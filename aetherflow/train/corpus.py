"""语料的无损压缩存储 + 逐批惰性解码。

训练数据侧的内存大头是**整份数据集的稠密表示**：token 流本身很小，但
one-hot 展开（``lm_batch`` 的默认形态）会把一份语料放大成
``(N, T, V)`` 的 f32 张量，并且整个 epoch 都驻留。

这个模块给两件事：

1. :func:`compress_tokens` / :func:`decompress_tokens`——整数 token 流的
   无损编解码（长度头 + delta → zigzag → LEB128）。无损由 roundtrip 判据
   钉死，不是"压了就算成功"；delta 编码对文本/数字这类相邻相关的序列
   给出数量级压缩比。
2. :class:`TokenCorpus`——**只**保存压缩字节，原始列表在构造结束时丢弃。
   ``iter_minibatches`` 每次只解当前批需要的窗口、只稠密化当前批。

第 2 条是"不需要解压就能训练"的准确含义：解压没有被取消，它被**下推到
批次粒度**——"整个数据集的展开形态"不再是任何时刻存在的对象。这条由
tracemalloc 的峰值实测在测试里给出数字，而不是靠叙述。

批次组装没有复用 ``train.lm.lm_batch``，是刻意的：那个函数把"最后一个
监督位"与"padding"绑在同一份左填充里，喂进来的行必须多带一位 target，
而窗口形状是调用方定的；``labels="index"`` 的扁平下标又绑定 (B, steps)
布局，先按 T+1 生成再切片会算错偏移。语料层的窗口是右移一位的固定关系
（``targets[t] = ids[t+1]``），自己写 30 行比套用它再纠正 off-by-one 更
不容易出错，也更便宜——不产生任何临时的大 one-hot 中间量。
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["compress_tokens", "decompress_tokens", "TokenCorpus", "peak_bytes"]

# ---------------------------------------------------------------------------
# 编解码
# ---------------------------------------------------------------------------


def _zigzag(n: int) -> int:
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _unzigzag(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)


def _put_varint(out: bytearray, value: int) -> None:
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)


def _get_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError(f"压缩流在第 {pos} 字节处截断")
        byte = data[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint 超过 64 位")


def compress_tokens(sequence: Sequence[int]) -> bytes:
    """一条整数 token 序列 → 字节。**无损**；相邻相近时极省。

    格式：``varint(长度)`` + 每个 delta 的 zigzag varint。delta 用 Python
    int 相减（不取模），token id 上限只受内存约束。长度头让单条记录可以
    独立解码、可以停在需要的位置。
    """
    seq = [int(x) for x in sequence]
    out = bytearray()
    _put_varint(out, len(seq))
    prev = 0
    for x in seq:
        _put_varint(out, _zigzag(x - prev))
        prev = x
    return bytes(out)


def decompress_tokens(blob: bytes) -> List[int]:
    """:func:`compress_tokens` 的逆。截断或多余字节都在这里报错，不静默补零。"""
    pos = 0
    n, pos = _get_varint(blob, 0)
    seq: List[int] = []
    prev = 0
    for _ in range(n):
        z, pos = _get_varint(blob, pos)
        prev += _unzigzag(z)
        seq.append(prev)
    if pos != len(blob):
        raise ValueError(f"压缩流多出 {len(blob) - pos} 字节")
    return seq


# ---------------------------------------------------------------------------
# 内存实测
# ---------------------------------------------------------------------------


def peak_bytes():
    """上下文管理器：块内的 tracemalloc 峰值字节（对照口径）。

    ``with peak_bytes() as box: ...`` 结束后 ``box["peak"]`` 即块内峰值。
    它统计的是 **Python 侧分配的内存增量峰值**，numpy 的巨型数组走的是
    这条路径，所以够用；但不包含 CUDA 显存——那要靠后端自己的账。
    """
    import tracemalloc

    return _Peak(box_tracemalloc=tracemalloc)


class _Peak:
    def __init__(self, box_tracemalloc) -> None:
        self._tm = box_tracemalloc
        self._was = box_tracemalloc.is_tracing()

    def __enter__(self) -> Dict[str, int]:
        if not self._was:
            self._tm.start()
        self._tm.reset_peak()
        self.box: Dict[str, int] = {"peak": 0}
        return self.box

    def __exit__(self, *exc) -> bool:
        _, peak = self._tm.get_traced_memory()
        self.box["peak"] = int(peak)
        if not self._was:
            self._tm.stop()
        return False


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------


class TokenCorpus:
    """只存压缩字节的 token 语料；按批惰性解码 + 按批稠密化。

    构造完就不再持有解压形态的语料：``self._blobs`` 是唯一真相。
    ``iter_minibatches`` 每次解出的只有当前批那几行、每行只有窗口长度。
    """

    def __init__(self, sequences: Sequence[Sequence[int]], vocab_size: int) -> None:
        if not sequences:
            raise ValueError("语料不能为空")
        self.vocab_size = int(vocab_size)
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size 至少为 2，得到 {self.vocab_size}")
        for i, s in enumerate(sequences):
            if len(s) < 2:
                raise ValueError(f"第 {i} 条序列长度不足 2，凑不出监督窗口")
            if any(not 0 <= int(x) < self.vocab_size for x in s):
                raise ValueError(f"第 {i} 条序列里有超出 [0, {self.vocab_size}) 的 token")
        self._blobs: List[bytes] = [compress_tokens(s) for s in sequences]
        self._lengths: List[int] = [len(s) for s in sequences]
        self._n_tokens = int(sum(self._lengths))
        self.steps = min(self._lengths) - 1

    # -- 尺寸账 -------------------------------------------------------

    @property
    def n_sequences(self) -> int:
        return len(self._blobs)

    @property
    def stored_bytes(self) -> int:
        return sum(len(b) for b in self._blobs)

    def storage_report(self) -> dict:
        """存储账：压缩字节 vs 两种"未压缩"口径。

        ``onehot_all_bytes`` 是"整份语料一次性 one-hot"的 f32 字节数——
        那正是这条压缩要消掉的驻留对象。
        """
        raw = self._n_tokens * 4
        onehot = self._n_tokens * self.vocab_size * 4
        return {
            "sequences": self.n_sequences,
            "tokens": self._n_tokens,
            "compressed_bytes": self.stored_bytes,
            "raw_i32_bytes": int(raw),
            "onehot_all_bytes": int(onehot),
            "ratio_vs_raw": raw / max(self.stored_bytes, 1),
            "ratio_vs_onehot": onehot / max(self.stored_bytes, 1),
        }

    # -- 解码 ---------------------------------------------------------

    def decode(self, index: int) -> List[int]:
        """整条解回（判据与调试用；训练路径只解窗口）。"""
        return decompress_tokens(self._blobs[index])

    def window(self, index: int, start: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """第 ``index`` 条从 ``start`` 起的 ``(ids[length], targets[length])``。

        ``targets[t] == ids[t+1]``，最后一个 target 取自 ``start+length``
        处的那个 token——所以序列至少要 ``length + 1`` 位。全程只解这一条。
        """
        if length < 1:
            raise ValueError(f"窗口长度至少 1，得到 {length}")
        seq = decompress_tokens(self._blobs[index])
        need = start + length + 1
        if start < 0 or need > len(seq):
            raise ValueError(
                f"窗口 [{start}, {start + length}) 需要第 {need} 位，"
                f"但这条只有 {len(seq)} 位"
            )
        ids = np.asarray(seq[start : start + length], dtype=np.int32)
        targets = np.asarray(seq[start + 1 : start + length + 1], dtype=np.int32)
        return ids, targets

    def iter_minibatches(
        self,
        batch: int,
        *,
        window: Optional[int] = None,
        start: int = 0,
        labels: str = "index",
        dtype=np.float32,
        drop_last: bool = True,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """按批产出 feed；每次只解 / 只稠密化当前批。

        键与 ``models/lm`` 的 LM 图对齐：``ids`` (B,T) i32、
        ``targets``（``"index"``：(B,T) i32 扁平下标；``"onehot"``：
        (B,T,V) f32）、``mask`` 与 ``reset`` 都是 (B,T) f32。

        ``reset`` 在第 0 列为 1：每批都是独立会话的起点（分块续接的另一半
        语义由调用方负责，与 ``train.lm.lm_batch`` 的约定一致）。

        ``drop_last`` 默认 **True**：丢掉不足 batch 的尾批。这不是照抄
        PyTorch 的参数（那边默认 False），而是静态形状框架的必然——图的
        batch 维在构图期写死（``f32[4, T]``），3 行的尾批进 4 行的图会在
        执行器入口报 ``ShapeMismatchError``，那个报错离"语料行数不整除"
        太远，调用方只会看到形状不匹配。要保留尾批传 False（配一个
        batch=尾长 的独立图，或者只做推理）。
        """
        if labels not in ("index", "onehot"):
            raise ValueError(f"labels 只能是 index / onehot，得到 {labels!r}")
        if batch < 1:
            raise ValueError(f"batch 至少 1，得到 {batch}")
        win = int(self.steps if window is None else window)
        if win < 1 or start + win + 1 > min(self._lengths):
            raise ValueError(
                f"窗口 {win} + 起点 {start} 超过最短序列 {min(self._lengths)} 位"
            )

        limit = self.n_sequences
        if drop_last:
            limit -= limit % batch
            if limit == 0:
                raise ValueError(
                    f"语料只有 {self.n_sequences} 行，不足 batch={batch}：丢弃尾批"
                    "后一批都不剩。静默产出空迭代等于无声地什么都不训，"
                    f"请改小 batch、增大全量语料，或显式传 drop_last=False"
                )
        for i in range(0, limit, batch):
            rows = list(range(i, min(i + batch, self.n_sequences)))
            b = len(rows)
            ids = np.zeros((b, win), dtype=np.int32)
            tg = np.zeros((b, win), dtype=np.int32)
            for r, index in enumerate(rows):
                ids[r], tg[r] = self.window(index, start, win)
            mask = np.ones((b, win), dtype=dtype)
            reset = np.zeros((b, win), dtype=dtype)
            reset[:, 0] = 1.0
            if labels == "index":
                # 扁平下标：(b*T + t)*V + y —— 与 masked_lm_loss_indexed 对齐
                flat = (
                    (np.arange(b, dtype=np.int64)[:, None] * win
                     + np.arange(win, dtype=np.int64)[None, :]) * self.vocab_size
                    + tg.astype(np.int64)
                )
                targets = flat.astype(np.int32)
            else:
                targets = np.zeros((b, win, self.vocab_size), dtype=dtype)
                np.put_along_axis(
                    targets, tg[:, :, None].astype(np.int64), dtype(1.0), axis=-1
                )
            yield {
                "ids": ids,
                "targets": targets,
                "mask": mask,
                "reset": reset,
            }

    def __repr__(self) -> str:
        rep = self.storage_report()
        return (
            f"TokenCorpus(sequences={rep['sequences']}, tokens={rep['tokens']}, "
            f"compressed={rep['compressed_bytes']}B, "
            f"onehot全量={rep['onehot_all_bytes']}B)"
        )
