"""压缩态训练：参数以低秩因子 / int8 底驻留，训练全程不物化稠密 f32 权重。

与 ``passes/param_compress.py`` 的分工：pass 负责**图**的改写（参数输入换
成因子与冻结底），这个模块负责**数值**的准备（怎么把一份稠密 f32 权重变成
因子与底）与**参数表**的接线（哪些进优化器、哪些必须冻结）。

三条内存判据，对应"研制模型内存压缩"的三个承诺：

1. **可训练量**：``W (m,n) f32`` 换 ``A (m,r) + B (r,n) f32``，优化器槽位
   （Adam 的 m/v）跟着参数走，槽位同比例缩。qlora 里 int8 底与 scale 不进
   参数表，零槽位。
2. **驻留表示即真相**：qlora 的底以 int8 数组交付，dequant 只作为图节点
   存在；参数表里找不到任何一份 ``m×n`` 的 f32 数组（``assert_dense_free``
   在运行时检查这件事，而不是靠读代码保证）。
3. **数据侧配套**：``compress_tokens`` 给 token 流做无损压缩（delta +
   zigzag + LEB128，对 i32 基线实测 ~3-9×），配合 ``TokenCorpus.iter_minibatches``
   的惰性解码，语料的完整 f32 one-hot（或全量解压副本）在训练中从不出现。

**exclude 名单由本模块生成**（``CompressedModel.exclude``）。这不是便利，
是正确性：``ParameterStore.from_graph`` 会把任何未排除的 f32 图输入当成
可训练参数——qlora 的 ``__scale`` 是 f32，漏排除它会连带量"跟着训练一起
漂"，冻结的前提从内部被破坏。i8 的 ``__base`` 漏排除则直接在 from_graph
报"不能作为可训练参数"，至少是响的；scale 那条是静默的，所以必须替调用方
兜住。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ..ir.graph import Graph
from ..ir.types import f32, i8
from ..passes.param_compress import ParamCompression, plan_param_compression

__all__ = [
    "CompressedModel",
    "compress_for_training",
    "low_rank_factors",
    "quantize_int8",
    "reconstruct_dense_weights",
    "memory_report",
    "assert_dense_free",
]


def quantize_int8(
    weight: np.ndarray, *, axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """对称 per-column int8 量化：``W ≈ base * scale``。

    返回 ``(base int8 (m,n), scale f32 (n,))``。scale 取每列最大绝对值 /
    127，格点误差以 ``scale/2`` 为界，所以 ``|W - dequant| <= scale/2`` 逐列
    成立——量化质量不靠事后调参，判据写在构造函数里。

    ``axis`` 只支持 -1（按输出列归一）：per-column 与图里
    ``mul(base_f32, scale)`` 的广播形态一一对应，改成别的轴 dequant 的
    形状就不对了，与其运行时算错不如现在拒绝。
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or axis != -1:
        raise ValueError(
            f"只支持二维权重的 per-column 量化（axis=-1），得到 ndim={w.ndim} axis={axis}"
        )
    # 每**列**（输出通道 = 最后一轴）一个 scale：沿行取最大，shape (n,)
    amax = np.max(np.abs(w), axis=0)
    scale = (amax / 127.0).astype(np.float32)
    scale[scale == 0.0] = 1.0  # 全零列：scale=1、base=0，dequant 仍是 0
    base = np.round(w / scale[np.newaxis, :]).astype(np.int8)
    return base, scale


def low_rank_factors(
    weight: np.ndarray, rank: int, *, init_b_zero: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """截断 SVD 给 ``(A (m,r) f32, B (r,n) f32)``，使 ``A @ B ≈ W``。

    ``A = U_r * sqrt(S_r)``、``B = sqrt(S_r) * V_r^T``——两边对称，因子
    的尺度与权重同阶，热启动（先分解再训练）时梯度不会偏向一边。
    截断误差是 ``sigma_{r+1}``（Frobenius 意义下
    ``sqrt(sum_{i>r} sigma_i^2)``），这是 SVD 的最优性保证。

    ``init_b_zero=True`` 是给**从零开始的 adapter 训练**的：LoRA 惯例是
    B 置零，第一步就恒等地从冻结底出发；热启动分解时不要用——那是
    "拟合 W"，不是"从 W 的扰动起步"。
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"低秩分解只支持二维权重，得到 ndim={w.ndim}")
    m, n = w.shape
    if not 1 <= rank <= min(m, n):
        raise ValueError(f"秩 {rank} 必须落在 [1, min({m}, {n})]")
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    A = (u[:, :rank] * np.sqrt(s[:rank])[np.newaxis, :]).astype(np.float32)
    B = (np.sqrt(s[:rank])[:, np.newaxis] * vt[:rank, :]).astype(np.float32)
    if init_b_zero:
        B = np.zeros_like(B)
    return A, B


class CompressedModel:
    """改写后的图 + 可训练因子表 + 冻结底表，一份接线好的训练现场。"""

    def __init__(
        self,
        graph: Graph,
        trainable: Dict[str, np.ndarray],
        frozen: Dict[str, np.ndarray],
        data_inputs: List[str],
        policy: Mapping[str, Tuple[str, int]],
    ) -> None:
        self.graph = graph
        self.trainable = trainable
        self.frozen = frozen
        self.data_inputs = data_inputs
        self.policy = dict(policy)

    # -- 接线 ---------------------------------------------------------

    @property
    def exclude(self) -> List[str]:
        """给 ``ParameterStore.from_graph`` 的完整非参数名单。"""
        return [*self.data_inputs, *self.frozen]

    def feed(self, batch: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """一次 step 的 batch feed：冻结底 + 本批数据（**不拷贝**）。

        冻结数组每次传的是同一个对象，执行器只读不改，所以没有复制成本。
        """
        merged = dict(self.frozen)
        if batch:
            clash = set(batch) & set(self.frozen)
            if clash:
                raise KeyError(f"batch 覆盖了冻结输入 {sorted(clash)}")
            merged.update(batch)
        return merged

    def dense_weights(self, trained: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """从**训练后的**驻留形态重建稠密 f32 参数表（交给推理引擎）。

        ``trained`` 应是 ``store.arrays()``：优化器原地更新的是 store 里的
        数组，不是 ``self.trainable`` 那份初始分解。缺省或传 ``None`` 会
        直接报错，而不是悄悄回落初始因子——"忘了传训练结果"的后果是一份
        看起来合理、实际从没训练过的权重，那必须是显式失败。
        """
        if trained is None:
            raise TypeError(
                "dense_weights 需要训练后的数组（store.arrays()）；"
                "self.trainable 只是初始分解，用它重建等于把训练结果丢掉"
            )
        return reconstruct_dense_weights(trained, self.frozen)

    def __repr__(self) -> str:
        return (
            f"CompressedModel(graph={self.graph.name}, "
            f"trainable={sum(int(a.size) for a in self.trainable.values())} 元素, "
            f"frozen={sum(int(a.size) for a in self.frozen.values())} 元素)"
        )


def compress_for_training(
    graph: Graph,
    weights: Mapping[str, np.ndarray],
    *,
    ranks: Mapping[str, int],
    keep_base: Optional[Mapping[str, bool]] = None,
    data_inputs: Iterable[str] = (),
    lora_b_zero: bool = False,
) -> CompressedModel:
    """从"原图 + 全部稠密权重"一步得到可训练的压缩现场。

    ``weights`` 必须覆盖**所有非数据的 f32 图输入**：``ranks`` 里的走压缩，
    其余的（LayerNorm 的 gamma/beta、bias 这类本就该留 f32 的参数）**原样
    进可训练表**。只收因子不收未压参数会让接好线的 Trainer 在第一个 step
    就报缺输入——把"哪些参数没被压"交给调用方记忆，正是这条接线层要消灭
    的那类错误。

    qlora 模式下先 per-column 量化成 int8 底，再用**底 + 适配器**共同
    拟合原权重：``A@B ≈ W - dequant(base)``（对残差做 SVD，适配器学的正是
    量化丢掉的部分）。这一步是 qlora 数值成立的关键——直接对 W 做 SVD、
    底又是 W 的量化，两者各拟合一次会双重计数。
    """
    plan = plan_param_compression(graph, ranks=ranks, keep_base=keep_base)
    policy = {p: plan[p] for p in plan}
    new_graph = ParamCompression(policy).run(graph)

    data = set(data_inputs)
    param_like = [
        v.name
        for v in graph.inputs
        if v.name not in data and v.type.dtype.value in ("f32", "f64")
    ]
    missing = sorted(set(param_like) - set(weights))
    if missing:
        raise KeyError(
            f"weights 里缺少这些非数据浮点输入的值: {missing}；"
            "不压缩的参数也要进 weights（它们会原样留在可训练表里）"
        )
    extra = sorted(set(weights) - set(param_like))
    if extra:
        raise KeyError(f"weights 里有图输入之外的名字，或数据输入: {extra}")

    trainable: Dict[str, np.ndarray] = {}
    frozen: Dict[str, np.ndarray] = {}
    for pname in param_like:
        w = np.asarray(weights[pname], dtype=np.float32)
        declared = next(v for v in graph.inputs if v.name == pname)
        if w.shape != tuple(int(d) for d in declared.type.shape):
            raise ValueError(f"参数 {pname} 的值形状 {w.shape} 与图声明不符")
        if pname not in policy:
            trainable[pname] = w  # 不压：原样可训练
            continue
        mode, rank = policy[pname]
        if mode == "qlora":
            base, scale = quantize_int8(w)
            residual = w - base.astype(np.float32) * scale[np.newaxis, :]
            A, B = low_rank_factors(residual, rank, init_b_zero=lora_b_zero)
            frozen[f"{pname}__base"] = base
            frozen[f"{pname}__scale"] = scale
        else:
            A, B = low_rank_factors(w, rank, init_b_zero=lora_b_zero)
        trainable[f"{pname}__A"] = A
        trainable[f"{pname}__B"] = B

    return CompressedModel(new_graph, trainable, frozen, list(data_inputs), policy)


def reconstruct_dense_weights(
    trainable: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
    *,
    extra: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """把驻留的压缩形态重建成稠密 f32 参数表（推理引擎的接口）。

    训练世界与推理世界的边界就在这里：训练全程只持有因子与 int8 底，
    ``models.lm.generate`` / ``LMRunner`` 那套推理路径要的是"参数名 ->
    稠密 f32"。重建式是 ``W = dequant(base) + A @ B``（qlora）或
    ``W = A @ B``（low_rank-only）。

    ``trainable`` 必须传**优化器正在原地更新的那份数组**（通常是
    ``store.arrays()``），不是 :class:`CompressedModel.trainable` 里的初始
    因子——后者只是"分解方案"，训练一步都不会改它。拿错份的后果是"训练
    完全白做、推理静默用初始权重"，属于最难查的那类错误，所以判据写在
    参数名上（见 :meth:`CompressedModel.dense_weights`）。

    ``extra`` 是没有被压缩的参数（LayerNorm 的 gamma/beta、bias 等），
    原样带上——推理图的参数表必须完整。
    """
    out: Dict[str, np.ndarray] = {}
    consumed = set()
    for name in trainable:
        if not name.endswith("__A"):
            continue
        base_name = name[: -len("__A")]
        bname = f"{base_name}__B"
        if bname not in trainable:
            raise KeyError(f"{name} 找不到配对的 {bname}")
        a = np.asarray(trainable[name], dtype=np.float32)
        b = np.asarray(trainable[bname], dtype=np.float32)
        dense = a @ b
        qkey = f"{base_name}__base"
        if qkey in frozen:
            qbase = np.asarray(frozen[qkey])
            if qbase.dtype != np.int8:
                raise TypeError(
                    f"{qkey} 的 dtype 是 {qbase.dtype}，重建要求 int8 驻留"
                )
            scale = np.asarray(frozen[f"{base_name}__scale"], dtype=np.float32)
            dense = dense + qbase.astype(np.float32) * scale[np.newaxis, :]
            consumed.add(f"{base_name}__scale")
        consumed.add(qkey)
        out[base_name] = dense.astype(np.float32)
        consumed.update({name, bname})

    for name, arr in trainable.items():
        if name in consumed:
            continue
        if name.endswith("__B"):
            base_name = name[: -len("__B")]
            raise KeyError(
                f"{name} 找不到配对的 {base_name}__A；半套因子会被当成"
                "普通参数原样搬进推理权重表，那是最安静的错法"
            )
        out[name] = np.asarray(arr, dtype=np.float32)
    for name, arr in (extra or {}).items():
        if name in out:
            raise KeyError(f"extra 里的 {name} 与重建出的参数重名")
        out[name] = np.asarray(arr, dtype=np.float32)
    leftover = sorted((set(frozen) | set(trainable)) - consumed - set(out))
    if leftover:
        raise KeyError(f"这些量没有被任何因子消费，无法重建: {leftover}")
    return out


def memory_report(
    original: Mapping[str, np.ndarray], compressed: CompressedModel
) -> dict:
    """压缩前后的参数字节数对账（驻留口径：数组 nbytes 之和）。

    冻结的 int8 底计入，因为它就是参数现在唯一的驻留形态。
    """
    base_bytes = sum(int(np.asarray(v).nbytes) for v in original.values())
    new_bytes = sum(int(v.nbytes) for v in compressed.trainable.values()) + sum(
        int(v.nbytes) for v in compressed.frozen.values()
    )
    return {
        "original_bytes": base_bytes,
        "compressed_bytes": new_bytes,
        "ratio": base_bytes / max(new_bytes, 1),
        "trainable_bytes": sum(int(v.nbytes) for v in compressed.trainable.values()),
        "frozen_bytes": sum(int(v.nbytes) for v in compressed.frozen.values()),
    }


def assert_dense_free(store, frozen: Mapping[str, np.ndarray],
                      shapes: Mapping[str, Tuple[int, ...]]) -> None:
    """运行时判据：压缩确实生效，而不是"配置里看起来压了"。

    三条硬检查，都按**名字**，不按形状：

    1. 被压参数的原名不在参数表里——稠密 f32 副本如果存在，它就是原来
       那个名字；store 从改写后的图构建，被压的名字根本不该出现。
    2. 因子表里不许混进 ``__base`` / ``__scale``——冻结量进了优化器，
       "冻结"这个前提从内部就破了。
    3. 每个 ``__base`` 的驻留 dtype 必须是 int8。

    为什么**不**按"同形同类型的 f32 数组"来抓副本：真实模型里形状巧合是
    常态——LM 图的 ``l0_A_raw`` 与被压的 ``l0_B_w`` 恰好同形，按形状查会
    把合法参数当泄漏。判据要能区分"就是那份 W"和"另一份碰巧同形的参数"，
    只有名字做得到；形状巧合靠第 1 条（原名）兜住，靠 store 从新图构建
    这个流程本身兜住。
    """
    for pname in shapes:
        if pname in store.names:
            raise AssertionError(
                f"参数 {pname} 的原名仍在参数表里——稠密副本没被消掉"
            )
    for name in store.names:
        if name.endswith(("__base", "__scale")):
            raise AssertionError(
                f"冻结量 {name} 混进了参数表；它不该被优化器持有"
            )
    for name, arr in frozen.items():
        if name.endswith("__base") and arr.dtype != np.int8:
            raise AssertionError(f"{name} 的驻留 dtype 是 {arr.dtype}，必须是 int8")
