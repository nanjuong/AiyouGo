"""SSM 语言模型：embedding → Mamba 栈 → final RMSNorm → 词表头。

审计 AF-34 指出"能表达一个 block 不等于具备语言模型闭环"。这里把闭环拆成
**图里能验收的部分**，每一块都有可执行的判据：

    词表嵌入   embedding(table, ids)                      一个已有算子
    主干       N 层 Mamba（复用 models.mamba）             同一张图
    末端归一化 rmsnorm
    词表头     matmul(flat, W)；tie_weights 时 W = embed^T
    损失       masked_lm_loss：按**有效 token 数**归一，padding 位不贡献梯度
               masked_lm_loss_indexed：同一损失，标签是整数下标而非 one-hot
    解码       stateful 构图：prefill 整段 → 逐 token decode，状态跨调用带出
    会话       LMRunner 跨请求复用图与驻留权重（每次 generate 都重新构图是
               一笔固定开销：构图 + 首次捕获约 200～250 ms）
    采样       greedy / temperature / top-k / top-p / 重复惩罚，纯主机侧实现
    束搜索     num_beams 条假设放进图的 batch 维，状态按父假设重排
    停止       EOS 或 max_new_tokens
    批量       generate_batch：左侧补 pad + 逐行 reset，B 条序列一起解码，
               每行独立停止（行与行之间不共享计数）
    服务       ServingEngine：请求队列 + 连续批处理，槽位一空就准入新请求

不在本模块的：tokenizer、chat template、工具调用的受约束格式。那些是产品层
的数据与协议，不是图语义；它们不该借"多一个算子"来假装已经具备。

**形状约定**：训练与 prefill/decode 都用静态形状（Mamba 的卷积是按时间展开
的，逆拓扑里没有符号时间维）。``with_steps`` 给出同一套参数名下的另一种长度，
prefill 的 ``state_out`` 直接就是 decode 的输入。
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32, i8, i32
from .mamba import MambaModelSpec, build_mamba_layers

__all__ = [
    "LMSpec",
    "build_language_model",
    "build_language_model_stateful",
    "masked_lm_loss",
    "masked_lm_loss_indexed",
    "sample_next_token",
    "generate",
    "generate_batch",
    "generate_beam",
    "LMRunner",
    "ServingEngine",
]


# 存档凭证覆盖的字段：形状 + 词表约定 + 归一化 eps。少一个都会让"同名参数
# 装进另一份模型"变成运行期才发现的错位。
_SPEC_FIELDS = (
    "vocab_size",
    "d_model",
    "num_layers",
    "d_state",
    "d_conv",
    "expand",
    "steps",
    "batch",
    "tie_weights",
    "pad_id",
    "eos_id",
    "eps",
)


@dataclass(frozen=True)
class LMSpec:
    """一张语言模型图的形状与词表约定。

    ``tie_weights=True`` 时词表头复用嵌入表（``embed^T``），少两组参数；
    这不是精度优化，而是让"小词表 + 小模型"的验收图参数更少、错配更少。
    """

    vocab_size: int = 64
    d_model: int = 32
    num_layers: int = 2
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    steps: int = 16
    batch: int = 2
    tie_weights: bool = True
    pad_id: int = 0
    eos_id: int = 1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "num_layers",
            "d_state",
            "d_conv",
            "expand",
            "steps",
            "batch",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须为正，得到 {getattr(self, name)}")
        for name in ("pad_id", "eos_id"):
            value = int(getattr(self, name))
            if not 0 <= value < self.vocab_size:
                raise ValueError(
                    f"{name} 必须落在 [0, {self.vocab_size})，得到 {value}"
                )

    @property
    def mamba(self) -> MambaModelSpec:
        return MambaModelSpec(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            steps=self.steps,
            batch=self.batch,
            num_layers=self.num_layers,
        )

    def with_steps(self, steps: int) -> "LMSpec":
        """换序列长度：prefill 用整段，decode 用 ``with_steps(1)``。"""
        return replace(self, steps=int(steps))

    def with_batch(self, batch: int) -> "LMSpec":
        """换批大小：束搜索把 ``num_beams`` 条假设放进同一张图的 batch 维。"""
        return replace(self, batch=int(batch))

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {
            "embed": (self.vocab_size, self.d_model)
        }
        shapes.update(self.mamba.param_shapes)
        shapes["final_norm"] = (self.d_model,)
        if not self.tie_weights:
            shapes["head_w"] = (self.d_model, self.vocab_size)
            shapes["head_b"] = (self.vocab_size,)
        return shapes

    @property
    def state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self.mamba.state_shapes

    def init_params(self, seed: int = 0) -> Dict[str, np.ndarray]:
        """一次给出全部参数；嵌入表用 ``1/sqrt(d)`` 缩放的小随机量。

        Mamba 那部分交给 :meth:`MambaModelSpec.init_params`（半衰期网格的
        专属初始化），这里不重复实现。
        """
        rng = np.random.default_rng(seed)
        params: Dict[str, np.ndarray] = {
            "embed": (
                rng.standard_normal((self.vocab_size, self.d_model))
                / np.sqrt(self.d_model)
            ).astype(np.float32)
        }
        params.update(self.mamba.init_params(seed=seed + 1))
        params["final_norm"] = np.ones(self.d_model, dtype=np.float32)
        if not self.tie_weights:
            params["head_w"] = (
                rng.standard_normal((self.d_model, self.vocab_size))
                / np.sqrt(self.d_model)
            ).astype(np.float32)
            params["head_b"] = np.zeros(self.vocab_size, dtype=np.float32)
        return params

    def carry_state(self, state_out: Mapping[str, Any]) -> Dict[str, Any]:
        """:func:`build_language_model_stateful` 的 ``state_out`` → 下一段的输入。"""
        return self.mamba.carry_state(state_out)

    def meta(self) -> Dict[str, object]:
        """存档用的配置凭证（键带 ``lm_`` 前缀，与其它 meta 不冲突）。

        用法是 ``trainer.save_checkpoint(path, meta=spec.meta())``，加载后
        ``spec.check_meta(meta)``。这是一个**显式**绑定：存档格式不偷偷塞
        配置，读的一方必须拿着 spec 来核对。
        """
        return {f"lm_{field}": getattr(self, field) for field in _SPEC_FIELDS}

    def check_meta(self, meta: Mapping[str, object]) -> None:
        """与存档里的凭证逐项比对；缺项按"未绑定"直接拒绝。

        浮点只比 ``eps``，用 1e-12 的绝对容差；布尔先于整数判断——``bool``
        是 ``int`` 的子类，顺序写反会把 ``False`` 当成 ``0`` 放过去。
        """
        missing = [f"lm_{field}" for field in _SPEC_FIELDS if f"lm_{field}" not in meta]
        if missing:
            raise ValueError(
                f"存档没有绑定语言模型配置，缺少 {missing}；"
                "请用 save_checkpoint(path, meta=spec.meta()) 重存"
            )
        diff = []
        for field in _SPEC_FIELDS:
            want = getattr(self, field)
            got = meta[f"lm_{field}"]
            if isinstance(want, bool):
                same = bool(got) is want
            elif isinstance(want, float):
                same = abs(float(got) - want) <= 1e-12
            else:
                same = int(got) == int(want)
            if not same:
                diff.append(f"{field}: 存档 {got!r} / 当前 {want!r}")
        if diff:
            raise ValueError("语言模型配置与存档不一致：\n  " + "\n  ".join(diff))


def _head(
    g: Graph,
    spec: LMSpec,
    y: Value,
    embed: Value,
    head_w: Optional[Value],
    head_b: Optional[Value],
    *,
    last_only: bool = False,
) -> Value:
    """``(B,T,D)`` → ``(B,T,V)``；平铺成二维再乘，避免三维 matmul。

    ``last_only=True`` 只对最后位置做投影，返回 ``(B,1,V)``。推理只需要
    下一个 token 的分布，把整段 logits 物化成 ``(B,T,V)`` 是纯粹的浪费：
    32K 上下文、V=32K 时那是 4 GiB。RMSNorm 逐位置独立，切片后算与整段算
    逐个相等，不是近似。
    """
    if last_only:
        last = g.slice(y, dim=1, start=spec.steps - 1, stop=spec.steps)
        flat = g.reshape(last, shape=(spec.batch, spec.d_model))
        positions = 1
    else:
        flat = g.reshape(y, shape=(spec.batch * spec.steps, spec.d_model))
        positions = spec.steps
    if spec.tie_weights:
        logits = g.matmul(flat, g.transpose(embed, perm=(1, 0)))
    else:
        assert head_w is not None and head_b is not None
        logits = g.add(g.matmul(flat, head_w), head_b)
    return g.reshape(logits, shape=(spec.batch, positions, spec.vocab_size))


def _model_head_inputs(
    g: Graph, spec: LMSpec, name
) -> Tuple[Value, Value, Value, Optional[Value], Optional[Value], Dict[str, Value]]:
    ids = g.input(name("ids"), i32[spec.batch, spec.steps])
    embed = g.input(name("embed"), f32[(spec.vocab_size, spec.d_model)])
    norm = g.input(name("final_norm"), f32[(spec.d_model,)])
    head_w = head_b = None
    extras: Dict[str, Value] = {}
    if not spec.tie_weights:
        head_w = g.input(name("head_w"), f32[(spec.d_model, spec.vocab_size)])
        head_b = g.input(name("head_b"), f32[(spec.vocab_size,)])
        extras[name("head_w")] = head_w
        extras[name("head_b")] = head_b
    return ids, embed, norm, head_w, head_b, extras


def build_language_model(
    g: Graph,
    spec: Optional[LMSpec] = None,
    *,
    prefix: str = "",
    with_reset: bool = False,
) -> Tuple[Dict[str, Value], Value]:
    """训练/prefill 口径：``(inputs, logits)``，logits 形状 ``(B, T, V)``。

    ``with_reset=True`` 时多一张 ``reset (B, T)`` 图输入，直接交给整栈的
    卷积与 SSM：左填充的 pad token 既不推进状态，也不进入真实 token 的卷积
    窗口。``train.lm.lm_batch`` 返回的 ``reset`` 就是这张输入。
    """
    spec = spec or LMSpec()

    def name(base: str) -> str:
        return f"{prefix}{base}"

    ids, embed, norm, head_w, head_b, extras = _model_head_inputs(g, spec, name)
    x = g.embedding(embed, ids)
    reset = g.input(name("reset"), f32[spec.batch, spec.steps]) if with_reset else None
    layer_inputs, y, _ = build_mamba_layers(
        g, spec.mamba, x, prefix=prefix, reset=reset
    )
    y = g.rmsnorm(y, norm, eps=spec.eps)
    logits = _head(g, spec, y, embed, head_w, head_b)
    inputs = {
        **layer_inputs,
        name("ids"): ids,
        name("embed"): embed,
        name("final_norm"): norm,
        **extras,
    }
    if reset is not None:
        inputs[name("reset")] = reset
    return inputs, logits


def build_language_model_stateful(
    g: Graph,
    spec: Optional[LMSpec] = None,
    *,
    prefix: str = "",
    with_reset: bool = False,
    head: str = "all",
) -> Tuple[Dict[str, Value], Optional[Value], Dict[str, Value]]:
    """流式口径：``(inputs, logits, state_out)``。

    ``state_out`` 的键与 :meth:`LMSpec.carry_state` 对应；``with_steps(1)`` 的
    同一份构图就是 decode，无需另一套参数名。``with_reset=True`` 与训练口径
    同义：``reset`` 是 ``(B, T)`` 图输入，1 表示该位置开始新序列。

    ``head`` 决定要不要算词表投影：

    ``"all"``
        全部位置，logits ``(B, T, V)``；训练/prefill 对照用。
    ``"last"``
        只算最后位置，logits ``(B, 1, V)``；生成的下一步只需要它。
    ``"none"``
        不算投影，``logits`` 是 ``None``；固定块预填充的中间块只带出状态，
        既不物化 ``(B,T,V)``，也不做词表头的大矩阵乘。
    """
    if head not in ("all", "last", "none"):
        raise ValueError(f"head 只能是 'all' / 'last' / 'none'，得到 {head!r}")
    spec = spec or LMSpec()

    def name(base: str) -> str:
        return f"{prefix}{base}"

    ids, embed, norm, head_w, head_b, extras = _model_head_inputs(g, spec, name)
    x = g.embedding(embed, ids)
    reset = g.input(name("reset"), f32[spec.batch, spec.steps]) if with_reset else None
    layer_inputs, y, state_out = build_mamba_layers(
        g, spec.mamba, x, prefix=prefix, with_states=True, reset=reset
    )
    logits: Optional[Value] = None
    if head != "none":
        y = g.rmsnorm(y, norm, eps=spec.eps)
        logits = _head(g, spec, y, embed, head_w, head_b, last_only=head == "last")
    inputs = {
        **layer_inputs,
        name("ids"): ids,
        name("embed"): embed,
        name("final_norm"): norm,
        **extras,
    }
    if reset is not None:
        inputs[name("reset")] = reset
    return inputs, logits, state_out


def _masked_mean(g: Graph, nll: Value, mask: Value, dtype) -> Value:
    """按有效 token 数归一：``sum(nll * mask) / (sum(mask) + eps)``。

    全 padding 的一批分母取 0，损失因此是 0 而不是 NaN——"这一批没有可学的
    东西"不该让整个训练步变成非有限值。f32 下 ``1e-12`` 加到任何 >= 1 的
    计数上都不可见；全 padding 时 ``0/eps = 0``。
    """
    weighted = g.mul(nll, mask)
    total = g.reduce_sum(g.reduce_sum(weighted, axis=0), axis=0)
    count = g.reduce_sum(g.reduce_sum(mask, axis=0), axis=0)
    return g.div(total, g.add(count, g.constant(1e-12, dtype=dtype)))


def masked_lm_loss(g: Graph, logits: Value, targets: Value, mask: Value) -> Value:
    """按有效 token 数归一的交叉熵（全程 f32）。

    ``targets`` 是与 logits 同形的 one-hot，``mask`` 是 ``(B, T)`` 的 0/1。
    与 ``train.loss.softmax_cross_entropy`` 的差别是**分母**：那个把
    ``B*T*V`` 一起平均，padding 位会稀释梯度；这里只除以被选中的 token 数。

    输入先统一到 f32 再算：f16 下 ``1e-12`` 会下溢成 0，全空批变成 ``0/0``
    的 NaN，梯度也非有限；而损失的精度本来就不该跟着算子走。代价是 f16 前向
    多一次 cast，反向同理，但只在图上多两个节点。

    one-hot 把 ``B*T*V`` 个 float 带进图里；V 大时请改用
    :func:`masked_lm_loss_indexed`，两条路径的损失与梯度逐位相同。
    """
    if targets.type.shape != logits.type.shape:
        raise ValueError(
            f"masked_lm_loss 的 targets 形状必须与 logits 一致，"
            f"得到 {targets.type.shape} 与 {logits.type.shape}"
        )
    dtype = f32
    if logits.type.dtype != dtype:
        logits = g.cast(logits, dtype=dtype)
    if targets.type.dtype != dtype:
        targets = g.cast(targets, dtype=dtype)
    if mask.type.dtype != dtype:
        mask = g.cast(mask, dtype=dtype)
    shift = g.reduce_max(logits, axis=-1, keepdims=True)
    shifted = g.sub(logits, shift)
    log_z = g.log(g.reduce_sum(g.exp(shifted), axis=-1, keepdims=True))
    logp = g.sub(shifted, log_z)
    picked = g.reduce_sum(g.mul(logp, targets), axis=-1, keepdims=False)
    nll = g.mul(picked, g.constant(-1.0, dtype=dtype))
    return _masked_mean(g, nll, mask, dtype)


def masked_lm_loss_indexed(
    g: Graph, logits: Value, targets: Value, mask: Value
) -> Value:
    """与 :func:`masked_lm_loss` 同义，``targets`` 是 ``(B, T)`` 的整数下标。

    下标是**扁平**的 ``(b*T + t)*V + y``（``train.lm.lm_batch(labels="index")``
    产出）。图里把 ``(B,T,V)`` 的 logits 拉直成一维"表"，再交给 ``embedding``
    的查表语义取出目标位的 logit——``embedding`` 就是"按整数下标沿第一维
    取值"，把 logits 拉直之后它正好是"沿最后一维取一格"，反向也直接是
    ``embedding_backward`` 的散射累加，不需要新算子。``targets`` 保持
    ``(B,T)`` 参与查表，不 reshape（设备后端的 ``reshape`` 只承接 f16/f32）。

    为什么下标在主机侧算：CUDA / HIP 的逐元素核只承接 f16 / f32，图内没有
    可用的整数偏移运算；靠 cast 到 f32 再算则会在大词表下丢精度（``B*T*V``
    超过 2^24 时）。主机侧只多 ``B*T`` 个 int32，比 one-hot 少 ``V`` 倍。

    ``mask``、归一化与 f32 抬升都与 one-hot 路径一致。前向**逐位相同**：
    ``sub`` 始终作用在同一对值上，one-hot 的 ``reduce_sum`` 只是把一个非零项
    与一串 0 相加。反向梯度在 1 ulp 内一致——one-hot 侧多经过一次
    ``(B,T,V)`` 的广播乘与归约，求和顺序不同。
    """
    shape = logits.type.shape
    if len(shape) != 3:
        raise ValueError(
            f"masked_lm_loss_indexed 需要 (B, T, V) 的 logits，得到 {shape}"
        )
    if not all(isinstance(d, int) for d in shape):
        raise ValueError(f"整数标签路径需要静态形状，得到 {shape}")
    batch, steps, vocab = (int(d) for d in shape)
    if tuple(targets.type.shape) != (batch, steps):
        raise ValueError(
            f"targets 应为 (B, T)=({batch}, {steps}) 的整数下标，"
            f"得到 {targets.type.shape}"
        )
    if targets.type.dtype not in (i32, i8):
        raise TypeError(
            f"targets 必须是整数下标（i32 / i8），得到 {targets.type.dtype.value}"
        )
    dtype = f32
    if logits.type.dtype != dtype:
        logits = g.cast(logits, dtype=dtype)
    if mask.type.dtype != dtype:
        mask = g.cast(mask, dtype=dtype)

    flat = g.reshape(logits, shape=(batch * steps * vocab,))
    picked = g.embedding(flat, targets)

    shift = g.reduce_max(logits, axis=-1, keepdims=True)
    log_z = g.log(
        g.reduce_sum(g.exp(g.sub(logits, shift)), axis=-1, keepdims=True)
    )
    # 与 one-hot 路径同样的运算顺序：(logit - max) - log(sum exp)。
    centered = g.sub(picked, g.reshape(shift, shape=(batch, steps)))
    logp = g.sub(centered, g.reshape(log_z, shape=(batch, steps)))
    nll = g.mul(logp, g.constant(-1.0, dtype=dtype))
    return _masked_mean(g, nll, mask, dtype)


def _apply_repetition_penalty(
    z: np.ndarray, recent: Sequence[int], penalty: float
) -> np.ndarray:
    """CTRL 式重复惩罚：已出现过的 token，正 logit 除以惩罚、负 logit 乘惩罚。

    除法与乘法都让"更可能"的那个 logit 绝对值变小，因此已出现 token 的
    概率只会降不会升。``penalty == 1.0`` 是严格无操作（连副本都不做）。
    重复的 token 只惩罚一次（``set`` 语义），越界的 id 显式报错——静默
    忽略会让调用方以为惩罚生效了。
    """
    if penalty == 1.0:
        return z
    ids = np.fromiter((int(t) for t in recent), dtype=np.int64, count=len(recent))
    if ids.size == 0:
        return z
    if ids.min() < 0 or ids.max() >= z.shape[-1]:
        raise ValueError(
            f"repetition_penalty 的历史 token 越界（词表 {z.shape[-1]}）；"
            f"范围 {ids.min()}～{ids.max()}"
        )
    row = np.array(z, dtype=np.float64, copy=True)
    hit = np.unique(ids)
    picked = row[..., hit]
    row[..., hit] = np.where(picked > 0, picked / penalty, picked * penalty)
    return row


def sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    recent: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """从最后一维的 logits 采一个 token。``temperature <= 0`` 表示贪心。

    返回形状是 ``logits.shape[:-1]``、dtype 为 ``i32``。top-k 与 top-p 都在
    归一化之前截断：被截掉的项取 ``-inf``，归一化后概率严格为 0，不会被采到。
    处理顺序固定为**重复惩罚 → 贪心判定 → top-k → 温度 → 归一化 → top-p**：
    先截断再除温度会让"温度极低"把截断边界之外的项重新放大成候选，顺序
    反了就不是同一个分布。

    ``-inf`` 是**合法的输入**：受约束解码就是靠它把不合法的 token 屏蔽掉。
    拒绝的是 NaN 与 ``+inf``（前向数值已坏），以及"整行都被屏蔽"（无候选）。
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 1 or z.shape[-1] == 0:
        raise ValueError(f"sample_next_token 需要非空的最后一维，得到 {z.shape}")
    if np.any(np.isnan(z)) or np.any(np.isposinf(z)):
        raise ValueError("logits 含 NaN 或 +inf，无法采样；请先检查前向数值")
    if not np.all(np.isfinite(z).any(axis=-1)):
        raise ValueError("logits 的某一行没有可选 token（全被屏蔽了）")

    penalty = float(repetition_penalty)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(
            f"repetition_penalty 必须是有限正数，得到 {repetition_penalty!r}"
        )
    if recent is not None:
        z = _apply_repetition_penalty(z, recent, penalty)
    elif penalty != 1.0:
        raise ValueError(
            "repetition_penalty != 1 时必须提供 recent（已生成/已输入的 token）；"
            "没有历史就没有可惩罚的对象"
        )

    if temperature <= 0.0:
        return np.argmax(z, axis=-1).astype(np.int32)
    temperature = float(temperature)
    if not np.isfinite(temperature):
        raise ValueError(f"temperature 必须是有限数，得到 {temperature!r}")

    if top_k and int(top_k) > 0:
        k = min(int(top_k), z.shape[-1])
        if k < z.shape[-1]:
            cut = np.partition(z, -k, axis=-1)[..., -k]
            z = np.where(z < cut[..., None], -np.inf, z)
    # 先减最大值再除温度：极小温度下 x/temperature 会先溢出成 ±inf，
    # 随后的 inf - inf 变成 NaN，整行概率不可用。减法在前则只有非最大项
    # 变成 -inf（exp 后严格为 0），最大项是 0，结果仍是确定性的 one-hot。
    shifted = z - np.max(z, axis=-1, keepdims=True)
    shifted = shifted / temperature
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    nucleus = float(top_p)
    if not np.isfinite(nucleus) or nucleus <= 0.0:
        raise ValueError(f"top_p 必须是有限正数，得到 {top_p!r}")
    if nucleus < 1.0:
        # 按概率降序累计；"自身之前的累计质量仍小于 top_p"的 token 保留，
        # 因此首次越过阈值的那一个也在内——top_p 很小时不至于把唯一候选切掉。
        order = np.argsort(-probs, axis=-1, kind="stable")
        ordered = np.take_along_axis(probs, order, axis=-1)
        before = np.cumsum(ordered, axis=-1) - ordered
        kept = before < nucleus
        keep = np.empty_like(kept)
        np.put_along_axis(keep, order, kept, axis=-1)
        probs = np.where(keep, probs, 0.0)
        total = np.sum(probs, axis=-1, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError(
                "top_p 截断后没有剩余候选；请提高 top_p 或检查 logits"
            )
        probs /= total

    generator = np.random.default_rng() if rng is None else rng
    cdf = np.cumsum(probs, axis=-1)
    # 用"第一个累计概率超过 u 的位置"，再夹到上界防止浮点误差越界。
    draw = generator.random(probs.shape[:-1])[..., None]
    index = np.sum(cdf < draw, axis=-1)
    return np.minimum(index, probs.shape[-1] - 1).astype(np.int32)



def _masked(logits: np.ndarray, allowed) -> np.ndarray:
    """把 ``allowed`` 之外的 logits 置 ``-inf``，返回 float64 副本。"""
    row = np.array(logits, dtype=np.float64, copy=True)
    allowed = np.fromiter(
        (int(t) for t in allowed), dtype=np.int64, count=len(allowed)
    )
    if allowed.size == 0:
        raise ValueError("约束在这一步没有任何可用 token")
    if allowed.min() < 0 or allowed.max() >= row.shape[-1]:
        raise ValueError(
            f"约束给出的 token 越界（词表 {row.shape[-1]}，"
            f"范围 {allowed.min()}～{allowed.max()}）"
        )
    keep = np.zeros(row.shape[-1], dtype=bool)
    keep[allowed] = True
    row[~keep] = -np.inf
    return row


def _stream_graph(steps: int, spec: LMSpec, head: str):
    """一张固定形状的流式图：``(graph, inputs, state_out)``。

    同一 ``(steps, head)`` 只建一次，循环里反复 execute 同一张图——固定块
    预填充的全部意义就在这里：块长固定，执行计划与编译结果才能跨块复用。
    """
    g = Graph(f"lm_{head}_{steps}")
    inputs, logits, state_out = build_language_model_stateful(
        g, spec.with_steps(steps), with_reset=True, head=head
    )
    if logits is not None:
        g.output(logits, "logits")
    # 状态必须登记成图输出，否则 execute 只回 logits，续接就断了
    for sname, value in state_out.items():
        g.output(value, sname)
    return g, inputs, state_out


def _host_row(backend, row) -> np.ndarray:
    """把（可能驻留在设备上的）logits 行取回主机。

    ``sample_next_token`` 与 ``_masked`` 都是 numpy 实现；CUDA/HIP 的数组
    不能被 ``np.asarray`` 隐式转换。每步取回的只有 ``V`` 个 float，而不是
    ``(B,T,V)`` 整段。
    """
    if isinstance(row, np.ndarray):
        return row
    convert = getattr(backend, "to_host", None)
    return np.asarray(convert(row) if convert is not None else row)


def _check_request(
    spec: LMSpec,
    prompt: Sequence[int],
    chunk_size: Optional[int],
    max_new_tokens: int,
    eos_id: Optional[int],
    constraint,
) -> Tuple[Sequence[int], int, int, int]:
    """校验一次生成请求，返回 ``(prompt, chunk, limit, stop)``。

    ``generate`` 与 ``LMRunner.generate`` 共用同一份校验：两个入口的报错
    文本与判定顺序因此不会分叉。``limit == 0`` 的提前返回由调用方处理
    （它要在建图之前就结束）。
    """
    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prompt 不能为空（至少需要一个 BOS/token）")
    if chunk_size is None:
        chunk = len(prompt)
        if chunk > spec.steps:
            raise ValueError(
                f"prompt 长度 {len(prompt)} 超过 prefill 的 steps={spec.steps}；"
                "请传 chunk_size 分块预填充，或先把 prompt 截断"
            )
    else:
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError(f"chunk_size 必须为正，得到 {chunk}")
        if chunk > spec.steps:
            raise ValueError(
                f"chunk_size={chunk} 超过单块上限 spec.steps={spec.steps}"
            )
    limit = int(max_new_tokens)
    if limit < 0:
        raise ValueError(f"max_new_tokens 不能为负，得到 {limit}")
    stop = int(spec.eos_id if eos_id is None else eos_id)
    if limit == 0:
        # 上限为 0 是"只做 prefill、不生成"的合法请求，不能顺手多吐一个 token
        if constraint is not None and not constraint.done:
            raise ValueError(
                "max_new_tokens=0 时受约束解码不可能走完文法；请提高上限"
            )
    elif constraint is not None and stop != int(spec.eos_id):
        raise ValueError(
            "受约束解码的停止符由文法决定（它只允许 spec.eos_id），"
            f"不能把它改成 {stop}"
        )
    return prompt, chunk, limit, stop


def _require_resident(backend, resident: bool, who: str) -> None:
    """确认后端能做权重驻留；三个入口的报错因此不会分叉。"""
    if not resident:
        return
    put = getattr(backend, "resident_parameters", None)
    drop = getattr(backend, "release_parameters", None)
    if put is None or drop is None:
        raise ValueError(
            "这个后端没有 resident_parameters / release_parameters，"
            "不能把权重登记成常驻"
        )
    if getattr(backend, "resident", None):
        raise ValueError(
            f"后端已有驻留参数；{who} 需要空的驻留表，"
            "避免结束时把调用方的权重一并释放"
        )


def _row_constraints(constraint, batch: int) -> List[Optional[Any]]:
    """把 ``constraint`` 规范成逐行的文法列表（长度等于 ``batch``）。

    三种合法形态：``None``（不受约束）、长度等于 ``batch`` 的序列（元素为
    文法或 ``None``）、以及 ``batch=1`` 时的单个文法。**单个实例不能共享给
    多行**：文法的推进状态是逐序列的，共享会把两条序列的写法搅在一起。
    """
    if constraint is None:
        return [None] * batch
    if isinstance(constraint, Sequence) and not isinstance(constraint, (str, bytes)):
        items = list(constraint)
        if len(items) != batch:
            raise ValueError(
                f"受约束解码的 constraint 序列有 {len(items)} 项，"
                f"与 batch={batch} 不符；每行要一份独立的文法实例（或 None）"
            )
    else:
        if batch != 1:
            raise ValueError(
                "受约束解码在批量里需要逐行的文法实例："
                f"batch={batch} 时要传入长度相同的序列（元素为文法或 None），"
                "不能共用一个实例（文法状态是逐序列的）"
            )
        items = [constraint]
    for index, item in enumerate(items):
        if item is None:
            continue
        for attr in ("allowed", "feed", "done"):
            if not hasattr(item, attr):
                raise ValueError(
                    f"受约束解码的第 {index} 行对象缺少 {attr}；"
                    "需要 models.lm_tools 的文法接口（allowed / feed / done）"
                )
    return items


def _check_batch_request(
    spec: LMSpec,
    prompts: Sequence[Sequence[int]],
    chunk_size: Optional[int],
    max_new_tokens: int,
    eos_id: Optional[int],
    constraint,
) -> Tuple[List[List[int]], int, int, int, List[Optional[Any]]]:
    """校验一次批量请求，返回 ``(prompts, chunk, limit, stop, constraints)``。

    逐条复用 :func:`_check_request` 的规则（空 prompt、块长、上限、停止符），
    因此单条与批量的报错文本一致。受约束解码按行接受：``constraint`` 可以
    是长度等于 ``batch`` 的序列（元素为文法或 ``None``），``batch=1`` 时也
    可以直接给一个文法实例；单个实例共享给多行是显式拒绝的。
    """
    rows = [[int(token) for token in row] for row in prompts]
    if not rows:
        raise ValueError("prompts 不能为空（至少需要一条序列）")
    if len(rows) != spec.batch:
        raise ValueError(
            f"prompts 有 {len(rows)} 条，与 spec.batch={spec.batch} 不符；"
            "图按固定 batch 构建，请用匹配的 spec 或补齐序列"
        )
    constraints = _row_constraints(constraint, spec.batch)
    checked = [
        _check_request(spec, row, chunk_size, max_new_tokens, eos_id, held)
        for row, held in zip(rows, constraints)
    ]
    _, chunk, limit, stop = checked[0]
    if chunk_size is None:
        # "整段一次算完"对批量意味着按最长的那条来：用第一条的长度会让
        # 短 prompt 决定所有行的块长，白白多切几块。
        chunk = max(len(row) for row in checked)
    return [item[0] for item in checked], chunk, limit, stop, constraints


def _run_stateless(
    backend,
    params: Mapping[str, np.ndarray],
    spec: LMSpec,
    prompts: Sequence[Sequence[int]],
    *,
    chunk: int,
    limit: int,
    stop: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    constraint,
    resident: bool,
    who: str,
) -> List[List[int]]:
    """建图缓存与权重驻留的"用完即放"生命周期，两个无状态入口共用。"""
    _require_resident(backend, resident, who)
    cache = {}
    owned = {"flag": False}

    def graph_for(length: int, head: str):
        key = (length, head)
        if key not in cache:
            cache[key] = _stream_graph(length, spec, head)
        graph, inputs, state_out = cache[key]
        if resident and not owned["flag"]:
            backend.resident_parameters(graph, params)
            owned["flag"] = True
        return graph, inputs, state_out

    try:
        return _decode(
            backend,
            params,
            prompts,
            spec,
            chunk=chunk,
            limit=limit,
            stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            constraint=constraint,
            resident=resident,
            graph_for=graph_for,
        )
    finally:
        if owned["flag"]:
            backend.release_parameters()


def generate(
    backend,
    params: Mapping[str, np.ndarray],
    prompt: Sequence[int],
    spec: Optional[LMSpec] = None,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
    seed: int = 0,
    constraint=None,
    chunk_size: Optional[int] = None,
    resident: bool = False,
) -> Sequence[int]:
    """固定块预填充 → 逐 token decode，直到 EOS 或 ``max_new_tokens``。

    ``temperature <= 0`` 是贪心解码（与 :func:`sample_next_token` 同一约定）。
    返回**生成的部分**，不含 prompt。

    ``top_p`` / ``repetition_penalty`` 与 :func:`sample_next_token` 同义：
    top-p 在归一化前截断；重复惩罚的历史是 **prompt + 已生成 token**（与
    HuggingFace 的口径一致），贪心解码同样生效——它改的是 logits，不是采样。

    只支持 ``batch == 1``：批内多序列的停止点不同，逐 token 解码会退化成
    "最短序列决定整批"，那是另一个要显式设计的问题，不该在这里悄悄发生。
    需要多条序列时用 :func:`generate_batch`——它按左侧补 pad 对齐，每行独立
    停止，语义是显式定义的，不靠"最短序列兜底"。

    ``chunk_size``
        prefill 的固定块长。``None`` 表示整段一次算完（旧行为，要求
        ``len(prompt) <= spec.steps``）。给定时 prompt 可以远长于
        ``spec.steps``：中间块只带状态、不算词表头（``head="none"``），
        最后一块只对末位置投影（``head="last"``）。块长必须是同一张图，
        执行计划才能跨块复用；尾块不足一块时按实际长度构图，避免用 pad
        污染状态。

    ``resident``
        把权重一次性上传到设备并登记常驻（后端需提供
        ``resident_parameters`` / ``release_parameters``）。要求后端驻留表
        为空，结束时释放——不会悄悄覆盖调用方已有的驻留权重。

    ``constraint`` 是受约束解码的文法（``models/lm_tools.py`` 的
    :class:`JsonGrammar`）：每步先用它的 ``allowed()`` 屏蔽 logits，采样后再
    ``feed`` 推进状态。**约束从第一个生成 token 起生效**，prompt 里不要预置
    半个结构——那些字节文法还没见过。文法完成时它只允许 EOS，因此
    ``eos_id`` 必须与文法用的那个一致，不能在这里改成别的停止符。
    """
    spec = spec or LMSpec()
    if spec.batch != 1:
        raise ValueError(
            f"generate 只支持 batch=1，得到 {spec.batch}；"
            "多条序列请用 generate_batch"
        )
    prompt, chunk, limit, stop = _check_request(
        spec, prompt, chunk_size, max_new_tokens, eos_id, constraint
    )
    if limit == 0:
        return []
    return _run_stateless(
        backend,
        params,
        spec,
        [prompt],
        chunk=chunk,
        limit=limit,
        stop=stop,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        constraint=constraint,
        resident=resident,
        who="generate(resident=True)",
    )[0]


def generate_batch(
    backend,
    params: Mapping[str, np.ndarray],
    prompts: Sequence[Sequence[int]],
    spec: Optional[LMSpec] = None,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
    seed: int = 0,
    constraint=None,
    chunk_size: Optional[int] = None,
    resident: bool = False,
) -> List[List[int]]:
    """批量解码：``B`` 条序列共用一张图，每条独立停止。

    ``prompts`` 的条数必须等于 ``spec.batch``（图按固定 batch 构建）。长度
    不同的 prompt 按**左侧补 ``pad_id``** 对齐：每行首个真实 token 的位置置
    ``reset=1``，pad 既不推进状态也不进入卷积窗口，所以行与行之间互不影响
    ——这正是 ``train.lm.lm_batch`` 的同一套约定，不是近似。

    每行的采样器相互独立：第 ``i`` 行用 ``seed + i`` 派生，贪心（
    ``temperature <= 0``）时不消耗随机数。停止与上限也逐行判定：某一行
    遇到 EOS 或攒够 ``max_new_tokens`` 后就不再采样，其余行照常继续；已结束
    的行在后续 decode 里喂 ``pad_id``，它的输出被丢弃。

    返回**生成的部分**（不含 prompt），形状是 ragged 的 ``List[List[int]]``，
    第 ``i`` 条就是第 ``i`` 行的结果。

    受约束解码（``constraint``）**按行接受**：给一个长度等于 ``spec.batch``
    的序列，元素是该行的文法实例或 ``None``（该行不受约束）。文法的推进
    状态是逐序列的，因此不能把同一个实例共享给多行；``batch=1`` 时仍可直接
    传单个文法（等价于长度为 1 的序列）。
    """
    spec = spec or LMSpec()
    checked, chunk, limit, stop, constraints = _check_batch_request(
        spec, prompts, chunk_size, max_new_tokens, eos_id, constraint
    )
    if limit == 0:
        return [[] for _ in checked]
    return _run_stateless(
        backend,
        params,
        spec,
        checked,
        chunk=chunk,
        limit=limit,
        stop=stop,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        constraint=constraints,
        resident=resident,
        who="generate_batch(resident=True)",
    )


def _host_array(backend, value) -> np.ndarray:
    """把（可能驻留在设备上的）任意数组取回主机；束搜索按 batch 维重排状态要用。"""
    if isinstance(value, np.ndarray):
        return value
    convert = getattr(backend, "to_host", None)
    if convert is None:
        raise ValueError(
            "束搜索需要后端提供 to_host：状态要在 batch 维按父假设重排，"
            "没有回传就没有可索引的数组"
        )
    return np.asarray(convert(value))


def _log_softmax(z: np.ndarray) -> np.ndarray:
    """按最后一维做 log-softmax；``-inf`` 列保持严格 0 概率。"""
    shifted = z - np.max(z, axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_z


def generate_beam(
    backend,
    params: Mapping[str, np.ndarray],
    prompt: Sequence[int],
    spec: Optional[LMSpec] = None,
    *,
    num_beams: int = 4,
    max_new_tokens: int = 16,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    chunk_size: Optional[int] = None,
    resident: bool = False,
) -> Sequence[int]:
    """束搜索：同时维护 ``num_beams`` 条假设，返回归一化分数最高的那一条。

    与 :func:`generate` 的差别是**选路规则**而不是采样：每一步对
    ``(num_beams, V)`` 的 log-softmax 加前缀分数，取全局前 ``num_beams``
    个候选；生成到 EOS 的假设进入完成池、不再占活动槽位
    （所以候选池按 ``2 * num_beams`` 取，避免 EOS 扎堆时活动槽位填不满）。

    ``length_penalty`` 按 ``score / len**p`` 归一化：``p=0`` 是纯对数概率和
    （偏向短句），``p=1`` 是平均对数概率，``p>1`` 更偏向长句。分数比较用
    稳定排序，因此完全并列时取先生成的那条，结果是确定的。

    ``num_beams=1`` 时退化成贪心解码，与 ``generate(temperature=0)`` 逐位一致
    —— 这一点有测试固定住。

    束搜索是**单条序列**的算法：``spec.batch`` 必须为 1；内部换成
    ``spec.with_batch(num_beams)`` 的图，把 k 条假设放进 batch 维。每一行
    用各自的父状态重新排布（主机侧 gather，显式的代价），因此图本身仍是
    静态形状、可复用执行计划。

    ``repetition_penalty`` 的历史是**每条假设各自**的 prompt + 已生成 token。
    受约束解码不在束搜索里提供：调用方需要约束时用 ``generate``。
    """
    spec = spec or LMSpec()
    if isinstance(num_beams, bool) or not isinstance(num_beams, int):
        raise TypeError(f"num_beams 必须是 int，得到 {num_beams!r}")
    if num_beams < 1:
        raise ValueError(f"num_beams 必须 >= 1，得到 {num_beams}")
    if spec.batch != 1:
        raise ValueError(
            f"generate_beam 需要 spec.batch=1，得到 {spec.batch}；"
            "束宽由 num_beams 决定，不再借用批量维"
        )
    penalty = float(repetition_penalty)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(
            f"repetition_penalty 必须是有限正数，得到 {repetition_penalty!r}"
        )
    power = float(length_penalty)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError(f"length_penalty 必须是有限的非负数，得到 {length_penalty!r}")
    prompt, chunk, limit, stop = _check_request(
        spec, prompt, chunk_size, max_new_tokens, eos_id, None
    )
    if limit == 0:
        return []
    beam_spec = spec.with_batch(num_beams)
    _require_resident(backend, resident, "generate_beam(resident=True)")
    cache: Dict[Tuple[int, str], Any] = {}
    owned = {"flag": False}

    def graph_for(length: int, head: str):
        key = (length, head)
        if key not in cache:
            cache[key] = _stream_graph(length, beam_spec, head)
        graph, inputs, state_out = cache[key]
        if resident and not owned["flag"]:
            backend.resident_parameters(graph, params)
            owned["flag"] = True
        return graph, inputs, state_out

    try:
        return _beam_decode(
            backend,
            params,
            prompt,
            beam_spec,
            beam_width=num_beams,
            limit=limit,
            stop=stop,
            length_penalty=power,
            early_stopping=early_stopping,
            repetition_penalty=penalty,
            chunk=chunk,
            resident=resident,
            graph_for=graph_for,
        )
    finally:
        if owned["flag"]:
            backend.release_parameters()


def _beam_decode(
    backend,
    params: Mapping[str, np.ndarray],
    prompt: Sequence[int],
    spec: LMSpec,
    *,
    beam_width: int,
    limit: int,
    stop: int,
    length_penalty: float,
    early_stopping: bool,
    repetition_penalty: float,
    chunk: int,
    resident: bool,
    graph_for,
) -> List[int]:
    """束搜索的实现：``spec.batch`` 就是 ``beam_width``。

    状态、分数、token 序列都按行索引；每步用父行下标把状态 gather 到
    新顺序。死槽位（分数 ``-inf``）喂 ``pad_id`` 并沿用父行 0 的状态，
    它们的结果被丢弃——图是固定 batch，不能少喂一行。
    """
    rows = [list(prompt) for _ in range(beam_width)]
    state, logits = _prefill(
        backend, params, rows, spec, chunk=chunk, resident=resident,
        graph_for=graph_for,
    )
    scores = np.full(beam_width, -np.inf, dtype=np.float64)
    scores[0] = 0.0
    tokens: List[List[int]] = [[] for _ in range(beam_width)]
    histories: List[List[int]] = [list(prompt) for _ in range(beam_width)]
    done: List[Tuple[float, List[int]]] = []
    vocab = spec.vocab_size

    def normalise(score: float, length: int) -> float:
        return score / float(max(length, 1) ** length_penalty)

    row_logits = logits[:, -1, :]
    dec_graph = None
    dec_inputs = None
    for step in range(limit):
        logp = np.full((beam_width, vocab), -np.inf, dtype=np.float64)
        for index in range(beam_width):
            if tokens[index] is None or not np.isfinite(scores[index]):
                continue
            row = _host_row(backend, row_logits[index])
            if not np.isfinite(row).any():
                raise ValueError(
                    f"束搜索第 {index} 条假设没有可选 token（logits 全被屏蔽）"
                )
            if repetition_penalty != 1.0:
                row = _apply_repetition_penalty(
                    row, histories[index], repetition_penalty
                )
            logp[index] = _log_softmax(row)
        flat = (scores[:, None] + logp).reshape(-1)
        # 1 个 EOS 会从候选池里剔掉但不填活动槽位，因此多取一倍候选。
        order = np.argsort(-flat, kind="stable")[: 2 * beam_width]
        new_scores = np.full(beam_width, -np.inf, dtype=np.float64)
        new_tokens: List[Optional[List[int]]] = [None] * beam_width
        new_histories: List[Optional[List[int]]] = [None] * beam_width
        parents = np.zeros(beam_width, dtype=np.int64)
        filled = 0
        for index in order:
            if filled >= beam_width:
                break
            value = float(flat[index])
            if not np.isfinite(value):
                break
            parent, token = divmod(int(index), vocab)
            if token == stop:
                done.append((normalise(value, len(tokens[parent]) + 1),
                             list(tokens[parent])))
                continue
            new_scores[filled] = value
            new_tokens[filled] = tokens[parent] + [token]
            new_histories[filled] = histories[parent] + [token]
            parents[filled] = parent
            filled += 1
        if filled == 0:
            break
        scores = new_scores
        tokens = new_tokens
        histories = [item if item is not None else [] for item in new_histories]
        if early_stopping and len(done) >= beam_width:
            break
        if step == limit - 1:
            break

        if dec_graph is None:
            dec_graph, dec_inputs, _ = graph_for(1, "last")
        state = {name: _host_array(backend, value)[parents]
                 for name, value in state.items()}
        ids = np.full((beam_width, 1), spec.pad_id, dtype=np.int32)
        for index in range(filled):
            ids[index, 0] = tokens[index][-1]
        feed = {} if resident else dict(params)
        feed[dec_inputs["ids"].name] = ids
        feed[dec_inputs["reset"].name] = np.zeros((beam_width, 1), dtype=np.float32)
        for name in spec.state_shapes:
            feed[dec_inputs[name].name] = state[name]
        out = backend.execute(dec_graph, feed)
        state = spec.carry_state(out)
        row_logits = out["logits"][:, 0, :]

    candidates = list(done)
    for index in range(beam_width):
        if tokens[index] is not None and np.isfinite(scores[index]):
            candidates.append(
                (normalise(scores[index], len(tokens[index])), list(tokens[index]))
            )
    if not candidates:
        return []
    best = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
    return candidates[best][1]


def _decode(
    backend,
    params: Mapping[str, np.ndarray],
    prompts: Sequence[Sequence[int]],
    spec: LMSpec,
    *,
    chunk: int,
    limit: int,
    stop: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    constraint,
    resident: bool,
    graph_for,
) -> List[List[int]]:
    """解码循环：图与驻留的生命周期由调用方决定。

    权重登记与释放不在这里做——``generate`` 用完即放，``LMRunner`` 跨请求
    保留。两条生命周期的数值路径必须完全相同，所以这里只认两个事实：权重
    是否已驻留（``resident``）与图从哪里来（``graph_for``）。

    多序列按**左侧补 ``pad_id``** 对齐到同一宽度：每行首个真实 token 的
    位置置 ``reset=1``，pad 既不推进状态也不进入卷积窗口（与
    ``train.lm.lm_batch`` 同一约定）。因此行与行互不影响，``prompts`` 的
    长度可以不同；已停止的行在后续 decode 里喂 pad，其结果被丢弃。
    """
    state, logits = _prefill(
        backend,
        params,
        prompts,
        spec,
        chunk=chunk,
        resident=resident,
        graph_for=graph_for,
    )
    return _sample(
        backend,
        params,
        logits,
        spec,
        prompts=prompts,
        state=state,
        limit=limit,
        stop=stop,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        constraint=constraint,
        resident=resident,
        graph_for=graph_for,
    )


def _prefill(
    backend,
    params: Mapping[str, np.ndarray],
    prompts: Sequence[Sequence[int]],
    spec: LMSpec,
    *,
    chunk: int,
    resident: bool,
    graph_for,
    consumed: Optional[Sequence[int]] = None,
    state=None,
):
    """把 ``prompts`` 从 ``consumed`` 位置起送进图，返回 ``(state, logits)``。

    ``state is None`` 表示新序列：多行按左侧补 ``pad_id`` 对齐，每行首个真实
    token 处置 ``reset=1``，行与行互不影响。``state`` 不为 ``None`` 表示接着
    一段已消费的前缀继续：此时**不允许**左填充（pad 会推进状态，而 reset 又
    会把已消费的历史清掉），各行剩余的 token 数必须一致。
    """
    batch = spec.batch
    consumed = list(consumed) if consumed is not None else [0] * batch
    remain = [len(prompts[row]) - consumed[row] for row in range(batch)]
    if any(count <= 0 for count in remain):
        raise ValueError(
            "增量前缀至少要包含一个未消费的 token；前缀已经全部消费时"
            "直接采样末位置即可"
        )
    width = max(remain)
    pads = [width - count for count in remain]
    fresh = state is None
    if not fresh and any(pads):
        raise ValueError(
            "增量前缀要求各行长度一致：左侧补 pad 会推进状态，"
            "而 reset 又会把已消费的历史清掉"
        )

    # 固定块：中间块只带状态，最后一块只投影末位置。尾块按实际长度构图，
    # 不用 pad 填满——pad 会推进状态，而 reset 又会连带清掉真实历史。
    blocks = [(start, min(chunk, width - start))
              for start in range(0, width, chunk)]
    logits = None
    for index, (start, length) in enumerate(blocks):
        is_last = index == len(blocks) - 1
        graph, inputs, _ = graph_for(length, "last" if is_last else "none")
        ids = np.full((batch, length), spec.pad_id, dtype=np.int32)
        reset = np.zeros((batch, length), dtype=np.float32)
        for row in range(batch):
            # 该行在本块内的偏移：负数表示块从真实 prompt 中间开始，
            # 提前于真实起点表示这一块整段都是左填充。
            offset = pads[row] - start
            lo = max(offset, 0)
            hi = min(offset + remain[row], length)
            if hi > lo:
                begin = consumed[row] + lo - offset
                ids[row, lo:hi] = prompts[row][begin:begin + hi - lo]
            if fresh and 0 <= offset < length:
                reset[row, offset] = 1.0
        feed = {} if resident else dict(params)
        feed[inputs["ids"].name] = ids
        # 新序列从零状态开始，第一块的位置 0 就是序列起点（与训练侧同一套
        # reset 语义）；续接前缀时首块必须用传进来的状态，且不重置。
        if index == 0:
            if state is None:
                for name, shape in spec.state_shapes.items():
                    feed[inputs[name].name] = np.zeros(shape, dtype=np.float32)
            else:
                for name in spec.state_shapes:
                    feed[inputs[name].name] = state[name]
        else:
            for name in spec.state_shapes:
                feed[inputs[name].name] = state[name]
        feed[inputs["reset"].name] = reset
        out = backend.execute(graph, feed)
        state = spec.carry_state(out)
        if is_last:
            logits = out["logits"]
    return state, logits


def _sample(
    backend,
    params: Mapping[str, np.ndarray],
    logits,
    spec: LMSpec,
    *,
    prompts: Sequence[Sequence[int]],
    state,
    limit: int,
    stop: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    constraint,
    resident: bool,
    graph_for,
) -> List[List[int]]:
    """从 ``logits`` 的末位置起逐 token 解码，返回每行的生成部分。

    ``state`` 是 ``_prefill`` 交出的状态；本函数只管采样与推进，不再关心
    前缀是从哪里来的（全量 prompt 还是缓存前缀）。

    ``prompts`` 只用于重复惩罚的历史：每行从自己的 prompt 起算，再追加它
    自己生成的 token——行与行之间不共享历史（批量里左填充的 pad 也不进历史，
    它从来没有被"生成"过）。
    """
    batch = spec.batch
    constraints = _row_constraints(constraint, batch)
    rngs = [np.random.default_rng(seed + index) for index in range(batch)]
    generated: List[List[int]] = [[] for _ in range(batch)]
    history: List[List[int]] = [list(row) for row in prompts]
    active = [True] * batch

    def _pick(index: int, logits_row) -> int:
        row = _host_row(backend, logits_row)
        held = constraints[index]
        if held is not None:
            row = _masked(row, held.allowed())
        token = int(sample_next_token(
            row,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent=history[index],
            rng=rngs[index],
        ))
        if held is not None:
            held.feed(token)
        return token

    def _finish():
        # 半截 JSON 比"报错"危险得多：调用方会把它当成功结果去解析。
        for index, held in enumerate(constraints):
            if held is not None and not held.done:
                raise ValueError(
                    f"第 {index} 行的受约束解码没有在 max_new_tokens 内走完文法；"
                    "请提高上限，或检查 prompt 是否预置了半个结构"
                    "（约束从第一个生成 token 起生效）"
                )
        return generated

    # 本轮只采一次：结束后不再采样的行会进入 inactive，不再消耗随机数。
    last: List[Optional[int]] = [None] * batch
    for row in range(batch):
        token = _pick(row, logits[row, -1, :])
        if token == stop:
            active[row] = False
            continue
        generated[row].append(token)
        history[row].append(token)
        last[row] = token
        if len(generated[row]) >= limit:
            active[row] = False
    if not any(active):
        return _finish()

    dec_graph, dec_inputs, _ = graph_for(1, "last")
    for _ in range(limit - 1):
        if not any(active):
            break
        ids = np.full((batch, 1), spec.pad_id, dtype=np.int32)
        for row in range(batch):
            if active[row]:
                ids[row, 0] = last[row]
        step_feed = {} if resident else dict(params)
        step_feed[dec_inputs["ids"].name] = ids
        # decode 不重置：这一段必须接着 prefill 的状态
        step_feed[dec_inputs["reset"].name] = np.zeros((batch, 1), dtype=np.float32)
        for name in spec.state_shapes:
            step_feed[dec_inputs[name].name] = state[name]
        out = backend.execute(dec_graph, step_feed)
        state = spec.carry_state(out)
        for row in range(batch):
            if not active[row]:
                continue
            token = _pick(row, out["logits"][row, 0, :])
            if token == stop:
                active[row] = False
                continue
            generated[row].append(token)
            history[row].append(token)
            last[row] = token
            if len(generated[row]) >= limit:
                active[row] = False
    return _finish()


class LMRunner:
    """跨请求复用的推理会话：图缓存 + 可选权重驻留 + 同一套解码循环。

    ``generate`` 每次调用都重新构图、重新上传权重，换来的是"无状态、可随意
    并发"。代价在 5.187 的基准里是可见的：TTFT 里 200～250 ms 是构图与首次
    捕获的固定成本，1K prompt 的吞吐反而最低。本类把这两笔成本移出请求：
    同一 ``(长度, head)`` 的图只建一次，``resident=True`` 时权重只上传一次，
    直到 :meth:`close`。

    **数值路径与 ``generate`` 完全相同**：两者共用同一个解码循环，差异只在
    图与权重的生命周期。因此"会话里第一次跑"与"直接调 ``generate``"逐位
    一致，第二条请求只是省掉了重复成本。

    ``max_graphs`` 给图缓存封顶：尾块长度随 prompt 变化，不封顶的缓存会随
    请求数无界增长。超出时按插入顺序淘汰最久未用的那张（``OrderedDict``），
    被淘汰的图下次请求重建即可，不影响正确性。

    会话绑定**一个固定的 batch**（图按 ``spec.batch`` 构建）：单条序列用
    :meth:`generate`（要求 ``spec.batch == 1``），多条序列用
    :meth:`generate_batch`（要求 ``len(prompts) == spec.batch``）。两者命中
    同一份图缓存，只是请求条数不同。

    ``prime`` 把一段前缀（例如固定的 system prompt）预先算完并保存状态，
    之后的 ``generate`` 只把新增 token 送进图。前缀必须是新 prompt 的真前缀，
    否则显式报错；生成结束后缓存自动失效（decode 推进过状态），需要重新
    ``prime``。批量会话不做前缀续接。

    ``resident=True`` 要求后端提供 ``resident_parameters`` /
    ``release_parameters``，且进入时会话看到的后端驻留表必须为空——
    会话结束时它只会释放自己登记的那一份，不清掉别人的权重。
    权重更新后要重建会话（或显式 ``close`` 后新建）：本类不做"检测到权重
    变化就悄悄重传"，那会让"用旧权重算出看起来正常的结果"更难发现。
    """

    def __init__(
        self,
        backend,
        params: Mapping[str, np.ndarray],
        spec: Optional[LMSpec] = None,
        *,
        resident: bool = False,
        max_graphs: int = 8,
    ) -> None:
        spec = spec or LMSpec()
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int):
            raise TypeError(f"max_graphs 必须是 int，得到 {max_graphs!r}")
        if max_graphs < 1:
            raise ValueError(f"max_graphs 必须 >= 1，得到 {max_graphs}")
        if resident:
            _require_resident(backend, True, "LMRunner(resident=True)")
        self.backend = backend
        self.params = dict(params)
        self.spec = spec
        self.resident = bool(resident)
        self.max_graphs = max_graphs
        self._graphs: "OrderedDict[Tuple[int, str], Any]" = OrderedDict()
        self._resident_ready = False
        self._closed = False
        # 缓存前缀：最近一次 ``prime`` 已消费的 token 与带出的状态。
        self._prefix: List[int] = []
        self._state = None
        self._logits = None

    # -- 图与驻留 -----------------------------------------------------

    def _graph_for_batch(self, spec: LMSpec, length: int, head: str):
        """按 ``spec.batch`` 取图：束搜索与会话的 batch 不同，键必须带上它。"""
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        key = (spec.batch, length, head)
        item = self._graphs.get(key)
        if item is None:
            item = _stream_graph(length, spec, head)
            self._graphs[key] = item
            while len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        if self.resident and not self._resident_ready:
            # 驻留表是"输入名 → 数组"，与具体哪张图无关；用第一张图校验
            # 名字与形状即可，后续图直接共享。
            self.backend.resident_parameters(item[0], self.params)
            self._resident_ready = True
        return item

    def _graph_for(self, length: int, head: str):
        return self._graph_for_batch(self.spec, length, head)

    @property
    def graph_count(self) -> int:
        """当前缓存的图数量（``max_graphs`` 是它的上界）。"""
        return len(self._graphs)

    # -- 生成 ---------------------------------------------------------

    def prime(self, prompt: Sequence[int], *, chunk_size: Optional[int] = None) -> int:
        """预先消费一段前缀并保存状态，后续 ``generate`` 只算新增的 token。

        典型用法是服务里的固定 system prompt：``prime`` 一次，之后每次请求
        只把新用户输入接上去。返回已消费的 token 数。

        再次 ``prime`` 时新 prompt 必须**以已缓存前缀开头**（这是缓存能复用
        的前提）：更长的 prompt 只跑增量，等长直接返回，变短或分叉都显式
        报错并要求先 :meth:`reset`。

        只支持 ``batch == 1``：续接要求各行增量等长，而左侧补 pad 会推进
        状态、破坏已消费的历史，批量路径不做前缀续接。
        """
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        if self.spec.batch != 1:
            raise ValueError(
                "prime 只支持 batch=1 的会话：续接前缀要求各行增量等长，"
                "而左侧补 pad 会破坏已消费的历史；批量请直接 generate_batch"
            )
        prompt, chunk, _, _ = _check_request(
            self.spec, prompt, chunk_size, 1, None, None
        )
        if self._prefix:
            cached = len(self._prefix)
            if len(prompt) < cached or list(prompt[:cached]) != self._prefix:
                raise ValueError(
                    "prime 的新 prompt 必须以已缓存前缀开头；要换前缀先 reset()"
                )
            if len(prompt) == cached:
                return cached
            consumed, state = [cached], self._state
        else:
            consumed, state = [0], None
        self._state, self._logits = _prefill(
            self.backend,
            self.params,
            [prompt],
            self.spec,
            chunk=chunk,
            resident=self.resident,
            graph_for=self._graph_for,
            consumed=consumed,
            state=state,
        )
        self._prefix = list(prompt)
        return len(prompt)

    def reset(self) -> None:
        """丢弃缓存前缀与状态；图缓存与驻留权重保留。"""
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        self._clear_prefix()

    def _clear_prefix(self) -> None:
        self._prefix = []
        self._state = None
        self._logits = None

    def _advance(self, prompt: Sequence[int], chunk: int):
        """把会话推进到 ``prompt`` 末尾，返回 ``(state, logits)``。

        没有缓存前缀时等价于一次全量预填充；有缓存前缀时只把新增 token 送
        进图，前缀部分直接复用已保存的状态。
        """
        if not self._prefix:
            return _prefill(
                self.backend,
                self.params,
                [prompt],
                self.spec,
                chunk=chunk,
                resident=self.resident,
                graph_for=self._graph_for,
            )
        cached = len(self._prefix)
        if len(prompt) < cached or list(prompt[:cached]) != self._prefix:
            raise ValueError(
                "prompt 必须以会话已缓存的前缀开头；要换前缀先 reset()"
            )
        if len(prompt) == cached:
            return self._state, self._logits
        return _prefill(
            self.backend,
            self.params,
            [prompt],
            self.spec,
            chunk=chunk,
            resident=self.resident,
            graph_for=self._graph_for,
            consumed=[cached],
            state=self._state,
        )

    def generate(
        self,
        prompt: Sequence[int],
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        seed: int = 0,
        constraint=None,
        chunk_size: Optional[int] = None,
    ) -> Sequence[int]:
        """与模块级 :func:`generate` 同参同返回值，只复用图与驻留权重。

        若会话里有 :meth:`prime` 缓存的前缀，``prompt`` 必须包含它，前缀
        部分不再重算。生成结束后前缀与状态一并丢弃（decode 已把状态推进到
        生成序列末尾，继续复用会让"前缀"与状态对不上）。
        """
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        if self.spec.batch != 1:
            raise ValueError(
                f"这个会话按 batch={self.spec.batch} 构建；"
                "单条序列需要 batch=1 的 spec，或多条一起用 generate_batch"
            )
        prompt, chunk, limit, stop = _check_request(
            self.spec, prompt, chunk_size, max_new_tokens, eos_id, constraint
        )
        if limit == 0:
            return []
        state, logits = self._advance(prompt, chunk)
        # 无论生成是否成功，状态都已经不是"前缀末尾"的那一份了。
        self._clear_prefix()
        return _sample(
            self.backend,
            self.params,
            logits,
            self.spec,
            prompts=[prompt],
            state=state,
            limit=limit,
            stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            constraint=constraint,
            resident=self.resident,
            graph_for=self._graph_for,
        )[0]

    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        seed: int = 0,
        constraint=None,
        chunk_size: Optional[int] = None,
    ) -> List[List[int]]:
        """与模块级 :func:`generate_batch` 同参同返回值，只复用图与驻留权重。"""
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        if self._prefix:
            raise ValueError(
                "会话里已有缓存前缀；请先 reset() 再批量生成"
                "（批量路径不做前缀续接）"
            )
        checked, chunk, limit, stop, constraints = _check_batch_request(
            self.spec, prompts, chunk_size, max_new_tokens, eos_id, constraint
        )
        if limit == 0:
            return [[] for _ in checked]
        return _decode(
            self.backend,
            self.params,
            checked,
            self.spec,
            chunk=chunk,
            limit=limit,
            stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            constraint=constraints,
            resident=self.resident,
            graph_for=self._graph_for,
        )

    def generate_beam(
        self,
        prompt: Sequence[int],
        *,
        num_beams: int = 4,
        max_new_tokens: int = 16,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> Sequence[int]:
        """与模块级 :func:`generate_beam` 同参同返回值，复用同一份图缓存。

        束宽对应的图按 ``(batch=num_beams, 长度, head)`` 缓存在会话里，和
        单条生成的图互不覆盖；``resident=True`` 时权重也只有一份。
        有缓存前缀时显式拒绝（先 ``reset()``）：束搜索的 k 条假设会把状态
        推进到分叉后的不同末尾，前缀无法再代表它们。
        """
        if self._closed:
            raise ValueError("LMRunner 已关闭，请新建一个会话")
        if self.spec.batch != 1:
            raise ValueError(
                f"这个会话按 batch={self.spec.batch} 构建；束搜索需要 batch=1 的会话"
            )
        if self._prefix:
            raise ValueError(
                "会话里已有缓存前缀；请先 reset() 再束搜索"
                "（k 条假设会把状态推进到不同末尾，前缀无法复用）"
            )
        if isinstance(num_beams, bool) or not isinstance(num_beams, int):
            raise TypeError(f"num_beams 必须是 int，得到 {num_beams!r}")
        if num_beams < 1:
            raise ValueError(f"num_beams 必须 >= 1，得到 {num_beams}")
        penalty = float(repetition_penalty)
        if not np.isfinite(penalty) or penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty 必须是有限正数，得到 {repetition_penalty!r}"
            )
        power = float(length_penalty)
        if not np.isfinite(power) or power < 0.0:
            raise ValueError(
                f"length_penalty 必须是有限的非负数，得到 {length_penalty!r}"
            )
        prompt, chunk, limit, stop = _check_request(
            self.spec, prompt, chunk_size, max_new_tokens, eos_id, None
        )
        if limit == 0:
            return []
        beam_spec = self.spec.with_batch(num_beams)
        return _beam_decode(
            self.backend,
            self.params,
            prompt,
            beam_spec,
            beam_width=num_beams,
            limit=limit,
            stop=stop,
            length_penalty=power,
            early_stopping=early_stopping,
            repetition_penalty=penalty,
            chunk=chunk,
            resident=self.resident,
            graph_for=lambda length, head: self._graph_for_batch(
                beam_spec, length, head
            ),
        )

    # -- 生命周期 -----------------------------------------------------

    def close(self) -> None:
        """释放本会话登记的驻留权重并丢弃图缓存；可重复调用。"""
        if self._closed:
            return
        if self._resident_ready:
            self.backend.release_parameters()
            self._resident_ready = False
        self._graphs.clear()
        self._clear_prefix()
        self._closed = True

    def __enter__(self) -> "LMRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"LMRunner(graphs={len(self._graphs)}, resident={self.resident}, "
            f"closed={self._closed})"
        )


class _ServingRequest:
    """一次推理请求在调度器里的全部状态。

    字段分三类：**契约**（prompt / limit / stop / 采样参数 / 种子，submit 之后
    只读）、**会话**（state / logits / feed_token，由 `step` 推进）、**输出**
    （generated / history，只追加）。历史与 `generate` 同口径：prompt + 已生成
    token，因此重复惩罚在两种入口下逐位一致。
    """

    __slots__ = (
        "rid",
        "prompt",
        "limit",
        "stop",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "seed",
        "constraint",
        "rng",
        "history",
        "generated",
        "state",
        "logits",
        "feed_token",
    )

    def __init__(
        self,
        rid: int,
        prompt: List[int],
        limit: int,
        stop: int,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        constraint,
    ) -> None:
        self.rid = rid
        self.prompt = prompt
        self.limit = limit
        self.stop = stop
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.constraint = constraint
        self.rng = np.random.default_rng(seed)
        self.history = list(prompt)
        self.generated: List[int] = []
        self.state: Optional[Dict[str, Any]] = None
        self.logits = None
        self.feed_token: Optional[int] = None


class ServingEngine:
    """请求队列 + 连续批处理的解码调度器（在 :class:`LMRunner` 之上）。

    **它解决什么**：`generate_batch` 要求"一批请求同时到齐、长度补齐、一起
    跑完"。真实服务不是这样——请求随时到达、长度各异、完成时间不同。这里的
    调度单位是**解码步**而不是请求：

    1. 先用上一轮交出的 logits 为每个活动槽位采样一个 token（停就输出并腾槽）；
    2. 把等待队列里的请求填进空槽，按"本次准入的那几条"做一次批量预填充；
    3. 对需要往里喂 token 的槽位跑一次批量 decode（图的 batch 固定为
       ``capacity``，空槽喂 ``pad_id`` 并丢弃结果）。

    槽位一空就立刻准入新请求，所以慢请求不会拖住新请求的**首 token**——
    这就是 continuous batching 与"整批一起跑"的差别。

    **判据**：同一批请求在 ``capacity=1`` 与 ``capacity=4`` 下的输出逐位相同
    （行与行不共享状态、不共享随机流），且与逐个调用 :func:`generate` 相同。

    **显式的代价**：

    - decode 图的 batch 固定为 ``capacity``，空槽也让内核白跑；调小 capacity
      或等队列积压后再扩容是调用方的选择，这里不做动态扩图。
    - 每一步都要把状态按槽位拼成 batch 再执行、执行完再拆回槽位；非驻留
      后端还会重复上传权重。参考后端上是纯 numpy 拼接，设备上是一次
      D2H/H2D（与束搜索同一套 `_host_array`）。
    - 预填充按"本次准入的条数"建图，因此图缓存键含 batch；准入条数变化时
      会建新图，`max_graphs` 用 LRU 封顶。
    - 只支持 `spec.batch=1` 的会话级构造：capacity 是调度器自己的概念，
      不借用 spec 的 batch 维。
    """

    def __init__(
        self,
        backend,
        params: Mapping[str, np.ndarray],
        spec: Optional[LMSpec] = None,
        *,
        capacity: int = 4,
        chunk_size: Optional[int] = None,
        resident: bool = False,
        max_graphs: int = 16,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity 必须是 int，得到 {capacity!r}")
        if capacity < 1:
            raise ValueError(f"capacity 必须 >= 1，得到 {capacity}")
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int):
            raise TypeError(f"max_graphs 必须是 int，得到 {max_graphs!r}")
        if max_graphs < 1:
            raise ValueError(f"max_graphs 必须 >= 1，得到 {max_graphs}")
        self.spec = spec or LMSpec()
        if self.spec.batch != 1:
            raise ValueError(
                f"ServingEngine 需要 spec.batch=1，得到 {self.spec.batch}；"
                "并发度由 capacity 决定"
            )
        self.backend = backend
        self.params = params
        self.capacity = capacity
        self.chunk = int(chunk_size) if chunk_size is not None else self.spec.steps
        if not 1 <= self.chunk <= self.spec.steps:
            raise ValueError(
                f"chunk_size 必须落在 [1, {self.spec.steps}]，得到 {self.chunk}"
            )
        self.resident = bool(resident)
        _require_resident(backend, self.resident, "ServingEngine(resident=True)")
        self.max_graphs = max_graphs
        self._graphs: "OrderedDict[Tuple[int, int, str], Any]" = OrderedDict()
        self._resident_ready = False
        self._closed = False
        self._pending: "deque[_ServingRequest]" = deque()
        self._slots: List[Optional[_ServingRequest]] = [None] * capacity
        self._ready: List[Tuple[int, List[int]]] = []
        self._next_id = 0

    # -- 图缓存 -------------------------------------------------------

    def _graph_for(self, batch: int, length: int, head: str):
        if self._closed:
            raise ValueError("ServingEngine 已关闭，请新建一个调度器")
        key = (batch, length, head)
        item = self._graphs.get(key)
        if item is None:
            item = _stream_graph(length, self.spec.with_batch(batch), head)
            self._graphs[key] = item
            while len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        if self.resident and not self._resident_ready:
            self.backend.resident_parameters(item[0], self.params)
            self._resident_ready = True
        return item

    # -- 请求生命周期 -------------------------------------------------

    def submit(
        self,
        prompt: Sequence[int],
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
        seed: int = 0,
        constraint=None,
    ) -> int:
        """入队一条请求，返回请求 id。参数与 :func:`generate` 同义。

        ``constraint`` 必须是**每条请求各自一份**的对象（它有内部状态）；
        同一个对象交给两条请求会让文法状态互相污染。这一点无法在这里校验，
        只能写清楚。

        ``max_new_tokens=0`` 的请求不构图，直接进完成队列，由下一次
        ``step`` 返回。
        """
        if self._closed:
            raise ValueError("ServingEngine 已关闭，请新建一个调度器")
        prompt, _chunk, limit, stop = _check_request(
            self.spec, prompt, self.chunk, max_new_tokens, eos_id, None
        )
        penalty = float(repetition_penalty)
        if not np.isfinite(penalty) or penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty 必须是有限正数，得到 {repetition_penalty!r}"
            )
        nucleus = float(top_p)
        if not np.isfinite(nucleus) or nucleus <= 0.0:
            raise ValueError(f"top_p 必须是有限正数，得到 {top_p!r}")
        request = _ServingRequest(
            self._next_id,
            list(prompt),
            limit,
            stop,
            temperature=temperature,
            top_k=top_k,
            top_p=nucleus,
            repetition_penalty=penalty,
            seed=int(seed),
            constraint=constraint,
        )
        self._next_id += 1
        if limit == 0:
            self._ready.append((request.rid, []))
        else:
            self._pending.append(request)
        return request.rid

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def active(self) -> int:
        return sum(1 for slot in self._slots if slot is not None)

    def step(self) -> List[Tuple[int, List[int]]]:
        """推进一个解码步，返回**本步完成**的 ``(请求 id, 生成 token)``。

        没有活动请求也没有等待请求时返回空列表（重复调用是安全的）。
        """
        if self._closed:
            raise ValueError("ServingEngine 已关闭，请新建一个调度器")
        finished = self._ready
        self._ready = []

        # 1) 采样：有 logits 的槽位各采一个 token
        for index, request in enumerate(self._slots):
            if request is None or request.logits is None:
                continue
            token = self._pick(request)
            if token == request.stop:
                finished.append((request.rid, list(request.generated)))
                self._slots[index] = None
                continue
            request.generated.append(token)
            request.history.append(token)
            request.logits = None
            if len(request.generated) >= request.limit:
                finished.append((request.rid, list(request.generated)))
                self._slots[index] = None
            else:
                request.feed_token = token

        # 2) 准入：空槽按 FIFO 填入等待请求，只对本次准入的做一次批量预填充
        admitted: List[_ServingRequest] = []
        for index, request in enumerate(self._slots):
            if request is not None or not self._pending:
                continue
            request = self._pending.popleft()
            self._slots[index] = request
            admitted.append(request)
        if admitted:
            batch = len(admitted)
            state, logits = _prefill(
                self.backend,
                self.params,
                [request.prompt for request in admitted],
                self.spec.with_batch(batch),
                chunk=self.chunk,
                resident=self.resident,
                graph_for=lambda length, head: self._graph_for(batch, length, head),
            )
            host_logits = _host_array(self.backend, logits)
            for row, request in enumerate(admitted):
                request.state = {
                    name: _host_array(self.backend, value)[row]
                    for name, value in state.items()
                }
                request.logits = host_logits[row, 0, :]

        # 3) 解码：需要喂 token 的槽位走一次容量为 capacity 的批量 decode
        feeding = [
            index
            for index, request in enumerate(self._slots)
            if request is not None and request.feed_token is not None
        ]
        if feeding:
            graph, inputs, _ = self._graph_for(self.capacity, 1, "last")
            ids = np.full((self.capacity, 1), self.spec.pad_id, dtype=np.int32)
            rows: Dict[str, List[np.ndarray]] = {
                name: [] for name in self.spec.state_shapes
            }
            for index, request in enumerate(self._slots):
                if request is None or request.state is None:
                    for name, shape in self.spec.state_shapes.items():
                        # state_shapes 带 batch 维；槽位里存的是"单行"，去掉首维
                        rows[name].append(np.zeros(shape[1:], dtype=np.float32))
                    continue
                for name in self.spec.state_shapes:
                    rows[name].append(request.state[name])
                if request.feed_token is not None:
                    ids[index, 0] = request.feed_token
                    request.feed_token = None
            feed = {} if self.resident else dict(self.params)
            feed[inputs["ids"].name] = ids
            feed[inputs["reset"].name] = np.zeros(
                (self.capacity, 1), dtype=np.float32
            )
            for name in self.spec.state_shapes:
                feed[inputs[name].name] = np.stack(rows[name], axis=0)
            out = self.backend.execute(graph, feed)
            state = self.spec.carry_state(out)
            host_state = {
                name: _host_array(self.backend, value)
                for name, value in state.items()
            }
            host_logits = _host_array(self.backend, out["logits"])
            for index in feeding:
                request = self._slots[index]
                request.state = {
                    name: host_state[name][index] for name in self.spec.state_shapes
                }
                request.logits = host_logits[index, 0, :]
        return finished

    def run(self) -> List[Tuple[int, List[int]]]:
        """推进到所有请求完成，按**完成先后**返回 ``(请求 id, 生成 token)``。

        ``step`` 里每个活动请求每步至少前进一个 token（或直接完成），
        所以循环一定终止：总步数不超过"所有请求的 max_new_tokens 之和"。
        """
        outputs: List[Tuple[int, List[int]]] = []
        while True:
            done = self.step()
            outputs.extend(done)
            if not self._pending and self.active == 0:
                return outputs

    def _pick(self, request: _ServingRequest) -> int:
        row = np.array(request.logits, dtype=np.float64, copy=True)
        if request.constraint is not None:
            row = _masked(row, request.constraint.allowed())
        token = int(sample_next_token(
            row,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            recent=request.history,
            rng=request.rng,
        ))
        if request.constraint is not None:
            request.constraint.feed(token)
        return token

    # -- 生命周期 -----------------------------------------------------

    def close(self) -> None:
        """释放驻留权重并丢弃图缓存；队列里的请求一并作废。可重复调用。"""
        if self._closed:
            return
        if self._resident_ready:
            self.backend.release_parameters()
            self._resident_ready = False
        self._graphs.clear()
        self._pending.clear()
        self._slots = [None] * self.capacity
        self._ready = []
        self._closed = True

    def __enter__(self) -> "ServingEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"ServingEngine(capacity={self.capacity}, active={self.active}, "
            f"pending={self.pending}, graphs={len(self._graphs)}, "
            f"closed={self._closed})"
        )
