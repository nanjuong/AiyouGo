"""Mamba block 构件（selective state space + 门控 + 深度卷积）。

这是对整套 IR 最有说服力的检验：一个现代架构里只有**一个**算子是为它新加的
（``scan_linear``），其余全部由已有语义组合：

    归一化    rmsnorm                                    （5.21）
    输入投影  matmul + slice（切成 x 分支与 z 门控）
    深度卷积  concat + slice + slice/mul/add 的时间展开    （kernel 很小，展开即最优）
    激活      silu
    离散化    softplus(dt) = log(1+exp(x))、A = -exp(A_raw)、a = exp(dt*A)
    递归      scan_linear
    读出      mul + reduce_sum(-1)（在 models/ssm.py 里）
    门控      mul(silu(z))
    输出投影  matmul，加残差

为什么卷积是展开而不是算子：Mamba 的 ``conv1d`` 是**因果**的、kernel 只有 3~4，
展开成 ``K`` 次移位相加的节点数远小于"新增一个算子 + 它的反向 + 它的后端实现"
的成本。序列长度因此必须是静态的（与 RNN/SNN 的展开理由相同）。

为什么要状态出参：整序列前向只是训练口径。推理要按 chunk 甚至逐 token 喂，
就必须把"递归状态"和"卷积窗口"跨调用带出来。两者都是普通图输入/输出——
没有新增算子，也没有把某个后端的状态藏进运行期对象里：

    conv0/h0  ->  build_mamba_block_stateful  ->  y + state_out{conv, h}

``steps=1`` 的同一份函数就是 decode：图不同（T 写死为 1），语义完全一致。

这是**受 Mamba 启发的选择性 SSM 块**，不是 mamba-ssm 的复刻，因此不承诺
官方权重兼容或性能继承。与官方 Mamba-1 有三处结构差异：

1. dt 用逐通道的 ``dt_w * x + dt_b``；官方先把输入低秩投影到 ``dt_rank``，
   再投影回 ``d_inner``；
2. 读出前多一次 RMSNorm（官方 ``out_proj`` 前没有）；
3. 归一化与门控的位置只在"输入 → 卷积 → scan → 门控 → 投影"这条链上近似。

保留的是 selective 的核心性质：步长随输入变化、B/C 随步变化。扫描公式本身
（``a = exp(Δ·A)``、``b = Δ·B·x``）与官方参考实现一致——后者也采用这一形式，
所以不能据它判定"不是真 SSM"；对拍见 ``mamba_reference.official_selective_scan``。
但"公式一致"不等于"整块等价"，参数迁移的拒绝规则见
``mamba_reference.map_official_params``。要评估表达力与优化行为，需要按
相同数据、参数量、训练 token、优化器与算力做消融，而不是沿用论文的性能结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ir.graph import Graph, Value
from ..ir.types import f32
from .ssm import build_selective_scan_stateful

__all__ = [
    "MambaSpec",
    "MambaModelSpec",
    "causal_depthwise_conv1d",
    "causal_depthwise_conv1d_stateful",
    "build_mamba_block",
    "build_mamba_block_stateful",
    "build_mamba_model",
    "build_mamba_model_stateful",
    "init_mamba_params",
    "effective_half_lives",
    "decay_diagnostics",
]


def _conv_shapes(x: Value, weight: Value, bias: Value):
    """校验深度卷积的三个输入，返回 ``(batch, steps, channels, kernel)``。"""
    shape = x.type.shape
    if len(shape) != 3:
        raise ValueError(f"深度卷积要求 (B, T, C) 输入，得到 {x.type}")
    batch, steps, channels = shape
    if not isinstance(steps, int):
        raise ValueError(f"时间维必须静态（展开发生在构图期），得到 {steps!r}")
    if weight.type.rank != 2 or weight.type.shape[0] != channels:
        raise ValueError(f"weight 形状应为 (C={channels}, K)，得到 {weight.type}")
    kernel = weight.type.shape[1]
    if not isinstance(kernel, int) or kernel < 1:
        raise ValueError(f"卷积核长度必须为正整数，得到 {kernel!r}")
    if bias.type.shape != (channels,):
        raise ValueError(f"bias 形状应为 ({channels},)，得到 {bias.type}")
    return batch, steps, channels, kernel


def _reset_prefix(g: Graph, reset: Value, batch: int, steps: int) -> Value:
    """``cs[b, t] = sum_{s<=t} reset[b, s]``，形状 ``(B, T)``。

    卷积窗口只需知道"两个位置之间有没有重置"，所以累积和就够了：
    ``cs_t != cs_p`` 等价于区间 ``(p, t]`` 内有 reset。累积和用
    ``scan_linear`` 表达（a 全 1、h0 全 0 的线性递归），不为它新增算子。
    """
    dtype = reset.type.dtype
    rt = g.transpose(reset, perm=(1, 0))  # (T, B)
    if steps == 1:
        # 单步的累积和就是它自己，省掉一个递归节点（decode 的热路径）
        cs_t = rt
    else:
        ones = g.constant(1.0, dtype=dtype, shape=(steps, batch))
        zeros = g.constant(0.0, dtype=dtype, shape=(batch,))
        cs_t = g.scan_linear(ones, rt, zeros)
    return g.transpose(cs_t, perm=(1, 0))


def _mask_stale_conv_state(
    g: Graph,
    new_state: Value,
    cs: Value,
    batch: int,
    steps: int,
    kernel: int,
    dtype,
) -> Value:
    """把"最后一次 reset 之前"的帧从带出的卷积窗口里清零。

    ``new_state`` 的第 ``s`` 行对应 x 的位置 ``p = steps - (K-1) + s``。若
    ``(p, steps-1]`` 里发生过 reset，这一帧就属于上一段，下一块再读它会把
    reset 之前的输入重新带进窗口（``[10,20,30,1]`` 的 ``30`` 泄漏到 ``[2]``）。
    只清严格在 reset 之前的帧：reset 位置本身是新段的第一帧，必须保留。

    ``p < 0`` 的行来自传进来的 ``state``，它们属于"本块之前"，因此只在整块
    没有 reset 时才有效。``cs`` 是块内累积和，单调不减，所以"区间内有 reset"
    等价于 ``cs[end] - cs[p] > 0``。
    """
    end = g.slice(cs, dim=1, start=steps - 1, stop=steps)  # (B, 1)
    keep = None
    for s in range(kernel - 1):
        pos = steps - (kernel - 1) + s
        if pos < 0:
            ref = g.constant(0.0, dtype=dtype, shape=(batch, 1))
        else:
            ref = g.slice(cs, dim=1, start=pos, stop=pos + 1)
        crossed = g.step(g.sub(end, ref))  # (B, 1)：1 表示这一帧已过期
        valid = g.sub(g.constant(1.0, dtype=dtype, shape=()), crossed)
        row = g.reshape(valid, shape=(batch, 1, 1))
        keep = row if keep is None else g.concat(keep, row, axis=1)
    return g.mul(new_state, keep)


def causal_depthwise_conv1d(
    g: Graph, x: Value, weight: Value, bias: Value
) -> Value:
    """逐通道因果一维卷积，stride=1，输出长度与输入相同。

    ``x (B, T, C)``、``weight (C, K)``、``bias (C,)``。左侧补 ``K-1`` 个零，
    所以位置 t 只看 t-K+1..t——这正是自回归模型需要的"看不到未来"。
    """
    out, _ = causal_depthwise_conv1d_stateful(g, x, weight, bias)
    return out


def causal_depthwise_conv1d_stateful(
    g: Graph,
    x: Value,
    weight: Value,
    bias: Value,
    state: Optional[Value] = None,
    reset: Optional[Value] = None,
) -> Tuple[Value, Value]:
    """同 :func:`causal_depthwise_conv1d`，但接受并返回卷积窗口状态。

    ``state`` 形状 ``(B, K-1, C)``，是"序列开始前的 K-1 个输入"；缺省为零常量
    （等价于左侧补零）。返回值中的新状态是这次输入的最后 ``K-1`` 个位置，
    直接喂给下一次调用即可续上，边界不丢也不重。

    ``reset (B, T)`` 是重置掩码，1 表示该位置开始一段新序列：窗口不得跨过它去
    读更早的输入。实现上把跨越重置的抽头乘 0——重置之后的输入照常进入窗口。
    带出的新状态同样按**最后一次 reset** 清掉更早的帧，否则下一块会把 reset
    之前的输入重新读进窗口（跨块泄漏）。
    """
    batch, steps, channels, kernel = _conv_shapes(x, weight, bias)
    dtype = x.type.dtype
    expected = (batch, kernel - 1, channels)

    if state is None:
        state = g.constant(0.0, dtype=dtype, shape=expected)
    elif tuple(state.type.shape) != expected:
        raise ValueError(f"卷积状态形状应为 {expected}，得到 {state.type.shape}")

    if reset is not None and tuple(reset.type.shape) != (batch, steps):
        raise ValueError(
            f"reset 形状应为 ({batch}, {steps})，得到 {reset.type.shape}"
        )

    cs = None if reset is None else _reset_prefix(g, reset, batch, steps)

    if kernel > 1:
        # 历史 K-1 帧与本次输入拼起来，第 j 个抽头就是 source[t+j]
        source = g.concat(state, x, axis=1)
        new_state = g.slice(source, dim=1, start=steps, stop=steps + kernel - 1)
        if cs is not None:
            new_state = _mask_stale_conv_state(
                g, new_state, cs, batch, steps, kernel, dtype
            )
    else:
        source = x
        new_state = g.constant(0.0, dtype=dtype, shape=expected)

    acc = g.broadcast_to(
        g.reshape(bias, shape=(1, 1, channels)), shape=(1, steps, channels)
    )

    if cs is not None and kernel > 1:
        # 历史部分的累积和按 0 计：conv 状态被视为"属于当前段之前"
        head = g.constant(0.0, dtype=dtype, shape=(batch, kernel - 1))
        cs_src = g.concat(head, cs, axis=1)
    else:
        cs_src = cs

    for j in range(kernel):
        # 第 j 个抽头：source[t + j] 对应 x[t + j - (K-1)]
        tap = g.slice(source, dim=1, start=j, stop=j + steps)
        if cs_src is not None:
            gap = g.sub(g.slice(cs_src, dim=1, start=j, stop=j + steps), cs)
            crossed = g.add(g.step(gap), g.step(g.neg(gap)))
            valid = g.sub(g.constant(1.0, dtype=dtype, shape=()), crossed)
            tap = g.mul(tap, g.reshape(valid, shape=(batch, steps, 1)))
        w_j = g.reshape(g.slice(weight, dim=1, start=j, stop=j + 1), shape=(1, 1, channels))
        acc = g.add(acc, g.mul(tap, w_j))
    return acc, new_state


@dataclass(frozen=True)
class MambaSpec:
    """单层 Mamba block 的形状。``steps`` 与 ``batch`` 必须静态。

    ``expand`` 是内部宽度相对 ``d_model`` 的倍数（原论文取 2），
    ``d_state`` 是状态维 N（典型 16），``d_conv`` 是深度卷积的核长（典型 4）。
    """

    d_model: int = 32
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    steps: int = 12
    batch: int = 4

    def __post_init__(self) -> None:
        for name in ("d_model", "d_state", "d_conv", "expand", "steps", "batch"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须为正，得到 {getattr(self, name)}")
        if self.d_conv < 1:
            raise ValueError("d_conv 必须 >= 1")

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def conv_state_shape(self) -> Tuple[int, int, int]:
        """卷积窗口状态 ``(B, K-1, D_inner)``。``d_conv=1`` 时中间维为 0。"""
        return (self.batch, self.d_conv - 1, self.d_inner)

    @property
    def ssm_state_shape(self) -> Tuple[int, int, int]:
        """SSM 递归状态 ``(B, D_inner, N)``。"""
        return (self.batch, self.d_inner, self.d_state)

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        d, di, n, k = self.d_model, self.d_inner, self.d_state, self.d_conv
        return {
            "norm_g": (d,),
            "w_in": (d, 2 * di),
            "conv_w": (di, k),
            "conv_b": (di,),
            "dt_w": (di,),
            "dt_b": (di,),
            "A_raw": (di, n),
            "B_w": (di, n),
            "C_w": (di, n),
            "D_skip": (di,),
            "norm2_g": (di,),
            "w_out": (di, d),
        }


def _log_grid(bounds: Tuple[float, float], count: int) -> np.ndarray:
    """在 ``[lo, hi]`` 上按对数等间隔取 ``count`` 个值。

    时间尺度天然是对数的：1、8、64、512 与 1、2、3、4 相比，前者的"记号间隔"
    才是相同的。所以网格在 log 域等距，指数还原。
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"区间必须满足 0 < lo <= hi，得到 {bounds}")
    if count < 1:
        raise ValueError(f"采样点数必须为正，得到 {count}")
    if count == 1:
        return np.array([lo], dtype=np.float64)
    frac = np.linspace(0.0, 1.0, int(count))
    return lo * (hi / lo) ** frac


def _inv_softplus(x: np.ndarray) -> np.ndarray:
    """``softplus`` 的反函数 ``log(exp(x) - 1)``，定义域 ``x > 0``。"""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0.0):
        raise ValueError("softplus 的反函数只在正值上有定义")
    return np.log(np.expm1(x))


def init_mamba_params(
    spec: MambaSpec,
    seed: int = 0,
    *,
    prefix: str = "",
    half_life_range: Tuple[float, float] = (1.0, 32768.0),
    dt_range: Tuple[float, float] = (1e-3, 1e-1),
) -> Dict[str, np.ndarray]:
    """Mamba 专属初始化：让各通道覆盖不同的记忆时间尺度。

    通用初始化（``train/params.py`` 的 Xavier / 零偏置）在这里是**有偏**的：
    ``A_raw ≈ 0`` 意味着 ``|A| ≈ 1``，``dt_b = 0`` 意味着 ``Δ ≈ log 2``，
    于是 ``a = exp(ΔA) ≈ 0.5``——所有通道的半衰期都约一个 token。用这种初始化
    训练，"复制上一步"就能解释大部分损失，长依赖没有梯度压力。

    官方 Mamba 的做法是两件独立的事：``dt_b`` 逐通道 log-uniform、``A`` 用
    S4D-Real 的倍频序列。这里把二者合并成一个可读的网格，直接指定**目标半衰期**
    ``H``：

    具体地，``(d_inner, d_state)`` 的每个单元分到一个目标半衰期
    ``H ∈ half_life_range``（在 ``d_inner * d_state`` 个点上对数等距），
    ``dt_b`` 逐通道在 ``dt_range`` 上对数等距给出名义步长 ``Δ``，再由

        ``|A| = log 2 / (H · Δ)``，``A_raw = log|A|``

    反解衰减率（``A = -exp(A_raw)`` 仍由模型公式负责，这里只给 raw）。
    于是初始化后的半衰期**逐单元精确等于**网格值（``x_act = 0`` 参考点），
    短/中/长记忆各占多少通道是数出来的，不是猜的；训练前后都可用
    :func:`effective_half_lives` 报告分布。

    ``dt_w`` 初始化为零：``Δ = softplus(dt_w·x_act + dt_b)`` 里的输入项一旦
    非零，激活幅度就会盖过跨四个数量级的 ``Δ`` 网格，"多时间尺度"在初始化
    时便名不副实。零起点让分层只由 ``dt_b`` 决定，输入调制由训练学出来。

    ``seed`` 只影响随机参数（投影、卷积、A/Δ 以外的部分）；A 与 Δ 的网格是
    确定性的，换 seed 做消融时时间尺度分配不会漂移。

    ``prefix`` 只参与随机流（``init_array`` 把它混进名字的 CRC），返回的字典
    键不带前缀；整栈初始化靠它让各层拿到不同的随机权重。
    """
    from ..train.params import init_array  # 延迟导入：构图层不在模块层依赖训练层

    di, n = spec.d_inner, spec.d_state
    params = {
        name: init_array(prefix + name, shape, f32, seed=seed)
        for name, shape in spec.param_shapes.items()
    }

    # Δ：逐通道对数间隔，从"一步就忘"的短步长到"慢慢累积"的长步长
    dt_target = _log_grid(dt_range, di)
    params["dt_b"] = _inv_softplus(dt_target).astype(np.float32)
    # 初始 Δ 完全由 dt_b 决定，见上面的说明
    params["dt_w"] = np.zeros((di,), dtype=np.float32)

    # A：每个 (通道, 状态) 单元一个目标半衰期；Δ 逐通道广播
    half_lives = _log_grid(half_life_range, di * n).reshape(di, n)
    rate = np.log(2.0) / (half_lives * dt_target[:, None])
    params["A_raw"] = np.log(rate).astype(np.float32)
    return params


def effective_half_lives(
    spec: MambaSpec, params: Dict[str, Any]
) -> np.ndarray:
    """逐单元的"旧状态半衰期"（token 数），形状 ``(d_inner, d_state)``。

    ``a = exp(ΔA)``，问 ``a^H = 1/2`` 得 ``H = log 2 / (Δ|A|)``。``Δ`` 取
    ``softplus(dt_b)``，即 ``x_act = 0`` 时的参考步长——实际前向里
    ``dt_w·x_act`` 还会推动它，所以这是初始化语义与训练后对照的同一把尺子。

    参数可以是 :func:`init_mamba_params` 的输出，也可以是训练后取出的当前值。
    """
    A = np.exp(np.asarray(params["A_raw"], dtype=np.float64))
    z = np.asarray(params["dt_b"], dtype=np.float64)
    dt = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    if A.shape != (spec.d_inner, spec.d_state) or dt.shape != (spec.d_inner,):
        raise ValueError(
            f"参数形状应为 A_raw{(spec.d_inner, spec.d_state)} 与 "
            f"dt_b{(spec.d_inner,)}，得到 A_raw{A.shape} / dt_b{dt.shape}"
        )
    return np.log(2.0) / (dt[:, None] * A)


def _column_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    """有限值的最小 / 中位 / p95 / 最大；用最近秩，不插值。"""
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    ordered = np.sort(flat)

    def at(q: float) -> float:
        rank = max(1, math.ceil(q * ordered.size))
        return float(ordered[min(rank, ordered.size) - 1])

    return {
        "min": float(ordered[0]),
        "p50": at(0.5),
        "p95": at(0.95),
        "max": float(ordered[-1]),
    }


def decay_diagnostics(
    spec: MambaSpec,
    params: Dict[str, Any],
    *,
    activation: Optional[np.ndarray] = None,
    horizon: int = 0,
    state_init_norm: float = 0.0,
    dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """AF-23：把"Δ>0、A<0 所以稳定"从局部论证变成可检查的条件。

    审计的反对意见不是公式错了，而是这条论证只约束了 ``a`` 本身：输入注入
    ``ΔBu``、读出 ``C``、门控、残差、投影都能放大信号；``Δ→0`` 时 ``a→1``，
    没有统一的严格收缩常数；f16 里很小的衰减还会舍入成 ``a=1``。因此这里只
    报告**递归本身**的条件，并明确不覆盖递归之外的放大：

        a = exp(-Δ·|A|),   q = max(a),   b = Δ·B·x,   M = max|b|
        |h_t| ≤ q^t·|h0| + M/(1-q)      仅当 q < 1 且 M 有限

    给 ``activation`` 才能算 ``M``（否则不知道输入幅度，不虚构界）；``dtype``
    给 ``"f16"`` 时额外统计 ``a`` 舍入成 1 的单元数——那是"看起来有衰减、
    实际每步都保留旧状态"的失效形态。

    这是诊断，不是证明：``q<1`` 只说明递归收缩，深堆叠的梯度与特征尺度仍由
    残差、门控与投影决定。返回的 ``state_bound`` 为 ``None`` 时，``reason``
    说明缺哪一个条件。
    """
    A = np.exp(np.asarray(params["A_raw"], dtype=np.float64))
    w = np.asarray(params["dt_w"], dtype=np.float64)
    z = np.asarray(params["dt_b"], dtype=np.float64)
    if A.shape != (spec.d_inner, spec.d_state):
        raise ValueError(
            f"A_raw 形状应为 {(spec.d_inner, spec.d_state)}，得到 {A.shape}"
        )
    if w.shape != (spec.d_inner,) or z.shape != (spec.d_inner,):
        raise ValueError(
            f"dt_w / dt_b 形状应为 {(spec.d_inner,)}，得到 {w.shape} / {z.shape}"
        )

    if activation is None:
        x_act = np.zeros((1, spec.d_inner), dtype=np.float64)
    else:
        x_act = np.asarray(activation, dtype=np.float64)
        if x_act.ndim == 0 or x_act.shape[-1] != spec.d_inner:
            raise ValueError(
                f"activation 的末维应为 d_inner={spec.d_inner}，得到 {x_act.shape}"
            )

    # Δ = softplus(dt_w·x + dt_b)，与图里的稳定写法一致
    logits = w * x_act + z
    dt = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    a = np.exp(-dt[..., None] * A)

    finite_a = a[np.isfinite(a)]
    q = float(np.max(finite_a)) if finite_a.size else float("nan")

    f16_one: Optional[int] = None
    if dtype is not None and str(dtype).lower() in ("f16", "float16", "half"):
        f16_one = int(np.count_nonzero(a.astype(np.float16) == np.float16(1.0)))

    bad_dt = int(np.count_nonzero(~np.isfinite(dt)))
    bad_a = int(np.count_nonzero(~np.isfinite(a)))
    first_bad = [list(map(int, idx)) for idx in np.argwhere(~np.isfinite(a))[:5]]

    # 有非有限值时 ``finite_a`` 会把它们过滤掉，q 看起来仍然 < 1：
    # "收缩"这个结论就建立在被丢掉的单元上，必须显式判否。
    contractive = bool(
        bad_a == 0 and bad_dt == 0 and np.isfinite(q) and q < 1.0
    )

    m: Optional[float] = None
    if activation is not None:
        bw = np.asarray(params["B_w"], dtype=np.float64)
        if bw.shape != (spec.d_inner, spec.d_state):
            raise ValueError(
                f"B_w 形状应为 {(spec.d_inner, spec.d_state)}，得到 {bw.shape}"
            )
        # 注入是 Δ·B·x，其中 B_t = x·B_w 是**沿 d_inner 求和后**的 (…, N)；
        # 逐单元乘 B_w 会把 (d_inner, N) 当成 (…, N) 广播，上界偏小一半以上。
        bt = x_act @ bw
        b = (dt * x_act)[..., None] * np.expand_dims(bt, axis=-2)
        finite_b = b[np.isfinite(b)]
        m = float(np.max(np.abs(finite_b))) if finite_b.size else None

    reason: Optional[str] = None
    state_bound: Optional[float] = None
    if bad_a or bad_dt:
        reason = (
            f"存在非有限单元：Δ {bad_dt} 个、a {bad_a} 个；"
            "收缩性只对被丢弃的单元之外的那部分成立，不能报 contractive"
        )
    elif not contractive:
        reason = f"没有严格收缩常数：q={q!r} 不满足 q < 1"
    elif m is None:
        reason = "未给 activation，无法估计输入注入上界 M"
    else:
        state_bound = q ** int(horizon) * float(state_init_norm) + m / (1.0 - q)

    with np.errstate(divide="ignore"):
        log_a = np.log(a)

    return {
        "delta": _column_stats(dt),
        "log_a": _column_stats(log_a),
        "q": q,
        "contractive": contractive,
        "f16_a_rounds_to_one": f16_one,
        "nonfinite": {"delta": bad_dt, "a": bad_a, "first": first_bad},
        "injection_bound": m,
        "state_bound": state_bound,
        "state_bound_reason": reason,
        "horizon": int(horizon),
    }



def _softplus(g: Graph, value: Value) -> Value:
    """``softplus(x) = log(1 + exp(x))`` 的稳定写法。

    直接算 ``exp(x)`` 在 x > 88 时会溢出（f32），dt 由参数与激活决定，
    训练中一旦出现大的预激活就是 inf，反向再变成 NaN。改成

        ``(x + |x|) / 2 + log(1 + exp(-|x|))``

    指数项的入参被压到非正半轴，``exp`` 恒落在 (0, 1]，两头都不溢出；
    x 很大时结果就是 x 本身，x 很负时是 exp(x) 的一阶近似。

    前半段写成 ``(x + |x|) / 2`` 而不是 ``relu(x)``：两者在 x != 0 时逐个
    相等，但 ``relu`` 在 0 处的子梯度取 0，会让整条链的导数在 0 处变成 0
    （真值是 ``sigmoid(0) = 0.5``）；``|x|`` 在 0 处的导数同样是 0，于是
    ``(x + |x|) / 2`` 的导数是 ``(1 + 0) / 2 = 0.5``，与解析值一致。
    绝对值仍用 ``relu(x) + relu(-x)`` 拼出来，不为它新增算子。
    """
    dtype = value.type.dtype
    one = g.constant(1.0, dtype=dtype, shape=())
    half = g.constant(0.5, dtype=dtype, shape=())
    magnitude = g.add(g.relu(value), g.relu(g.neg(value)))
    positive = g.mul(half, g.add(value, magnitude))
    return g.add(positive, g.log(g.add(one, g.exp(g.neg(magnitude)))))


def _mamba_layer(
    g: Graph,
    spec: MambaSpec,
    x: Value,
    params: Dict[str, Value],
    prefix: str,
    conv0: Optional[Value],
    h0: Optional[Value],
    reset: Optional[Value],
) -> Tuple[Value, Dict[str, Value]]:
    """一层的公共主体，参数从 ``params[prefix + 名字]`` 取。

    返回 ``(y, {"conv": 新窗口, "h": 新状态})``。``conv0`` / ``h0`` / ``reset``
    为 None 时分别取零常量 / 不重置。
    """
    di = spec.d_inner

    def p(base: str) -> Value:
        return params[prefix + base]

    h = g.rmsnorm(x, p("norm_g"), axes=(-1,), eps=1e-6)

    # 输入投影后切成 x 分支与 z 门控
    xg = g.matmul(h, p("w_in"))
    x_branch = g.slice(xg, dim=2, start=0, stop=di)
    z = g.slice(xg, dim=2, start=di, stop=2 * di)

    x_conv, conv_out = causal_depthwise_conv1d_stateful(
        g, x_branch, p("conv_w"), p("conv_b"), state=conv0, reset=reset
    )
    x_act = g.silu(x_conv)

    # 转到时间优先 (T, B, D)：scan 的时间维固定在 0
    xt = g.transpose(x_act, perm=(1, 0, 2))

    dt = _softplus(g, g.add(g.mul(x_act, p("dt_w")), p("dt_b")))
    dtt = g.transpose(dt, perm=(1, 0, 2))

    # A 恒为负：-exp(raw) 保证衰减，不需要参数初始化刻意取负
    A = g.neg(g.exp(p("A_raw")))
    B_t = g.matmul(xt, p("B_w"))  # (T, B, N)
    C_t = g.matmul(xt, p("C_w"))

    reset_t = None if reset is None else g.transpose(reset, perm=(1, 0))
    y, h_out = build_selective_scan_stateful(
        g, xt, dtt, A, B_t, C_t, D_skip=p("D_skip"), h0=h0, reset=reset_t
    )
    y = g.transpose(y, perm=(1, 0, 2))  # 回到 (B, T, D)

    y = g.mul(y, g.silu(z))
    y = g.matmul(g.rmsnorm(y, p("norm2_g"), axes=(-1,), eps=1e-6), p("w_out"))
    return g.add(x, y), {"conv": conv_out, "h": h_out}


def _mamba_body(
    g: Graph,
    spec: MambaSpec,
    name,
    conv0: Optional[Value],
    h0: Optional[Value],
    reset: Optional[Value],
):
    """声明单层的输入并构建主体，返回 ``(inputs, out, state_out)``。"""
    d, batch, steps = spec.d_model, spec.batch, spec.steps

    inputs: Dict[str, Value] = {"x": g.input(name("x"), f32[batch, steps, d])}
    for pname, shape in spec.param_shapes.items():
        inputs[pname] = g.input(name(pname), f32[shape])

    out, state_out = _mamba_layer(g, spec, inputs["x"], inputs, "", conv0, h0, reset)
    return inputs, out, state_out


def build_mamba_block(
    g: Graph, spec: MambaSpec | None = None
) -> Tuple[Dict[str, Value], Value]:
    """构建一个 Mamba block，返回 ``(输入表, 输出 Value)``，形状均为 ``(B, T, d)``。

    这是训练 / 整序列口径：初始状态与卷积窗口都是零常量，图自包含。
    需要跨调用携带状态（prefill 分块、逐 token decode）时用
    :func:`build_mamba_block_stateful`。
    """
    spec = spec or MambaSpec()
    inputs, out, _ = _mamba_body(g, spec, lambda s: s, None, None, None)
    return inputs, out


def build_mamba_block_stateful(
    g: Graph,
    spec: MambaSpec | None = None,
    *,
    prefix: str = "",
    with_reset: bool = False,
) -> Tuple[Dict[str, Value], Value, Dict[str, Value]]:
    """带持久状态的 Mamba block：``(inputs, y, state_out)``。

    输入表额外包含：

    ``conv0`` ``(B, K-1, d_inner)``
        上一次调用留下的卷积窗口；全零即"这是序列的第一段"。
    ``h0`` ``(B, d_inner, N)``
        上一次调用留下的 SSM 状态；全零即"没有历史"。
    ``reset`` ``(B, T)``（仅 ``with_reset=True``）
        重置掩码，1 表示该位置开始新序列：SSM 状态与卷积窗口都不再跨过它。

    返回的 ``state_out`` 是同名的两个值（键为 ``"conv"`` / ``"h"``），形状与
    ``conv0`` / ``h0`` 相同，可以直接作为下一次调用的输入。``spec.steps=1``
    时这次调用就是 decode：图本身不同（T 写死为 1），语义与 prefill 一致。

    ``prefix`` 只改图里的输入名，字典键仍是 ``x`` / ``conv0`` / ``h0``；要在同一
    张图里放多个 block 且能一起喂数据，用 :func:`build_mamba_model_stateful`
    （那里的键自带层号，不会重复）。
    """
    spec = spec or MambaSpec()
    batch, steps = spec.batch, spec.steps

    def name(base: str) -> str:
        return f"{prefix}{base}"

    conv0 = g.input(name("conv0"), f32[spec.conv_state_shape])
    h0 = g.input(name("h0"), f32[spec.ssm_state_shape])
    reset = None
    if with_reset:
        reset = g.input(name("reset"), f32[batch, steps])

    inputs, out, state_out = _mamba_body(g, spec, name, conv0, h0, reset)
    # _mamba_body 已经按 name() 命名了 x 与参数，这里只需补上状态输入
    inputs["conv0"] = conv0
    inputs["h0"] = h0
    if reset is not None:
        inputs["reset"] = reset
    return inputs, out, state_out


@dataclass(frozen=True)
class MambaModelSpec:
    """多层 Mamba 栈的形状：每层形状相同，参数与状态各自独立。

    输入表里的名字都带层号前缀（``l0_w_in``、``l0_conv0``…），所以整栈是**一张**
    图，而不是"每层一张图手工串"。``num_layers=1`` 时与单个 block 等价。
    """

    d_model: int = 32
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    steps: int = 12
    batch: int = 4
    num_layers: int = 4

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "d_state",
            "d_conv",
            "expand",
            "steps",
            "batch",
            "num_layers",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须为正，得到 {getattr(self, name)}")

    @property
    def layer(self) -> MambaSpec:
        """单层的形状规格。"""
        return MambaSpec(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            steps=self.steps,
            batch=self.batch,
        )

    def with_steps(self, steps: int) -> "MambaModelSpec":
        """换一个序列长度：prefill 用整段，decode 用 ``with_steps(1)``。

        两张图共享同一套参数名与状态名，prefill 的 ``state_out`` 可以直接作为
        decode 的输入——名字相同正是为了免掉搬运代码。
        """
        return replace(self, steps=steps)

    @property
    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.num_layers):
            for base, shape in self.layer.param_shapes.items():
                shapes[f"l{i}_{base}"] = shape
        return shapes

    @property
    def state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """每层的卷积窗口与 SSM 状态，键名与图输入一致。"""
        layer = self.layer
        shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.num_layers):
            shapes[f"l{i}_conv0"] = layer.conv_state_shape
            shapes[f"l{i}_h0"] = layer.ssm_state_shape
        return shapes

    def init_params(
        self, seed: int = 0, **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """整栈的模型专属初始化，键与 :attr:`param_shapes` 一一对应。

        逐层调用 :func:`init_mamba_params`（``seed`` 相同也不要紧：随机参数的
        随机流混入了各自的名字 ``l{i}_*``，层间不会共用同一串数）。A 与 Δ 的
        时间尺度网格逐层一致——各层都有短中长通道，这是设计而非偷懒。
        """
        params: Dict[str, np.ndarray] = {}
        for i in range(self.num_layers):
            prefix = f"l{i}_"
            for base, arr in init_mamba_params(
                self.layer, seed=seed, prefix=prefix, **kwargs
            ).items():
                params[prefix + base] = arr
        return params

    def carry_state(self, state_out: Dict[str, Any]) -> Dict[str, Any]:
        """把 :func:`build_mamba_model_stateful` 的 ``state_out`` 翻成输入表。

        输出名是 ``l{i}_conv`` / ``l{i}_h``（与单个 block 的 ``conv`` / ``h``
        同构），输入名是 ``l{i}_conv0`` / ``l{i}_h0``——两者只差末尾那个 0，
        因为图里不能拿同一个名字同时做输入与输出。于是分块续接是

            feed.update(spec.carry_state(state_out))
        """
        return {name: state_out[name[:-1]] for name in self.state_shapes}


def _model_inputs(
    g: Graph, spec: MambaModelSpec, name, with_states: bool
) -> Dict[str, Value]:
    inputs: Dict[str, Value] = {
        "x": g.input(name("x"), f32[spec.batch, spec.steps, spec.d_model])
    }
    for pname, shape in spec.param_shapes.items():
        inputs[pname] = g.input(name(pname), f32[shape])
    if with_states:
        for sname, shape in spec.state_shapes.items():
            inputs[sname] = g.input(name(sname), f32[shape])
    return inputs


def _run_layers(
    g: Graph,
    spec: MambaModelSpec,
    inputs: Dict[str, Value],
    stateful: bool,
    reset: Optional[Value],
) -> Tuple[Value, Dict[str, Value]]:
    """逐层串联，返回 ``(y, state_out)``；非 stateful 时 ``state_out`` 为空。"""
    layer = spec.layer
    x = inputs["x"]
    state_out: Dict[str, Value] = {}
    for i in range(spec.num_layers):
        prefix = f"l{i}_"
        conv0 = inputs[prefix + "conv0"] if stateful else None
        h0 = inputs[prefix + "h0"] if stateful else None
        x, got = _mamba_layer(g, layer, x, inputs, prefix, conv0, h0, reset)
        if stateful:
            state_out[prefix + "conv"] = got["conv"]
            state_out[prefix + "h"] = got["h"]
    return x, state_out


def build_mamba_layers(
    g: Graph,
    spec: MambaModelSpec,
    x: Value,
    *,
    prefix: str = "",
    with_states: bool = False,
    reset: Optional[Value] = None,
) -> Tuple[Dict[str, Value], Value, Dict[str, Value]]:
    """从**已有 Value** 出发叠 N 层，参数单独建输入。

    与 :func:`build_mamba_model` 的区别只有一处：它不让本函数创建 ``x`` 输入。
    语言模型要把 embedding 的输出接进来（``x`` 是一张算子的结果而不是图输入），
    但又需要叠层函数返回同一套参数名与状态名。``inputs`` 的键已带 ``prefix``，
    且**不含** ``x``。
    """

    def name(base: str) -> str:
        return f"{prefix}{base}"

    raw: Dict[str, Value] = {}
    for pname, shape in spec.param_shapes.items():
        # 键保持原名（``_run_layers`` 按 ``l0_w_in`` 查找），图输入名才带前缀。
        raw[pname] = g.input(name(pname), f32[shape])
    if with_states:
        for sname, shape in spec.state_shapes.items():
            raw[sname] = g.input(name(sname), f32[shape])
    out, state_out = _run_layers(
        g, spec, {**raw, "x": x}, stateful=with_states, reset=reset
    )
    return {value.name: value for value in raw.values()}, out, state_out


def build_mamba_model(
    g: Graph, spec: MambaModelSpec | None = None, *, prefix: str = ""
) -> Tuple[Dict[str, Value], Value]:
    """训练口径的多层 Mamba 栈：``(inputs, y)``，状态全为零常量。

    每层是 ``rmsnorm -> 投影 -> 因果卷积 -> selective scan -> 门控 -> 投影``，
    残差加在层内（见 :func:`_mamba_layer`），所以输出与输入同形 ``(B, T, d)``，
    可直接叠到任何同宽的结构上。
    """
    spec = spec or MambaModelSpec()

    def name(base: str) -> str:
        return f"{prefix}{base}"

    inputs = _model_inputs(g, spec, name, with_states=False)
    out, _ = _run_layers(g, spec, inputs, stateful=False, reset=None)
    return inputs, out


def build_mamba_model_stateful(
    g: Graph,
    spec: MambaModelSpec | None = None,
    *,
    prefix: str = "",
    with_reset: bool = False,
) -> Tuple[Dict[str, Value], Value, Dict[str, Value]]:
    """流式口径的多层 Mamba 栈：``(inputs, y, state_out)``。

    输入表在 :func:`build_mamba_model` 的基础上多出每层的 ``l{i}_conv0`` /
    ``l{i}_h0``（形状见 :attr:`MambaModelSpec.state_shapes`）；``with_reset``
    再补一张**所有层共用**的 ``reset (B, T)``——"这个位置开始新序列"是数据层面
    的事实，不该逐层各说一遍。

    ``state_out`` 的键是 ``l{i}_conv`` / ``l{i}_h``（不能与输入的 ``l{i}_conv0`` /
    ``l{i}_h0`` 同名），用 :meth:`MambaModelSpec.carry_state` 翻回输入表即可续接：

        feed.update(spec.carry_state(state_out))

    ``spec.with_steps(1)`` 的同一份构图就是 decode，状态名不变，无需转换。
    """
    spec = spec or MambaModelSpec()

    def name(base: str) -> str:
        return f"{prefix}{base}"

    inputs = _model_inputs(g, spec, name, with_states=True)
    reset = None
    if with_reset:
        reset = inputs["reset"] = g.input(name("reset"), f32[spec.batch, spec.steps])

    out, state_out = _run_layers(g, spec, inputs, stateful=True, reset=reset)
    return inputs, out, state_out
