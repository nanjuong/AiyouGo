"""SSM（状态空间模型）的模型级 IR 构件。

这里实现 Mamba 的 **selective scan（S6）**，但没有为它新增任何"SSM 专用"IR：

    离散化   a_t = exp(dt_t * A)                 -> exp / mul（已有）
    输入项   b_t = dt_t * B_t * x_t              -> mul（已有，靠广播）
    递归     h_t = a_t * h_{t-1} + b_t           -> scan_linear（新算子）
    读出     y_t = sum_n C_t[n] * h_t[..., n]    -> mul + reduce_sum(-1)（已有）
    跳连     y_t += D * x_t                      -> mul / add（已有）

只有**递归**必须是一个算子：它无法展开成固定图（T 会变成节点数），也没有
数值可用的闭式解（连乘 a 在 T 长时下溢）。其余全是已有语义的组合，所以
SSM、SNN、Transformer 共用同一套算子与同一套反向机制。

流式推理（prefill / decode）需要两样图外的东西：**末状态出参**与**重置**。
两者都不改算子，只是图上的组合：

    末状态   h_last = h[T-1]                     -> slice + reshape
    重置     h_t = keep_t * (a_t * h_{t-1} + b_t) -> 把 a 在 reset 位置乘 0

``keep = 1 - reset`` 让 reset 位置只用输入项 ``b_t`` 重建状态，等价于"从这里
开始一段新序列"；由于乘法发生在递归内部，前缀段的状态不会以任何形式泄漏。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..ir.graph import Graph, Value

__all__ = ["build_selective_scan", "build_selective_scan_stateful"]


def _check_static(value: Value, what: str) -> tuple:
    shape = tuple(value.type.shape)
    if not value.type.is_static:
        raise ValueError(
            f"{what} 需要静态形状，得到 {value.type}；"
            "scan 的递归步数与反向的切片位置都写在图里，不能随运行期变化"
        )
    return shape


def _extend(g: Graph, value: Value, shape: Sequence[int]) -> Value:
    """reshape 到显式形状（等价于 numpy 的 [..., None] 组合）。"""
    return g.reshape(value, shape=tuple(int(d) for d in shape))


def _selective_scan(g, x, dt, A, B_t, C_t, D_skip, h0, reset):
    """公共主体：返回 ``(y, h, h0, lead, dims, states)``。"""
    _, lead, dims, states = _check_inputs(x, dt, A, B_t, C_t, D_skip)
    dtype = x.type.dtype

    if h0 is None:
        h0 = g.constant(0.0, dtype=dtype, shape=(*lead[1:], dims, states))
    elif tuple(h0.type.shape) != (*lead[1:], dims, states):
        raise ValueError(
            f"初始状态形状应为 {(*lead[1:], dims, states)}，得到 {h0.type.shape}"
        )

    x_pre = _extend(g, x, (*lead, dims, 1))
    dt_pre = _extend(g, dt, (*lead, dims, 1))
    b_pre = _extend(g, B_t, (*lead, 1, states))
    c_pre = _extend(g, C_t, (*lead, 1, states))

    # a_t = exp(dt_t * A)：dt 是 (T,...,D,1)，A 是 (D,N)，广播成 (T,...,D,N)
    a = g.exp(g.mul(dt_pre, A))
    # b_t = (dt_t * B_t) * x_t
    b = g.mul(g.mul(dt_pre, b_pre), x_pre)

    if reset is not None:
        shape = _check_static(reset, "selective_scan 的 reset")
        if shape != tuple(lead):
            raise ValueError(
                f"reset 的形状应为 {tuple(lead)}（与 x 去掉末维一致），得到 {shape}"
            )
        keep = g.sub(g.constant(1.0, dtype=dtype, shape=()), reset)
        # (T, ...) -> (T, ..., 1, 1)，与 a 的 (T, ..., D, N) 对齐
        keep = _extend(g, keep, (*lead, 1, 1))
        a = g.mul(a, keep)

    h = g.scan_linear(a, b, h0)

    # y_t[..., d] = sum_n h_t[..., d, n] * C_t[..., n]
    y = g.reduce_sum(g.mul(h, c_pre), axis=-1)
    if D_skip is not None:
        y = g.add(y, g.mul(D_skip, x))
    return y, h, h0, lead, dims, states


def _check_inputs(x, dt, A, B_t, C_t, D_skip):
    """只做形状校验，不构造常量（常量必须在 Graph 上建）。"""
    x_shape = _check_static(x, "selective_scan 的 x")
    dt_shape = _check_static(dt, "selective_scan 的 dt")
    if x_shape != dt_shape:
        raise ValueError(f"x 与 dt 形状必须一致，得到 {x_shape} 与 {dt_shape}")
    if len(x_shape) < 2:
        raise ValueError(f"selective_scan 至少需要 (T, D)，得到 {x_shape}")

    lead = x_shape[:-1]
    dims = int(x_shape[-1])

    a_shape = tuple(A.type.shape)
    if len(a_shape) != 2 or a_shape[0] != dims:
        raise ValueError(f"A 的形状应为 ({dims}, N)，得到 {a_shape}")
    states = int(a_shape[1])

    for name, value in (("B_t", B_t), ("C_t", C_t)):
        shape = _check_static(value, f"selective_scan 的 {name}")
        if shape != (*lead, states):
            raise ValueError(
                f"{name} 的形状应为 {(*lead, states)}（(T, ..., N)），得到 {shape}"
            )

    if D_skip is not None and tuple(D_skip.type.shape) != (dims,):
        raise ValueError(f"D 的形状应为 ({dims},)，得到 {D_skip.type.shape}")
    return x_shape, lead, dims, states


def build_selective_scan(
    g: Graph,
    x: Value,
    dt: Value,
    A: Value,
    B_t: Value,
    C_t: Value,
    D_skip: Optional[Value] = None,
    h0: Optional[Value] = None,
    reset: Optional[Value] = None,
) -> Value:
    """构造一次 selective scan，返回 ``y``，形状与 ``x`` 相同 ``(T, ..., D)``。

    参数
    ----
    x
        输入序列 ``(T, ..., D)``。
    dt
        每步的时间步长 ``(T, ..., D)``，必须已过 softplus 之类的正值变换。
    A
        状态转移矩阵 ``(D, N)``，连续时间系数，通常取负值。
    B_t, C_t
        随步变化的输入 / 输出投影 ``(T, ..., N)``（这是 "selective" 的来源：
        普通 SSM 的 B/C 与时间无关）。
    D_skip
        可选跳连 ``(D,)``。
    h0
        可选初始状态 ``(..., D, N)``；缺省为零常量，图因此仍然自包含。
    reset
        可选重置掩码 ``(T, ...)``，1 表示"该位置开始一段新序列"：状态在那一
        步只由输入项重建，旧状态被丢弃。缺省表示整段连续。
    """
    y, _, _, _, _, _ = _selective_scan(g, x, dt, A, B_t, C_t, D_skip, h0, reset)
    return y


def build_selective_scan_stateful(
    g: Graph,
    x: Value,
    dt: Value,
    A: Value,
    B_t: Value,
    C_t: Value,
    D_skip: Optional[Value] = None,
    h0: Optional[Value] = None,
    reset: Optional[Value] = None,
) -> Tuple[Value, Value]:
    """同 :func:`build_selective_scan`，但额外返回末状态 ``h_last``。

    ``h_last`` 的形状与 ``h0`` 相同（``(..., D, N)``），可以直接作为下一次调用
    的 ``h0``：把长序列切成若干段、用末状态串起来，与整段前向在数学上等值。
    流式推理的 decode 就是 ``T=1`` 的这次调用。
    """
    y, h, h0, lead, dims, states = _selective_scan(
        g, x, dt, A, B_t, C_t, D_skip, h0, reset
    )
    steps = int(lead[0])
    last = g.slice(h, dim=0, start=steps - 1, stop=steps)
    h_last = g.reshape(last, shape=tuple(h0.type.shape))
    return y, h_last
