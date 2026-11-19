"""IR 算子表。

每个算子由三部分组成，缺一不可：

1. ``infer``：类型推导。纯符号运算，必须能在符号维存在时工作。
2. ``reference``：参考语义。这是整个框架的"唯一真值"。
   任何后端（CUDA、其他硬件）的输出都必须与此语义等价。
   参考实现刻意只用 numpy 的标量/逐元素语义，不绑定任何加速库，
   也不约束累加顺序——位级一致性由后端自己负责，语义一致性以此为准。
3. ``portability``：可移植性标注。这是防止 IR 退化成"某一个后端调度器"
   的强制手段，verifier 会据此检查图是否可被目标后端承接。
4. ``effects``：副作用声明。空集表示纯函数，CSE / DCE / 常量折叠 / 激活重算
   都可以按"同样输入必定同样输出、删掉不留痕"来推理；一旦声明了 RNG /
   STATE / IO，这些优化必须让路（见 ``Effect``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .types import (
    DType,
    Dim,
    SymExpr,
    TensorType,
    _as_shape,
    dim_static,
    dim_symbols,
    sym_add,
    sym_binary,
)


class NoGradientError(ValueError):
    """这张图在该位置没有定义梯度。

    刻意不是 ``NotImplementedError``：它描述的是"数学上这里没定义/没给规则"，
    而不是"这个后端还没实现"，换一个后端也不会变好。反向规则缺失与
    算子缺失是两类问题，混用异常类型会把排查引向错误方向。
    """


class Portability(str, Enum):
    """节点可移植性。

    PORTABLE  通用数学语义，任何后端都必须支持或能拆解实现。
    LIBRARY   绑定某个外部库的能力与约束（如 cuDNN 卷积），仅该后端承接。
    VENDOR    厂商专有硬件能力，不可移植，禁止进入共享优化路径。
    """

    PORTABLE = "portable"
    LIBRARY = "library"
    VENDOR = "vendor"


class Effect(str, Enum):
    """算子的可观察效果。

    三者对优化的约束不同，不能压成一个"有副作用/没副作用"的布尔量：

    - ``RNG``：采样。输出由 seed / key 完全决定，但语义是"每次重新采样"。
      因此它**可以**在激活重算里重放（同一 key 复现同一位模式），但**不能**
      被常量折叠（噪声被固化）、也**不能**被 CSE 合并（两次采样被并成一次）。
    - ``STATE``：读写图外可变状态（内存池、缓存、文件句柄）。重放会推进或
      改写状态，所以不能进激活重算；删除未使用的节点也不是"清理"，而是改变
      行为。
    - ``IO``：对外可见的输入输出（打印、日志、主机传输）。同样不能被删或重放。

    空集表示纯函数——这正是 CSE / DCE / 折叠 / 重算默认成立的前提。
    """

    RNG = "rng"
    STATE = "state"
    IO = "io"


#: 不能在重算段里重复执行的效果。``RNG`` 不在其中：显式 key 下重放可复现。
UNREPLAYABLE_EFFECTS: FrozenSet[Effect] = frozenset({Effect.STATE, Effect.IO})


InferFn = Callable[[Tuple[TensorType, ...], dict], TensorType]
RefFn = Callable[[List[np.ndarray], dict], np.ndarray]
# 反向规则。签名 ``(ctx) -> 每个输入的梯度``，``ctx`` 见 ``ir/grad.py`` 的 GradCtx。
# 返回 None 的元素表示"该输入不接收梯度"（例如 softmax 的 scale、步进函数）。
VjpFn = Callable[[Any], Sequence[Optional[Any]]]


@dataclass(frozen=True)
class Op:
    name: str
    infer: InferFn
    reference: RefFn
    portability: Portability = Portability.PORTABLE
    attrs_schema: Tuple[str, ...] = field(default_factory=tuple)
    doc: str = ""
    vjp: Optional[VjpFn] = None
    """反向规则；None 表示这张图的这个算子不支持求导。

    "None" 与"规则里返回 None"是两种语义：前者在 ``backward`` 时**报错**，
    后者表示该输入确实不接收梯度。前者防止把"忘了写规则"静默当成"梯度为零"。
    """
    nullary: bool = False
    """是否允许零输入节点。目前只有 ``constant`` 需要。"""

    effects: FrozenSet[Effect] = field(default_factory=frozenset)
    """副作用集合。空集 = 纯函数，可被 CSE / DCE / 常量折叠 / 激活重算处理。"""

    def __post_init__(self) -> None:
        # 插件可能传 set / list / 裸字符串；统一成冻结集合，并拒绝拼错的名字。
        if not isinstance(self.effects, frozenset):
            object.__setattr__(self, "effects", frozenset(self.effects))
        bad = [e for e in self.effects if not isinstance(e, Effect)]
        if bad:
            raise TypeError(f"算子的 effects 必须是 Effect，得到 {bad}")

    @property
    def pure(self) -> bool:
        """没有任何可观察效果。优化 pass 的默认推理前提。"""
        return not self.effects

    def check_attr_names(self, attrs: dict) -> None:
        unknown = set(attrs) - set(self.attrs_schema)
        if unknown:
            raise TypeError(f"算子 {self.name} 不接受属性: {sorted(unknown)}")


OPS: Dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.name in OPS:
        raise ValueError(f"算子重复注册: {op.name}")
    OPS[op.name] = op
    return op


def get_op(name: str) -> Op:
    op = OPS.get(name)
    if op is None:
        raise KeyError(f"未知算子: {name}")
    return op


# --------------------------------------------------------------------------
# 形状工具
# --------------------------------------------------------------------------


def _broadcast_shape(a: Sequence[Dim], b: Sequence[Dim]) -> Tuple[Dim, ...]:
    """numpy 风格广播，但对符号维做保守处理。

    - 相同 → 保留
    - 一方为 1 → 取另一方
    - 两边都是静态且不等 → 报错
    - 其余（符号 vs 静态非 1、两个不同符号）→ 构图期拒绝

    最后一条是刻意的：符号维只知道自己"可能是 1"，不知道具体值，于是结果的
    那一维可能是静态值、也可能是符号，IR 里没有能表达这种选择的广播维表达式。
    此前取符号一方，等于把一个未必成立的假设写进类型——``(B, 2) + (3, 2)`` 在
    B=1 时本应是 ``(3, 2)``，运行期就会撞上"实际形状与 IR 推导不一致"。与其
    等到执行才报形状不符，不如在构图期就说清楚"这一维无法表示"。
    """
    out: List[Dim] = []
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            raise ValueError(f"无法广播的维度: {x} 与 {y}")
        else:
            raise ValueError(
                f"符号维 {x} 与 {y} 的广播结果无法在 IR 中表示："
                f"其中任一方都可能取 1。请把该维改成静态形状，"
                f"或先显式 reshape 到已知形状再相加"
            )
    longer = a if len(a) >= len(b) else b
    out.extend(reversed(longer[: abs(len(a) - len(b))]))
    return tuple(reversed(out))


def _norm_axes(axis, rank: int) -> Tuple[int, ...]:
    if axis is None:
        # 显式拒绝，而不是让 tuple(None) 抛一句 "'NoneType' object is not
        # iterable"：全归约在 IR 里没有独立的语义位，写清楚该写什么。
        raise ValueError(
            "axis=None（全归约）不受支持；"
            f"请显式列出所有轴，例如 axis=tuple(range({rank}))"
        )
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    norm = tuple(a + rank if a < 0 else a for a in axes)
    for a in norm:
        if a < 0 or a >= rank:
            raise ValueError(f"轴越界: {a}，秩为 {rank}")
    if len(set(norm)) != len(norm):
        raise ValueError(f"重复的轴: {norm}")
    return norm


def _elementwise_binary_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b = inputs
    if a.dtype != b.dtype:
        raise TypeError(f"二元算子要求两侧 dtype 一致，得到 {a.dtype.value} 与 {b.dtype.value}；请显式插入 cast")
    return TensorType(a.dtype, _broadcast_shape(a.shape, b.shape))


def _elementwise_unary_infer(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return inputs[0]


def _binary_ref(np_fn):
    """二元逐元素：溢出 / 无效值按 IEEE 返回，且不在宿主上打警告。

    f16 的 ``add`` 溢出成 ``inf`` 是位宽决定的语义，不是错误；numpy 默认会
    往 stderr 打一条 RuntimeWarning。这条口径与 ``div``、``exp/log/sqrt``
    一致——同一张图在参考后端与设备后端上不该有"日志差异"。
    """

    def ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np_fn(args[0], args[1])

    return ref


# --------------------------------------------------------------------------
# 逐元素算子
# --------------------------------------------------------------------------


for _name, _np_fn in (
    ("add", np.add),
    ("sub", np.subtract),
    ("mul", np.multiply),
):
    register(
        Op(
            name=_name,
            infer=_elementwise_binary_infer,
            reference=_binary_ref(_np_fn),
            doc=f"逐元素 {_name}，支持 numpy 广播",
        )
    )


def _div_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``div``：浮点按 IEEE，整数分母为 0 必须显式报错。

    浮点除零是**定义过的**（±inf / nan），这里不拦；整数没有这个语义——numpy
    会先升到 float64 再落回整数，``inf`` 转 i32 得到 INT_MIN，产物是垃圾值而
    警告在默认配置下被忽略。整数除法本身按截断定义（与参考执行器的 final cast
    一致），只有分母为 0 这件事无解，所以当场拒绝。
    """
    a, b = args
    if a.dtype.kind not in "fc" and np.any(b == 0):
        raise ValueError(
            f"div 的整数分母不能为 0（收到 dtype={a.dtype}）；"
            f"需要 IEEE 的 inf/nan 语义请先 cast 到浮点"
        )
    # 浮点除零是正常语义，不该顺带在宿主上打 numpy 警告
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b)


register(
    Op(
        name="div",
        infer=_elementwise_binary_infer,
        reference=_div_ref,
        doc="逐元素 div，支持 numpy 广播；浮点除零按 IEEE，整数分母为 0 显式拒绝",
    )
)


def _infer_silu(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (1, 2):
        raise ValueError(f"silu 需要 (x) 或 (x, gate) 输入，得到 {len(inputs)}")
    if len(inputs) == 2 and inputs[1] != inputs[0]:
        raise TypeError(f"silu 的 gate 必须与 x 同 dtype、同形状，得到 {inputs[1]} 与 {inputs[0]}")
    return inputs[0]


def _silu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    # clip 只是为了避免 exp 溢出警告，不改变数学语义
    out = x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    if len(args) > 1:
        # 可选门控：silu(x) * gate。SwiGLU 的逐元素乘法因此不需要单独物化。
        out = out * args[1]
    return out


register(
    Op(
        name="silu",
        infer=_infer_silu,
        reference=_silu_ref,
        doc="SiLU / swish: x * sigmoid(x)；可选第 2 个输入 gate 表示 silu(x) * gate"
        "（SwiGLU 的门控乘法，ADR-010 的输入扩展）",
    )
)


def _relu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.maximum(args[0], 0.0)


register(Op(name="relu", infer=_elementwise_unary_infer, reference=_relu_ref, doc="ReLU"))


_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_GELU_K = 0.044715


def _gelu_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + _GELU_K * x * x * x)))


register(
    Op(
        name="gelu",
        infer=_elementwise_unary_infer,
        reference=_gelu_ref,
        doc="GELU 的 tanh 近似：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))。"
        "不用 erf 精确式，因为 erf 不是已有算子；本式与 PyTorch 的 "
        "approximate='tanh' 逐点一致，是 GPT/BERT 系列实际使用的那个定义",
    )
)


# 自动微分需要的初等一元函数。写成一元算子而不是"反向专用算子"，
# 因为反向规则本身应当由**已有语义**组合出来：dx 只依赖 x、y 与基本算术，
# 这样反向图天然是 PORTABLE 的，任何能跑前向的后端都能跑反向。
# 只补 AD 与真实模型真正需要的那几个，不做成通用数学库。
# ``erf`` 是精确 GELU（PyTorch ``approximate='none'``）的必要项：
# 它无法由已有初等函数精确表出，所以只能作为一元算子存在。
def _sigmoid_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = np.clip(args[0], -60.0, 60.0)  # 只为避免 exp 溢出告警，不改语义
    return 1.0 / (1.0 + np.exp(-x))


# numpy 本身没有 erf（那是 scipy 的能力）。参考实现只负责定义语义，
# 用 math.erf 逐元素求值即可；慢一点没关系，精度是 float64。
_ERF_VEC = np.vectorize(math.erf, otypes=[np.float64])


def _erf_ref(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    return _ERF_VEC(x).astype(x.dtype, copy=False)


def _ieee_math(fn):
    """定义域外按 IEEE 返回，且不在宿主上打 numpy RuntimeWarning。

    ``log(0) = -inf``、``log(-1) = nan``、``sqrt(-1) = nan``、
    ``exp(x)`` 溢出成 ``inf`` 都是浮点语义的一部分（设备内核同样如此），
    不是错误；numpy 默认还会往 stderr 打一条 RuntimeWarning，同一张图在
    参考后端上跑就会污染日志。``div`` 早已按这条口径处理，这里补齐数学族。
    """

    def reference(args: List[np.ndarray], attrs: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            return fn(args[0])

    return reference


for _name, _ref, _doc in (
    ("exp", _ieee_math(np.exp), "自然指数 e^x"),
    ("log", _ieee_math(np.log), "自然对数 ln(x)"),
    ("sqrt", _ieee_math(np.sqrt), "平方根"),
    ("neg", lambda a, at: -a[0], "取负"),
    ("tanh", lambda a, at: np.tanh(a[0]), "双曲正切"),
    ("sigmoid", _sigmoid_ref, "logistic 函数 1/(1+e^-x)"),
    ("erf", _erf_ref, "误差函数 erf(x)"),
):
    register(Op(name=_name, infer=_elementwise_unary_infer, reference=_ref, doc=_doc))


def _ref_step(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    threshold = float(attrs.get("threshold", 0.0))
    return (args[0] > threshold).astype(args[0].dtype)


register(
    Op(
        name="step",
        infer=_elementwise_unary_infer,
        reference=_ref_step,
        attrs_schema=("threshold",),
        doc="Heaviside 阶跃：x > threshold → 1，否则 0（0 点取 0，与 ReLU 导数一致）",
    )
)


def _ref_spike(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    # 前向与 step 完全相同：只判断"是否越过阈值"，与膜电位的历史无关。
    # 阈值默认取 1.0（神经元语义），而 step 默认 0.0（通用阶跃）。
    threshold = float(attrs.get("threshold", 1.0))
    return (args[0] > threshold).astype(args[0].dtype)


register(
    Op(
        name="spike",
        infer=_elementwise_unary_infer,
        reference=_ref_spike,
        attrs_schema=("threshold", "alpha"),
        doc="脉冲发放：x > threshold → 1，否则 0。前向同 step，反向用替代梯度"
        "（窗宽 alpha 的三角窗），使 SNN 的发放过程可训练",
    )
)


# --------------------------------------------------------------------------
# SNN 事件驱动计算
# --------------------------------------------------------------------------


def _check_event_inputs(
    indices: TensorType,
    values: TensorType,
    offsets: TensorType,
    *,
    op: str,
) -> int:
    """校验事件三件套，返回 ``offsets`` 的**静态**长度（步数 + 1）。

    ``indices`` / ``values`` 的长度是事件数 ``E``：它随输入脉冲变化，在静态
    图里只能写成符号维（同一次执行里两者必须等长）。``offsets`` 的长度决定
    输出步数，必须静态——否则 ``infer`` 给不出输出形状。
    """
    if indices.rank != 1 or values.rank != 1 or offsets.rank != 1:
        raise ValueError(
            f"{op} 的 indices / values / offsets 都必须是一维，"
            f"得到 {indices.shape} / {values.shape} / {offsets.shape}"
        )
    if indices.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"{op} 的 indices 必须是 i32/i8，得到 {indices.dtype}")
    if offsets.dtype != DType.I32:
        raise TypeError(f"{op} 的 offsets 必须是 i32，得到 {offsets.dtype}")
    if indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"{op} 的 indices 与 values 必须等长，得到 "
            f"{indices.shape[0]} 与 {values.shape[0]}"
        )
    slots = dim_static(offsets.shape[0])
    if slots is None:
        raise ValueError(
            f"{op} 的 offsets 长度必须静态：它决定输出步数，"
            f"符号维 {offsets.shape[0]} 无法表达"
        )
    if slots < 1:
        raise ValueError(f"{op} 的 offsets 至少要有一个元素，得到 {slots}")
    return slots


def _check_event_offsets(offsets: np.ndarray, nnz: int, op: str) -> None:
    """``offsets`` 必须是 ``[0, nnz]`` 上的非降序列。

    内核按它切段；倒序或越界会读到别人的事件（甚至越界访存），因此这是
    **系统边界**上的校验，参考实现与后端都不能省略。
    """
    prev = int(offsets[0]) if offsets.size else 0
    if prev < 0 or prev > nnz:
        raise ValueError(f"{op} 的 offsets[0] 必须是 [0, {nnz}]，得到 {prev}")
    for s in range(1, int(offsets.shape[0])):
        cur = int(offsets[s])
        if cur < prev or cur > nnz:
            raise ValueError(
                f"{op} 的 offsets 必须是 [0, {nnz}] 上非降序列，"
                f"第 {s} 项是 {cur}（前一项 {prev}）"
            )
        prev = cur


def _infer_sparse_linear(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    weights, indices, values, offsets = inputs
    if weights.rank != 2:
        raise ValueError(
            f"sparse_linear 的 weights 必须是 (K, N) 二维，得到 {weights.shape}"
        )
    slots = _check_event_inputs(indices, values, offsets, op="sparse_linear")
    if values.dtype != weights.dtype:
        raise TypeError(
            f"sparse_linear 的 values 与 weights 必须同 dtype，"
            f"得到 {values.dtype} 与 {weights.dtype}"
        )
    return TensorType(values.dtype, (slots - 1, weights.shape[1]))


def _ref_sparse_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    weights, indices, values, offsets = args
    steps = int(offsets.shape[0]) - 1
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    out = np.zeros((steps, cols), dtype=values.dtype)
    # f16 的累加在内核里是 float；参考实现用同一宽度，对拍的才不是舍入差。
    acc = np.float32 if values.dtype == np.float16 else values.dtype
    w = weights.astype(acc, copy=False)
    v = values.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            out[s] = (v[lo:hi] @ w[idx[lo:hi]]).astype(values.dtype)
    return out


register(
    Op(
        name="sparse_linear",
        infer=_infer_sparse_linear,
        reference=_ref_sparse_linear,
        doc="事件驱动的稀疏线性：按 CSR 事件 (indices, values, offsets) 把"
        "weights[K, N] 的行加权求和成 values.dtype[S, N]。稠密的 S×K 脉冲"
        "矩阵从不物化，每个输出元素只读该步真正发放的事件；indices/offsets 是"
        "i32，事件数 E 可以是符号维",
    )
)


def _infer_sparse_linear_weight_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dout, indices, values, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"sparse_linear_weight_grad 的 dout 必须是 (S, N) 二维，得到 {dout.shape}"
        )
    slots = _check_event_inputs(
        indices, values, offsets, op="sparse_linear_weight_grad"
    )
    if dout.shape[0] != slots - 1:
        raise ValueError(
            f"sparse_linear_weight_grad 的 dout 步数应为 {slots - 1}，"
            f"得到 {dout.shape[0]}"
        )
    if values.dtype != dout.dtype:
        raise TypeError(
            f"sparse_linear_weight_grad 的 values 与 dout 必须同 dtype，"
            f"得到 {values.dtype} 与 {dout.dtype}"
        )
    rows = attrs.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise TypeError(
            f"sparse_linear_weight_grad 的 rows 必须是正整数，得到 {rows!r}"
        )
    return TensorType(dout.dtype, (rows, dout.shape[1]))


def _ref_sparse_linear_weight_grad(
    args: List[np.ndarray], attrs: dict
) -> np.ndarray:
    dout, indices, values, offsets = args
    rows = int(attrs["rows"])
    steps, cols = int(dout.shape[0]), int(dout.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear_weight_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear_weight_grad 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    dw = np.zeros((rows, cols), dtype=acc)
    v = values.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            # add.at 是"重复下标累加"：同一个神经元在一个 batch 里发放多次
            # 是常态，普通下标赋值只会留下最后一次贡献。
            contrib = v[lo:hi][:, None] * dout[s].astype(acc, copy=False)[None, :]
            np.add.at(dw, idx[lo:hi], contrib)
    return dw.astype(dout.dtype)


register(
    Op(
        name="sparse_linear_weight_grad",
        infer=_infer_sparse_linear_weight_grad,
        reference=_ref_sparse_linear_weight_grad,
        attrs_schema=("rows",),
        doc="sparse_linear 对 weights 的梯度：把 values[e] * dout[s] 散射累加"
        "回 indices[e] 那一行。rows 是权重行数 K——反向图里不再有 weights，"
        "只能由属性带过来",
    )
)


def _infer_sparse_linear_values_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    """``(dout, weights, indices, offsets) -> (E,)``。

    输出形状直接取 ``indices`` 的长度：它是事件数，可以和前向一样是符号维，
    且每个事件恰好产出一个标量。这里没有 ``values`` 这个输入，于是把
    ``indices`` 自己传给 ``_check_event_inputs`` 顶替"等长"这一条——事件数
    本来就是由 ``indices`` 定义的，与 ``values`` 等长在前向构图时已经查过。
    """
    dout, weights, indices, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 必须是 (S, N) 二维，"
            f"得到 {dout.shape}"
        )
    if weights.rank != 2:
        raise ValueError(
            f"sparse_linear_values_grad 的 weights 必须是 (K, N) 二维，"
            f"得到 {weights.shape}"
        )
    slots = _check_event_inputs(indices, indices, offsets, op="sparse_linear_values_grad")
    if dout.shape[1] != weights.shape[1]:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 与 weights 最后一维必须一致，"
            f"得到 {dout.shape[1]} 与 {weights.shape[1]}"
        )
    if dout.shape[0] != slots - 1:
        raise ValueError(
            f"sparse_linear_values_grad 的 dout 步数应为 {slots - 1}，"
            f"得到 {dout.shape[0]}"
        )
    if dout.dtype != weights.dtype:
        raise TypeError(
            f"sparse_linear_values_grad 的 dout 与 weights 必须同 dtype，"
            f"得到 {dout.dtype} 与 {weights.dtype}"
        )
    return TensorType(dout.dtype, indices.shape)


def _ref_sparse_linear_values_grad(
    args: List[np.ndarray], attrs: dict
) -> np.ndarray:
    """``dvalues[e] = dout[s(e), :] · weights[indices[e], :]``。

    ``s(e)`` 是事件 ``e`` 所属的步，由 ``offsets`` 唯一决定。与权重梯度不同，
    这里**没有重复下标问题**：每个事件只写一次，不需要 ``add.at``。
    """
    dout, weights, indices, offsets = args
    steps, cols = int(dout.shape[0]), int(dout.shape[1])
    rows = int(weights.shape[0])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "sparse_linear_values_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise ValueError(
            f"sparse_linear_values_grad 的 indices 必须落在 [0, {rows})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    d = dout.astype(acc, copy=False)
    w = weights.astype(acc, copy=False)
    dv = np.zeros(nnz, dtype=acc)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            dv[lo:hi] = (w[idx[lo:hi]] * d[s][None, :]).sum(axis=1)
    return dv.astype(dout.dtype)


register(
    Op(
        name="sparse_linear_values_grad",
        infer=_infer_sparse_linear_values_grad,
        reference=_ref_sparse_linear_values_grad,
        doc="sparse_linear 对事件幅度 values 的梯度：dvalues[e] = dout[s(e)] · "
        "weights[indices[e]]。s(e) 由 offsets 决定，每个事件只写一次，"
        "不需要原子加。输出长度等于事件数 E，可以是符号维",
    )
)


def _check_spike_matrix(spikes: TensorType, op: str) -> Tuple[int, int]:
    if spikes.rank != 2:
        raise ValueError(
            f"{op} 的输入必须是 (T, K) 二维脉冲矩阵，得到 {spikes.shape}"
        )
    if spikes.dtype not in (DType.F32, DType.F16):
        raise TypeError(f"{op} 的输入必须是 f32/f16，得到 {spikes.dtype}")
    rows = dim_static(spikes.shape[0])
    cols = dim_static(spikes.shape[1])
    if rows is None or cols is None:
        raise ValueError(
            f"{op} 的 T 与 K 必须静态：容量与输出长度都写在属性/形状里，"
            f"得到 {spikes.shape}"
        )
    return rows, cols


def _infer_spike_offsets(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    (spikes,) = inputs
    rows, _ = _check_spike_matrix(spikes, "spike_offsets")
    # 每行起点：长度为 T+1，最后一项是总事件数
    return TensorType(DType.I32, (rows + 1,))


def _ref_spike_offsets(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    (spikes,) = args
    rows = int(spikes.shape[0])
    offsets = np.zeros(rows + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(
        np.count_nonzero(spikes.astype(np.float32), axis=1), dtype=np.int64
    )
    return offsets


register(
    Op(
        name="spike_offsets",
        infer=_infer_spike_offsets,
        reference=_ref_spike_offsets,
        doc="统计 (T, K) 脉冲矩阵每行的非零数，给出长度为 T+1 的 CSR 起点表；"
        "末项就是总事件数。与 spike_csr 合起来构成层间压缩的两半",
    )
)


def _check_spike_offsets(
    spikes: TensorType, offsets: TensorType, op: str
) -> Tuple[int, int]:
    """校验 ``(T, K)`` 脉冲矩阵与长度 ``T+1`` 的 i32 起点表，返回 ``(T, K)``。"""
    rows, cols = _check_spike_matrix(spikes, op)
    if offsets.rank != 1 or offsets.dtype != DType.I32:
        raise TypeError(
            f"{op} 的 offsets 必须是一维 i32，得到 {offsets.shape}/{offsets.dtype}"
        )
    slots = dim_static(offsets.shape[0])
    if slots != rows + 1:
        raise ValueError(f"{op} 的 offsets 长度必须是 T+1={rows + 1}，得到 {slots}")
    return rows, cols


def _check_event_capacity(attrs: dict, rows: int, cols: int, op: str) -> int:
    """校验事件容量：必须是正整数，且不超过 ``T×K``。"""
    capacity = attrs.get("capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise TypeError(f"{op} 的 capacity 必须是正整数，得到 {capacity!r}")
    if capacity > rows * cols:
        raise ValueError(
            f"{op} 的 capacity={capacity} 超过 T×K={rows * cols}："
            "非零元素不可能比元素总数还多"
        )
    return capacity


def _infer_spike_csr(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    spikes, offsets = inputs
    rows, cols = _check_spike_offsets(spikes, offsets, "spike_csr")
    capacity = _check_event_capacity(attrs, rows, cols, "spike_csr")
    return TensorType(DType.I32, (capacity,))


def _ref_spike_csr(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    spikes, offsets = args
    rows = int(spikes.shape[0])
    capacity = int(attrs["capacity"])
    indices = np.zeros(capacity, dtype=np.int32)
    for t in range(rows):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        if hi < lo or hi > capacity:
            raise ValueError(
                f"spike_csr 的 offsets 必须在 [0, {capacity}] 上非降，"
                f"第 {t} 段是 [{lo}, {hi})"
            )
        cols_hit = np.nonzero(spikes[t].astype(np.float32))[0]
        if cols_hit.size != hi - lo:
            raise ValueError(
                f"spike_csr 的 offsets 与脉冲不一致：第 {t} 步有 "
                f"{cols_hit.size} 个非零，offsets 给出 {hi - lo}"
            )
        indices[lo:hi] = cols_hit
    return indices


register(
    Op(
        name="spike_csr",
        infer=_infer_spike_csr,
        reference=_ref_spike_csr,
        attrs_schema=("capacity",),
        doc="按 spike_offsets 给出的起点表，把 (T, K) 脉冲矩阵的非零列号写进"
        "长度固定的 indices[capacity]。容量固定是静态图的前提，溢出必须报错",
    )
)


def _check_event_indices(indices: TensorType, op: str) -> None:
    """校验 CSR 列号是一维 i32/i8。"""
    if indices.rank != 1 or indices.dtype not in (DType.I32, DType.I8):
        raise TypeError(
            f"{op} 的 indices 必须是 i32/i8 一维，"
            f"得到 {indices.shape}/{indices.dtype}"
        )


def _check_spike_linear_inputs(
    spikes: TensorType,
    weights: TensorType,
    indices: TensorType,
    offsets: TensorType,
    op: str,
) -> Tuple[int, int]:
    """校验融合事件线性的四个输入，返回 ``(步数, 权重列数)``。"""
    rows, cols = _check_spike_offsets(spikes, offsets, op)
    if weights.rank != 2:
        raise ValueError(f"{op} 的 weights 必须是 (K, N) 二维，得到 {weights.shape}")
    if weights.shape[0] != cols:
        raise ValueError(
            f"{op} 的 weights 行数应等于脉冲宽度 K={cols}，得到 {weights.shape[0]}"
        )
    if weights.dtype != spikes.dtype:
        raise TypeError(
            f"{op} 的 spikes 与 weights 必须同 dtype，"
            f"得到 {spikes.dtype} 与 {weights.dtype}"
        )
    _check_event_indices(indices, op)
    # 不把 N 转成 int：与 sparse_linear 一致，允许最后一维是符号维。
    return rows, weights.shape[1]


def _infer_spike_linear(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    spikes, weights, indices, offsets = inputs
    rows, out_cols = _check_spike_linear_inputs(
        spikes, weights, indices, offsets, "spike_linear"
    )
    return TensorType(spikes.dtype, (rows, out_cols))


def _ref_spike_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    spikes, weights, indices, offsets = args
    rows, cols = int(spikes.shape[0]), int(weights.shape[1])
    width = int(weights.shape[0])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "spike_linear")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= width):
        raise ValueError(
            f"spike_linear 的 indices 必须落在 [0, {width})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    out = np.zeros((rows, cols), dtype=spikes.dtype)
    acc = np.float32 if spikes.dtype == np.float16 else spikes.dtype
    sp = spikes.astype(acc, copy=False)
    w = weights.astype(acc, copy=False)
    for s in range(rows):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            hit = idx[lo:hi]
            out[s] = (sp[s, hit] @ w[hit]).astype(spikes.dtype)
    return out


register(
    Op(
        name="spike_linear",
        infer=_infer_spike_linear,
        reference=_ref_spike_linear,
        doc="事件驱动的脉冲线性：out[s, n] = Σ_e spikes[s, indices[e]] * W[indices[e], n]，"
        "e 只扫第 s 步实际发放的事件。值直接从脉冲矩阵读，不物化 values 缓冲，"
        "因此反向对 spikes 就是普通的 matmul，未发放位置的 surrogate 梯度不会丢",
    )
)


def _infer_spike_linear_weight_grad(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dout, spikes, indices, offsets = inputs
    if dout.rank != 2:
        raise ValueError(
            f"spike_linear_weight_grad 的 dout 必须是 (S, N) 二维，得到 {dout.shape}"
        )
    steps, width = _check_spike_offsets(spikes, offsets, "spike_linear_weight_grad")
    if dout.shape[0] != steps:
        raise ValueError(
            f"spike_linear_weight_grad 的 dout 步数应为 {steps}，得到 {dout.shape[0]}"
        )
    if dout.dtype != spikes.dtype:
        raise TypeError(
            f"spike_linear_weight_grad 的 spikes 与 dout 必须同 dtype，"
            f"得到 {spikes.dtype} 与 {dout.dtype}"
        )
    _check_event_indices(indices, "spike_linear_weight_grad")
    return TensorType(dout.dtype, (width, dout.shape[1]))


def _ref_spike_linear_weight_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    dout, spikes, indices, offsets = args
    steps, width = int(dout.shape[0]), int(spikes.shape[1])
    cols = int(dout.shape[1])
    nnz = int(indices.shape[0])
    _check_event_offsets(np.asarray(offsets), nnz, "spike_linear_weight_grad")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= width):
        raise ValueError(
            f"spike_linear_weight_grad 的 indices 必须落在 [0, {width})，"
            f"得到 [{int(idx.min())}, {int(idx.max())}]"
        )
    acc = np.float32 if dout.dtype == np.float16 else dout.dtype
    dw = np.zeros((width, cols), dtype=acc)
    sp = spikes.astype(acc, copy=False)
    do = dout.astype(acc, copy=False)
    for s in range(steps):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        if hi > lo:
            hit = idx[lo:hi]
            # add.at 是"重复下标累加"：同一神经元在同一步多次发放是常态。
            np.add.at(dw, hit, sp[s, hit][:, None] * do[s][None, :])
    return dw.astype(dout.dtype)


register(
    Op(
        name="spike_linear_weight_grad",
        infer=_infer_spike_linear_weight_grad,
        reference=_ref_spike_linear_weight_grad,
        doc="spike_linear 对权重的梯度：把 spikes[s, idx[e]] * dout[s] 散射累加回 "
        "idx[e] 那一行。值由脉冲矩阵现场读取，不需要单独的 values 缓冲",
    )
)


def _infer_constant(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return TensorType(DType(attrs["dtype"]), _as_shape(attrs["shape"]))


def _ref_constant(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    shape = _as_shape(attrs["shape"])
    dtype = DType(attrs["dtype"]).np_dtype
    data = attrs.get("data")
    if data is None:
        return np.full(shape, float(attrs.get("value", 0.0)), dtype=dtype)
    # data 是常量折叠产出的任意张量；JSON 里是嵌套 list，读回来可能是 tuple。
    return np.array(np.asarray(data, dtype=dtype).reshape(shape), dtype=dtype)


register(
    Op(
        name="constant",
        infer=_infer_constant,
        reference=_ref_constant,
        attrs_schema=("dtype", "shape", "value", "data"),
        doc="常量张量。零输入节点；值写在属性里，因此图仍然可序列化、可独立执行。"
        "``value`` 表示标量填充，``data`` 表示任意常量张量（常量折叠的产物），"
        "两者互斥",
        nullary=True,
    )
)


# --------------------------------------------------------------------------
# 矩阵算子
# --------------------------------------------------------------------------


def _infer_matmul(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b = inputs
    if a.rank < 2 or b.rank < 2:
        raise ValueError(f"matmul 需要秩 >= 2，得到 {a.rank} 与 {b.rank}")

    ka, kb = a.shape[-1], b.shape[-2]
    static_ka, static_kb = dim_static(ka), dim_static(kb)
    if static_ka is not None and static_kb is not None:
        if static_ka != static_kb:
            raise ValueError(f"matmul 内维不匹配: {ka} vs {kb}")
    elif static_ka is None and static_kb is None and ka != kb:
        raise ValueError(f"matmul 内维符号不一致: {ka} vs {kb}")

    batch = _broadcast_shape(a.shape[:-2], b.shape[:-2])
    return TensorType(a.dtype, batch + (a.shape[-2], b.shape[-1]))


def _ref_matmul(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.matmul(args[0], args[1])


register(
    Op(
        name="matmul",
        infer=_infer_matmul,
        reference=_ref_matmul,
        doc="批量矩阵乘：(..., M, K) x (..., K, N) -> (..., M, N)",
    )
)


# --------------------------------------------------------------------------
# 卷积
# --------------------------------------------------------------------------


def _pair(value, name: str) -> Tuple[int, int]:
    """把 int 或二元组规范成 ``(h, w)``。

    JSON 往返后 tuple 会变成 list，所以两种容器都必须接受。
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} 不能是 bool")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"{name} 必须是 int 或二元组，得到 {value!r}")


def _conv_out_dim(size: Dim, kernel: int, stride: int, padding: int, dilation: int) -> Dim:
    """NCHW 单个空间维的输出长度。

    静态维按定义式 ``floor((size + 2p - span) / stride) + 1`` 计算。

    符号维只在**结果恰好等于输入**时沿用原符号（kernel=1、stride=1、
    padding=0、dilation=1）；其余符号情形返回一个派生表达式，运行期绑定后
    求值。表达式只引用输入里已有的符号，不编造新名字。
    """
    span = dilation * (kernel - 1) + 1
    if not isinstance(size, int):
        if span == 1 and padding == 0 and stride == 1:
            return size
        # 其余符号情形用表达式表达，运行期绑定后再求值。符号维必须来自
        # 已出现的输入符号（与 reshape 的约定一致），表达式不引入新符号。
        out = sym_add(sym_binary("floordiv", sym_add(size, 2 * padding - span), stride), 1)
        return out
    out = (int(size) + 2 * padding - span) // stride + 1
    if out <= 0:
        raise ValueError(
            f"conv2d 输出尺寸非法: 输入 {size}、有效核 {span}、padding {padding}、stride {stride}"
        )
    return out


def _infer_conv2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (2, 3):
        raise ValueError(f"conv2d 需要 2 或 3 个输入（x, weight[, bias]），得到 {len(inputs)}")
    x, weight = inputs[0], inputs[1]
    if x.rank != 4 or weight.rank != 4:
        raise ValueError(f"conv2d 要求 x/weight 都是 NCHW 秩 4，得到 {x.rank} 与 {weight.rank}")
    if x.dtype != weight.dtype:
        raise TypeError(
            f"conv2d 要求 x 与 weight 的 dtype 一致，得到 {x.dtype.value} 与 {weight.dtype.value}"
        )

    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d 的 groups 必须 >= 1，得到 {groups}")

    sh, sw = _pair(attrs.get("stride", (1, 1)), "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    dh, dw = _pair(attrs.get("dilation", (1, 1)), "dilation")
    if sh < 1 or sw < 1 or dh < 1 or dw < 1:
        raise ValueError(f"conv2d 的 stride/dilation 必须 >= 1，得到 {(sh, sw)} 与 {(dh, dw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"conv2d 的 padding 必须 >= 0，得到 {(ph, pw)}")

    n, c, h, w = x.shape
    o, cg, kh, kw = weight.shape
    if not isinstance(kh, int) or not isinstance(kw, int):
        raise ValueError(f"conv2d 的卷积核尺寸必须静态已知，得到 {kh!r}x{kw!r}")
    if isinstance(o, int) and o % groups != 0:
        raise ValueError(f"conv2d 输出通道 {o} 不能被 groups={groups} 整除")
    if isinstance(c, int) and isinstance(cg, int):
        if c % groups != 0:
            raise ValueError(f"conv2d 输入通道 {c} 不能被 groups={groups} 整除")
        if cg != c // groups:
            raise ValueError(
                f"conv2d 权重通道 {cg} 应等于 输入通道/groups = {c // groups}"
            )
    elif not isinstance(c, int) and not isinstance(cg, int) and groups == 1 and c != cg:
        raise ValueError(f"conv2d 输入通道符号 {c} 与权重通道符号 {cg} 不一致")

    ho = _conv_out_dim(h, kh, sh, ph, dh)
    wo = _conv_out_dim(w, kw, sw, pw, dw)

    if len(inputs) == 3:
        bias = inputs[2]
        if bias.rank != 1:
            raise ValueError(f"conv2d 的 bias 必须是秩 1，得到秩 {bias.rank}")
        if bias.dtype != x.dtype:
            raise TypeError(
                f"conv2d 要求 bias 与 x 的 dtype 一致，得到 {bias.dtype.value} 与 {x.dtype.value}"
            )
        if isinstance(bias.shape[0], int) and isinstance(o, int) and bias.shape[0] != o:
            raise ValueError(f"conv2d bias 长度 {bias.shape[0]} 与输出通道 {o} 不一致")

    return TensorType(x.dtype, (n, o, ho, wo))


def _conv2d_cross(x, weight, sh, sw, ph, pw, dh, dw) -> np.ndarray:
    """单组 NCHW 交叉相关；调用方保证 ``x`` 通道数 == ``weight`` 第二维。"""
    n, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    out = np.zeros((n, o, ho, wo), dtype=np.result_type(x.dtype, weight.dtype))
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ]
            out += np.einsum("nchw,oc->nohw", patch, weight[:, :, i, j])
    return out


def _ref_conv2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 交叉相关（与 PyTorch ``conv2d`` 一致，不翻转核）。

    用 ``einsum`` 逐 (kh, kw) 位置累加，刻意不做 im2col：
    参考实现的职责是定义语义，不是定义性能。累加顺序固定，因而确定。
    分组时按输入/输出通道切片逐组计算，再沿输出通道拼回——与
    PyTorch 的 ``groups`` 语义一致：第 g 组只看 ``C/groups`` 个输入通道。
    """
    x, weight = args[0], args[1]
    sh, sw = _pair(attrs.get("stride", (1, 1)), "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    dh, dw = _pair(attrs.get("dilation", (1, 1)), "dilation")
    groups = int(attrs.get("groups", 1))

    if groups == 1:
        out = _conv2d_cross(x, weight, sh, sw, ph, pw, dh, dw)
    else:
        c = x.shape[1]
        o = weight.shape[0]
        cg, og = c // groups, o // groups
        parts = [
            _conv2d_cross(
                x[:, gi * cg : (gi + 1) * cg],
                weight[gi * og : (gi + 1) * og],
                sh, sw, ph, pw, dh, dw,
            )
            for gi in range(groups)
        ]
        out = np.concatenate(parts, axis=1)
    if len(args) > 2:
        out += args[2].reshape(1, -1, 1, 1)
    return out


register(
    Op(
        name="conv2d",
        infer=_infer_conv2d,
        reference=_ref_conv2d,
        attrs_schema=("stride", "padding", "dilation", "groups"),
        doc="NCHW 二维交叉相关：x(N,C,H,W) * weight(O,C/groups,kh,kw) + bias(O,)"
        " -> (N,O,Hout,Wout)；attrs 为 stride/padding/dilation 二元组与 groups，"
        "groups>1 即分组卷积，groups==C==O 即 depthwise",
    )
)


def _pool_out_dim(
    size: Dim, kernel: int, stride: int, padding: int, ceil_mode: bool = False
) -> Dim:
    """池化单个空间维的输出长度；符号维规则与 ``conv2d`` 一致。

    ``ceil_mode`` 用 PyTorch 的规则：先向上取整，再把"窗口起点落在输入（含左侧
    padding）之外"的那个多余窗口去掉（``(out - 1) * stride >= size + padding``）。
    第二条修正不能省——它保证每个窗口至少覆盖一个真实元素，也让 ceil 与 floor
    在整除时给出同一个结果。符号维上无法表达这个条件分支，因此显式报错。
    """
    if not isinstance(size, int):
        if ceil_mode:
            raise ValueError("ceil_mode 需要静态空间维：输出长度的上限修正依赖具体的 H / W")
        if kernel == 1 and padding == 0 and stride == 1:
            return size
        # 与卷积同构：符号维上保留派生表达式，运行期绑定后求值。
        return sym_add(
            sym_binary("floordiv", sym_add(size, 2 * padding - kernel), stride), 1
        )
    if ceil_mode:
        out = -((int(size) + 2 * padding - kernel) // -stride) + 1
        if (out - 1) * stride >= int(size) + padding:
            out -= 1
    else:
        out = (int(size) + 2 * padding - kernel) // stride + 1
    if out <= 0:
        raise ValueError(
            f"池化输出尺寸非法: 输入 {size}、核 {kernel}、padding {padding}、"
            f"stride {stride}、ceil_mode={ceil_mode}"
        )
    return out


def _pool_flags(attrs: dict) -> Tuple[bool, bool]:
    """池化的布尔属性 ``(ceil_mode, count_include_pad)``，缺省都是 False / True。

    ``count_include_pad`` 只对平均池化有意义，但放在这里统一解析：语义上它就是
    "分母是否含 padding 的零"，与 ``ceil_mode`` 一起决定窗口与除数。
    """
    ceil_mode = attrs.get("ceil_mode", False)
    if not isinstance(ceil_mode, bool):
        raise ValueError(f"ceil_mode 必须是 bool，得到 {ceil_mode!r}")
    count_include_pad = attrs.get("count_include_pad", True)
    if not isinstance(count_include_pad, bool):
        raise ValueError(f"count_include_pad 必须是 bool，得到 {count_include_pad!r}")
    return ceil_mode, count_include_pad


def _infer_maxpool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"maxpool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")

    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    if kh < 1 or kw < 1 or sh < 1 or sw < 1:
        raise ValueError(f"maxpool2d 的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"maxpool2d 的 padding 必须 >= 0，得到 {(ph, pw)}")
    ceil_mode, _ = _pool_flags(attrs)

    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    return TensorType(x.dtype, (n, c, ho, wo))


def _ref_maxpool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 最大池化；padding 区域不参与取 max（等价于填 -inf）。"""
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)

    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)

    # ceil_mode 的最后一个窗口会越过 H + 2p：补到窗口实际够用的长度，切片才
    # 不会短于输出长度（短了就无法与累积结果逐元素比较）。
    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (ph, need_h - h - ph), (pw, need_w - w - pw)),
        constant_values=-np.inf,
    )
    out = np.full((n, c, ho, wo), -np.inf, dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ]
            out = np.maximum(out, patch)
    return out


register(
    Op(
        name="maxpool2d",
        infer=_infer_maxpool2d,
        reference=_ref_maxpool2d,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode"),
        doc="NCHW 最大池化：kernel 必填，stride 默认等于 kernel，padding 默认 0；"
        "ceil_mode=True 时输出向上取整并去掉起点越界的多余窗口",
    )
)


def _pool_geometry(attrs: dict) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """池化的 (kernel, stride, padding)；stride 缺省等于 kernel。"""
    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")
    if kh < 1 or kw < 1 or sh < 1 or sw < 1:
        raise ValueError(f"池化的 kernel/stride 必须 >= 1，得到 {(kh, kw)} 与 {(sh, sw)}")
    if ph < 0 or pw < 0:
        raise ValueError(f"池化的 padding 必须 >= 0，得到 {(ph, pw)}")
    return (kh, kw), (sh, sw), (ph, pw)


def _infer_avgpool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"avgpool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    if x.dtype not in (DType.F16, DType.BF16, DType.F32, DType.F64):
        raise TypeError(f"avgpool2d 需要浮点输入，得到 {x.dtype.value}")
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    return TensorType(
        x.dtype,
        (
            n,
            c,
            _pool_out_dim(h, kh, sh, ph, ceil_mode),
            _pool_out_dim(w, kw, sw, pw, ceil_mode),
        ),
    )


def _pool_count_window(attrs: dict, h: int, w: int, ho: int, wo: int,
                       need_h: int, need_w: int) -> np.ndarray:
    """每个输出格的分母：窗口落在 padded 输入里的格数。

    ``count_include_pad=True`` 时 padding 计入（值按 0），区间是
    ``[0, H + 2p)`` / ``[0, W + 2p)``；``False`` 时只数真实输入，区间是
    ``[p, p + H)``。``ceil_mode`` 让最后一格越过 padded 输入时，越界部分两边
    都不计入——PyTorch 的 ``avg_pool2d`` 正是这么定义的，只按 ``KH * KW``
    除会在这种角落上偏小。分母兜底到 1，防零窗口除法。
    """
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    _, count_include_pad = _pool_flags(attrs)
    if count_include_pad:
        lo_h, hi_h = 0, h + 2 * ph
        lo_w, hi_w = 0, w + 2 * pw
    else:
        lo_h, hi_h = ph, ph + h
        lo_w, hi_w = pw, pw + w
    keep = np.zeros((1, 1, need_h, need_w), dtype=np.float64)
    keep[:, :, lo_h:hi_h, lo_w:hi_w] = 1.0
    count = np.zeros((1, 1, ho, wo), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            count += keep[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ]
    np.maximum(count, 1.0, out=count)
    return count


def _ref_avgpool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """NCHW 平均池化。

    分母是窗口落在 padded 输入里的格数：``count_include_pad=True`` 时 padding
    的零也计入，``False`` 时只数真实输入；``ceil_mode`` 的越界部分两边都不
    计入。窗口外（``ceil_mode`` 的最后一格）按 0 参与求和。
    """
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)

    # 与 maxpool2d 一样补到窗口实际够用的长度：ceil 的最后一格会越过 H + 2p。
    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.pad(
        x, ((0, 0), (0, 0), (ph, need_h - h - ph), (pw, need_w - w - pw))
    )
    count = _pool_count_window(attrs, h, w, ho, wo, need_h, need_w)
    acc = np.zeros((n, c, ho, wo), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            window = (
                slice(None),
                slice(None),
                slice(i, i + sh * (ho - 1) + 1, sh),
                slice(j, j + sw * (wo - 1) + 1, sw),
            )
            acc += padded[window]
    return (acc / count).astype(x.dtype)


register(
    Op(
        name="avgpool2d",
        infer=_infer_avgpool2d,
        reference=_ref_avgpool2d,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode", "count_include_pad"),
        doc="NCHW 平均池化：kernel 必填，stride 默认等于 kernel，padding 默认 0；"
        "count_include_pad=True（默认）时分母固定为 kernel 面积，False 时只数窗口"
        "与输入的重叠面积；ceil_mode 与 maxpool2d 同规则",
    )
)


def _infer_avgpool2d_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 NCHW 四元组，得到 {shape}")
    return TensorType(dy.dtype, shape)


def _ref_avgpool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """平均池化对输入的梯度：每个窗口把它那格的 dy/分母 均摊回去。

    分母与 ``avgpool2d`` 前向逐格对应（``_pool_count_window``）：padding 计不
    计入、``ceil_mode`` 的越界格都不进分母，所以不同输出格的分母可能不同，
    梯度不再是"乘一个常数"。
    """
    dy = args[0]
    n, c, h, w = (int(d) for d in attrs["input_shape"])
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    if dy.shape != (n, c, ho, wo):
        raise ValueError(
            f"avgpool2d_backward 的 dy 形状应为 {(n, c, ho, wo)}，得到 {dy.shape}"
        )

    need_h = max(h + 2 * ph, (ho - 1) * sh + kh)
    need_w = max(w + 2 * pw, (wo - 1) * sw + kw)
    padded = np.zeros((n, c, need_h, need_w), dtype=np.float64)
    share = dy.astype(np.float64) / _pool_count_window(
        attrs, h, w, ho, wo, need_h, need_w
    )
    for i in range(kh):
        for j in range(kw):
            padded[
                :, :, i : i + sh * (ho - 1) + 1 : sh, j : j + sw * (wo - 1) + 1 : sw
            ] += share
    return padded[:, :, ph : ph + h, pw : pw + w].astype(dy.dtype)


register(
    Op(
        name="avgpool2d_backward",
        infer=_infer_avgpool2d_backward,
        reference=_ref_avgpool2d_backward,
        attrs_schema=(
            "input_shape",
            "kernel",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
        ),
        doc="平均池化对输入的梯度：dy(N,C,HO,WO) -> dx(N,C,H,W)，"
        "input_shape 为原输入形状；除数与前向的 ceil_mode / count_include_pad 一致，"
        "与 maxpool2d_backward 一样只定义一阶",
    )
)


def _adaptive_window(index: int, out_size: int, in_size: int) -> Tuple[int, int]:
    """PyTorch 自适应池化的窗口 ``[start, end)``。

    定义是 ``start = floor(i*H/OH)``、``end = ceil((i+1)*H/OH)``。这里用
    非负整数的整除形式表达，避免浮点误差：``start = i*H // OH``，
    ``end = ((i+1)*H + OH - 1) // OH``。当 H 不是 OH 的整数倍时窗口可以
    重叠，反向因此必须累加而不是覆盖。
    """
    start = (index * in_size) // out_size
    end = ((index + 1) * in_size + out_size - 1) // out_size
    return start, end


def _adaptive_output_size(attrs: dict) -> Tuple[int, int]:
    raw = attrs["output_size"]
    if isinstance(raw, int) and not isinstance(raw, bool):
        oh = ow = int(raw)
    elif isinstance(raw, (tuple, list)) and len(raw) in (1, 2):
        oh = int(raw[0])
        ow = int(raw[-1])
    else:
        raise ValueError(f"output_size 必须是 int 或长度 1/2 的序列，得到 {raw!r}")
    if oh < 1 or ow < 1:
        raise ValueError(f"adaptive_avg_pool2d 的 output_size 必须 >= 1，得到 {(oh, ow)}")
    return oh, ow


def _infer_adaptive_avg_pool2d(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"adaptive_avg_pool2d 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    if x.dtype not in (DType.F16, DType.BF16, DType.F32, DType.F64):
        raise TypeError(f"adaptive_avg_pool2d 需要浮点输入，得到 {x.dtype.value}")
    oh, ow = _adaptive_output_size(attrs)
    n, c, _, _ = x.shape
    return TensorType(x.dtype, (n, c, oh, ow))


def _ref_adaptive_avg_pool2d(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """自适应平均池化：每个输出格是它那个窗口的均值。

    与 PyTorch ``adaptive_avg_pool2d`` 一致（``output_size=(1, 1)`` 就是
    全局平均池化）。窗口大小随输出位置变化，因此没有固定分母——这正是
    它与 ``avgpool2d`` 需要分成两个算子的原因。
    """
    x = args[0]
    n, c, h, w = x.shape
    oh, ow = _adaptive_output_size(attrs)
    out = np.empty((n, c, oh, ow), dtype=np.result_type(x.dtype, np.float32))
    for i in range(oh):
        hs, he = _adaptive_window(i, oh, h)
        for j in range(ow):
            ws, we = _adaptive_window(j, ow, w)
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(axis=(-2, -1))
    return out.astype(x.dtype)


register(
    Op(
        name="adaptive_avg_pool2d",
        infer=_infer_adaptive_avg_pool2d,
        reference=_ref_adaptive_avg_pool2d,
        attrs_schema=("output_size",),
        doc="NCHW 自适应平均池化：output_size 为 (OH, OW)，窗口按 "
        "start=floor(i*H/OH)、end=ceil((i+1)*H/OH) 取；output_size=(1,1) 即全局池化",
    )
)


def _infer_adaptive_avg_pool2d_backward(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 NCHW 四元组，得到 {shape}")
    if dy.rank != 4:
        raise ValueError(f"adaptive_avg_pool2d_backward 要求 dy 是秩 4，得到秩 {dy.rank}")
    return TensorType(dy.dtype, shape)


def _ref_adaptive_avg_pool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """自适应平均池化对输入的梯度：每个窗口把 dy/窗口面积 均摊回去。"""
    dy = args[0]
    n, c, h, w = (int(d) for d in attrs["input_shape"])
    oh, ow = _adaptive_output_size(attrs)
    if dy.shape != (n, c, oh, ow):
        raise ValueError(
            f"adaptive_avg_pool2d_backward 的 dy 形状应为 {(n, c, oh, ow)}，得到 {dy.shape}"
        )

    dx = np.zeros((n, c, h, w), dtype=np.float64)
    for i in range(oh):
        hs, he = _adaptive_window(i, oh, h)
        for j in range(ow):
            ws, we = _adaptive_window(j, ow, w)
            share = dy[:, :, i, j].astype(np.float64) / float((he - hs) * (we - ws))
            # 窗口可以重叠（H 不是 OH 的整数倍时），必须累加；
            # 这里用 += 而不是赋值，与 CUDA 侧的 atomicAdd 对应。
            dx[:, :, hs:he, ws:we] += share[:, :, None, None]
    return dx.astype(dy.dtype)


register(
    Op(
        name="adaptive_avg_pool2d_backward",
        infer=_infer_adaptive_avg_pool2d_backward,
        reference=_ref_adaptive_avg_pool2d_backward,
        attrs_schema=("input_shape", "output_size"),
        doc="自适应平均池化对输入的梯度：dy(N,C,OH,OW) -> dx(N,C,H,W)，"
        "input_shape 为原输入形状；窗口重叠处累加",
    )
)


# --------------------------------------------------------------------------
# 归一化与归约
# --------------------------------------------------------------------------


def _infer_softmax(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x = inputs[0]
    if len(inputs) > 1:
        scale = inputs[1]
        if not (scale.is_static and scale.numel() == 1):
            raise ValueError(f"softmax 的 scale 必须是标量，得到 {scale}")
    return x


def _ref_softmax(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    if len(args) > 1:
        x = x * args[1]
    axis = attrs.get("axis", -1)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


register(
    Op(
        name="softmax",
        infer=_infer_softmax,
        reference=_ref_softmax,
        attrs_schema=("axis",),
        doc="沿 axis 的数值稳定 softmax；可选第二输入为标量缩放，等价于 softmax(x * scale)",
    )
)


def _infer_attention(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(f"attention 需要 3 或 4 个输入（q, k, v[, mask]），得到 {len(inputs)}")
    q, k, v = inputs[0], inputs[1], inputs[2]
    for label, t in (("q", q), ("k", k), ("v", v)):
        if t.rank < 2:
            raise ValueError(f"attention 的 {label} 至少是 2 维，得到 {t}")
        if t.dtype != q.dtype:
            raise TypeError(
                f"attention 的 q/k/v dtype 必须一致，得到 {q.dtype.value} 与 {t.dtype.value}"
            )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"attention 的 q/k 内维必须相等，得到 {q.shape[-1]} 与 {k.shape[-1]}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"attention 的 k/v 序列长度必须相等，得到 {k.shape[-2]} 与 {v.shape[-2]}"
        )
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError(
            f"attention 要求 q/k/v 的批维完全一致，得到 "
            f"{q.shape[:-2]} / {k.shape[:-2]} / {v.shape[:-2]}；"
            "批维广播会让反向必须做运行期归约（与 conv 的符号维拒绝同一条理由）"
        )
    if len(inputs) == 4:
        mask = inputs[3]
        if mask.dtype != q.dtype:
            raise TypeError(
                f"attention 的 mask dtype 必须与 q 一致，得到 {mask.dtype.value} 与 {q.dtype.value}"
            )
        if mask.rank < 2:
            raise ValueError(f"attention 的 mask 至少是 2 维，得到 {mask}")
        if isinstance(mask.shape[-1], int) and isinstance(k.shape[-2], int) and mask.shape[-1] != k.shape[-2]:
            raise ValueError(
                f"attention 的 mask 最后一维 {mask.shape[-1]} 必须等于 k 的序列长度 {k.shape[-2]}"
            )
    float(attrs.get("scale", 1.0))
    p = float(attrs.get("dropout", 0.0))
    if not 0.0 <= p < 1.0:
        raise ValueError(f"attention 的 dropout 必须在 [0, 1) 内，得到 {p}")
    return TensorType(q.dtype, q.shape[:-2] + (q.shape[-2], v.shape[-1]))


def _ref_attention(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """``softmax(q @ k^T * scale [+ mask]) @ v``，可选对概率做 dropout。

    这里刻意物化 ``(..., S, T)`` 的分数矩阵：参考后端的职责是**定义语义**，
    不是省内存。分块、不物化、按 causal 跳过上三角块都是后端实现的事，
    用它们替换这里的实现不会改变任何数值定义。
    """
    q, k, v = args[0], args[1], args[2]
    scale = float(attrs.get("scale", 1.0))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if len(args) > 3:
        mask = args[3]
        # mask 可以比 scores 多前导维（例如 (1,1,S,T)）：先去掉长度 1 的
        # 前导维，否则 numpy 会把 (B,H,S,T) + (1,1,S,T) 广播成 (1,B,H,S,T)，
        # 输出凭空多一维——IR 推导与运行期形状校验都会拒绝这种结果。
        while mask.ndim > scores.ndim:
            if mask.shape[0] != 1:
                raise ValueError(
                    f"attention 的 mask 前导维必须是 1，得到 {mask.shape}"
                )
            mask = mask[0]
        scores = scores + mask
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    dropout = float(attrs.get("dropout", 0.0))
    if dropout > 0.0:
        probs = probs * _ref_dropout_mask(
            [probs], {"p": dropout, "seed": attrs.get("seed", 0)}
        )
    return np.matmul(probs, v)


register(
    Op(
        name="attention",
        infer=_infer_attention,
        reference=_ref_attention,
        attrs_schema=("scale", "dropout", "seed"),
        doc="融合注意力：softmax(q @ k^T * scale [+ mask]) @ v，对 (..., S, D) x "
        "(..., T, D) x (..., T, E) 求值，输出 (..., S, E)。mask 是可选的加性偏置"
        "（禁止位取大负数，见 causal_mask_bias），dropout 作用在 softmax 概率上。"
        "IR 里只留一个节点：中间分数矩阵不进入图，反向图也因此只有一份；"
        "是否分块、是否跳过上三角由后端决定，不改变这里的语义",
    )
)


def _infer_layernorm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (3, 4):
        raise ValueError(
            f"layernorm 需要 (x, gamma, beta) 或 (x, gamma, beta, residual) 输入，得到 {len(inputs)}"
        )
    x = inputs[0]
    if len(inputs) == 4:
        res = inputs[3]
        if res != x:
            raise TypeError(
                f"layernorm 的 residual 必须与 x 同 dtype、同形状，得到 {res} 与 {x}"
            )
    return x


def _ref_layernorm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma, beta = args[0], args[1], args[2]
    if len(args) > 3:
        # 残差在归一化之前相加：整段融合的关键是 add 的输出从不物化。
        # 用 numpy 的原 dtype 相加（f16 相加会按 f16 舍入），与内核的
        # "先舍入回存储类型、再累加"保持一致。
        x = x + args[3]
    axes = _norm_axes(attrs.get("axes", (-1,)), x.ndim)
    eps = float(attrs.get("eps", 1e-5))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


register(
    Op(
        name="layernorm",
        infer=_infer_layernorm,
        reference=_ref_layernorm,
        attrs_schema=("axes", "eps"),
        doc="LayerNorm(x, gamma, beta)，gamma/beta 广播到最后若干维；"
        "第 4 个可选输入 residual 使 LayerNorm(x + residual) 成为一个节点"
        "（ADR-010 的输入扩展，残差因此不需要先物化）",
    )
)


def _infer_rmsnorm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"rmsnorm 需要 (x, gamma) 两个输入，得到 {len(inputs)}")
    x, gamma = inputs
    if x.dtype != gamma.dtype:
        raise TypeError(
            f"rmsnorm 要求 x 与 gamma 同 dtype，得到 {x.dtype.value} 与 {gamma.dtype.value}"
        )
    if gamma.rank > x.rank:
        raise ValueError(f"gamma 的秩 {gamma.rank} 大于 x 的秩 {x.rank}")
    if x.is_static and gamma.is_static:
        try:
            np.broadcast_shapes(x.shape, gamma.shape)
        except ValueError as exc:
            raise ValueError(f"gamma 形状 {gamma.shape} 无法广播到 {x.shape}: {exc}") from exc
    return TensorType(x.dtype, x.shape)


def _ref_rmsnorm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma = args
    axes = _norm_axes(attrs.get("axes", (-1,)), x.ndim)
    eps = float(attrs.get("eps", 1e-6))
    ms = np.mean(x * x, axis=axes, keepdims=True)
    return x / np.sqrt(ms + eps) * gamma


register(
    Op(
        name="rmsnorm",
        infer=_infer_rmsnorm,
        reference=_ref_rmsnorm,
        attrs_schema=("axes", "eps"),
        doc="RMSNorm(x, gamma) = x / sqrt(mean(x^2) + eps) * gamma；"
        "没有均值中心化与 beta，比 LayerNorm 少一次归约（LLaMA/Mamba 的默认归一化）",
    )
)


def _norm_channel_axis(x: TensorType, attrs: dict, name: str) -> int:
    """通道轴规范化。BatchNorm 需要一个明确的通道维，不像 LayerNorm 可以给一组轴。"""
    if x.rank < 2:
        raise ValueError(f"{name} 要求秩 >= 2（batch 维 + 通道维），得到 {x.rank}")
    axis = int(attrs.get("axis", 1))
    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"{name} 的 axis 越界: {axis}，秩为 {x.rank}")
    return axis


def _infer_batch_norm(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 3:
        raise ValueError(f"batch_norm 需要 (x, gamma, beta) 三个输入，得到 {len(inputs)}")
    x, gamma, beta = inputs
    if x.dtype != gamma.dtype or x.dtype != beta.dtype:
        raise TypeError(
            f"batch_norm 要求三者同 dtype，得到 {x.dtype.value} / "
            f"{gamma.dtype.value} / {beta.dtype.value}"
        )
    axis = _norm_channel_axis(x, attrs, "batch_norm")
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t.rank != 1:
            raise ValueError(f"batch_norm 的 {name} 必须是一维 (C,)，得到秩 {t.rank}")
        ch = x.shape[axis]
        if isinstance(ch, int) and isinstance(t.shape[0], int) and t.shape[0] != ch:
            raise ValueError(f"batch_norm 的 {name} 长度 {t.shape[0]} 与通道数 {ch} 不符")
    return TensorType(x.dtype, x.shape)


def _ref_batch_norm(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x, gamma, beta = args
    axis = int(attrs.get("axis", 1)) % x.ndim
    eps = float(attrs.get("eps", 1e-5))
    red = tuple(i for i in range(x.ndim) if i != axis)
    mean = np.mean(x, axis=red, keepdims=True)
    var = np.var(x, axis=red, keepdims=True)
    bshape = [1] * x.ndim
    bshape[axis] = -1
    return (x - mean) / np.sqrt(var + eps) * gamma.reshape(bshape) + beta.reshape(bshape)


register(
    Op(
        name="batch_norm",
        infer=_infer_batch_norm,
        reference=_ref_batch_norm,
        attrs_schema=("axis", "eps"),
        doc="BatchNorm（训练态）：沿除 axis 外的所有轴求均值与方差，再按 "
        "(x-mean)/sqrt(var+eps)*gamma+beta 归一化，gamma/beta 形状 (C,)。"
        "running stats 不在图内：训练循环用 batch_norm_stats 取 mean/var 后自己维护，"
        "因为“更新动量”是训练策略，不是张量语义",
    )
)


def _infer_batch_norm_stats(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 1:
        raise ValueError(f"batch_norm_stats 需要 1 个输入，得到 {len(inputs)}")
    x = inputs[0]
    axis = _norm_channel_axis(x, attrs, "batch_norm_stats")
    return TensorType(x.dtype, (2, x.shape[axis]))


def _ref_batch_norm_stats(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    axis = int(attrs.get("axis", 1)) % x.ndim
    red = tuple(i for i in range(x.ndim) if i != axis)
    return np.stack([np.mean(x, axis=red), np.var(x, axis=red)]).astype(x.dtype)


register(
    Op(
        name="batch_norm_stats",
        infer=_infer_batch_norm_stats,
        reference=_ref_batch_norm_stats,
        attrs_schema=("axis",),
        doc="BatchNorm 的批统计量：返回 (2, C)，第 0 行是均值、第 1 行是方差。"
        "单独成算子是为了让训练循环能更新 running stats；它不参与反向，"
        "所以没有 vjp 规则——把它接进损失会在 backward 时明确报错，而不是静默给零",
    )
)


def _infer_reduce(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(t.dtype, shape)


def _reduce_axis(attrs: dict):
    """把 axis 归一化成 numpy 认识的形式。

    JSON 往返与反向规则都会把 axis 存成 list；numpy 只认 tuple/int。
    """
    axis = attrs.get("axis", -1)
    return tuple(axis) if isinstance(axis, list) else axis


def _ref_reduce_sum(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.sum(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_sum",
        infer=_infer_reduce,
        reference=_ref_reduce_sum,
        attrs_schema=("axis", "keepdims"),
        doc="求和归约",
    )
)


def _ref_reduce_max(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.max(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_max",
        infer=_infer_reduce,
        reference=_ref_reduce_max,
        attrs_schema=("axis", "keepdims"),
        doc="最大值归约。反向把梯度均分给所有等于最大值的元素（与 PyTorch 一致）",
    )
)


def _ref_reduce_mean(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.mean(
        args[0], axis=_reduce_axis(attrs), keepdims=attrs.get("keepdims", False)
    )


register(
    Op(
        name="reduce_mean",
        infer=_infer_reduce,
        reference=_ref_reduce_mean,
        attrs_schema=("axis", "keepdims"),
        doc="均值归约（求和不带缩放，缩放放在反向）",
    )
)


def _infer_argmax(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    # numpy 的 argmax 一次只归约一个轴；这里同样只接受单轴，避免
    # "多轴时索引该按什么顺序编号"这种必须自己定义语义的模糊地带。
    axes = _norm_axes(attrs.get("axis", -1), t.rank)
    if len(axes) != 1:
        raise ValueError(f"argmax 只接受一个轴，得到 {axes}")
    if attrs.get("keepdims", False):
        shape = tuple(1 if i in axes else d for i, d in enumerate(t.shape))
    else:
        shape = tuple(d for i, d in enumerate(t.shape) if i not in axes)
    return TensorType(DType.I32, shape)


def _ref_argmax(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    axis = _norm_axes(attrs.get("axis", -1), args[0].ndim)[0]
    out = np.argmax(args[0], axis=axis)
    if attrs.get("keepdims", False):
        out = np.expand_dims(out, axis)
    return out.astype(np.int32)


register(
    Op(
        name="argmax",
        infer=_infer_argmax,
        reference=_ref_argmax,
        attrs_schema=("axis", "keepdims"),
        doc="沿单个轴取最大值的下标，输出 i32。它不可导：接进损失会在 backward 时"
        "明确报错，而不是静默给零",
    )
)


# --------------------------------------------------------------------------
# 形状与类型算子
# --------------------------------------------------------------------------


def _infer_reshape(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    target = _as_shape(attrs["shape"])

    # reshape 不允许引入新的未知量：符号维必须在源类型中出现过
    for d in target:
        if isinstance(d, SymExpr):
            raise ValueError("reshape 的目标形状暂不支持符号表达式，请用裸符号或静态整数")
        unknown = dim_symbols(d) - t.symbols
        if unknown:
            raise ValueError(f"reshape 引入了未定义的符号维: {sorted(unknown)}")

    static_target = [d for d in target if isinstance(d, int)]
    if t.is_static and all(isinstance(d, int) for d in target):
        n_src, n_dst = t.numel(), 1
        for d in static_target:
            n_dst *= d
        if n_src != n_dst:
            raise ValueError(f"reshape 元素个数不守恒: {n_src} -> {n_dst}")

    return TensorType(t.dtype, target)


def _ref_reshape(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.reshape(args[0], attrs["shape"])


register(
    Op(
        name="reshape",
        infer=_infer_reshape,
        reference=_ref_reshape,
        attrs_schema=("shape",),
        doc="改变形状，元素顺序不变；不接受 -1，请显式给全维",
    )
)


def _infer_transpose(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    perm = tuple(attrs["perm"])
    if sorted(perm) != list(range(t.rank)):
        raise ValueError(f"非法置换 {perm}，秩为 {t.rank}")
    return TensorType(t.dtype, tuple(t.shape[i] for i in perm))


def _ref_transpose(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.transpose(args[0], attrs["perm"])


register(
    Op(
        name="transpose",
        infer=_infer_transpose,
        reference=_ref_transpose,
        attrs_schema=("perm",),
        doc="维度置换",
    )
)


def _infer_cast(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return TensorType(DType(attrs["dtype"]), inputs[0].shape)


def _ref_cast(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return args[0].astype(DType(attrs["dtype"]).np_dtype)


register(
    Op(
        name="cast",
        infer=_infer_cast,
        reference=_ref_cast,
        attrs_schema=("dtype",),
        doc="元素类型转换",
    )
)


def _infer_slice(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    dim = int(attrs["dim"]) % t.rank
    start = int(attrs.get("start", 0))
    stop = attrs.get("stop", t.shape[dim])
    stop = t.shape[dim] if stop is None else int(stop)
    if stop <= start:
        raise ValueError(f"slice 区间非法: start={start} stop={stop}")
    shape = list(t.shape)
    shape[dim] = stop - start
    return TensorType(t.dtype, tuple(shape))


def _ref_slice(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    dim = int(attrs["dim"]) % x.ndim
    start = int(attrs.get("start", 0))
    stop = attrs.get("stop", x.shape[dim])
    stop = x.shape[dim] if stop is None else int(stop)
    index = [slice(None)] * x.ndim
    index[dim] = slice(start, stop)
    return x[tuple(index)]


register(
    Op(
        name="slice",
        infer=_infer_slice,
        reference=_ref_slice,
        attrs_schema=("dim", "start", "stop"),
        doc="沿单个维度切片 [start, stop)，step 固定为 1",
    )
)


def _infer_broadcast_to(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    target = _as_shape(attrs["shape"])
    if t.is_static and all(isinstance(d, int) for d in target):
        try:
            np.broadcast_shapes(t.shape, target)
        except ValueError as exc:
            raise ValueError(f"broadcast_to 无法把 {t.shape} 广播到 {target}: {exc}") from exc
    return TensorType(t.dtype, target)


def _ref_broadcast_to(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.array(np.broadcast_to(args[0], _as_shape(attrs["shape"])))


register(
    Op(
        name="broadcast_to",
        infer=_infer_broadcast_to,
        reference=_ref_broadcast_to,
        attrs_schema=("shape",),
        doc="显式广播到目标形状（物料化，不返回视图）",
    )
)


def _infer_pad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    t = inputs[0]
    dim = int(attrs["dim"]) % t.rank
    before = int(attrs.get("before", 0))
    after = int(attrs.get("after", 0))
    if before < 0 or after < 0:
        raise ValueError(f"pad 的 before/after 必须 >= 0，得到 {before} 与 {after}")
    shape = list(t.shape)
    if isinstance(shape[dim], int):
        shape[dim] = int(shape[dim]) + before + after
    else:
        raise ValueError(
            f"pad 的第 {dim} 维是符号维 {shape[dim]!r}，加上常数后需要派生符号；"
            "请给出静态维"
        )
    return TensorType(t.dtype, tuple(shape))


def _ref_pad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    x = args[0]
    dim = int(attrs["dim"]) % x.ndim
    before = int(attrs.get("before", 0))
    after = int(attrs.get("after", 0))
    widths = [(0, 0)] * x.ndim
    widths[dim] = (before, after)
    return np.pad(x, widths)


register(
    Op(
        name="pad",
        infer=_infer_pad,
        reference=_ref_pad,
        attrs_schema=("dim", "before", "after"),
        doc="沿单个维度两侧补零；是 slice 的反向，语义与切片严格对称",
    )
)


def _infer_concat(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if not inputs:
        raise ValueError("concat 至少需要一个输入")
    first = inputs[0]
    ax = _norm_axes(attrs.get("axis", 0), first.rank)[0]
    shape = list(first.shape)
    total = 0
    for t in inputs:
        if t.dtype != first.dtype:
            raise TypeError(
                f"concat 要求所有输入 dtype 一致，得到 {first.dtype.value} 与 {t.dtype.value}"
            )
        if t.rank != first.rank:
            raise ValueError(f"concat 要求所有输入同秩，得到 {first.rank} 与 {t.rank}")
        for i in range(first.rank):
            if i == ax:
                continue
            if t.shape[i] != first.shape[i]:
                raise ValueError(
                    f"concat 的非拼接维必须一致：第 {i} 维为 {first.shape[i]} 与 {t.shape[i]}"
                )
        dim = t.shape[ax]
        if not isinstance(dim, int):
            raise ValueError(
                f"concat 的拼接维必须是静态量，得到符号维 {dim!r}；"
                "输出形状需要它求和，而符号维相加不是已定义的符号运算"
            )
        total += dim
    shape[ax] = total
    return TensorType(first.dtype, tuple(shape))


def _ref_concat(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    return np.concatenate(args, axis=_norm_axes(attrs.get("axis", 0), args[0].ndim)[0])


register(
    Op(
        name="concat",
        infer=_infer_concat,
        reference=_ref_concat,
        attrs_schema=("axis",),
        doc="沿 axis 拼接任意个同秩、非拼接维一致的张量；拼接维必须静态（输出形状要相加）",
    )
)


def _infer_embedding(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) != 2:
        raise ValueError(f"embedding 需要 (table, ids) 两个输入，得到 {len(inputs)}")
    table, ids = inputs
    if table.rank < 1:
        raise ValueError("embedding 的 table 至少是秩 1（V, ...）")
    if ids.dtype not in (DType.I32, DType.I8):
        raise TypeError(f"embedding 的 ids 必须是整数下标，得到 {ids.dtype.value}")
    if not isinstance(table.shape[0], int):
        raise ValueError(f"embedding 的词表大小必须静态，得到 {table.shape[0]!r}")
    return TensorType(table.dtype, tuple(ids.shape) + tuple(table.shape[1:]))


def _check_embedding_ids(ids: np.ndarray, rows: int, op: str) -> np.ndarray:
    """下标必须落在 ``[0, rows)``，越界显式报错。

    正向越界 numpy 自己会抛 IndexError，但**负下标**会静默回绕到倒数几行
    （``table[-1]`` 就是最后一行），反向的 ``np.add.at`` 同样会回绕——那是一种
    "看起来算完了"的错误结果。所以在参考语义里统一检查一次。
    """
    idx = np.asarray(ids)
    if idx.size:
        low, high = int(idx.min()), int(idx.max())
        if low < 0 or high >= rows:
            raise ValueError(
                f"{op} 的下标必须落在 [0, {rows})，得到 [{low}, {high}]"
            )
    return idx


def _ref_embedding(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """按第一维查表：``out[i] = table[ids[i]]``，越界显式报错。"""
    table, ids = args
    _check_embedding_ids(ids, int(table.shape[0]), "embedding")
    return table[ids]


register(
    Op(
        name="embedding",
        infer=_infer_embedding,
        reference=_ref_embedding,
        doc="查表：table(V, ...) 与整数 ids(...) -> (ids.shape..., table.shape[1:])；"
        "词表大小必须静态（越界检查与反向积累都依赖它）",
    )
)


def _infer_embedding_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy = inputs[0]
    shape = _as_shape(attrs["table_shape"])
    if len(shape) < 1:
        raise ValueError(f"table_shape 至少需要一维，得到 {shape}")
    if isinstance(shape[0], str):
        raise ValueError(f"词表大小必须静态，得到 {shape[0]!r}")
    return TensorType(dy.dtype, shape)


def _ref_embedding_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """查表的反向：把每一条 dy 散射累加回对应行。

    重复下标必须**累加**，不能用赋值——同一个 token 在一个 batch 里出现多次
    是常态。``np.add.at`` 正是为此存在的无缓冲散射。
    """
    dy, ids = args
    shape = tuple(int(d) for d in attrs["table_shape"])
    expected = tuple(ids.shape) + shape[1:]
    if dy.shape != expected:
        raise ValueError(f"embedding_backward 的 dy 形状应为 {expected}，得到 {dy.shape}")
    _check_embedding_ids(ids, shape[0], "embedding_backward")
    out = np.zeros(shape, dtype=dy.dtype)
    np.add.at(out, ids, dy)
    return out


register(
    Op(
        name="embedding_backward",
        infer=_infer_embedding_backward,
        reference=_ref_embedding_backward,
        attrs_schema=("table_shape",),
        doc="embedding 对表参数的梯度：dy(..., D) 与 ids(...) -> dtable(V, D)；重复下标累加",
    )
)


# --------------------------------------------------------------------------
# 序列扫描（SSM / 线性递归）
# --------------------------------------------------------------------------
# 递归 h_t = a_t * h_{t-1} + b_t 无法用逐元素算子展开成固定图（T 会变成节点数），
# 也没有闭式解可用：连乘 a 会在 T 长时下溢。所以它必须是一个语义算子。
# 注意这里只定义**最朴素的线性递归**，Mamba 的离散化（exp(dt*A)、dt*B*x）
# 全部由已有算子在前端组合出来——算子越简单，反向与跨后端实现越可控。


def _infer_scan_linear(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, b, h0 = inputs
    if a.dtype != b.dtype or a.dtype != h0.dtype:
        raise TypeError(
            f"scan_linear 要求三个输入同 dtype，得到 "
            f"{a.dtype.value} / {b.dtype.value} / {h0.dtype.value}"
        )
    if a.shape != b.shape:
        raise ValueError(f"scan_linear 的 a 与 b 形状必须一致，得到 {a.shape} 与 {b.shape}")
    if a.rank < 2:
        raise ValueError(f"scan_linear 至少需要 (T, D) 两个维度，得到 {a.shape}")
    if h0.shape != a.shape[1:]:
        raise ValueError(
            f"初始状态形状应为 {a.shape[1:]}（去掉时间维），得到 {h0.shape}"
        )
    if not isinstance(a.shape[0], int):
        raise ValueError(
            f"scan_linear 的时间维必须是静态整数，得到符号维 {a.shape[0]!r}；"
            "递归的边界与反向的切片位置都依赖它"
        )
    return a


def _ref_scan_linear(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    a, b, h0 = args
    steps = a.shape[0]
    out = np.empty_like(a)
    h = np.array(h0, copy=True)
    for t in range(steps):
        h = a[t] * h + b[t]
        out[t] = h
    return out


register(
    Op(
        name="scan_linear",
        infer=_infer_scan_linear,
        reference=_ref_scan_linear,
        attrs_schema=(),
        doc="线性递归 h_t = a_t * h_{t-1} + b_t（h_{-1} = h0）；SSM 的 selective scan 核心",
    )
)


def _infer_scan_linear_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    a, dy = inputs
    if a.dtype != dy.dtype:
        raise TypeError(
            f"scan_linear_backward 要求 a 与 dy 同 dtype，得到 "
            f"{a.dtype.value} 与 {dy.dtype.value}"
        )
    if a.shape != dy.shape:
        raise ValueError(
            f"scan_linear_backward 的 a 与 dy 形状必须一致，得到 {a.shape} 与 {dy.shape}"
        )
    if a.rank < 2:
        raise ValueError(f"scan_linear_backward 至少需要 (T, D) 两个维度，得到 {a.shape}")
    if not isinstance(a.shape[0], int):
        raise ValueError(
            f"scan_linear_backward 的时间维必须是静态整数，得到符号维 {a.shape[0]!r}"
        )
    return dy


def _ref_scan_linear_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """反向递归 dh_t = dy_t + a_{t+1} * dh_{t+1}，边界 dh_T = 0。

    这是线性递归的伴随：前向从 0 走到 T-1，反向从 T-1 走回 0。它同样不能
    用逐元素算子展开，所以与前向并列成为一个语义算子——不是"某个后端的内核"。
    """
    a, dy = args
    steps = a.shape[0]
    out = np.empty_like(dy)
    dh = np.zeros_like(dy[0])
    for t in range(steps - 1, -1, -1):
        nxt = a[t + 1] * dh if t + 1 < steps else 0.0
        dh = dy[t] + nxt
        out[t] = dh
    return out


register(
    Op(
        name="scan_linear_backward",
        infer=_infer_scan_linear_backward,
        reference=_ref_scan_linear_backward,
        attrs_schema=(),
        doc="线性递归的伴随：dh_t = dy_t + a_{t+1} * dh_{t+1}；scan_linear 反向图的组成部分",
    )
)


# --------------------------------------------------------------------------
# 卷积 / 池化的反向
# --------------------------------------------------------------------------


def _conv_geometry(attrs: dict):
    stride = _pair(attrs.get("stride", (1, 1)), "stride")
    padding = _pair(attrs.get("padding", (0, 0)), "padding")
    dilation = _pair(attrs.get("dilation", (1, 1)), "dilation")
    return stride, padding, dilation


def _infer_conv2d_input_grad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    dy, weight = inputs
    if dy.rank != 4 or weight.rank != 4:
        raise ValueError(f"conv2d_input_grad 要求 dy/weight 都是秩 4，得到 {dy.rank} 与 {weight.rank}")
    if dy.dtype != weight.dtype:
        raise TypeError(
            f"conv2d_input_grad 要求 dy 与 weight dtype 一致，得到 "
            f"{dy.dtype.value} 与 {weight.dtype.value}"
        )
    shape = _as_shape(attrs["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"input_shape 必须是 4 维，得到 {shape}")
    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d_input_grad 的 groups 必须 >= 1，得到 {groups}")
    c = shape[1]
    cw = weight.shape[1]
    if isinstance(c, int) and isinstance(cw, int) and cw * groups != c:
        raise ValueError(
            f"conv2d_input_grad 的权重通道 {cw} * groups={groups} 与 input_shape 通道 {c} 不一致"
        )
    return TensorType(dy.dtype, shape)


def _conv2d_input_grad_single(dy, weight, n, c, h, w, sh, sw, ph, pw, dh, dw):
    """单组卷积对输入的梯度；``dy``/``weight`` 都已按组切好。"""
    _, _, kh, kw = weight.shape
    _, _, ho, wo = dy.shape

    padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=dy.dtype)
    for i in range(kh):
        for j in range(kw):
            contrib = np.einsum("nohw,oc->nchw", dy, weight[:, :, i, j])
            padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ] += contrib
    return padded[:, :, ph : ph + h, pw : pw + w]


def _ref_conv2d_input_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """卷积 w.r.t. 输入的梯度。

    ``dx[n,c,ih,iw] = sum_{o,ki,kj} dy[n,o,ho,wo] * w[o,c,ki,kj]``，其中
    ``ih = ho*sh - ph + ki*dh``。这就是"转置卷积"的那条散射关系，但这里
    不引入转置卷积的概念，只定义输入梯度——IR 里不该出现两套卷积语义。
    分组时每组独立做一次，再沿输入通道拼回。
    """
    dy, weight = args[0], args[1]
    (sh, sw), (ph, pw), (dh, dw) = _conv_geometry(attrs)
    n, c, h, w = (int(v) for v in _as_shape(attrs["input_shape"]))
    groups = int(attrs.get("groups", 1))
    cw = weight.shape[1]
    if cw * groups != c:
        raise ValueError(
            f"conv2d_input_grad 的权重通道 {cw} * groups={groups} 与 "
            f"input_shape 通道 {c} 不一致"
        )
    if groups == 1:
        return _conv2d_input_grad_single(dy, weight, n, c, h, w, sh, sw, ph, pw, dh, dw)

    og = weight.shape[0] // groups
    parts = [
        _conv2d_input_grad_single(
            dy[:, gi * og : (gi + 1) * og],
            weight[gi * og : (gi + 1) * og],
            n, cw, h, w, sh, sw, ph, pw, dh, dw,
        )
        for gi in range(groups)
    ]
    return np.concatenate(parts, axis=1)


register(
    Op(
        name="conv2d_input_grad",
        infer=_infer_conv2d_input_grad,
        reference=_ref_conv2d_input_grad,
        attrs_schema=("input_shape", "stride", "padding", "dilation", "groups"),
        doc="conv2d 对 x 的梯度：dy(N,O,HO,WO) 与 w(O,C,kh,kw) -> dx(N,C,H,W)，"
        "几何参数与 conv2d 同名同义",
    )
)


def _infer_conv2d_weight_grad(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    x, dy = inputs
    if x.rank != 4 or dy.rank != 4:
        raise ValueError(
            f"conv2d_weight_grad 要求 x/dy 都是秩 4，得到 {x.rank} 与 {dy.rank}"
        )
    shape = _as_shape(attrs["weight_shape"])
    if len(shape) != 4:
        raise ValueError(f"weight_shape 必须是 4 维，得到 {shape}")
    groups = int(attrs.get("groups", 1))
    if groups < 1:
        raise ValueError(f"conv2d_weight_grad 的 groups 必须 >= 1，得到 {groups}")
    c = x.shape[1]
    cw = shape[1]
    if isinstance(c, int) and isinstance(cw, int) and cw * groups != c:
        raise ValueError(
            f"conv2d_weight_grad 的 x 通道 {c} 与 weight_shape 通道 {cw} * groups={groups} 不一致"
        )
    return TensorType(x.dtype, shape)


def _conv2d_weight_grad_single(x, dy, o, kh, kw, sh, sw, ph, pw, dh, dw):
    """单组卷积对权重的梯度；``x``/``dy`` 都已按组切好。"""
    n, c, h, w = x.shape
    _, _, ho, wo = dy.shape

    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    dw_out = np.zeros((o, c, kh, kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = padded[
                :,
                :,
                i * dh : i * dh + sh * (ho - 1) + 1 : sh,
                j * dw : j * dw + sw * (wo - 1) + 1 : sw,
            ]
            dw_out[:, :, i, j] = np.einsum("nohw,nchw->oc", dy, patch)
    return dw_out


def _ref_conv2d_weight_grad(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """卷积 w.r.t. 权重的梯度：``dw[o,c,ki,kj] = sum_{n,ho,wo} dy*x``。"""
    x, dy = args[0], args[1]
    (sh, sw), (ph, pw), (dh, dw) = _conv_geometry(attrs)
    n, c, h, w = x.shape
    o, cw, kh, kw = _as_shape(attrs["weight_shape"])
    groups = int(attrs.get("groups", 1))
    if cw * groups != c:
        raise ValueError(
            f"conv2d_weight_grad 的 x 通道 {c} 与 weight_shape 通道 {cw} * groups={groups} 不一致"
        )
    _, _, ho, wo = dy.shape

    if groups == 1:
        return _conv2d_weight_grad_single(x, dy, o, kh, kw, sh, sw, ph, pw, dh, dw)

    og = o // groups
    parts = [
        _conv2d_weight_grad_single(
            x[:, gi * cw : (gi + 1) * cw],
            dy[:, gi * og : (gi + 1) * og],
            og, kh, kw, sh, sw, ph, pw, dh, dw,
        )
        for gi in range(groups)
    ]
    return np.concatenate(parts, axis=0)


register(
    Op(
        name="conv2d_weight_grad",
        infer=_infer_conv2d_weight_grad,
        reference=_ref_conv2d_weight_grad,
        attrs_schema=("weight_shape", "stride", "padding", "dilation", "groups"),
        doc="conv2d 对 weight 的梯度：x(N,C,H,W) 与 dy(N,O,HO,WO) -> dw(O,C,kh,kw)",
    )
)


def _infer_maxpool2d_backward(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    return inputs[0]


def _ref_maxpool2d_backward(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """最大池化的输入梯度：把 dy 散射到每个窗口的最大值位置。

    并列最大值取**第一个**（行优先），因此结果确定、可对拍。
    padding 区域不参与竞争，与 ``maxpool2d`` 的前向语义严格对应。
    """
    x, dy = args[0], args[1]
    kh, kw = _pair(attrs["kernel"], "kernel")
    raw_stride = attrs.get("stride")
    sh, sw = _pair(attrs["kernel"] if raw_stride is None else raw_stride, "stride")
    ph, pw = _pair(attrs.get("padding", (0, 0)), "padding")

    n, c, h, w = x.shape
    _, _, ho, wo = dy.shape
    dx = np.zeros_like(x)
    for ni in range(n):
        for ci in range(c):
            for hi in range(ho):
                for wi in range(wo):
                    best = -np.inf
                    best_ij = None
                    for i in range(kh):
                        ih = hi * sh - ph + i
                        if ih < 0 or ih >= h:
                            continue
                        for j in range(kw):
                            iw = wi * sw - pw + j
                            if iw < 0 or iw >= w:
                                continue
                            v = x[ni, ci, ih, iw]
                            if best_ij is None or v > best:
                                best = v
                                best_ij = (ih, iw)
                    if best_ij is not None:
                        dx[ni, ci, best_ij[0], best_ij[1]] += dy[ni, ci, hi, wi]
    return dx


register(
    Op(
        name="maxpool2d_backward",
        infer=_infer_maxpool2d_backward,
        reference=_ref_maxpool2d_backward,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode"),
        doc="最大池化对输入的梯度：x(N,C,H,W) 与 dy(N,C,HO,WO) -> dx(N,C,H,W)；"
        "并列最大值取第一个；ceil_mode 只改变前向的输出长度，梯度沿同一组窗口散射",
    )
)


def _infer_maxpool2d_indices(
    inputs: Tuple[TensorType, ...], attrs: dict
) -> TensorType:
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"maxpool2d_indices 要求 NCHW 秩 4 输入，得到秩 {x.rank}")
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    return TensorType(
        DType.I32,
        (
            n,
            c,
            _pool_out_dim(h, kh, sh, ph, ceil_mode),
            _pool_out_dim(w, kw, sw, pw, ceil_mode),
        ),
    )


def _ref_maxpool2d_indices(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """最大池化的取值位置：每个窗口里最大元素的 ``ih * W + iw``（平面内展平下标）。

    这与 PyTorch ``max_pool2d(..., return_indices=True)`` 的第二个返回值同一个
    定义，也是 ``maxpool2d_backward`` 内部使用的定位规则——并列取第一个，因此
    三个算子（前向、位置、反向）在并列时选中同一个元素。
    """
    x = args[0]
    (kh, kw), (sh, sw), (ph, pw) = _pool_geometry(attrs)
    ceil_mode, _ = _pool_flags(attrs)
    n, c, h, w = x.shape
    ho = _pool_out_dim(h, kh, sh, ph, ceil_mode)
    wo = _pool_out_dim(w, kw, sw, pw, ceil_mode)
    out = np.full((n, c, ho, wo), -1, dtype=np.int32)
    for ni in range(n):
        for ci in range(c):
            for hi in range(ho):
                for wi in range(wo):
                    best = -np.inf
                    bi = bj = -1
                    for i in range(kh):
                        ih = hi * sh - ph + i
                        if ih < 0 or ih >= h:
                            continue
                        for j in range(kw):
                            iw = wi * sw - pw + j
                            if iw < 0 or iw >= w:
                                continue
                            v = x[ni, ci, ih, iw]
                            if bi < 0 or v > best:
                                best = v
                                bi, bj = ih, iw
                    if bi >= 0:
                        out[ni, ci, hi, wi] = bi * w + bj
    return out


register(
    Op(
        name="maxpool2d_indices",
        infer=_infer_maxpool2d_indices,
        reference=_ref_maxpool2d_indices,
        attrs_schema=("kernel", "stride", "padding", "ceil_mode"),
        doc="最大池化的取值位置（i32，平面内展平下标 ih*W+iw）："
        "对应 PyTorch max_pool2d(return_indices=True) 的第二个输出；"
        "并列最大值取第一个，与 maxpool2d_backward 的散射位置一致",
    )
)


# --------------------------------------------------------------------------
# 随机
# --------------------------------------------------------------------------

# SplitMix64 的三个常数。选它不是为了"够随机"，而是因为它只由整数乘法与
# 异或构成：numpy 参考实现与 CUDA 内核可以写出**逐位相同**的序列，
# 于是随机算子也能像确定性算子一样跨后端对拍。
# 用 numpy 内置的 Generator 会绑死 PCG64 的位级实现，而 CUDA 侧没有对应物。
_SM64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SM64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MUL2 = np.uint64(0x94D049BB133111EB)
_F32_2P24 = np.float32(16777216.0)  # 2^24，掩码只用 24 位尾数


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """逐元素的 SplitMix64 混合函数（uint64 数组）。"""
    z = (z ^ (z >> np.uint64(30))) * _SM64_MUL1
    z = (z ^ (z >> np.uint64(27))) * _SM64_MUL2
    return z ^ (z >> np.uint64(31))


def _infer_dropout_mask(inputs: Tuple[TensorType, ...], attrs: dict) -> TensorType:
    if len(inputs) not in (1, 2):
        raise ValueError(
            "dropout_mask 需要 1 或 2 个输入（x 只取形状与 dtype；"
            f"可选的第二个是运行时 key），得到 {len(inputs)}"
        )
    p = float(attrs.get("p", 0.5))
    if not 0.0 <= p < 1.0:
        # p = 1 时缩放因子 1/(1-p) 无定义；与其产生 inf，不如在构图期拒绝。
        raise ValueError(f"dropout 的 p 必须在 [0, 1) 内，得到 {p}")
    if len(inputs) == 2:
        key = inputs[1]
        if key.dtype is not DType.I32:
            raise ValueError(
                f"dropout_mask 的 key 必须是 i32 输入，得到 {key.dtype}"
            )
        if tuple(key.shape) not in ((), (1,)):
            raise ValueError(
                f"dropout_mask 的 key 必须是标量或长度 1 的向量，得到形状 {key.shape}"
            )
    return inputs[0]


def _dropout_seed(args: List[np.ndarray], attrs: dict) -> np.uint64:
    """取本次调用的 seed：运行时 key 输入优先，其次是 attrs 里的常量。

    ``offset`` 让同一张图里的多个 dropout 站点共用**一个** key 输入，却
    拿到互不相同的掩码；否则同形状的两层会得到逐位相同的掩码。
    """
    if len(args) > 1:
        seed = int(np.asarray(args[1]).reshape(-1)[0])
    else:
        seed = int(attrs.get("seed", 0))
    offset = int(attrs.get("offset", 0))
    return np.uint64((seed + offset) & 0xFFFFFFFFFFFFFFFF)


def _ref_dropout_mask(args: List[np.ndarray], attrs: dict) -> np.ndarray:
    """按扁平索引生成缩放后的掩码：保留 → 1/(1-p)，丢弃 → 0。

    输入 x **只用于确定形状与 dtype，数值完全不参与**——因此这个算子对 x
    没有梯度（见 ``grad_rules``），梯度全部经由 ``mul(x, mask)`` 那条边回传。
    """
    x = args[0]
    p = float(attrs.get("p", 0.5))
    seed = _dropout_seed(args, attrs)

    n = int(x.size)
    idx = np.arange(n, dtype=np.uint64)
    z = _splitmix64(seed + idx * _SM64_GOLDEN)
    # 取高 24 位：恰好是 float32 的尾数宽度，numpy 与 CUDA 得到同一个 float
    u = ((z >> np.uint64(40)) & np.uint64(0xFFFFFF)).astype(np.float32) / _F32_2P24
    keep = (u >= np.float32(p)).astype(np.float32)
    scale = np.float32(1.0 / (1.0 - p))
    return (keep * scale).reshape(x.shape).astype(x.dtype)


register(
    Op(
        name="dropout_mask",
        infer=_infer_dropout_mask,
        reference=_ref_dropout_mask,
        attrs_schema=("p", "seed", "offset"),
        doc="dropout 掩码：按 (seed + offset, 扁平索引) 用 SplitMix64 生成，"
        "保留的元素取 1/(1-p)、丢弃的元素取 0。输入 x 只提供形状与 dtype，"
        "数值不参与计算，也不接收梯度。seed 可以写进 attrs（图自包含、"
        "可复现），也可以由第二个 i32 输入在运行期给出——训练循环靠它"
        "逐步推进随机状态，同一 optimizer step 的前向与重算用同一个值；"
        "offset 让同图多个站点共用一个 key 却拿到不同掩码。"
        "参考后端与 CUDA 用同一套位运算，因此掩码可以逐位对拍",
        # 显式 seed/key 让结果确定，但语义是"每次重新采样"：不能折叠成常量，
        # 也不能被 CSE 合并成一次采样。重算段里可以重放（同一 key 同一结果）。
        effects=frozenset({Effect.RNG}),
    )
)
