"""AetherFlow IR 类型系统。

设计约定
--------
1. 形状维三种形态：``int`` 是编译期已知的静态维，``str`` 是裸符号维，
   ``SymExpr`` 是由符号维与常量组成的整数表达式（如 ``(S + 2 - 3) / 2 + 1``）。
   符号维存在的意义：同一个 IR 可以同时服务不同 batch/序列长度/分辨率，
   而无需每次重新构图。
2. 类型是 IR 的一部分，不携带任何后端信息。任何后端都不得在类型里
   附加私有字段——这是保证 IR 硬件中立的底线。
3. 参考语义执行器只在语义层面校验类型（形状绑定），不做位级约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple, Union


# 二元整数运算的白名单。形状推导只需要这些：卷积/池化的输出长度是
# ``floor((S + 2p - span) / s) + 1``，自适应池化的窗口是 ``floor(i*S/O)``。
_SYM_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "floordiv": 2,
    "max": 2,
    "min": 2,
}
_SYM_FN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "floordiv": lambda a, b: a // b,
    "max": max,
    "min": min,
}
_SYM_TEXT = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "floordiv": "/",
    "max": "max",
    "min": "min",
}


@dataclass(frozen=True)
class SymExpr:
    """符号维上的整数表达式。

    只承载纯整数算术，求值发生在运行期符号绑定之后。表达式是**结构相等**
    的（``frozen`` dataclass），所以 ``(S + 1)`` 与 ``(S + 1)`` 天然相同，
    不需要额外的规约过程。
    """

    op: str
    args: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.op not in _SYM_ARITY:
            raise ValueError(f"未知的符号表达式运算: {self.op!r}")
        if len(self.args) != _SYM_ARITY[self.op]:
            raise ValueError(
                f"符号表达式 {self.op} 需要 {_SYM_ARITY[self.op]} 个操作数，"
                f"得到 {len(self.args)}"
            )
        for a in self.args:
            # 表达式内部的常量可以是负数（如 ``H - 1``），只有形状维本身
            # 才要求非负；这条区分由 ``allow_negative`` 明确表达。
            _check_dim(a, allow_negative=True)

    @property
    def is_static(self) -> bool:
        """所有叶子都是常量时为 True（构造器会把它折叠成 int，正常不会出现）。"""
        return all(isinstance(a, int) for a in self.args)

    @property
    def symbols(self) -> frozenset:
        found = set()
        for a in self.args:
            found |= dim_symbols(a)
        return frozenset(found)

    def __str__(self) -> str:
        # 完全括号化：形状串可能被写进日志或 IR 文本，宁可多一层括号，
        # 也不要在优先级上做假设。
        left, right = (_dim_str(a) for a in self.args)
        if self.op in ("max", "min"):
            return f"{self.op}({left}, {right})"
        if self.op == "add" and isinstance(self.args[1], int) and self.args[1] < 0:
            return f"({left} - {abs(self.args[1])})"
        return f"({left} {_SYM_TEXT[self.op]} {right})"


Dim = Union[int, str, SymExpr]


def _check_dim(d: Any, *, allow_negative: bool = False) -> None:
    if isinstance(d, bool):
        raise TypeError("形状维不能是 bool")
    if isinstance(d, int):
        if d < 0 and not allow_negative:
            raise ValueError(f"静态维不能为负: {d}")
        return
    if isinstance(d, SymExpr):
        return
    if isinstance(d, str):
        if not d.isidentifier():
            raise ValueError(f"符号维必须是合法标识符: {d!r}")
        return
    raise TypeError(f"非法的形状维类型: {type(d).__name__}")


def _dim_str(d: Dim) -> str:
    return str(d)


def dim_static(d: Dim):
    """所有叶子都是常量时返回 int，否则返回 None。"""
    if isinstance(d, int):
        return d
    if isinstance(d, SymExpr):
        values = [dim_static(a) for a in d.args]
        if any(v is None for v in values):
            return None
        return _SYM_FN[d.op](*values)
    return None


def dim_symbols(d: Dim) -> frozenset:
    if isinstance(d, SymExpr):
        return d.symbols
    if isinstance(d, str):
        return frozenset((d,))
    return frozenset()


def dim_resolve(d: Dim, binding: dict) -> int:
    """把维求值成具体整数。未绑定的符号抛 ``KeyError``。"""
    if isinstance(d, int):
        return d
    if isinstance(d, SymExpr):
        return _SYM_FN[d.op](*(dim_resolve(a, binding) for a in d.args))
    if d not in binding:
        raise KeyError(d)
    return int(binding[d])


def sym_binary(op: str, a: Dim, b: Dim) -> Dim:
    """构造 ``op(a, b)``，并做常量折叠与少量恒等式化简。

    化简不是为了好看：卷积与池化会套出 ``((H - 1) + 1)`` 这种链，
    不化简的话，"结果仍是 H"这件事在结构上就看不出来，后续算子之间
    的形状比较会凭空多出不必要的差异。
    """
    static = dim_static(a), dim_static(b)
    if static[0] is not None and static[1] is not None:
        if op == "floordiv" and static[1] == 0:
            raise ValueError(f"符号表达式除以零: {a} / {b}")
        return _SYM_FN[op](*static)

    sa, sb = static
    if op == "add":
        if sb == 0:
            return a
        if sa == 0:
            return b
        absorbed = _absorb_constant(a, b, sa, sb)
        if absorbed is not None:
            return absorbed
    elif op == "sub":
        if sb == 0:
            return a
    elif op == "mul":
        if sb == 1:
            return a
        if sa == 1:
            return b
        if sb == 0 or sa == 0:
            return 0
    elif op == "floordiv":
        if sb == 1:
            return a
    return SymExpr(op, (a, b))


def _absorb_constant(a: Dim, b: Dim, sa, sb) -> Any:
    """把 ``(x ± c1) + c2`` 合并成 ``x ± (c1 ± c2)``。"""
    for head, addend in ((a, sb), (b, sa)):
        if addend is None or not isinstance(head, SymExpr):
            continue
        if head.op not in ("add", "sub"):
            continue
        tail_value = dim_static(head.args[1])
        if tail_value is None or not isinstance(head.args[1], int):
            continue
        base = head.args[0]
        new = tail_value + addend if head.op == "add" else addend - tail_value
        return sym_binary("add", base, new)
    return None


def sym_add(a: Dim, b: Dim) -> Dim:
    return sym_binary("add", a, b)


def sym_mul(a: Dim, b: Dim) -> Dim:
    return sym_binary("mul", a, b)


class DType(str, Enum):
    """IR 支持的元素类型。

    枚举值同时用作 IR 文本表示，例如 ``f32[B, S, 64]``。
    """

    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I32 = "i32"
    BOOL = "bool"

    @property
    def np_dtype(self) -> str:
        """numpy 中用于承载该 dtype 的类型名。

        BF16 在 numpy 中没有原生类型，参考语义借用 float32 承载，
        只用于验证数学语义，不代表位宽。
        """
        return _NP_DTYPE[self]

    def __getitem__(self, shape: Union[Dim, Iterable[Dim]]) -> "TensorType":
        """``f32["B", "S", 64]`` 语法糖，用于在构图时声明张量类型。"""
        return TensorType(self, _as_shape(shape))


_DTYPE_NAMES = frozenset(d.value for d in DType)


_NP_DTYPE = {
    DType.F16: "float16",
    DType.BF16: "float32",
    DType.F32: "float32",
    DType.F64: "float64",
    DType.I8: "int8",
    DType.I32: "int32",
    DType.BOOL: "bool",
}


def _as_shape(shape: Union[Dim, Iterable[Dim]]) -> Tuple[Dim, ...]:
    if isinstance(shape, (int, str, SymExpr)):
        return (shape,)
    return tuple(shape)


@dataclass(frozen=True)
class TensorType:
    """张量类型：dtype + 形状。形状维可以是整数、符号名或符号表达式。"""

    dtype: DType
    shape: Tuple[Dim, ...]

    def __post_init__(self) -> None:
        for d in self.shape:
            _check_dim(d)
        # 形状符号与 dtype 名共用一套字符串：``f32[f32, 4]`` 无论让人还是让
        # 执行器都分不清哪一处是 dtype、哪一处是维。属性里的 ``dtype="f32"``
        # 也会被符号替换命中。这类图能构出来，却一定在运行期或序列化后出错，
        # 所以在进入 IR 的地方就拒绝。
        clash = self.symbols & _DTYPE_NAMES
        if clash:
            raise ValueError(
                f"形状符号 {sorted(clash)} 与 dtype 名冲突，"
                f"请改用不与 {sorted(_DTYPE_NAMES)} 重名的符号"
            )

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_static(self) -> bool:
        """所有维都是静态整数时为 True。"""
        return all(isinstance(d, int) for d in self.shape)

    @property
    def symbols(self) -> frozenset:
        found = set()
        for d in self.shape:
            found |= dim_symbols(d)
        return frozenset(found)

    def numel(self) -> int:
        """静态元素个数。含符号维时抛出 ValueError。"""
        if not self.is_static:
            raise ValueError(f"类型含符号维，无法计算元素个数: {self}")
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def with_dtype(self, dtype: DType) -> "TensorType":
        return TensorType(dtype, self.shape)

    def resolve(self, binding: dict) -> "TensorType":
        """把符号维替换成绑定值，返回完全静态的类型。"""
        resolved = tuple(dim_resolve(d, binding) for d in self.shape)
        return TensorType(self.dtype, resolved)

    def __str__(self) -> str:
        return f"{self.dtype.value}[{', '.join(str(d) for d in self.shape)}]"

    def __repr__(self) -> str:
        return f"TensorType({self.dtype.value}, {self.shape})"


# 常用 dtype 的短别名，构图时更接近数学写法
f16 = DType.F16
bf16 = DType.BF16
f32 = DType.F32
f64 = DType.F64
i8 = DType.I8
i32 = DType.I32
boolean = DType.BOOL
