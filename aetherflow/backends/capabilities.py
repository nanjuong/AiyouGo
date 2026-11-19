"""后端能力契约：从"算子名集合"到"算子 × 形态"。

5.9 的准入面只要求 ``capabilities()`` 返回算子名集合。那能回答"有没有这个
算子"，回答不了"这个 dtype 行不行""是不是只支持最后一维""能不能进反向图
或图捕获"。只声明名字时，跑通一种形态就会被读成全面支持——这是审计 AF-32
第二点指出的缺口。

这里的做法是**扩展而不替换**：

- ``BackendCapabilities`` 是 ``frozenset`` 的子类。``set(caps)``、``len``、
  ``in``、与普通 ``frozenset`` 的相等比较全部照旧；后端仍可只返回一个普通
  集合，含义是"只承诺算子名"。
- 想说得更细的后端，为算子附一份 ``OpCapability``：dtype / axis / layout
  是**排除项**，``None`` 表示未声明限制；backward / capture / deterministic
  是**用途承诺**，默认全为 True。
- ``refined`` / ``unrefined`` 把"哪些算子说了形态细节、哪些只说了名字"
  暴露给调用方，避免把"没有排除"当成"已经验证"。

明细不参与相等与哈希：同一个算子集合可以有不同细粒度，但``==`` 仍按集合
比较。这是为了不打断既有的对账代码（如 CUDA 后端把 ``capabilities()`` 与
静态算子表直接比较）。需要比较细粒度时用 ``describe()``。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from ..ir.types import DType

if TYPE_CHECKING:  # 只用于注解，避免运行时把 ir 的图对象拖进来
    from ..ir.graph import Graph, Node

DTypeLike = Union[DType, str]
Axes = Union[int, Iterable[int]]


def _dtype_set(values: Optional[Iterable[DTypeLike]]) -> Optional[FrozenSet[DType]]:
    if values is None:
        return None
    return frozenset(v if isinstance(v, DType) else DType(v) for v in values)


def _axis_set(values: Optional[Axes]) -> Optional[FrozenSet[int]]:
    if values is None:
        return None
    if isinstance(values, int):
        return frozenset((values,))
    return frozenset(int(v) for v in values)


def _string_set(values: Optional[Iterable[str]]) -> Optional[FrozenSet[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return frozenset((values,))
    return frozenset(str(v) for v in values)


def _norm_axes(axes: Axes, rank: Optional[int]) -> Tuple[int, ...]:
    """把轴集合规范到 ``[0, rank)``；没有秩时原样返回。"""
    items = (axes,) if isinstance(axes, int) else tuple(axes)
    if rank is None or rank <= 0:
        return tuple(int(a) for a in items)
    return tuple(int(a) % rank for a in items)


@dataclass(frozen=True)
class OpCapability:
    """单个算子的形态明细。

    字段分两类，混在一起看会误读：

    - ``dtypes`` / ``axes`` / ``layouts`` 是**排除项**。``None`` 表示这个
      维度没有声明限制（不是"已验证全部取值"）；给了集合就表示只接收集合
      里的取值，其余在构图期/执行前就该被拒绝。
    - ``backward`` / ``capture`` / ``deterministic`` 是**承诺**，默认 True。
      它们不来自图，而是来自用途：训练要看 backward，图捕获要看 capture，
      一致性基准要看 deterministic。

    ``axes`` 只约束归约/归一化类算子的 ``axis`` / ``axes`` 属性，且声明里
    写正向或负向下标都可以（判定时按节点输入的秩一起规范化）。
    """

    dtypes: Optional[FrozenSet[DType]] = None
    axes: Optional[FrozenSet[int]] = None
    layouts: Optional[FrozenSet[str]] = None
    backward: bool = True
    capture: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtypes", _dtype_set(self.dtypes))
        object.__setattr__(self, "axes", _axis_set(self.axes))
        object.__setattr__(self, "layouts", _string_set(self.layouts))

    def explain(
        self,
        *,
        dtype: Optional[DTypeLike] = None,
        axis: Optional[Axes] = None,
        layout: Optional[str] = None,
        rank: Optional[int] = None,
        backward: bool = False,
        capture: bool = False,
        deterministic: bool = False,
    ) -> Optional[str]:
        """说明这一形态为什么不满足声明；全部满足返回 ``None``。

        ``dtype`` / ``axis`` / ``layout`` 传 ``None`` 表示"这一维没有问"，
        未声明的维度也不会拒绝。三个布尔参数表示"这次用途需要该能力"。
        """
        if dtype is not None and self.dtypes is not None:
            want = dtype if isinstance(dtype, DType) else DType(dtype)
            if want not in self.dtypes:
                allowed = sorted(d.value for d in self.dtypes)
                return f"dtype {want.value} 不在支持范围 {allowed}"

        if axis is not None and self.axes is not None:
            want_axes = _norm_axes(axis, rank)
            allowed_axes = _norm_axes(self.axes, rank)
            bad = sorted({a for a in want_axes if a not in allowed_axes})
            if bad:
                return f"axis {bad} 不在支持范围 {sorted(allowed_axes)}"

        if layout is not None and self.layouts is not None and layout not in self.layouts:
            return f"layout {layout!r} 不在支持范围 {sorted(self.layouts)}"

        if backward and not self.backward:
            return "不支持反向（backward）"
        if capture and not self.capture:
            return "不支持图捕获（capture）"
        if deterministic and not self.deterministic:
            return "不保证确定性（deterministic）"
        return None


class BackendCapabilities(frozenset):
    """算子名集合＋可选的形态明细。

    集合语义保持不变，明细是附加信息。没有明细的算子仍然算"已承接"，只是
    承诺粒度停留在算子名；``unrefined`` 会把这些算子单独列出来。
    """

    def __new__(
        cls,
        ops: Iterable[str] = (),
        details: Optional[Mapping[str, OpCapability]] = None,
    ) -> "BackendCapabilities":
        names = frozenset(ops)
        if details is None and isinstance(ops, BackendCapabilities):
            details = ops.details
        clean = dict(details or {})

        extra = sorted(set(clean) - names)
        if extra:
            raise ValueError(f"能力明细里出现了未声明承接的算子: {extra}")
        wrong = sorted(name for name, cap in clean.items() if not isinstance(cap, OpCapability))
        if wrong:
            raise TypeError(f"能力明细必须是 OpCapability，这些不是: {wrong}")

        obj = super().__new__(cls, names)
        # frozenset 子类实例带 __dict__，用 object.__setattr__ 绕开不可变限制；
        # 包成 MappingProxyType 是为了让"明细"也跟集合一样只读。
        object.__setattr__(obj, "_details", MappingProxyType(clean))
        return obj

    # -- 集合之外的只读面 ---------------------------------------------

    @property
    def details(self) -> Mapping[str, OpCapability]:
        return self._details

    @property
    def refined(self) -> FrozenSet[str]:
        """给了形态明细的算子。"""
        return frozenset(self._details)

    @property
    def unrefined(self) -> FrozenSet[str]:
        """只声明了名字、没有形态明细的算子。"""
        return frozenset(self) - self.refined

    def describe(self, op_name: str) -> Optional[OpCapability]:
        return self._details.get(op_name)

    def explain(self, op_name: str, **features) -> Optional[str]:
        """按给定形态检查某个算子；未承接或未细化返回 ``None`` / 原因。"""
        if op_name not in self:
            return f"未承接算子 {op_name}"
        detail = self._details.get(op_name)
        if detail is None:
            return None
        return detail.explain(**features)

    def supports(self, op_name: str, **features) -> bool:
        """True 只表示声明没有排除这一形态，不代表已被用例覆盖。"""
        return self.explain(op_name, **features) is None

    def explain_node(self, node: "Node") -> Optional[str]:
        """按节点实际形态（输出 dtype、axis/axes、layout）给出拒绝原因。"""
        name = node.op.name
        if name not in self:
            return f"未承接算子 {name}"
        detail = self._details.get(name)
        if detail is None:
            return None

        attrs = node.attrs
        rank = node.inputs[0].type.rank if node.inputs else None
        reason = detail.explain(
            dtype=node.output.type.dtype,
            axis=attrs.get("axis", attrs.get("axes")),
            layout=attrs.get("layout"),
            rank=rank,
        )
        if reason is None:
            return None
        return f"{name}: {reason}"

    def supports_node(
        self,
        node: "Node",
        *,
        backward: bool = False,
        capture: bool = False,
        deterministic: bool = False,
    ) -> bool:
        """节点形态与用途同时满足声明时返回 True。"""
        if self.explain_node(node) is not None:
            return False
        detail = self._details.get(node.op.name)
        if detail is None:
            return True
        return (
            detail.explain(
                backward=backward, capture=capture, deterministic=deterministic
            )
            is None
        )

    # -- 杂项 ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"BackendCapabilities({len(self)} 个算子，细化 {len(self._details)} 个)"

    def __reduce__(self):
        # pickle 默认只会还原 frozenset 的元素，明细会静默丢失
        return (type(self), (frozenset(self), dict(self._details)))


def to_capabilities(capabilities: Iterable[str]) -> BackendCapabilities:
    """把后端交出的能力声明统一成 ``BackendCapabilities``（不复制明细）。"""
    if isinstance(capabilities, BackendCapabilities):
        return capabilities
    return BackendCapabilities(capabilities)


def capability_gaps(
    graph: "Graph", capabilities: Iterable[str]
) -> List[Tuple[str, str]]:
    """返回 ``(算子名, 原因)``，按节点出现顺序、按算子名去重。

    只声明算子名的后端，gap 只有"未承接算子"一种；给了明细的后端还会因为
    dtype / axis / layout 不匹配产生 gap。
    """
    caps = to_capabilities(capabilities)
    gaps: List[Tuple[str, str]] = []
    seen = set()
    for node in graph.nodes:
        reason = caps.explain_node(node)
        if reason is None or node.op.name in seen:
            continue
        seen.add(node.op.name)
        gaps.append((node.op.name, reason))
    return gaps
