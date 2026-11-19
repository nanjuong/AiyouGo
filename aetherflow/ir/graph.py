"""计算图与构图 DSL。

Graph 是 IR 的容器：一种有向无环图，节点顺序即拓扑顺序（构图即排序）。
所有 Value / Node 都归属于唯一的 Graph，跨图引用会被立即拒绝，
这样后端插件拿到的图一定是自包含的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ops import OPS, Op, get_op
from .types import DType, TensorType
from .version import IR_VERSION


def _nested_shape(value) -> Tuple[int, ...]:
    """求嵌套 list/tuple 的形状（只沿第 0 个元素下降）。"""
    shape: List[int] = []
    while isinstance(value, (list, tuple)):
        shape.append(len(value))
        if not value:
            break
        value = value[0]
    return tuple(shape)


def _attr_repr(key: str, value) -> str:
    """属性打印。``constant.data`` 可能很大，只打形状摘要，避免 IR 文本刷屏。"""
    if key == "data" and isinstance(value, (list, tuple)):
        shape = "x".join(str(d) for d in _nested_shape(value)) or "scalar"
        return f"data=<{shape}>"
    return f"{key}={value!r}"


def _freeze_payload(value: Any) -> Any:
    """把常量载荷转成不可变的嵌套结构（list / tuple -> tuple）。

    ``data`` 是唯一允许嵌套的 attrs：只把顶层包成 tuple 的话，
    ``attrs["data"][0].append(...)`` 与"调用方手里还留着原 list"这两条路
    都能在 ``seal()`` 之后原地改掉常量内容，而 ``revision`` 不变——执行计划
    与捕获缓存会继续命中一张已经变了内容的图。转换一次的成本与写入常量同阶。
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _freeze_attrs(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    """把节点属性冻结成只读映射。

    顶层直接包 ``MappingProxyType``，不复制属性表本身。形状类属性
    （shape / perm / axes / output_size）在构建期多是 list，这里就地转成
    tuple：只包外壳的话，``node.attrs["shape"].append(...)`` 仍能绕过只读，
    而形状一变、执行计划却按旧签名命中，正是 seal 要堵的那条路。

    ``data`` 例外——常量载荷可能很大，逐层复制不划算；它在写入时就已经
    冻结成嵌套 tuple（见 ``constant``），这里不需要再处理。
    """
    frozen: Dict[str, Any] = {}
    for key, value in attrs.items():
        if key != "data" and isinstance(value, list):
            value = tuple(value)
        frozen[key] = value
    return MappingProxyType(frozen)


class _FrozenRecord:
    """seal 之后拒绝改写任何字段的基类。

    为什么不是"约定只读"：执行计划与捕获计划都按图缓存，判失效只看
    ``Graph.revision``。如果值名、类型或生产者能在缓存之后被改掉，缓存
    仍然命中而图已经变了，错误会表现为"某个后端结果对不上"——这种静默
    失效的排查成本远高于这里多抛一个异常。
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} 已冻结（图 seal 后不可改）：{name}；"
                "需要不同取值请在新图里重建对应记录"
            )
        object.__setattr__(self, name, value)


@dataclass(eq=False)
class Value(_FrozenRecord):
    """图中的一条数据边。

    比较与哈希都用**对象身份**（``eq=False``）：两张图各自声明的第一个输入
    可能同名同类型，结构相等并不代表它们是同一条边。按结构相等放行会让
    "把 A 图的输入塞进 B 图"这种错误在构图期静默通过，等到执行时才发现
    读的是同名但无关的张量。
    """

    id: int
    type: TensorType
    name: str
    producer: Optional["Node"] = None
    _frozen: bool = field(default=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type}"


@dataclass(eq=False)
class Node(_FrozenRecord):
    """图中的一次算子应用。当前实现为单输出，多输出留给后续扩展。"""

    op: Op
    inputs: Tuple[Value, ...]
    attrs: Mapping[str, Any]
    output: Value
    _frozen: bool = field(default=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return f"{self.op.name}({', '.join('%' + v.name for v in self.inputs)})"


class Graph:
    """可执行的 IR 图。"""

    def __init__(self, name: str = "main") -> None:
        self.name = name
        # 图的 IR 版本。进程内构造的图永远是当前版本；从 JSON 读入时，
        # 这里记录的是"它来自哪个版本"，供 verifier 与再次导出使用。
        self.ir_version = IR_VERSION
        # 读入时走过迁移的话，记下原始版本；进程内构造与会话内新图都是 None。
        # 它只作凭证：再次导出写的是迁移后的 ``ir_version``，不是这个。
        self.migrated_from: Optional[Tuple[int, int]] = None
        self._values: List[Value] = []
        self._nodes: List[Node] = []
        self._inputs: List[Value] = []
        self._outputs: List[Value] = []
        self._output_aliases: List[Optional[str]] = []
        self._by_name: Dict[str, Value] = {}
        # 结构版本号：每次新增输入 / 节点 / 输出都自增。执行器把与形状无关的
        # 执行计划（分组、输入位置、静态 attrs）挂在图旁边缓存，靠它判断失效。
        # 图的构建是唯一的修改入口，``seal()`` 之后连这条路也关掉，版本号
        # 因此成为真正的"结构不可变"凭证，而不是"调用方自觉"。
        self._revision = 0
        self._sealed = False

    # -- 构建 ---------------------------------------------------------

    def _check_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError(
                f"图 {self.name!r} 已 seal：执行计划与捕获缓存按 revision 复用，"
                "结构不能再改；需要不同结构请另建一张图"
            )

    def seal(self) -> "Graph":
        """冻结图：此后结构不可增删，值与节点的字段也不可改写。

        执行计划（``Executor._plan``）与捕获计划（``CudaBackend._graph_plan``）
        都以 ``revision`` 判失效；只要还能改图，"计划仍命中但图已经变了"就是
        一条静默错误通道。seal 把它变成显式失败：

        - ``input`` / ``add_node`` / ``output``（含 ``g.matmul(...)`` 语法糖）抛
          ``RuntimeError``；
        - ``Value`` / ``Node`` 的字段改写抛 ``AttributeError``；
          ``Node.attrs`` 变成只读映射，``attrs[k] = v`` 抛 ``TypeError``，
          ``attrs[k].append(...)`` 抛 ``AttributeError``（顶层 list 已换 tuple）。

        返回自身，便于 ``g = build().seal()``。重复 seal 是幂等的。
        """
        if self._sealed:
            return self
        for node in self._nodes:
            object.__setattr__(node, "attrs", _freeze_attrs(node.attrs))
            node._frozen = True
        for value in self._values:
            value._frozen = True
        self._sealed = True
        return self

    def input(self, name: str, type: TensorType) -> Value:
        """声明一个图输入（权重、激活都走这里）。

        可以在构图的任何时刻调用，包括已经有节点之后——eager 前端就是
        "用到时再声明"这样长大的。序列化因此不依赖"输入都排在节点前"：
        节点输出带显式 SSA 名，回读时把输入全部先重建只是顺序不同，
        引用按名字解析，不会断。值 id 会在回读时重排，但 id 不进存档，
        对外的契约是名字与类型。
        """
        self._check_mutable()
        if name in self._by_name:
            raise ValueError(f"名字重复: {name}")
        v = Value(id=len(self._values), type=type, name=name)
        self._values.append(v)
        self._inputs.append(v)
        self._by_name[name] = v
        self._revision += 1
        return v

    def add_node(
        self,
        op_name: str,
        args: Sequence[Value],
        attrs: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> Value:
        """新增一个节点，返回它的输出值。

        ``name`` 是显式 SSA 名，反序列化靠它无损重放：图允许"先建节点、
        后声明输入"（eager 前端就是这样长大的），而回读一律先重建全部输入，
        自动名所依赖的"当前值个数"与构图时不同——只有把名字写进存档，
        节点之间的引用才不会在重放时指错边。不传时按 ``v<id>`` 自动命名，
        并避开已经用掉的输入名与别名。
        """
        self._check_mutable()
        op = get_op(op_name)
        attrs = dict(attrs or {})
        op.check_attr_names(attrs)

        if not args and not op.nullary:
            raise ValueError(f"算子 {op_name} 至少需要一个输入")
        for a in args:
            if not isinstance(a, Value):
                raise TypeError(f"算子 {op_name} 的输入必须是 Value，得到 {type(a).__name__}")
            if a not in self._values:
                raise ValueError(f"算子 {op_name} 引用了不属于本图的 Value: %{a.name}")

        out_type = op.infer(tuple(a.type for a in args), attrs)
        index = len(self._values)
        if name is None:
            # 自动 SSA 名必须避开用户已经用掉的名字。用户完全可以把输入命名为
            # ``v1``，若这里直接取 ``v<len>`` 就会覆盖 ``_by_name``，序列化按名字
            # 重放时引用的就变成了另一条边（输出会静默变化）。名字与 id 因此分开
            # 分配：id 继续按顺序递增，名字只保证唯一。
            name = f"v{index}"
            suffix = 0
            while name in self._by_name:
                suffix += 1
                name = f"v{index}_{suffix}"
        elif name in self._by_name:
            raise ValueError(f"名字重复: {name}")
        out = Value(id=index, type=out_type, name=name)
        node = Node(op=op, inputs=tuple(args), attrs=attrs, output=out)
        out.producer = node

        self._values.append(out)
        self._nodes.append(node)
        self._by_name[name] = out
        self._revision += 1
        return out

    def output(self, value: Value, name: Optional[str] = None) -> None:
        """声明图输出。name 只是对外暴露的别名，不改变 SSA 名。

        同一个 Value 可以被声明为多个输出（不同别名），
        这在多个输出被 CSE 合并到同一计算时会真实发生。
        """
        self._check_mutable()
        if value not in self._values:
            raise ValueError(f"输出不属于本图: %{value.name}")
        if name is not None:
            if name in self._by_name:
                raise ValueError(f"名字重复: {name}")
            self._by_name[name] = value
        self._outputs.append(value)
        self._output_aliases.append(name)
        self._revision += 1

    @property
    def revision(self) -> int:
        """结构版本号：每新增一个输入 / 节点 / 输出就自增。

        执行器把与形状无关的执行计划按图缓存，用这个号判断是否失效。
        ``seal()`` 之后版本号不再变化——这正是缓存可以长期复用的依据。
        """
        return self._revision

    @property
    def sealed(self) -> bool:
        """图是否已冻结。"""
        return self._sealed

    @property
    def output_names(self) -> List[str]:
        return [v.name if alias is None else alias for v, alias in zip(self._outputs, self._output_aliases)]

    def constant(self, value: float = 0.0, dtype: Optional[DType] = None, shape=(), data=None) -> Value:
        """插入一个常量节点。

        不用"额外图输入"承载常量：输入要由调用方在每次执行时喂进来，
        常量一旦变成输入，图就不再自包含，反向图也就不能独立跑了。

        ``value`` 是标量填充；``data`` 给出任意常量（常量折叠的产物），
        两者互斥。data 按 JSON 原生类型存储（嵌套 list / tuple 均可），
        写入时转成嵌套 tuple：``seal()`` 之后既改不动它，调用方手里的原
        list 也不能再影响图。
        """
        attrs = {
            "dtype": (dtype or DType.F32).value,
            "shape": list(shape) if isinstance(shape, (tuple, list)) else [shape],
        }
        if data is None:
            attrs["value"] = float(value)
        else:
            attrs["data"] = _freeze_payload(data)
        return self.add_node("constant", [], attrs)

    def __getattr__(self, name: str):
        """构图语法糖：``g.matmul(x, w)`` 等价于 ``g.add_node("matmul", [x, w])``。"""
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in OPS:
            raise AttributeError(f"Graph 没有属性 {name!r}，也不是已知算子")
        op_name = name

        def call(*args: Value, **attrs) -> Value:
            return self.add_node(op_name, args, attrs)

        return call

    # -- 访问 ---------------------------------------------------------

    @property
    def inputs(self) -> Tuple[Value, ...]:
        return tuple(self._inputs)

    @property
    def outputs(self) -> Tuple[Value, ...]:
        return tuple(self._outputs)

    @property
    def nodes(self) -> Tuple[Node, ...]:
        return tuple(self._nodes)

    @property
    def values(self) -> Tuple[Value, ...]:
        return tuple(self._values)

    def find(self, name: str) -> Value:
        return self._by_name[name]

    def op_names(self) -> List[str]:
        return [n.op.name for n in self._nodes]

    # -- 打印 ---------------------------------------------------------

    def to_ir(self) -> str:
        """打印类 MLIR 的文本 IR，便于 review 与 diff。"""
        ins = ", ".join(f"%{v.name}: {v.type}" for v in self._inputs)
        outs = ", ".join(str(v.type) for v in self._outputs) if self._outputs else "()"
        lines = [f"func @{self.name}({ins}) -> {outs} {{"]
        for n in self._nodes:
            args = ", ".join(f"%{a.name}" for a in n.inputs)
            attr_str = ", ".join(_attr_repr(k, v) for k, v in n.attrs.items())
            sep = ", " if args and attr_str else ""
            port = "" if n.op.portability.value == "portable" else f"  // {n.op.portability.value}"
            lines.append(f"  %{n.output.name} = {n.op.name}({args}{sep}{attr_str}) : {n.output.type}{port}")
        if self._outputs:
            lines.append(f"  return {', '.join('%' + v.name for v in self._outputs)}")
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_ir()

    def __repr__(self) -> str:
        return f"Graph({self.name!r}, nodes={len(self._nodes)})"


def attrs_needed(args: str, attr_str: str) -> bool:
    return bool(args) and bool(attr_str)
