"""跨图控制流：按数据选择子图（``Conditional``）与按数据重复子图（``WhileLoop``）。

静态图是 DAG：节点序列里没有"这一层走哪支"的语义位，也不允许回边。此前
数据相关的分支只能写在两次 ``run`` 之间（见 ``aetherflow/dynamic.py``），
代价是每次分支都要把状态搬回 Python、重新取图。这一层补的是同一件事的
**显式表达**：控制流是"子图之间"的关系，每张子图仍是普通 ``Graph``——
照样过 verifier、照样能序列化、照样交给任何后端执行。

为什么不做成 IR 算子（把子图塞进节点属性）
------------------------------------------
1. 节点属性必须能完整 JSON 序列化，而 ``Graph`` 是活对象（值 id、生产者
   指针、seal 状态），塞进 ``attrs`` 之后 ``dumps`` 会立刻失效；
2. 设备后端要递归编译嵌套子图，等于给每个后端各写一遍控制流语义，而这
   正是"IR 是唯一契约"要避免的事；
3. 谓词必须回主机读一次真值。这一点绕不过：数据相关的分支本来就不在
   CUDA Graph 可捕获的范围内（捕获要求路径固定），把它做进 IR 不会让任何
   后端跑得更快，只会让语义边界更模糊。

因此控制流落在"图与图之间"：``ControlRegion`` 组合若干张图，执行时由
``run`` 在主机侧读回谓词、决定执行哪张图。子图可以嵌套控制节点，所以
``if`` 里面套 ``while`` 不需要额外机制。

契约（都是显式失败，不做隐式降级）
----------------------------------
- 谓词/条件：恰好一个输出，且是 0 维标量；bool 或数值都接受，取 Python
  真值（``0`` / ``False`` 为假，``NaN`` 为真——与 Python 语义一致）。
- ``Conditional``：两个分支的输出**名字与顺序必须一致**，同名输出的类型
  必须相等。只有被选中的分支执行，未选中分支里出错也不会被触发。
- ``WhileLoop``：循环携带量（``carried``）不能为空、不能重名，必须是循环体
  的输入，且循环体必须逐一同名同类型地输出它们；``max_steps`` 必须显式
  给出，达到上限仍未退出就抛 ``ControlFlowError``，而不是静默截断。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .graph import Graph
from .serialize import from_dict, to_dict
from .types import DType, TensorType

__all__ = [
    "ControlFlowError",
    "ControlRegion",
    "GraphRegion",
    "Conditional",
    "WhileLoop",
    "to_spec",
    "from_spec",
]


class ControlFlowError(RuntimeError):
    """控制流无法按契约收敛或求值。

    与 ``ValueError`` 区分：参数不合法是构图期的问题（在构造器里当场抛），
    这里是"图本身合法，但这次执行没有结束"——例如 ``max_steps`` 用尽。
    """


class ControlRegion:
    """控制流图的统一接口：一组输入名/类型、一组输出名/类型、一次执行。

    统一成接口而不是让 ``Conditional`` / ``WhileLoop`` 各写一套，是为了让
    它们可以互相嵌套：``Conditional`` 的分支可以是另一个 ``Conditional``，
    ``WhileLoop`` 的循环体也可以是，验证与执行都不需要分支判断。
    """

    @property
    def inputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        raise NotImplementedError

    @property
    def outputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        raise NotImplementedError

    def run(self, feeds: Mapping[str, np.ndarray], backend=None) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def _backend_or_default(backend):
    if backend is not None:
        return backend
    from ..backends.reference_backend import ReferenceBackend

    return ReferenceBackend()


class GraphRegion(ControlRegion):
    """把一张普通 ``Graph`` 当成控制流区域。

    输入/输出名字就是图上的输入名与输出别名——控制流的连线全靠名字，
    不另造一套映射表，否则"图里叫什么"与"程序里叫什么"迟早对不上。
    """

    def __init__(self, graph: Graph) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"GraphRegion 需要 Graph，得到 {type(graph).__name__}")
        self.graph = graph

    @property
    def inputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return tuple((value.name, value.type) for value in self.graph.inputs)

    @property
    def outputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return tuple(
            (name, value.type)
            for name, value in zip(self.graph.output_names, self.graph.outputs)
        )

    def run(self, feeds: Mapping[str, np.ndarray], backend=None) -> Dict[str, np.ndarray]:
        return _backend_or_default(backend).execute(self.graph, feeds)

    def __repr__(self) -> str:
        return f"GraphRegion({self.graph.name!r}, nodes={len(self.graph.nodes)})"


def _as_region(value: Any, where: str) -> ControlRegion:
    if isinstance(value, ControlRegion):
        return value
    if isinstance(value, Graph):
        return GraphRegion(value)
    raise TypeError(
        f"{where} 必须是 Graph 或 ControlRegion，得到 {type(value).__name__}"
    )


def _check_scalar_predicate(region: ControlRegion, where: str) -> None:
    outputs = region.outputs
    if len(outputs) != 1:
        names = [name for name, _ in outputs]
        raise ValueError(
            f"{where} 必须恰好有一个输出作为谓词，得到 {len(outputs)} 个: {names}"
        )
    name, type_ = outputs[0]
    if type_.rank != 0:
        raise ValueError(
            f"{where} 的谓词 %{name} 必须是 0 维标量，得到形状 {type_.shape}；"
            "分支选择需要单一真值，元素张量无法定义走哪一支"
        )
    if type_.dtype not in (DType.BOOL, DType.F16, DType.BF16, DType.F32, DType.F64,
                           DType.I8, DType.I32):
        raise TypeError(f"{where} 的谓词 %{name} 不能是 {type_.dtype.value}")


def _truth(region: ControlRegion, produced: Mapping[str, np.ndarray], where: str) -> bool:
    name = region.outputs[0][0]
    if name not in produced:
        raise ControlFlowError(f"{where} 的输出 %{name} 没有出现在执行结果里")
    value = np.asarray(produced[name])
    if value.size != 1:
        raise ControlFlowError(
            f"{where} 的谓词 %{name} 运行期形状为 {value.shape}，"
            "与 IR 声明的 0 维标量不一致"
        )
    return bool(value.reshape(()).item())


def _merge_inputs(regions: Sequence[Tuple[str, ControlRegion]]) -> Tuple[Tuple[str, TensorType], ...]:
    """合并多个区域的输入，同名必须同类型；顺序按首次出现。"""
    merged: List[Tuple[str, TensorType]] = []
    seen: Dict[str, TensorType] = {}
    for where, region in regions:
        for name, type_ in region.inputs:
            previous = seen.get(name)
            if previous is None:
                seen[name] = type_
                merged.append((name, type_))
            elif previous != type_:
                raise ValueError(
                    f"输入 %{name} 在 {where} 里声明为 {type_}，"
                    f"与先前的 {previous} 冲突：同名输入必须同类型"
                )
    return tuple(merged)


class Conditional(ControlRegion):
    """按数据选择执行哪一张子图。

    ``then`` / ``otherwise`` 可以是 ``Graph``，也可以是另一个控制区域
    （例如嵌套的 ``Conditional``），两者都只暴露"输入名/类型"与"输出名/类型"。

    分支必须暴露**同名、同序、同类型**的输出：控制流的汇合点没有第二套类型
    系统，靠"两边的 y 是同一个 y"来保证图外看到的接口是确定的。
    """

    def __init__(
        self,
        predicate: Any,
        then: Any,
        otherwise: Any,
        *,
        name: str = "cond",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"name 必须是非空字符串，得到 {name!r}")
        self.name = name
        self.predicate = _as_region(predicate, "predicate")
        self.then = _as_region(then, "then")
        self.otherwise = _as_region(otherwise, "otherwise")

        _check_scalar_predicate(self.predicate, f"Conditional({name}) 的 predicate")
        then_outputs = self.then.outputs
        else_outputs = self.otherwise.outputs
        then_names = [n for n, _ in then_outputs]
        else_names = [n for n, _ in else_outputs]
        if then_names != else_names:
            raise ValueError(
                f"Conditional({name}) 的两个分支输出不一致："
                f"then={then_names}，otherwise={else_names}；"
                "控制流汇合要求同名同序的输出"
            )
        for (then_name, then_type), (_, else_type) in zip(then_outputs, else_outputs):
            if then_type != else_type:
                raise ValueError(
                    f"Conditional({name}) 的输出 %{then_name} 类型不一致："
                    f"then={then_type}，otherwise={else_type}"
                )
        self._inputs = _merge_inputs(
            [("predicate", self.predicate), ("then", self.then), ("otherwise", self.otherwise)]
        )

    @property
    def inputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self._inputs

    @property
    def outputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self.then.outputs

    def run(self, feeds: Mapping[str, np.ndarray], backend=None) -> Dict[str, np.ndarray]:
        backend = _backend_or_default(backend)
        produced = self.predicate.run(feeds, backend)
        branch = self.then if _truth(self.predicate, produced, f"Conditional({self.name})") else self.otherwise
        return branch.run(feeds, backend)

    def __repr__(self) -> str:
        return f"Conditional({self.name!r}, outputs={[n for n, _ in self.outputs]})"


class WhileLoop(ControlRegion):
    """按数据重复执行循环体，循环携带量（``carried``）在轮次之间传递。

    契约比 ``Conditional`` 更细，因为循环有一个"状态必须封闭"的要求。
    ``carried`` 是 **状态名 → 循环体输出名** 的映射：状态名既是调用方给初值
    的名字，也是循环体的输入名（下一轮从这里进）；输出名是这一轮算出的新
    状态（不能与输入重名——图的名字空间是共享的，输入已占用该名字）。

    - 状态名必须是循环体的输入，输出名必须是循环体的输出，两者类型相同；
    - 条件区域只能看到"自由输入 + 当前状态"，且同名必须同类型；
    - ``max_steps`` 必须显式给出。不设默认值不是苛刻：默认值会让"这个循环
      本该收敛"的错误变成一次静默截断，而截断后的数值看起来完全合法。

    程序对外暴露的输出就是状态名（按声明顺序），不额外把自由输入透传出去
    ——那样调用方会以为它们可能被循环改写。
    """

    def __init__(
        self,
        condition: Any,
        body: Any,
        *,
        carried: Mapping[str, str],
        max_steps: int,
        name: str = "while",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"name 必须是非空字符串，得到 {name!r}")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps 必须是正整数，得到 {max_steps!r}")
        if not isinstance(carried, Mapping):
            raise TypeError(
                "carried 必须是「状态名 -> 循环体输出名」的映射，"
                f"得到 {type(carried).__name__}"
            )
        carried = dict(carried)
        if not carried:
            raise ValueError("carried 不能为空：没有循环携带量就无法表达迭代状态")
        for state, out in carried.items():
            if not isinstance(state, str) or not state:
                raise ValueError(f"carried 的状态名必须是非空字符串，得到 {state!r}")
            if not isinstance(out, str) or not out:
                raise ValueError(
                    f"carried[{state!r}] 的循环体输出名必须是非空字符串，得到 {out!r}"
                )
            if state == out:
                raise ValueError(
                    f"carried[{state!r}] 的输入名与输出名相同：图的名字空间是共享的，"
                    "输出别名不能复用输入名；请给新状态另一个名字"
                )

        self.name = name
        self.max_steps = max_steps
        self.carried = carried
        self.condition = _as_region(condition, "condition")
        self.body = _as_region(body, "body")

        _check_scalar_predicate(self.condition, f"WhileLoop({name}) 的 condition")

        body_inputs = dict(self.body.inputs)
        body_outputs = dict(self.body.outputs)
        for state, out in carried.items():
            if state not in body_inputs:
                raise ValueError(
                    f"WhileLoop({name}) 的循环携带量 %{state} 不是循环体的输入："
                    f"下一轮的状态没有入口。循环体输入为 {sorted(body_inputs)}"
                )
            if out not in body_outputs:
                raise ValueError(
                    f"WhileLoop({name}) 的循环携带量 %{state} 的出口 %{out} "
                    f"不是循环体的输出。循环体输出为 {sorted(body_outputs)}"
                )
            if body_inputs[state] != body_outputs[out]:
                raise ValueError(
                    f"WhileLoop({name}) 的循环携带量 %{state} 类型在体输入 "
                    f"{body_inputs[state]} 与体输出 %{out} {body_outputs[out]} 之间不一致"
                )

        cond_inputs = dict(self.condition.inputs)
        free: List[Tuple[str, TensorType]] = []
        for item, type_ in self.body.inputs:
            if item in carried:
                continue
            free.append((item, type_))
        for item, type_ in cond_inputs.items():
            previous = dict(free).get(item)
            if previous is None:
                if item in carried:
                    if body_inputs[item] != type_:
                        raise ValueError(
                            f"WhileLoop({name}) 的条件把 %{item} 声明为 {type_}，"
                            f"与循环体的 {body_inputs[item]} 不一致"
                        )
                    continue
                free.append((item, type_))
            elif previous != type_:
                raise ValueError(
                    f"WhileLoop({name}) 的输入 %{item} 在条件 {type_} 与循环体 "
                    f"{previous} 之间不一致"
                )
        self._free = tuple(free)
        self._outputs = tuple((state, body_inputs[state]) for state in carried)

    @property
    def inputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self._free

    @property
    def outputs(self) -> Tuple[Tuple[str, TensorType], ...]:
        return self._outputs

    def run(self, feeds: Mapping[str, np.ndarray], backend=None) -> Dict[str, np.ndarray]:
        backend = _backend_or_default(backend)
        missing = [state for state in self.carried if state not in feeds]
        if missing:
            raise ValueError(
                f"WhileLoop({self.name}) 缺少初始状态 {missing}；"
                "循环携带量必须由调用方给出初值"
            )
        state: Dict[str, np.ndarray] = {
            name: np.asarray(feeds[name]) for name in self.carried
        }
        for step in range(self.max_steps):
            scope = dict(feeds)
            scope.update(state)
            produced = self.condition.run(scope, backend)
            if not _truth(self.condition, produced, f"WhileLoop({self.name}) 的 condition"):
                return dict(state)
            produced = self.body.run(scope, backend)
            missing = [out for out in self.carried.values() if out not in produced]
            if missing:
                # 理论上构图期已经验过，但控制区域可以嵌套，运行期的实际输出
                # 仍要以真实结果为准：宁可在这里停住，也不要拿上一轮的状态
                # 继续跑，那等于把"没更新"伪装成"已收敛"。
                raise ControlFlowError(
                    f"WhileLoop({self.name}) 第 {step} 轮的循环体没有产生 {missing}"
                )
            state = {
                name: np.asarray(produced[out]) for name, out in self.carried.items()
            }
        raise ControlFlowError(
            f"WhileLoop({self.name}) 达到 max_steps={self.max_steps} 仍未退出条件；"
            "请调大 max_steps，或检查条件是否真的会变为假"
        )

    def __repr__(self) -> str:
        return f"WhileLoop({self.name!r}, carried={list(self.carried)}, max_steps={self.max_steps})"


# --------------------------------------------------------------------------
# 序列化：控制节点 = 若干子图 + 少量结构属性
# --------------------------------------------------------------------------


def to_spec(region: ControlRegion) -> Dict[str, Any]:
    """导出成纯 JSON 结构。子图走 ``ir.serialize.to_dict``，不重复定义格式。"""
    if isinstance(region, GraphRegion):
        return {"kind": "graph", "graph": to_dict(region.graph)}
    if isinstance(region, Conditional):
        return {
            "kind": "cond",
            "name": region.name,
            "predicate": to_spec(region.predicate),
            "then": to_spec(region.then),
            "otherwise": to_spec(region.otherwise),
        }
    if isinstance(region, WhileLoop):
        return {
            "kind": "while",
            "name": region.name,
            "condition": to_spec(region.condition),
            "body": to_spec(region.body),
            "carried": dict(region.carried),
            "max_steps": region.max_steps,
        }
    raise TypeError(f"不认识的区域类型: {type(region).__name__}")


def from_spec(spec: Mapping[str, Any]) -> ControlRegion:
    """从 JSON 结构还原控制流。构造期的全部契约检查都会重新跑一遍。"""
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec 必须是映射，得到 {type(spec).__name__}")
    kind = spec.get("kind")
    if kind == "graph":
        return GraphRegion(from_dict(spec["graph"]))
    if kind == "cond":
        return Conditional(
            from_spec(spec["predicate"]),
            from_spec(spec["then"]),
            from_spec(spec["otherwise"]),
            name=spec.get("name", "cond"),
        )
    if kind == "while":
        return WhileLoop(
            from_spec(spec["condition"]),
            from_spec(spec["body"]),
            carried=dict(spec["carried"]),
            max_steps=spec["max_steps"],
            name=spec.get("name", "while"),
        )
    raise ValueError(f"未知的控制流类型: {kind!r}；可用: graph / cond / while")
