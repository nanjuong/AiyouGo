"""中层 IR 优化框架。

三条硬约定
----------
1. **Pass 是函数式的**：输入一张图，输出一张新图，绝不原地修改。
   这样每个 pass 都可以单独测试，也能任意重排顺序。
2. **每个 pass 之后必须过 verifier**，否则立即失败。
   优化把图改坏是这类框架最常见的事故，宁可当场崩。
3. **正确性由差分测试保证**：同一输入下，优化前后输出必须数值等价。
   pass 的单测不能只看"节点少了"，必须看"结果没变"。

重写工具 ``GraphRewriter`` 承担所有"旧值 -> 新值"的搬运，
避免每个 pass 自己维护 id 映射。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional

from ..ir.graph import Graph, Node, Value
from ..ir.verifier import ERROR, VerifierError, verify


class Pass(ABC):
    """一个 IR 变换。"""

    name: str = "pass"

    @abstractmethod
    def run(self, graph: Graph) -> Graph:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class PassManager:
    """按顺序执行多个 pass，并在每个 pass 之后校验 IR 合法性。"""

    def __init__(self, passes: Iterable[Pass], check: bool = True) -> None:
        self.passes: List[Pass] = list(passes)
        self.check = check

    def run(self, graph: Graph) -> Graph:
        current = graph
        for p in self.passes:
            current = p.run(current)
            if self.check:
                errors = [i for i in verify(current) if i.level == ERROR]
                if errors:
                    raise VerifierError(errors)
        return current

    def __repr__(self) -> str:
        return f"PassManager({', '.join(p.name for p in self.passes)})"


class GraphRewriter:
    """把旧图逐节点搬运成新图，同时维护 old.id -> new.Value 的映射。"""

    def __init__(self, graph: Graph) -> None:
        self.old = graph
        self.new = Graph(graph.name)
        self.map: Dict[int, Value] = {}

    def copy_inputs(self, keep: Optional[Callable[[Value], bool]] = None) -> None:
        for v in self.old.inputs:
            if keep is None or keep(v):
                self.map[v.id] = self.new.input(v.name, v.type)

    def emit(self, node: Node) -> Value:
        args = [self.map[a.id] for a in node.inputs]
        out = self.new.add_node(node.op.name, args, node.attrs)
        if out.type != node.output.type:
            raise AssertionError(
                f"重写节点 {node.op.name} 时类型发生变化: {node.output.type} -> {out.type}"
            )
        self.map[node.output.id] = out
        return out

    def remap(self, node: Node, new_value: Value) -> None:
        """把一个被消除或合并的节点映射到已有的新值上。"""
        self.map[node.output.id] = new_value

    def finish(self) -> Graph:
        for v, name in zip(self.old.outputs, self.old.output_names):
            if v.id not in self.map:
                raise ValueError(f"图输出 %{v.name} 在重写后丢失")
            nv = self.map[v.id]
            self.new.output(nv, None if name == nv.name else name)
        return self.new
