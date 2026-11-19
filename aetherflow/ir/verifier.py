"""IR 合法性校验。

verifier 承担两个职责：

1. 结构完整性：拓扑顺序、类型是否可复推、输出是否声明。
2. 可移植性策略：给定目标后端拥有的算子集合，指出哪些节点无法承接。
   这一项是防止 IR 悄悄退化成"单一后端调度器"的强制闸门——
   只有第二个后端接入时才发现硬件假设泄漏，代价太高。

``capabilities`` 可以是普通算子名集合，也可以是带形态明细的
``BackendCapabilities``（这里是鸭子类型，只认 ``explain_node`` 方法，避免
ir 反过来依赖 backends 包）：后者会进一步按节点的 dtype / axis / layout
判定，把"声明了算子名、但这一形态接不住"在构图期就指出来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .graph import Graph
from .ops import Portability
from .version import IRVersionError, check_version

ERROR = "error"
WARNING = "warning"


@dataclass
class Issue:
    level: str
    message: str
    node: Optional[str] = None

    def __str__(self) -> str:
        loc = f" @{self.node}" if self.node else ""
        return f"[{self.level}]{loc} {self.message}"


class VerifierError(RuntimeError):
    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = list(issues)
        super().__init__("\n".join(str(i) for i in self.issues))


def verify(graph: Graph, capabilities: Optional[Iterable[str]] = None) -> List[Issue]:
    """检查图是否合法；若给出 capabilities，同时检查目标后端能否承接。"""
    issues: List[Issue] = []

    try:
        check_version(graph.ir_version)
    except IRVersionError as exc:
        # 版本不认识就不该继续往下校验：后面的规则可能已经变了
        issues.append(Issue(ERROR, str(exc)))
        return issues

    if not graph.outputs:
        issues.append(Issue(ERROR, "图没有声明任何输出"))

    seen = {v.id for v in graph.inputs}
    for node in graph.nodes:
        for arg in node.inputs:
            if arg.id not in seen:
                issues.append(Issue(ERROR, f"输入 %{arg.name} 在该节点之前未被定义", str(node)))
        try:
            expected = node.op.infer(tuple(a.type for a in node.inputs), node.attrs)
        except Exception as exc:  # 类型推导失败本身就是 IR 错误
            issues.append(Issue(ERROR, f"类型推导失败: {exc}", str(node)))
            expected = None
        if expected is not None and expected != node.output.type:
            issues.append(
                Issue(
                    ERROR,
                    f"类型不一致：声明为 {node.output.type}，推导为 {expected}",
                    str(node),
                )
            )
        seen.add(node.output.id)

    if capabilities is not None:
        # 名字集合与形态判定分开：普通集合只走第一层，带 explain_node 的
        # 能力对象才会进入 dtype / axis / layout 的细判。
        names = set(capabilities)
        explain_node = getattr(capabilities, "explain_node", None)
        if not callable(explain_node):
            explain_node = None
        for node in graph.nodes:
            if node.op.name not in names:
                issues.append(
                    Issue(
                        ERROR,
                        f"目标后端不支持算子 {node.op.name}（可移植性 {node.op.portability.value}）",
                        str(node),
                    )
                )
                continue
            if explain_node is not None:
                reason = explain_node(node)
                if reason:
                    issues.append(
                        Issue(
                            ERROR,
                            f"目标后端的声明不支持该节点的形态：{reason}",
                            str(node),
                        )
                    )

    return issues


def verify_portable_only(graph: Graph) -> List[Issue]:
    """只允许 PORTABLE 节点。用于检查 IR 是否可以下发给任意硬件后端。"""
    issues: List[Issue] = []
    for node in graph.nodes:
        if node.op.portability is not Portability.PORTABLE:
            issues.append(
                Issue(
                    ERROR,
                    f"节点不可移植（{node.op.portability.value}），不能出现在跨后端共享路径中",
                    str(node),
                )
            )
    return issues


def require(graph: Graph, capabilities: Optional[Iterable[str]] = None) -> None:
    issues = verify(graph, capabilities)
    errors = [i for i in issues if i.level == ERROR]
    if errors:
        raise VerifierError(errors)
