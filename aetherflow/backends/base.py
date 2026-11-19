"""后端插件契约。

这里只定义最小可用的一套接口，刻意保持"薄"：

- ``capabilities``：能力声明。后端必须如实列出自己承接的算子名。
  能力协商的意义在于，小硬件插件不必硬着头皮实现全部算子——
  前端可以根据这份声明决定拆分、降级或拒绝编译。
  返回普通集合表示"只承诺算子名"；返回 ``BackendCapabilities`` 可以附加
  dtype / axis / layout / 反向 / 捕获 / 确定性等形态明细（见 capabilities.py）。
- ``execute``：执行一个图。

后续会在此基础上拆成三套独立契约（编译期 CodeGen、运行期 Runtime、
离线调优 Autotune），因为它们生命周期不同，混在一起会让插件接口过重。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from ..ir.graph import Graph
from .capabilities import capability_gaps


class BackendCapabilityError(RuntimeError):
    """目标后端无法承接给定图。"""


@runtime_checkable
class Backend(Protocol):
    """后端插件必须实现的最小接口。"""

    name: str

    def capabilities(self) -> FrozenSet[str]:
        """返回该后端能原生执行的算子名集合。

        可以是 ``frozenset``，也可以是 ``BackendCapabilities``——后者仍是
        ``frozenset`` 子类，同时能说明"支持哪些形态"。
        """
        ...

    def execute(self, graph: Graph, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行图并返回输出。不支持的算子应抛 BackendCapabilityError。"""
        ...


def unsupported_ops(graph: Graph, capabilities: Iterable[str]) -> list:
    """返回图中目标后端不支持的算子（按出现顺序去重）。

    能力声明带形态明细时，dtype / axis / layout 不匹配也算不支持——
    "声明了算子名"不等于"任何形态都接得住"。只有名字的声明行为不变。
    """
    return [name for name, _ in capability_gaps(graph, capabilities)]
