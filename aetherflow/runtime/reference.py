"""参考语义执行器。

这是全局唯一真值：任何后端（CUDA 或其它硬件）的输出都必须与这里等价。
它刻意不依赖任何加速库、不做算子融合、不共享中间缓冲——
它只回答一个问题：这个 IR 在数学上应该算出什么。

符号绑定、运行期形状校验、输出组装都在 ``Executor`` 里，
本文件只负责"一个节点 = 调用一次参考语义函数"。
"""

from __future__ import annotations

from ..ir.graph import Node
from .executor import Executor, ShapeMismatchError, resolve_attrs

__all__ = ["ReferenceExecutor", "ShapeMismatchError", "resolve_attrs"]


class ReferenceExecutor(Executor):
    name = "reference"

    def invoke(self, node: Node, args: list, attrs: dict):
        return node.op.reference(args, attrs)


# 旧名字兼容：早期版本里这个函数叫 _resolve_attrs
_resolve_attrs = resolve_attrs
