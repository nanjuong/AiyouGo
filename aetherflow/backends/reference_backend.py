"""参考后端。

它不是"某个硬件"的模拟，而是 IR 的语义基准实现。
存在的意义有两个：
1. 在第一个真实硬件后端出现之前，就能验证 IR 的表达力与类型系统；
2. 任何新后端加入时，用它做数值对齐的参照。

它只承接 PORTABLE 算子；LIBRARY / VENDOR 节点会被明确拒绝——
这条拒绝本身就是在提醒：那段图已经不可能移植到别的硬件上了。

能力声明带有形态明细（``OpCapability``）：只写有依据的维度，其余留
``None``。``None`` 是"没有排除"，不是"逐项验证过"——参考执行器用 numpy
承载，对位宽与访存顺序本来就没有承诺（BF16 借用 f32 承载）。
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from ..ir.graph import Graph
from ..ir.ops import OPS, Portability
from ..ir.verifier import require
from ..runtime.reference import ReferenceExecutor
from .base import BackendCapabilityError, unsupported_ops
from .capabilities import BackendCapabilities, OpCapability


class ReferenceBackend:
    name = "reference"

    def __init__(self) -> None:
        self._executor = ReferenceExecutor()
        self._capabilities: Optional[BackendCapabilities] = None
        self._capability_count = -1

    def capabilities(self) -> BackendCapabilities:
        """参考后端的能力声明。

        - ``backward``：算子表里有 VJP 才声明。没有规则的算子在反向图上会被
          拒绝，能力声明不能比这更宽。
        - ``capture=False``：这里只有逐节点 numpy 求值，没有图捕获路径；声明
          成 True 会让上层误以为它能进 CUDA 的捕获计划。
        - ``deterministic=True``：随机性全部来自显式 seed（如 dropout），同一
          输入同一 seed 的结果可复现。
        - dtype / axis / layout 不设限：参考语义只做数学对齐，不承诺位宽与
          访存布局。

        结果按算子表长度缓存：``register`` 只允许新增、不允许重名，所以长度
        变化就足以判定失效。
        """
        count = len(OPS)
        if self._capabilities is None or self._capability_count != count:
            details = {
                name: OpCapability(backward=op.vjp is not None, capture=False)
                for name, op in OPS.items()
                if op.portability is Portability.PORTABLE
            }
            self._capabilities = BackendCapabilities(details, details=details)
            self._capability_count = count
        return self._capabilities

    def execute(self, graph: Graph, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        caps = self.capabilities()
        missing = unsupported_ops(graph, caps)
        if missing:
            raise BackendCapabilityError(
                f"参考后端不支持这些算子: {', '.join(missing)}；"
                "它们可能绑定在特定硬件或外部库上，或者超出了已声明的 dtype/axis 范围"
            )
        require(graph, caps)
        return self._executor.run(graph, inputs)
