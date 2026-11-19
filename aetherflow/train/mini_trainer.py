"""压缩路径用的极简训练循环：一步 = 前向 + 反向 + 裁剪 + 优化器。

为什么不 vendor 主项目的 ``train.loop.Trainer``：那个 Trainer 挂着
autocast / checkpoint / recompute / schedule / 梯度累积等整条训练栈，
对"压缩技术"这个包体是纯负债。本模块只保留 demo 与测试需要的四件事，
语义与主项目逐步一致（同样的 backward、同样的 clip、同样的 Adam）。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..backends.reference_backend import ReferenceBackend
from ..ir.graph import Graph
from ..ir.grad import backward
from .grads import clip_global_norm
from .optim import Optimizer
from .params import ParameterStore

__all__ = ["MiniTrainer"]


class MiniTrainer:
    """在参考后端上把一张可微图跑起来的最小执行器。

    参数表按名字绑定到 ``ParameterStore``：每步把**当前**参数值灌进
    feed，前向/反向共用同一份输入字典，这样 ``store.arrays()`` 与
    优化器更新的永远是同一批 numpy 数组（原地更新语义由优化器保证）。
    """

    def __init__(
        self,
        graph: Graph,
        store: ParameterStore,
        optimizer: Optimizer,
        *,
        output: str = "loss",
        clip: Optional[float] = None,
        executor: Any = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.optimizer = optimizer
        self.output = output
        self.clip = clip
        self.executor = executor if executor is not None else ReferenceBackend()
        self.params_list = [graph.find(n) for n in store.names]
        self.bwd = backward(graph, wrt=self.params_list)
        self._executed = 0

    def step(self, extra_feed: Mapping[str, Any]) -> float:
        """一次完整更新；``extra_feed`` 是除参数外的图输入（ids/targets/...）。"""
        feed: Dict[str, Any] = dict(extra_feed)
        feed.update(self.store.arrays())
        # 反向图的种子梯度：损失是标量，vjp 对 dY 线性，dloss=1 即取梯度本身。
        # 与主项目 Trainer 同义；显式传入时以传入为准（支持加权反传）。
        feed.setdefault("dloss", np.float32(1.0))
        out = self.executor.execute(self.graph, feed)
        loss = float(out[self.output])
        if not np.isfinite(loss):
            raise FloatingPointError(f"step {self._executed + 1}: 损失为 {loss}")
        grads_host = self.executor.execute(self.bwd, feed)
        grads = {n: grads_host["d" + n] for n in self.store.names}
        if self.clip is not None:
            grads, _ = clip_global_norm(grads, self.clip)
        self.optimizer.step(grads)
        self._executed += 1
        return loss
