"""梯度处理：全局范数裁剪与梯度累积。

两者都发生在"反向图算完之后、优化器更新之前"这一段，所以放在同一个模块：

    grads -> clip_global_norm -> accumulator.add -> optimizer.step

**顺序不能反。** 先累积再裁剪，得到的是"整个累积窗口的总范数上限"；先裁剪再
累积，每个微批各自被裁到 ``max_norm``，窗口总范数最多是 ``N * max_norm``——
那不是同一个约束，而且窗口越长越松，等于放大 batch 时悄悄放松了正则。

为什么裁剪要按**全局**范数而不是逐张量：逐张量裁会把方向也改掉（各分量被
不同程度的缩放），而全局范数只缩整体尺度，梯度方向保持不变。裁剪的目的是
防止单步走太远，不是改变下降方向。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ._arrays import array_module, host_scalar, to_host

__all__ = ["global_norm", "clip_global_norm", "GradientAccumulator"]


def global_norm(grads: Mapping[str, np.ndarray]) -> float:
    """所有梯度拼成一个大向量后的 L2 范数。

    平方和在 float64 上累加：f16 的梯度平方会直接下溢成 0，
    f32 在参数很多时也会丢掉小项。

    设备态下平方和在显存里累加，最后只把那个标量搬回主机——逐张量回主机
    等于把梯度全拷一遍，裁剪就成了设备态训练的短路点。
    """
    if not grads:
        raise ValueError("global_norm 需要至少一个梯度")
    total = None
    for grad in grads.values():
        xp = array_module(grad)
        arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
        part = xp.sum(arr.astype(xp.float64, copy=False) ** 2)
        if total is None:
            total = part
        elif array_module(total) is xp:
            total = total + part
        else:  # 主机与设备混着传：退回主机侧累加（罕见，但要能算对）
            total = to_host(total) + to_host(part)
    return math.sqrt(host_scalar(total))


def clip_global_norm(
    grads: Mapping[str, np.ndarray],
    max_norm: float,
    eps: float = 1e-6,
) -> Tuple[Dict[str, np.ndarray], float]:
    """按全局范数裁剪，返回 ``(裁剪后的梯度副本, 裁剪前的范数)``。

    返回裁剪前的范数，是为了让调用方能把它记进日志：训练中途范数突然飙升，
    是"即将发散"最早可见的信号，比等损失变成 nan 早得多。

    非有限范数直接报错而不是替换成 ``max_norm``：梯度里出现 inf/nan 说明前向
    或反向已经坏了，把它裁成一个"看起来正常"的值只会把故障推迟一步暴露。
    f16 训练里的溢出由 ``GradScaler`` 负责跳过，与这里是两条互不替代的路径。
    """
    if max_norm <= 0.0:
        raise ValueError(f"max_norm 必须为正，得到 {max_norm}")
    total = global_norm(grads)
    if not math.isfinite(total):
        raise ValueError(
            f"梯度全局范数为 {total}，说明梯度里已经含 inf/nan；"
            "请检查前向是否溢出，而不是裁剪掉这个信号"
        )

    # 与 PyTorch 同一条公式：不裁时 scale >= 1，直接返回副本，
    # 避免"乘了一个略小于 1 的系数"这种看不见的扰动。
    scale = max_norm / (total + eps)
    if scale >= 1.0:
        return {
            name: array_module(grad).array(grad, copy=True)
            for name, grad in grads.items()
        }, total

    clipped: Dict[str, np.ndarray] = {}
    for name, grad in grads.items():
        xp = array_module(grad)
        arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
        clipped[name] = (arr * scale).astype(arr.dtype, copy=False)
    return clipped, total


class GradientAccumulator:
    """把多个微批的梯度累加成一个等效大批次的梯度。

    典型用法::

        if acc.add(grads):
            clipped, norm = clip_global_norm(acc.flush(), max_norm)
            opt.step(clipped)

    三条约束：

    1. **内部用 float32 缓冲。** f16 梯度累加几次就会丢精度，而累积的目的
       恰恰是"用等效更大的 batch"，精度掉了就失去了意义。
    2. **flush 除以权重之和，不是微批数。** 默认每个微批权重为 1，因此
       "除以微批数"是它的特例；语言模型这类按**有效 token** 归一的损失，
       把 ``weight`` 传成这一批的有效 token 数，窗口平均就精确等于"按总有效
       token 求均值"——变长与 padding 比例不同也不会把某个微批过度加权。
    3. **最后一个窗口不足 ``steps`` 个微批时按实际权重归一。** 除以
       ``steps`` 会把梯度整体缩小，参数少走一步——这种偏差在损失曲线上
       完全看不出来。
    4. **满了必须先 flush。** 窗口满后再 ``add`` 直接报错，而不是自动清空：
       自动清空会把上一次的累积悄悄丢掉，训练看起来在跑、实际少更新了一次。
    """

    def __init__(self, steps: int, dtype=np.float32) -> None:
        if int(steps) < 1:
            raise ValueError(f"累积步数必须 >= 1，得到 {steps}")
        self.steps = int(steps)
        self.dtype = np.dtype(dtype)
        self._sum: Dict[str, np.ndarray] = {}
        self._count = 0
        self._weight = 0.0

    @property
    def pending(self) -> int:
        """当前窗口里已累积的微批数。"""
        return self._count

    @property
    def weight(self) -> float:
        """当前窗口里已累积的权重之和（默认等于微批数）。"""
        return self._weight

    @property
    def full(self) -> bool:
        return self._count >= self.steps

    def add(self, grads: Mapping[str, np.ndarray], weight: float = 1.0) -> bool:
        """加入一个微批的梯度，返回窗口是否已满。

        ``weight`` 是这个微批在窗口平均里的份量。默认 1（逐个微批等权）；
        按 token 归一的损失应传"这一批的有效 token 数"，窗口平均才等于
        按总有效 token 的均值。非有限或非正的权重直接拒绝：它会让整窗的
        分母出错，而错误只在参数更新后才看得到。
        """
        if self.full:
            raise RuntimeError(
                f"累积窗口已满（{self._count}/{self.steps}），请先 flush()；"
                "自动清空会丢掉已累积的梯度"
            )
        if not grads:
            raise ValueError("梯度不能为空")
        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            raise ValueError(f"累积权重必须是有限正数，得到 {weight!r}")

        if not self._sum:
            for name, grad in grads.items():
                # 按第一个微批所在的设备分配：设备态下累加缓冲也留在显存
                self._sum[name] = array_module(grad).array(
                    grad, dtype=self.dtype, copy=True
                )
            if w != 1.0:
                for name in self._sum:
                    self._sum[name] *= w
        else:
            if set(grads) != set(self._sum):
                missing = sorted(set(self._sum) - set(grads))
                extra = sorted(set(grads) - set(self._sum))
                raise KeyError(
                    f"这个窗口里的梯度集合与第一个微批不一致：缺 {missing}，多 {extra}"
                )
            for name, grad in grads.items():
                xp = array_module(self._sum[name])
                arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
                if tuple(arr.shape) != tuple(self._sum[name].shape):
                    raise ValueError(
                        f"参数 {name} 的梯度形状 {arr.shape} 与本窗口首个微批 "
                        f"{self._sum[name].shape} 不一致"
                    )
                self._sum[name] += arr if w == 1.0 else arr * w

        self._count += 1
        self._weight += w
        return self.full

    def flush(self) -> Optional[Dict[str, np.ndarray]]:
        """取出窗口的加权平均梯度并清空；窗口为空时返回 ``None``。

        返回 float32（累积缓冲的 dtype）；优化器会在 ``step`` 里按参数 dtype
        转换，这里不做提前降精度。
        """
        if self._count == 0:
            return None
        total = float(self._weight)
        out = {name: arr / total for name, arr in self._sum.items()}
        self._sum.clear()
        self._count = 0
        self._weight = 0.0
        return out

    def discard(self) -> int:
        """丢掉当前窗口并返回被丢掉的微批数。

        AMP 溢出时用：被 inf/nan 污染的微批留在窗口里，后面几批再"平均"也
        只是把坏值稀释掉，梯度依旧是坏的——整个窗口必须作废，不能只丢出问题
        的那一批（单批的贡献无法从累加和里减掉）。
        """
        count = self._count
        self._sum.clear()
        self._count = 0
        self._weight = 0.0
        return count

    def state_dict(self) -> Dict[str, object]:
        """窗口的完整状态：窗口长度、已累积微批数与累加和（主机副本）。

        存档必须带上这三样。只存"已更新到的参数"会让续训从窗口中途重新
        开始：已经算过的几个微批的梯度直接消失，等效 batch 悄悄变小，损失
        曲线上只能看到一次轻微的跳变。
        """
        return {
            "steps": self.steps,
            "count": self._count,
            "weight": self._weight,
            "sum": {
                name: np.array(to_host(arr), copy=True)
                for name, arr in self._sum.items()
            },
        }

    def load_state_dict(
        self, state: Mapping[str, object], *, like: Mapping[str, object]
    ) -> None:
        """恢复窗口；``like`` 是参数表，用来决定累加和落在哪个设备、什么 dtype。

        只接受本类自己产出的形状：``steps`` 必须与当前窗口长度一致（换窗口
        长度等于换训练配置，静默接受会让"等效 batch"与预期不同），累加和
        必须覆盖全部参数且形状一致。
        """
        steps = int(state.get("steps", self.steps))
        if steps != self.steps:
            raise ValueError(
                f"存档里的累积窗口长度 {steps} 与当前 {self.steps} 不一致；"
                "换窗口长度必须从头开始，不能接着旧窗口"
            )
        count = int(state.get("count", 0))
        if not 0 <= count <= self.steps:
            raise ValueError(
                f"存档里的已累积微批数 {count} 不在 [0, {self.steps}]"
            )
        sums = dict(state.get("sum") or {})
        weight = float(state.get("weight", float(count)))
        if count == 0:
            if sums:
                raise ValueError("已累积微批数为 0，却带了梯度累加和")
            self._sum.clear()
            self._count = 0
            self._weight = 0.0
            return
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"存档里的累积权重 {weight!r} 不是有限正数")
        if set(sums) != set(like):
            raise KeyError(
                "存档里的梯度累加和与当前参数集合不一致：缺少 "
                f"{sorted(set(like) - set(sums))}，多出 {sorted(set(sums) - set(like))}"
            )
        prepared: Dict[str, np.ndarray] = {}
        for name, arr in sums.items():
            shape = np.shape(like[name])
            if np.shape(arr) != shape:
                raise ValueError(
                    f"存档里 {name} 的梯度累加和形状 {np.shape(arr)} 与参数 {shape} 不一致"
                )
            prepared[name] = array_module(like[name]).asarray(arr, dtype=self.dtype)
        self._sum = prepared
        self._count = count
        self._weight = weight

    def __repr__(self) -> str:
        return f"GradientAccumulator({self._count}/{self.steps})"
