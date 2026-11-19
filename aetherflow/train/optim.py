"""优化器：SGD（含动量/Nesterov）与 Adam。

三条刻意的约束：

1. **优化器只接收 ``dict[str, ndarray]``，不接收图、不接收 ParameterStore。**
   "梯度从哪来"是自动微分的事，"参数怎么走一步"是优化器的事，混在一起
   会让"换个优化器"变成改训练流程。
2. **原地更新。** 参数数组被 ``p -= ...`` 直接改写，不返回新字典。
   否则调用方忘了替换引用，训练会看起来在跑、实际一步没动。
3. **缺梯度就报错，不补零。** 某个参数没有被损失依赖时，``grads`` 里没有它；
   这时静默跳过会掩盖"这一层根本没参与前向"的真实缺陷。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping

import numpy as np

from ._arrays import array_module, to_host

__all__ = ["Optimizer", "SGD", "Adam"]


def _host_copy(arr) -> np.ndarray:
    """槽位数组的主机副本：存档里放的永远是 numpy。"""
    return np.array(to_host(arr), copy=True)


class Optimizer:
    """优化器基类：负责参数表校验与步数计数。"""

    def __init__(self, params: MutableMapping[str, np.ndarray], lr: float) -> None:
        if lr <= 0.0:
            raise ValueError(f"学习率必须为正，得到 {lr}")
        self.params = params
        self.lr = float(lr)
        # 构造时的 lr 是"配置"，调度器每步写 ``self.lr`` 的是"轨迹"。
        # 存档超参必须比配置：调度器把 lr 移到别处再加载旧存档，配置没变，
        # 不该被当成超参不一致拒绝（外部审计 F06）。
        self._base_lr = float(lr)
        self.step_count = 0

    def rebind(self, params: MutableMapping[str, np.ndarray]) -> None:
        """换成另一张参数表，名字与形状必须与当前完全一致。

        设备态训练需要它：参数从主机数组换成显存数组之后，"优化器改的是哪
        一份数据"必须跟着换。名字集合不许变——动量与一阶/二阶矩都按名字
        索引，悄悄少一个名字会让续训的轨迹与预期不同。
        """
        if set(params) != set(self.params):
            raise KeyError(
                "新参数表的名字与当前不一致："
                f"多 {sorted(set(params) - set(self.params))}，"
                f"少 {sorted(set(self.params) - set(params))}"
            )
        for name, arr in params.items():
            if tuple(arr.shape) != tuple(self.params[name].shape):
                raise ValueError(
                    f"参数 {name} 的形状从 {tuple(self.params[name].shape)} "
                    f"变成 {tuple(arr.shape)}；换表不能改形状"
                )
        self.params = params

    # -- 子类实现 -----------------------------------------------------

    def _update(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        raise NotImplementedError

    # -- 对外接口 -----------------------------------------------------

    def step(self, grads: Mapping[str, np.ndarray]) -> None:
        """用一批梯度走一步。``grads`` 必须覆盖全部参数。

        先对**全部**参数做形状 / dtype 校验，再统一更新。逐个"校验一个更新
        一个"会在中途失败时留下半步状态：前面的参数已经走了一步，步数却还是
        旧的，续训与调试都会看到不一致的参数。
        """
        missing = [n for n in self.params if n not in grads]
        if missing:
            raise KeyError(
                f"这些参数没有梯度: {missing}。"
                "通常意味着损失不依赖它们；请检查构图，而不是让优化器跳过一个不动的参数"
            )
        unknown = [n for n in grads if n not in self.params]
        if unknown:
            raise KeyError(f"梯度里出现了不认识的参数: {unknown}")

        prepared = {
            name: self._coerce_grad(name, param, grads[name])
            for name, param in self.params.items()
        }
        for name, param in self.params.items():
            self._update(name, param, prepared[name])
        self.step_count += 1

    @staticmethod
    def _coerce_grad(name: str, param: np.ndarray, grad) -> np.ndarray:
        """把梯度对齐到参数所在的设备与 dtype。

        主机态下这就是原来的 ``np.asarray``；设备态下梯度已经是显存数组，
        原样返回（既不做 D2H 也不做拷贝）。
        """
        xp = array_module(param)
        arr = grad if isinstance(grad, xp.ndarray) else xp.asarray(grad)
        if arr.dtype != param.dtype:
            arr = arr.astype(param.dtype, copy=False)
        if tuple(arr.shape) != tuple(param.shape):
            raise ValueError(
                f"参数 {name} 的梯度形状 {tuple(arr.shape)} 与参数形状 {tuple(param.shape)} 不一致"
            )
        return arr

    # -- 状态 ---------------------------------------------------------

    def hyperparams(self) -> dict:
        """影响更新语义的超参。存档把它与槽位绑在一起，加载时逐项校验。

        只存权重与槽位是不够的：``lr`` 从 0.01 改成 0.001 后加载旧存档，
        参数、动量、步数全都"对得上"，续训轨迹却与未中断相差一个量级。

        存的是**构造时**的 lr（``_base_lr``），不是当前 ``self.lr``：调度器
        每步会改写 ``self.lr``，把"轨迹上的 lr"当配置存下来，会让"同一配置、
        带调度器"的存档在重建 Trainer 时被误判成超参不一致（外部审计 F06）。
        """
        return {"lr": float(self._base_lr)}

    def _slots(self) -> Mapping[str, MutableMapping[str, np.ndarray]]:
        """按名字暴露内部槽位（动量 / 一阶矩 / 二阶矩），供存档使用。

        返回的是内部字典本身，不是副本：``state_dict`` 负责拷贝，
        ``load_state_dict`` 负责原地覆盖，两边都不会各自维护第二份真相。
        """
        return {}

    def state_dict(self) -> dict:
        """优化器状态：步数 + 各槽位数组的副本（**不含参数本身**）。

        参数与优化器状态分开存：参数属于模型，槽位属于这次训练过程。
        把两者混在一个对象里，会导致"换优化器就得重存权重"。
        """
        return {
            "step": self.step_count,
            "hyper": self.hyperparams(),
            "slots": {
                slot: {name: _host_copy(arr) for name, arr in arrays.items()}
                for slot, arrays in self._slots().items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """恢复步数与槽位。槽位名、参数名、形状任何一处不匹配都直接报错。

        不"能恢复多少恢复多少"：少一个动量数组，续训的轨迹就与没存过不同，
        而这种偏差在损失曲线上几乎看不出来。

        恢复分两步：**先全部校验、构造好待提交的槽位，再一次性换上去**。
        中途失败不会留下"步数已经变成 100、动量还是旧的"这种自相矛盾的状态。
        训练过的存档必须带上全部槽位、且每个槽位覆盖全部参数——只检查
        "存档里出现的名字"，会让 ``{'step': 100, 'slots': {}}`` 这种残缺
        存档把旧槽位原封不动地留下来。
        """
        slots = dict(state.get("slots") or {})
        if "hyper" in state:
            saved = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in dict(state.get("hyper") or {}).items()
            }
            current = self.hyperparams()
            bad = [
                f"{key}: 存档 {saved.get(key)!r} / 当前 {current.get(key)!r}"
                for key in sorted(set(saved) | set(current))
                if saved.get(key) != current.get(key)
            ]
            if bad:
                raise ValueError(
                    "存档里的优化器超参与当前不一致（"
                    + "；".join(bad)
                    + "）。超参决定更新语义，换了超参不能接着旧状态续训"
                )
        known = self._slots()
        unknown = set(slots) - set(known)
        if unknown:
            raise KeyError(
                f"存档里的槽位 {sorted(unknown)} 当前优化器没有；"
                f"可用: {sorted(known)}。通常是优化器类型或超参换了"
            )
        step = int(state.get("step", 0))
        if slots or step > 0:
            missing_slots = set(known) - set(slots)
            if missing_slots:
                raise KeyError(
                    f"存档缺少槽位 {sorted(missing_slots)}（步数 {step}）；"
                    "训练过的优化器必须恢复全部槽位，否则续训轨迹与未中断不同"
                )
            for slot in known:
                missing_names = set(self.params) - set(slots.get(slot, {}))
                if missing_names:
                    raise KeyError(
                        f"槽位 {slot} 缺少参数 {sorted(missing_names)}；"
                        "残缺槽位会让这些参数从零动量重新开始"
                    )

        prepared: Dict[str, Dict[str, np.ndarray]] = {}
        for slot, arrays in slots.items():
            bucket: Dict[str, np.ndarray] = {}
            for name, arr in arrays.items():
                if name not in self.params:
                    raise KeyError(f"槽位 {slot} 里的参数 {name!r} 不在当前参数表中")
                shape = self.params[name].shape
                if np.shape(arr) != shape:
                    raise ValueError(
                        f"槽位 {slot}/{name} 的形状 {np.shape(arr)} 与参数形状 {shape} 不一致"
                    )
                bucket[name] = array_module(self.params[name]).array(
                    arr, dtype=self.params[name].dtype, copy=True
                )
            prepared[slot] = bucket

        for slot, bucket in prepared.items():
            known[slot].clear()
            known[slot].update(bucket)
        self.step_count = step

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr:g}, steps={self.step_count})"


class SGD(Optimizer):
    """随机梯度下降。

    ``momentum`` 为 0 时退化为最朴素的 SGD；``nesterov=True`` 时按
    ``g + mu * buf`` 而不是 ``buf`` 更新（注意这里 buf 已经包含了当前梯度，
    与 PyTorch 的写法一致）。``weight_decay`` 是 L2 惩罚，加在**梯度**上
    （而不是直接缩参数），所以它与动量/自适应学习率的交互是可解释的。
    """

    def __init__(
        self,
        params: MutableMapping[str, np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"momentum 必须落在 [0, 1)，得到 {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov 需要 momentum > 0")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay 不能为负，得到 {weight_decay}")
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)
        self._buf: Dict[str, np.ndarray] = {}

    def _slots(self) -> Mapping[str, MutableMapping[str, np.ndarray]]:
        # momentum=0 时没有状态：返回空表，而不是一堆永不使用的零数组
        return {"buf": self._buf} if self.momentum else {}

    def hyperparams(self) -> dict:
        return {
            **super().hyperparams(),
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": self.weight_decay,
        }

    def _update(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        g = grad
        if self.weight_decay:
            g = g + self.weight_decay * param
        if not self.momentum:
            param -= self.lr * g
            return
        buf = self._buf.get(name)
        if buf is None:
            # 按参数所在设备分配：设备态下槽位也必须留在显存
            buf = array_module(param).zeros_like(param)
            self._buf[name] = buf
        buf *= self.momentum
        buf += g
        direction = g + self.momentum * buf if self.nesterov else buf
        param -= self.lr * direction


class Adam(Optimizer):
    """Adam（含偏差修正）。

    偏差修正不是可选项：第一步 ``m`` 是 ``(1-beta1)*g``，不修正时更新量
    只有应有值的 1/10（默认 beta1=0.9），学习率实际上被悄悄缩小了一个量级。
    """

    def __init__(
        self,
        params: MutableMapping[str, np.ndarray],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas 必须落在 [0, 1)，得到 {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps 必须为正，得到 {eps}")
        # 半精度参数不能直接交给 Adam：``eps=1e-8`` 在 f16 里下溢成 0，
        # 梯度为零时 ``v=0``，更新量是 ``0 / (sqrt(0) + 0) = NaN``，一次
        # 零梯度就能把参数写成 NaN。f32 master 权重 + GradScaler 是这里的
        # 正确用法（GradScaler 只管反向的动态范围，管不了这个 eps）。
        for name, param in params.items():
            dt = np.dtype(param.dtype)
            if dt.kind == "f" and dt.itemsize < 4:
                raise TypeError(
                    f"参数 {name} 的 dtype 是 {dt}，Adam 不能直接更新半精度参数；"
                    "请用 f32 master 权重（更新后再转半精度存储），"
                    "反向的动态范围交给 GradScaler"
                )
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay 不能为负，得到 {weight_decay}")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}

    def _slots(self) -> Mapping[str, MutableMapping[str, np.ndarray]]:
        return {"m": self._m, "v": self._v}

    def hyperparams(self) -> dict:
        # 逐个标量展开：存档写的是 manifest 标量，列表要先自己拆开
        return {
            **super().hyperparams(),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def _update(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        xp = array_module(param)
        g = grad
        if self.weight_decay:
            g = g + self.weight_decay * param

        m = self._m.get(name)
        v = self._v.get(name)
        if m is None:
            m = xp.zeros_like(param)
            v = xp.zeros_like(param)
            self._m[name] = m
            self._v[name] = v

        m *= self.beta1
        m += (1.0 - self.beta1) * g
        v *= self.beta2
        v += (1.0 - self.beta2) * (g * g)

        t = self.step_count + 1
        m_hat = m / (1.0 - math.pow(self.beta1, t))
        v_hat = v / (1.0 - math.pow(self.beta2, t))
        param -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
