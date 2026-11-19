"""训练层共用的数组工具：按数组所在设备挑模块。

优化器、梯度处理、EMA、存档四处都要回答同一个问题——"这个数组在主机还是
显存"。放在一处，而不是四处各写一遍 ``if cupy``：任何一处漏判，要么变成
隐式 D2H（设备态训练的收益当场消失），要么直接抛 ``TypeError``
（cupy 拒绝隐式回主机）。

判断依据是类所在的顶层包，而不是 ``import cupy`` 之后 ``isinstance``：
主机环境不一定装了 cupy，导不进来就不该有分支。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["array_module", "to_host", "host_scalar"]


def array_module(arr) -> Any:
    """``arr`` 所属的数组模块：numpy、cupy 或 HIP 运行时。

    HIP 的容器（``backends/hip/runtime.py::DeviceArray``）不是 numpy 子类，
    但它提供了优化器与梯度工具需要的那一面（``ndarray`` / ``asarray`` /
    ``asnumpy`` / ``zeros_like`` / ``sqrt`` 与逐元素运算符）。把运行时模块
    本身当作 ``xp`` 交出去，训练层的分派代码一行都不用改。
    """
    module = type(arr).__module__
    root = module.split(".", 1)[0]
    if root == "cupy":
        import cupy

        return cupy
    if module.startswith("aetherflow.backends.hip"):
        from ..backends.hip import runtime

        return runtime
    return np


def to_host(arr) -> np.ndarray:
    """转成主机 numpy 数组（设备数组会同步拷回）。"""
    xp = array_module(arr)
    return arr if xp is np else xp.asnumpy(arr)


def host_scalar(value) -> float:
    """把可能是 0 维设备数组的值收成一个 Python float（一次 D2H）。"""
    xp = array_module(value)
    return float(value if xp is np else xp.asnumpy(value))
