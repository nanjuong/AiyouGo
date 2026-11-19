"""后端抽象与 CPU 参考后端。

参考后端只承接 IR 中**无状态**的算子子集：带状态转移的模型算子
由参考实现直接拒绝，模型层负责展开。
"""

from .base import Backend, BackendCapabilityError, unsupported_ops
from .capabilities import BackendCapabilities, OpCapability, capability_gaps, to_capabilities
from .contracts import DeviceInfo, MemoryStats
from .reference_backend import ReferenceBackend

__all__ = [
    "Backend",
    "BackendCapabilityError",
    "unsupported_ops",
    "BackendCapabilities",
    "OpCapability",
    "capability_gaps",
    "to_capabilities",
    "DeviceInfo",
    "MemoryStats",
    "ReferenceBackend",
]
