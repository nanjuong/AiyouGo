"""中层图优化 IR（vendor 来自 AetherFlow）。

对外导出：
    from aetherflow.passes import ParamCompression, plan_param_compression
"""

from .param_compress import ParamCompression, plan_param_compression, MODES

__all__ = ["ParamCompression", "plan_param_compression", "MODES"]
