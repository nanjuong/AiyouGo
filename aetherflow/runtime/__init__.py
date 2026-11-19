"""运行时：符号绑定与执行器协议。"""

from .executor import Executor, ShapeMismatchError, resolve_attrs
from .reference import ReferenceExecutor

__all__ = ["Executor", "ReferenceExecutor", "ShapeMismatchError", "resolve_attrs"]
