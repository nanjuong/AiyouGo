"""AetherFlow 统一前端 IR。

分层结构（当前只实现第一层，后两层后续补齐）：

    高层语义 IR   -> 本文作实现的 ir 包：算子语义 + 类型 + 图
    中层图优化 IR -> 融合 ✅、常量折叠 ✅、死代码消除 ✅（均在语义 IR 上）
    底层硬件 IR   -> 待实现：由后端插件内部产生，不进入公共契约
"""

from .control import (
    Conditional,
    ControlFlowError,
    ControlRegion,
    GraphRegion,
    WhileLoop,
    from_spec,
    to_spec,
)
from .grad import GradCtx, backward
from .second_order import hessian_vector_product
from .graph import Graph, Node, Value
from .liveness import ALIGNMENT, Footprint, MemoryBudget, plan_memory
from .ops import (
    OPS,
    UNREPLAYABLE_EFFECTS,
    Effect,
    NoGradientError,
    Op,
    Portability,
    get_op,
    register,
)
from .serialize import IRVersionError, dumps, from_dict, loads, to_dict
from .types import DType, TensorType, bf16, boolean, f16, f32, f64, i8, i32
from .verifier import Issue, VerifierError, require, verify, verify_portable_only
from .version import (
    IR_VERSION,
    check_version,
    migrate,
    migration_path,
    register_migration,
    registered_migrations,
    unregister_migration,
    version_text,
)

__all__ = [
    "Graph",
    "Node",
    "Value",
    "IR_VERSION",
    "IRVersionError",
    "check_version",
    "version_text",
    "migrate",
    "migration_path",
    "register_migration",
    "unregister_migration",
    "registered_migrations",
    "to_dict",
    "from_dict",
    "dumps",
    "loads",
    "Footprint",
    "MemoryBudget",
    "plan_memory",
    "ALIGNMENT",
    "Op",
    "OPS",
    "Effect",
    "UNREPLAYABLE_EFFECTS",
    "NoGradientError",
    "GradCtx",
    "backward",
    "hessian_vector_product",
    "ControlFlowError",
    "ControlRegion",
    "GraphRegion",
    "Conditional",
    "WhileLoop",
    "to_spec",
    "from_spec",
    "Portability",
    "get_op",
    "register",
    "DType",
    "TensorType",
    "f16",
    "bf16",
    "f32",
    "f64",
    "i8",
    "i32",
    "boolean",
    "Issue",
    "VerifierError",
    "verify",
    "verify_portable_only",
    "require",
]
