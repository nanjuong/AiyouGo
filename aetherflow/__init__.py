"""aiyougo compression — 模型与数据的轻量级压缩技术。

本项目包含 AetherFlow 训练框架中用于"不物化稠密 f32 权重"与"不驻留全量语料"
的两条压缩线：

1. 参数压缩（``aetherflow.train.compress`` / ``aetherflow.passes.param_compress``）：
   - low_rank：W(m,n) f32 → A(m,r) + B(r,n)，可训练量 (m+n)·r，优化器槽位同比例缩。
   - qlora：冻结的 int8 底 + 每列 scale + 低秩适配器，底只 1 字节/元素。
   - 三条判据：可训练量 / 驻留表示即真相 / assert_dense_free。
2. 数据压缩（``aetherflow.train.corpus``）：
   - compress_tokens / decompress_tokens：token 流 delta → zigzag → LEB128 无损编解码（~9×）。
   - TokenCorpus：只保存压缩字节，按批惰性解码，整个数据集的展开形态不驻留。

压缩 pass 作用在 IR 图上，因此本包自带完整 IR 层（``aetherflow.ir``）、
CPU 参考后端（``aetherflow.backends.ReferenceBackend``）与语言模型构图
（``aetherflow.models.lm``），入口脚本开箱即用。

依赖：仅 Python 标准库 + numpy。
"""

from .backends import ReferenceBackend
from .ir import backward, f32, f64, f16, bf16, i8, i32, Graph
from .runtime import ReferenceExecutor

from . import ir
from . import passes
from . import train

__all__ = [
    "Graph",
    "backward",
    "ReferenceBackend",
    "ReferenceExecutor",
    "f32",
    "f64",
    "f16",
    "bf16",
    "i8",
    "i32",
    "ir",
    "passes",
    "train",
]
