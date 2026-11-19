"""训练侧压缩模块。

对外导出：
    from aetherflow.train import (
        compress_for_training, low_rank_factors, quantize_int8,
        reconstruct_dense_weights, memory_report, assert_dense_free,
        TokenCorpus, compress_tokens, decompress_tokens, peak_bytes,
        Adam, SGD, ParameterStore, clip_global_norm, Trainer,
    )
"""

from .compress import (
    CompressedModel,
    assert_dense_free,
    compress_for_training,
    low_rank_factors,
    memory_report,
    quantize_int8,
    reconstruct_dense_weights,
)
from .corpus import TokenCorpus, compress_tokens, decompress_tokens, peak_bytes
from .grads import clip_global_norm, global_norm
from .mini_trainer import MiniTrainer
from .optim import SGD, Adam, Optimizer
from .params import ParameterStore

# demo/test 里写的是 Trainer；本包提供的是极简版（见 mini_trainer.py 的
# 存在理由说明），以 Trainer 之名导出以保持入口脚本可跑。
Trainer = MiniTrainer

__all__ = [
    "CompressedModel",
    "compress_for_training",
    "low_rank_factors",
    "quantize_int8",
    "reconstruct_dense_weights",
    "memory_report",
    "assert_dense_free",
    "TokenCorpus",
    "compress_tokens",
    "decompress_tokens",
    "peak_bytes",
    "clip_global_norm",
    "global_norm",
    "Adam",
    "SGD",
    "Optimizer",
    "ParameterStore",
    "MiniTrainer",
    "Trainer",
]
