"""可训练参数的容器与初始化。

为什么参数要单独有容器，而不是"一堆 numpy 数组"：

1. **参数是图输入，但语义上不是数据。** 训练循环每次要喂的是
   ``参数 + 这一批的 x``，两者混在一起最容易出现的 bug 是"忘了更新某个参数"
   或"把 x 也当成参数更新了"。容器把这件事显式化。
2. **初始化的默认值必须能跑通。** 全零初始化会让 LayerNorm 的 gamma 把梯度
   乘成零、让对称的隐层永远学不出差异。这里按名字与秩给出"不会自锁"的默认值。
3. **优化器只认 ``dict[str, ndarray]``。** 容器不绑定任何优化器接口，
   优化器也不认识容器——两者之间只传数组，避免出现"优化器要求某种参数对象"
   这种把训练流程写死的耦合。

命名约定（与 ``models`` 里的构件一致）：矩阵权重是 ``w*``（秩 >= 2），
偏置/截距以 ``b`` 结尾，归一化的缩放以 ``g`` 结尾。默认初始化据此选择。
"""

from __future__ import annotations

import zlib
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.graph import Graph
from ..ir.types import DType, TensorType

__all__ = ["init_array", "default_init", "ParameterStore"]


_FLOAT_DTYPES = (DType.F32, DType.F64)


def default_init(name: str, shape: Sequence[int]) -> str:
    """按名字与秩选默认初始化策略。

    秩 >= 2 视为权重 → Xavier；秩 1 且名字以 ``b``/``bias``/``beta`` 结尾
    视为偏置 → 零；其余秩 1 视为缩放（LayerNorm 的 gamma）→ 一。
    秩 0 视为标量偏置 → 零。
    """
    if len(shape) >= 2:
        return "xavier"
    if len(shape) == 1:
        low = name.lower()
        if low.endswith(("b", "bias", "beta")):
            return "zeros"
        return "ones"
    return "zeros"


def _fan_in_out(shape: Sequence[int]) -> Tuple[int, int]:
    """扇入/扇出。

    两种布局都要正确，否则初始化方差会差一个数量级：

    - 矩阵权重 ``(in, out)``：与 ``x @ W`` 的用法一致，扇入在 dim 0；
    - 卷积权重 ``(out, in, kh, kw)``：扇入在 dim 1，并乘上感受野。
    """
    shape = tuple(int(d) for d in shape)
    if len(shape) < 2:
        n = shape[0] if shape else 1
        return n, n
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = 1
    for d in shape[2:]:
        receptive *= d
    return shape[1] * receptive, shape[0] * receptive


def init_array(
    name: str,
    shape: Sequence[int],
    dtype: DType = DType.F32,
    init: Optional[str] = None,
    seed: int = 0,
    value: float = 0.0,
) -> np.ndarray:
    """按策略生成一个参数数组。

    ``seed`` 会与参数名的 CRC 混合，所以同一张图重复构造得到同一组初始化，
    而不同参数之间又不会共享同一串随机数（全图共用一个 rng 时，只要模型
    结构改一处，后面所有参数的初始值都会漂移，实验就无法复现了）。
    """
    strategy = init or default_init(name, shape)
    np_dtype = DType(dtype).np_dtype
    dims = tuple(int(d) for d in shape)
    rng = np.random.default_rng((seed + zlib.crc32(name.encode("utf-8"))) & 0xFFFFFFFF)

    if strategy == "zeros":
        return np.zeros(dims, dtype=np_dtype)
    if strategy == "ones":
        return np.ones(dims, dtype=np_dtype)
    if strategy == "constant":
        return np.full(dims, float(value), dtype=np_dtype)

    fan_in, fan_out = _fan_in_out(dims)
    if strategy == "xavier":
        limit = float(np.sqrt(6.0 / max(fan_in + fan_out, 1)))
        return rng.uniform(-limit, limit, dims).astype(np_dtype)
    if strategy == "he":
        std = float(np.sqrt(2.0 / max(fan_in, 1)))
        return (rng.standard_normal(dims) * std).astype(np_dtype)
    if strategy == "normal":
        std = float(value) if value else 0.02
        return (rng.standard_normal(dims) * std).astype(np_dtype)
    raise ValueError(
        f"未知初始化策略 {strategy!r}；可用: zeros / ones / constant / xavier / he / normal"
    )


class ParameterStore:
    """一组可训练参数，按名字索引。

    ``arrays`` 返回的是**内部字典的引用**：优化器靠它原地更新参数，
    这样"参数"与"数组"不会出现两份副本（副本一旦分叉，训练就是静默失效）。
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        if not arrays:
            raise ValueError("ParameterStore 不能为空：至少需要一个参数")
        self._arrays: Dict[str, np.ndarray] = {}
        for name, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"参数 {name!r} 必须是 numpy 数组，得到 {type(arr).__name__}")
            if not np.issubdtype(arr.dtype, np.floating):
                raise TypeError(f"参数 {name!r} 必须是浮点类型，得到 {arr.dtype}")
            self._arrays[name] = arr

    # -- 构造 ---------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        exclude: Iterable[str] = (),
        init: Optional[Mapping[str, str]] = None,
        seed: int = 0,
    ) -> "ParameterStore":
        """从图输入里挑出参数并初始化。

        ``exclude`` 是**数据输入**的名字（通常是 ``"x"``、``"ids"``）。
        默认不做任何猜测：少写一个 exclude 会让 "x" 变成参数，这是设计上
        允许的显式代价——反过来"自动猜哪些是参数"会在形状巧合时静默猜错。
        """
        excluded = set(exclude)
        overrides = dict(init or {})
        arrays: Dict[str, np.ndarray] = {}
        for value in graph.inputs:
            if value.name in excluded:
                continue
            if value.type.dtype not in _FLOAT_DTYPES:
                raise TypeError(
                    f"图输入 %{value.name} 的类型是 {value.type.dtype.value}，"
                    "不能作为可训练参数（只有 f32/f64 需要梯度）"
                )
            if not value.type.is_static:
                raise ValueError(
                    f"图输入 %{value.name} 的形状 {value.type} 含符号维，"
                    "参数形状必须静态已知（符号维是给激活用的）"
                )
            shape = tuple(int(d) for d in value.type.shape)
            arrays[value.name] = init_array(
                value.name, shape, value.type.dtype, overrides.get(value.name), seed
            )
        missing = set(overrides) - set(arrays)
        if missing:
            raise KeyError(f"init 里指定的参数不在图输入中: {sorted(missing)}")
        return cls(arrays)

    # -- 访问 ---------------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._arrays)

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {name: tuple(arr.shape) for name, arr in self._arrays.items()}

    def arrays(self) -> Dict[str, np.ndarray]:
        """内部数组（优化器原地更新的唯一入口）。"""
        return self._arrays

    def feed(self, extra: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """构造完整的执行输入：参数 + 本批数据。

        名字冲突直接报错：数据输入与参数同名时，无论谁覆盖谁都是错的。
        """
        feed = dict(self._arrays)
        for name, arr in (extra or {}).items():
            if name in feed:
                raise KeyError(
                    f"数据输入 {name!r} 与参数同名；请给数据换个名字（参数名由图输入决定）"
                )
            feed[name] = arr
        return feed

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        if name not in self._arrays:
            raise KeyError(f"未知参数: {name}")
        if tuple(value.shape) != tuple(self._arrays[name].shape):
            raise ValueError(
                f"参数 {name} 的形状必须保持 {self._arrays[name].shape}，得到 {value.shape}"
            )
        self._arrays[name] = np.asarray(value, dtype=self._arrays[name].dtype)

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def __len__(self) -> int:
        return len(self._arrays)

    def __iter__(self):
        return iter(self._arrays)

    def __repr__(self) -> str:
        return f"ParameterStore({len(self._arrays)} 个参数: {', '.join(self.names)})"
