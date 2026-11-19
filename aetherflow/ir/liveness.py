"""编译期的静态显存预算。

为什么需要它：``Arena`` 报的峰值是**捕获时实际发生**的峰值——它必须先跑一遍
（或者在真机上捕获一次）才知道答案。编译器应该在看图的时候就能回答
"这张图至少要多少显存"，否则显存规划无法参与"要不要选这张图/这个分块"的
决策，也没法在没有 GPU 的机器上做容量评估。

算法的核心只有一条：**值的生命周期就是"被定义"到"最后一次被使用"**。
图按拓扑序执行，所以：

- 节点输出的定义时刻 = 它在 ``graph.nodes`` 里的下标；
- 图输入在时刻 -1 就已存在（由调用方 / 参数驻留持有）；
- 最后一次使用 = 把它当输入的最大节点下标；图输出则一直活到最后。

峰值取所有时刻活跃值字节数之和的最大值。

**这不是"任何实现都绕不过去的严格下界"，而是指定存储假设下的 IR 估计。**
成立需要三条假设，每条都有已知的方向性偏差：

1. **每个 Value 独占一块存储。** 后端把连续张量上的 ``reshape`` / ``slice``
   实现成视图时，预算会**高估**（视图不额外占存储）。
2. **中间值的生命周期由 IR 决定。** 见下面的 ``release`` 参数：只有
   "用完就摘"（``release=True``）时峰值才是这个数；默认的普通执行把中间值
   留到图结束，实际占用等于 ``total_bytes``。
3. **只算 IR 里看得见的张量。** 内核临时区（共享内存之外的全局暂存）、
   cuBLAS / cuDNN 的 workspace、优化器槽位、捕获缓存与主机副本都不在内，
   这部分会**低估**。

``total_bytes`` 是"不复用"的占用，``peak_live_bytes`` 是"按 liveness 复用"
的占用，两者之比 ``reuse_ratio`` 是复用能带来的空间倍数——与 5.4 的实测
（70 → 34 MiB）是同一件事，只是这里不用跑。``release=False`` 时两者相等，
``reuse_ratio`` 恒为 1。

与 ``Arena`` 的关系：``Arena`` 的 ``peak_live_bytes`` 统计的是**中间缓冲**
（图输入来自参数驻留或调用方，不经过 arena），所以对照时用
``intermediate_peak_bytes``。两者不要求严格相等：arena 把每个请求向
512 字节对齐、后端的某些小临时（如 softmax 的 scale 标量）不体现在 IR
里，这些让实测偏大；后端把连续张量上的 reshape 实现成视图时又会偏小。
在 5.36 的实测里两者相差 8192 / 6656 B，远小于一个数量级——静态预算
因此可以直接用于容量评估，而不必先跑一遍。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .graph import Graph
from .types import DType, TensorType

__all__ = ["Footprint", "MemoryBudget", "plan_memory", "ALIGNMENT"]

ALIGNMENT = 512

_DTYPE_BYTES = {
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.F64: 8,
    DType.I8: 1,
    DType.I32: 4,
    DType.BOOL: 1,
}


def _align_up(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class Footprint:
    """一个值的字节数与生命周期。"""

    name: str
    bytes: int
    aligned_bytes: int
    defined_at: int
    last_used_at: int
    is_input: bool
    is_output: bool

    @property
    def live_for(self) -> int:
        return self.last_used_at - self.defined_at


@dataclass(frozen=True)
class MemoryBudget:
    """一张图的静态显存预算。

    ``release`` 记录这份预算对应哪种执行策略：``True`` 按 liveness 用完即摘
    （``Executor.run_device(release=True)``，也是捕获路径的前置），``False``
    把中间值留到图结束（普通执行的默认）。同一张图在两种策略下的
    ``peak_live_bytes`` 可以相差好几倍，所以引用数字时必须带上这个口径。
    """

    values: Tuple[Footprint, ...]
    peak_live_bytes: int
    intermediate_peak_bytes: int
    total_bytes: int
    intermediate_total_bytes: int
    aligned_peak_bytes: int
    aligned_intermediate_peak_bytes: int
    symbols: Mapping[str, int]
    release: bool = True

    @property
    def reuse_ratio(self) -> float:
        """不复用与峰值之比；``release=False`` 时恒为 1。

        ``<= 1`` 表示没有可复用的空间（例如每个值都活到最后）。
        """
        if self.peak_live_bytes <= 0:
            return 1.0
        return self.total_bytes / self.peak_live_bytes

    def as_dict(self) -> dict:
        return {
            "release": self.release,
            "peak_live_bytes": self.peak_live_bytes,
            "intermediate_peak_bytes": self.intermediate_peak_bytes,
            "total_bytes": self.total_bytes,
            "intermediate_total_bytes": self.intermediate_total_bytes,
            "aligned_peak_bytes": self.aligned_peak_bytes,
            "aligned_intermediate_peak_bytes": self.aligned_intermediate_peak_bytes,
            "reuse_ratio": self.reuse_ratio,
            "symbols": dict(self.symbols),
        }


def _bytes_of(type_: TensorType, symbols: Mapping[str, int]) -> int:
    try:
        resolved = type_.resolve(dict(symbols))
    except KeyError as exc:
        raise ValueError(
            f"静态显存预算需要绑定符号维：%{exc.args[0]}（类型 {type_}）。"
            "请在 symbols= 里给出每个符号维的取值"
        ) from None
    return resolved.numel() * _DTYPE_BYTES[resolved.dtype]


def plan_memory(
    graph: Graph,
    symbols: Optional[Mapping[str, int]] = None,
    release: bool = True,
) -> MemoryBudget:
    """从 IR 直接算出显存预算，不执行、不需要 GPU。

    ``symbols`` 用于绑定形状里的符号维；图完全静态时可以省略。

    ``release`` 与 ``Executor.run_device`` 的同名开关一一对应，必须按实际
    执行策略选：普通执行默认不回收中间值（``release=False``），此时每个
    中间值都活到图结束，峰值等于总量；只有 ``release=True`` 的路径（捕获
    与显式回收）才拿到 liveness 的复用收益。默认 ``True`` 是为了与既有
    调用方和 ``Arena`` 的对照口径保持一致。
    """
    bindings: Dict[str, int] = dict(symbols or {})
    nodes = graph.nodes
    count = len(nodes)

    last_use: Dict[int, int] = {}
    for index, node in enumerate(nodes):
        for value in node.inputs:
            last_use[value.id] = index
    outputs = {v.id for v in graph.outputs}
    position = {node.output.id: i for i, node in enumerate(nodes)}

    footprints = []
    for value in graph.values:
        is_input = value.producer is None
        is_output = value.id in outputs
        defined_at = -1 if is_input else position[value.id]
        if not release:
            # 不回收：环境里的每个值都活到图结束，包括图输入。
            used_at = count
        elif is_output:
            used_at = count
        else:
            used_at = last_use.get(value.id, count if is_input else defined_at)
        size = _bytes_of(value.type, bindings)
        footprints.append(
            Footprint(
                name=value.name,
                bytes=size,
                aligned_bytes=_align_up(size),
                defined_at=defined_at,
                last_used_at=used_at,
                is_input=is_input,
                is_output=is_output,
            )
        )

    def peak(selected, aligned: bool) -> int:
        """事件扫描：``[defined_at, last_used_at]`` 闭区间上的最大同时活跃量。

        每个值在 ``defined_at`` 加一次、在 ``last_used_at + 1`` 减一次
        （半开区间等价于闭区间）。同一时刻先处理减、后处理加——排序时
        负数在前正好给出这个顺序，因为相邻值之间"前者最后一步就是后者的
        定义步"时，前者在该时刻已经不再活跃。
        """
        events = []
        for f in selected:
            size = f.aligned_bytes if aligned else f.bytes
            if size <= 0:
                continue
            events.append((f.defined_at, size))
            events.append((f.last_used_at + 1, -size))
        events.sort()
        live = best = 0
        for _, delta in events:
            live += delta
            if live > best:
                best = live
        return best

    intermediates = [f for f in footprints if not f.is_input]
    return MemoryBudget(
        values=tuple(footprints),
        peak_live_bytes=peak(footprints, aligned=False),
        intermediate_peak_bytes=peak(intermediates, aligned=False),
        total_bytes=sum(f.bytes for f in footprints),
        intermediate_total_bytes=sum(f.bytes for f in intermediates),
        aligned_peak_bytes=peak(footprints, aligned=True),
        aligned_intermediate_peak_bytes=peak(intermediates, aligned=True),
        symbols=bindings,
        release=release,
    )
