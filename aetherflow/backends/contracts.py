"""插件契约的可选层。

``base.Backend`` 是准入面，只有三个成员：``name`` / ``capabilities()`` /
``execute()``。任何后端都必须有，缺一个就不算后端。这里是它之外的**可选契约**：
后端具备对应能力才实现，不具备就不实现，前端与工具靠探测决定用不用。

为什么拆成独立协议，而不是继续往 ``Backend`` 上加方法：

- 生命周期不同。设备信息是"启动时问一次"，内存统计是"每次运行后关心"，
  而调优是离线的事，codegen 是编译期的事。混在一个接口里，最小准入面会变成
  所有插件都必须假装填满的巨型接口——第二个后端接入时，第一件事是写一堆
  返回空值的占位实现，那等于没有契约。
- 不实现的后端不受影响。``ReferenceBackend`` 一个可选契约都不实现，
  照样跑全部用例、照样通过 conformance。

**只在两个实现共享同一语义时才写契约。** ``CodegenContract`` / ``TuningContract``
在 5.163 提炼：CUDA 与 HIP 的 ``_kernel(key, source, func)`` 签名与语义一致，
调优面（``tunable_ops`` / ``tuning_space`` / ``set_tuning`` / ``clear_tuning`` /
``tuning_config``）也由同一个 ``DeviceBackend`` 提供。两者都是"离线/编译期"的
能力，和运行期的设备信息、内存统计生命周期不同，所以仍然分协议。

不写进契约的部分同样是刻意的：codegen 不约束句柄类型（CUDA 是
``cupy.RawKernel``，HIP 是模块函数），调优不约束搜索策略与存储格式——那些是
工具层的事，写进契约就等于要求每个后端自带一套调优器。

**探测的语义必须说清楚。** ``isinstance(obj, Protocol)`` 只检查方法**是否存在**，
不检查签名，更不检查返回值含义。所以 ``implemented_contracts()`` 的结论是
"这个后端声称有这项能力"，不是"这项能力是对的"。后者只能靠实际调用与
conformance 证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    FrozenSet,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:  # 只为注解，避免 contracts 在导入期把 tuning 的 numpy 依赖带进来
    from ..tuning.space import TuningSpace

__all__ = [
    "DeviceInfo",
    "MemoryStats",
    "DeviceContract",
    "DeviceManagerContract",
    "CodegenContract",
    "TuningContract",
    "MemoryContract",
    "StreamContract",
    "AllocatorContract",
    "CONTRACTS",
    "supports",
    "implemented_contracts",
]


@dataclass(frozen=True)
class DeviceInfo:
    """一块可用于计算的设备的静态信息。

    只收"插件自己知道、且上层真的会看"的三项。显存总量用字节而不是 GB，
    格式化是展示层的事，契约里不做四舍五入。
    """

    name: str
    runtime: int
    total_memory_bytes: int

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / 1024**3, 2)


@dataclass(frozen=True)
class MemoryStats:
    """分层分配器的一次快照，单位字节。

    四个数各有各的用途，缺一个都会让结论失真：

    - ``live_bytes``：当前仍被占用的量
    - ``peak_bytes``：历史峰值存活量，即这张图绕不过去的显存下界
    - ``total_bytes``：向驱动实际申请的总量（真实占用）
    - ``requested_bytes``：累计请求量，即"完全不复用需要多少"

    ``total_bytes`` 与 ``peak_bytes`` 的差额是分配器自身的浪费（块内碎片、对齐）。
    """

    live_bytes: int
    peak_bytes: int
    total_bytes: int
    requested_bytes: int


@runtime_checkable
class DeviceContract(Protocol):
    """设备管理器：能被询问"你用的是什么设备"。"""

    def device_info(self) -> DeviceInfo:
        ...


@runtime_checkable
class DeviceManagerContract(Protocol):
    """多设备：枚举、查询当前设备、切换。

    5.164 提炼。此前 ``StreamContract`` 的注释明确写着"不写多设备枚举与
    切换：HIP 侧没有对应实现"。HIP 侧补上之后，两家在三个方法上语义一致：

    - ``enumerate_devices()``：按索引顺序返回进程可见的全部设备；
    - ``current_device()``：当前上下文所在的设备索引；
    - ``set_device(index)``：切换当前设备，越界或类型不对**当场**报错。

    切换之后 ``device_info()`` 与自动 block 用的 SM 数必须跟着走——否则
    "切了设备但仍在旧卡上铺块"是最难查的一类错。这条要求写在契约说明里，
    因为协议只查方法存在性，查不到语义。

    与 ``DeviceContract`` 分开：单设备后端也能回答"我用的是什么设备"，
    但回答不了"还有哪几块"。生命周期上两者都是启动时问一次，拆开只是
    为了让能力粒度如实反映实现。
    """

    def enumerate_devices(self) -> Tuple[DeviceInfo, ...]:
        ...

    def current_device(self) -> int:
        ...

    def set_device(self, index: int) -> None:
        ...


@runtime_checkable
class CodegenContract(Protocol):
    """编译期：能把渲染好的厂商源码编成可调用句柄。

    CUDA 与 HIP 的私有实现是 ``_kernel(key, source, func)``；公开入口是
    ``DeviceBackend.kernel``，签名完全相同：

    - ``key``：编译缓存键，必须可哈希且稳定（通常是算子的渲染签名）。同一
      key 重复调用返回同一个句柄，不重复走编译器。
    - ``source``：厂商语言源码。CUDA 是 CUDA C，HIP 是它接受的 HIP C++——
      契约不要求两者同一种方言，只要求"这个后端的源码"。
    - ``func``：入口函数名。

    返回值的类型由后端决定，契约只承诺"可调用且能按该后端的启动方式使用"。
    公开入口 ``DeviceBackend.kernel()`` 把两种后端的形态收成一个带
    ``launch(...)`` 的包装句柄（同时保留 ``__call__``），跨后端工具因此不必
    关心"这个后端靠 ``__call__`` 还是 ``launch`` 启动"。
    编译失败必须抛异常，不要返回 ``None``——``None`` 会让失败延后到启动时，
    那时的报错已经不知道是哪段源码。
    """

    def kernel(self, key: Any, source: Any, func: str) -> Any:
        ...


@runtime_checkable
class TuningContract(Protocol):
    """调优面：枚举可调算子、给出搜索空间、读写覆盖配置。

    这是一组**离线**接口：调优工具在进程里反复改配置、重测，正式执行时只读
    ``tuning_config()``。它不负责搜索算法，也不负责任何测量——那些在
    ``aetherflow.tuning`` 里，对任何实现了本契约的后端都通用。

    配置的语义由后端定义（块大小、线程铺法、tile 形状……），契约只要求
    "``set_tuning`` 接受的东西，``tuning_config`` 能原样读回"，以及非法配置
    在 ``set_tuning`` 当场报错，而不是等到执行时才发现。
    """

    def tunable_ops(self) -> FrozenSet[str]:
        ...

    def tuning_space(self, op: str) -> "TuningSpace":
        ...

    def set_tuning(self, op: str, config: Mapping[str, Any]) -> None:
        ...

    def clear_tuning(self, op: Optional[str] = None) -> None:
        ...

    def tuning_config(self) -> Mapping[str, Mapping[str, Any]]:
        ...


@runtime_checkable
class StreamContract(Protocol):
    """流管理：能开一条用于捕获与 replay 的非阻塞流。

    5.89 把这半件事从 CUDA 插件内部提出来：CUDA 与 HIP 现在都已实现，且
    语义一致（非阻塞；图捕获必须发生在它上面，默认流上捕获会被驱动拒绝）。
    它和 ``DeviceContract`` 分开，是模块头那条"生命周期不同就拆协议"的直接
    应用——设备信息是启动时问一次，流是每个捕获/每步训练都要开的东西。

    仍然不写多设备枚举与切换：那三项在 5.164 独立成 ``DeviceManagerContract``，
    而不是塞进流契约——开流与选卡是两件事。
    """

    def new_stream(self) -> Any:
        ...


@runtime_checkable
class MemoryContract(Protocol):
    """内存分层：能被询问"上一次运行用了多少、峰值多少"，并能交出一个分配器。

    没有可报告的对象时返回 ``None``，不要返回一串 0——0 是一个具体的观测值，
    会让"没有统计"和"用了 0 字节"混为一谈。

    ``allocator()`` 返回的是一个**新**的分配器（不与上一次捕获共享地址：
    地址写死在图里，共享会让两张图的中间值互相踩），它的形状由
    ``AllocatorContract`` 描述。
    """

    def memory_stats(self) -> Optional[MemoryStats]:
        ...

    def allocator(self) -> Any:
        ...


@runtime_checkable
class AllocatorContract(Protocol):
    """一次捕获期内的显存规划器。

    只在两个真实实现共享的那五个方法上提炼（``cuda/memory.Arena`` 与
    ``hip/memory.Arena``）。``alias`` / ``unalias`` 不在契约里——那是 cupy
    零拷贝视图特有的记账，HIP 的 ``DeviceArray`` 没有这一层。

    ``recycle`` 也不在契约里：只有 HIP 需要它（CUDA 侧预热轮的块由池回收），
    把它写进契约就等于要求 CUDA 实现一个永远不用的方法。

    注意它**不挂在后端上**，所以不进 ``CONTRACTS`` 表：那张表回答的是
    "这个后端能做什么"，而分配器是后端交出来的对象。
    """

    def malloc(self, size: int) -> Any:
        ...

    def free(self, ptr: Any) -> None:
        ...

    def stats(self) -> MemoryStats:
        ...

    def reset_stats(self) -> None:
        ...

    def close(self) -> None:
        ...


CONTRACTS: Mapping[str, type] = {
    "device": DeviceContract,
    "device_manager": DeviceManagerContract,
    "codegen": CodegenContract,
    "tuning": TuningContract,
    "memory": MemoryContract,
    "stream": StreamContract,
}


def supports(backend: object, contract: type) -> bool:
    """后端是否**声称**实现了某个契约（只查方法存在性）。"""
    return isinstance(backend, contract)


def implemented_contracts(backend: object) -> Tuple[str, ...]:
    """按 ``CONTRACTS`` 的声明顺序，返回后端声称实现的契约名。"""
    return tuple(name for name, proto in CONTRACTS.items() if supports(backend, proto))
