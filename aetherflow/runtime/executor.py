"""后端无关的图执行骨架。

所有后端共享同一套语义边界，只替换"单个节点怎么算"这一步：

    to_array -> 绑定符号维 -> 逐节点 invoke -> 形状校验 -> cast -> to_host

这样做的原因不是为了让代码短，而是为了保证**不同后端的形状校验和符号绑定
行为完全一致**——如果每个后端各写一遍，它们迟早会在边界条件上分叉，
而那种分叉表现为"某个后端精度对不齐"，排查成本极高。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
from weakref import WeakKeyDictionary

import numpy as np

from ..ir.graph import Graph, Node
from ..ir.types import TensorType, dim_resolve


class ShapeMismatchError(ValueError):
    pass


class _GroupPlan:
    """一个执行组里与运行期形状无关的部分。

    ``args`` 是"每个节点的每个输入在环境里看哪一格"，``pending`` 标记组内
    前序节点的输出（那几格运行时是 ``None``）。``attrs`` 与 ``expected``
    只在整张图静态时才有值：那时解析结果与绑定无关，可以直接复用 IR 上的
    对象；含符号维的图按本次绑定解析，结果记在单格备忘里（见 ``resolve``）。
    """

    __slots__ = ("nodes", "args", "attrs", "expected", "dtype", "kernels", "constants", "_memo")

    def __init__(self, nodes: Tuple[Node, ...], static: bool) -> None:
        self.nodes = nodes
        # 批量派发用：逐节点的内核函数，由后端的 ``_prepare_plan`` 填。
        # ``None`` = 这份计划不能跳过运行期校验，仍走 ``invoke``。
        self.kernels = None
        # 常量节点的设备副本：值在属性里、与运行期输入无关，因此可以随这张
        # 图的计划一起缓存。设备后端在 ``_prepare_plan`` 里填，骨架在
        # ``run_device`` 里优先取用——省掉每次执行的 H2D 与重新分配。
        self.constants: Dict[int, Any] = {}
        pending = {node.output.id for node in nodes[:-1]}
        self.args = [
            tuple((a.id, a.id in pending) for a in node.inputs) for node in nodes
        ]
        self.attrs = [node.attrs for node in nodes] if static else None
        self.expected = nodes[-1].output.type if static else None
        # dtype 与形状无关，构建期解析成 np.dtype 存下来：``np.asarray(x, "float32")``
        # 每次都要做一次字符串→dtype 的解析，而这是每个节点的必经之路。
        self.dtype = np.dtype(nodes[-1].output.type.dtype.np_dtype)
        self._memo = None

    def resolve(self, binding: Dict[str, int]):
        """返回本组的 ``(attrs, expected)``。

        静态图在构建期就算好了，直接给。含符号维时二者只取决于本次绑定：
        同一组形状重复执行会命中上一次的结果，省掉每步一次的递归替换与
        ``TensorType`` 重建——24 节点的符号图实测每步因此省下 0.1 ms 量级。
        备忘只留最后一格：形状切换就重算，语义不受影响，内存也不会膨胀。
        """
        if self.attrs is not None:
            return self.attrs, self.expected
        key = tuple(binding.items())
        memo = self._memo
        if memo is not None and memo[0] == key:
            return memo[1], memo[2]
        attrs = [resolve_attrs(node.attrs, binding) for node in self.nodes]
        expected = self.nodes[-1].output.type.resolve(binding)
        self._memo = (key, attrs, expected)
        return attrs, expected


def resolve_attrs(attrs: dict, binding: Dict[str, int]) -> dict:
    """把属性里出现的符号维替换成运行期绑定值。

    只替换能在 binding 中命中的字符串，所以 ``dtype="f32"`` 这类
    字面量属性不会被误改——前提是形状符号不叫 ``f32``，而这一点由
    ``TensorType`` 构造期拒绝（见 ``ir/types.py``）。
    """

    def resolve(v):
        if isinstance(v, str):
            return binding.get(v, v)
        if isinstance(v, (tuple, list)):
            return type(v)(resolve(x) for x in v)
        return v

    return {k: resolve(v) for k, v in attrs.items()}


def _check_input_dtype(arr: Any, want: np.dtype, name: str) -> None:
    """入口处拒绝有损的隐式转换；无损的浮点对齐仍照常做。

    - 浮点 -> 浮点：允许。声明 f32 就必须按 f32 算（AF-10），f64 输入在此
      对齐，这是协议而不是降级。
    - 浮点 -> 整数：拒绝。``astype`` 会静默截断（``0.9`` 变 ``0``），token
      与下标在入口处就被改掉，后面谁都看不出来。
    - 整数窄化：先验范围。``int64(2**32)`` 塞进 i32 会包成 0，越界必须报错。
    - 布尔 -> 整数 / 数值 -> 布尔：拒绝。``True`` 不是下标。
    """
    src = np.dtype(arr.dtype)
    if src == want:
        return
    if want.kind in "iu":
        if src.kind == "b":
            raise TypeError(
                f"输入 %{name} 声明为 {want.name}，收到布尔数组；布尔不是整数"
            )
        if src.kind == "f":
            raise TypeError(
                f"输入 %{name} 声明为 {want.name}，收到浮点 {src.name}；"
                "浮点转整数必须由调用方显式转换"
            )
        if src.kind in "iu" and arr.size:
            # 同位宽也必须验范围：uint32 -> int32 不"窄化"，itemsize 相等，
            # 但 2**31..2**32-1 会静默回绕成负数（外部审计 F10）。真正要
            # 跳过检查的只有无损的加宽（如 i32 -> i64）。
            src_info, want_info = np.iinfo(src), np.iinfo(want)
            if want_info.min > src_info.min or want_info.max < src_info.max:
                info = want_info
                low, high = int(arr.min()), int(arr.max())
                if low < info.min or high > info.max:
                    raise ValueError(
                        f"输入 %{name} 声明为 {want.name}，取值 [{low}, {high}] "
                        f"超出范围 [{info.min}, {info.max}]"
                    )
    elif want.kind == "b" and src.kind != "b":
        raise TypeError(
            f"输入 %{name} 声明为 bool，收到 {src.name}；布尔输入必须显式给出"
        )


class Executor:
    """模板方法式执行器。子类只需实现 ``invoke`` 与容器相关的三个钩子。"""
    name: str = "executor"

    # ---- 容器钩子（默认按 numpy 处理） ----
    def to_array(self, value: Any) -> Any:
        return np.asarray(value)

    def cast(self, value: Any, dtype: np.dtype) -> Any:
        return np.asarray(value, dtype=dtype)

    def to_host(self, value: Any) -> Any:
        return value

    def dispose(self, value: Any) -> None:
        """告诉后端某个中间值已经不会再被读取。默认什么都不做。

        numpy 侧回收由 GC 负责；CUDA 捕获期需要它把地址还给 arena，
        因为 cupy 14 的 ``MemoryPointer`` 销毁时不会回调自定义分配器。
        """

    # ---- 后端必须实现的部分 ----
    def invoke(self, node: Node, args: list, attrs: dict) -> Any:
        raise NotImplementedError

    # ---- 可选：节点分组（语义级融合） ----
    def group_nodes(self, graph: Graph) -> List[Tuple[Node, ...]]:
        """把拓扑序的节点划分成执行组。默认一个节点一组。

        分组是纯计划：哪些节点可以放在一次内核里算，由后端按自己的能力决定
        （融合需要"中间值只被链内使用""形状相同"这类前提）。执行骨架只保证
        两件事：组内按原顺序执行；组结束后只把**最后一个节点**的输出写回环境，
        中间值因此不会被物化，也不会被后续节点读到。
        """
        return [(node,) for node in graph.nodes]

    def invoke_group(self, group: Tuple[Node, ...], args: list, attrs: list,
                     plan: Optional["_GroupPlan"] = None) -> Any:
        """执行一组节点，返回最后一个节点的输出。默认逐个 ``invoke``。

        ``args`` / ``attrs`` 与 ``group`` 一一对应（每个节点的输入值与属性）。
        组内前序节点的输出还没有写回环境，``args`` 里对应位置是 ``None``；
        逐个执行时直接用刚刚算出的结果补上，语义与物化后重读完全一致。

        ``plan`` 是这组在缓存计划里的那份记录。它带 ``kernels`` 时说明后端
        已在计划构建期完成校验并绑定好内核函数，跳过 ``invoke`` 的逐次检查
        直接调用——这就是批量派发的快路径（见 ``invoke_planned``）。
        """
        if plan is not None and plan.kernels is not None:
            if len(group) == 1:
                # 单节点组没有"组内前序输出"，args[0] 就是它要的全部输入，
                # 不必再走一遍补位循环。
                return plan.kernels[0](group[0], args[0], attrs[0])
            return self.invoke_planned(group, args, attrs, plan.kernels)
        produced: Dict[int, Any] = {}
        out = None
        for node, node_args, node_attrs in zip(group, args, attrs):
            resolved = [
                produced[a.id] if a.id in produced else arr
                for a, arr in zip(node.inputs, node_args)
            ]
            out = self.invoke(node, resolved, node_attrs)
            produced[node.output.id] = out
        return out

    def invoke_planned(
        self, group: Tuple[Node, ...], args: list, attrs: list, kernels: tuple
    ) -> Any:
        """按已绑定的内核函数逐节点执行，不再过 ``invoke`` 的校验。

        语义与 ``invoke_group`` 的默认实现逐字对应，只把"查算子、查 dtype、
        选候选"这一层换成计划期绑定好的函数对象。校验并没有被取消，而是被
        ``_prepare_plan`` 前移到计划构建期——只有"必然通过"的组才会带
        ``kernels``，其余仍走原路径，报错时机与文案都不变。
        """
        produced: Dict[int, Any] = {}
        out = None
        for node, node_args, node_attrs, fn in zip(group, args, attrs, kernels):
            resolved = [
                produced[a.id] if a.id in produced else arr
                for a, arr in zip(node.inputs, node_args)
            ]
            out = fn(node, resolved, node_attrs)
            produced[node.output.id] = out
        return out

    # ---- 固定流程 ----
    def run(self, graph: Graph, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """跑完整张图，只返回输出。

        这里固定用 ``release=True``：``run`` 只读图输出，环境本身不返回，
        摘掉已经没人再读的中间值不会改变任何可观察结果，却能让分配器立刻
        复用那些缓冲。需要完整环境（捕获、逐值调试）请直接调
        ``run_device``，它的默认值仍是"不回收"。
        """
        env = self.run_device(graph, inputs, release=True)
        outputs = {
            name: self.to_host(env[v.id])
            for v, name in zip(graph.outputs, graph.output_names)
        }
        # 输出已经拷到主机，设备侧不再有使用者。tiered 模式下这一步才让
        # arena 的 live 在执行结束时归零——否则每轮的上传副本与输出块都会
        # 留着，看起来像"分层没有复用"，实际是逐轮累积。
        for value in graph.outputs:
            self.dispose(env[value.id])
        # 输入同理：只有"本轮为 numpy 输入新建的设备副本"才还块。调用方
        # 直接传设备数组（或驻留参数）时 env 里就是它本人，不能释放。
        for value in graph.inputs:
            device = env[value.id]
            if inputs.get(value.name) is not device:
                self.dispose(device)
        return outputs

    def run_device(
        self, graph: Graph, inputs: Mapping[str, Any], release: bool = False
    ) -> Dict[int, Any]:
        """在设备上跑完整张图，返回 ``值 id -> 容器值`` 的环境，不做 ``to_host``。

        单独拆出来是为了 CUDA Graph 捕获：捕获期间不允许任何主机同步，
        而 ``run`` 结尾的 ``to_host`` 恰好是一次强制同步。

        ``release=True`` 时按"最后一次被使用"回收中间值：节点执行完就把它
        那些不再被任何节点读取、也不是图输出的输入从环境里摘掉。这是复用
        中间缓冲的前提——若不摘，全部中间值会一起活到图结束，分配器只能
        一路向上要新块（实测因此多占一倍显存）。

        ``run`` 走这条路径（它不需要环境）；``run_device`` 默认保持关闭，
        因为它的契约就是"交出完整环境"，摘掉键会破坏调用方的预期。容量
        评估因此要选对口径：``plan_memory(graph, release=False)`` 对应这里
        的默认行为，``release=True`` 对应回收后的峰值。
        """
        binding: Dict[str, int] = {}
        env: Dict[int, Any] = {}
        plan = self._plan(graph)

        remaining: Dict[int, int] = {}
        if release:
            keep = {v.id for v in graph.inputs} | {v.id for v in graph.outputs}
            # 常量是计划的一部分：它们由后端缓存、跨执行复用以省掉 H2D，
            # 因此不能按"最后一次被读"回收——回收掉就等于把缓存打空，
            # 下一轮重新上传，退回到修这个缺陷之前的行为。
            for group in plan:
                keep.update(group.constants)
            for node in graph.nodes:
                for a in node.inputs:
                    remaining[a.id] = remaining.get(a.id, 0) + 1

        for value in graph.inputs:
            if value.name not in inputs:
                raise KeyError(f"缺少图输入: {value.name}")
            arr = self.to_array(inputs[value.name])
            # 输入在入口就对齐到 IR 声明的 dtype。此前只校验形状，于是
            # "声明 f32、传 f64"会让整条链按 f64 算、直到输出才降回去：
            # 类型成了协议，精度却不是。BF16 的承载本来就是 f32
            # （见 ``TensorType`` 的 np_dtype），这里不会对它做位宽转换。
            want = np.dtype(value.type.dtype.np_dtype)
            if arr.dtype != want:
                _check_input_dtype(arr, want, value.name)
                arr = self.cast(arr, want)
            self._bind(value.type, tuple(arr.shape), f"输入 %{value.name}", binding)
            env[value.id] = arr

        for group in plan:
            # 组内前序节点的输出还没进环境（融合时它们甚至不会被物化），
            # 这里留 None 占位；``invoke_group`` 决定是内联计算还是逐个补读。
            # 单节点组不存在待补位的槽，直接按输入顺序取值，省掉一层循环。
            if len(group.nodes) == 1:
                args = [[env[a.id] for a in group.nodes[0].inputs]]
            else:
                args = [
                    [None if pending else env[vid] for vid, pending in slots]
                    for slots in group.args
                ]
            attrs, expected = group.resolve(binding)
            single = group.nodes[0] if len(group.nodes) == 1 else None
            if single is not None and single.output.id in group.constants:
                # 常量：计划里已有设备副本，直接复用；不重新上传、不重新分配。
                out = group.constants[single.output.id]
            else:
                out = self.invoke_group(group.nodes, args, attrs, plan=group)
            last = group.nodes[-1]
            if out.shape != expected.shape:
                raise ShapeMismatchError(
                    f"节点 %{last.output.name}（{last.op.name}）运行期形状 {tuple(out.shape)} "
                    f"与 IR 推导 {expected.shape} 不一致"
                )
            env[last.output.id] = self.cast(out, group.dtype)
            out = None

            if release:
                # 按组回收：融合后的中间值从未进过环境，这里只是把它们的
                # 引用计数扣到 0；真正的回收发生在组的最后一个消费者之后。
                for node in group.nodes:
                    for a in node.inputs:
                        remaining[a.id] -= 1
                        if remaining[a.id] == 0 and a.id not in keep:
                            dead = env.pop(a.id, None)
                            if dead is not None:
                                self.dispose(dead)
                # 局部引用也得断：args 会把这个节点的输入一直持有到下一轮赋值，
                # 那样"最后一次使用"就晚了一个节点，复用的空档被白白吃掉。
                args = None

        return env

    def _plan(self, graph: Graph) -> List[_GroupPlan]:
        """图的执行计划：分组与每组的静态部分，按图缓存一次。

        分组（``group_nodes``）与 attrs 解析都与运行期形状无关，此前却每步
        重算一遍——24 节点的图每步因此多花约 0.1 ms。缓存以图的结构版本号
        失效：``Graph`` 的构建入口会自增版本，节点 attrs 构建后只读。
        """
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = WeakKeyDictionary()
            self._plan_cache = cache
        hit = cache.get(graph)
        if hit is not None and hit[0] == graph.revision:
            return hit[1]
        static = all(v.type.is_static for v in graph.inputs)
        plan = [_GroupPlan(group, static) for group in self.group_nodes(graph)]
        self._prepare_plan(plan)
        cache[graph] = (graph.revision, plan)
        return plan

    def _prepare_plan(self, plan: List[_GroupPlan]) -> None:
        """计划构建后的钩子：后端可以在这里把校验结果提前算好。

        默认什么都不做——参考执行器没有"每节点一次的固定检查"可省。
        设备后端把"算子是否注册、输入 dtype 是否受支持"判定一次，通过的组
        记下内核函数，运行期走 ``invoke_planned`` 的批量派发路径。
        """

    def _drop_plan(self, graph: Graph) -> None:
        """丢掉某张图的计划缓存，下一次执行重新绑定内核与常量。

        捕获流程必须在"预热轮"与"捕获轮"之间调用它：常量在预热期上传到
        捕获 arena 里，而捕获轮会把预热期的地址整体收回（HIP 的
        ``Arena.recycle``），继续复用那份副本就是读到后来别人写进去的数据
        ——表现是 loss 直接错到 10^2 量级，而不是报错。重建计划让常量在
        捕获轮里重新上传，地址因此写进被捕获的图里。
        """
        cache = getattr(self, "_plan_cache", None)
        if cache is not None:
            cache.pop(graph, None)

    def _clear_plan_values(self) -> None:
        """丢掉计划里缓存的设备副本（目前只有常量），并强制重建计划。

        切卡、关闭 arena 这类"地址整体失效"的事件必须调用它：里面的设备
        数组指向旧设备或已释放的内存块，继续用会读到别人的数据。只清值
        不够——``_plan`` 按 ``graph.revision`` 缓存，清空后不会自动重跑
        ``_prepare_plan``，常量就再也回不到缓存里。因此这里直接清计划缓存，
        由下一次执行按新设备重新上传；计划重建的代价是每张图一次。
        """
        cache = getattr(self, "_plan_cache", None)
        if cache is not None:
            cache.clear()

    @staticmethod
    def _bind(type_: TensorType, actual: tuple, where: str, binding: Dict[str, int]) -> None:
        """把运行期形状绑定到符号维，同时校验静态维。"""
        if len(actual) != type_.rank:
            raise ShapeMismatchError(f"{where} 秩不匹配：期望 {type_.rank}，实际 {len(actual)}")
        for declared, real in zip(type_.shape, actual):
            if isinstance(declared, int):
                if declared != real:
                    raise ShapeMismatchError(f"{where} 静态维不匹配：期望 {declared}，实际 {real}")
            elif isinstance(declared, str):
                prev = binding.get(declared)
                if prev is None:
                    binding[declared] = int(real)
                elif prev != real:
                    raise ShapeMismatchError(
                        f"{where} 符号维 {declared} 绑定冲突：先前为 {prev}，本次为 {real}"
                    )
            else:
                # 派生维（如卷积的 (S - 1) / 2 + 1）不能反解出新符号，
                # 只能用已绑定的符号求值后核对。要求所依赖的符号在更早的
                # 维或更早的输入里出现过——这与"符号必须先出现"的约定一致。
                try:
                    expected = dim_resolve(declared, binding)
                except KeyError as exc:
                    raise ShapeMismatchError(
                        f"{where} 的派生维 {declared} 依赖未绑定的符号 {exc.args[0]}"
                    )
                if expected != real:
                    raise ShapeMismatchError(
                        f"{where} 派生维 {declared} 求值为 {expected}，实际 {real}"
                    )
