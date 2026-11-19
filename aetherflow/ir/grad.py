"""在语义 IR 上做反向展开（reverse-mode 自动微分）。

设计上的三个决定
----------------

1. **不新增 IR 层。** 反向就是"再构一张图"：前向节点原样重放，反向节点
   由每个算子自己的 ``vjp`` 规则组合出来。反向图仍然是同一套 PORTABLE IR，
   因此 ``verifier``、序列化、任意后端都一视同仁——"一套 IR"这条主线不为 AD
   开特例。DESIGN 1212 行的待决项（AD 放语义 IR 层还是新增一层）就此落定。

2. **规则挂在算子上，不在引擎里写 if/else。** 引擎只负责拓扑逆序、cotangent
   累加与缺规则时报错；"这个算子怎么求导"的知识属于算子本身。SNN 的非标准
   反传（替代梯度）也因此只是替换某个算子的 vjp，而不是改引擎。

3. **反向图是自包含的。** 它重新计算前向（值不靠外部喂），输入只有原图输入
   加每个输出的种子梯度。代价是重复计算，换来的是一张可以被单独导出、单独
   执行、单独对拍的图。先正确，再谈检查点与重算最小化。

第一版的边界（都写进 DESIGN 5.17）
----------------------------------
- 种子梯度是**图输入**，不是常量：IR 里没有"无来源的值"，让调用方显式给。
- 与广播相关的反向（求和归约回原形状）需要静态形状；符号维在那一处显式报错。
- 反向图里的 ``conv2d_*_grad`` / ``maxpool2d_backward`` 没有定义二阶导数，
  对反向图再求导会得到 ``NoGradientError``，而不是静默的二阶零。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from . import grad_rules  # noqa: F401  导入即把 vjp 规则挂到 OPS 上
from .graph import Graph, Node, Value
from .ops import NoGradientError

__all__ = ["backward", "GradCtx", "NoGradientError"]


@dataclass(frozen=True)
class GradCtx:
    """一条反向规则能看到的全部上下文。

    ``fwd`` 是"前向 Value.id -> 反向图中的同名 Value"的映射：反向图重放了
    前向，所以每个前向值在反向图里都有一个对应的 ``Value``，规则只管用它构图。
    """

    b: Graph
    node: Node
    g: Value
    fwd: Mapping[int, Value]

    def input(self, i: int) -> Value:
        return self.fwd[self.node.inputs[i].id]

    @property
    def output(self) -> Value:
        return self.fwd[self.node.output.id]

    @property
    def attrs(self) -> dict:
        return self.node.attrs


def backward(
    graph: Graph,
    wrt: Optional[Sequence[Value]] = None,
    seeds: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    active: Optional[Sequence[str]] = None,
    minimize: bool = False,
) -> Graph:
    """返回 ``graph`` 的反向图。

    参数
    ----
    wrt
        对哪些 Value 求梯度，默认全部图输入。可以是中间值（对激活求导）。
    seeds
        输出别名 -> 种子输入名。默认对每个输出 ``y`` 生成输入 ``dy``。
        种子形状与 dtype 必须与对应输出一致（由图输入声明强制）。
    name
        反向图的名字，默认为 ``<graph.name>_backward``。
    active
        参与反向的输出别名，默认全部。**未激活的输出会被真正剪掉**：
        不给它建种子输入，也不重放只有它能到达的前向节点。

        不能用"给它喂零种子"代替：零种子只在导数链上每一环都有限时才等价于
        不传播。``loss=sum(x*x)`` 配一个诊断输出 ``sum(log(x))``，在 x=0 处
        零种子仍然算出 ``0/x``，得到 NaN，把对的梯度污染成 NaN。剪枝是在
        构图期按可达性做的，不存在"算出来再丢掉"的那一步。
    minimize
        只对"求导目标的下游"调用 vjp：``wrt`` 无关的分支不产生反向节点。
        默认 ``False``（保持既有构图行为）；``True`` 时反向图的数值与默认
        路径逐位一致，节点数不增。前向重放范围不变——被剪掉的分支仍可能
        作为某个 vjp 的输入值被读取，而"读值"与"求它的导数"是两件事。

    返回图的输出命名为 ``d<wrt 的 SSA 名>``，顺序与 ``wrt`` 一致。

    为什么不能"自动补零梯度"：IR 里没有无来源的值，造零只能靠 ``sub(x, x)``
    之类的假节点，那会把"这个参数与损失无关"这件事藏进图里。这里选择报错，
    让调用方自己决定是删掉这个参数还是显式给零。
    """
    if not isinstance(minimize, bool):
        raise TypeError(
            f"minimize 必须是 bool，得到 {type(minimize).__name__}；"
            "这个开关改变的是构图范围，不接受『真值即可』的隐式转换"
        )
    if wrt is None:
        wrt = graph.inputs
    wrt = tuple(wrt)
    for v in wrt:
        if v not in graph.values:
            raise ValueError(f"求导目标 %{v.name} 不属于图 {graph.name!r}")

    aliases = graph.output_names
    if seeds is None:
        seed_names = {a: f"d{a}" for a in aliases}
    else:
        unknown = set(seeds) - set(aliases)
        if unknown:
            raise KeyError(f"seeds 里有图输出中不存在的别名: {sorted(unknown)}")
        seed_names = {a: seeds.get(a, f"d{a}") for a in aliases}

    if active is None:
        active_aliases = tuple(aliases)
    else:
        active_aliases = tuple(active)
        unknown = set(active_aliases) - set(aliases)
        if unknown:
            raise KeyError(f"active 里有图输出中不存在的别名: {sorted(unknown)}")
        if not active_aliases:
            raise ValueError("active 不能为空：至少要有一个输出参与反向")
    active_set = set(active_aliases)

    b = Graph(name or f"{graph.name}_backward")

    # 1) 前向输入与种子梯度。种子**先于**所有节点声明：SSA 编号必须满足
    #    "输入在前"，否则 JSON 往返（先恢复全部输入、再重放节点）会让名字错位。
    fwd: Dict[int, Value] = {}
    for v in graph.inputs:
        fwd[v.id] = b.input(v.name, v.type)

    seed_pairs = []
    for value, alias in zip(graph.outputs, aliases):
        if alias not in active_set:
            continue
        sname = seed_names[alias]
        try:
            seed = b.input(sname, value.type)
        except ValueError as exc:  # 与图输入或已用别名撞名
            raise ValueError(
                f"种子输入名 {sname!r} 与前向图的已有名字冲突，请用 seeds= 指定别的名字"
            ) from exc
        seed_pairs.append((value, seed))

    # 2) 只重放激活输出可达的前向节点。节点按拓扑序排列，因此一次逆序扫描
    #    就能把"能到达某个激活输出"的集合算出来：被访问到的节点，它的输入
    #    也一定在集合里。未激活分支从此不在反向图里，既不算也不导数。
    live = {v.id for v, alias in zip(graph.outputs, aliases) if alias in active_set}
    for node in reversed(graph.nodes):
        if node.output.id in live:
            for a in node.inputs:
                live.add(a.id)

    for node in graph.nodes:
        if node.output.id not in live:
            continue
        args = [fwd[a.id] for a in node.inputs]
        fwd[node.output.id] = b.add_node(node.op.name, args, dict(node.attrs))

    # 2b) ``minimize``：算出"求导目标的下游"值集合，vjp 只在这个集合上调用。
    #     沿消费者方向从前向图 BFS 一次即可：某个值若不能从任何 ``wrt`` 沿前向
    #     走到，它的 cotangent 就永远不会影响被求导的值。这不是"梯度为零"，
    #     而是"不在任何一条通往目标的链上"——跳过它不改变结果，也不改变累加
    #     顺序：留下来的节点仍按原逆拓扑序处理，浮点加法逐位相同。
    downstream: Optional[set] = None
    if minimize:
        consumers: Dict[int, List[Value]] = {}
        for node in graph.nodes:
            for a in node.inputs:
                consumers.setdefault(a.id, []).append(node.output)
        downstream = set()
        frontier = [v.id for v in wrt]
        while frontier:
            vid = frontier.pop()
            for out in consumers.get(vid, ()):
                if out.id not in downstream:
                    downstream.add(out.id)
                    frontier.append(out.id)

    # 3) 多个输出指向同一个 Value 时，种子必须相加而不是后者覆盖前者。
    cot: Dict[int, Value] = {}
    for value, seed in seed_pairs:
        prev = cot.get(value.id)
        cot[value.id] = seed if prev is None else b.add_node("add", [prev, seed])

    # 4) 逆拓扑序传播 cotangent。
    #    ``wrt`` 里的中间值在这一步会被 ``pop`` 出去（它的生产节点要用这份
    #    cotangent），所以先留一份引用，收集时才取得到。对中间值求导仍然
    #    照常向它的祖先传播：反向图是一次构建好的，多算的那部分不影响返回
    #    值，也不改变调用方看到的结果。
    wanted = {v.id for v in wrt}
    saved: Dict[int, Value] = {}
    for node in reversed(graph.nodes):
        if downstream is not None and node.output.id not in downstream:
            continue  # 这个节点到不了任何求导目标：不求它的导数
        g = cot.pop(node.output.id, None)
        if g is None:
            continue  # 这个节点不在任何求导路径上
        if node.output.id in wanted:
            saved[node.output.id] = g
        if node.op.vjp is None:
            raise NoGradientError(
                f"算子 {node.op.name}（节点 %{node.output.name}）没有反向规则；"
                "如果它本应可微，请在 ir/grad_rules.py 里补一条 vjp"
            )
        grads = node.op.vjp(GradCtx(b=b, node=node, g=g, fwd=fwd))
        if len(grads) != len(node.inputs):
            raise RuntimeError(
                f"算子 {node.op.name} 的 vjp 返回了 {len(grads)} 个梯度，"
                f"但节点有 {len(node.inputs)} 个输入"
            )
        for inp, gi in zip(node.inputs, grads):
            if gi is None:
                continue
            prev = cot.get(inp.id)
            cot[inp.id] = gi if prev is None else b.add_node("add", [prev, gi])

    # 5) 收集结果。
    for v in wrt:
        g = saved.get(v.id)
        if g is None:
            g = cot.get(v.id)
        if g is None:
            raise NoGradientError(
                f"%{v.name} 不在任何激活输出（{list(active_aliases)}）的依赖路径上，"
                "没有梯度可返回。自动微分不会替你造一张零张量"
            )
        b.output(g, f"d{v.name}")
    return b
