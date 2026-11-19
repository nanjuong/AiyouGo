"""参数压缩重写：把稠密 f32 权重输入换成"压缩表示 + 图内展开"。

训练大模型时参数内存的大头不是中间激活，而是 **参数本身 + 优化器槽位**
（Adam 每参数多 2 份 f32）。这个 pass 把目标参数输入改写为压缩形态，
让"完整稠密 f32 参数缓冲"在主机与显存里**从此不存在**——展开只发生在
消费它的算子内部（GPU 后端可融合进 kernel；参考后端里它是执行器的瞬时
中间值，按计划逐组释放，不进任何长期缓冲）。

两种模式（按参数逐个选择）：

- ``low_rank``：``W ≈ A @ B``，A 是 (m, r) f32、B 是 (r, n) f32。原输入
  整个删除，可训练量 (m+n)·r，优化器槽位同比例缩小。
- ``qlora``：冻结的 int8 底 ``base``（1 字节/元素）+ 每列 scale（f32）
  + 低秩适配器 ``A/B``（f32）。只有适配器可训练。底权重**永远以 int8
  驻留**，dequant 写成 ``cast`` + ``mul`` 两个 IR 节点。

新输入命名：``<p>__A`` / ``<p>__B``（可训练因子）、``<p>__base``（int8
冻结底）、``<p>__scale``（每列 scale，冻结）。分隔符用 ``__`` 而不是
``/``：图输入要能直接当 feed 键，也要过序列化的 JSON 字段名。

**只改写被 matmul 直接消费的参数。** 判据是"消费方式确定"——layernorm
的 gamma/beta 本来就一维、embedding 表的压缩有另一套语义（哈希共享而非
低秩），把它们混进同一条 pass 会让"哪些参数被悄悄改了形态"重新变成一件
需要读源码才能知道的事。不匹配的参数原样保留，并在 :func:`plan_param_compression`
里当场报错，而不是构图时留一个间接错误。

**数值判据。** 改写前后的前向差只应来自被丢弃的奇异值（low_rank）或量化
格点（qlora），与手算的 SVD 截断界 / 每列量化界一致；梯度只流向可训练因子，
冻结输入不进 ``backward`` 的 ``wrt`` 就不产生任何 vjp 成本。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from ..ir.graph import Graph, Value
from ..ir.types import f32, i8
from .base import Pass

__all__ = ["ParamCompression", "plan_param_compression", "MODES"]

MODES = ("low_rank", "qlora")


def _is_param(graph: Graph, value: Value) -> bool:
    """这个操作数是图输入（参数），不是中间值。"""
    return value.producer is None and any(v.id == value.id for v in graph.inputs)


def compressed_inputs(graph: Graph) -> Dict[str, str]:
    """新图里"哪些原始参数被压了、压成什么模式"，从输入名后缀反推。"""
    found: Dict[str, str] = {}
    for value in graph.inputs:
        name = value.name
        if name.endswith("__A"):
            base = name[: -len("__A")]
            found.setdefault(base, "low_rank")
        elif name.endswith("__base"):
            found[name[: -len("__base")]] = "qlora"
    return found


class ParamCompression(Pass):
    """按 policy 改写参数输入；产出的图里被压参数不再以 f32 稠密出现。

    ``policy`` 是"参数名 -> (mode, rank)"。未列出的参数一律原样保留——
    不做"自动检测该压哪个"，那是调优层的事。
    """

    name = "param-compress"

    def __init__(self, policy: Mapping[str, Tuple[str, int]]) -> None:
        self.policy: Dict[str, Tuple[str, int]] = {}
        for pname, (mode, rank) in dict(policy).items():
            if mode not in MODES:
                raise ValueError(
                    f"参数 {pname!r} 的模式 {mode!r} 不认识；用 {list(MODES)}"
                )
            if int(rank) < 1:
                raise ValueError(f"参数 {pname!r} 的秩必须 >= 1，得到 {rank}")
            self.policy[pname] = (mode, int(rank))

    def run(self, graph: Graph) -> Graph:
        targets = {v.name: v for v in graph.inputs if v.name in self.policy}
        if not targets:
            raise KeyError(
                f"policy 里的名字在这张图的输入里一个都不存在: {sorted(self.policy)}"
            )

        # 消费检查：每个目标参数必须**只**出现在 matmul 的权重位上。
        # 出现在别的地方（加法、逐元素乘）意味着低秩替换会改变语义。
        consumers: Dict[str, List[Tuple[object, int]]] = {n: [] for n in targets}
        for node in graph.nodes:
            for slot, arg in enumerate(node.inputs):
                if arg.producer is None and arg.name in consumers:
                    consumers[arg.name].append((node, slot))
        for pname, uses in consumers.items():
            for node, slot in uses:
                if node.op.name != "matmul":
                    raise ValueError(
                        f"参数 {pname} 被 {node.op.name} 消费；本 pass 只改写 "
                        "matmul 的权重操作数（低秩替换在别的消费方式下不保语义）"
                    )
                shape = targets[pname].type.shape
                if len(shape) != 2:
                    raise ValueError(
                        f"参数 {pname} 的形状 {tuple(shape)} 不是二维，无法低秩分解"
                    )

        out = Graph(graph.name)
        mapping: Dict[int, Value] = {}
        parts: Dict[str, Dict[str, Value]] = {}

        for value in graph.inputs:
            if value.name not in targets:
                mapping[value.id] = out.input(value.name, value.type)
                continue
            mode, rank = self.policy[value.name]
            m, n = (int(d) for d in value.type.shape)
            if rank > min(m, n):
                raise ValueError(
                    f"参数 {value.name} 的秩 {rank} 超过形状 {m}x{n} 的短边"
                )
            p: Dict[str, Value] = {}
            if mode == "qlora":
                p["base"] = out.input(f"{value.name}__base", i8[m, n])
                p["scale"] = out.input(f"{value.name}__scale", f32[n])
            p["A"] = out.input(f"{value.name}__A", f32[m, rank])
            p["B"] = out.input(f"{value.name}__B", f32[rank, n])
            parts[value.name] = p

        def expand(x: Value, pname: str) -> Value:
            """把"数据 @ W"重写成压缩形态；W 永不以稠密 f32 存在。"""
            pp = parts[pname]
            mode = self.policy[pname][0]
            if mode == "qlora":
                dense = out.mul(
                    out.add_node("cast", [pp["base"]], {"dtype": "f32"}),
                    pp["scale"],
                )
                return out.add(out.matmul(x, dense), out.matmul(out.matmul(x, pp["A"]), pp["B"]))
            return out.matmul(out.matmul(x, pp["A"]), pp["B"])

        for node in graph.nodes:
            hit = [
                (slot, arg.name)
                for slot, arg in enumerate(node.inputs)
                if arg.producer is None and arg.name in parts
            ]
            if node.op.name == "matmul" and hit:
                if len(hit) != 1:
                    raise ValueError(
                        "一个 matmul 里同时出现多个被压缩的操作数；"
                        "两侧都是参数时无法判定谁是数据、谁是权重"
                    )
                slot, pname = hit[0]
                if slot != 1:
                    raise NotImplementedError(
                        "首版只支持权重在右侧（x @ W）；W @ x 形态需要转置重写"
                    )
                x = mapping[node.inputs[0].id]
                y = expand(x, pname)
                mapping[node.output.id] = y
                continue

            args: List[Value] = []
            for a in node.inputs:
                args.append(mapping[a.id] if a.id in mapping else a)
            mapping[node.output.id] = out.add_node(node.op.name, args, node.attrs)

        for value, name in zip(graph.outputs, graph.output_names):
            out.output(mapping[value.id], name)
        return out


def plan_param_compression(
    graph: Graph,
    *,
    ranks: Mapping[str, int],
    keep_base: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Tuple[str, int]]:
    """把"哪些参数、秩多少、留不留 int8 底"变成 policy，并当场筛非法项。

    留底 = ``qlora``（int8 冻结底 + 可训练适配器）；不留 = ``low_rank``
    （因子完全替代原参数）。名字打错、参数不是 matmul 权重，都在这里报错。
    """
    names = {v.name for v in graph.inputs}
    unknown = sorted(set(ranks) - names)
    if unknown:
        raise KeyError(f"这些名字不是图输入: {unknown}")

    # 消费方式预检：与 run() 同一条判据，早报错比晚报错好用
    used: Dict[str, List[str]] = {}
    for node in graph.nodes:
        for arg in node.inputs:
            if arg.producer is None and arg.name in ranks:
                used.setdefault(arg.name, []).append(node.op.name)
    bad = {n: sorted(set(ops)) for n, ops in used.items() if set(ops) - {"matmul"}}
    if bad:
        raise ValueError(
            "这些参数不是只被 matmul 消费的权重，低秩替换不保语义: "
            + "; ".join(f"{n} -> {ops}" for n, ops in sorted(bad.items()))
        )
    missing = sorted(set(ranks) - set(used))
    if missing:
        raise ValueError(f"这些参数没有被任何算子消费（图里没用到）: {missing}")

    keep = dict(keep_base or {})
    return {
        pname: ("qlora" if keep.get(pname, False) else "low_rank", int(rank))
        for pname, rank in ranks.items()
    }
