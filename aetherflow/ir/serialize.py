"""IR 的 JSON 序列化。

为什么需要它：没有序列化，图永远由当前代码在进程内构造，"IR 版本号"就是
死代码。序列化让图能离开进程交给第三方后端，版本检查才真正起作用——
这是 ADR-001（IR 是唯一契约）从口号变成可交付物的一步。

格式刻意保持扁平、只用 JSON 原生类型，第三方实现不必依赖本项目的任何类：

    {
      "ir_version": [1, 0],
      "name": "block",
      "inputs": [{"name": "x", "type": {"dtype": "f32", "shape": [2, "B", 32]}}],
      "nodes": [
        {"op": "matmul", "inputs": ["x", "w"],
         "attrs": {}, "output": {"dtype": "f32", "shape": [2, "B", 32]}}
      ],
      "outputs": [{"value": "v5", "name": "y"}]
    }

节点只按名字引用前序值（SSA 名由构图顺序决定，读回来是确定的），
每个节点带上输出类型，读的时候与本实现的推导结果对照——不一致就说明
图作者与实现的语义已经分叉，必须立刻失败而不是继续跑。

读回来的顺序是固定的：先重建 ``inputs`` 里的全部输入，再按存档顺序重放
``nodes``。构图却允许输入与节点交错（``Graph.input`` 可以在任何时刻调用），
两者能对上是因为**节点输出带显式名字**（``nodes[].name``）：输入先重建只
改变了值 id 的分配顺序，引用全靠名字解析，不会指向另一条边。值 id 不进
存档，回读后可能与构图时不同，对外的契约是名字与类型。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

from .graph import Graph
from .types import DType, SymExpr, TensorType
from .version import IR_VERSION, IRVersionError, check_version, migrate

__all__ = [
    "IRVersionError",
    "to_dict",
    "from_dict",
    "dumps",
    "loads",
]


# ---- 类型 ----------------------------------------------------------------


def _dim_to_json(dim) -> Any:
    """形状维编码。``int`` / 裸符号原样保留，表达式展开成 dict。"""
    if isinstance(dim, SymExpr):
        return {"expr": dim.op, "args": [_dim_to_json(a) for a in dim.args]}
    return dim


def _dim_from_json(data) -> Any:
    if isinstance(data, dict):
        return SymExpr(str(data["expr"]), tuple(_dim_from_json(a) for a in data["args"]))
    return data


def _type_to_dict(type_: TensorType) -> Dict[str, Any]:
    return {"dtype": type_.dtype.value, "shape": [_dim_to_json(d) for d in type_.shape]}


def _type_from_dict(data: Dict[str, Any]) -> TensorType:
    return TensorType(DType(data["dtype"]), tuple(_dim_from_json(d) for d in data["shape"]))


# ---- 属性 ----------------------------------------------------------------
# JSON 没有 tuple。构图时属性多为 tuple（如 perm / shape / axes），
# 读回来统一还原成 tuple，保证"写出去再读回来"与原件逐字段相等。


def _to_json(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_to_json(x) for x in value]
    # 按 Mapping 而不是 dict 判断：seal 之后的图把属性换成了只读映射
    # （MappingProxyType），它不是 dict 的子类，但导出必须照常工作。
    if isinstance(value, Mapping):
        return {k: _to_json(v) for k, v in value.items()}
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_from_json(x) for x in value)
    if isinstance(value, dict):
        return {k: _from_json(v) for k, v in value.items()}
    return value


# ---- 序列化 --------------------------------------------------------------


def to_dict(graph: Graph) -> Dict[str, Any]:
    """把图导出成纯 JSON 结构。"""
    return {
        "ir_version": list(graph.ir_version),
        "name": graph.name,
        "inputs": [
            {"name": v.name, "type": _type_to_dict(v.type)} for v in graph.inputs
        ],
        "nodes": [
            {
                "op": n.op.name,
                "inputs": [a.name for a in n.inputs],
                "attrs": _to_json(n.attrs),
                # 输出名必须显式写出来：图允许"先建节点、后声明输入"（eager
                # 前端会这样长大），重放时"当前值个数"与构图时不同，靠自动
                # 编号会得到另一串名字。
                "name": n.output.name,
                "output": _type_to_dict(n.output.type),
            }
            for n in graph.nodes
        ],
        "outputs": [
            {"value": v.name, "name": alias}
            for v, alias in zip(graph.outputs, graph.output_names)
        ],
    }


def from_dict(data: Dict[str, Any]) -> Graph:
    """从 JSON 结构还原图。版本不兼容、算子未知、类型对不上都会明确失败。

    次版本低于当前实现时先走 ``version.migrate``：迁移在**结构**上做，而不是
    先重建再改图——旧图可能含当前实现不认识的算子，重建会在构图期失败。
    迁移过就记下来源版本（``Graph.migrated_from``），否则为 ``None``。
    """
    source_version = check_version(data.get("ir_version"))
    data = migrate(data)
    version = check_version(data.get("ir_version"))

    graph = Graph(data.get("name", "main"))
    if source_version != version:
        graph.migrated_from = source_version
    for spec in data.get("inputs", []):
        graph.input(spec["name"], _type_from_dict(spec["type"]))

    for spec in data.get("nodes", []):
        try:
            args = [graph.find(name) for name in spec["inputs"]]
        except KeyError as exc:
            raise ValueError(f"节点引用了未定义的值 {exc}") from None
        out = graph.add_node(
            spec["op"],
            args,
            _from_json(spec.get("attrs", {})),
            name=spec.get("name"),
        )
        declared = _type_from_dict(spec["output"])
        if out.type != declared:
            raise ValueError(
                f"节点 {spec['op']} 的声明类型 {declared} 与本实现推导的 {out.type} 不一致："
                "图作者与实现的算子语义已经分叉"
            )

    for spec in data.get("outputs", []):
        try:
            value = graph.find(spec["value"])
        except KeyError as exc:
            raise ValueError(f"图输出引用了未定义的值 {exc}") from None
        alias = spec.get("name")
        graph.output(value, None if alias == value.name else alias)

    graph.ir_version = version
    return graph


def dumps(graph: Graph, indent: int = 2) -> str:
    return json.dumps(to_dict(graph), ensure_ascii=False, indent=indent)


def loads(text: str) -> Graph:
    return from_dict(json.loads(text))
