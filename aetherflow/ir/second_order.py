"""对反向图再求导：二阶导数（Hessian-vector product）的显式入口。

为什么单列一个模块而不是让调用方自己 ``backward(backward(...))``
------------------------------------------------------------------
一阶反向图的输出名是 ``d<wrt 名>``、输入是"前向输入 + 一阶种子
``d<输出别名>``"。要对它再求导，得自己在一阶图里按名字找回 ``wrt`` 对应的
那个 ``Value``、再把二阶种子名对齐——错一步就得到一张"对 dy 求导"的图，
形状可能还正好对得上，于是静默算错。这里把这段映射固定下来，并显式声明
二阶图的三组输入。

二阶图的输入与输出
------------------
输入 = 前向图的全部输入 + 每个激活输出的种子 ``d<别名>``（一阶反向往回算
要用）+ 每个 ``wrt`` 的二阶种子 ``dd<wrt 名>``（向量 ``v``）。
输出 = 每个 ``wrt`` 一项 ``d<wrt 名>``，即 ``(H · v)`` 的对应分量。

因此调用方喂的是"两次前向输入"？不是：前向重放只需要一份输入，一阶节点
与二阶节点共用同一份图输入；一阶种子与二阶种子各一组。

与一阶一样的边界
----------------
- **曲率为零不会得到零张量，而是 ``NoGradientError``。** ``f(x) = sum(x*w)``
  对 ``x`` 的一阶梯度不含 ``x``，二阶反向时 ``x`` 不在任何激活输出的依赖
  路径上；IR 里没有"无来源的零"，自动微分也不替调用方造。要二阶为零的
  语义就显式把零写进图里。
- **算子必须自带二阶规则。** 一阶反向图里出现的 ``conv2d_input_grad`` /
  ``conv2d_weight_grad`` / ``maxpool2d_backward`` 等算子本身没有 vjp，
  对它们再求导会得到 ``NoGradientError`` 并指出是哪个算子——这不是"二阶
  为零"，是"这里没有定义"。
- 与一阶相同：广播相关的反向要求静态形状，符号维在那一处显式报错。
"""

from __future__ import annotations

from typing import Optional, Sequence

from .graph import Graph, Value
from .grad import backward

__all__ = ["hessian_vector_product"]


def hessian_vector_product(
    graph: Graph,
    wrt: Optional[Sequence[Value]] = None,
    *,
    active: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
) -> Graph:
    """返回 ``graph`` 的 Hessian-vector product 图。

    参数
    ----
    graph
        前向图。
    wrt
        对哪些图输入求二阶导，默认全部图输入。顺序决定输出顺序。
    active
        参与一阶反向的输出别名，默认全部；语义与 :func:`backward` 一致
        （未激活输出在构图期被真正剪掉，不是喂零种子）。
    seeds
        二阶种子输入名，按 ``wrt`` 顺序给出，默认为 ``dd<wrt 名>``。
        与已有名字冲突时用这个参数改名。
    name
        二阶图的名字，默认为 ``<graph.name>_hvp``。

    返回图的两个"种子"输入都需要调用方自己喂：一阶种子
    ``d<输出别名>``（通常是与输出同形的全 1，等价于对输出求和）与二阶
    种子 ``dd<wrt 名>``（向量 ``v``）。
    """
    if wrt is None:
        wrt = graph.inputs
    wrt = tuple(wrt)
    if not wrt:
        raise ValueError("wrt 不能为空：没有求导目标就没有二阶导数")

    if seeds is None:
        second_names = [f"dd{v.name}" for v in wrt]
    else:
        second_names = list(seeds)
        if len(second_names) != len(wrt):
            raise ValueError(
                f"seeds 有 {len(second_names)} 个名字，但 wrt 有 {len(wrt)} 项"
            )
        if len(set(second_names)) != len(second_names):
            raise ValueError(f"seeds 里有重名：{second_names}")

    # 一阶反向图。名字固定成 ``<graph>_backward``：它只是中间产物。
    first = backward(graph, wrt=wrt, active=active)

    # 在一阶图里按名字找回 wrt 对应的 Value。backward 用前向输入名声明
    # 一阶图的输入，因此名字是一一对应的。
    by_name = {value.name: value for value in first.inputs}
    targets = []
    for value in wrt:
        target = by_name.get(value.name)
        if target is None:
            raise RuntimeError(
                f"一阶反向图里没有名为 {value.name!r} 的输入；"
                "这是 backward 的输入命名约定被改动了，不是调用方的问题"
            )
        targets.append(target)

    seed_map = {f"d{value.name}": seed for value, seed in zip(wrt, second_names)}
    return backward(
        first,
        wrt=targets,
        seeds=seed_map,
        name=name or f"{graph.name}_hvp",
    )
