"""IR 版本号、兼容策略与跨版本迁移。

版本号是 ``(主, 次)`` 二元组，规则只有三条：

- 主版本不同：拒绝。结构不兼容，猜就是错的。
- 次版本高于本实现：拒绝。图用了本实现还不认识的特性，按旧规则解读会静默出错。
- 次版本低于本实现：接受。新增字段一律可选，未知字段忽略。

"接受"不等于"读出来就是对的"：旧版本可能缺少新字段、或者某个属性的默认值
在新版本里变了。这类差异只能由迁移函数补齐，所以这里再提供一个显式的迁移
注册表：

- 每个源版本最多注册**一条**出边，路径因此唯一，不需要在分叉里挑。
- 目标必须严格高于源，迁移只升不降。
- 缺哪一段就报哪一段，并列出已注册的边；不允许"跳过中间版本直接猜"。

迁移作用于**序列化后的 JSON 结构**（``serialize.to_dict`` 的产物），不是
``Graph`` 对象：旧版本的图可能含有当前实现已经不认识的算子或属性，先重建
再迁移会在构图期失败。结构化数据则可以先改写再交给 ``from_dict``。

放在单独模块是为了避免 ``graph`` 与 ``serialize`` 互相导入。
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

IR_VERSION: Tuple[int, int] = (1, 0)

Version = Tuple[int, int]
Migration = Callable[[Dict[str, Any]], Dict[str, Any]]

# 源版本 -> (目标版本, 迁移函数)。源版本唯一，路径因此是链式的。
_MIGRATIONS: Dict[Version, Tuple[Version, Migration]] = {}


class IRVersionError(ValueError):
    """图的 IR 版本与本实现不兼容，或迁移路径缺失。"""


def _coerce_version(version: Any) -> Version:
    """把 ``[主, 次]`` 规范成二元组；格式不对抛 ``IRVersionError``。"""
    try:
        pair = tuple(int(x) for x in version)
    except Exception:
        raise IRVersionError(
            f"IR 版本必须是 [主, 次] 两个整数，得到 {version!r}"
        ) from None
    if len(pair) != 2:
        raise IRVersionError(
            f"IR 版本必须是 [主, 次] 两个整数，得到 {version!r}"
        )
    if pair[0] < 0 or pair[1] < 0:
        raise IRVersionError(f"IR 版本不能为负，得到 {version!r}")
    return pair


def check_version(version: Any) -> Tuple[int, int]:
    """校验版本号，兼容时返回 ``(主, 次)``，否则抛 ``IRVersionError``。"""
    major, minor = _coerce_version(version)

    want_major, want_minor = IR_VERSION
    if major != want_major:
        raise IRVersionError(
            f"图的 IR 主版本是 {major}.{minor}，本实现只支持 {want_major}.x："
            "主版本不同意味着结构不兼容，不做猜测性兼容"
        )
    if minor > want_minor:
        raise IRVersionError(
            f"图的 IR 版本 {major}.{minor} 高于本实现 {want_major}.{want_minor}："
            "它可能用到本版本还不认识的特性，拒绝按旧规则解读"
        )
    return (major, minor)


def version_text(version: Tuple[int, int] = IR_VERSION) -> str:
    return f"{version[0]}.{version[1]}"


# ---- 迁移注册表 ----------------------------------------------------------


def register_migration(
    source: Any,
    target: Any,
    migrate: Migration,
) -> Migration:
    """注册一条 ``source -> target`` 的迁移，返回函数本身（可当装饰器用）。

    迁移函数收到一份**深拷贝**的 JSON 结构，返回改写后的结构。返回同一个
    对象也可以；返回 ``None`` 或其他类型会被拒绝，避免"跑完没改"与"返回错
    类型"混在一起。
    """
    src = _coerce_version(source)
    dst = _coerce_version(target)
    if dst <= src:
        raise ValueError(
            f"迁移目标 {version_text(dst)} 必须严格高于源 {version_text(src)}："
            "迁移只升不降"
        )
    if not callable(migrate):
        raise TypeError(f"迁移必须是可调用对象，得到 {type(migrate).__name__}")
    if src in _MIGRATIONS:
        existing, _ = _MIGRATIONS[src]
        raise ValueError(
            f"源版本 {version_text(src)} 已经注册过到 {version_text(existing)} 的迁移："
            "每个源版本只允许一条出边，分叉会让迁移路径依赖注册顺序"
        )
    _MIGRATIONS[src] = (dst, migrate)
    return migrate


def unregister_migration(source: Any) -> Tuple[Version, Migration]:
    """移除一条迁移，返回 ``(目标版本, 函数)``；不存在时抛 ``KeyError``。"""
    src = _coerce_version(source)
    if src not in _MIGRATIONS:
        raise KeyError(f"源版本 {version_text(src)} 没有注册迁移")
    return _MIGRATIONS.pop(src)


def registered_migrations() -> Tuple[Tuple[Version, Version], ...]:
    """已注册的边，按源版本升序；用于自省与错误信息。"""
    return tuple(
        (src, _MIGRATIONS[src][0]) for src in sorted(_MIGRATIONS)
    )


def migration_path(source: Any, target: Any = None) -> Tuple[Version, ...]:
    """``source`` 到 ``target`` 的迁移路径（每条边的源版本序列）。

    版本相同时返回空元组。缺某一段时抛 ``IRVersionError``，消息里会指出
    缺的那一段与已注册的全部边。
    """
    src = _coerce_version(source)
    dst = IR_VERSION if target is None else _coerce_version(target)
    if dst == src:
        return ()
    if dst < src:
        raise IRVersionError(
            f"不能把 IR 从 {version_text(src)} 降级到 {version_text(dst)}："
            "迁移只升不降，降级需要另一套语义（旧实现不认识新字段）"
        )

    path: List[Version] = []
    current = src
    while current != dst:
        step = _MIGRATIONS.get(current)
        if step is None:
            known = ", ".join(
                f"{version_text(a)}→{version_text(b)}"
                for a, b in registered_migrations()
            )
            raise IRVersionError(
                f"IR 版本 {version_text(current)} 没有迁移到 {version_text(dst)} 的路径："
                f"缺 {version_text(current)} 的出边；已注册的迁移为 "
                f"[{known or '无'}]。请先用 register_migration 注册这一段，"
                "不要跳过中间版本直接猜"
            )
        nxt = step[0]
        if nxt > dst:
            raise IRVersionError(
                f"IR 版本 {version_text(current)} 的迁移目标是 {version_text(nxt)}，"
                f"越过了请求的 {version_text(dst)}：路径不能跳过终点"
            )
        path.append(current)
        current = nxt
    return tuple(path)


def migrate(data: Dict[str, Any], target: Any = None) -> Dict[str, Any]:
    """把一份序列化结构迁移到 ``target``（默认当前 ``IR_VERSION``）。

    版本已经相同时原样返回入参，不做拷贝——正常路径（读当前版本的图）因此
    没有额外开销。需要迁移时才深拷贝，迁移函数可以放心就地改写。
    """
    if not isinstance(data, dict):
        raise TypeError(f"migrate 需要序列化后的 dict，得到 {type(data).__name__}")
    src = check_version(data.get("ir_version"))
    dst = IR_VERSION if target is None else _coerce_version(target)
    path = migration_path(src, dst)
    if not path:
        return data

    current = copy.deepcopy(data)
    for step in path:
        nxt, fn = _MIGRATIONS[step]
        result = fn(current)
        if not isinstance(result, dict):
            raise TypeError(
                f"迁移 {version_text(step)}→{version_text(nxt)} 返回了 "
                f"{type(result).__name__}，必须是 dict"
            )
        current = result
    current["ir_version"] = list(dst)
    return current
