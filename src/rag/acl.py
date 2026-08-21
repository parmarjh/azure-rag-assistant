from __future__ import annotations

from .models import Chunk, UserContext


def department_groups(department: str) -> list[str]:
    from .config import get_config
    return list(get_config().acl_map.get(department, [department.lower()]))


def allowed(chunk: Chunk, user: UserContext | None) -> bool:
    if user is None:
        return True
    return bool(set(chunk.security_groups) & set(user.groups))


def build_filter(user: UserContext | None, department: str | None = None,
                 current_only: bool = True) -> dict:
    from .config import get_config
    groups = user.groups if user else sorted({
        group for values in get_config().acl_map.values() for group in values
    })
    return {"groups": groups, "department": department, "is_current": current_only}
