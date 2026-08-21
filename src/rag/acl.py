from __future__ import annotations

from .models import Chunk, UserContext

DEPARTMENT_GROUPS = {"HR": ["hr", "all-staff"], "Finance": ["finance", "all-staff"],
                     "IT": ["it", "all-staff"], "Legal": ["legal", "all-staff"],
                     "Sales": ["sales", "all-staff"]}


def allowed(chunk: Chunk, user: UserContext | None) -> bool:
    if user is None:
        return True
    return bool(set(chunk.security_groups) & set(user.groups))


def build_filter(user: UserContext | None, department: str | None = None,
                 current_only: bool = True) -> dict:
    groups = user.groups if user else ["all-staff", "hr", "finance", "it", "legal", "sales"]
    return {"groups": groups, "department": department, "is_current": current_only}
