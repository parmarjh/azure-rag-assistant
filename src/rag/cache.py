from __future__ import annotations

import hashlib
import time


class TTLCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._data: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        item = self._data.get(key)
        if not item or time.time() - item[0] > self.ttl:
            return None
        return item[1]

    def set(self, key: str, value: object):
        self._data[key] = (time.time(), value)


def cache_key(question: str, filters: dict, mode: str, groups: list[str]) -> str:
    raw = f"{question.strip().lower()}|{sorted(filters.items())}|{mode}|{sorted(groups)}"
    return hashlib.sha256(raw.encode()).hexdigest()
