from __future__ import annotations

import re

from .models import Retrieved


def rerank(query: str, items: list[Retrieved], threshold: float = 0.1) -> list[Retrieved]:
    terms = set(re.findall(r"[a-z0-9$]+", query.lower()))
    ranked = []
    for item in items:
        words = set(re.findall(r"[a-z0-9$]+", item.content.lower()))
        exact = len(terms & words) / max(1, len(terms))
        score = min(1.0, item.score * 0.45 + exact * 0.55)
        ranked.append(Retrieved(item.chunk, score))
    ranked.sort(key=lambda x: (-x.score, x.chunk_id))
    return [x for x in ranked if x.score >= threshold]
