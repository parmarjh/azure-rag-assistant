from __future__ import annotations

import re

from .models import Retrieved


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%]+", text.lower())


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def rerank(
    query: str,
    items: list[Retrieved],
    threshold: float = 0.1,
    index=None,
    embedder=None,
) -> list[Retrieved]:
    terms = list(dict.fromkeys(_terms(query)))
    query_vector = embedder.embed([query])[0] if embedder is not None else []
    bm25_values = [
        index.bm25_score(query, item.chunk) if hasattr(index, "bm25_score")
        else sum(term in _terms(item.content) for term in terms)
        for item in items
    ]
    cosine_values = [
        index.vector_score(query_vector, item.chunk) if hasattr(index, "vector_score")
        else item.score
        for item in items
    ]
    bm25_values = _normalise(bm25_values)
    cosine_values = _normalise(cosine_values)
    ranked = []
    for item in items:
        position = len(ranked)
        words = set(_terms(item.content))
        weighted_total = weighted_match = 0.0
        for term in terms:
            weight = index.idf(term) if hasattr(index, "idf") else 1.0
            weight = max(weight, 0.1)
            weighted_total += weight
            if term in words or term[:6] in {word[:6] for word in words if len(word) >= 6}:
                weighted_match += weight
        coverage = weighted_match / max(weighted_total, 0.1)
        score = (
            0.50 * bm25_values[position]
            + 0.25 * cosine_values[position]
            + 0.25 * coverage
        )
        ranked.append(Retrieved(item.chunk, score))
    ranked.sort(key=lambda x: (-x.score, x.chunk_id))
    return [x for x in ranked if x.score >= threshold]
