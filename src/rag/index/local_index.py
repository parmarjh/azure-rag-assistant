from __future__ import annotations

import re

from ..models import Chunk, Retrieved


class LocalSearchIndex:
    def __init__(self, embedder):
        self.embedder, self.chunks = embedder, []

    def create(self): self.chunks = []
    def upload(self, chunks): self.chunks.extend(chunks)

    def _allowed(self, c: Chunk, filters: dict) -> bool:
        if filters.get("is_current") and not c.is_current:
            return False
        if filters.get("department") and c.department.lower() != filters["department"].lower():
            return False
        groups = filters.get("groups")
        return not groups or bool(set(c.security_groups) & set(groups))

    def search(self, query, vector, filters, top_k):
        qwords = set(re.findall(r"[a-z0-9$]+", query.lower()))
        scored = []
        for c in self.chunks:
            if not self._allowed(c, filters):
                continue
            words = set(re.findall(r"[a-z0-9$]+", c.content.lower()))
            text_score = len(qwords & words) / max(1, len(qwords))
            vec_score = 0.0
            if vector and c.embedding:
                vec_score = sum(a*b for a,b in zip(vector, c.embedding))
            # Hybrid local scoring; baseline passes a vector-only marker.
            score = vec_score if filters.get("_baseline") else 0.65 * text_score + 0.35 * vec_score
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [Retrieved(c, float(s)) for s, c in scored[:top_k]]
