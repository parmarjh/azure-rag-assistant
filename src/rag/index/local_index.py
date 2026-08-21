from __future__ import annotations

import math
import re
from collections import Counter

from ..models import Chunk, Retrieved


class LocalSearchIndex:
    def __init__(self, embedder):
        self.embedder, self.chunks = embedder, []
        self._idf: dict[str, float] = {}
        self._doc_terms: list[Counter[str]] = []
        self._avgdl = 0.0

    def create(self):
        self.chunks = []
        self._idf, self._doc_terms, self._avgdl = {}, [], 0.0

    def upload(self, chunks):
        self.chunks.extend(chunks)
        self._doc_terms = [Counter(self._terms(c.content)) for c in self.chunks]
        document_frequency = Counter()
        for terms in self._doc_terms:
            document_frequency.update(terms.keys())
        total = len(self.chunks)
        self._idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._avgdl = sum(sum(terms.values()) for terms in self._doc_terms) / max(1, total)

    @staticmethod
    def _terms(text: str) -> list[str]:
        return re.findall(r"[a-z0-9$%]+", text.lower())

    def _bm25(self, query: str, index: int) -> float:
        query_terms = self._terms(query)
        if not query_terms or not self._doc_terms:
            return 0.0
        terms = self._doc_terms[index]
        length = sum(terms.values())
        score = 0.0
        for term in query_terms:
            frequency = terms.get(term, 0)
            if not frequency:
                continue
            numerator = frequency * 2.2
            denominator = frequency + 1.2 * (
                1 - 0.75 + 0.75 * length / max(1.0, self._avgdl))
            score += self._idf.get(term, 0.0) * numerator / denominator
        return score

    def _allowed(self, c: Chunk, filters: dict) -> bool:
        if filters.get("is_current") and not c.is_current:
            return False
        if filters.get("department") and c.department.lower() != filters["department"].lower():
            return False
        groups = filters.get("groups")
        return not groups or bool(set(c.security_groups) & set(groups))

    def search(self, query, vector, filters, top_k):
        scored = []
        for index, c in enumerate(self.chunks):
            if not self._allowed(c, filters):
                continue
            score = sum(a * b for a, b in zip(vector, c.embedding)) if vector and c.embedding else 0.0
            if query:
                score = self._bm25(query, index)
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [Retrieved(c, float(s)) for s, c in scored[:top_k]]
