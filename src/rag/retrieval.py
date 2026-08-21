from __future__ import annotations

from collections import defaultdict

from .context import assemble
from .models import Retrieved
from .rerank import rerank


def reciprocal_rank_fusion(result_lists: list[list[Retrieved]], k: int = 60) -> list[Retrieved]:
    """Fuse independently searched query lists with deterministic RRF scores."""
    scores = defaultdict(float)
    values = {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            scores[item.chunk_id] += 1.0 / (k + rank)
            values[item.chunk_id] = item
    fused = [Retrieved(values[chunk_id].chunk, score)
             for chunk_id, score in scores.items()]
    return sorted(fused, key=lambda x: (-x.score, x.chunk_id))


def retrieve(index, embedder, query: str, filters: dict, top_k: int = 8,
             threshold: float = 0.1, token_budget: int = 3000,
             per_document_cap: int = 3) -> list[Retrieved]:
    """Run hybrid search, fusion, reranking, and context assembly."""
    vector = embedder.embed([query])[0]
    text_results = index.search(query, None, filters, 30)
    vector_results = index.search("", vector, filters, 30)
    fused = reciprocal_rank_fusion([text_results, vector_results])
    return assemble(rerank(query, fused, threshold), token_budget, per_document_cap)[:top_k]
