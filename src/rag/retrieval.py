from __future__ import annotations

from collections import defaultdict

from .context import assemble
from .models import Retrieved
from .rerank import rerank
from .telemetry import Span


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
             per_document_cap: int = 3, candidate_k: int = 30,
             use_rerank: bool = True, neighbour_expansion: bool = True,
             subqueries: list[str] | None = None,
             timings: dict[str, float] | None = None) -> list[Retrieved]:
    """Run hybrid search, fusion, reranking, neighbour expansion and assembly."""
    queries = subqueries or [query]
    lists: list[list[Retrieved]] = []
    query_result_scores: list[dict[str, float]] = []
    with Span("search") as search_span:
        for subquery in queries:
            vector = embedder.embed([subquery])[0]
            text_results = index.search(subquery, None, filters, candidate_k)
            vector_results = index.search("", vector, filters, candidate_k)
            lists.extend((text_results, vector_results))
            query_result_scores.append({
                item.chunk_id: max(
                    score for score in (
                        next(
                            (
                                result.score
                                for result in text_results
                                if result.chunk_id == item.chunk_id
                            ),
                            0.0,
                        ),
                        next(
                            (
                                result.score
                                for result in vector_results
                                if result.chunk_id == item.chunk_id
                            ),
                            0.0,
                        ),
                    )
                )
                for item in (*text_results, *vector_results)
            })
    if timings is not None:
        timings["search"] = search_span.elapsed_ms
    fused = reciprocal_rank_fusion(lists)
    with Span("rerank") as rerank_span:
        ranked = (
            rerank(query, fused, threshold, index=index, embedder=embedder)
            if use_rerank else fused
        )
    if timings is not None:
        timings["rerank"] = rerank_span.elapsed_ms
    if len(queries) > 1:
        required: list[Retrieved] = []
        for result_scores in query_result_scores:
            for best_id in sorted(
                result_scores, key=result_scores.get, reverse=True
            )[:2]:
                match = next(
                    (
                        item for item in ranked
                        if item.chunk_id == best_id
                    ),
                    None,
                )
                if match is not None and match.chunk_id not in {
                    item.chunk_id for item in required
                }:
                    required.append(match)
        if required:
            required_ids = {item.chunk_id for item in required}
            ranked = required + [
                item for item in ranked if item.chunk_id not in required_ids
            ]
    else:
        required = []
    if neighbour_expansion and hasattr(index, "chunks"):
        selected = {item.chunk_id for item in ranked[:top_k]}
        expanded = list(ranked)
        for item in ranked[:top_k]:
            for neighbour in index.chunks:
                same_section = (
                    neighbour.doc_id == item.doc_id
                    and neighbour.section_path == item.section_path
                    and abs(neighbour.chunk_index - item.chunk.chunk_index) == 1
                    and neighbour.chunk_id not in selected
                )
                if same_section:
                    expanded.append(Retrieved(neighbour, item.score * 0.98))
                    selected.add(neighbour.chunk_id)
        expanded = sorted(expanded, key=lambda x: (-x.score, x.chunk_id))
        if required:
            required_ids = {item.chunk_id for item in required}
            ranked = required + [
                item for item in expanded if item.chunk_id not in required_ids
            ]
        else:
            ranked = expanded
    return assemble(ranked, token_budget, per_document_cap)[:top_k]
