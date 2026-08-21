from __future__ import annotations

from .models import Retrieved


def assemble(items: list[Retrieved], token_budget: int = 3000, per_document_cap: int = 3) -> list[Retrieved]:
    result, counts, used = [], {}, 0
    for item in items:
        count = counts.get(item.doc_id, 0)
        if count >= per_document_cap or used + item.chunk.token_count > token_budget:
            continue
        result.append(item)
        counts[item.doc_id] = count + 1
        used += item.chunk.token_count
    return result
