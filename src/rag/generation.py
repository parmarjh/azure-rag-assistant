from __future__ import annotations

import re

from .models import Citation, Retrieved, Usage


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def generate(question: str, items: list[Retrieved], allow_general: bool = False):
    if not items:
        return "I don't have that information in the knowledge base.", [], Usage()
    qwords = set(re.findall(r"[a-z0-9$]+", question.lower()))
    candidates: list[tuple[float, str, Retrieved]] = []
    for item in items:
        for sentence in _sentences(item.content):
            words = set(re.findall(r"[a-z0-9$]+", sentence.lower()))
            overlap = len(qwords & words)
            if overlap:
                # Prefer a sentence that contains the query's distinctive entities.
                exact = overlap / max(1, len(qwords))
                candidates.append((exact + item.score * 0.2, sentence, item))
    candidates.sort(key=lambda x: (-x[0], x[2].chunk_id, x[1]))
    selected = [(sentence, item) for _, sentence, item in candidates[:5]]
    if not selected:
        selected = [(_sentences(x.content)[0], x) for x in items if _sentences(x.content)]
    citations, parts, seen = [], [], {}
    for sentence, item in selected:
        if item.chunk_id not in seen:
            seen[item.chunk_id] = len(citations) + 1
            c = item.chunk
            citations.append(Citation(c.chunk_id, c.doc_id, c.filename, c.doc_title, c.section_path, c.page))
        parts.append(f"{sentence} [{seen[item.chunk_id]}]")
    text = " ".join(parts)
    tokens = len(text.split())
    return text, citations, Usage(prompt_tokens=len(question.split()) + sum(x.chunk.token_count for x in items),
                                  completion_tokens=tokens, estimated_cost_usd=0.0)
