from __future__ import annotations

import re

STOP_WORDS = {
    "a", "an", "and", "are", "do", "for", "get", "how", "i", "in", "is",
    "many", "of", "our", "per", "the", "this", "to", "what", "which",
}


def scrub(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    return re.sub(r"(?i)(ignore (?:all )?previous instructions|system prompt)", "[redacted]", text)


def has_sufficient_evidence(query: str, items) -> bool:
    terms = {x for x in re.findall(r"[a-z0-9$]+", query.lower()) if x not in STOP_WORDS}
    context = " ".join(x.content.lower() for x in items)
    if not terms:
        return False
    # A query's distinctive head nouns must occur in retrieved evidence. This
    # prevents generic enterprise headers from making an unrelated answer look grounded.
    distinctive = {x for x in terms if len(x) > 4}
    context_terms = set(re.findall(r"[a-z0-9$]+", context))
    matched = len(distinctive & context_terms)
    return matched / max(1, len(distinctive)) >= 0.6


def confidence(query: str, items) -> float:
    if not items:
        return 0.0
    terms = set(re.findall(r"[a-z0-9$]+", query.lower()))
    context = " ".join(x.content.lower() for x in items)
    coverage = len([x for x in terms if x in context]) / max(1, len(terms))
    top = items[0].score
    margin = max(0.0, top - (items[2].score if len(items) > 2 else 0.0))
    return max(0.0, min(1.0, 0.55 * top + 0.25 * margin + 0.20 * coverage))
