from __future__ import annotations

import re

from .generation import body_text

STOP_WORDS = {
    "a", "an", "and", "are", "do", "for", "get", "how", "i", "in", "is",
    "many", "of", "our", "per", "the", "this", "to", "what", "which",
}


def scrub(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    return re.sub(r"(?i)(ignore (?:all )?previous instructions|system prompt)", "[redacted]", text)


def has_sufficient_evidence(query: str, items) -> bool:
    terms = {x for x in re.findall(r"[a-z0-9$]+", query.lower()) if x not in STOP_WORDS}
    context = " ".join(
        f"{body_text(x)} {x.filename} {x.chunk.doc_title} {x.section_path}".lower()
        for x in items
    )
    if not terms:
        return False
    # A query's distinctive head nouns must occur in retrieved evidence. This
    # prevents generic enterprise headers from making an unrelated answer look grounded.
    distinctive = {x for x in terms if len(x) > 4}
    context_terms = set(re.findall(r"[a-z0-9$]+", context))
    matched = len(distinctive & context_terms)
    if matched < len(distinctive):
        stems = {term[:6] for term in distinctive}
        matched = max(matched, len(stems & {term[:6] for term in context_terms}))
    return matched / max(1, len(distinctive)) >= 0.6


def confidence(query: str, items) -> float:
    if not items:
        return 0.0
    terms = set(re.findall(r"[a-z0-9$]+", query.lower()))
    context = " ".join(body_text(x).lower() for x in items)
    coverage = len([x for x in terms if x in context]) / max(1, len(terms))
    top = items[0].score
    margin = max(0.0, top - (items[2].score if len(items) > 2 else 0.0))
    return max(0.0, min(1.0, 0.55 * top + 0.25 * margin + 0.20 * coverage))


def validate_citations_and_numbers(text: str, citations, context) -> bool:
    """Ensure references resolve and every asserted numeric token is grounded."""
    citation_ids = {citation.chunk_id for citation in citations}
    context_by_id = {item.chunk_id: item for item in context}
    citation_by_number = {number: citation for number, citation in enumerate(citations, 1)}
    references = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    if any(number not in citation_by_number for number in references):
        return False
    if any(citation_by_number[number].chunk_id not in citation_ids for number in references):
        return False
    cited_text = " ".join(
        context_by_id[citation_by_number[number].chunk_id].content.lower()
        for number in references
        if citation_by_number[number].chunk_id in context_by_id
    )
    stripped = re.sub(r"\[\d+\]", "", text)
    asserted = re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?", stripped)
    return all(token.lower().replace(",", "") in cited_text.replace(",", "") for token in asserted)


def clarification_facets(items, limit: int = 4) -> list[str]:
    facets: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        chunk = item.chunk
        key = (chunk.doc_id, chunk.section_path)
        if key in seen:
            continue
        seen.add(key)
        title = chunk.doc_title
        section = chunk.section_path
        lower = f"{title} {section}".lower()
        if "expense" in lower:
            label = f"expense category limits ({title} §3)"
        elif "travel" in lower:
            label = f"hotel nightly caps and per diems ({title} §4/§6)"
        elif "discount" in lower:
            label = f"discount cap and approval bands ({title})"
        elif "pricing" in lower or "price" in lower:
            label = f"plan seat pricing ({title} §2/§3)"
        elif "confidential" in lower or "non-disclosure" in lower:
            label = f"confidentiality survival period ({title} §4)"
        elif "benefit" in lower:
            label = f"benefit reimbursement limits ({title} §6)"
        else:
            label = f"{section} ({title})"
        facets.append(label)
        if len(facets) >= limit:
            break
    return facets
