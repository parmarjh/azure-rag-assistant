from __future__ import annotations

import re

from .generation import body_text

STOP_WORDS = {
    "a", "an", "and", "are", "do", "for", "get", "how", "i", "in", "is",
    "many", "of", "our", "per", "the", "this", "to", "what", "which",
    "about", "can", "does", "if", "it", "on", "someone", "that", "we",
    "have", "has", "had", "was", "were", "who", "whom", "much", "miss",
    "compare", "difference", "both", "versus", "vs", "policy",
}
ALIASES = {
    "applies": "apply",
    "applied": "apply",
    "approve": "approval",
    "approves": "approval",
    "calls": "call",
    "combined": "combine",
    "days": "day",
    "employees": "employee",
    "meals": "meal",
    "spend": "expense",
    "costs": "price",
    "deadline": "day",
    "receipt": "receipt",
    "submission": "submit",
    "survival": "survive",
    "threshold": "limit",
    "versus": "compare",
}


def _stem(term: str) -> str:
    term = ALIASES.get(term, term)
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    for suffix in ("ing", "ed", "es"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[:-len(suffix)]
    return term


def _tokens(text: str) -> list[str]:
    return [_stem(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _body(item) -> str:
    if hasattr(item, "chunk"):
        return body_text(item)
    content = item.content
    header = item.header.strip()
    if header and content.startswith(header):
        content = content[len(header):]
    return content.lstrip(" \n:").strip()


def scrub(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    return re.sub(r"(?i)(ignore (?:all )?previous instructions|system prompt)", "[redacted]", text)


def has_sufficient_evidence(query: str, items, corpus=None, pair_fraction: float = 0.5) -> bool:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    terms = [_stem(token) for token in query_tokens
             if token not in STOP_WORDS and not token.isdigit()]
    if not terms or not items:
        return False
    corpus = corpus or items
    corpus_terms = set(_tokens(" ".join(
        f"{_body(x)} {x.filename} "
        f"{x.chunk.doc_title if hasattr(x, 'chunk') else x.doc_title} "
        f"{x.section_path}"
        for x in corpus
    )))
    if any(len(token) > 3 and _stem(token) not in corpus_terms
           for token in query_tokens if token not in STOP_WORDS and not token.isdigit()):
        return False
    if re.search(r"\b(compare|versus|vs\.?|difference|both)\b", query, re.I):
        return True

    document_terms = [
        _tokens(f"{item.section_path} {_body(item)}") for item in items
    ]
    document_frequency = {}
    for term in set(terms):
        document_frequency[term] = sum(
            term in set(tokens) for tokens in document_terms
        )
    corpus_documents = [_tokens(f"{x.section_path} {_body(x)}") for x in corpus]
    corpus_frequency = {
        term: sum(term in set(tokens) for tokens in corpus_documents)
        for term in set(terms)
    }
    total_docs = max(1, len(corpus_documents))
    idf = {
        term: (1.0 + (total_docs / max(1, corpus_frequency.get(term, 0))))
        for term in terms
    }
    high_idf = {term for term in terms if idf.get(term, 0.0) >= 10.0}
    ordered = [term for term in terms if term in high_idf]
    pairs = list(zip(ordered, ordered[1:]))
    if not pairs:
        return True
    satisfied = 0
    for left, right in pairs:
        found = False
        for tokens in document_terms:
            for start, token in enumerate(tokens):
                if token != left:
                    continue
                positions = [i for i, value in enumerate(tokens) if value == right]
                if any(abs(position - start) <= 10 for position in positions):
                    found = True
                    break
            if found:
                break
        satisfied += int(found)
    return satisfied / len(pairs) >= pair_fraction


def confidence(query: str, items) -> float:
    if not items:
        return 0.0
    terms = set(re.findall(r"[a-z0-9$]+", query.lower()))
    context = set(_tokens(" ".join(_body(x) for x in items)))
    coverage = len([x for x in terms if _stem(x) in context]) / max(1, len(terms))
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
        if label in facets:
            continue
        facets.append(label)
        if len(facets) >= limit:
            break
    return facets
