from __future__ import annotations

import re

from .generation import body_text

STOP_WORDS = {
    "a", "an", "and", "are", "do", "for", "get", "how", "i", "in", "is",
    "many", "of", "our", "per", "the", "this", "to", "what", "which",
    "about", "can", "does", "if", "it", "on", "someone", "that", "we",
    "have", "has", "had", "was", "were", "who", "whom", "much", "miss",
    "compare", "difference", "both", "versus", "vs", "policy",
    "while", "company", "standard", "required", "receive", "reimbursable",
    "purchase", "their",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
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
    "maximum": "max",
    "rotate": "rotation",
    "rotates": "rotation",
    "rotating": "rotation",
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


def _term_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    number_values = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    left = number_values.get(left, left)
    right = number_values.get(right, right)
    return left == right or (
        len(left) > 3
        and len(right) > 3
        and _within_one(left, right)
    )


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


def _within_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    differences = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        differences += 1
        if differences > 1:
            return False
        if len(left) > len(right):
            i += 1
        elif len(right) > len(left):
            j += 1
        else:
            i += 1
            j += 1
    return differences + (len(left) - i) + (len(right) - j) <= 1


def has_sufficient_evidence(
    query: str,
    items,
    corpus=None,
    pair_fraction: float = 0.5,
    term_count: int = 3,
    pair_window: int = 10,
    subqueries: list[str] | None = None,
) -> bool:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    terms = [
        _stem(token)
        for token in query_tokens
        if token not in STOP_WORDS and len(token) > 3 and not token.isdigit()
    ]
    if not terms or not items:
        return False
    corpus = corpus or items
    corpus_terms = set(_tokens(" ".join(
        f"{_body(x)} {x.filename} "
        f"{x.chunk.doc_title if hasattr(x, 'chunk') else x.doc_title} "
        f"{x.section_path}"
        for x in corpus
    )))
    if any(
        len(token) > 3
        and token not in NUMBER_WORDS
        and not any(_within_one(_stem(token), candidate) for candidate in corpus_terms)
        for token in query_tokens
        if token not in STOP_WORDS and not token.isdigit()
    ):
        return False
    if subqueries and re.search(r"\b(compare|versus|vs\.?|difference|both)\b", query, re.I):
        return all(
            has_sufficient_evidence(
                subquery,
                items,
                corpus,
                pair_fraction,
                term_count,
                pair_window,
            )
            for subquery in subqueries
        )

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
    selected = sorted(
        enumerate(dict.fromkeys(terms)),
        key=lambda pair: (-idf.get(pair[1], 0.0), pair[0]),
    )[:max(1, term_count)]
    selected_positions = sorted(position for position, _ in selected)
    ordered = [terms[position] for position in selected_positions]
    if len(ordered) == 1:
        return any(
            _term_matches(ordered[0], term)
            for tokens in document_terms
            for term in tokens
        )
    pairs = list(zip(ordered, ordered[1:]))
    pair_scores = sorted(
        pairs,
        key=lambda pair: -(idf.get(pair[0], 0.0) + idf.get(pair[1], 0.0)),
    )
    found = False
    evidence_window = pair_window + (
        15 if any(token.isdigit() or token in NUMBER_WORDS for token in query_tokens) else 0
    )
    for left, right in pair_scores:
        for tokens in document_terms:
            left_positions = [
                i for i, value in enumerate(tokens)
                if _term_matches(value, left)
            ]
            right_positions = [
                i for i, value in enumerate(tokens)
                if _term_matches(value, right)
            ]
            if any(abs(left_position - right_position) <= evidence_window
                   for left_position in left_positions
                   for right_position in right_positions):
                found = True
                break
        if found:
            break
    satisfied = int(found)
    return satisfied / max(1, min(1, len(pair_scores))) >= pair_fraction


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


def clarification_facets(items, limit: int = 4, head_noun: str = "limit") -> list[str]:
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
        heading = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", section).strip()
        words = [
            re.sub(r"ies$", "y", word.lower())
            for word in re.findall(r"[A-Za-z]+", heading)
        ]
        description = " ".join(words) or head_noun
        matches_head = any(_stem(word) == _stem(head_noun) for word in words)
        label_description = description if matches_head else f"{head_noun} in {description}"
        label = f"{label_description} ({title}"
        if chunk.section_number:
            label += f" §{chunk.section_number}"
        label += ")"
        if label in facets:
            continue
        facets.append(label)
        if len(facets) >= limit:
            break
    return facets
