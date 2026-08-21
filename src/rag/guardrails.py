from __future__ import annotations

import re

from .generation import body_text

STOP_WORDS = {
    "a", "an", "and", "are", "do", "for", "get", "how", "i", "in", "is",
    "many", "of", "our", "per", "the", "this", "to", "what", "which",
    "about", "can", "does", "if", "it", "on", "someone", "that", "we",
    "have", "has", "had", "was", "were", "who", "whom", "much", "miss",
    "compare", "difference", "both", "versus", "vs",
    "while", "their", "based", "give", "need", "request", "receive",
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
    "lockout": "lock",
}
ANSWER_HEADS = {
    "amount", "cap", "cost", "deadline", "limit", "long", "period", "percentage",
    "price", "rate", "threshold",
}


def _stem(term: str) -> str:
    term = ALIASES.get(term, term)
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    for suffix in ("ation", "able", "ible", "ment", "ate", "al", "ing", "ed", "es"):
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


def _unknown_term_matches(left: str, right: str) -> bool:
    return left == right or (
        abs(len(left) - len(right)) == 1
        and len(left) > 3
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


def _evidence_units(item) -> list[list[str]]:
    body = _body(item)
    prefix = f"{item.filename} {item.chunk.doc_title} {item.section_path}"
    if "|" in body or "Sheet:" in body:
        return [
            _tokens(f"{prefix} {line}")
            for line in body.splitlines()
            if line.strip()
        ]
    return [_tokens(f"{prefix} {body}")]


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
    term_count: int = 2,
    subqueries: list[str] | None = None,
) -> bool:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    terms = [
        _stem(token)
        for token in query_tokens
        if (
            token not in STOP_WORDS
            and len(token) >= 3
            and not token.isdigit()
            and _stem(token) not in ANSWER_HEADS
        )
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
        and not any(
            _unknown_term_matches(_stem(token), candidate)
            for candidate in corpus_terms
        )
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
                term_count,
            )
            for subquery in subqueries
        )

    document_terms = [_evidence_units(item) for item in items]
    def canonical(term: str) -> str:
        if term in corpus_terms:
            return term
        matches = sorted(candidate for candidate in corpus_terms
                         if _term_matches(term, candidate))
        return matches[0] if matches else term

    document_frequency = {}
    for term in set(terms):
        corpus_term = canonical(term)
        document_frequency[term] = sum(
            any(corpus_term in set(unit) for unit in units)
            for units in document_terms
        )
    corpus_documents = [_tokens(f"{x.section_path} {_body(x)}") for x in corpus]
    corpus_frequency = {
        term: sum(canonical(term) in set(tokens) for tokens in corpus_documents)
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

    def cooccurs(selected_terms: list[str]) -> bool:
        return any(
            all(
                any(
                    _term_matches(value, selected_term)
                    for value in tokens
                )
                for selected_term in selected_terms
            )
            for units in document_terms
            for tokens in units
        )

    if len(ordered) == 1:
        return any(
            _term_matches(ordered[0], term)
            for units in document_terms
            for tokens in units
            for term in tokens
        )
    found = cooccurs(ordered)
    value_query = bool(re.search(
        r"\b(?:how much|how many|price|cost|amount|maximum|rate|approval|approve|required)\b"
        r"|\$\s*\d",
        query,
        re.I,
    ))
    if not found and term_count == 2 and value_query:
        ranked_terms = [
            term for _, term in sorted(
                enumerate(dict.fromkeys(terms)),
                key=lambda pair: (-idf.get(pair[1], 0.0), pair[0]),
            )
        ]
        for index, left in enumerate(ranked_terms):
            for right in ranked_terms[index + 1:]:
                if cooccurs([left, right]):
                    ordered = [left, right]
                    found = True
                    break
            if found:
                break
    named_terms = [
        _stem(token)
        for index, token in enumerate(re.findall(r"[A-Za-z0-9]+", query))
        if index > 0 and token[0].isupper() and len(token) >= 3
    ]
    if named_terms and re.search(r"\b(?:tier|plan)\b", query, re.I):
        value_pattern = re.compile(r"\$|%|\b\d+(?:\.\d+)?\b")
        named_evidence = any(
            all(
                any(
                    _term_matches(named, value)
                    for value in _tokens(_body(item))
                )
                for named in named_terms
            )
            and bool(value_pattern.search(_body(item)))
            and bool(re.search(r"\b(?:tiers?|plans?)\b", item.section_path, re.I))
            for item in items
        )
        found = found and named_evidence
    return found


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


def clarification_facets(
    items,
    limit: int = 4,
    head_noun: str = "limit",
) -> list[str]:
    ranked: list[tuple[float, object]] = []
    seen: set[tuple[str, str]] = set()
    generic_sections = {"purpose", "overview", "introduction", "scope", "definitions"}
    head_terms = {
        _stem(word)
        for word in re.findall(r"[a-z]+", head_noun.lower())
        if len(word) > 3 and word not in STOP_WORDS
    }
    if len(re.findall(r"[a-z]+", head_noun.lower())) > 1:
        head_terms = set()
    anchor_terms = {
        _stem(word)
        for word in ("limit", "threshold", "cap", "cost", "price", "spend", "expense", "rate")
    }
    for item in items:
        chunk = item.chunk
        key = (chunk.doc_id, chunk.section_path)
        if key in seen:
            continue
        seen.add(key)
        heading = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", chunk.section_path).strip()
        body = _body(item)
        body_tokens = _tokens(body)
        heading_tokens = _tokens(heading)
        value_positions = [
            index for index, token in enumerate(body_tokens)
            if token.isdigit() or token in NUMBER_WORDS
        ]
        if re.search(r"[$€£%]", body):
            value_positions.extend(
                index for index, token in enumerate(body_tokens)
                if re.search(r"\d", token)
            )
        head_positions = [
            index for index, token in enumerate(body_tokens)
            if not head_terms or token in head_terms
        ]
        near_value = (
            bool(value_positions)
            and (
                not head_terms
                or any(
                    abs(head_position - value_position) <= 12
                    for head_position in head_positions
                    for value_position in value_positions
                )
            )
        )
        has_anchor = bool(anchor_terms & (set(body_tokens) | set(heading_tokens)))
        if not has_anchor:
            continue
        if heading.casefold() in generic_sections and not near_value:
            continue
        if not near_value:
            continue
        relevance = item.score
        if head_terms and head_positions:
            relevance += 0.20
        if value_positions:
            relevance += 0.05
        score = relevance
        ranked.append((score, item))
    facets: list[str] = []
    for _, item in sorted(ranked, key=lambda pair: -pair[0]):
        chunk = item.chunk
        heading = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", chunk.section_path).strip()
        label = f"{heading} ({chunk.doc_title}"
        if chunk.section_number:
            label += f" §{chunk.section_number}"
        label += ")"
        if label in facets:
            continue
        facets.append(label)
        if len(facets) >= limit:
            break
    return facets
