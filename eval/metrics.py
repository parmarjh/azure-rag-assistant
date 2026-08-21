"""Deterministic metric primitives for the RAG evaluation harness.

Every metric here is computable without an LLM so that baseline-vs-improved runs are
reproducible and comparable. `judge.py` layers an optional LLM judge on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s$%./-]")
# Numeric claims we can verify verbatim against the retrieved context: currency amounts,
# percentages, and bare numbers with an optional unit word attached by the answer text.
_NUMERIC = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?%|\b\d[\d,]*(?:\.\d+)?\b")

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirty": "30", "sixty": "60", "ninety": "90",
}

# Numbers that carry no factual weight on their own: citation markers, list ordinals and
# section numbers would otherwise dominate the groundedness signal.
_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5"}


def normalize(text: str) -> str:
    text = (text or "").lower().replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _numeric_variants(token: str) -> set[str]:
    """Variants of a numeric token so `$5,250` matches `5250` and `12 weeks` matches `12`."""
    raw = token.strip().lower().replace(" ", "")
    stripped = raw.lstrip("$").rstrip("%").replace(",", "")
    out = {raw, stripped, stripped.rstrip("0").rstrip(".") if "." in stripped else stripped}
    if stripped.endswith(".0"):
        out.add(stripped[:-2])
    try:
        as_float = float(stripped)
    except ValueError:
        return {v for v in out if v}
    if as_float.is_integer():
        out.add(str(int(as_float)))
        out.add(f"{int(as_float):,}")
    return {v for v in out if v}


def _numeric_tokens(text: str) -> set[str]:
    """Every numeric token in `text`, expanded to its comparable variants."""
    tokens: set[str] = set()
    for match in _NUMERIC.findall(text or ""):
        tokens |= _numeric_variants(match)
    return tokens


def fact_present(fact: str, text: str) -> bool:
    """True if `fact` is asserted in `text`, tolerating formatting differences."""
    if not fact:
        return False
    hay = normalize(text)
    needle = normalize(fact)
    if needle and needle in hay:
        return True
    # Compare whole numeric tokens: substring matching would let "(cid:127)" satisfy "$12".
    if _NUMERIC.fullmatch(fact.strip()):
        return bool(_numeric_variants(fact) & _numeric_tokens(text))
    # Spelled-out numbers ("three (3) years" vs "3 years").
    words = needle.split()
    if len(words) == 1 and words[0] in _NUM_WORDS:
        return _NUM_WORDS[words[0]] in _numeric_tokens(text)
    return False


def numeric_claims(text: str) -> list[str]:
    claims: list[str] = []
    for match in _NUMERIC.findall(text or ""):
        token = match.strip()
        bare = token.lstrip("$").rstrip("%").replace(",", "").strip()
        if bare in _TRIVIAL_NUMBERS and not token.startswith("$") and not token.endswith("%"):
            continue
        claims.append(token)
    return claims


def _docnames(items: Iterable[Any]) -> list[str]:
    return [getattr(i, "filename", "") or "" for i in items]


@dataclass
class ItemResult:
    item_id: str
    category: str
    difficulty: str
    question: str
    answer_text: str = ""
    behavior: str = "answer"
    behavior_expected: str = "answer"
    behavior_correct: bool = False
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_sections: list[str] = field(default_factory=list)
    cited_docs: list[str] = field(default_factory=list)
    hit_at_k: bool = False
    doc_recall: float = 0.0
    section_hit: float = 0.0
    mrr: float = 0.0
    chunk_precision: float = 0.0
    context_fact_recall: float = 0.0
    answer_correctness: float = 0.0
    forbidden_leak: bool = False
    groundedness: float = 1.0
    ungrounded_claims: list[str] = field(default_factory=list)
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    hallucinated: bool = False
    confidence: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _behavior_of(answer: Any) -> str:
    if getattr(answer, "clarification", None):
        return "clarify"
    if getattr(answer, "abstained", False):
        return "abstain"
    return "answer"


def score_item(item: dict[str, Any], answer: Any) -> ItemResult:
    """Score one dataset item against a pipeline `Answer`."""
    expected_docs = [d.lower() for d in item.get("expected_docs", [])]
    expected_sections = [str(s).lower() for s in item.get("expected_sections", [])]
    expected_facts = item.get("expected_facts", [])
    forbidden = item.get("forbidden_facts", [])
    expected_behavior = item.get("expected_behavior", "answer")

    retrieved = list(getattr(answer, "retrieved", []) or [])
    retrieved_docs = [d.lower() for d in _docnames(retrieved)]
    retrieved_sections = [str(getattr(r, "section_path", "") or "") for r in retrieved]
    context = "\n".join(getattr(r, "content", "") or "" for r in retrieved)
    citations = list(getattr(answer, "citations", []) or [])
    cited_docs = [d.lower() for d in _docnames(citations)]
    answer_text = getattr(answer, "text", "") or ""
    behavior = _behavior_of(answer)

    res = ItemResult(
        item_id=item["id"],
        category=item.get("category", "unknown"),
        difficulty=item.get("difficulty", "unknown"),
        question=item["question"],
        answer_text=answer_text,
        behavior=behavior,
        behavior_expected=expected_behavior,
        behavior_correct=(behavior == expected_behavior),
        retrieved_docs=retrieved_docs,
        retrieved_sections=retrieved_sections,
        cited_docs=cited_docs,
        confidence=float(getattr(answer, "confidence", 0.0) or 0.0),
        latency_ms=dict(getattr(answer, "latency_ms", {}) or {}),
    )

    usage = getattr(answer, "usage", None)
    if usage is not None:
        get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0))
        res.prompt_tokens = int(get("prompt_tokens") or 0)
        res.completion_tokens = int(get("completion_tokens") or 0)
        res.cost_usd = float(get("estimated_cost_usd") or 0.0)

    # --- retrieval -------------------------------------------------------------
    if expected_docs:
        found = [d for d in expected_docs if any(d in r for r in retrieved_docs)]
        res.hit_at_k = bool(found)
        res.doc_recall = len(found) / len(expected_docs)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if any(e in doc for e in expected_docs):
                res.mrr = 1.0 / rank
                break
        relevant = sum(1 for d in retrieved_docs if any(e in d for e in expected_docs))
        res.chunk_precision = relevant / len(retrieved_docs) if retrieved_docs else 0.0
        if expected_sections:
            hits = 0
            for section in expected_sections:
                for got_doc, got_section in zip(retrieved_docs, retrieved_sections):
                    if not any(e in got_doc for e in expected_docs):
                        continue
                    got = got_section.lower()
                    if got.startswith(section) or f" {section} " in f" {got} " or section in got:
                        hits += 1
                        break
            res.section_hit = hits / len(expected_sections)
    else:
        # No-answer / ambiguous items: no retrieval target, so retrieval scores are N/A
        # and left at 0; behaviour correctness is what matters for them.
        res.hit_at_k = False

    if expected_facts:
        res.context_fact_recall = sum(fact_present(f, context) for f in expected_facts) / len(expected_facts)

    # --- generation ------------------------------------------------------------
    if expected_behavior == "answer":
        if expected_facts:
            res.answer_correctness = sum(fact_present(f, answer_text) for f in expected_facts) / len(expected_facts)
        else:
            res.answer_correctness = 1.0 if answer_text.strip() else 0.0
    else:
        res.answer_correctness = 1.0 if res.behavior_correct else 0.0

    res.forbidden_leak = any(fact_present(f, answer_text) for f in forbidden)

    claims = numeric_claims(answer_text)
    if claims:
        ungrounded = [c for c in claims if not fact_present(c, context)]
        res.ungrounded_claims = ungrounded
        res.groundedness = 1.0 - len(ungrounded) / len(claims)

    if cited_docs:
        if expected_docs:
            res.citation_precision = sum(
                1 for c in cited_docs if any(e in c for e in expected_docs)
            ) / len(cited_docs)
            res.citation_recall = sum(
                1 for e in expected_docs if any(e in c for c in cited_docs)
            ) / len(expected_docs)
        else:
            # Citing anything on a no-answer question is a false-grounding signal.
            res.citation_precision = 0.0
            res.citation_recall = 0.0
    elif not expected_docs:
        res.citation_precision = 1.0
        res.citation_recall = 1.0

    # A hallucination is: answering a question the corpus cannot support, asserting a
    # number absent from the retrieved context, or leaking a superseded value.
    res.hallucinated = bool(
        (expected_behavior in {"abstain"} and behavior == "answer")
        or res.ungrounded_claims
        or res.forbidden_leak
    )
    return res


def _mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _pct(values: Sequence[bool]) -> float:
    return round(100.0 * sum(1 for v in values if v) / len(values), 2) if values else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def aggregate(results: Sequence[ItemResult]) -> dict[str, Any]:
    retrievable = [r for r in results if r.behavior_expected == "answer"]
    answerable = retrievable
    totals = {
        "n_items": len(results),
        "retrieval": {
            "hit_rate_pct": _pct([r.hit_at_k for r in retrievable]),
            "doc_recall": _mean([r.doc_recall for r in retrievable]),
            "section_hit_rate": _mean([r.section_hit for r in retrievable]),
            "mrr": _mean([r.mrr for r in retrievable]),
            "chunk_precision": _mean([r.chunk_precision for r in retrievable]),
            "context_fact_recall": _mean([r.context_fact_recall for r in retrievable]),
        },
        "generation": {
            "answer_correctness": _mean([r.answer_correctness for r in results]),
            "answer_correctness_answerable": _mean([r.answer_correctness for r in answerable]),
            "groundedness": _mean([r.groundedness for r in results]),
            "citation_precision": _mean([r.citation_precision for r in answerable]),
            "citation_recall": _mean([r.citation_recall for r in answerable]),
            "hallucination_rate_pct": _pct([r.hallucinated for r in results]),
            "behavior_accuracy_pct": _pct([r.behavior_correct for r in results]),
            "stale_version_leak_pct": _pct([r.forbidden_leak for r in results]),
        },
        "system": {
            "latency_ms_mean": _mean([r.latency_ms.get("total", 0.0) for r in results]),
            "latency_ms_p95": _percentile([r.latency_ms.get("total", 0.0) for r in results], 95),
            "prompt_tokens_mean": _mean([float(r.prompt_tokens) for r in results]),
            "completion_tokens_mean": _mean([float(r.completion_tokens) for r in results]),
            "cost_usd_total": round(sum(r.cost_usd for r in results), 6),
        },
        "by_category": {},
    }
    categories = sorted({r.category for r in results})
    for cat in categories:
        subset = [r for r in results if r.category == cat]
        sub_retrievable = [r for r in subset if r.behavior_expected == "answer"]
        totals["by_category"][cat] = {
            "n": len(subset),
            "hit_rate_pct": _pct([r.hit_at_k for r in sub_retrievable]),
            "answer_correctness": _mean([r.answer_correctness for r in subset]),
            "behavior_accuracy_pct": _pct([r.behavior_correct for r in subset]),
            "hallucination_rate_pct": _pct([r.hallucinated for r in subset]),
            "groundedness": _mean([r.groundedness for r in subset]),
        }
    return totals
