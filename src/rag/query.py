from __future__ import annotations

import re

from .models import ChatTurn

_COMPARISON = re.compile(r"\b(compare|versus|vs\.?|difference|both)\b", re.I)
_HEADS = re.compile(r"\b(limit|threshold|cap|deadline|cost|price|how much can i spend)\b", re.I)


def _slots(text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    plan = re.search(r"\b([A-Za-z][\w-]*)\s+(?:plan|tier)\b", text, re.I)
    if plan and plan.group(1).lower() not in {"as", "the", "a", "what"}:
        slots["plan"] = plan.group(1)
    year = re.search(r"\b(19|20)\d{2}\b", text)
    if year:
        slots["year"] = year.group(0)
    return slots


def _subject(text: str) -> str:
    text = text.rstrip(" ?.")
    patterns = (
        r"^(?:how many|how much)\s+(.+?)\s+(?:do|does|can|may|per|each|for)\b",
        r"^what\s+(?:is|are)\s+(.+?)(?:\s+(?:for|in|per|each)\b|$)",
        r"^is\s+(.+?)\s+(?:required|allowed|available)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    return ""


def _follow_up(question: str) -> bool:
    words = question.split()
    lowered = question.lower()
    return (
        len(words) <= 10
        or lowered.startswith(("what about", "and ", "how about"))
        or bool(re.match(
            r"^(do|does|did|are|is|can|could|which)\s+(they|it|there|those|these)\b",
            lowered,
        ))
        or bool(re.match(r"^(do|does|did|are|is|can|could)\s+there\b", lowered))
    )


def _condense(question: str, history: list[ChatTurn], entities: dict[str, str]) -> str:
    if not history or not _follow_up(question):
        return question
    previous = history[-1].question
    if previous.lower().startswith(("what about", "and ", "how about")) and len(history) > 1:
        previous = history[-2].question
    current_slots = _slots(question)
    if current_slots.get("plan"):
        old_slots = _slots(previous)
        if old_slots.get("plan"):
            return re.sub(
                rf"\b{re.escape(old_slots['plan'])}\b",
                current_slots["plan"],
                previous,
                count=1,
                flags=re.I,
            )
        return f"{previous.rstrip('?')} for the {current_slots['plan']} tier?"
    topic = _subject(previous)
    if topic:
        if re.match(r"^(?:what|how)\s+about\b", question, re.I):
            tail = re.sub(r"^(?:what|how)\s+about\b", "", question, flags=re.I).strip()
            return f"{topic} for {tail}".rstrip("?") + "?"
        if re.match(r"^(do|does|did|are|is|can|could)\s+they\b", question, re.I):
            tail = re.sub(
                r"^(do|does|did|are|is|can|could)\s+they\b",
                "",
                question,
                flags=re.I,
            ).strip()
            auxiliary = "Do" if question.lower().startswith("do ") else "Are"
            return f"{auxiliary} {topic} {tail}".rstrip("?") + "?"
        if question.lower().startswith(("which ", "what ")):
            return f"{question.rstrip('?')} for {topic}?"
        if re.match(r"^(?:are|is|do|does|can|could)\s+there\b", question, re.I):
            return f"{question.rstrip('?')} about {topic}?"
        return f"{topic} {question[0].lower() + question[1:]}".rstrip("?") + "?"
    return f"{previous.rstrip('?')}; {question}"


def decompose(question: str, known_entities: list[str] | None = None) -> list[str]:
    if not _COMPARISON.search(question):
        return [question]
    normalized = re.sub(r"\s+", " ", question).strip().rstrip("?")
    match = re.search(r"\b(?:versus|vs\.?|difference between)\b", normalized, re.I)
    if match:
        left, right = normalized[:match.start()], normalized[match.end():]
        left_entity = re.search(r"\bin\s+(?:the\s+)?(.+)$", left, re.I)
        topic = re.sub(r"^\s*compare\s+", "", left, flags=re.I)
        topic = re.sub(r"\s+in\s+.+$", "", topic, flags=re.I).strip()
        right_entity = re.sub(
            r"^\s*(?:the\s+)?(?:one\s+)?in\s+", "", right, flags=re.I
        ).strip()
        if left_entity and right_entity:
            return [f"{topic} {left_entity.group(1)}", f"{topic} {right_entity}"]
    with_match = re.search(r"\bin\s+(?:the\s+)?(.+?)\s+with\s+the\s+one\s+in\s+(.+)$",
                           normalized, re.I)
    if with_match:
        topic = re.sub(r"^\s*compare\s+", "", normalized[:with_match.start()], flags=re.I)
        topic = re.sub(r"\s+in\s+$", "", topic, flags=re.I).strip()
        return [f"{topic} {with_match.group(1)}", f"{topic} {with_match.group(2)}"]
    entities = [
        entity
        for entity in (known_entities or [])
        if len(entity) > 2 and re.search(rf"\b{re.escape(entity)}\b", question, re.I)
    ]
    if len(entities) >= 2:
        topic = re.sub(r"^\s*compare\s+", "", normalized, flags=re.I)
        topic = re.sub(r"\bin\s+.+$", "", topic, flags=re.I).strip()
        return [f"{topic} {entity}" for entity in entities[:2]]
    return [question]


def understand(
    question: str,
    history: list[ChatTurn] | None = None,
    known_entities: list[str] | None = None,
) -> dict:
    history = history or []
    original = question.strip()
    entities: dict[str, str] = {}
    for turn in history[-3:]:
        entities.update(turn.entities)
    entities.update(_slots(original))
    rewritten = _condense(original, history, entities)
    entities.update(_slots(rewritten))
    comparison = bool(_COMPARISON.search(rewritten))
    head = _HEADS.search(rewritten)
    version_intent = bool(
        re.search(r"\b(19|20)\d{2}\b|\b(previous|prior|old|historical)\b", rewritten, re.I)
    )
    head_match = _HEADS.search(rewritten)
    query_words = re.findall(r"[a-z]+", rewritten.lower())
    head_words = re.findall(r"[a-z]+", head_match.group(0)) if head_match else []
    modifiers = [
        word for word in query_words
        if word not in {"what", "is", "the", "a", "an", "how", "much", "can", "i", "on"}
        and word not in head_words
        and len(word) > 1
    ]
    disambiguating_modifiers = [
        word for word in modifiers if word not in {"approval"}
    ]
    ambiguous_head = (
        bool(head_match)
        and len(rewritten.split()) <= 8
        and not any(entities.get(key) for key in ("plan", "department", "year"))
        and (
            len(head_words) > 1
            or not disambiguating_modifiers
            or disambiguating_modifiers == ["approval"]
        )
    )
    entities.setdefault(
        "topic",
        "pricing" if re.search(r"\b(price|tier)\b", rewritten, re.I) else _subject(rewritten),
    )
    return {
        "rewritten": rewritten,
        "entities": entities,
        "version_intent": version_intent,
        "comparison": comparison,
        "head_noun": head.group(1).lower() if head else None,
        "ambiguous": ambiguous_head and not history,
        "ambiguous_head": ambiguous_head,
        "subqueries": decompose(rewritten, known_entities),
    }
