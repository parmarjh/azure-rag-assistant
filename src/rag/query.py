from __future__ import annotations

import re

from .models import ChatTurn


def understand(question: str, history: list[ChatTurn] | None = None) -> dict:
    history = history or []
    q = question.strip()
    entities: dict[str, str] = {}
    for turn in history[-3:]:
        entities.update(turn.entities)
    if re.search(r"\b(standard|professional|enterprise|starter)\b", q, re.I):
        entities["plan"] = re.search(r"\b(standard|professional|enterprise|starter)\b", q, re.I).group(1)
    for key, pat in {"department": r"\b(HR|Finance|IT|Legal|Sales)\b",
                     "year": r"\b(202[0-9])\b"}.items():
        m = re.search(pat, q, re.I)
        if m:
            entities[key] = m.group(1)
    if q.lower() in {"what about standard?", "what about professional?", "what about enterprise?"} and history:
        topic = entities.get("topic", "the plan")
        q = f"What is the {entities.get('plan', '')} plan {topic}?"
    version_intent = bool(re.search(r"\b(202[0-9]|previous|prior|old|historical)\b", q, re.I))
    comparison = bool(re.search(r"\b(compare|versus|vs\.?|difference|both)\b", q, re.I))
    ambiguous = bool(re.search(r"^\s*what is the limit\??\s*$", q, re.I))
    entities.setdefault("topic", "pricing" if "price" in q.lower() or "tier" in q.lower() else "")
    return {"rewritten": q, "entities": entities, "version_intent": version_intent,
            "comparison": comparison, "ambiguous": ambiguous}
