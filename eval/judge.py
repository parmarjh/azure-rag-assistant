"""Optional LLM-as-judge layer.

The deterministic metrics in `metrics.py` are the primary signal (they run offline and are
reproducible). When a real LLM is configured (Azure OpenAI), `--judge llm` adds
LLM-graded correctness / groundedness / relevance, mirroring the Azure AI Foundry
evaluators so the two can be cross-checked.
"""

from __future__ import annotations

import json
from typing import Any

JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. You are given a QUESTION, the reference ANSWER, the "
    "CONTEXT that was retrieved, and the SYSTEM ANSWER. Score the system answer only on "
    "the evidence given.\n"
    "Return JSON with exactly these keys and no prose:\n"
    '{"correctness": 0-1 float, "groundedness": 0-1 float, "relevance": 0-1 float, '
    '"hallucinated": true|false, "reason": "one sentence"}\n'
    "correctness: does the system answer convey the same facts as the reference answer "
    "(partial credit allowed)?\n"
    "groundedness: is every claim in the system answer supported by CONTEXT?\n"
    "relevance: does the system answer address the question?\n"
    "hallucinated: true if the answer asserts anything not supported by CONTEXT, or "
    "answers at all when the reference says the information is absent."
)


def _prompt(item: dict[str, Any], result: Any, context: str) -> list[dict[str, str]]:
    user = (
        f"QUESTION:\n{item['question']}\n\n"
        f"REFERENCE ANSWER:\n{item.get('expected_answer', '(the knowledge base does not contain this)')}\n\n"
        f"CONTEXT:\n{context[:12000]}\n\n"
        f"SYSTEM ANSWER:\n{result.answer_text or '(no answer / abstained)'}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_item(llm: Any, item: dict[str, Any], result: Any, context: str) -> dict[str, Any]:
    """Grade one item with an LLM. Returns {} when the judge is unavailable or unparseable."""
    if llm is None:
        return {}
    try:
        completion = llm.chat(_prompt(item, result, context), temperature=0.0, max_tokens=300)
        text = getattr(completion, "text", None) or str(completion)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return {"error": "unparseable judge output"}
        parsed = json.loads(text[start : end + 1])
        usage = getattr(completion, "usage", None)
        if usage is not None:
            parsed["judge_tokens"] = (
                usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                if isinstance(usage, dict)
                else getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
            )
        return parsed
    except Exception as exc:  # a judge failure must never fail the evaluation run
        return {"error": f"{type(exc).__name__}: {exc}"}


def get_judge(kind: str) -> Any:
    """kind: 'none' | 'llm'. Returns an LLM provider or None."""
    if kind != "llm":
        return None
    from rag.config import get_config
    from rag.providers.llm import get_llm_provider

    cfg = get_config(mode="improved")
    if getattr(cfg, "provider", "local") != "azure":
        raise SystemExit(
            "--judge llm requires Azure OpenAI configuration; the local extractive provider "
            "cannot act as a judge. Configure AZURE_OPENAI_* env vars or use --judge none."
        )
    return get_llm_provider(cfg)
