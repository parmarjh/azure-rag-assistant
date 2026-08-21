from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RagConfig:
    mode: str = "improved"
    provider: str = "local"
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o"
    search_endpoint: str | None = None
    openai_endpoint: str | None = None
    search_index: str = "northwind-knowledge"
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 80
    section_aware: bool = True
    prepend_header: bool = True
    top_k: int = 8
    candidate_k: int = 30
    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_rewrite: bool = True
    use_subquery_decomposition: bool = True
    filter_current_only: bool = True
    per_doc_cap: int = 3
    neighbour_expansion: bool = True
    rerank_threshold: float = 0.10
    abstain_threshold: float = 0.19
    evidence_term_count: int = 2
    enable_guardrails: bool = True
    enable_clarification: bool = True
    context_token_budget: int = 3000
    prompt_cost_per_1k: float = 0.00015
    completion_cost_per_1k: float = 0.00060
    cache_ttl_seconds: int = 3600
    acl_map: dict[str, list[str]] = field(default_factory=lambda: {
        "HR": ["hr"], "Finance": ["finance"], "Legal": ["legal"],
        "Sales": ["sales"], "IT": ["it", "all-staff"],
    })


BASELINE_CONFIG = RagConfig(
    mode="baseline", chunk_size_tokens=1000, chunk_overlap_tokens=0,
    section_aware=False, prepend_header=False, top_k=3, candidate_k=3,
    use_hybrid=False, use_rerank=False, use_query_rewrite=False,
    use_subquery_decomposition=False, filter_current_only=False, per_doc_cap=99,
    neighbour_expansion=False, rerank_threshold=0.0, abstain_threshold=0.0,
    evidence_term_count=0,
    enable_guardrails=False, enable_clarification=False, context_token_budget=5000,
)
IMPROVED_CONFIG = RagConfig()


def get_config(mode: str = "improved") -> RagConfig:
    if mode not in {"baseline", "improved"}:
        raise ValueError("mode must be baseline or improved")
    base = BASELINE_CONFIG if mode == "baseline" else IMPROVED_CONFIG
    provider = os.getenv("RAG_PROVIDER", "")
    if not provider:
        provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_SEARCH_ENDPOINT") else "local"
    return RagConfig(**{**base.__dict__, "provider": provider,
                        "search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                        "openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                        "abstain_threshold": float(os.getenv(
                            "RAG_ABSTAIN_THRESHOLD", base.abstain_threshold)),
                        "rerank_threshold": float(os.getenv(
                            "RAG_RERANK_THRESHOLD", base.rerank_threshold)),
                        "evidence_term_count": int(os.getenv(
                            "RAG_EVIDENCE_TERM_COUNT", base.evidence_term_count)),
                        "context_token_budget": int(os.getenv(
                            "RAG_CONTEXT_TOKEN_BUDGET", base.context_token_budget)),
                        "prompt_cost_per_1k": float(os.getenv(
                            "RAG_PROMPT_COST_PER_1K", base.prompt_cost_per_1k)),
                        "completion_cost_per_1k": float(os.getenv(
                            "RAG_COMPLETION_COST_PER_1K", base.completion_cost_per_1k)),
                        "chat_model": os.getenv("RAG_CHAT_MODEL", base.chat_model),
                        "embedding_model": os.getenv(
                            "RAG_EMBEDDING_MODEL", base.embedding_model)})
