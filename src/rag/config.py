from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RagConfig:
    mode: str = "improved"
    provider: str = "local"
    embedding_model: str = "text-embedding-3-large"
    answer_deployment: str = "gpt-4o"
    search_endpoint: str | None = None
    openai_endpoint: str | None = None
    search_index: str = "northwind-knowledge"
    top_k: int = 8
    rerank_threshold: float = 0.10
    abstain_threshold: float = 0.19
    token_budget: int = 3000
    per_document_cap: int = 3
    cache_ttl_seconds: int = 3600
    acl_map: dict[str, list[str]] = field(default_factory=lambda: {
        "HR": ["hr", "all-staff"], "Finance": ["finance", "all-staff"],
        "IT": ["it", "all-staff"], "Legal": ["legal", "all-staff"],
        "Sales": ["sales", "all-staff"],
    })


BASELINE_CONFIG = RagConfig(mode="baseline", top_k=3, rerank_threshold=0.0,
                            abstain_threshold=0.0, token_budget=5000)
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
                        "openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")})
