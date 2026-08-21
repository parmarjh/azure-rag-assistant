from __future__ import annotations

import logging
import time

from .acl import build_filter
from .cache import TTLCache, cache_key
from .chunking import baseline_chunks, improved_chunks
from .config import RagConfig
from .context import assemble
from .conversation import SessionStore
from .generation import generate
from .guardrails import confidence, has_sufficient_evidence
from .index.azure_search import AzureSearchIndex
from .index.local_index import LocalSearchIndex
from .models import Answer, ChatTurn, Chunk, IngestStats, Usage, UserContext
from .parsing import parse_directory
from .providers.embeddings import AzureOpenAIEmbeddingProvider, LocalEmbeddingProvider
from .query import understand
from .rerank import rerank

log = logging.getLogger("rag")


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        if config.provider == "azure":
            import os
            self.embedder = AzureOpenAIEmbeddingProvider(config.openai_endpoint or "",
                config.embedding_model, os.getenv("AZURE_OPENAI_API_KEY"))
        else:
            self.embedder = LocalEmbeddingProvider()
        if config.provider == "azure":
            import os
            self.index = AzureSearchIndex(config.search_endpoint or "", config.search_index,
                                          api_key=os.getenv("AZURE_SEARCH_API_KEY"))
        else:
            self.index = LocalSearchIndex(self.embedder)
        self.sessions = SessionStore()
        self.cache = TTLCache(config.cache_ttl_seconds)
        self.chunks: list[Chunk] = []

    @classmethod
    def from_config(cls, cfg): return cls(cfg)

    def ingest(self, source_dir: str) -> IngestStats:
        docs = parse_directory(source_dir)
        self.index.create()
        chunks = []
        for doc in docs:
            chunks.extend(baseline_chunks(doc) if self.config.mode == "baseline" else improved_chunks(doc))
        vectors = self.embedder.embed([c.content for c in chunks])
        for c, vector in zip(chunks, vectors):
            c.embedding = vector
        self.index.upload(chunks)
        self.chunks = chunks
        return IngestStats(len(docs), len(chunks), self.config.mode)

    def answer(self, question: str, session_id: str | None = None,
               user: UserContext | None = None) -> Answer:
        started = time.perf_counter()
        history = self.sessions.history(session_id) if session_id else []
        understanding = understand(question, history)
        rewritten = understanding["rewritten"]
        filters = ({"_baseline": True} if self.config.mode == "baseline" else
                   build_filter(user, None, not understanding["version_intent"]))
        if understanding["ambiguous"] and self.config.mode == "improved":
            candidates = sorted({c.section_path for c in self.chunks if c.is_current})[:4]
            text = "Could you clarify which limit you mean? I found: " + ", ".join(candidates)
            return Answer(text, [], [], 0.0, False, text, self.config.mode,
                          {"rewrite": 0.0, "search": 0.0, "rerank": 0.0,
                           "generate": 0.0, "total": (time.perf_counter()-started)*1000},
                          Usage(), self.config.provider,
                          rewritten, filters)
        key = cache_key(rewritten, filters, self.config.mode, user.groups if user else [])
        if not session_id or not history:
            cached = self.cache.get(key)
            if cached:
                return cached
        vector = self.embedder.embed([rewritten])[0]
        candidates = self.index.search(rewritten, vector, filters, 3 if self.config.mode == "baseline" else 30)
        ranked = candidates if self.config.mode == "baseline" else rerank(rewritten, candidates, self.config.rerank_threshold)
        context = assemble(ranked, self.config.token_budget, 99 if self.config.mode == "baseline" else self.config.per_document_cap)
        conf = confidence(rewritten, context)
        if self.config.mode == "improved" and (
                conf < self.config.abstain_threshold or not has_sufficient_evidence(rewritten, context)):
            text = "I don't have that information in the knowledge base."
            result = Answer(text, [], context, conf, True, None, self.config.mode,
                            {"rewrite": 0.0, "search": 0.0, "rerank": 0.0, "generate": 0.0,
                             "total": (time.perf_counter()-started)*1000},
                            Usage(), self.config.provider,
                            rewritten, filters)
        else:
            text, citations, usage = generate(rewritten, context, self.config.mode == "baseline")
            result = Answer(text, citations, context, conf, False, None, self.config.mode,
                            {"rewrite": 0.0, "search": 0.0, "rerank": 0.0, "generate": 0.0,
                             "total": (time.perf_counter()-started)*1000},
                            usage, self.config.provider, rewritten, filters)
        if session_id:
            self.sessions.add(session_id, ChatTurn(question, result.text, understanding["entities"]))
        if not session_id or not history:
            self.cache.set(key, result)
        return result
