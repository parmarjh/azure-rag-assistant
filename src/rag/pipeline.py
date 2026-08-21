from __future__ import annotations

import logging
import os
import time

from .acl import build_filter
from .cache import TTLCache, cache_key
from .chunking import baseline_chunks, improved_chunks
from .config import RagConfig
from .context import assemble
from .conversation import SessionStore
from .generation import generate
from .guardrails import (
    clarification_facets,
    confidence,
    has_sufficient_evidence,
    validate_citations_and_numbers,
)
from .index.azure_search import AzureSearchIndex
from .index.local_index import LocalSearchIndex
from .models import Answer, ChatTurn, Chunk, IngestStats, Usage, UserContext
from .parsing import parse_directory
from .providers.embeddings import AzureOpenAIEmbeddingProvider, LocalEmbeddingProvider
from .providers.llm import AzureOpenAILLMProvider, LocalLLMProvider
from .query import understand
from .retrieval import retrieve
from .telemetry import Span, emit_request

log = logging.getLogger("rag")


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        if config.provider == "azure":
            self.embedder = AzureOpenAIEmbeddingProvider(
                config.openai_endpoint or "",
                config.embedding_model,
                os.getenv("AZURE_OPENAI_API_KEY"),
            )
            self.llm = AzureOpenAILLMProvider(
                config.openai_endpoint or "",
                config.chat_model,
                os.getenv("AZURE_OPENAI_API_KEY"),
            )
            self.index = AzureSearchIndex(
                config.search_endpoint or "",
                config.search_index,
                api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            )
        else:
            self.embedder = LocalEmbeddingProvider()
            self.llm = LocalLLMProvider()
            self.index = LocalSearchIndex(self.embedder)
        self.sessions = SessionStore()
        self.cache = TTLCache(config.cache_ttl_seconds)
        self.chunks: list[Chunk] = []

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def ingest(self, source_dir: str) -> IngestStats:
        docs = parse_directory(source_dir)
        self.index.create()
        chunks = []
        for doc in docs:
            if self.config.section_aware:
                chunks.extend(improved_chunks(
                    doc,
                    target=self.config.chunk_size_tokens,
                    overlap=self.config.chunk_overlap_tokens,
                    with_header=self.config.prepend_header,
                ))
            else:
                chunks.extend(baseline_chunks(doc, size=self.config.chunk_size_tokens))
        vectors = self.embedder.embed([c.content for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        self.index.upload(chunks)
        self.chunks = chunks
        return IngestStats(len(docs), len(chunks), self.config.mode)

    def _baseline_retrieve(self, query: str, filters: dict) -> list:
        vector = self.embedder.embed([query])[0]
        return assemble(
            self.index.search(query="", vector=vector, filters=filters, top_k=self.config.candidate_k),
            self.config.context_token_budget,
            self.config.per_doc_cap,
        )[:self.config.top_k]

    def _known_entities(self) -> list[str]:
        values = set()
        for chunk in self.chunks:
            values.update((chunk.filename.rsplit(".", 1)[0], chunk.doc_title))
        return sorted(values, key=lambda value: (-len(value), value))

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        user: UserContext | None = None,
    ) -> Answer:
        started = time.perf_counter()
        history = self.sessions.history(session_id) if session_id else []
        latency: dict[str, float] = {}
        with Span("rewrite") as span:
            understanding = understand(
                question,
                history,
                self._known_entities() if self.config.use_subquery_decomposition else None,
            )
        latency["rewrite"] = span.elapsed_ms
        rewritten = understanding["rewritten"] if self.config.use_query_rewrite else question.strip()
        filters = build_filter(
            user,
            user.department if user else None,
            self.config.filter_current_only and not understanding["version_intent"],
        )
        if not self.config.filter_current_only:
            filters["is_current"] = False
        filters["_baseline"] = not self.config.use_hybrid
        key = cache_key(rewritten, filters, self.config.mode, user.groups if user else [])
        if not session_id or not history:
            cached = self.cache.get(key)
            if cached:
                return cached
        retrieval_timings: dict[str, float] = {}
        if self.config.use_hybrid:
            candidates = retrieve(
                self.index,
                self.embedder,
                rewritten,
                filters,
                top_k=self.config.top_k,
                threshold=self.config.rerank_threshold if self.config.use_rerank else 0.0,
                token_budget=self.config.context_token_budget,
                per_document_cap=self.config.per_doc_cap,
                candidate_k=self.config.candidate_k,
                use_rerank=self.config.use_rerank,
                neighbour_expansion=self.config.neighbour_expansion,
                subqueries=(
                    understanding["subqueries"]
                    if self.config.use_subquery_decomposition
                    else [rewritten]
                ),
                timings=retrieval_timings,
            )
        else:
            with Span("search") as span:
                candidates = self._baseline_retrieve(rewritten, filters)
            retrieval_timings["search"] = span.elapsed_ms
            retrieval_timings["rerank"] = 0.0
        latency.update(retrieval_timings)

        distinct_topics = {(item.doc_id, item.section_path) for item in candidates}
        if (
            self.config.enable_clarification
            and understanding["ambiguous_head"]
            and len(distinct_topics) >= 2
            and not history
        ):
            facets = clarification_facets(candidates, head_noun=understanding["head_noun"] or "limit")
            text = "Could you clarify which limit you mean? I found: " + "; ".join(facets[:4])
            prompt_tokens = len(rewritten.split()) + sum(item.chunk.token_count for item in candidates)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                estimated_cost_usd=prompt_tokens * self.config.prompt_cost_per_1k / 1000,
            )
            result = Answer(
                text, [], candidates, 0.0, False, text, self.config.mode,
                {**latency, "generate": 0.0, "total": (time.perf_counter() - started) * 1000},
                usage, self.config.provider, rewritten, filters,
            )
            emit_request(self._telemetry_record(result, question))
            return result

        conf = confidence(rewritten, candidates)
        should_abstain = self.config.enable_guardrails and (
            conf < self.config.abstain_threshold
            or not has_sufficient_evidence(
                rewritten,
                candidates,
                self.chunks,
                self.config.evidence_pair_fraction,
                self.config.evidence_term_count,
                self.config.evidence_pair_window,
                understanding["subqueries"] if understanding["comparison"] else None,
            )
        )
        citations = []
        usage = Usage()
        if should_abstain:
            text = "I don't have that information in the knowledge base."
            abstained = True
            latency["generate"] = 0.0
        else:
            with Span("generate") as span:
                text, citations, usage = generate(
                    rewritten,
                    candidates,
                    allow_general=not self.config.enable_guardrails,
                    provider=self.config.provider,
                    llm=self.llm,
                    prompt_rate=self.config.prompt_cost_per_1k,
                    completion_rate=self.config.completion_cost_per_1k,
                    focus_queries=(
                        understanding["subqueries"]
                        if self.config.use_subquery_decomposition
                        else None
                    ),
                )
            latency["generate"] = span.elapsed_ms
            abstained = False
            if self.config.enable_guardrails and not validate_citations_and_numbers(
                text, citations, candidates
            ):
                with Span("generate_retry") as span:
                    text, citations, usage = generate(
                        rewritten + " Use only supported facts and cite every numeric claim.",
                        candidates,
                        allow_general=False,
                        provider=self.config.provider,
                        llm=self.llm,
                        prompt_rate=self.config.prompt_cost_per_1k,
                        completion_rate=self.config.completion_cost_per_1k,
                        focus_queries=(
                            understanding["subqueries"]
                            if self.config.use_subquery_decomposition
                            else None
                        ),
                    )
                latency["generate"] += span.elapsed_ms
                if not validate_citations_and_numbers(text, citations, candidates):
                    text = "I don't have sufficient supported information in the knowledge base."
                    citations, abstained = [], True

        latency["total"] = (time.perf_counter() - started) * 1000
        result = Answer(
            text, citations, candidates, conf, abstained, None, self.config.mode,
            latency, usage, self.config.provider, rewritten, filters,
        )
        if session_id:
            self.sessions.add(session_id, ChatTurn(question, result.text, understanding["entities"]))
        if not session_id or not history:
            self.cache.set(key, result)
        emit_request(self._telemetry_record(result, question))
        return result

    @staticmethod
    def _telemetry_record(result: Answer, question: str) -> dict:
        return {
            "question": question,
            "rewritten_query": result.rewritten_query,
            "filters": result.filters,
            "retrieved": [
                {"doc_id": item.doc_id, "score": item.score} for item in result.retrieved
            ],
            "confidence": result.confidence,
            "abstained": result.abstained,
            "clarified": bool(result.clarification),
            "latency_ms": result.latency_ms,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "estimated_cost_usd": result.usage.estimated_cost_usd,
        }
