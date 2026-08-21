# Step 5 — Architecture & problem-solving answers

## 1. Retrieval quality: 5 chunks retrieved, only 1 relevant

**Debug in this order, because each step tells you whether the next one matters.**

1. **Is the answer even in the index?** Grep the chunk store for the expected fact. If it is
   missing, this is an ingestion bug (parser dropped a table, chunk boundary split the row,
   document never indexed), not a retrieval bug. This is the single most common cause and the
   cheapest to check.
2. **Is it retrievable at all?** Run the same query with `top_k=50` and no filters. If the
   right chunk appears at rank 23, it is a *ranking* problem; if it never appears, it is a
   *representation/filter* problem (wrong filter excluded it, embedding mismatch, or the
   chunk's text lacks the query's discriminating terms).
3. **Split lexical vs semantic.** Run BM25-only and vector-only separately. Vector wins and
   BM25 loses → paraphrase/vocabulary gap; BM25 wins and vector loses → the embedding is
   diluted (chunk too large, or many near-identical chunks). Hybrid + RRF then dominates both.
4. **Inspect what the 4 irrelevant chunks are.** Boilerplate (headers/footers/signature
   blocks) means the chunker is emitting junk; near-duplicates from the same document means
   no per-document cap; chunks from a superseded version means missing `is_current` filtering.
5. **Measure, do not eyeball.** Score hit rate / MRR / section-hit on the golden set before and
   after each change (this repo's `eval/run_eval.py`), one change at a time.

**Fixes, in the order of payoff I actually observed on this corpus:** heading-breadcrumb
enriched, section-aware chunks (~350 tokens, 80 overlap) → hybrid + RRF → semantic reranking
of a large candidate pool (30-50 → 8) → metadata filters (`is_current`, department) →
per-document cap + neighbour expansion → query rewriting/decomposition. Raising `top_k` alone
is the worst fix: it dilutes the prompt, raises cost and latency, and hides the real problem.

## 2. Latency: 3 s → 12 s

Latency is additive across stages, so **instrument per stage first, guess never**. With
Application Insights the request trace already carries spans for rewrite / search / rerank /
generation / post-validation, so start by comparing p50 and p95 per span against the baseline.

Typical culprits, and the signal that identifies each:
- **Generation tokens grew** (most common): completion or prompt token count in telemetry
  climbed — a bigger `top_k`, longer chunks, unbounded conversation history, or a prompt
  change. Fix: token budget, per-doc cap, condensed history, `max_tokens`, streaming.
- **Azure OpenAI throttling / queueing**: 429s and rising `retry-after`; PTU utilisation at
  100%. Fix: PTU scale-out or a second region with a load balancer, backoff with jitter.
- **Model or deployment change**: someone moved from a mini model to a large one, or the
  region changed. Check deployment name + region in the span attributes.
- **Search side**: semantic ranker enabled on a huge candidate set, partition/replica pressure,
  or a filter that forces a full scan. Search's own diagnostics give per-query duration.
- **Extra hops added**: a new reranker, a multi-query fan-out done sequentially instead of
  concurrently, a cache that started missing (check cache hit rate first — a cache regression
  looks exactly like a global slowdown).
- **Cold starts / networking**: Functions or Container Apps scaled to zero, DNS/private-endpoint
  changes, or client-side gzip disabled.

Then fix the dominant span only: parallelise independent sub-queries, cap the candidate pool,
cache aggressively (embedding + semantic answer cache), stream the first token, and set an
end-to-end deadline with graceful degradation (skip reranking rather than time out).

## 3. Scale: 10k → 5M documents

- **Shard the index**: index-per-domain/tenant behind aliases, with a routing layer that picks
  the candidate index set from the user's claims. Smaller ANN graphs → better recall per ms and
  a hard data-isolation boundary.
- **Compress vectors**: truncate `text-embedding-3-large` to 1024 dims, scalar/binary
  quantization, `stored=false` for raw vectors, oversample + rescore to recover recall. Vector
  storage, not document count, is what actually breaks the bank.
- **Two-stage retrieval**: cheap filtered candidate generation (BM25 and/or compressed ANN,
  top 200) → cross-encoder/semantic rerank of the top 50. Never rerank thousands.
- **Industrialise ingestion**: queue-based (Service Bus/Event Hubs) partitioned workers,
  separate backfill and incremental lanes, chunk-level checksums so re-ingestion re-embeds
  only what changed, dead-letter + replay, and idempotent upserts keyed by content hash.
- **Freshness and lifecycle**: tombstones, `expiry_date`, nightly compaction, alias-based
  blue/green rebuilds gated on the evaluation suite.
- **Metadata store becomes load-bearing**: Cosmos DB as the document registry (version chains,
  ACLs, checksums, lineage) — you cannot answer "why did we return this?" at 5M docs without it.
- **Operate on quality signals**: per-domain golden sets in CI, sampled online judging,
  alerts on hit-rate/abstention drift per index, and cost per answered question as a KPI.
- Cost posture flips to PTU + caching; embeddings become a batch-economics problem
  (batch API, off-peak, dimension reduction).

## 4. Security: HR documents must never be retrievable by Engineering users

Enforce access **at retrieval**, derived from identity — never in the prompt, never after
generation.

1. **Identity**: Entra ID sign-in; APIM validates the JWT; the app reads group/app-role claims
   from the validated token. A client-supplied department is untrusted input.
2. **Labels at ingestion**: every chunk carries `security_groups` (a collection field) derived
   from the source of truth (SharePoint/Graph permissions, or the container/folder for a simple
   corpus), plus `department` and classification. Ingestion fails closed: no ACL → no index.
3. **Filter inside the query**: `security_groups/any(g: search.in(g, 'hr,all-staff'))` combined
   with the user's other filters. Azure AI Search applies this as part of retrieval, so the
   restricted chunk never reaches the model and never enters a cache key that another user
   could hit (cache keys include the group set).
4. **Defence in depth**: post-retrieval assertion in the app that drops (and alarms on) any
   chunk the caller may not read; per-department index or per-BU search service when the
   isolation requirement is contractual/regulatory rather than best-effort; separate storage
   containers with distinct RBAC; CMK for the most sensitive domain.
5. **Prove it continuously**: negative test cases in the golden set (an Engineering user asking
   an HR question must abstain — item `C01` in this repo's eval set), audit logs of every
   retrieval with user, filters and returned doc ids, and periodic access reviews.
6. **Don't leak through side channels**: no cross-user answer cache without groups in the key,
   no document titles in "sources" lists the user cannot open, no error messages that confirm a
   document exists.

## 5. Cost: Azure OpenAI spend spikes

**Identify** (Cost Management + App Insights, in that order):
1. Split spend by deployment and by model in Cost Management — embeddings vs chat, and which
   deployment moved.
2. In App Insights, group requests by department/user/endpoint over the spike window and look at
   *tokens per request* and *requests per minute* separately. Four distinct shapes:
   traffic growth, prompt bloat, retry storms, or a re-ingestion that re-embedded the corpus.
3. Check the cache hit rate. A silent cache regression (bad key, TTL of 0, Redis eviction)
   presents exactly as a cost spike with flat traffic.
4. Check for loops: a failed post-validation causing endless regeneration, an agent retry loop,
   or a health check that calls the model.

**Optimise**, roughly in order of return:
- **Tokens**: cut the context, not the answer — smaller/denser chunks, per-doc cap, top 5-8
  reranked chunks instead of 20 unranked ones, condensed conversation history, `max_tokens`
  caps, and no whole-document stuffing.
- **Caching**: exact + semantic answer cache keyed by (normalised question, filters, groups);
  embedding cache keyed by content hash. Repeated FAQs are the majority of enterprise traffic.
- **Model routing**: gpt-4o-mini (or smaller) for rewrite, classification, ambiguity detection
  and judging; the expensive model only for the final grounded answer. Route "easy" questions
  (high retrieval confidence, single-doc) to the cheap model entirely.
- **Embeddings**: embed only changed chunks (checksums), batch, and reduce dimensions.
- **Commercial**: PTU/reservations once the p95 load is predictable; quotas per department at
  APIM so one team cannot spend the whole budget; budget alerts and a cost-per-answer SLO.

## 6. Production failure: occasionally very wrong answers with valid-looking citations

This is the most dangerous failure mode because the citation makes it credible. It almost
always means **the answer was generated from a chunk that is topically right but factually
wrong for the question** (wrong version, wrong plan tier, wrong department) — or the citation
was attached to a claim that came from a different chunk.

Debugging walk, stage by stage, on the *specific* logged request (this is why every request
logs the rewritten query, filters, retrieved ids and scores):

1. **User query**: was it ambiguous or a follow-up? Check the *rewritten* query — a bad
   condensation ("What about Standard?" → "What is Standard?") sends retrieval somewhere else
   while the answer still sounds confident.
2. **Retrieval**: were the correct chunks in the candidate set at all? Replay the logged query
   with a larger `top_k` and no filters. If the right chunk was absent, look for over-filtering
   (`is_current` on a genuinely historical question) or a chunking defect.
3. **Ranking**: look at the score distribution. A flat distribution (top1 ≈ top8) is the
   signature of near-duplicate documents — the 2025 vs 2026 rate-card case — and means ranking
   cannot be the tie-breaker; metadata must be.
4. **Context**: reconstruct the exact prompt from telemetry. Common findings: the relevant
   chunk was truncated at the token budget; two conflicting chunks were both present with no
   date to distinguish them; a table was split so a row lost its header.
5. **Prompt**: does it *require* citing per claim and forbid outside knowledge? Does it tell the
   model what to do with conflicting sources (prefer the current effective date and say so)?
6. **LLM**: temperature > 0, a model/deployment change, or a long prompt where the fact sits in
   the middle (lost-in-the-middle) — re-run the same context at temperature 0 and with the
   chunk order reversed to see if the answer is order-sensitive.
7. **Citation binding**: verify the citation was *derived* from the sentence's source chunk and
   not appended afterwards. Post-validation must check that every numeric claim in the answer
   appears in a cited chunk (this repo does exactly that and abstains on failure) — otherwise
   "valid-looking citation" is unfalsifiable.

**Systemic fixes**: version/effective-date metadata + current-only default filtering (kills the
most common instance), sentence-level citation validation with regeneration-then-abstain,
confidence gating, temperature 0, deterministic tie-breaking, conflict-awareness in the prompt,
and adding every reproduced incident to the golden set so a regression fails CI rather than a
customer.
