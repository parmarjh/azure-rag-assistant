# 5-minute demo & architecture video — running order

Everything below is runnable from this repository with no credentials. Start the API first
(`uvicorn api.main:app`) and keep a terminal open next to the browser.

## 0:00-1:15 — Architecture (screen: `docs/diagrams/architecture.png`)

Cover, in this order:

- **Shape of the system:** event-driven ingestion (Blob → Event Grid → Durable Functions →
  Document Intelligence → Azure OpenAI embeddings → Azure AI Search) is deliberately separate
  from the synchronous request path (Front Door → APIM → App Service orchestrator → Search +
  Azure OpenAI). Ingestion is bursty and re-runnable; answering is latency-sensitive. Coupling
  them means a re-index storm degrades chat.
- **Why Azure AI Search:** it is the only component here that gives BM25, HNSW vectors, a
  trained semantic reranker, and `filter`-time security trimming in *one* query — so ACL
  enforcement and version filtering happen inside retrieval rather than as a post-filter that
  silently shrinks top-k.
- **Hybrid + semantic, not either/or:** vectors alone miss exact identifiers ("Tier 2",
  "$5,250", "gpt-4o"); BM25 alone misses paraphrase. Fuse with RRF, then rerank the top ~50.
- **Identity and secrets:** Entra ID group claims flow from the token into the search filter;
  managed identity everywhere; no keys in app config.
- **10k → 5M documents:** the answer is *not* a bigger index — it is partitioning by domain
  with an alias, PTU-backed embeddings with a queue, and incremental checksum-based re-embedding.

## 1:15-2:15 — Working chatbot (screen: browser at `127.0.0.1:8000`)

Three questions, chosen to show grounding, versioning and refusal:

1. `How many weeks of paid parental leave do eligible employees receive?` — answers with the
   citation; click the citation to show document + section, not just filename.
2. `What is the list price per seat per month for the Enterprise tier?` — answers `$109` from
   the 2026 rate card. Say: both rate cards are indexed; the 2025 one was filtered out by
   `is_current`, and it *does* come back for "What was the price in 2025?".
3. `What is Northwind Traders' stock ticker symbol?` — abstains. Say: this is the metric that
   went 100% → 0% hallucination in that category.

## 2:15-3:30 — Two failure diagnoses (screen: terminal)

Pick two of these; the detail is in `docs/FAILURE_ANALYSIS.md`.

**(a) Correct document, wrong chunk.** Show the baseline answering the Tier-2 hotel cap
question from the wrong part of the travel policy:

```bash
python -m rag.cli ask "What is the nightly hotel rate cap for a Tier 2 city?" --mode baseline
python -m rag.cli ask "What is the nightly hotel rate cap for a Tier 2 city?" --mode improved --show-retrieved
```

The line to say: *the evaluation told me where to look* — context fact recall was 1.0 while
answer correctness was 0.0, so retrieval had the fact and generation lost it. Root causes were
fixed-window chunking that split the table from its heading, a lexical scorer with no IDF, and
carrying the RRF fusion score forward as the relevance score, which collapsed every candidate
into a 0.002-wide band and broke ranking, the confidence margin and the abstain threshold at
once. Now the right section scores 0.91 against 0.48.

**(b) Wrong answer with a valid-looking citation — a bug in my own pipeline.** The question
"For a 300-seat annual prepaid contract in 2026, what combined discount applies and who has to
approve **it**?" answered from `PasswordPolicy.docx` and `VPNGuide.pdf`, correctly cited. Cause:
the department filter was regex-derived from the question text, case-insensitively, so the
pronoun "it" was read as the IT department and hard-filtered the corpus. The fix is
architectural, not a regex fix: a department filter is an authorization decision and must come
from the caller's identity, never from question text.

## 3:30-4:20 — Evaluation, before vs after (screen: `eval/results/comparison.md`)

Show the table and read the shape of it, not every row: retrieval hit rate 37.9% → 100%,
section hit rate 0 → 0.95, MRR 0.28 → 0.96, answer correctness 0.15 → 0.80, hallucination
15.8% → 0%, citation precision 0.35 → 0.90, correct behaviour 76.3% → 92.1%, at roughly 3x the
offline latency and +47% prompt tokens.

One diagnosis worth calling out: retrieval hit 100% while correctness was still 0.63, because
answers quoted the sentence *next to* the number ("...must meet the following minimum
requirements:" instead of the `12 characters` row). Retrieval metrics can't see that — it is an
answer-unit selection defect, and fixing it (header-bound table rows, answer-shape scoring,
whole sentences) moved correctness to 0.80 without touching the guardrails.

Then attribute changes to causes: section-aware chunking and hybrid+rerank moved *retrieval*;
version metadata killed stale-version leakage; sufficiency gates and citation/numeric
validation moved *hallucination*; decomposition and per-document caps moved the multi-document
category. Groundedness stayed at 1.0 in both modes and I would not present that as a win — the
offline generator is extractive, so it is near-tautological, and it is the first metric to
watch when the Azure generative path is switched on.

Then the most interesting slide-free moment: my *first* sufficiency gate scored 63.2% correct
behaviour — below the 76.3% baseline — with retrieval already perfect. It required the two
rarest query concepts to co-occur in one chunk, so one incidental rare word ("the **list** price
per seat") could veto a correct answer. Replacing it with IDF-mass coverage took it to 92.1%.
A guardrail is a precision/recall trade and has to be measured like a ranker.

And the honest residual: 3 of 38 items are still wrong, all over/under-abstention rather than
hallucination — e.g. "What about the Starter tier?" abstains because the tier and the
cancellation terms live in different sections. I kept the gate.

## 4:20-5:00 — What I would change before production

- Replace the lexical/heuristic sufficiency gate with an LLM grader plus Azure AI Foundry
  continuous evaluation on a golden set gated in CI; the heuristic is a stand-in for offline
  determinism.
- Document-level evidence aggregation to fix the residual over-abstention without weakening
  guardrails.
- Move ACL from department strings to Entra group object IDs on every chunk, with a
  negative-path test in CI (an Engineering token must never retrieve HR).
- Semantic ranker + PTU capacity planning, answer/embedding caching keyed by identity, and
  per-department cost dashboards from the per-stage telemetry already emitted.
- The Azure path in this repository is real code but has not been run against live resources —
  first production step is a subscription and an end-to-end smoke test.
