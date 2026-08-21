# Enterprise Knowledge Assistant — RAG on the Azure AI stack

A grounded, citation-first question-answering assistant over a small enterprise document set
(HR, Finance, IT, Legal, Sales). The point of this repository is not the chatbot: it is the
**measured** difference between a naive RAG pipeline and a hardened one, and the diagnosis
that connects the two.

| | Baseline | Improved |
|---|---|---|
| Chunking | fixed 1000 characters, no overlap, filename metadata only | section-aware, ~350 tokens with ~80 overlap, heading breadcrumb + version metadata |
| Retrieval | pure vector, top-3, no filters | hybrid BM25 + vector, RRF fusion, reranking, ACL + effective-date filters, neighbour expansion |
| Query handling | question used verbatim | follow-up condensation, comparison decomposition, ambiguity detection |
| Answering | answer whatever was retrieved | sufficiency gates, abstention, clarification, citation + numeric validation |

Headline evaluation deltas over the 38-question set (full table in
[`eval/results/comparison.md`](eval/results/comparison.md)):

| Metric | Baseline | Improved |
|---|---:|---:|
| Retrieval hit rate @k | 37.9% | **100%** |
| Section hit rate | 0 | **0.95** |
| MRR | 0.28 | **0.96** |
| Answer correctness | 0.15 | **0.80** |
| Citation precision / recall | 0.35 / 0.31 | **0.90 / 0.90** |
| Correct behaviour (answer vs abstain vs clarify) | 76.3% | **92.1%** |
| Hallucination rate | 15.8% | **0%** |
| Stale-version leak | 0% | 0% |
| Mean latency (offline, local provider) | 9.7 ms | 32.4 ms |
| Mean prompt tokens | 426 | 627 |

The trade is explicit: ~3x the offline work and +47% prompt tokens for +62 points of hit rate
and no hallucinations. Three of 38 items remain wrong in improved mode and are documented as
residual failures rather than tuned away — see
[the residual section](docs/FAILURE_ANALYSIS.md#residual-failure-modes-what-i-would-fix-next).

## Contents

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — production Azure architecture, service
  choices, semantic vs vector vs hybrid, 10k vs 10M scaling, security, cost. Rendered diagram:
  [`docs/diagrams/architecture.png`](docs/diagrams/architecture.png).
- [`docs/DESIGN.md`](docs/DESIGN.md) — the implementation design this repository follows.
- [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) — the six failure scenarios: symptom,
  root cause found by instrumenting the pipeline, fix, and the metric that moved.
- [`docs/ANSWERS.md`](docs/ANSWERS.md) — Step 5 architecture and problem-solving answers
  (retrieval quality, latency, scale, security, cost, wrong-answer-with-valid-citation).

## Pipeline

```text
Documents ──► Parsing ──► Chunking ──► Embeddings ──► Azure AI Search
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
        Query understanding (condense / decompose / classify)
                        ▼
        Hybrid retrieval ──► RRF ──► rerank ──► filters (ACL, effective date)
                        ▼
        Context assembly (token budget, per-document cap, neighbours)
                        ▼
        LLM ──► sufficiency + citation/numeric validation
                        ▼
        Grounded answer + citations  |  abstention  |  clarifying question
```

## Two providers, one pipeline

Every stage is behind an interface with two implementations, selected by `RAG_PROVIDER`:

| Stage | `azure` | `local` (default) |
|---|---|---|
| Embeddings | Azure OpenAI `text-embedding-3-large` | deterministic feature-hash vectors |
| Index | Azure AI Search (hybrid + semantic ranker, security filters) | in-process BM25 + vector index with the same filter semantics |
| Generation | Azure OpenAI chat deployment, grounded prompt, temperature 0 | deterministic extractive composer |
| Telemetry | Application Insights / OpenTelemetry | structured logs |

The `local` provider exists so that the pipeline, the guardrails and the whole evaluation are
**reproducible offline with no keys and no network** — which is also what makes the
baseline-vs-improved comparison in this repository trustworthy rather than sampled from a
non-deterministic endpoint. The Azure path is real code (SDK calls, index schema, semantic
configuration, managed-identity auth) but, as noted in [Limitations](#limitations), it has not
been executed against live Azure resources.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# ingest + ask (offline, no credentials needed)
python -m rag.cli ingest --source data/KnowledgeBase --mode improved
python -m rag.cli ask "How many weeks of paid parental leave do eligible employees receive?"

# same question against the naive pipeline
python -m rag.cli ask "What is the list price per seat per month for the Enterprise tier?" --mode baseline

# API + minimal chat UI on http://127.0.0.1:8000
uvicorn api.main:app --reload

# evaluation: writes eval/results/{baseline,improved}.json and comparison.md
python -m eval.run_eval --mode both
pytest tests/ -q
```

To run against Azure, set the following and the pipeline switches provider automatically
(API keys are optional — without them it uses `DefaultAzureCredential`):

```bash
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
RAG_CHAT_MODEL=gpt-4o                 # deployment name
RAG_EMBEDDING_MODEL=text-embedding-3-large
APPLICATIONINSIGHTS_CONNECTION_STRING=...   # optional
```

`infra/main.bicep` provisions the corresponding resources; `python -m rag.cli provision`
creates the search index with its vector profile, semantic configuration and filterable
security/version fields.

## Repository layout

```text
src/rag/
  parsing.py      PDF / DOCX / XLSX → sections (headings, pages, tables, boilerplate stripped)
  metadata.py     version chain, effective dates, is_current, department ACL groups
  chunking.py     baseline (fixed-size) and improved (section-aware + breadcrumb) chunkers
  providers/      embeddings + LLM (Azure OpenAI / local)
  index/          Azure AI Search and local BM25+vector index behind one interface
  query.py        follow-up condensation, comparison decomposition, ambiguity detection
  retrieval.py    hybrid search, RRF fusion, rerank, neighbour expansion, assembly
  guardrails.py   sufficiency gates, confidence, citation + numeric validation, clarification
  generation.py   grounded prompt (Azure) / extractive composer (local), citation binding
  pipeline.py     the orchestrator: RagPipeline.ingest() / .answer()
  acl.py conversation.py cache.py telemetry.py context.py rerank.py config.py
api/              FastAPI endpoints (/chat, /health) + a minimal static UI
eval/             dataset, deterministic metrics, optional LLM judge, runner, results
tests/            pipeline behaviour tests + evaluation-harness tests
```

## Failure scenarios

Each scenario is reproducible from the CLI and covered by the evaluation set; the root-cause
analysis is in [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md).

| # | Scenario | Fix in this repository |
|---|---|---|
| 1 | Correct document, wrong chunk | section-aware chunking with heading breadcrumbs, hybrid retrieval with IDF-weighted rerank, table rows as first-class units |
| 2 | Answer spread across sections/documents | sub-query decomposition per entity, RRF fusion, per-document cap so one document cannot monopolise context, neighbour expansion |
| 3 | Conflicting versions (2025 vs 2026 rate card) | version chain + effective dates in metadata, `is_current` filter by default, relaxed when the question is explicitly historical |
| 4 | Hallucination / missing information | corpus-grounded sufficiency gates (unknown-concept and concept co-occurrence), confidence threshold, citation + numeric validation with regenerate-then-abstain |
| 5 | Ambiguous query ("What is the limit?") | detect an under-specified head noun resolving to multiple documents, and ask a clarifying question that names the candidate facets instead of guessing |
| 6 | Conversational follow-ups | condense the follow-up into a standalone question using carried entity slots; only the condensed question reaches retrieval, never raw history |

## Evaluation

`eval/dataset.jsonl` holds 38 questions across straightforward, multi-document,
version-conflict, no-answer, ambiguous, follow-up and access-control categories, each with
expected documents/sections, expected facts and — where relevant — **forbidden** facts (e.g.
the superseded `$99` price), which is how stale-version leakage is measured.

Metrics are computed deterministically (no LLM required): hit rate, expected-document recall,
section hit rate, MRR, retrieved-chunk precision, context fact recall, answer correctness,
groundedness, citation precision/recall, correct-behaviour rate (answer vs abstain vs clarify),
hallucination rate, stale-version leak rate, latency percentiles and token/cost estimates.
`--judge llm` adds an Azure OpenAI judge for correctness and groundedness when credentials
exist.

## Bonus features

Query rewriting · hybrid search · reranking · metadata filtering · confidence scoring ·
guardrails and abstention · document-level access control (department groups, enforced at
search time and re-checked after retrieval) · answer/embedding caching keyed by identity ·
automated evaluation pipeline · Application Insights / OpenTelemetry instrumentation with
per-stage latency, tokens and cost per request.

## Limitations

- **The Azure path has not been executed against live Azure resources** — no subscription was
  available for this exercise. All reported numbers come from the `local` provider.
- The local generator is extractive, so its groundedness is near-tautological; groundedness is
  the metric to watch first when the Azure generative path is enabled.
- Feature-hash embeddings are not semantically meaningful. Lexical/BM25 signal therefore does
  more of the work offline than it would with `text-embedding-3-large`.
- Latency and cost figures are offline estimates and are not representative of Azure OpenAI.
