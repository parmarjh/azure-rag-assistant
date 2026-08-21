# Failure Analysis — symptom, root cause, fix, measured effect

The baseline is deliberately a naive RAG pipeline: fixed 1000-character windows, no overlap,
filename-only metadata, pure vector search, top-3, no rewriting, no reranking, no filters and
no sufficiency gate. Everything below was found by running the evaluation set against it (and
then against the improved pipeline), reading the per-item retrieval traces, and instrumenting
the stage that the trace implicated. Numbers refer to `eval/results/comparison.md`.

Two of these were bugs in **my own** improved implementation, found by evaluation rather than
by reading the code — they are included because how a pipeline is debugged is the point of the
exercise.

---

## Scenario 1 — Correct document, wrong chunk

**Symptom.** "What is the nightly hotel rate cap for a Tier 2 city?" retrieved
`TravelPolicy.docx` but answered with the pre-approval sentence rather than the `$250` row.
The evaluation separated these two failures for me: context fact recall was 1.0 while answer
correctness was 0.0, so retrieval had the fact and generation lost it.

**Root causes.** Three distinct ones, in order of impact:

1. *Chunk boundaries ignored structure.* Fixed 1000-character windows cut tables in half and
   dropped the heading that gives a row meaning, so the embedded text of the winning chunk did
   not contain the words "Tier 2" and "nightly cap" together. Section hit rate on the baseline
   was **0**.
2. *Lexical scoring had no IDF.* The local hybrid scorer combined an unweighted term-overlap
   ratio with cosine similarity as a weighted sum of two incomparable scales, so common words
   dominated: "What is the price per seat for the Enterprise tier?" ranked `Discounts.xlsx` and
   `Benefits.pdf` above `Pricing2026.pdf`.
3. *A degenerate score band.* After adopting RRF I carried the fusion score (`1/(60+rank)`)
   forward as the relevance score. Every candidate then scored 0.287-0.289, which silently
   broke ranking, the confidence margin term and the abstain threshold at the same time — the
   top four results for one query were ExpensePolicy §5, TravelPolicy §5, VPNGuide §5 and
   PasswordPolicy §4, all within 0.002 of each other.

**Fix.** Section-aware chunking (~350 tokens, ~80 overlap) with a document/section breadcrumb
prepended before embedding; table rows kept intact and made first-class retrieval and answer
units; BM25 with real corpus IDF as the lexical scorer; RRF used only for candidate fusion and
ordering, with a scored rerank stage (normalised BM25 + cosine + IDF-weighted coverage)
producing the number that everything downstream consumes.

**Effect.** Hit rate 37.9% → 100%, section hit rate 0 → 0.93, MRR 0.28 → 0.96, retrieved-chunk
precision 0.16 → 0.35. The Tier-2 query now scores TravelPolicy §4 at 0.91 against 0.48 for
the runner-up — a usable margin instead of a coin flip — and answers `Tier 2 | Chicago,
Denver, Austin, Toronto | $250`.

---

## Scenario 2 — Information spread across sections and documents

**Symptom.** "Compare the confidentiality survival period in the NDA template with the one in
the vendor service agreement" retrieved neither contract on the baseline — it returned
`LeavePolicy.pdf`, `ExpensePolicy.pdf` and `TravelPolicy.docx` and answered with a sentence
about unplanned absences. Multi-document hit rate was 33.3%. Even with retrieval fixed, a
single fused query and an uncapped top-k tend to fill the context with whichever document
matches most densely, so one side of the comparison is silently missing.

**Fix.** Detect comparison intent from the query, decompose into one sub-query per entity,
retrieve each independently, fuse with RRF, and cap chunks per document so no document can
monopolise the context window. Neighbour expansion pulls the adjacent chunk of a selected
section when the answer straddles a boundary, within the token budget.

**Effect.** Multi-document hit rate 33.3% → 100% with 100% correct behaviour in that category;
both documents are now cited, in reference order. The comparison answers state each side with its own citation, e.g. three years
(NDA §4) versus five years (Vendor Agreement §4).

---

## Scenario 3 — Conflicting versions

**Symptom.** On the baseline, "What is the list price per seat per month for the Enterprise
tier?" retrieved `Benefits.pdf` and answered "Employee stock purchase plan: 15% discount" —
neither rate card was retrieved at all (version-conflict hit rate 0). The subtler failure
appears once retrieval works: with the current-only filter disabled, "How much does the
Advanced Analytics add-on cost per seat?" ranks `Pricing2026.pdf` §5 at 0.777 and the
superseded `Pricing2025.pdf` §4 at 0.734. The two rate cards are near-identical in wording, so
a 0.04 margin is all that separates the current price from the stale one — relevance ranking
is not a version-resolution mechanism.

**Fix.** Parse version, effective date and supersedes-chain into metadata at ingestion, mark
one document per chain `is_current`, and filter to current documents by default. When the
question carries historical intent (an explicit year, "previous", "prior"), the filter is
relaxed so the superseded card can be retrieved, and the answer states the effective date.

**Effect.** Version-conflict hit rate 0 → 100%, stale-version leak rate 0. The evaluation
enforces this from both directions: `$109` is a required fact and `$99` a *forbidden* fact for
the current-price question, while "What was the Enterprise tier price in 2025?" requires
`$99` from `Pricing2025.pdf`.

---

## Scenario 4 — Hallucination and missing information

**Symptom.** Every no-answer question was answered by the baseline (100% hallucination rate in
that category) — "What is Northwind Traders' stock ticker symbol?" came back with a sentence
about VPN access. An intermediate build of the improved pipeline, before the sufficiency gates
were in place, did the same thing more convincingly: the same question produced "This policy
defines what business expenses Northwind Traders will reimburse..." and "Who is the Chief
Executive Officer?" produced the vendor agreement preamble — fluent, on-brand, correctly
cited, and completely unrelated to the question.

**Root cause of my first attempt failing.** My initial gate required 60% of the query's
distinctive terms to appear in the retrieved context. That is a *coverage* test, and coverage
is the wrong signal: it punished long comparison questions whose evidence was fully present
(M01 abstained with context fact recall 1.0) while passing unanswerable questions whose words
happen to be common across the corpus.

**Fix — sufficiency as concept presence, not term coverage.** Two gates, both grounded in the
indexed corpus rather than in the model's opinion:

1. **Unknown-concept gate.** If a non-numeric content term of the query occurs nowhere in the
   corpus (with edit-distance-1 and morphological tolerance so "travelling"/"traveling" and
   "cancellation"/"cancellable" are not false positives), the concept is not in the knowledge
   base: abstain. This catches "ticker", "symbol", "refund", "Germany".
2. **IDF-mass coverage gate.** Take the query's three most informative concepts, weight each by
   its corpus IDF, and require that a *single* retrieved chunk covers at least 60% of that mass.
   This is what separates answerable from unanswerable questions whose individual words all
   exist in the corpus somewhere.

   I first implemented gate 2 as "the two rarest concepts must co-occur in one chunk", which
   looked principled and was too brittle to ship: one incidental rare word decided the outcome,
   so "What is the **list** price per seat per month for the Enterprise tier?" abstained while
   citing the correct chunk. That version scored **63.2% correct behaviour — worse than the
   baseline's 76.3%** even with perfect retrieval, which is the useful lesson: a guardrail is a
   precision/recall trade and has to be measured like a ranker, not asserted. Weighted mass
   coverage degrades gracefully (a missing incidental term cannot veto; missing *rare* concepts
   still do) and took correct behaviour to 92.1%. The threshold was picked from a sweep —
   0.4/0.5/0.6 were behaviourally identical on the calibration set and 0.7+ started refusing
   answerable questions, so 0.6 is the conservative end of the flat region rather than a
   fitted value.
3. **Entity grounding on the citation.** After generation, entity mentions in the question must
   actually occur in the chunks the answer cites. "What is the Standard tier price per seat?"
   used to answer "Enterprise $109" with a valid citation — there is no Standard tier in this
   corpus — and now abstains, because the cited Subscription Tiers chunk contains Starter,
   Professional and Enterprise but not Standard. This is the wrong-answer-with-valid-citation
   failure caught structurally rather than by prompt instruction.

Confidence (top score, margin over rank 3, query coverage) is a third, threshold-configurable
gate, and every answer is finally checked by citation and numeric validation: each `[n]` must
resolve to a context chunk and every currency/percentage/number asserted must appear in a
cited chunk, otherwise the answer is regenerated once and then abstained.

**Effect.** Hallucination rate 15.8% → 0%, no-answer behaviour 0% → 100% correct, citation
precision 0.35 → 0.90. The cost is over-abstention, which is the residual failure mode below.

---

## Scenario 5 — Ambiguous query

**Decision: clarify rather than guess, but only when the ambiguity is real.** "What is the
limit?" maps to expense category limits, hotel caps, per diems, discount caps, API call limits
and wellness/tuition maxima. Retrieving anyway means a confidently wrong answer; asking always
is annoying. The implemented rule: an under-specified head noun ("limit", "threshold", "cap",
"how much can I spend") with no disambiguating entity in the query **and none carried by the
conversation**, whose candidates resolve to two or more distinct documents, triggers a
clarifying question naming the candidate facets (derived from section headings and document
titles, capped at four). If the conversation already disambiguates it, the assistant answers.

**Effect.** Ambiguous behaviour accuracy 0% → 66.7% (2 of 3); the baseline answered all three
ambiguous questions with whatever ranked first. The rule is deliberately conservative about
firing — an earlier version clarified "What is the password lockout threshold?", which is not
ambiguous at all, because it keyed on the head noun alone and ignored the modifiers.

---

## Scenario 6 — Conversational context

**Symptom.** Concatenating history into the retrieval query pollutes it: after two turns the
embedding drifts toward whatever was discussed earlier. Sending only the follow-up is worse —
"What about the Starter tier?" carries no topic at all, and the baseline retrieved
`LeavePolicy.pdf`, `TravelPolicy.docx` and `ExpensePolicy.pdf`, answering "Benefits Guide."

**Fix.** Condense, don't concatenate. The follow-up is detected structurally (very short, or
opening with "what about"/"and", or pronoun-only such as "Do they carry over?"), then rewritten
into a standalone question by substituting the new entity into the previous resolved question,
or by re-attaching the previous topic. Only the condensed question reaches retrieval; raw
history never does. Entity slots (plan/tier, department, year) are carried in session state so
"What about SMS?" → "Are there any exceptions?" still resolves to the MFA policy.

**Effect.** Follow-up hit rate 20% → 100%.

---

## Bugs the evaluation found in my own improved pipeline

1. **The department filter was derived from the question text — and matched a pronoun.**
   "For a 300-seat annual prepaid contract in 2026, what combined discount applies and who has
   to approve **it**?" produced `department='it'` because the department regex was
   case-insensitive, so the corpus was hard-filtered to IT and the answer came from
   `PasswordPolicy.docx` and `VPNGuide.pdf`. Beyond the regex, the design was wrong: a
   department *filter* is an authorization/scoping decision and must come from the caller's
   identity, never from free text in a question, where it can silently hide documents (or, with
   a different index layout, be used to probe them). Filters now derive from `UserContext`
   only.
2. **Retrieval was fixed long before answers were.** With hit rate already at 100% and section
   hit rate at 0.95, answer correctness sat at 0.63: the pipeline cited the right chunk and
   quoted the sentence *next to* the fact — "All Company account passwords must meet the
   following minimum requirements:" instead of the `12 characters` row, "It replaces the 2025
   rate card." instead of the `$45` dinner per diem, the `3 days or fewer` notice band for a
   five-day request. Fifteen of 38 items were of this shape. The diagnosis is worth stating
   because it is invisible to retrieval metrics: when the expected fact is present in the
   assembled context but correctness is zero, the defect is in answer-unit selection, not in
   search. Fixes: table rows carry their header as one unit (so a row is never orphaned from
   its column name), prose units are whole sentences rather than wrapped lines, candidate units
   are scored against the *answer shape* the question implies (currency, quantity, duration,
   bound, actor) with pointer units ("...as follows:", cross-references) demoted, quantities in
   the question select the matching band, and multi-part questions must place at least one unit
   per sub-query. Answer correctness 0.63 → 0.80 and citation precision 0.82 → 0.90, with no
   change to the guardrails.
3. **A metric bug in my own evaluation harness.** Two items were flagged as leaking the
   superseded `$12` price. They did not: the PDF bullet artifact `(cid:127)` contains the
   digits `127`, and my numeric matcher compared against a whitespace-stripped haystack, so
   `12` matched as a substring. Facts are now compared as whole numeric tokens
   (`tests/test_eval_metrics.py` locks this), and the artifacts are stripped at parse time so
   they never reach an answer.

---

## Residual failure modes (what I would fix next)

Three of 38 items are still wrong in improved mode. They are listed here rather than tuned
away, because every one of them is a symptom of a real limit of a lexical guardrail:

- **`F01` — over-abstention on cross-section inference.** "What about the Starter tier?"
  following a cancellation-policy question abstains: the rate card states cancellation terms
  once for all tiers, so no single chunk carries enough of the question's IDF mass. I chose to
  keep the gate rather than weaken the property that holds hallucination at zero; the real fix
  is document-level evidence aggregation (or an LLM sufficiency judge), not a looser threshold.
- **`V03` — over-abstention with the right chunk retrieved.** "What is the API call limit on
  the Professional tier?" retrieves the 2026 tier table at rank 1 and still abstains: the
  answer lives in a wide table row where "API", "Professional" and "limit" are spread across
  columns, so per-chunk mass coverage under-counts it. Table rows need their own evidence
  accounting (header + cell as a unit), which is the same fix as document-level aggregation.
- **`A02` — under-clarification.** "What's the approval threshold?" is answered from the travel
  policy's pre-approval sentence instead of asking which approval. The modifier "approval" is
  itself specific enough for the ambiguity rule to stand down, yet three documents define
  different approval thresholds. Fixing it properly means deciding ambiguity from the *spread*
  of candidate answers (do the top chunks disagree on the value?) rather than from query shape
  — a better rule I would rather implement than fake with a word list.
- **Groundedness is near-tautological offline.** The local generator is extractive, so it
  cannot state an ungrounded number; groundedness stays at 1.0 in both modes and is therefore
  the least informative metric here. It becomes meaningful — and is the first metric to watch —
  once the Azure generative path is enabled.
- **Latency and cost went up** (mean 9.7ms → 32.4ms offline, prompt tokens 426 → 627): the
  improved pipeline runs more searches (one per sub-query), reranks a larger candidate set and
  assembles more context. Offline milliseconds are not predictive of Azure latency, but the
  shape of the trade — roughly 3x work for +62 points of hit rate and zero hallucination — is,
  and per-stage timings are recorded on every request so the cost of each stage stays visible
  in production.
