"""Baseline-vs-improved RAG evaluation runner.

    python -m eval.run_eval --mode both
    python -m eval.run_eval --mode improved --judge llm     # needs Azure OpenAI

Writes eval/results/<mode>.json (per-item + aggregate) and, for --mode both,
eval/results/comparison.md with the side-by-side table used in the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from eval import judge as judge_mod
from eval.metrics import ItemResult, aggregate, score_item

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CORPUS = ROOT / "data" / "KnowledgeBase"

# Evaluation default: a reader with access to every department, so retrieval quality is
# measured without ACL interference. ACL items override this with their own groups.
ALL_GROUPS = ["hr", "finance", "it", "legal", "sales", "all-staff"]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(json.loads(line))
    return items


def run_mode(mode: str, items: list[dict[str, Any]], judge_kind: str, ingest: bool) -> dict[str, Any]:
    from rag.config import get_config
    from rag.models import UserContext
    from rag.pipeline import RagPipeline

    cfg = get_config(mode=mode)
    pipeline = RagPipeline.from_config(cfg)
    if ingest:
        stats = pipeline.ingest(str(CORPUS))
        print(f"[{mode}] ingested: {stats}")

    judge = judge_mod.get_judge(judge_kind)
    results: list[ItemResult] = []
    started = time.time()

    for idx, item in enumerate(items, start=1):
        session_id = f"eval-{mode}-{item['id']}"
        user = UserContext(
            user_id=f"eval-{item['id']}",
            groups=item.get("user_groups", ALL_GROUPS),
            department=item.get("user_department"),
        )
        # Replay conversation history through the same session so follow-up items are
        # answered with the same context-condensation path a real user would hit.
        for prior in item.get("history", []):
            pipeline.answer(prior, session_id=session_id, user=user)

        answer = pipeline.answer(item["question"], session_id=session_id, user=user)
        result = score_item(item, answer)
        if judge is not None:
            context = "\n\n".join(getattr(r, "content", "") or "" for r in (answer.retrieved or []))
            result.llm_judge = judge_mod.judge_item(judge, item, result, context)
        results.append(result)
        flag = "ok " if result.behavior_correct and not result.hallucinated else "BAD"
        print(
            f"[{mode}] {idx:>2}/{len(items)} {flag} {item['id']:<4} "
            f"hit={int(result.hit_at_k)} corr={result.answer_correctness:.2f} "
            f"grnd={result.groundedness:.2f} beh={result.behavior}/{result.behavior_expected} "
            f"{result.latency_ms.get('total', 0):.0f}ms"
        )

    agg = aggregate(results)
    if judge is not None:
        judged = [r.llm_judge for r in results if r.llm_judge and "correctness" in r.llm_judge]
        if judged:
            agg["llm_judge"] = {
                "n_judged": len(judged),
                "correctness": round(statistics.fmean(float(j["correctness"]) for j in judged), 4),
                "groundedness": round(statistics.fmean(float(j["groundedness"]) for j in judged), 4),
                "relevance": round(statistics.fmean(float(j["relevance"]) for j in judged), 4),
                "hallucination_rate_pct": round(
                    100.0 * sum(1 for j in judged if j.get("hallucinated")) / len(judged), 2
                ),
            }

    provider = next((getattr(r, "provider", None) for r in results if getattr(r, "provider", None)), None)
    payload = {
        "mode": mode,
        "provider": provider or _provider_of(cfg),
        "judge": judge_kind,
        "dataset": str(DATASET.relative_to(ROOT)),
        "wall_time_s": round(time.time() - started, 2),
        "config": _config_snapshot(cfg),
        "aggregate": agg,
        "items": [r.to_dict() for r in results],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{mode}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[{mode}] wrote {out}")
    return payload


def _provider_of(cfg: Any) -> str:
    return str(getattr(cfg, "provider", "local"))


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    keys = [
        "provider", "mode", "chunk_size_tokens", "chunk_overlap_tokens", "section_aware",
        "prepend_header", "top_k", "candidate_k", "use_hybrid", "use_rerank",
        "use_query_rewrite", "use_subquery_decomposition", "filter_current_only",
        "per_doc_cap", "neighbour_expansion", "abstain_threshold", "enable_guardrails",
        "enable_clarification", "context_token_budget", "embedding_model", "chat_model",
    ]
    snapshot = {}
    for key in keys:
        if hasattr(cfg, key):
            snapshot[key] = getattr(cfg, key)
    return snapshot


_ROWS = [
    ("Retrieval", "Hit rate @k (%)", ("retrieval", "hit_rate_pct"), "up"),
    ("Retrieval", "Expected-doc recall", ("retrieval", "doc_recall"), "up"),
    ("Retrieval", "Section hit rate", ("retrieval", "section_hit_rate"), "up"),
    ("Retrieval", "MRR", ("retrieval", "mrr"), "up"),
    ("Retrieval", "Relevant-chunk precision", ("retrieval", "chunk_precision"), "up"),
    ("Retrieval", "Context fact recall", ("retrieval", "context_fact_recall"), "up"),
    ("Generation", "Answer correctness", ("generation", "answer_correctness"), "up"),
    ("Generation", "Groundedness", ("generation", "groundedness"), "up"),
    ("Generation", "Citation precision", ("generation", "citation_precision"), "up"),
    ("Generation", "Citation recall", ("generation", "citation_recall"), "up"),
    ("Generation", "Correct behaviour (%)", ("generation", "behavior_accuracy_pct"), "up"),
    ("Generation", "Hallucination rate (%)", ("generation", "hallucination_rate_pct"), "down"),
    ("Generation", "Stale-version leak (%)", ("generation", "stale_version_leak_pct"), "down"),
    ("System", "Mean latency (ms)", ("system", "latency_ms_mean"), "down"),
    ("System", "P95 latency (ms)", ("system", "latency_ms_p95"), "down"),
    ("System", "Mean prompt tokens", ("system", "prompt_tokens_mean"), "down"),
    ("System", "Total est. cost (USD)", ("system", "cost_usd_total"), "down"),
]


def _get(agg: dict[str, Any], path: tuple[str, str]) -> float:
    return float(agg.get(path[0], {}).get(path[1], 0.0) or 0.0)


def write_comparison(baseline: dict[str, Any], improved: dict[str, Any]) -> Path:
    lines = [
        "# Baseline vs Improved RAG — evaluation results",
        "",
        f"- Dataset: `{baseline['dataset']}` ({baseline['aggregate']['n_items']} items)",
        f"- Provider: `{baseline['provider']}` (baseline) / `{improved['provider']}` (improved)"
        f", judge: `{improved['judge']}`",
        "- Generated by `python -m eval.run_eval --mode both`",
        "",
        "| Layer | Metric | Baseline | Improved | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for layer, label, path, direction in _ROWS:
        base = _get(baseline["aggregate"], path)
        imp = _get(improved["aggregate"], path)
        delta = imp - base
        arrow = "" if abs(delta) < 1e-9 else (" ✅" if (delta > 0) == (direction == "up") else " ⚠️")
        lines.append(f"| {layer} | {label} | {base:.3g} | {imp:.3g} | {delta:+.3g}{arrow} |")

    lines += ["", "## Per-category behaviour accuracy (%)", "",
              "| Category | n | Baseline hit rate | Improved hit rate | Baseline correct behaviour | Improved correct behaviour | Baseline hallucination | Improved hallucination |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    categories = sorted(set(baseline["aggregate"]["by_category"]) | set(improved["aggregate"]["by_category"]))
    for cat in categories:
        b = baseline["aggregate"]["by_category"].get(cat, {})
        i = improved["aggregate"]["by_category"].get(cat, {})
        lines.append(
            f"| {cat} | {i.get('n', b.get('n', 0))} | {b.get('hit_rate_pct', 0):.3g} | {i.get('hit_rate_pct', 0):.3g} | "
            f"{b.get('behavior_accuracy_pct', 0):.3g} | {i.get('behavior_accuracy_pct', 0):.3g} | "
            f"{b.get('hallucination_rate_pct', 0):.3g} | {i.get('hallucination_rate_pct', 0):.3g} |"
        )

    regressions = [
        r for r in baseline["items"]
        if next((i for i in improved["items"] if i["item_id"] == r["item_id"]), {}).get("behavior_correct")
        is False and r["behavior_correct"] is True
    ]
    fixed = [
        r for r in improved["items"]
        if r["behavior_correct"]
        and not next((b for b in baseline["items"] if b["item_id"] == r["item_id"]), {}).get("behavior_correct", False)
    ]
    lines += ["", "## Items fixed by the improved pipeline", ""]
    lines += [f"- `{r['item_id']}` ({r['category']}): {r['question']}" for r in fixed] or ["- none"]
    lines += ["", "## Items regressed by the improved pipeline", ""]
    lines += [f"- `{r['item_id']}` ({r['category']}): {r['question']}" for r in regressions] or ["- none"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--judge", choices=["none", "llm"], default="none")
    parser.add_argument("--no-ingest", action="store_true", help="reuse the existing index")
    parser.add_argument("--filter-category", default=None)
    args = parser.parse_args()

    items = load_dataset(Path(args.dataset))
    if args.filter_category:
        items = [i for i in items if i.get("category") == args.filter_category]
    if not items:
        raise SystemExit("no dataset items selected")

    modes = ["baseline", "improved"] if args.mode == "both" else [args.mode]
    payloads = {m: run_mode(m, items, args.judge, ingest=not args.no_ingest) for m in modes}
    if len(modes) == 2:
        write_comparison(payloads["baseline"], payloads["improved"])


if __name__ == "__main__":
    main()
