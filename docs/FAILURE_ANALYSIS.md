# Failure Analysis

The baseline is intentionally a fixed-window, vector-only retriever with no metadata
filters, rewriting, reranking, or sufficiency gate. Improved mode adds the corresponding
fixes described in `DESIGN.md` section 11: section-aware breadcrumbed chunks, hybrid search,
version resolution, decomposition and reranking, strict grounding, clarification, and
conversation condensation.
