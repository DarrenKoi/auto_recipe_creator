---
type: Research Workflow Guide
title: CV Evaluation and Document Knowledge Pipelines
description: Offline evaluation workflows that validate align-key changes before production and the screenshot/digital-first document extraction, chart-RAG, retrieval, and reconstruction subsystem.
resource: "poc/workflow_2"
tags: [evaluation, golden-set, document-extraction, rag, research]
---

# Research pipelines

## CV evaluation and porting

`poc/workflow_2` is active despite its historical name. It is the evaluation/A-B/tuning environment for [production align correction](align-fail.md), and imports the workflow 3 engine rather than the reverse. A change should be proven here, then narrowly ported into `poc/workflow_3/align` with parity tests and an operational kill switch where appropriate.

Primary drivers:

- `golden_localization_eval_cond.py` — RCP localization.
- `golden_consensus_eval_cond.py` — consensus A/B.
- `golden_combined_eval_cond.py` — routed production-like evaluation: consensus when eligible, RCP fallback otherwise.
- `golden_registration_eval_cond.py` — leave-one-out registration/verifier experiments and coverage decomposition.
- `golden_reregister_report_cond.py` — template-quality/remediation classification.

Edit-often bench configuration lives in gitignored `golden_eval_config.py`, copied from `golden_eval_config.example.py`. `golden_eval_config_loader.seed_env()` applies constants only when the real shell environment has not already supplied a value. This import ordering matters for settings read at module import time.

### Reading the metrics

Registration evaluation separates failure mechanisms:

- **Proposer miss:** ground truth absent from top-K. Improve proposal/template registration; reranking cannot recover it.
- **Rank error:** ground truth in top-K but not rank 1. A verifier/reranker may help.
- **Rank-1 OK:** proposal and selection already succeed.

The recent registration worklist aggregates proposer misses by recipe and modality and writes debug/report JSON. It is a prioritization artifact, not an automatic re-registration action. Review sample count, mismatch count, and hook errors before trusting rankings; one miss is enough for inclusion and no confidence interval is enforced.

### Recent routing decision

Git history documents an empirical sequence rather than a one-shot implementation. Registration arms first established candidate-verifier behavior, then production pseudo-arms compared baseline, MIND, ECC, and combinations. The final production route is:

- OM: baseline/NCC order fused with MIND rank.
- SEM: complete switch to ECC order, not fusion.

The production implementation and the bench implementation duplicate constants/algorithms. This is an intentional but fragile parity contract; always run both workflow 2 and workflow 3 tests when changing it. See the [testing matrix](../operations/testing.md#change-to-check-matrix).

## Document extraction and chart RAG

`side_projects/document_extraction` is a secondary product domain that turns Office/PDF content into visual evidence, structured text, RAG chunks, retrieval results, and optionally reconstructed Marp slides. It depends on [model and search integrations](../architecture/integrations.md#document-extraction-and-search-systems) but is operationally separate from equipment automation.

### Two ingestion paths

**Screenshot-first** (`extract.py`, `extraction/extract_screenshot.py`):

```text
Stage 0 capture/export
-> preprocess
-> OCR
-> layout detection
-> optional crop refinement
-> evidence merge
-> synthesis
-> review packet
-> RAG chunks
```

PPT capture uses slideshow screenshots; Word/Excel usually export to PDF; PDFs use PyMuPDF. PDF/Word can fall back to viewer automation for DRM-constrained documents. Excel has no generic DRM viewer fallback.

**Digital-first** (`harvest/harvest_pdf.py`, `extraction/build_rag_chunks.py`): preserve pre-DRM text coordinates/fonts, tables, figures, renders, TOC, links, and a resumable manifest, then reconstruct headings and query-oriented chunks. This path preserves richer structure when a digital PDF is still accessible.

### Model roles and fallbacks

- PaddleOCR-VL — OCR/document parsing.
- UI-Venus — page layout/regions.
- MAI-UI — optional dense crop refinement.
- Kimi-family service — image-aware synthesis.
- GLM-family service — evidence-only text fallback.

A failed OCR/layout call can switch the stage runner into offline stubs while allowing the pipeline to complete structurally. Synthesis separately falls back Kimi -> GLM -> offline. Therefore successful process exit is not evidence of extraction quality: inspect `stage_log` and `summary_model_sources`, and reject offline outputs in real evaluations.

### Retrieval architecture

The chart-RAG path stores text/summary plus page/crop image provenance. `opensearch_index.py` builds BM25 and 1024-dimensional kNN mappings; `embeddings.py` calls a bge-m3-compatible service or deterministic offline stub; `hybrid_search.py` fuses BM25 and dense ranks client-side with RRF and creates a deferred visual-reader payload.

The reranker hook is currently passthrough, and tests use fake transport. Real OpenSearch mappings, embedding endpoints, reranking, and chart-heavy golden relevance remain office integration work.

### Reconstruction and evaluation

`marp/` converts evidence into a deck, renders through Marp/Chromium, computes SSIM against captures, and can degrade low-fidelity slides to raster. The extraction benchmark measures structured outputs; the retrieval benchmark supports BM25/dense/hybrid arms and Recall@k/MRR by query tier.

Recent commits show the domain's progression from screenshot extraction to DRM fallback and reconstruction, then to provenance-heavy chart RAG and retrieval benchmarking. Code scaffolding is broad, but current `docs/status.md` correctly states that most remaining risk is real office data/model/render accuracy rather than pure transformation logic.

## Practical entrypoints

```bash
# Model-free structural extraction smoke.
DOC_EXTRACT_OFFLINE=1 uv run python \
  -m side_projects.document_extraction.extraction.extract_screenshot

# Hybrid retrieval and benchmark entrypoints.
uv run python -m side_projects.document_extraction.extraction.hybrid_search
uv run python -m side_projects.document_extraction.benchmark.run_retrieval_benchmark

# Evidence-to-Marp pipeline.
uv run python -m side_projects.document_extraction.marp.build_marp
```

Many modules use edit-in-source path constants rather than command-line arguments. Confirm the current source and [operations guidance](../operations/runbook.md) before treating examples as immutable deployment commands.
