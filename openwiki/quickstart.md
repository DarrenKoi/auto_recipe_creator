---
type: Quickstart
title: Auto Recipe Creator Engineering Wiki
description: Entry point for the CD-SEM/VeritySEM recipe automation repository, covering production align-fail handling, CV evaluation, VLM services, document extraction, operations, and tests.
resource: "repository://auto_recipe_creator"
tags: [cd-sem, automation, computer-vision, vlm, python]
---

# Auto Recipe Creator

This Python monorepo automates CD-SEM/VeritySEM recipe operations. Its primary product is the `poc/workflow_3` real-time loop: detect Align Fail alarm `ALID=9006`, connect to the affected tool through RCS, locate the align key with classical computer vision, optionally reposition/confirm it, preserve evidence, notify an engineer when needed, and tear down safely. `poc/workflow_2` is the active offline evaluation bench; `poc/workflow_3e` adds new alarm jobs without branching the core loop.

The repository also hosts the Flask VLM proxy/deployment stack and a substantial document-extraction, reconstruction, and chart-RAG side project. Start with the [architecture overview](architecture/overview.md), then use the [source map](source-map.md) to reach implementation files.

## Current system at a glance

- **Production automation:** [`poc/workflow_3`](../poc/workflow_3/README.md) owns the live alarm cycle, RCS GUI control, matching, recording, notification, and audit artifacts.
- **Validated extension:** [`poc/workflow_3e`](../poc/workflow_3e/README.md) runs a unified supervisor for align alarms plus a double-gated measurement-fail abort job.
- **CV research-to-production path:** `poc/workflow_2` evaluates localization, consensus, registration, and rerank changes before selected changes are ported into workflow 3. See [research pipelines](workflows/research-pipelines.md).
- **Model infrastructure:** `web_main.py`, `flask_api/vlm_serve`, and `deploy_vlms` expose local VLM/OCR services used by automation and extraction. See [integrations](architecture/integrations.md).
- **Secondary product domain:** `side_projects/document_extraction` captures or harvests documents, extracts structured evidence, produces RAG chunks, performs hybrid retrieval, and reconstructs Marp decks.

## Setup

Python 3.10+ and `uv` are the normal development path (`pyproject.toml`). Do not read or commit live `.env` files; use documented environment variable names and sample configuration only.

```bash
uv sync --extra dev

# Safe synthetic/replay loop; requires a replay CSV fixture.
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py

# Core synthetic tests that do not require RCS.
uv run python poc/workflow_3/align/matching/test_engine.py
uv run pytest poc/workflow_3/recording_filter
uv run pytest test/flask_api test/deploy_vlms
```

Live RCS automation is Windows/office-only. Keep correction dry-run enabled until capture overlays and coordinates have been reviewed; real correction requires both `SAFE_MODE=0` and `ALIGN_FAIL_CORRECTION_DRY_RUN=0`. Follow the staged [operations runbook](operations/runbook.md), not a direct jump to actuation.

## Reading paths

1. [Architecture overview](architecture/overview.md) — package boundaries, dependency rules, runtime/data flow, and repository evolution.
2. [Align-fail workflow](workflows/align-fail.md) — alarm semantics, matching, consensus, OM/SEM routing, safety, and the measurement-abort extension.
3. [Research pipelines](workflows/research-pipelines.md) — CV golden benches and document extraction/chart RAG.
4. [Integrations](architecture/integrations.md) — MES/RCS adapters, VLM proxy, model deployment, and external systems.
5. [Operations runbook](operations/runbook.md) — safe execution, configuration, artifacts, service checks, and troubleshooting.
6. [Testing guidance](operations/testing.md) — change-oriented test commands and limits of local validation.
7. [Source map](source-map.md) — authoritative docs, entrypoints, ownership boundaries, legacy code, and generated artifacts.

## Engineering invariants

- `workflow_3` is the production core and must not import `workflow_1` or `workflow_2`; research/legacy code may depend on production, never the reverse.
- OpenCV is coordinate authority. VLMs locate UI regions, OCR text, or explain ambiguity; they do not override weak CV evidence or invent an align coordinate.
- GUI actuation is fail-safe and serialized. The system has one RCS application/cursor; `workflow_3e` therefore dispatches both job classes from one process.
- The registered/MES asset tree is equipment-keyed, while the production consensus cache is recipe-keyed and deliberately equipment-independent. See [align-fail data contracts](workflows/align-fail.md#data-contracts).
- Configuration is environment/settings based; scripts intentionally avoid CLI argument parsers. Shell environment overrides local scratch config, which overrides source defaults.
- Current source beats stale prose. In particular, `poc/workflow_3/__init__.py` now defaults `ALIGN_IMAGES_DIR` to `poc/workflow_3/align_images`, despite older migration text in `poc/workflow_3/README.md`.

## Backlog

- **Legacy CCTV/DVR experiments** — `poc/workflow_1`; mapped in the source map but deferred because the package is frozen except for the retained CCTV path.
- **Older experimental automation** — `poc/work2` and portions of `test/vlm_input_control`; deferred because production ownership has moved to `poc/workflow_3`.
- **Video frame parser** — `test/video_frame_parser`; CLIP/MongoDB/FAISS subsystem is testable but independent of the active align loop, so only its entrypoints are mapped initially.
- **Historical journals and reports** — `docs/journals`, `docs/project_progress`, and `poc/workflow_3/docs`; useful decision evidence, but not individually summarized in this first-pass wiki.
