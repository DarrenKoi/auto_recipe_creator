---
type: Source Map
title: Repository Source Map
description: Engineer-oriented navigation map of entrypoints, ownership boundaries, authoritative documentation, tests, legacy areas, and generated artifacts in the auto recipe creator repository.
resource: "repository://auto_recipe_creator"
tags: [source-map, navigation, ownership, repository]
---

# Source map

Use this page after the [quickstart](quickstart.md) to find the smallest authoritative source set for a change. Existing README/docs are primary evidence, but current code wins when migration notes have drifted.

## Root and application infrastructure

| Path | Role | Start here when |
|---|---|---|
| `pyproject.toml` | Python 3.10+, dependencies, package targets, pytest paths | changing setup/dependencies |
| `CLAUDE.md` | broad project map, current workstreams, conventions, commands | orienting across the monorepo |
| `CONTEXT.md` | Korean domain glossary | interpreting Align Fail terms; note some workflow-number prose is historical |
| `web_main.py`, `index.py` | Flask app factory and WSGI alias | changing API/dashboard hosting |
| `flask_api/` | `/api` registration and VLM proxy | changing proxy routes/health |
| `gpu_dashboard/` | placeholder dashboard JSON endpoints | extending GPU UI/health |

The Flask stack [integrates model services](architecture/integrations.md#flask-vlm-proxy) but does not launch the equipment automation loop.

## Production automation

| Path | Ownership |
|---|---|
| `poc/workflow_3/README.md` | detailed loop/office checklist; verify path statements against source |
| `poc/workflow_3/__init__.py` | package constants, DPI awareness, asset/cache roots |
| `poc/workflow_3/config.py` | authoritative settings schema and environment readers |
| `poc/workflow_3/workflow_3_config_loader.py` | gitignored scratch-config bridge |
| `poc/workflow_3/monitor/align_fail_monitor.py` | main poll/edge-trigger entrypoint |
| `poc/workflow_3/monitor/align_fail_monitor_only_check.py` | check-only connect/capture/feasibility flow |
| `poc/workflow_3/monitor/cycle.py` | per-alarm steps, domain executors, teardown coordination |
| `poc/workflow_3/monitor/integration_loader.py` | office adapter discovery/contracts |
| `poc/workflow_3/monitor/recording.py` | change-sensitive screenshot recording |
| `poc/workflow_3/runner/` | steps, conditions, results, workflow journals |
| `poc/workflow_3/rcs/` | RCS launch/login/tool/window/capture/close automation |
| `poc/workflow_3/sem_monitor/` | panel location and equipment controller |
| `poc/workflow_3/vlm/` | workflow service registry, clients, prompts |
| `poc/workflow_3/recording_filter/` | offline recording-to-interaction-timeline processing |

Read the [architecture overview](architecture/overview.md) before changing package dependencies and the [align workflow](workflows/align-fail.md) before changing business behavior.

## Align domain

| Path | Ownership |
|---|---|
| `align/assets.py` | central MES-tree resolver/loader |
| `align/templates.py`, `cond_template.py`, `cond_file.py` | RCP templates and hidden sidecar interpretation |
| `align/consensus_*` | success-image staging, crops, template build, fallback routing |
| `align/matching/engine.py` | candidate/result/policy types and coordinate authority |
| `align/matching/ensemble.py` | proposer ensemble and rank fusion support |
| `align/matching/mind_rerank.py` | OM MIND and SEM ECC production routing |
| `align/correction.py` | visibility gate, correction outcomes, primary/fallback orchestration |
| `align/live_search.py` | bounded pan/zoom fallback behind controller protocol |
| `align/diagnostics/` | office/dev probes and capture analysis |

Tests are colocated. For any algorithmic change, pair this area with [workflow 2 evaluation](workflows/research-pipelines.md#cv-evaluation-and-porting).

## Alarm-job extensions

`poc/workflow_3e/README.md` is the domain overview. `monitor.py` is the unified supervisor; `dispatch.py` owns edge triggering/manifests; `abort_cycle.py` reuses workflow 3 executors; `abort_button.py` owns VLM location; `config.py` adds `MEAS_FAIL_*` settings. This package [extends the production architecture](architecture/overview.md#extension-architecture) one-way.

## CV bench and legacy

| Path | Status |
|---|---|
| `poc/workflow_2/` | active offline golden evaluation, A/B, tuning, and reports |
| `poc/workflow_2/docs/study/runbooks/` | CV procedures/history |
| `poc/workflow_2/docs/study/adr/` | historical decisions, some pre-migration paths |
| `poc/workflow_1/` | frozen early GUI/CCTV work; retained CCTV path and historical fixtures |
| `poc/work2/` | older experimental area; not the current production package |

Do not infer status from numbering. Workflow 2 is active research; workflow 3 is production; workflow 1 is legacy.

## Model services and deployment

| Path | Role |
|---|---|
| `flask_api/vlm_serve/config.py` | proxy route/service registry |
| `flask_api/vlm_serve/service_template.py` | buffered forwarding, headers, timeouts, error handling |
| `deploy_vlms/config/` | common and per-model deployment settings |
| `deploy_vlms/scripts/` | start/stop/check/foreground service operations |
| `docs/setup_vlms/` | detailed deployment and capacity notes; check scripts for drift |
| `test/flask_api/`, `test/deploy_vlms/` | proxy contracts and sizing tests |

See [integrations](architecture/integrations.md) and [service operations](operations/runbook.md#vlm-service-operations).

## Document extraction

| Path | Role |
|---|---|
| `side_projects/document_extraction/README.md` | high-level side-project overview |
| `docs/status.md` | most current implementation/validation status |
| `docs/pipeline_overview.md` | screenshot-first pipeline |
| `docs/rag_chart_heavy_architecture.md` | chart-RAG design and phases |
| `extract.py`, `*_handler.py` | Stage 0 source dispatch/capture/export |
| `harvest/` | digital-first pre-DRM PDF preservation |
| `extraction/` | OCR/layout/merge/synthesis/chunks/search/OpenSearch |
| `benchmark/` | extraction and retrieval golden metrics/drivers |
| `marp/` | reconstruction, rendering, refinement, SSIM/downgrade |
| `util/viewer_capture.py` | DRM viewer fallback |

The [research-pipeline guide](workflows/research-pipelines.md#document-extraction-and-chart-rag) explains how these pieces relate and what remains unvalidated.

## Tests and secondary experiments

- `test/video_frame_parser/` — CLIP frame extraction/analysis with MongoDB and FAISS; run with pytest under its tests directory.
- `test/vlm_input_control/` — screen capture, VLM analysis, and input-control integration experiments.
- `test/workflow_extractor/`, `test/work2/` — historical/experimental support.
- `docs/journals/`, `docs/project_progress/`, `poc/workflow_3/docs/` — rationale, specs, runbooks, and reports. Use recent, relevant files selectively rather than treating every journal as current API documentation.

See [testing guidance](operations/testing.md) for supported commands and platform limits.

## Generated and local-only paths

Do not treat these as source or commit their contents unless explicitly intended:

- `poc/workflow_3/align_images/`, consensus cache, captures, debug images, logs, and workflow journals.
- `poc/workflow_2/debug_images/`, bench outputs, and local golden config.
- `deploy_vlms/runtime/` process/log files.
- Office adapter modules and local scratch config files documented as gitignored.
- `_workspace/`, caches, virtual environments, and test outputs.

When changing a path contract, update the [operations runbook](operations/runbook.md), corresponding tests, and the authoritative source comments together.
