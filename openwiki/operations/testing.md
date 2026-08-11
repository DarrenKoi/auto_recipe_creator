---
type: Testing Guide
title: Testing and Validation Strategy
description: Test layers, commands, change-to-check mapping, platform boundaries, and production validation expectations across align automation, VLM proxying, and document extraction.
resource: "pyproject.toml"
tags: [testing, pytest, smoke-tests, validation, safety]
---

# Testing and validation

The repository mixes pytest suites, directly executable smoke scripts, golden-set evaluations, and office-only calibration. A green local suite proves pure logic and integration contracts; it does not prove real RCS actuation, model accuracy, MES schema fidelity, or GPU service readiness.

## Fast baseline

```bash
uv run pytest poc/workflow_3/recording_filter
uv run pytest test/flask_api test/deploy_vlms
uv run python poc/workflow_3/align/matching/test_engine.py
uv run python poc/workflow_3/align/matching/test_engine_ensemble.py
uv run python poc/workflow_3/align/matching/test_mind_rerank.py
uv run python poc/workflow_3/align/test_correction.py
```

Tests under workflow 3 often run directly despite pytest-style names. Follow the file's existing invocation pattern. Some tests and modules write logs/debug artifacts; review git status afterward.

## Production workflow tests

### Matching and correction

- `align/matching/test_engine.py` — synthetic scoring, scale, decisions, and geometry.
- `align/matching/test_engine_ensemble.py` and `test_ensemble.py` — proposer/fusion behavior.
- `align/matching/test_mind_rerank.py` — OM/SEM routing, kill switches, invalid candidates, and existing-coordinate invariant.
- `align/test_correction.py` — visibility gate, outcomes, and error paths.
- `align/test_consensus_*.py` — gathering, crops, template build, and RCP fallback.
- `align/diagnostics/test_match_on_captured_frames.py` — requires office fixtures.

### Monitor, RCS, and extension jobs

```bash
uv run python poc/workflow_3/rcs/test_tool_name_match.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
uv run python poc/workflow_3e/test_config.py
uv run python poc/workflow_3e/test_detector.py
uv run python poc/workflow_3e/test_abort_button.py
uv run python poc/workflow_3e/test_abort_cycle.py
uv run python poc/workflow_3e/test_dispatch.py
```

These exercise contracts and dry-run behavior without establishing that a Windows window, button, or tool behaves as expected. The [operations activation sequence](runbook.md#office-activation-sequence) is the required next layer.

## CV golden evaluation

Workflow 2 golden drivers measure data-dependent changes that synthetic tests cannot:

```bash
uv run python poc/workflow_2/golden_localization_eval_cond.py
uv run python poc/workflow_2/golden_consensus_eval_cond.py
uv run python poc/workflow_2/golden_combined_eval_cond.py
uv run python poc/workflow_2/golden_registration_eval_cond.py
```

Configure `golden_eval_config.py` from its example and preserve shell-env precedence. Evaluate digest plus mismatch/hook-error counts, not only headline accuracy. Any production rerank/template change should maintain bench/production bit parity and include a rollback switch when it can alter candidate selection.

## VLM proxy and deployment tests

`test/flask_api` uses Flask's test client and monkeypatched requests to cover route registration, URL mapping, API-key injection, proxy errors, logging, health aggregation, and buffered UI-TARS responses. `test/deploy_vlms` checks memory/KV sizing logic.

These tests do not launch models. Production validation requires direct `/v1/models`, proxied health, one representative inference per service, and GPU/process/log inspection as described in the [runbook](runbook.md#vlm-service-operations).

## Document extraction tests

Representative offline smoke commands:

```bash
uv run python -m side_projects.document_extraction.extraction.test_extraction_smoke
uv run python -m side_projects.document_extraction.extraction.test_opensearch_smoke
uv run python -m side_projects.document_extraction.benchmark.test_retrieval_benchmark_smoke
uv run python -m side_projects.document_extraction.marp.test_render_smoke
uv run python -m side_projects.document_extraction.marp.test_verify_smoke
uv run python -m side_projects.document_extraction.util.test_viewer_capture_smoke
```

They cover deterministic transformations, fallback chains, provenance, fake OpenSearch transport, RRF math, benchmark metrics, Marp planning, SSIM, and viewer frame-difference logic. They do not validate live VLM/OCR quality, Windows COM, DRM key sequences, real OpenSearch mappings, embedding/reranker endpoints, or Chromium fidelity.

## Change-to-check matrix

| Change area | Minimum checks | Additional validation |
|---|---|---|
| Matcher policy/proposers | engine + ensemble + correction tests | workflow 2 localization/combined golden evaluation |
| MIND/ECC routing | `test_mind_rerank.py`, engine ensemble | registration golden arms; compare parity constants in bench and production |
| Consensus/cache | consensus unit tests + success gather | verify recipe-only cache path on office and RCP fallback |
| Alarm dispatch/cycle | monitor tests + replay SAFE_MODE run | office real-alarm dry-run, manifest and teardown review |
| RCS/SEM input | geometry/tool-name tests | elevated Windows capture-overlay calibration; pilot one tool |
| Workflow 3e abort | all `workflow_3e/test_*.py` | notify-only office run and physical postcondition review |
| VLM proxy route/config | `test/flask_api` | direct and proxied model smoke, timeout/log review |
| VLM deployment sizing | `test/deploy_vlms` | actual GPU placement, memory, process, and model-name checks |
| Extraction stages | extraction smokes | office source files, real models, ground truth |
| Retrieval/indexing | OpenSearch + benchmark smokes | real index/embedding/reranker and chart-heavy relevance set |
| Marp reconstruction | render/refine/verify smokes | actual Chromium render and SSIM distribution calibration |

## Review invariants

Before merging a change, confirm:

- No new dependency points from workflow 3 into workflow 1/2.
- No actuation bypasses global and domain-specific gates.
- Rerankers still select an existing candidate and preserve all-rejected fallback.
- Consensus paths remain recipe-keyed while MES assets remain equipment-keyed.
- New alarm jobs remain serialized with the single RCS cursor.
- Logs/manifests preserve enough context to diagnose failure without exposing credentials.
- Platform-specific limitations and test evidence are stated in the change description.

Use the [source map](../source-map.md) to locate authoritative implementation and docs for the area under test.
