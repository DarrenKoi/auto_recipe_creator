---
type: Integration Guide
title: External Systems and Model Services
description: Integration topology for MES alarms, office-only RCS adapters, Flask VLM proxy routes, direct LLM gateways, model deployment, and document-search infrastructure.
resource: "flask_api/vlm_serve"
tags: [integrations, mes, rcs, vlm, flask, opensearch]
---

# Integrations

## MES and office-only adapters

[The align-fail workflow](../workflows/align-fail.md) expects tabular alarm rows with equipment, alarm, time, and recipe context. `monitor/alarm_source.py` supports the office source and replay CSVs. Office modules are intentionally local/gitignored and discovered by `monitor/integration_loader.py`; missing modules degrade the corresponding integration with a warning instead of breaking imports.

Canonical adapter location is `poc/workflow_3/monitor/`. Important contracts include:

- `office_align_fail_alarm.py` — alarm query/filter provider.
- `office_rich_notify.py` — engineer/cube notification adapter.
- `office_success_downloader.py` — recent successful S-image gatherer for consensus.
- `office_rcp_msr_downloader.py` — optional synchronous RCP downloader for sites where MES does not populate `ALIGN_IMAGES_DIR` directly. Runtime should request RCP only; MSR is offline-bench data.
- `poc/workflow_3e/office_meas_many_fails.py` — optional dedicated measurement-fail provider exposing `get_measurement_fail_alarms()`.

Do not put data download side effects inside the notification adapter: check-only flows may not notify and would then miss required assets. Use the staged [office activation runbook](../operations/runbook.md#office-activation-sequence).

## RCS and Windows desktop control

`poc/workflow_3/rcs` and `sem_monitor` integrate with `RcsMainHD.exe`, Remote Monitoring windows, Win32/pywinauto window control, MSS screenshots, and pynput input. Workflow 3 enables per-monitor DPI awareness during package import so window rectangles, captures, and click coordinates share physical pixels.

Real operation depends on Windows foreground/UIPI behavior. Forced foreground and `BlockInput` may silently fail when the monitor is not elevated. Both monitor entrypoints report elevation status. `SAFE_MODE`, action settings, and domain dry-run gates remain the final protection; see [operations](../operations/runbook.md#safety-gates).

## Flask VLM proxy

`web_main.py` registers `flask_api` at `/api` and the placeholder GPU dashboard at `/gpu-dashboard`. `flask_api/vlm_serve/config.py` defines enabled local routes:

| Route slug | Upstream port | Served model |
|---|---:|---|
| `ui-venus` | 8001 | `ui-venus-1.5-8b` |
| `mai-ui` | 8002 | `mai-ui-8b` |
| `ui-tars` | 8003 | `ui-tars-1.5-7b` |
| `paddleocr-vl-1.5` | 8004 | `paddleocr-vl-1.5` |
| `got-ocr` | 8005 | `got-ocr-2.0-hf` |

Useful endpoints:

```text
GET  /api/health
GET  /api/vlm_serve/health
ANY  /api/vlm_serve/<route_slug>/v1/*
```

`service_template.py` forwards query/body/headers and can inject an upstream API key from `VLM_SERVE_UPSTREAM_API_KEY`; never document or log the value. Per-service base URL overrides use `VLM_SERVE_<SERVICE>_BASE_URL`, while `VLM_SERVE_UPSTREAM_HOST` changes the shared host.

The proxy is buffered, not streaming: it calls `requests` with `stream=False`. UI-TARS requests may be rewritten to `stream=true` for upstream compatibility, but the Flask client still receives the response only after buffering. Long responses therefore remain subject to the read timeout.

Aggregate health requires careful interpretation: inspect each service's nested health state rather than treating the outer API status as proof that models are reachable.

## Workflow VLM clients

`poc/workflow_3/vlm/flask_vlm.py` is the client-side service hub. It supports:

- **Proxy routes** for local UI/OCR models.
- **Direct company gateway routes** for larger multimodal/text models used by workflows such as extraction synthesis.

Prompts under `poc/workflow_3/vlm/prompts` have narrow jobs: coarse UI boxes, refined points, OCR extraction, or measurement-counter grounding. The [architecture](overview.md#perception-and-actuation-split) requires these models to support, not replace, CV coordinate decisions.

## Model deployment

`deploy_vlms/config/common.env` and `deploy_vlms/config/models/*.env` define local model paths, aliases, ports, GPU placement, and sizing. Scripts validate local/offline model roots and launch vLLM-compatible servers; GOT-OCR is a separate Flask/Transformers service rather than a standard chat-completions server.

Operational commands and diagnostics live in the [runbook](../operations/runbook.md#vlm-service-operations). The detailed source docs are `docs/setup_vlms/`, but verify commands against current scripts: some older prose describes a superseded `check_vlm.py` argument shape.

## Document extraction and search systems

[The document-extraction pipeline](../workflows/research-pipelines.md#document-extraction-and-chart-rag) reuses workflow VLM clients and also integrates with:

- Microsoft Office COM and PyMuPDF for capture/export.
- Viewer keyboard automation for DRM fallback.
- Kimi/GLM-style gateway services for synthesis.
- bge-m3-compatible embedding HTTP services.
- OpenSearch BM25 and kNN indexes, fused client-side with reciprocal-rank fusion.
- Marp/Chromium for reconstructed slide rendering.

Current tests fake OpenSearch transport and model responses. Office COM, live VLM accuracy, DRM viewer behavior, real embedding/index compatibility, and Marp visual fidelity remain integration validations, not guarantees from unit tests.
