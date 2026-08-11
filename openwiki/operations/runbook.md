---
type: Operations Runbook
title: Development and Office Operations Runbook
description: Safe setup, configuration precedence, staged office activation, artifacts, VLM service operations, and troubleshooting for equipment automation and supporting model services.
resource: "poc/workflow_3/README.md"
tags: [operations, runbook, windows, safety, deployment]
---

# Operations runbook

## Environments

Development is split across macOS/dev machines and an office Windows machine:

- Pure CV, replay alarms, proxy tests, and many extraction smokes run without RCS.
- RCS window control, real input, Office COM, MES adapters, and equipment calibration require Windows/office access.
- Model deployment targets local company GPU infrastructure and local/offline weights.

Use Python 3.10+ with `uv`:

```bash
uv sync --extra dev
```

Windows-only optional dependencies are under the `windows` extra in `pyproject.toml`. Never inspect or commit live `.env` files. Configuration values flow through the shell environment and settings loaders.

## Configuration precedence

For workflow 3, current precedence is:

```text
real shell environment
> gitignored workflow_3_config.py seeded into environment
> Workflow3Settings source defaults
```

Copy `poc/workflow_3/workflow_3_config.example.py` for local convenience. `workflow_3_config_loader.seed_env()` uses set-if-absent semantics. It cannot define settings that `config.py` does not read.

`ALIGN_IMAGES_DIR` and `ALIGN_CONSENSUS_CACHE_DIR` are package-import constants, so set them in the real environment before importing `poc.workflow_3`; the local scratch loader runs too late for them. The same precedence pattern applies to workflow 2's golden config.

## Safety gates

- `SAFE_MODE=1` disables interactive action globally.
- Real align correction requires `SAFE_MODE=0` **and** `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.
- Real measurement abort requires `SAFE_MODE=0` **and** `MEAS_FAIL_ABORT_DRY_RUN=0`.
- `ALIGN_FAIL_BLOCK_INPUT` is opt-in, applies only outside safe mode, and should be enabled only after elevation/foreground behavior is verified.
- Occupied RCS popups are detection-only; do not automate share/terminate choices.
- `ALIGN_FAIL_MIND_RERANK=0`, `ALIGN_FAIL_ECC_RERANK=0`, `ALIGN_FAIL_CONSENSUS=0`, and `ALIGN_FAIL_COND_BOX_CROP=0` provide focused CV rollback paths.

Default correction and abort settings are dry-run/notify-only even if the global safe mode default differs. Never arm a destructive path based only on synthetic tests.

## Safe local runs

```bash
# Replay one or more synthetic alarms through the main loop.
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py

# Unified align + measurement-fail supervisor in dry-run.
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  MEAS_FAIL_ALID=<alid> uv run python poc/workflow_3e/monitor.py

# Check-only office variant: connects/captures/closes without production correction or recording.
uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py
```

Replay CSVs use `EQP_ID,ALID,ALARM_NAME,UTC9,RECIPE_ID,OPERATION_DESC,LOT_TYPE_CD`; the replay source refreshes timestamps so rows pass the recent-window filter.

## Office activation sequence

1. Put office adapters in `poc/workflow_3/monitor/` and verify imports. Do not rely on the old workflow 1 fallback.
2. Confirm MES RCP images and hidden `cond.txt` sidecars resolve under the effective `ALIGN_IMAGES_DIR`. Current source default is `poc/workflow_3/align_images`; override if office MES still writes elsewhere.
3. Run an import sweep:
   ```bash
   uv run python -c "import poc.workflow_3.monitor.align_fail_monitor"
   ```
4. Run `SAFE_MODE=1` against a real alarm: expect no clicks; inspect journals, manifest, notifications, and captures.
5. Run record-only with `ALIGN_FAIL_CORRECTION=0` and verify connect/capture/record/close parity.
6. Enable correction but retain `ALIGN_FAIL_CORRECTION_DRY_RUN=1`; review selected points and OK-button overlays.
7. Verify SEM panel, mode/PM reading, foreground behavior, double-click recenter, wheel/dropdown magnification, and engineer-done counter on each relevant tool model.
8. Pilot one tool with the correction dry-run gate disabled. Keep focused kill switches available.
9. For workflow 3e, first run notify-only and confirm the captured Abort/Stop coordinate and alarm detail; only then disable `MEAS_FAIL_ABORT_DRY_RUN`.

`poc/workflow_3/README.md` contains useful detailed checklists but also stale migration text. Prefer `poc/workflow_3/__init__.py`, `config.py`, and the current [align data contract](../workflows/align-fail.md#data-contracts) when they disagree.

## Artifacts and observability

Primary outputs:

- `poc/workflow_3/logs/align_fail_alarms.txt` — detected alarms.
- `poc/workflow_3/logs/align_fail_cycles.csv` — one summary row per cycle.
- `poc/workflow_3/logs/workflow_runs/<run_id>.../` — step journals.
- `poc/workflow_3/logs/vlm_calls.log` and general rotating logs — detailed logging depending on settings.
- `<ALIGN_IMAGES_DIR>/.../captured_img_from_rcs/<tag>/` — captures, marked overlays, feasibility/zoom JSON, and `recording/` frames.
- `poc/workflow_3e/measurement_abort_cycles.csv` or configured workflow 3e manifest location — abort summaries.
- `deploy_vlms/runtime/logs/<service>.log` — model server logs.
- `logs/vlm_service/vlm_serve.log` — Flask proxy logs.

Use `WORKFLOW3_FILE_LOG_DETAIL=1` temporarily for successful/info VLM events; defaults emphasize warnings/errors. Keep screenshots JPEG locally and WebP for VLM payloads.

## VLM service operations

Run from repository root:

```bash
python deploy_vlms/scripts/start_all.py
python deploy_vlms/scripts/start_model.py ui-venus
python deploy_vlms/scripts/serve_vlm.py ui-venus       # foreground diagnosis
python deploy_vlms/scripts/check_vlm.py                # all configured services
python deploy_vlms/scripts/stop_model.py ui-venus
python deploy_vlms/scripts/stop_model.py all

curl http://127.0.0.1:8001/v1/models
curl http://<flask-host>/api/vlm_serve/health
curl http://<flask-host>/api/vlm_serve/ui-venus/v1/models
```

The current `check_vlm.py` accepts an optional host and checks configured services; ignore older docs suggesting URL/model arguments. GOT-OCR uses a distinct server implementation and has a route/process naming mismatch (`got-ocr` vs `got-ocr-2.0-hf`) worth checking during diagnosis.

## Troubleshooting

### Assets appear empty

Print the effective `ALIGN_IMAGES_DIR` early and inspect `<eqp>/<class>/<recipe>/align_img_from_rcp`. A common failure is MES writing the old workflow 1 path while code reads the workflow 3 default. Set the real environment before process import; local scratch config is too late.

### Consensus always falls back

Confirm the office success downloader exists, gather logs show successful staging, and cache paths are `<class>/<recipe>/events` without equipment ID. Check modality sample count and blur rejection. Do not move the cache into the MES tree.

### RCS window/click mismatch

Check elevation, DPI awareness, foreground logs, physical-vs-logical coordinates, and whether another engineer owns the tool. Do not bypass occupied-popup behavior. Validate with capture overlays before arming clicks.

### VLM proxy says healthy but models fail

Inspect nested per-service health, direct `/v1/models`, proxy logs, service logs, upstream host/base URL overrides, and configured served-model names. The outer aggregate status alone is not sufficient.

### Document extraction completes with poor output

Inspect `stage_log` and `summary_model_sources` for `offline` or fallback markers. Structural completion may be using stubs. Verify real model endpoints, crop-refine setting, OpenSearch/embedding wiring, and source-specific capture path. See [research pipelines](../workflows/research-pipelines.md#model-roles-and-fallbacks).
