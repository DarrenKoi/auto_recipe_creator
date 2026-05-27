# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen understanding and classical CV for coordinate decisions, driving GUI automation of the RCS metrology client to replace manual recipe creation.

## Active Workstreams

The two `poc/workflow_*` packages are the current focus. They split a single end-to-end goal into two stages.

- **`poc/workflow_1/`** — RCS login → Tool selection → **Align Fail alarm detection** (ALID=9006) → Tool screen capture. Owns the RCS client automation and the alarm polling loop.
- **`poc/workflow_2/`** — Given an Align-Fail tool, drive the SEM Monitor to find the on-wafer location matching the recipe-registered align key. CV decides coordinates; VLM only identifies regions / assesses feasibility.

The handoff between them is a **filesystem contract**, not a function call:

```
align_images/<eqp_id>/<class>/<recipe>/
├─ align_img_from_rcp/      IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe-registered align key (office MES)
├─ align_img_from_msr/      S*/E*                             # measurement trajectory (E = fail) (office MES)
└─ captured_img_from_rcs/   <tag>_rcs.jpg                     # fail-time RCS capture (workflow_1 writes)
```

- Root constant: `ALIGN_IMAGES_DIR` in `poc/workflow_1/__init__.py`.
- `workflow_1`'s `rcs_screenshot.py` writes `captured_img_from_rcs/`.
- `workflow_2`'s `align_fail_assets.resolve_assets_auto()` is the single reader, exposing `recipe_om` / `recipe_sem` / `current_sem` (override via `ALIGN_EQP_ID` / `ALIGN_CLASS_NAME` / `ALIGN_RECIPE_NAME`).

**Authoritative workflow_2 design doc:** `poc/workflow_2/docs/workflow_2_procedure.md` (single source of truth for steps, file mapping, and implementation status). ADRs under `poc/workflow_2/docs/adr/`.

## Repository Structure

```
poc/workflow_1/          # Stage 1: RCS login, tool select, align-fail alarm detection + capture
poc/workflow_1/prompts/  # VLM prompt builders (ui-venus / mai-ui login locators, OCR assist)
poc/workflow_1/util/     # env, image, json, mouse, time, window helpers
poc/workflow_2/          # Stage 2: align-key search (CV match engine + live SEM Monitor search)
poc/workflow_2/docs/     # workflow_2_procedure.md (authoritative), ADRs, algorithm explainers
flask_api/vlm_serve/     # Flask VLM proxy: service registry, health discovery, per-model blueprints
deploy_vlms/             # VLM deployment configs, scripts, operational docs
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU cluster)
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
docs/                    # Architecture research notes and setup guides
```

## Setup & Dependencies

`uv` with `pyproject.toml` (Python >= 3.10). Use uv-managed workflows by default.

```bash
uv sync --extra dev                      # Core project + dev tools
uv pip install -r requirements.txt       # All-in-one
uv pip install -r test/video_frame_parser/requirements.txt  # torch, opencv, pymongo, faiss
```

## Running Modules

All scripts run with just `uv run python <script>.py` (no CLI args — see Code Conventions).

```bash
# workflow_1 — RCS automation (Windows; Flask proxy, no per-PC .env needed)
uv run python poc/workflow_1/open_rcs.py                  # Start RcsMainHD.exe only
uv run python poc/workflow_1/workflow_login.py            # RCS login workflow
uv run python poc/workflow_1/login_rcs_ui_venus_mai.py    # 2-stage (ui-venus coarse → mai-ui refined) login locator
uv run python poc/workflow_1/view_list_tab_rcs.py         # Locate + click the List tab
uv run python poc/workflow_1/workflow_select_tool.py      # Find a tool in List tab and double-click it
uv run python poc/workflow_1/connect_tool.py              # Manual: connect to an arbitrary tool (reuses select_tool)
uv run python poc/workflow_1/workflow_close_tool.py       # Close the opened tool window (Remote Monitoring System) by tool id in title
uv run python poc/workflow_1/align_fail_alarm.py          # Poll alarms; on ALID=9006 notify + capture RCS frame
uv run python poc/workflow_1/monitor_align_fail.py        # Align-fail + open Tool DVR (CCTV) + capture CH4 frames
uv run python poc/workflow_1/rcs_screenshot.py            # Capture tool window into captured_img_from_rcs, then close

# workflow_2 — align-key search
uv run python poc/workflow_2/compare_align_images.py      # Step 3: static CV compare (falls back to synthetic self-test)
uv run python poc/workflow_2/vlm_align_key_box.py         # Step 1·2: VLM feasibility probe (office: Flask VLM required)
uv run python poc/workflow_2/live_align_search.py         # Step 4-7: two-phase live search demo (Mac: magnification mock)

# Video frame parser
uv run python -m test.video_frame_parser.example_usage
```

`workflow_1/workflow_runner.py` is a library, not an entry point: `WorkflowRunner` runs a `list[WorkflowStep]` sequentially and `ConditionChecker` evaluates step pre/post conditions; runs are journaled under `logs/workflow_runs/`.

## Testing

```bash
# workflow_2 match engine — synthetic smoke test (expect 10/10)
uv run python poc/workflow_2/test_align_key_match.py
uv run python poc/workflow_2/test_match_on_captured_frames.py   # against captured frames

# Video frame parser unit tests
uv run pytest test/video_frame_parser/tests/

# vlm_input_control integration (safe mode by default; toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules.
- **No `__future__` imports by default**: do not add `from __future__ import annotations` (or any `__future__` import) unless explicitly asked.
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/workflow_1/logger.py` uses Python `logging` with `RotatingFileHandler` for the audit trail (`poc/workflow_1/logs/vlm_calls.log` for VLM calls, `work2.log` for general events).
- **Absolute imports** within `poc/`: use `from poc.workflow_1.xxx import ...` / `from poc.workflow_2.xxx import ...`.
- **`__all__` in `__init__.py` is optional**: only add it when it provides clear value for a curated package API.
- **Image format convention**: save debug screenshots locally as **JPEG**; convert to **WebP** (quality=90) when sending to VLM APIs to cut payload size without hurting accuracy.
- **Safe mode**: interactive modules respect `SAFE_MODE` (blocks real mouse/keyboard output). `action_enabled`/`typing_enabled` default to the inverse of `SAFE_MODE` in `WorkflowSettings`.
- **No CLI arguments**: do not use `argparse` or flags. Configuration comes from `WorkflowSettings` (`workflow_config.py`), `flask_vlm.py` constants, or environment variables. Scripts must run with just `uv run python <script>.py`.

## Development Workflow

Development is **mixed macOS + Windows**:

- On **macOS**, Claude Code cannot see or drive the actual RCS application. Windows-only paths (RCS, pywinauto, pynput mouse/keyboard) are edited on Mac, pushed via git, pulled at the office, and run there; debugging relies on the user reporting console output and debug screenshots in `poc/workflow_1/debug_images/` (per-model subdirs).
- On **Windows** (office machine), Claude Code runs directly and can execute the automation scripts itself.

Pure-CV and synthetic-data work in `workflow_2` (e.g. `compare_align_images.py`, `test_align_key_match.py`) runs and is verified on Mac without RCS.

## Architecture Notes

### Flask Proxy VLM Architecture

Both workflows route VLM calls through a Flask proxy at the company server, which provides unified health discovery and per-service routing.

- **Service registry (server side)**: `flask_api/vlm_serve/config.py`, one `VLMServiceEntry` dataclass per model.
- **Registered services**: ui-venus (8001), mai-ui (8002), ui-tars (8003, disabled), paddleocr-vl-1.5 (8004), got-ocr (8005).
- **Health endpoint**: `GET /api/vlm_serve/health`.
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.

### `poc/workflow_1/flask_vlm.py` — client config hub

Defines `ALL_VLM_SERVICES` (a `list[VLMServiceEntry]`) plus `DEFAULT_*` service/model constants. Two connection modes:

- **`proxy`** — Flask-routed UI/OCR models: `ui-venus-1.5-8b` (primary screen analysis & tabs), `mai-ui-8b`, `paddleocr-vl-1.5` (OCR assist), `got-ocr`.
- **`direct`** — company LLM gateway (`http://common.llm.skhynix.com/v1`): `Kimi-K2.5`, `Qwen3-VL-30B-Instruct`.

Helpers: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`. Per-model debug dirs live under `debug_images/<model-slug>/` (slug via `resolve_debug_model_name()` in `poc/workflow_1/__init__.py`).

Run/step tuning lives separately in `WorkflowSettings` (`workflow_config.py`): retry budget, settle/poll timings, verify service (`paddleocr-vl-1.5`), and `service_fallback_order` (`ui-venus` → `mai-ui`). Build it with `load_workflow_settings()` (env overrides applied).

### `poc/workflow_1/` prompt builders

Each builder in `poc/workflow_1/prompts/` returns a `(system_message, user_message)` tuple and takes image `width`/`height` plus target params.

- `prompt_login_rcs_ui_venus.py` — coarse bbox for Server / UserID / Password / Login / Cancel / Shortcut.
- `prompt_login_rcs_mai_ui.py` — refined click point on the cropped+zoomed region (2-stage locator).
- `prompt_ocr_assist.py` — OCR text extraction.

### `poc/workflow_2/` — align-key search

Design rule (confirmed 2026-05-25): **OpenCV produces quantitative scores and final coordinates; VLM only identifies regions, explains ambiguous FOVs, and assesses feasibility.** Never let a VLM answer override a low CV score or decide a repeatable stage transition.

- `align_key_matcher.py` — Chamfer + ORB/RANSAC match engine (the coordinate authority). `MatchPolicy` / `DEFAULT_POLICY` / `STRUCTURE_POLICY` (cold-start thresholds, needs real-data calibration); scale bands `DEFAULT_SCALES` (immutable) and `BROAD_SCALES` (low-mag miniature search).
- `align_fail_assets.py` — resolves/loads the `align_images/...` tree (see Active Workstreams).
- `compare_align_images.py` (Step 3) — static registered-SEM vs current-SEM compare; emits score/decision + overlay + one-line verdict.
- `live_align_search.py` (Steps 4–7) — two-phase live search. Physical conventions: **double-click = recenter on click point, wheel = discrete FOV-centered zoom, template routing by OM/SEM mode.** Phase A broad zoom-out + spiral pan (budget 10); Phase B recenter → zoom-in → confirm at scale~1.0 + ORB. Real-equipment access is isolated behind the `SEMMonitorController` Protocol (Mac uses a magnification mock).
- Partially implemented / needs assets: `sem_panel_locator.py`, `vlm_sem_monitor_box.py`, `vlm_cursor_click_filter.py`, `filter_frames_by_change.py`, `search_align_key.py`. Not yet built (priority order in the procedure doc): real `SEMMonitorController` adapter, actuation safety gate, real-data calibration.

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis for GPU cluster environments. MongoDB for metadata, FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` with `PYTHONPATH=./test`.
