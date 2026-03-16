# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation in semiconductor metrology equipment.

## Repository Structure

```
poc/work2/               # Primary workstream: Flask proxy VLM routing, RCS rebuild scripts, shared helpers
poc/work2/prompts/       # VLM prompt builders (one module per screen/task)
poc/home/                # Personal study PoC: HuggingFace free API — NO office relation
flask_api/               # Flask API services (VLM proxy, health endpoints)
flask_api/vlm_serve/     # VLM service registry, health-driven discovery, per-model proxy blueprints
deploy_vlms/             # VLM deployment configs, scripts, and operational docs
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU cluster)
test/workflow_extractor/ # CCTV-to-knowledge ingestion pipeline
docs/                    # Architecture research notes and setup guides
```

### Current `poc/work2/` structure

```text
poc/work2/
├── __init__.py
├── connection_check.py
├── flask_vlm.py
├── logger.py
├── login_rcs.py
├── open_rcs.py
├── reading_check.py
├── vlm_client.py
├── debug_images/
│   └── .gitkeep
├── prompts/
│   ├── __init__.py
│   ├── ocr_assist.py
│   ├── rcs_login.py
│   ├── rcs_main_tabs.py
│   └── screen_analysis.py
└── util/
    ├── __init__.py
    ├── debug_image_utils.py
    ├── image_utils.py
    ├── json_utils.py
    ├── time_utils.py
    └── window_utils.py
```

## Setup & Dependencies

The project uses `uv` with `pyproject.toml` (Python >= 3.10). Each module also has its own `requirements.txt` for standalone installs.
Use uv-managed workflows by default: `uv sync`, `uv run`, and `uv pip ...`.

```bash
# Core project + dev tools
uv sync --extra dev

# Add home-study extras (HuggingFace)
uv sync --extra home

# All-in-one requirements install via uv-managed pip compatibility
uv pip install -r requirements.txt

# Per-module installs (alternative)
uv pip install -r test/vlm_input_control/requirements.txt
uv pip install -r test/video_frame_parser/requirements.txt  # torch, opencv, pymongo, faiss
```

## Running Modules

```bash
# poc/work2 — company automation (Flask proxy, no per-PC .env needed)
uv run python poc/work2/connection_check.py    # Verify Flask proxy + VLM service health
uv run python poc/work2/open_rcs.py            # Start RCS only
uv run python poc/work2/reading_check.py       # Multi-VLM UI component comparison
uv run python poc/work2/login_rcs.py           # Login dialog capture + VLM marking

# poc/home — personal study only
uv run python -m poc.home.test_setup           # Validate HuggingFace env
uv run python -m poc.home.demo                 # Mode configured in .env or hardcoded

# Video frame parser
uv run python -m test.video_frame_parser.example_usage
```

## Testing

```bash
# Unit tests
uv run pytest test/video_frame_parser/tests/
uv run pytest test/video_frame_parser/tests/test_analyzer.py -v

# Integration test (safe mode by default; toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules
- **No `__future__` imports by default**: Do not add `from __future__ import annotations` or any other `__future__` import unless the user explicitly asks for it.
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/work2/logger.py` uses Python `logging` with `RotatingFileHandler` for VLM call audit trail (`poc/work2/logs/vlm_calls.log`).
- **Absolute imports** within `poc/work2/`: use `from poc.work2.xxx import ...`.
- **`__all__` in `__init__.py` is optional**: Do not add or maintain explicit `__all__` exports unless they provide clear value for a curated package API.
- **Image format convention**: Save debug screenshots locally as **JPEG**. When sending images to VLM APIs, convert to **WebP** (quality=90) to reduce payload size without hurting recognition accuracy.
- **Safe mode**: Most interactive modules default to `SAFE_MODE=true` to prevent actual mouse or keyboard output.
- **No CLI arguments**: Do not use `argparse` or CLI flags. All configuration comes from `flask_vlm.SHARED_PIPELINE_SETTINGS`, environment variables, or hardcoded defaults in the source files. Scripts should run with just `uv run python <script>.py`.

## Development Workflow

Claude Code runs on macOS and cannot see or interact with the actual RCS application. All Windows-only automation paths in `poc/work2/` (RCS, pywinauto, pynput mouse or keyboard) are tested by the user at the office on a Windows machine. Updated Python files are pushed via git, pulled at the office, and run there. Debugging relies on the user reporting results such as console output and debug screenshots in `poc/work2/debug_images/`.

## Architecture Notes

### `poc/work2/` (Current Workstream)

Flat module with a `prompts/` sub-package for VLM prompt builders. `flask_vlm.py` is the central config hub. `SHARED_PIPELINE_SETTINGS` holds team-wide defaults such as Flask base URL, primary VLM service or model, OCR service or model, and pipeline flags.

- **Primary VLM**: `ui-venus-1.5-8b` (service slug: `ui-venus`)
- **OCR assist VLM**: `paddleocr-vl-1.5` (service slug: `paddleocr-vl-1.5`)
- Per-model debug dirs under `debug_images/<model-slug>/`
- Rotating file logger at `poc/work2/logs/vlm_calls.log` (10MB max, 5 backups)

Config resolution starts from `SHARED_PIPELINE_SETTINGS`, then applies environment overrides, then script-level handling.

### Flask Proxy VLM Architecture

`poc/work2/` routes through a Flask proxy at the company server. The proxy provides unified health discovery and per-service routing.

- **Service registry**: `flask_api/vlm_serve/config.py` with one `VLMServiceEntry` dataclass per model
- **Registered services**: ui-venus (8001), mai-ui (8002), ui-tars (8003, disabled), paddleocr-vl-1.5 (8004), got-ocr (8005)
- **Health endpoint**: `GET /api/vlm_serve/health`
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`
- `flask_vlm.py` helpers: `resolve_service_proxy_url()`, `fetch_vlm_health()`, `normalize_vlm_health_entries()`

### RCS Automation Workflow (`poc/work2/`)

Each step is a standalone script. All use Flask proxy routing via `flask_vlm.py`.

1. `connection_check.py` verifies Flask API health and probes each VLM service's `/v1/models` endpoint.
2. `open_rcs.py` starts `RcsMainHD.exe` only.
3. `login_rcs.py` captures the login dialog, runs a selected VLM on the image, and saves marked debug outputs.
4. `reading_check.py` captures a monitor screenshot, sends it to multiple UI VLMs in parallel, and compares component or coordinate responses.

### `poc/work2/` VLM Prompt Builders

Each prompt builder in `poc/work2/prompts/` returns a `(system_message, user_message)` tuple for the VLM request flow. They take image `width` or `height` and target-specific params.

- `build_ocr_assist_prompt()` for OCR text extraction
- `build_rcs_login_locator_prompt()` for Server, UserID, Password, LoginButton, CancelButton, and ShortcutButton coordinates
- `build_rcs_main_tab_locator_prompt()` for tab center coordinates with first-letter anchoring
- `build_state_recognition_prompt()` for general screen state recognition
- `build_measurement_judgment_prompt()` for measurement success or failure judgment with suggested adjustments
- `build_general_query_prompt()` for generic screen QA

### Shared Utilities (`poc/work2/util/`)

The old monolithic `rcs_utils.py` has been split into smaller helper modules.

- `image_utils.py` for `capture_window()` and `encode_image_webp()`
- `json_utils.py` for `extract_json()` and `parse_coords()`
- `debug_image_utils.py` for debug image pathing and local saves
- `window_utils.py` for window lookup and activation helpers

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis module for GPU cluster environments. Uses MongoDB for metadata and FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` when operating with `PYTHONPATH=./test`.
