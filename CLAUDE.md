# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation in semiconductor metrology equipment.

## Repository Structure

```
poc/work2/               # Phase 2 primary: Flask proxy VLM routing, multi-model benchmark, OCR pipeline
poc/work2/prompts/       # VLM prompt builders (one module per screen/task)
poc/work/                # Phase 1 legacy — shared utilities only (vlm_openai_client, screen_capture, config, rcs_common)
poc/home/                # Personal study PoC: HuggingFace free API — NO office relation
flask_api/               # Flask API services (VLM proxy, health endpoints)
flask_api/vlm_serve/     # VLM service registry, health-driven discovery, per-model proxy blueprints
deploy_vlms/             # VLM deployment configs, scripts, and operational docs
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU cluster)
test/workflow_extractor/ # CCTV-to-knowledge ingestion pipeline
docs/                    # Architecture research notes and setup guides
```

## Setup & Dependencies

The project uses `uv` with `pyproject.toml` (Python ≥ 3.10). Each module also has its own `requirements.txt` for standalone installs.
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
# poc/work2 — Phase 2 company automation (Flask proxy, no per-PC .env needed)
uv run python poc/work2/connection_check.py    # Verify Flask proxy + VLM service health
uv run python poc/work2/reading_check.py       # Multi-VLM UI component comparison
uv run python poc/work2/automate_rcs_login.py  # Multi-model benchmark + RCS login
uv run python poc/work2/click_rcs_view_mode.py # Tab switching with OCR assist
uv run python poc/work2/check_tool_screen.py   # Tool viewer detection + VLM analysis

# poc/home — personal study only
uv run python -m poc.home.test_setup       # Validate HuggingFace env
uv run python -m poc.home.demo             # mode configured in .env or hardcoded

# Video frame parser
uv run python -m test.video_frame_parser.example_usage
```

## Testing

```bash
# Unit tests
uv run pytest test/video_frame_parser/tests/
uv run pytest test/video_frame_parser/tests/test_analyzer.py -v

# Integration test (safe mode by default — no actual inputs sent, toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/work2/logger.py` uses Python `logging` with `RotatingFileHandler` for VLM call audit trail (`poc/work2/logs/vlm_calls.log`).
- **Absolute imports** within `poc/work2/`: use `from poc.work2.xxx import ...`. Cross-module imports from `poc.work` shared utilities use `from poc.work.xxx import ...` (e.g., `vlm_openai_client`, `screen_capture`, `config`, `rcs_common`).
- **Image format convention**: Save debug screenshots locally as **JPEG** (smaller file size for storage). When sending images to VLM APIs, convert to **WebP** (quality=90) to reduce API payload size — WebP compression does not hurt VLM coordinate/element recognition accuracy.
- **Safe mode**: Most interactive modules default to `SAFE_MODE=true` to prevent actual mouse/keyboard output
- **No CLI arguments**: Do not use `argparse` or CLI flags. All configuration comes from `flask_vlm.SHARED_PIPELINE_SETTINGS` (team defaults), environment variables (overrides), or hardcoded defaults in the source files. Scripts should run with just `uv run python <script>.py`

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `VLMScreenAnalyzer` | `poc/work2/vlm_screen_analysis.py` | Primary+OCR pipeline screen analysis |
| `ScreenAnalysisResult` | `poc/work2/vlm_screen_analysis.py` | State recognition result (state_id, confidence, ui_elements, suggested_actions) |
| `MeasurementJudgment` | `poc/work2/vlm_screen_analysis.py` | Measurement success/failure judgment |
| `OCRHintResult` | `poc/work2/pipeline_ocr.py` | OCR text extraction result (texts, focus_hits) |
| `ToolScreenSettings` | `poc/work2/check_tool_screen.py` | Tool viewer detection config (tool_name, timeout, backends, vlm_analyze) |
| `ChatImageRequest` | `poc/work/vlm_openai_client.py` | VLM API request builder (shared, imported by work2) |
| `LangChainOpenAICompatibleVLMClient` | `poc/work/vlm_openai_client.py` | OpenAI-compatible VLM client (shared, imported by work2) |
| `ScreenCapture` | `poc/work/screen_capture.py` | Screenshot utility via mss (shared, imported by work2) |
| `PocConfig` | `poc/work/config.py` | Legacy .env config loader (shared, imported by work2) |

## Development Workflow

Claude Code runs on macOS (dev machine) and **cannot see or interact with the actual RCS application**. All Windows-only automation paths in `poc/work2/` (RCS, pywinauto, pynput mouse/keyboard) are tested by the user at the office on a Windows machine. Updated Python files are pushed via git, pulled at the office, and run there. Debugging relies on the user reporting results (console output, debug screenshots in `poc/work2/debug_images/`) back to Claude Code.

## Architecture Notes

### poc/work2/ (Primary Workstream)

Flat module with a `prompts/` sub-package for VLM prompt builders. `flask_vlm.py` is the central config hub — `SHARED_PIPELINE_SETTINGS` dict holds team-wide defaults (Flask base URL, primary VLM service/model, OCR service/model, pipeline flags). No individual `.env` files needed; teammates share hardcoded defaults in `flask_vlm.py`.

- **Primary VLM**: `ui-venus-1.5-8b` (service slug: `ui-venus`)
- **OCR assist VLM**: `paddleocr-vl-1.5` (service slug: `paddleocr-vl-1.5`)
- Cross-module imports from `poc.work` for shared utilities: `vlm_openai_client` (ChatImageRequest, LangChainOpenAICompatibleVLMClient), `screen_capture` (ScreenCapture), `config` (PocConfig), `rcs_common`
- Per-model debug dirs under `debug_images/<model-slug>/`
- Rotating file logger at `poc/work2/logs/vlm_calls.log` (10MB max, 5 backups)

Config resolution order: `SHARED_PIPELINE_SETTINGS` → env var overrides → `apply_pipeline_env_defaults()` injects into `os.environ` for backward compatibility with `poc.work` code.

### Flask proxy VLM architecture

Instead of direct VLM API calls, `poc/work2/` routes through a Flask proxy at the company server. The proxy provides unified health discovery and per-service routing.

- **Service registry**: `flask_api/vlm_serve/config.py` — `VLMServiceEntry` dataclass per model (route_slug, display_name, model_name, upstream_port, enabled flag)
- **Registered services**: ui-venus (8001), mai-ui (8002), ui-tars (8003, disabled), paddleocr-vl-1.5 (8004), got-ocr (8005)
- **Health endpoint**: `GET /api/vlm_serve/health` — returns `vlm_statuses` array with per-service health_status (serving/unreachable/error), proxy_registered flag, detected models
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`
- `flask_vlm.py` helpers: `resolve_service_proxy_url()`, `fetch_vlm_health()`, `normalize_vlm_health_entries()`

### Two-stage VLM pipeline

Primary VLM + OCR assist (PaddleOCR-VL) for improved accuracy on GUI text elements.

1. `pipeline_ocr.collect_ocr_hint_result()` sends screenshot to PaddleOCR-VL, extracts visible text lines as `OCRHintResult` (texts + focus_hits matching target words)
2. `pipeline_ocr.build_ocr_extra_instructions()` converts OCR result to instruction tuples injected into the primary VLM prompt
3. Primary VLM (ui-venus) receives the image + OCR hints as `extra_instructions` and makes final coordinate/element decisions from pixels

OCR hints are advisory only — the primary VLM always makes final decisions from the actual image.

### RCS automation workflow (poc/work2/)

Each step is a standalone script. All use Flask proxy routing via `flask_vlm.py`:

1. **`connection_check.py`** — Verifies Flask API health + probes each VLM service's `/v1/models` endpoint. Renders status table.
2. **`automate_rcs_login.py`** — Multi-model VLM benchmark on RCS login screen:
   - Auto-discovers benchmark targets from Flask health (serving + proxy_registered)
   - Captures login window → single OCR call (shared) → loops over target services with `build_rcs_login_locator_prompt()`
   - Compares detection accuracy, saves per-model marked images, prints comparison table
   - Executes click using best result (default: login_button)
   - Waits for post-login main window via regex matching
3. **`click_rcs_view_mode.py`** — Tab switching with OCR assist. Uses first-letter anchoring ('V' for View, 'L' for List). Applies offset correction (list_tab.x = view_tab.x + 50px).
4. **`check_tool_screen.py`** — Polls for tool viewer window (`RcsViewerHD.exe`), optionally captures + analyzes with VLMScreenAnalyzer (OCR-assisted). Saves source JPEG + marked overlay.
5. **`reading_check.py`** — Captures monitor screenshot, sends to multiple UI VLMs in parallel, compares component/coordinate responses. Saves per-model overlays + normalized JSON.

### poc/work2/ VLM prompt builders

Each prompt builder in `poc/work2/prompts/` returns a `(system_message, user_message)` tuple for use with `ChatImageRequest`. They take image `width`/`height` and target-specific params. Current prompts:
- `build_ocr_assist_prompt()` — OCR text extraction task (PaddleOCR-VL format)
- `build_rcs_login_locator_prompt()` — Server/UserID/Password/LoginButton/CancelButton/ShortcutButton coordinates
- `build_rcs_main_tab_locator_prompt()` — Tab center coordinates with first-letter anchoring (View, List)
- `build_state_recognition_prompt()` — General screen state recognition (state_id, confidence, ui_elements, suggested_actions)
- `build_measurement_judgment_prompt()` — Measurement success/failure judgment with suggested adjustments
- `build_general_query_prompt()` — Generic screen QA

### Shared utilities (poc/work2/rcs_utils.py)

Key functions for RCS automation scripts:
- `capture_window()` — mss screenshot of pywinauto window region → PIL Image
- `encode_image_webp()` — PIL Image → base64 WebP (quality=90)
- `click_at()` — VLM coords → screen coords → click (with retry, falls back to pywinauto.mouse)
- `extract_json()` — Extracts JSON from VLM response text (handles markdown fences)
- `parse_coords()` — Validates/converts VLM coordinate dict to integers, checks bounds
- `find_existing_main_window()` — Desktop-wide window search across pywinauto backends
- `save_marked_image()` — Overlays crosshairs + circles + labels at detected coordinates
- `debug_image_path()` — Returns model-specific debug image path

### poc/work2/ vs poc/work/ (Phase comparison)

| Aspect | Phase 1 (poc/work/) | Phase 2 (poc/work2/) |
|--------|---------------------|----------------------|
| VLM routing | Direct API calls (per-PC .env) | Flask proxy (shared team config) |
| Model selection | Single model (VLM_MODEL_NAME) | Multi-model benchmark + health discovery |
| OCR assist | None | PaddleOCR-VL two-stage pipeline |
| Config source | `.env` via `PocConfig.load()` | `flask_vlm.SHARED_PIPELINE_SETTINGS` |
| Debug images | Flat directory | Per-model subdirectories |
| VLM logging | Print only | `logger.py` rotating file + print |

`poc/work/` remains in the repo for shared utilities (`vlm_openai_client`, `screen_capture`, `config`, `rcs_common`) that `poc/work2/` imports.

### test/video_frame_parser/

CLIP-based video frame extraction and analysis module for GPU cluster environments. Uses MongoDB for metadata, FAISS for similarity search. Import pattern for `test/` siblings: use `from video_frame_parser.xxx import Yyy` (requires `PYTHONPATH=./test`), always wrapped in `try/except ImportError` with `AVAILABLE` flag.
