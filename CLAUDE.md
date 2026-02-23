# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation in semiconductor metrology equipment.

## Repository Structure

```
automation/rcs/          # RCS GUI automation (Windows-only, pywinauto)
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU cluster)
test/workflow_extractor/ # CCTV-to-knowledge ingestion pipeline
poc/work/                # Company PoC: VLM screen analysis + RCS automation (Qwen3-VL API)
poc/work/prompts/        # VLM prompt builders (one module per screen/task, e.g. rcs_login.py)
poc/home/                # Personal study PoC: HuggingFace free API — NO office relation
docs/                    # Architecture research notes and setup guides
```

## Setup & Dependencies

The project uses `uv` with `pyproject.toml` (Python ≥ 3.10). Each module also has its own `requirements.txt` for standalone installs.

```bash
# Core project + dev tools
uv sync --extra dev

# Add home-study extras (HuggingFace)
uv sync --extra home

# All-in-one pip install (core + automation + RAG)
pip install -r requirements.txt

# Per-module pip installs (alternative)
pip install -r poc/work/requirements.txt                 # core: mss, pynput, Pillow, requests, python-dotenv
pip install -r test/vlm_input_control/requirements.txt
pip install -r test/video_frame_parser/requirements.txt  # torch, opencv, pymongo, faiss
```

## Running Modules

```bash
# poc/work — main company demo (requires .env with VLM_API_URL, VLM_API_KEY)
python -m poc.work.vlm_click_demo          # VLM click-point visualization (manager demo)
python -m poc.work.vlm_rcs_agent           # Observe-Think-Act agent loop for RCS

# poc/home — personal study only
uv run python -m poc.home.test_setup       # Validate HuggingFace env
uv run python -m poc.home.demo                        # mode configured in .env or hardcoded

# RCS auto-login (Windows only, config from .env)
python -m automation.rcs.run_login

# RCS auto-login via poc/work/ (Windows only, VLM-based, all config from .env)
# Saves debug_vlm_login.png with VLM-detected coordinates marked
python poc/work/automate_rcs_login.py

# Video frame parser
python -m test.video_frame_parser.example_usage
```

## Testing

```bash
# Unit tests
pytest test/video_frame_parser/tests/
pytest test/video_frame_parser/tests/test_analyzer.py -v

# Integration test (safe mode by default — no actual inputs sent, toggle via SAFE_MODE in .env)
python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module)
- **Absolute imports** within `poc/work/`: always use `from poc.work.xxx import ...` (not relative or bare imports). Scripts are run via `uv run python <script>.py` or `python -m poc.work.<module>`.
- **Import guards** for optional dependencies with `LIBRARY_AVAILABLE` flag:
  ```python
  try:
      import pywinauto
      PYWINAUTO_AVAILABLE = True
  except ImportError:
      PYWINAUTO_AVAILABLE = False
  ```
- **`@dataclass` config classes** with Korean field comments, loaded from `.env` via `python-dotenv`
- **`__all__` exports** in every `__init__.py` with relative imports
- **Enums** for categorical values (`FrameType`, `AnalysisStatus`, `VLMProvider`, `MouseButton`)
- **`to_dict()` / `from_dict()`** on data models for MongoDB serialization
- **Safe mode**: Most interactive modules default to `SAFE_MODE=true` to prevent actual mouse/keyboard output
- **No CLI arguments**: Do not use `argparse` or CLI flags. All configuration comes from `.env` (via `python-dotenv`) or hardcoded defaults in the source files. Scripts should run with just `python <script>.py`

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `PocConfig` | `poc/work/config.py` | Unified `.env` config (VLMConfig, RCSConfig, OperationConfig) |
| `VLMScreenAnalyzer` | `poc/work/vlm_screen_analysis.py` | Multi-provider VLM API (Qwen3-VL, GPT-4V, Claude, Kimi-2) |
| `VLMRCSAgent` | `poc/work/vlm_rcs_agent.py` | Observe-Think-Act loop for RCS GUI automation |
| `RCSConfig` | `automation/rcs/rcs_config.py` | RCS connection/login settings |
| `RCSLauncher` | `automation/rcs/rcs_launcher.py` | Orchestrates full RCS login sequence |
| `ScreenCapture` | `test/vlm_input_control/screen_capture.py` | Screen/region capture via mss |
| `MouseController` | `test/vlm_input_control/mouse_control.py` | Mouse input via pynput |
| `KeyboardController` | `test/vlm_input_control/keyboard_control.py` | Keyboard input via pynput |
| `VideoFrameParser` | `test/video_frame_parser/parser.py` | Main video processing pipeline |
| `FrameAnalyzer` | `test/video_frame_parser/analyzer.py` | CLIP-based frame embeddings |
| `BatchProcessor` | `test/video_frame_parser/batch_processor.py` | Multi-GPU batch processing |
| `DatabaseHandler` | `test/video_frame_parser/db_handler.py` | MongoDB + FAISS vector storage |

## Development Workflow

Claude Code runs on macOS (dev machine) and **cannot see or interact with the actual RCS application**. All Windows-only automation (RCS, pywinauto, pynput mouse/keyboard) is tested by the user at the office on a Windows machine. Updated Python files are pushed via git, pulled at the office, and run there. Debugging relies on the user reporting results (console output, debug screenshots like `debug_vlm_login.png`) back to Claude Code.

## Architecture Notes

### poc/work/ (Primary Workstream)
Mostly flat module with sub-packages: `prompts/` for VLM prompt builders and `steps/` for standalone step runners. New prompts go in `prompts/` and are re-exported from `prompts/__init__.py`. All internal imports use absolute paths (`from poc.work.xxx import ...`); `prompts/__init__.py` and `steps/__init__.py` keep relative imports since sub-packages are never run directly. Config loaded via `PocConfig.load()` which reads `.env` (copy from `.env.example`; `.env.example` now includes `RCS_EXE_PATH` for the path to `RcsMainHD.exe`). The `vlm_click_demo.py` is the primary manager-presentation entry point: it captures a screenshot, sends it to the VLM, then draws bounding boxes at the returned click coordinates. Coordinate chain: VLM output coords (resized image) → screenshot pixels → monitor-local coords → absolute mouse coords (offset for multi-monitor setups via `MONITOR_INDEX`).

`opensearch_handler.py` is intentionally kept but inactive — import-guarded and `opensearch-py` is not in `requirements.txt`. Do not delete; kept for re-enablement after company PoC approval.

### poc/work/ RCS automation
`automate_rcs_login.py` — VLM-based RCS login automation. pywinauto is only used to launch the exe and find the window by title ("Remote Control System"); internal control detection via pywinauto failed (legacy app doesn't expose ComboBox/Button to UIA or win32 backends). Instead uses: mss screenshot of window region → VLM coordinate extraction (asks for Server/UserID/Password/LoginButton click points) → pynput mouse clicks and keyboard typing at the returned positions. Config from `.env`: `RCS_EXE_PATH`, `RCS_SERVER`, `RCS_USERNAME`, `RCS_PASSWORD`, `VLM_API_URL`, `VLM_API_KEY`, `VLM_MODEL_NAME`. Coordinate chain: VLM coords (resized image) → ÷ resize_scale → screenshot coords → + window offset → absolute screen coords.

Current post-login success detection policy (`poc/work/automate_rcs_login.py`):
- Login success is determined by the final main window only. Updater window checks are intentionally skipped.
- Final title match uses regex (`RCS_MAIN_WINDOW_REGEX`) requiring both `RCS` and `[Server : ...]` semantics (default: `\brcs\b.*\[server\s*:[^\]]+\]`).
- Wait behavior defaults: `RCS_POST_LOGIN_DELAY_SEC=4.0`, `RCS_POST_LOGIN_MAIN_TIMEOUT_SEC=240.0`, `RCS_POST_LOGIN_POLL_SEC=0.5`.
- Window discovery order:
- 1) `app.windows()` from the launched process
- 2) desktop-wide fallback (`Desktop(...).windows(top_level_only=True, visible_only=True)`) because RCS may relaunch into another process
- Desktop backend priority defaults to `win32,uia` (`RCS_DESKTOP_SCAN_BACKENDS`).
- Debug title scan logs are controlled by `RCS_DEBUG_MAIN_WINDOW_TITLES` and default to off (`0`).

### poc/work/ vs test/vlm_input_control/
Both implement screen capture + VLM + input control, but `poc/work/` is self-contained (no shared imports with `test/`) and production-oriented. `test/vlm_input_control/` is an older integration prototype.

### automation/rcs/
Windows-only. Uses `uia` or `win32` pywinauto backends. `RCSLauncher.run()` orchestrates: launch exe → wait for window → select server → enter credentials → verify login, with retry logic.

### test/video_frame_parser/
Designed for H200 GPU cluster. Pipeline: extract frames (OpenCV) → generate CLIP embeddings (torch) → store in MongoDB + FAISS index. Factory shortcut: `create_h200_optimized_parser(num_gpus=8)`.

### Import Pattern for test/ sibling modules
When importing across `test/` sub-packages use `from video_frame_parser.xxx import Yyy` (not `from test.video_frame_parser...`). Requires `PYTHONPATH=./test` or running from within `test/`.

## Commit Style

Short imperative subjects following the project history pattern: `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`, `Fix ...`. Keep commits scoped to one logical change.
