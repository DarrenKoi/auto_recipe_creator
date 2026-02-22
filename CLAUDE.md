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
uv run python -m poc.home.demo --mode screen_analysis

# RCS auto-login (Windows only)
python -m automation.rcs.run_login --server SERVER --username USER --password PASS
python -m automation.rcs.run_login --debug           # dump pywinauto UI control tree

# RCS auto-login via poc/work/ (Windows only, all config from .env)
python poc/work/automate_rcs_login.py

# Video frame parser
python -m test.video_frame_parser.example_usage
```

## Testing

```bash
# Unit tests
pytest test/video_frame_parser/tests/
pytest test/video_frame_parser/tests/test_analyzer.py -v

# Integration test (safe mode by default — no actual inputs sent)
python -m test.vlm_input_control.integration_test
python -m test.vlm_input_control.integration_test --live   # CAUTION: sends real inputs
```

## Code Conventions

- **Korean docstrings** throughout all modules
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module)
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

## Architecture Notes

### poc/work/ (Primary Workstream)
Self-contained flat module — all files live directly in `poc/work/` with no sub-packages. Config loaded via `PocConfig.load()` which reads `.env` (copy from `.env.example`; `.env.example` now includes `RCS_EXE_PATH` for the path to `RcsMainHD.exe`). The `vlm_click_demo.py` is the primary manager-presentation entry point: it captures a screenshot, sends it to the VLM, then draws bounding boxes at the returned click coordinates. Coordinate chain: VLM output coords (resized image) → screenshot pixels → monitor-local coords → absolute mouse coords (offset for multi-monitor setups via `MONITOR_INDEX`).

`opensearch_handler.py` is intentionally kept but inactive — import-guarded and `opensearch-py` is not in `requirements.txt`. Do not delete; kept for re-enablement after company PoC approval.

### poc/work/ RCS automation
`automate_rcs_login.py` — simple pywinauto script that reads all config from `.env` (`RCS_EXE_PATH`, `RCS_SERVER`, `RCS_USERNAME`, `RCS_PASSWORD`). Launches RCS, finds the "Remote Control System" window, fills Server/User ID/Password, clicks Log In.

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
