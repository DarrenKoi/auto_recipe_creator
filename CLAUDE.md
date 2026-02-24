# CLAUDE.md

## Project Purpose

AI-powered automation system for CD-SEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation in semiconductor metrology equipment.

## Repository Structure

```
poc/work/                # Company PoC: VLM screen analysis + RCS automation (Qwen3-VL API)
poc/work/prompts/        # VLM prompt builders (one module per screen/task, e.g. rcs_login.py)
docs/                    # Architecture research notes and setup guides
```

## Setup & Dependencies

The project uses `uv` with `pyproject.toml` (Python ≥ 3.10). Each module also has its own `requirements.txt` for standalone installs.

# All-in-one pip install (core + automation + RAG)
pip install -r requirements.txt
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

```

## Code Conventions
- **Absolute imports** within `poc/work/`: always use `from poc.work.xxx import ...` (not relative or bare imports). Scripts are run via `uv run python <script>.py` or `python -m poc.work.<module>`.
- **No CLI arguments**: Do not use `argparse` or CLI flags. All configuration comes from `.env` (via `python-dotenv`) or hardcoded defaults in the source files. Scripts should run with just `python <script>.py`

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `PocConfig` | `poc/work/config.py` | Unified `.env` config (VLMConfig, RCSConfig, OperationConfig) |
| `VLMScreenAnalyzer` | `poc/work/vlm_screen_analysis.py` | Multi-provider VLM API (Qwen3-VL, Kimi-2) |
| `VLMRCSAgent` | `poc/work/vlm_rcs_agent.py` | Observe-Think-Act loop for RCS GUI automation |

## Development Workflow

Claude Code runs on macOS (dev machine) and **cannot see or interact with the actual RCS application**. All Windows-only automation (RCS, pywinauto, pynput mouse/keyboard) is tested by the user at the office on a Windows machine. Updated Python files are pushed via git, pulled at the office, and run there. Debugging relies on the user reporting results (console output, debug screenshots like `debug_vlm_login.png`) back to Claude Code.

## Architecture Notes

### poc/work/ (Primary Workstream)
Mostly flat module with a `prompts/` sub-package for VLM prompt builders. New prompts go in `prompts/` and are re-exported from `prompts/__init__.py`. All internal imports use absolute paths (`from poc.work.xxx import ...`); `prompts/__init__.py` keeps relative imports since sub-packages are never run directly. Config loaded via `PocConfig.load()` which reads `.env` (copy from `.env.example`; `.env.example` now includes `RCS_EXE_PATH` for the path to `RcsMainHD.exe`). The `vlm_click_demo.py` is the primary manager-presentation entry point: it captures a screenshot, sends it to the VLM, then draws bounding boxes at the returned click coordinates. Coordinate chain: VLM output coords (resized image) → screenshot pixels → monitor-local coords → absolute mouse coords (offset for multi-monitor setups via `MONITOR_INDEX`).

### poc/work/ RCS automation
`automate_rcs_login.py` — VLM-based RCS login automation. pywinauto is only used to launch the exe and find the window by title ("Remote Control System"); internal control detection via pywinauto failed (legacy app doesn't expose ComboBox/Button to UIA or win32 backends). Instead uses: mss screenshot of window region → VLM coordinate extraction (asks for Server/UserID/Password/LoginButton click points) → pynput mouse clicks and keyboard typing at the returned positions. Config from `.env`: `RCS_EXE_PATH`, `RCS_SERVER`, `RCS_USERNAME`, `RCS_PASSWORD`, `VLM_API_URL`, `VLM_API_KEY`, `VLM_MODEL_NAME`. Coordinate chain: VLM coords (resized image) → ÷ resize_scale → screenshot coords → + window offset → absolute screen coords.
