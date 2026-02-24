# Repository Guidelines

## Project Purpose
AI-powered automation system for CD-SEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation in semiconductor metrology equipment.

## Project Structure & Module Organization
- `poc/work/`: Main workstream POC — VLM-driven screen analysis and RCS GUI automation (Qwen3-VL API). Has its own `AGENTS.md` with module-level details.
- `poc/work/prompts/`: VLM prompt builders (one module per screen/task, e.g. `rcs_login.py`). New prompts go here and are re-exported from `prompts/__init__.py`.
- `poc/home/`: Personal/home-study variant using Hugging Face APIs. Not related to office work.
- `automation/rcs/`: Windows-focused RCS GUI automation (launcher/config/controller modules).
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests.
- `test/vlm_input_control/`: VLM-based mouse/keyboard control experiments.
- `test/workflow_extractor/`: CCTV-to-knowledge ingestion pipeline.
- `docs/`: Architecture research notes and setup guides.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `pip install -r requirements.txt`: Install all-in-one pip dependencies.
- `python -m poc.work.vlm_click_demo`: Run click-point visualization demo (manager presentation).
- `python -m poc.work.vlm_rcs_agent`: Run Observe-Think-Act RCS agent loop.
- `python -m automation.rcs.run_login`: Run RCS automation login flow (Windows only).
- `python poc/work/automate_rcs_login.py`: Run VLM-based RCS login automation (Windows only).
- `pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.

## Core Coding Rules (Must Follow)
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Korean docstrings/comments are common in this repo — keep them aligned with surrounding module conventions.
- Use print-based logging prefixes (`[INFO]`, `[WARNING]`, `[ERROR]`); do not introduce the `logging` module.
- **Absolute imports** within `poc/work/`: always use `from poc.work.xxx import ...` (not relative or bare imports). Sub-package `__init__.py` files may keep relative imports.
- Import guards for optional dependencies: `try/except ImportError` with `<LIB>_AVAILABLE` flag pattern.
- Follow existing patterns: `@dataclass` configs, enums for fixed categories, `__all__` exports in `__init__.py`.
- **No CLI arguments**: Do not use `argparse` or CLI flags. All configuration via `.env` (using `python-dotenv`) or hardcoded defaults. Scripts run with just `python <script>.py`.
- Prefer `.env` + `python-dotenv` for runtime config. Copy from `.env.example`.

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `PocConfig` | `poc/work/config.py` | Unified `.env` config (VLMConfig, RCSConfig, OperationConfig) |
| `VLMScreenAnalyzer` | `poc/work/vlm_screen_analysis.py` | Multi-provider VLM API (Qwen3-VL, Kimi-2) |
| `VLMRCSAgent` | `poc/work/vlm_rcs_agent.py` | Observe-Think-Act loop for RCS GUI automation |

## Testing Guidelines
- Primary framework: `pytest`.
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path and document required env vars.
- For imports across `test/` sibling modules, use `from video_frame_parser...` style with `PYTHONPATH=./test`.

## Platform & Workflow Notes
- Development assistant runs on macOS; Windows-only RCS/pywinauto/pynput behavior must be validated on office Windows machines.
- Do NOT remove Windows-only packages (e.g., pywinauto, pynput) from requirements files.
- Keep `poc/work/` self-contained; avoid cross-import coupling with `test/` prototypes.

## Commit & Pull Request Guidelines
- Short, imperative commit subjects: `Add ...`, `Fix ...`, `Refactor ...`, `Update ...`.
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, affected paths, and test evidence.
- For GUI automation changes, include screenshots or log snippets when possible.

## Security & Configuration
- Never commit API keys or credentials; keep them in `.env`.
- Document any new required environment variable in `.env.example`.
