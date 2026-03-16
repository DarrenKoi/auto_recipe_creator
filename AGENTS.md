# Repository Guidelines

## Project Purpose
AI-powered automation system for CD-SEM/VeritySEM recipe setup. The project combines VLM-based screen analysis with GUI automation to reduce manual recipe creation work.

## Project Structure & Module Organization
- `poc/work2/`: Primary workstream. Flask proxy-based VLM routing, shared model defaults, and Windows RCS automation experiments live here.
- `poc/home/`: Personal/home-study variant using Hugging Face APIs (`demo.py`, `test_setup.py`).
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests in `test/video_frame_parser/tests/`.
- `test/vlm_input_control/`, `test/workflow_extractor/`: Integration-style and extractor experiments.
- `docs/`: Architecture notes, GUI automation research, and setup guides.

### `poc/work2/` current structure
```text
poc/work2/
├── __init__.py
├── connection_check.py
├── flask_vlm.py
├── logger.py
├── login_rcs.py
├── open_rcs.py
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

- `poc/work2/flask_vlm.py`: Shared Flask proxy route/model registry. This is the single source of truth for coworker-facing service slug, model name, and endpoint mappings.
- `poc/work2/connection_check.py`: Flask `/api/vlm_serve/health` and per-service `/v1/models` connection checker.
- `poc/work2/vlm_client.py`: Simple image-capable client for calling Flask proxy VLM services by service slug from task scripts.
- `poc/work2/open_rcs.py`: Minimal RCS launcher. Only opens `RcsMainHD.exe` and checks whether an RCS window is already present.
- `poc/work2/login_rcs.py`: Login dialog capture + VLM coordinate marking script used during the rebuild.
- `poc/work2/util/`: Smaller helper modules for image, JSON, timing, debug-image, and window utilities.
- `poc/work2/prompts/`: Prompt builders retained under `work2`; some modules are reusable helpers for rebuild experiments rather than standalone entrypoints.
- `poc/work2/logger.py`: File logger for VLM latency / status / token usage under `poc/work2/logs/`.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `uv sync --extra home`: Install optional home-study dependencies.
- `uv run python poc/work2/connection_check.py`: Verify Flask proxy VLM/OCR routing and live `/v1/models` connectivity.
- `uv run python poc/work2/open_rcs.py`: Start `RcsMainHD.exe` only, without the old combined login automation flow.
- `uv run python poc/work2/login_rcs.py`: Capture the login dialog and save VLM-marked debug images (Windows).
- `uv run python -m poc.home.test_setup`: Validate local home-study environment.
- `uv run python -m poc.home.demo`: Run home-study VLM demo flow.
- `uv run pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.
- `uv run python -m test.vlm_input_control.integration_test`: Run input-control integration checks.
- `uv run python -m test.video_frame_parser.example_usage`: Run parser example pipeline.

## Core Coding Rules (Must Follow)
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Do not add `from __future__ import annotations` or other `__future__` imports unless the user explicitly asks for them.
- Use uv-managed execution only: run Python/test commands via `uv run ...` (do not use plain `python` or `pip`).
- Keep docstrings/comments aligned with surrounding module language conventions (Korean docstrings are common in this repo).
- Use print-based logging prefixes (`[INFO]`, `[WARNING]`, `[ERROR]`); do not introduce the `logging` module unless a file already depends on it.
- Use absolute imports within `poc/work2/` (`from poc.work2.xxx import ...`). Do not use `sys.path` hacks or `try/except` relative-vs-bare fallbacks. Sub-package `__init__.py` files may keep relative imports.
- Use import guards for optional dependencies with `<LIB>_AVAILABLE` flags.
- Follow existing module patterns: dataclass-based configs and enums for fixed categories. Explicit `__all__` exports in package initializers are optional, not required.
- Prefer `.env` + `python-dotenv` for runtime config loading.
- For data models used with storage layers (MongoDB/FAISS flows), keep `to_dict()` / `from_dict()` serialization patterns.
- Save debug screenshots locally as **JPEG** (smaller file size). Convert images to **WebP** (quality=90) before sending to VLM APIs to reduce payload size. WebP does not hurt VLM recognition accuracy.
- Default to safe/offline behavior in automation flows (`SAFE_MODE=true`) unless explicitly required otherwise.
- Do not add new CLI-flag driven entrypoints (`argparse`) for operational scripts; prefer `.env` + in-code defaults.
- No repo-wide formatter/linter config is committed; avoid introducing style drift within edited files.

## Testing Guidelines
- Primary framework: `pytest` (see `test/video_frame_parser/tests/test_*.py`).
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path and document required env vars.

## Platform & Workflow Notes
- Development assistant environment is macOS; Windows-only RCS/pywinauto/pynput behavior must be validated by running updated code on office Windows machines.
- Treat `poc/work2/` as the primary implementation surface. New automation and pipeline work should land there by default.
- `poc/work2/` must remain independent from server-side `flask_api` source code and its local env assumptions. Coworkers should be able to run `poc/work2` with only this repo and the hardcoded/shared client-side endpoint definitions in `poc/work2/flask_vlm.py`.
- Run `poc/work2` scripts via `uv run python poc/work2/<script>.py` unless a module explicitly requires another invocation form.
- For imports across `test/` sibling modules, use `from video_frame_parser...` style when operating with `PYTHONPATH=./test`.
- Canonical office gateway host is `itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com`; if repo docs mention `webpp`, treat that as stale and keep `webapp`.

## `poc/work2` Plan
- Goal: make `poc/work2` the main experimentation and delivery surface for Flask proxy-based VLM/OCR automation.
- Operating principles:
- Keep prompts scenario-specific and small (`login`, `main_tabs`, `screen_analysis`, `ocr_assist`) instead of building one universal prompt.
- Enforce strict JSON outputs and validate or repair fields before using them for clicks, OCR hints, or UI-state decisions.
- Use stepwise execution: `observe` -> `decide` -> `act` -> `verify`.
- Keep debug artifacts first-class: local JPEG screenshots, WebP payloads for VLM calls, marked overlays, and call logs.
- Preserve safe/default-off behavior in Windows automation unless a script explicitly needs to act.
- Current focus areas:
- Centralize team-default pipeline settings in `poc/work2/flask_vlm.py` so coworkers can run scripts without per-user `.env` sprawl.
- Use `poc/work2/connection_check.py` to validate Flask proxy health and route or model readiness before debugging automation behavior.
- Prefer direct model selection after `poc/work2/connection_check.py`: check available services, then hardcode the service slug each script wants to use.
- Consolidate reusable Windows automation helpers in `poc/work2/util/` and reusable prompt builders in `poc/work2/prompts/`.
- Validate the real RCS workflow on office Windows in this order: connection check -> open RCS -> login dialog analysis.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (history pattern: `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`, `Fix ...`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, affected paths, test evidence (commands + results), and screenshots or log snippets for GUI automation changes.
- Link related issues or docs and note platform constraints (for example, Windows-only behavior in `poc/work2/`).
