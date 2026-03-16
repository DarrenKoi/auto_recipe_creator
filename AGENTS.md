# Repository Guidelines

## Project Purpose
AI-powered automation system for CD-SEM/VeritySEM recipe setup. The project combines VLM-based screen analysis with GUI automation to reduce manual recipe creation work.

## Project Structure & Module Organization
- `poc/work2/`: Phase 2 primary workstream. Flask proxy-based VLM/OCR pipeline, shared model routing, and Windows RCS automation experiments live here.
- `poc/work/`: Phase 1 legacy/reference area. Keep only reusable concepts and temporary shared utilities needed during migration; do not add new feature work here unless it directly unblocks `poc/work2`.
- `poc/home/`: Personal/home-study variant using Hugging Face APIs (`demo.py`, `test_setup.py`).
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests in `test/video_frame_parser/tests/`.
- `test/vlm_input_control/`, `test/workflow_extractor/`: Integration-style and extractor experiments.
- `docs/`: Architecture notes, GUI automation research, and setup guides.

### `poc/work2/` detailed map (current)
- `poc/work2/flask_vlm.py`: Shared Phase 2 Flask proxy route/model registry. This is the single source of truth for coworker-facing service slug, model name, and endpoint mappings.
- `poc/work2/connection_check.py`: Flask `/api/vlm_serve/health` and per-service `/v1/models` connection checker.
- `poc/work2/vlm_client.py`: Simple image-capable client for calling Flask proxy VLM services by service slug from task scripts.
- `poc/work2/reading_check.py`: Single-monitor screenshot benchmark that compares multiple UI VLM services and optionally OCR assist output.
- `poc/work2/pipeline_ocr.py`: OCR assist stage for selected UI VLM pipelines.
- `poc/work2/vlm_screen_analysis.py`: Screen/state analysis built on a selected VLM service + optional OCR hints.
- `poc/work2/rcs_utils.py`: Shared capture/click/debug/window-search/JSON parsing helpers for Windows automation flows.
- `poc/work2/automate_rcs_login.py`: RCS login locator benchmark and login-click automation across multiple serving UI models.
- `poc/work2/click_rcs_view_mode.py`: View/List tab locator + click flow on the logged-in RCS main window.
- `poc/work2/check_tool_screen.py`: Tool window detection plus optional VLM UI analysis after a tool screen opens.
- `poc/work2/prompts/`: Phase 2 prompt builders (`rcs_login`, `rcs_main_tabs`, `screen_analysis`, `ocr_assist`).
- `poc/work2/logger.py`: File logger for VLM latency / status / token usage under `poc/work2/logs/`.

### Phase 1 retention policy
- Keep `poc/work/` only as legacy reference and temporary dependency source while `poc/work2` is being stabilized.
- Reuse only the core concepts from Phase 1: scenario-specific prompts, strict JSON outputs, observe/decide/act/verify loops, safe execution defaults, and screenshot-based debugging.
- If a `poc/work2` change still depends on `poc/work` utilities such as `vlm_openai_client`, `screen_capture`, `config`, or `rcs_common`, prefer migrating or wrapping that shared logic rather than extending Phase 1 scripts.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `uv sync --extra home`: Install optional home-study dependencies.
- `uv run python poc/work2/connection_check.py`: Verify Flask proxy VLM/OCR routing and live `/v1/models` connectivity.
- `uv run python poc/work2/reading_check.py`: Compare UI VLM services on a captured monitor screenshot.
- `uv run python poc/work2/automate_rcs_login.py`: Run the Phase 2 RCS login benchmark/automation flow (Windows).
- `uv run python poc/work2/click_rcs_view_mode.py`: Detect and click View/List tabs on the RCS main window (Windows).
- `uv run python poc/work2/check_tool_screen.py`: Detect a tool screen and analyze it with the Phase 2 pipeline (Windows).
- `uv run python -m poc.home.test_setup`: Validate local home-study environment.
- `uv run python -m poc.home.demo`: Run home-study VLM demo flow.
- `uv run pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.
- `uv run python -m test.vlm_input_control.integration_test`: Run input-control integration checks.
- `uv run python -m test.video_frame_parser.example_usage`: Run parser example pipeline.

## Core Coding Rules (Must Follow)
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Use uv-managed execution only: run Python/test commands via `uv run ...` (do not use plain `python` or `pip`).
- Keep docstrings/comments aligned with surrounding module language conventions (Korean docstrings are common in this repo).
- Use print-based logging prefixes (`[INFO]`, `[WARNING]`, `[ERROR]`); do not introduce the `logging` module unless a file already depends on it.
- Use absolute imports within `poc/work2/` (`from poc.work2.xxx import ...`) and keep existing absolute imports for legacy `poc/work/` modules. Do not use `sys.path` hacks or `try/except` relative-vs-bare fallbacks. Sub-package `__init__.py` files may keep relative imports.
- Use import guards for optional dependencies with `<LIB>_AVAILABLE` flags.
- Follow existing module patterns: dataclass-based configs and enums for fixed categories. Explicit `__all__` exports in package initializers are optional, not required.
- Prefer `.env` + `python-dotenv` for runtime config loading.
- For data models used with storage layers (MongoDB/FAISS flows), keep `to_dict()` / `from_dict()` serialization patterns.
- Save debug screenshots locally as **JPEG** (smaller file size). Convert images to **WebP** (quality=90) before sending to VLM APIs to reduce payload size — WebP does not hurt VLM recognition accuracy.
- Default to safe/offline behavior in automation flows (`SAFE_MODE=true`) unless explicitly required otherwise.
- Do not add new CLI-flag driven entrypoints (`argparse`) for operational scripts; prefer `.env` + in-code defaults.
- No repo-wide formatter/linter config is committed; avoid introducing style drift within edited files.

## Testing Guidelines
- Primary framework: `pytest` (see `test/video_frame_parser/tests/test_*.py`).
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path and document required env vars.

## Platform & Workflow Notes
- Development assistant environment is macOS; Windows-only RCS/pywinauto/pynput behavior must be validated by running updated code on office Windows machines.
- Treat `poc/work2/` as the primary implementation surface. New automation/pipeline work should land there by default.
- `poc/work/` is no longer the main workstream. Avoid adding new Phase 1 entrypoints, prompts, or workflow branches unless they are needed as temporary compatibility layers for `poc/work2`.
- `poc/work2/` currently still imports some legacy helpers from `poc/work/` (`vlm_openai_client`, `screen_capture`, `config`, `rcs_common`). When touching those boundaries, prefer reducing the dependency rather than growing it.
- `poc/work2/` must remain independent from server-side `flask_api` source code and its local env assumptions. Coworkers should be able to run `poc/work2` with only this repo and the hardcoded/shared client-side endpoint definitions in `poc/work2/flask_vlm.py`.
- Run Phase 2 scripts via `uv run python poc/work2/<script>.py` unless a module explicitly requires another invocation form.
- For imports across `test/` sibling modules, use `from video_frame_parser...` style when operating with `PYTHONPATH=./test`.
- Canonical office gateway host is `itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com`; if repo docs mention `webpp`, treat that as stale and keep `webapp`.

## Phase 2 Plan (`poc/work2`)
- Goal: make `poc/work2` the main experimentation and delivery surface for Flask proxy-based VLM/OCR automation while preserving only the useful design principles from Phase 1.
- Core concepts to keep from Phase 1:
- Keep prompts scenario-specific and small (`login`, `main_tabs`, `screen_analysis`, `ocr_assist`) instead of building one universal prompt.
- Enforce strict JSON outputs and validate/repair fields before using them for clicks, OCR hints, or UI-state decisions.
- Use stepwise execution: `observe` -> `decide` -> `act` -> `verify`.
- Keep debug artifacts first-class: local JPEG screenshots, WebP payloads for VLM calls, marked overlays, and call logs.
- Preserve safe/default-off behavior in Windows automation unless a script explicitly needs to act.
- Phase 2 focus areas:
- Centralize team-default pipeline settings in `poc/work2/flask_vlm.py` so coworkers can run scripts without per-user `.env` sprawl.
- Use `poc/work2/connection_check.py` to validate Flask proxy health and route/model readiness before debugging automation behavior.
- Use `poc/work2/reading_check.py` and `poc/work2/automate_rcs_login.py` to compare serving UI models on the same screen/task.
- Keep service-slug based VLM + OCR assist composition in `poc/work2/pipeline_ocr.py` and `poc/work2/vlm_screen_analysis.py`.
- Prefer direct model selection after `poc/work2/connection_check.py`: check available services, then hardcode the service slug each script wants to use.
- Consolidate reusable Windows automation helpers in `poc/work2/rcs_utils.py` and reusable prompt builders in `poc/work2/prompts/`.
- Validate the real RCS workflow on office Windows in this order: connection check -> login -> main tab interaction -> tool screen analysis.
- Migration direction:
- Do not expand Phase 1 scripts unless it is the shortest path to unblock Phase 2.
- When Phase 2 code needs legacy helpers, prefer porting the reusable helper into `poc/work2` or a future shared module instead of adding more Phase 1 business logic.
- Treat `poc/work/` as historical context plus temporary compatibility, not as the destination for new roadmap items.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (history pattern: `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`, `Fix ...`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, affected paths, test evidence (commands + results), and screenshots/log snippets for GUI automation changes.
- Link related issues/docs and note platform constraints (for example, Windows-only behavior in `poc/work2/`).
