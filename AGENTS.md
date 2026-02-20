# Repository Guidelines

## Project Structure & Module Organization
- `automation/rcs/`: Windows-focused RCS GUI automation (`run_login.py`, launcher/config/controller modules).
- `poc/work/`: Main workstream POC for VLM-driven screen analysis and input control. OpenSearch integration (`opensearch_handler.py`) is kept but inactive — to be re-enabled after company PoC approval.
- `poc/home/`: Personal/home-study variant using Hugging Face APIs (`demo.py`, `test_setup.py`).
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests in `test/video_frame_parser/tests/`.
- `test/vlm_input_control/`, `test/workflow_extractor/`: Integration-style and extractor experiments.
- `docs/`: Architecture notes, GUI automation research, and setup guides.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `uv sync --extra home`: Install optional home-study dependencies.
- `python -m automation.rcs.run_login --debug`: Run RCS automation in debug mode (Windows).
- `uv run python -m poc.home.test_setup`: Validate local home-study environment.
- `uv run python -m poc.home.demo --mode screen_analysis`: Run VLM demo flow.
- `pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.
- `python -m test.vlm_input_control.integration_test`: Run input-control integration checks.

## Coding Style & Naming Conventions
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Follow existing module patterns: dataclass-based configs, enums for fixed categories, and explicit `__all__` exports in package initializers.
- Keep comments/docstrings concise; match surrounding language conventions in each module.
- No repo-wide formatter/linter config is committed; avoid introducing style drift within edited files.

## Testing Guidelines
- Primary framework: `pytest` (see `test/video_frame_parser/tests/test_*.py`).
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path where possible and document required env vars.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (history pattern: `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, affected paths, test evidence (commands + results), and screenshots/log snippets for GUI automation changes.
- Link related issues/docs and note platform constraints (for example, Windows-only behavior in `automation/rcs/`).
