# Repository Guidelines

## Project Structure & Module Organization
This repository is the `poc/work/` package root used for the VLM-driven RCS proof-of-concept. Core runtime code is in the root:

- `config.py`: centralized runtime/settings helpers.
- `screen_capture.py`: screen capture and image utilities.
- `keyboard_control.py`, `mouse_control.py`: input-control helpers.
- `vlm_screen_analysis.py`: visual analysis flow.
- `vlm_rcs_agent.py`: observe-think-act orchestration entry logic.
- `vlm_click_demo.py`: runnable demo script.
- `opensearch_handler.py`: persistence/search integration layer — **currently inactive** (import-guarded; `opensearch-py` not in `requirements.txt`). Kept for re-enablement after company PoC approval. Do not delete.

Environment and dependency files live at `.env.example` and `requirements.txt`.
There is currently no dedicated assets directory in this workspace.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: start a local environment.
- `pip install -r requirements.txt`: install local runtime dependencies.
- `cp .env.example .env`: create local config before running workflows; fill provider/API keys and paths.
- `python vlm_click_demo.py`: run the available executable demo.
- `python -m <module>.py`: pattern for quick smoke checks when adding new runnable modules.

There is no project-level packaging step in this directory.

## Coding Style & Naming Conventions
- Python version target: 3.10+.
- 4-space indentation and PEP 8 naming (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`).
- Use focused modules and dataclass-based config objects where configuration grows beyond simple constants.
- Keep comments concise and close to non-obvious logic.
- No repo-enforced formatter/linter is pinned here; keep imports organized and type usage readable.

## Testing Guidelines
- This workspace currently has no committed automated tests.
- When adding tests, use `pytest` and name files/functions `test_<behavior>.py` / `test_<function>`.
- Run tests from repo root with `pytest`, and keep tests close to the code they validate.
- For code-touching changes, include at least one deterministic smoke check (e.g., import/CLI execution) in the PR notes.

## Commit & Pull Request Guidelines
- Commit history follows short, imperative subject lines, mostly with a leading verb:
  - `Add ...`
  - `Replace ...`
  - `Reorganize ...`
  - `Clarify ...`
- PRs should include:
  - purpose and behavior changes,
  - verification steps executed,
  - linked issue/task when applicable,
  - and notes on config impacts.
- For visual/input-control changes, include a short screenshot or recorded output snippet if possible.

## Security & Configuration
- Never commit API keys or credentials; keep them in `.env`.
- Prefer non-privileged test credentials in local experiments.
- Document any new required environment variable in `.env.example` and keep defaults safe.
