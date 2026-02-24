# Repository Guidelines

## Project Purpose
AI-powered automation system for CD-SEM/VeritySEM recipe setup. The project combines VLM-based screen analysis with GUI automation to reduce manual recipe creation work.

## Project Structure & Module Organization
- `automation/rcs/`: Windows-focused RCS GUI automation (`run_login.py`, launcher/config/controller modules).
- `poc/work/`: Main workstream POC for VLM-driven screen analysis and input control. OpenSearch integration (`opensearch_handler.py`) is kept but inactive — to be re-enabled after company PoC approval.
- `poc/home/`: Personal/home-study variant using Hugging Face APIs (`demo.py`, `test_setup.py`).
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests in `test/video_frame_parser/tests/`.
- `test/vlm_input_control/`, `test/workflow_extractor/`: Integration-style and extractor experiments.
- `docs/`: Architecture notes, GUI automation research, and setup guides.

### `poc/work/` detailed map (current)
- `poc/work/prompts/`: Screen-specific prompt builders.
- `poc/work/prompts/rcs_login.py`: Login window element locator prompt.
- `poc/work/prompts/rcs_main_tabs.py`: Main View/List tab locator prompt.
- `poc/work/prompts/rcs_select_tool.py`: Single tool-row locator prompt.
- `poc/work/prompts/rcs_tool_list.py`: Full visible tool-list reader prompt.
- `poc/work/vlm_openai_client.py`: OpenAI-compatible VLM client wrapper (`ChatImageRequest`, langchain-compatible client).
- `poc/work/rcs_common.py`: Shared env/window/tab helpers (`load_env`, `env_flag`, `env_float`, `connect_rcs_window`, `switch_tab`).
- `poc/work/select_tool.py`: Select a tool from List tab (VLM first, optional UIA fallback).
- `poc/work/list_up_tools.py`: On List tab, locate target tool (`MCD018`) and open with double-click.
- `poc/work/check_tool_screen.py`: Detect newly opened tool window by title regex (`Remote Monitoring System - ... - [TOOL_NAME] Server[...]`).
- `poc/work/click_rcs_view_mode.py`: VLM-based View/List tab click experiment.
- `poc/work/automate_rcs_login.py`: VLM-based login benchmark/automation flow.
- `poc/work/vlm_rcs_agent.py`: Observe-Think-Act loop with step history and single-action JSON output.
- `poc/work/vlm_screen_analysis.py`: State recognition / measurement judgment / general QA analysis prompts.
- `poc/work/screen_capture.py`, `mouse_control.py`, `keyboard_control.py`: Capture/input control primitives.
- `poc/work/config.py`: Dataclass config loading from `.env`.
- `poc/work/run_rcs.py`, `switching_tabs.py`: utility launch/tab switch entrypoints.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `uv sync --extra home`: Install optional home-study dependencies.
- `pip install -r requirements.txt`: Install all-in-one pip dependencies.
- `python -m poc.work.vlm_click_demo`: Run workstream click-point visualization demo.
- `python -m poc.work.vlm_rcs_agent`: Run Observe-Think-Act RCS agent loop.
- `python -m poc.work.list_up_tools`: On current List tab, locate and open target tool (double-click).
- `python -m poc.work.check_tool_screen`: Verify that target tool screen window is opened.
- `python -m automation.rcs.run_login`: Run RCS automation login flow (Windows).
- `python poc/work/automate_rcs_login.py`: Run VLM-based RCS login automation (Windows).
- `uv run python -m poc.home.test_setup`: Validate local home-study environment.
- `uv run python -m poc.home.demo`: Run home-study VLM demo flow.
- `pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.
- `python -m test.vlm_input_control.integration_test`: Run input-control integration checks.
- `python -m test.video_frame_parser.example_usage`: Run parser example pipeline.

## Core Coding Rules (Must Follow)
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Keep docstrings/comments aligned with surrounding module language conventions (Korean docstrings are common in this repo).
- Use print-based logging prefixes (`[INFO]`, `[WARNING]`, `[ERROR]`); do not introduce the `logging` module unless a file already depends on it.
- Use absolute imports within `poc/work/` (`from poc.work.xxx import ...`); do not use `sys.path` hacks or `try/except` relative-vs-bare fallbacks. Sub-package `__init__.py` files may keep relative imports.
- Use import guards for optional dependencies with `<LIB>_AVAILABLE` flags.
- Follow existing module patterns: dataclass-based configs, enums for fixed categories, and explicit `__all__` exports in package initializers.
- Prefer `.env` + `python-dotenv` for runtime config loading.
- For data models used with storage layers (MongoDB/FAISS flows), keep `to_dict()` / `from_dict()` serialization patterns.
- Default to safe/offline behavior in automation flows (`SAFE_MODE=true`) unless explicitly required otherwise.
- Do not add new CLI-flag driven entrypoints (`argparse`) for operational scripts; prefer `.env` + in-code defaults.
- No repo-wide formatter/linter config is committed; avoid introducing style drift within edited files.

## Testing Guidelines
- Primary framework: `pytest` (see `test/video_frame_parser/tests/test_*.py`).
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path and document required env vars.

## Platform & Workflow Notes
- Development assistant environment is macOS; Windows-only RCS/pywinauto/pynput behavior must be validated by running updated code on office Windows machines.
- Keep `poc/work/` self-contained; avoid introducing cross-import coupling with `test/` prototypes. All `poc/work/` modules use absolute imports (`from poc.work.xxx`) — scripts should be run via `uv run python <script>` or `python -m poc.work.<module>`.
- For imports across `test/` sibling modules, use `from video_frame_parser...` style when operating with `PYTHONPATH=./test`.

## Dynamic Prompt + Step-by-Step Roadmap (`poc/work`)
- Goal: support dynamic prompts by UI situation and execute deterministic step-by-step automation, then connect VLM decisions to actual control tools.
- Prompt strategy:
- Keep prompts scenario-specific and small (`login`, `tab_switch`, `tool_select`, `state_judgment`) instead of one giant universal prompt.
- Route prompts by current step and screen context (for example: login detected -> use `build_rcs_login_locator_prompt`; list detected -> use `build_rcs_select_tool_prompt`).
- Enforce strict JSON schema per step and reject/repair invalid fields before action.
- Step loop strategy (recommended standard):
- `observe`: capture screenshot + gather minimal window metadata.
- `decide`: call one prompt for the current step only.
- `act`: execute exactly one action (`click`, `double_click`, `type`, `scroll`, `hotkey`, `wait`).
- `verify`: check expected post-condition (title change, target control visible, or window count/title regex match).
- `advance/retry`: move to next step when verified; otherwise retry with failure history.
- Existing modules to reuse for this roadmap:
- Prompt builders: `poc/work/prompts/*.py`
- Step/action loop base: `poc/work/vlm_rcs_agent.py`
- Window/visibility checks: `poc/work/rcs_common.py`, `poc/work/check_tool_screen.py`
- Input execution: `poc/work/mouse_control.py`, `poc/work/keyboard_control.py`
- Integration direction for later VLM tool-control:
- Keep control tools as explicit adapters with stable methods (`click_at`, `double_click`, `type_text`, `hotkey`) and call them only after prompt JSON validation.
- Add step-specific verifier functions before chaining to the next step.
- Preserve safe execution mode (`SAFE_MODE=true`) for dry-run and debugging.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (history pattern: `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`, `Fix ...`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key changes, affected paths, test evidence (commands + results), and screenshots/log snippets for GUI automation changes.
- Link related issues/docs and note platform constraints (for example, Windows-only behavior in `automation/rcs/`).
