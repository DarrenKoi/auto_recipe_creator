# Repository Guidelines

## Project Purpose
AI-powered automation system for CD-SEM/VeritySEM recipe setup. The project combines VLM/OCR screen analysis, deterministic computer vision, and guarded GUI automation to reduce manual recipe creation and align-fail recovery work.

## Current Implementation Surfaces
- `poc/workflow_1/`: Primary RCS automation surface. Owns RCS startup, login, tool selection, align-fail monitoring, screen capture, shared VLM client helpers, debug artifacts, and workflow runner primitives.
- `poc/workflow_2/`: Primary align-key search and SEM Monitor matching surface. Owns align-fail asset loading, OpenCV/ORB/Chamfer matching, live search loops, SEM Monitor box detection probes, and current SEM capture/search work.
- `flask_api/vlm_serve/`: Server-side VLM/OCR service wrappers and Flask route handlers.
- `deploy_vlms/`: Model runtime configuration and scripts for starting, stopping, serving, and checking VLM/OCR models.
- `side_projects/document_extraction/`: Screenshot/document extraction side project, including extraction handlers and Korean RAG planning docs.
- `test/video_frame_parser/`: Video-frame parsing pipeline plus unit tests in `test/video_frame_parser/tests/`.
- `test/vlm_input_control/`, `test/workflow_extractor/`: Integration-style and extractor experiments.
- `docs/`: Architecture notes, setup guides, workflow docs, code guides, and project planning.

## Retired / Legacy Surface
- `poc/work2/` is legacy. Do not put new automation, prompts, shared helpers, or workflow code there.
- Do not use `poc/work2` as the default reference for new work. Prefer `poc/workflow_1`, `poc/workflow_2`, `flask_api/vlm_serve`, and `deploy_vlms`.
- Existing `poc/work2` files and tests may remain for compatibility or historical comparison. Touch them only when a requested compatibility fix explicitly requires it.

## Active Workflow Boundaries
- `workflow_1` is responsible for getting RCS into the right state: open RCS, find windows, log in, select tool/CH4 views, monitor align-fail alarms, download or organize align-fail assets, and capture frames.
- `workflow_2` starts from align-fail assets and SEM Monitor state: load recipe/current assets, match align keys, decide stage/FOV movement, capture live SEM frames, and prepare escalation/debug outputs.
- `workflow_2` may read asset roots defined by `workflow_1`, but do not move workflow orchestration back into `work2`.
- `flask_api/vlm_serve` and `deploy_vlms` are the source of server-side model service behavior. Client scripts should call those services through current workflow clients, not by copying service logic.

## Important Domain Decisions
- Read `CONTEXT.md` before changing align-fail, recipe, or asset-path behavior.
- Align-fail assets use an event folder model under `align_images/<eqp_id>/<class_name>/<recipe_name>/`.
- Registered recipe align-key images live under `align_img_from_rcp/`.
- Measurement/current-fail images live under `align_img_from_msr/` or are captured by workflow code, depending on the flow.
- Current SEM images are not downloaded from MES. `workflow_2` captures SEM Monitor live when it needs current SEM evidence. See `poc/workflow_2/docs/study/adr/0001-current-sem-is-live-captured-not-downloaded.md`.
- Prefer deterministic CV for align-key matching decisions. VLM probes can provide hints or feasibility checks, but clicks, movement, and match decisions need strict validation before use.

## Build, Test, and Development Commands
- `uv sync --extra dev`: Install core project and dev dependencies from `pyproject.toml`.
- `uv run pytest test/video_frame_parser/tests/`: Run unit tests for the video parser module.
- `uv run pytest test/flask_api/`: Run Flask VLM service tests.
- `uv run python poc/workflow_2/test_align_key_match.py`: Run synthetic align-key matcher smoke tests.
- `uv run python poc/workflow_2/search_align_key.py`: Run the workflow_2 search-loop smoke path.
- `uv run python poc/workflow_2/test_match_on_captured_frames.py`: Run captured-frame matcher checks when fixtures/artifacts are present.
- `uv run python poc/workflow_1/open_rcs.py`: Open RCS on Windows.
- `uv run python poc/workflow_1/workflow_login.py`: Run the guarded login workflow on Windows.
- `uv run python poc/workflow_1/workflow_select_tool.py`: Run guarded tool-selection workflow on Windows.
- `uv run python deploy_vlms/scripts/check_vlm.py`: Check configured VLM/OCR model service readiness.
- `uv run python side_projects/document_extraction/extract.py`: Run the document extraction side-project entrypoint.

## Core Coding Rules
- Target Python `>=3.10`; use 4-space indentation and PEP 8 naming (`snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants).
- Do not add `from __future__ import annotations` or other `__future__` imports unless the user explicitly asks for them.
- Use uv-managed execution only: run Python/test commands via `uv run ...` rather than plain `python` or `pip`.
- Keep docstrings/comments aligned with surrounding module language conventions. Korean docstrings and docs are common in workflow modules.
- Use print-based logging prefixes (`[INFO]`, `[WARNING]`, `[ERROR]`) unless editing a module that already uses another logger.
- Use absolute imports across package boundaries, for example `from poc.workflow_1.xxx import ...`, `from poc.workflow_2.xxx import ...`, and `from flask_api.vlm_serve.xxx import ...`.
- Do not add `sys.path` hacks or `try/except` relative-vs-bare import fallbacks. Sub-package `__init__.py` files may keep relative imports.
- Use import guards for optional dependencies with `<LIB>_AVAILABLE` flags.
- Follow existing module patterns: dataclass-based configs, enums for fixed categories, and explicit serialization helpers for stored data models.
- Prefer `.env` plus `python-dotenv` for runtime config loading.
- Save debug screenshots locally as JPEG when possible. Convert images to WebP with quality 90 before VLM API calls to reduce payload size.
- Default to safe/offline behavior in automation flows (`SAFE_MODE=true`) unless the user explicitly requests live action.
- Do not add new CLI-flag driven operational scripts with `argparse`; prefer `.env` plus in-code defaults.
- No repo-wide formatter/linter config is committed. Avoid broad formatting churn.

## Workflow 1 Guidelines
- Keep workflow steps small and inspectable: `observe -> decide -> act -> verify`.
- Use `workflow_types.py` and `workflow_runner.py` for reusable step definitions and execution behavior.
- Keep RCS window detection, coordinate targeting, and click/type actions guarded and easy to dry-run.
- Store debug artifacts under workflow-specific debug/log directories, not ad-hoc root folders.
- When changing login/tool-selection behavior, update or check `docs/codes/workflow_1/`.

## Workflow 2 Guidelines
- Keep align-key matching centered on `align_key_matcher.py`, `align_fail_assets.py`, `search_align_key.py`, and `live_align_search.py`.
- Keep asset layout changes synchronized with `CONTEXT.md`, the workflow_2 ADRs, and handoff docs.
- Treat `align_img_from_rcp` as recipe/reference input and live/current SEM capture as workflow output.
- Enforce bounded search behavior: explicit pan/zoom budgets, ROI constraints, safe gates, and escalation paths.
- Keep VLM output strict JSON where VLM is used, then validate or repair fields before using them for boxes, clicks, OCR hints, or UI-state decisions.
- Put durable workflow_2 design notes under `poc/workflow_2/docs/` or `docs/workflow_2/` depending on whether the doc is implementation-local or broader educational material.

## Testing Guidelines
- Primary framework: `pytest`.
- Name tests as `test_<behavior>.py` and keep fixtures close to the tested module.
- For hardware/API-dependent flows, provide a safe/offline path and document required env vars.
- Prefer focused tests for the touched surface before broad test runs.
- For GUI automation changes, include screenshots, debug-image paths, or log snippets in the verification notes when available.

## Platform & Workflow Notes
- RCS, pywinauto, pynput, and live screen automation are Windows-only and should be validated on office Windows machines.
- Non-GUI logic, matcher logic, Flask route tests, and document extraction code should remain runnable from the normal uv environment.
- For imports across `test/` sibling modules, use `from video_frame_parser...` style when operating with `PYTHONPATH=./test`.
- Canonical office gateway host is `itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com`; if repo docs mention `webpp`, treat that as stale and keep `webapp`.

## Documentation Guidelines
- Keep teammate-facing workflow docs Korean-first when the surrounding document is Korean. Preserve technical identifiers, file paths, commands, APIs, model names, and established English terms.
- Update docs together with code when changing workflow contracts, asset paths, model service assumptions, or safety behavior.
- Use `docs/codes/workflow_1/` for workflow_1 code guides.
- Use `docs/workflow_2/` for broader workflow_2 explanations and `poc/workflow_2/docs/` for implementation-local runbooks, ADRs, handoffs, and generated status artifacts.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects such as `Add ...`, `Reorganize ...`, `Replace ...`, `Clarify ...`, or `Fix ...`.
- Keep commits scoped to one logical change.
- PRs should include purpose, key changes, affected paths, test evidence, and screenshots or log snippets for GUI automation changes.
- Note platform constraints for Windows-only behavior in `poc/workflow_1/` and live SEM Monitor behavior in `poc/workflow_2/`.
