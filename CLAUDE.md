# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen understanding and classical CV for coordinate decisions, driving GUI automation of the RCS metrology client to replace manual recipe creation.

## Active Workstreams

**`poc/workflow_3/` is the production package and current focus.** It consolidates the former workflow_1 (RCS GUI automation) and workflow_2 (CV align-key correction) into one real-time loop:

```
alarm detection (ALID=9006) → connect to tool via RCS → CV align-fail correction
→ on failure: cube rich notification to engineer → always-on screenshot recording
  (captures engineer manual operations too) → close tool → wait for next alarm
```

Subpackages — 4-layer DAG: `util` (leaf) → `{vlm, runner}` (services) → `{align, rcs, sem_monitor, recording_filter}` (capabilities) → `monitor` (orchestrator). workflow_3 never imports workflow_1/2.

- **`monitor/`** — the loop. `align_fail_monitor.py` (primary entry: polling + edge-trigger + manifest), `align_fail_monitor_only_check.py` (light "check-only" variant: connect → capture one frame → close, no correction actuation / no recording / no engineer watch), `cycle.py` (per-alarm WorkflowRunner steps + guaranteed teardown; also the check-only cycle), `recording.py` (always-on RecordingSession), `notify.py` (popup + outcome-based cube notify), `engineer_done_align_adjustment.py` (detects engineer finishing manual align via Recipe Monitor measurement counter N→ stops recording early so teardown closes the tool), `success_gather.py` (non-blocking office glue around `align.consensus_gather`), `alarm_source.py` (office module 2-stage fallback + replay CSV), `integration_loader.py` (office adapter loading logs).
- **`rcs/`** — RCS GUI automation: open/login (`login_rcs_common`, `login_rcs_ui_venus_mai`)/tool select+match (`tool_name_match`)/close/screenshot. Tool-row click is coarse→fine 2-VLM (`ui-venus` bbox → `mai-ui` point) + a **row confirm gate** (`tool_row_verify`): the two VLMs are *not* independent votes (fine only sees the crop coarse chose), so after the point is picked a **single-row strip** is cropped and OCR'd to confirm the text is the target ID. Policy via `SELECT_TOOL_ROW_CONFIRM` = `lenient` (default; reject only on reading a *different* ID) | `strict` (require confirmation) | `off`. Crop tightness needs all three of `SELECT_TOOL_ROW_VERTICAL_PAD_RATIO` (0.35) / `SELECT_TOOL_ROW_VERTICAL_PAD_MIN_PX` (10) / `SELECT_TOOL_ROW_MIN_CROP_HEIGHT` (56) — lowering only the ratio is a no-op because the two floors dominate (list rows are ~24px). A mis-click now reports `failure_class="wrong_tool_opened"` (was indistinguishable from `rcs_occupied`), closes the stray tool window, and retries after the occupied cooldown. Model choice is benchmarked by `bench_tool_locator.py` (no alarm, no clicking).
- **`align/`** — Align fail correction domain. Flat domain modules + two subpackages:
  - `matching/` — coordinate authority: `engine` (match engine, `AlignKeyTemplate`/`build_template`), `ensemble`.
  - `diagnostics/` — offline review/compare entrypoints (`compare_align_images`, `crosshair_detect`, `search_align_key`, `align_review`, `feasibility_check`, `verify_cond_box_crop`, `test_match_on_captured_frames`).
  - domain: `assets` (reads the `align_images/...` tree), `correction` (primary entry: `correct_align_fail_auto(controller, ...) -> CorrectionOutcome`), `live_search` (fallback + `SEMMonitorController` Protocol + Mac mock), `templates` (recipe align image → `AlignKeyTemplate`, cond-aware), `ok_button` (VLM OK-button locator), `search_pattern` (square-spiral pan primitive), `cond_file`/`cond_template`/`clean_align_image`/`consensus_gather` (cond + consensus helpers).
- **`sem_monitor/`** — `panel_locator.py` (SEM Monitor panel locator) + `controller.py` (real `RCSSEMMonitor` adapter skeleton — double-click recenter / wheel zoom / OK click, uncalibrated).
- **`recording_filter/`** — offline, on-demand frame-filter package (NOT in the loop hot path). Turns `RecordingSession` frames into `interaction_timeline.json` via cv2 change-detection (`frame_reduce`) + VLM cursor-based click detection (`click_detect`); `run_filter` orchestrates, `settings` = `RecordingFilterSettings`.
- **`vlm/`** — Flask VLM client/config/prompts (`flask_vlm`, `vlm_client`, `ui_venus_mai_locator`, `ocr_spotting`). **`runner/`** — WorkflowRunner/step types/settings. **`util/`** — shared helpers. Top-level: `config.py` (`Workflow3Settings`), `logger.py` (audit trail), `debug_artifacts.py` (debug-file saver, no per-save console spam).

**Extension:** `poc/workflow_3e/` adds new MES-alarm jobs *on top of* workflow_3 without editing its core (imports workflow_3 one-way). First job: **measurement-fail abort** (MES fires a consecutive-fail threshold alarm → connect + abort the running measurement). Runs via a **unified supervisor** (`poc/workflow_3e/monitor.py`) that polls MES once and dispatches align rows to workflow_3's `process_fail_rows` and abort rows to workflow_3e's `process_abort_rows` — one process, so the single RCS cursor stays serialized (no lock; abort "can queue"). Ships **notify-only** behind a double gate (`SAFE_MODE=0` **and** `MEAS_FAIL_ABORT_DRY_RUN=0` to actually click). `MEAS_FAIL_*` env namespace (not `ALIGN_FAIL_*`). See `poc/workflow_3e/README.md` + spec/plan under `poc/workflow_3/docs/superpowers/`.

**Frozen:** `poc/workflow_1/` keeps only the CCTV/DVR path + early experiments (no active work; still the `align_images` data root).

**Active offline CV bench:** `poc/workflow_2/` is *not* frozen — it is the eval / A-B / tuning harness where matching, ensemble, threshold, and consensus changes are validated against golden sets, then ported into `workflow_3/align`. It imports the engine from `poc.workflow_3.align` (never the reverse) and forks it bit-parity for experiments via `ensemble_lab.py`; golden drivers are `golden_localization_eval_cond.py` (rcp localization), `golden_consensus_eval_cond.py` (consensus A/B), and `golden_combined_eval_cond.py` (**production routed pipeline** — consensus-if-eligible else rcp, reusing both drivers; 3 axes: (A) consensus scaling by `cons_pool_n`, (B) rcp-only arm = `edge_ncc`/lab testbed, (C) routed overall; prints a one-line `[DIGEST]` + `digest.txt` to relay results without re-typing the console). **Current transition:** prove a CV change in workflow_2 → port only the verified change into workflow_3; primary build focus is workflow_3 (the real-time loop).

- **Bench config (shared, no env/CLI):** the 3 golden drivers read `poc/workflow_2/golden_eval_config.py` (gitignored edit-often scratch; copy from `golden_eval_config.example.py`). `golden_eval_config_loader.seed_env()` bridges its constants into env at each driver's top (before `gce`'s import-time `CONSENSUS_MIN_S` read); real env still wins. Constants: `GOLDEN_ROOT` (align_images eval root), `HISTORY_ROOT` (consensus pool root), `LAB_MODE` (`""`|`edge_ncc`), `MIN_S`.
- **Consensus history pool:** lives in a **separate root keyed by `<class>/<recipe>` only (eqp-independent — same recipe shares one pool across tools)**: `<HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg` (+ `.<img>/cond.txt`), the same format `office_success_downloader` writes. Production `align/assets.py` is untouched; `gce._history_images` reads this root directly. `_consensus_template_ab` is **history-first + LOO fallback**: history pool ≥ `min_s` → consensus from that disjoint pool (eval on `from_msr` S, no leakage, no LOO); else the byte-identical `from_msr` leave-one-out path. Office collects **class·recipe·modality-wise ~8–10 most-recent S (rolling, S only)**.

The filesystem contract (office MES writes, `align` reads):

```
align_images/<eqp_id>/<class>/<recipe>/
├─ align_img_from_rcp/      IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe-registered align key (office MES)
├─ align_img_from_msr/      S*/E*                             # measurement trajectory (E = fail) (office MES)
└─ captured_img_from_rcs/   <tag>/…                           # fail-time captures + recording/ (workflow_3 writes)
```

- **Runtime no longer consumes `align_img_from_msr`** (2026-06-18): correction/feasibility match consensus(preferred)/rcp(fallback) templates into the live capture, so the production loop (`align_fail_monitor`, `align_fail_monitor_only_check`) downloads **rcp only** (`gather_rcp_msr(..., include_msr=False)`). msr is offline-bench-only — fetch it on demand with `poc/workflow_3/monitor/fetch_msr_offline.py` (`include_msr=True`).

- **Production consensus cache is eqp-independent** (`ALIGN_CONSENSUS_CACHE_DIR`, distinct from the eqp-keyed `align_images` tree above): `<cache_root>/<class>/<recipe>/events/<event_id>/S*.jpeg` — **no `<eqp_id>`** (same recipe pools across tools; matches the bench `HISTORY_ROOT` keying + what `office_success_downloader` writes). `consensus_gather._events_dir_for(recipe_id, cache_root)` is the **single** path-construction point and deliberately omits `eqp_id` so it can't be re-added (re-adding splits the pool per-tool and misses the eqp-less office writer → silent permanent rcp fallback). Coupled guard: `monitor/success_gather._IN_FLIGHT` dedupe key is `recipe_id` alone (shared per-recipe staging would otherwise race across tools). The `align_images/<eqp_id>/…` tree (rcp/msr + captures/recordings) stays eqp-keyed — separate MES-contract root. Verify at office (read-only, no RCS/download): `uv run python poc/workflow_3/align/diagnostics/verify_consensus_path.py` (prints `[DIGEST]`). Fixed 2026-06-26.

- Root constant: `ALIGN_IMAGES_DIR` in `poc/workflow_3/__init__.py` (env-overridable). **Default now resolves to `poc/workflow_3/align_images`** (moved 2026-06-11; `.gitignore` tracks the new location). Office MES historically writes align keys to `poc/workflow_1/align_images`, so at the office you MUST either repoint MES output to the workflow_3 tree or set `ALIGN_IMAGES_DIR` to the MES path — otherwise the code reads an empty root and rcp/msr appear absent (captures still land because the loop writes those itself). The check-only monitor prints a path-health report at startup (`_report_data_paths`) to surface this mismatch.
- `align/assets.resolve_assets_auto()` is the single reader (override via `ALIGN_EQP_ID` / `ALIGN_CLASS_NAME` / `ALIGN_RECIPE_NAME` or kwargs).
- `office_*` modules (`office_align_fail_alarm`, `office_rich_notify`) are gitignored and exist only on the office PC; copy them into `poc/workflow_3/monitor/` (the canonical location — workflow_3 loads office adapters only from there; the old `poc.workflow_1.office_*` import fallback has been removed, so a missing adapter just disables that integration with a warning). See `poc/workflow_3/README.md` for the office migration + staged-enablement checklist.

**Authoritative docs:** `poc/workflow_3/README.md` (loop, env, office checklist). New workflow_3 loop/ops docs (specs, ADRs, journals, runbooks) live under `poc/workflow_3/docs/` (authored + git-tracked; generated artifacts go to `debug_images/`, never `docs/`). CV procedure history stays in the bench: `poc/workflow_2/docs/study/runbooks/workflow_2_procedure.md` + ADRs under `poc/workflow_2/docs/study/adr/` (paths in older docs predate the workflow_3 migration).

## Repository Structure

```
poc/workflow_3/          # PRODUCTION: real-time align-fail monitoring loop
poc/workflow_3/monitor/  #   loop entry (+ check-only variant) + per-alarm cycle + recording + notify + engineer-done + office integrations
poc/workflow_3/rcs/      #   RCS GUI automation (open/login/select/close/screenshot)
poc/workflow_3/align/    #   align correction domain root; align/matching (engine+ensemble), align/diagnostics (offline review) + smoke tests
poc/workflow_3/recording_filter/ # offline frame-filter: RecordingSession frames -> interaction timeline (change-detect + VLM cursor)
poc/workflow_3/sem_monitor/ # SEM Monitor panel_locator + real RCSSEMMonitor controller adapter
poc/workflow_3/vlm/      #   Flask VLM client, service registry, prompt builders
poc/workflow_3/runner/   #   WorkflowRunner, step/condition types, WorkflowSettings
poc/workflow_3/util/     #   env, image, json, time + optional mouse/window helpers
poc/workflow_1/          # frozen: CCTV/DVR path + early experiments + align_images data root
poc/workflow_2/          # active offline CV bench: eval/AB/tuning + ensemble_lab (bit-parity fork) + golden drivers + docs
poc/workflow_3e/         # extension: new MES-alarm jobs on workflow_3 (measurement-fail abort) + unified supervisor (monitor.py)
flask_api/vlm_serve/     # Flask VLM proxy: service registry, health discovery, per-model blueprints
deploy_vlms/             # VLM deployment configs, scripts, operational docs
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU cluster)
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
docs/                    # Architecture research notes and setup guides
```

## Setup & Dependencies

`uv` with `pyproject.toml` (Python >= 3.10). Use uv-managed workflows by default.

```bash
uv sync --extra dev                      # Core project + dev tools
uv pip install -r requirements.txt       # All-in-one
uv pip install -r test/video_frame_parser/requirements.txt  # torch, opencv, pymongo, faiss
```

## Running Modules

All scripts run with just `uv run python <script>.py` (no CLI args — see Code Conventions).

```bash
# workflow_3 — production loop (office Windows)
uv run python poc/workflow_3/monitor/align_fail_monitor.py   # Real-time align-fail monitoring loop
uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py  # Check-only variant: connect + 1 capture + close (no correction/recording)

# dev-PC dry-run (no office modules; replay one synthetic alarm through the cycle)
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py

# workflow_3 — RCS building blocks (office Windows; each runnable standalone)
uv run python poc/workflow_3/rcs/open_rcs.py                 # Start RcsMainHD.exe only
uv run python poc/workflow_3/rcs/workflow_login.py           # RCS login workflow
uv run python poc/workflow_3/rcs/view_list_tab_rcs.py        # Locate + click the List tab
uv run python poc/workflow_3/rcs/workflow_select_tool.py     # Find a tool in List tab and double-click it
uv run python poc/workflow_3/rcs/workflow_close_tool.py      # Close the opened tool window by tool id in title
uv run python poc/workflow_3/rcs/rcs_screenshot.py           # Capture tool window into captured_img_from_rcs, then close

# workflow_3 — CV engine demos (run on Mac/dev PC, synthetic data)
uv run python poc/workflow_3/align/diagnostics/compare_align_images.py  # static CV compare (falls back to synthetic self-test)
uv run python poc/workflow_3/align/correction.py                       # primary reposition+OK demo (mock, dry-run)
uv run python poc/workflow_3/align/live_search.py                      # two-phase live search demo (mock)

# legacy workflow_1 — CCTV/DVR path only
uv run python poc/workflow_1/monitor_align_fail.py           # Align-fail + open Tool DVR (CCTV) + capture CH4 frames

# Video frame parser
uv run python -m test.video_frame_parser.example_usage
```

`runner/workflow_runner.py` is a library, not an entry point: `WorkflowRunner` runs a `list[WorkflowStep]` sequentially and `ConditionChecker` evaluates step pre/post conditions; runs are journaled under `poc/workflow_3/logs/workflow_runs/`. The per-alarm cycle (`monitor/cycle.py`) is built on it; cleanup (stop recording / close tool / popup backstop) is guaranteed by `try/finally`, not steps.

## Testing

```bash
# align engine — synthetic smoke tests
uv run python poc/workflow_3/align/matching/test_engine.py
uv run python poc/workflow_3/align/test_correction.py                 # incl. error paths
uv run python poc/workflow_3/align/matching/test_engine_ensemble.py
uv run python poc/workflow_3/align/matching/test_ensemble.py
uv run python poc/workflow_3/align/diagnostics/test_match_on_captured_frames.py  # needs office capture fixtures
uv run python poc/workflow_3/rcs/test_tool_name_match.py              # 9/9
uv run python poc/workflow_3/rcs/test_tool_row_verify.py              # 42/42 (row confirm gate + crop tightness)

# tool locator VLM combo bench (office; RCS logged in, List tab visible; no alarm, no clicking)
uv run python poc/workflow_3/rcs/bench_tool_locator.py
BENCH_REPEATS=1 uv run python poc/workflow_3/rcs/bench_tool_locator.py   # smoke first (48 runs); full default = 4 combos x 12 tools x 3 = 144 runs / ~432 VLM calls

# recording_filter — offline frame-filter unit tests (pytest-style, 18 tests)
uv run pytest poc/workflow_3/recording_filter

# monitor — engineer-done + success-gather smoke tests (run directly)
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_success_gather.py

# Video frame parser unit tests
uv run pytest test/video_frame_parser/tests/

# vlm_input_control integration (safe mode by default; toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules.
- **No `__future__` imports by default**: do not add `from __future__ import annotations` (or any `__future__` import) unless explicitly asked.
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/workflow_3/logger.py` uses Python `logging` with `RotatingFileHandler` for the audit trail (`poc/workflow_3/logs/vlm_calls.log` for VLM calls, `work2.log` for general events). Avoid em-dash (U+2014) inside `print()` strings — the office console is cp949 and cannot encode it (docstrings are fine).
- **Absolute imports** within `poc/`: use `from poc.workflow_3.xxx import ...`; legacy packages import from workflow_3, never the reverse.
- **`__all__` in `__init__.py` is optional**: only add it when it provides clear value for a curated package API.
- **Image format convention**: save debug screenshots locally as **JPEG**; convert to **WebP** (quality=90) when sending to VLM APIs to cut payload size without hurting accuracy.
- **Safe mode**: interactive modules respect `SAFE_MODE` (blocks real mouse/keyboard output). `action_enabled`/`typing_enabled` default to the inverse of `SAFE_MODE` in `WorkflowSettings`. CV correction has a second gate: real reposition/OK clicks require `SAFE_MODE=0` **and** `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.
- **No CLI arguments**: do not use `argparse` or flags. Configuration comes from `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings`), `vlm/flask_vlm.py` constants, or environment variables. Scripts must run with just `uv run python <script>.py`.

## Development Workflow

Development is **mixed macOS + Windows**:

- On **macOS**, Claude Code cannot see or drive the actual RCS application. Windows-only paths (RCS, pywinauto, pynput mouse/keyboard) are edited on Mac, pushed via git, pulled at the office, and run there; debugging relies on the user reporting console output and debug screenshots in `poc/workflow_3/debug_images/` (per-model subdirs).
- On **Windows** (office machine), Claude Code runs directly and can execute the automation scripts itself.

Pure-CV and synthetic-data work in `workflow_3/align` (e.g. `diagnostics/compare_align_images.py`, `matching/test_engine.py`) and the replay-source loop dry-run run and are verified on any dev machine without RCS.

## Architecture Notes

### Flask Proxy VLM Architecture

VLM calls route through a Flask proxy at the company server, which provides unified health discovery and per-service routing.

- **Service registry (server side)**: `flask_api/vlm_serve/config.py`, one `VLMServiceEntry` dataclass per model.
- **Registered services**: ui-venus (8001), mai-ui (8002), ui-tars (8003, disabled), paddleocr-vl-1.5 (8004), got-ocr (8005).
- **Health endpoint**: `GET /api/vlm_serve/health`.
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.

### `poc/workflow_3/vlm/flask_vlm.py` — client config hub

Defines `ALL_VLM_SERVICES` (a `list[VLMServiceEntry]`) plus `DEFAULT_*` service/model constants. Two connection modes:

- **`proxy`** — Flask-routed UI/OCR models: `ui-venus-1.5-8b` (primary screen analysis & tabs), `mai-ui-8b`, `paddleocr-vl-1.5` (OCR assist), `got-ocr`.
- **`direct`** — company LLM gateway (`http://common.llm.skhynix.com/v1`): `Kimi-K2.5`, `Qwen3-VL-30B-Instruct`.

Helpers: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`. Per-model debug dirs live under `debug_images/<model-slug>/` (slug via `resolve_debug_model_name()` in `poc/workflow_3/__init__.py`).

Run/step tuning lives in `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings` in `runner/workflow_config.py`): retry budget, settle/poll timings, verify service (`paddleocr-vl-1.5`), `service_fallback_order` (`ui-venus` → `mai-ui`), plus loop fields (poll/recording/watch intervals, correction toggles, alarm source). Build it with `load_workflow3_settings()` (env overrides applied; legacy `ALIGN_FAIL_*` env names preserved).

- **Local config (`workflow_3_config.py`, edit-often scratch — distinct from `config.py`):** `config.py` is the authoritative **schema/reader** (defines `Workflow3Settings` defaults + the `ALIGN_FAIL_*`/`SAFE_MODE` env names it reads); `workflow_3_config.py` is a **gitignored convenience front-end** of plain constants (copy from `workflow_3_config.example.py`) that `workflow_3_config_loader.seed_env()` bridges into env *before* `load_workflow3_settings()` runs — so you set toggles in one file instead of a long `ALIGN_FAIL_X=… uv run …` line. One-way flow: `workflow_3_config.py` constants → `seed_env()` (`os.environ`) → `config.py` reads env. **Precedence: real shell env > `workflow_3_config.py` > `config.py` defaults** (seed is setdefault; the loader prints which config values were ignored because env already set them). It can only set vars `config.py` already reads — it never adds a setting, and deleting it just falls back to `config.py` defaults (a malformed scratch file warns + falls back, doesn't crash). `seed_env()` is called in both monitors' `__main__` (`align_fail_monitor.py`, `align_fail_monitor_only_check.py`). Same pattern as workflow_2's `golden_eval_config.py`. (`ALIGN_IMAGES_DIR` is read at package import, *before* `seed_env()`, so it must come from real env or its default — not controllable here.)

Recently added env flags (all `Workflow3Settings` fields unless noted; defaults in parens):

- **Foreground takeover** (force RCS to the front + capture/act reliably even while the engineer is using another app — Windows foreground lock + UIPI otherwise silently defeats this): `ALIGN_FAIL_BLOCK_INPUT` (0, opt-in — freezes the user's physical mouse/keyboard during the automated connect→capture→close phase so the cursor-glide into the live SEM box isn't contested; gated on `SAFE_MODE=0`; Ctrl+Alt+Del always escapes), `ALIGN_FAIL_KEEP_AWAKE` (1). The foreground-lock bypass in `util/window_utils.py:_force_foreground_via_attach` (ForegroundLockTimeout=0 + AttachThreadInput) now also **keystroke-primes** with a synthetic Alt down/up so our thread is the "last input receiver" (the OS rule for honoring `SetForegroundWindow`); the prime is skipped under `SAFE_MODE` and always releases Alt. **Run the monitor elevated (as Administrator):** non-elevated, UIPI makes both force-foreground and `BlockInput` silently no-op over the user's app — `window_utils.print_elevation_status()` (via `is_elevated()`) prints a `[WARNING]` at startup of **both** monitors if not elevated.
- **SEM-box detect + PM mode** (check-only feasibility marking — picks OM/SEM from the on-screen PM box, matches inside the live SEM box, shifts the align point back to full-window coords): `ALIGN_FAIL_SEM_BOX_DETECT` (1), `ALIGN_FAIL_SEM_BOX_SERVICE` (`ui-venus`). PM read is one VLM call by default; `ALIGN_FAIL_PM_TWO_STAGE_OCR` (0) enables locate→crop→OCR re-read via `ALIGN_FAIL_PM_OCR_SERVICE` (`paddleocr-vl-1.5`). Module-level tunables (not settings): `ALIGN_FAIL_PM_OVERRIDE_MARGIN` (0.15, in `align/diagnostics/feasibility_check.py` — fall back to the score-winning modality if it beats the PM-picked one by more than this), `SEM_BOX_PM_OM_VALUES` (`104,210`), `SEM_BOX_PM_CROP_PAD_RATIO` (0.30), `SEM_BOX_GREY_LO/HI/CHROMA_TOL` (grey-frame edge-snap), all in `sem_monitor/sem_box_detect.py`. The PM box is drawn (cyan) on the `_marked.jpg` overlay for verification; the standalone PM-crop debug image is no longer saved in the loop (the `detect_sem_box(pm_crop_debug_path=...)` hook remains for offline diagnostics/tests).
- **Occupied `select`-popup detect** (another engineer holds the tool → back off without touching the share/terminate options): `ALIGN_FAIL_OCCUPIED_POPUP_DETECT` (1), `ALIGN_FAIL_OCCUPIED_POPUP_SERVICE` (`ui-venus`), `ALIGN_FAIL_OCCUPIED_COOLDOWN_SEC` (300 — occupied tools aren't marked active; retried after this cooldown). `ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS` default lowered 10 → **3** (popup is detected early, so fewer window-search attempts are needed).
- **Zoom in/out ladder probe** (`monitor/cycle.py:_run_zoom_ladder`; check-only only; when feasibility verdict is `ambiguous`/`not_visible` — can't tell which point is the align key — sweep the live SEM box with the mouse wheel both OUT (lower mag, wide context) and IN (higher mag, pin the point), capturing each rung): `ALIGN_FAIL_ZOOM_PROBE` (1, default on), `ALIGN_FAIL_ZOOM_PROBE_STEPS` (2, OUT steps), `ALIGN_FAIL_ZOOM_PROBE_IN_STEPS` (2, IN steps), `ALIGN_FAIL_ZOOM_PROBE_SCROLL_DY` (-1, negative = wheel down = OUT/lower mag; IN flips sign), `ALIGN_FAIL_ZOOM_PROBE_SCROLLS_PER_STEP` (5; notches per rung — 1-2 notches barely change mag, so 5 makes a meaningful step; tune from per-rung `magnification` in `zoom_ladder.json`), `ALIGN_FAIL_ZOOM_PROBE_SETTLE_SEC` (0.6, post-wheel redraw), `ALIGN_FAIL_ZOOM_AIM_SETTLE_SEC` (0.15, hover settle after moving cursor into the box, before the wheel — module const in cycle.py), `ALIGN_FAIL_ZOOM_PROBE_REMATCH` (1; re-match the rcp key per rung via `mark_align_feasibility` → per-rung `_marked.jpg` + verdict/score, and flag the `best_rung` where the key is present). **Wheel target is always the detected live SEM-box center** — the cursor is moved into the box + the window activated before each scroll; if the box can't be located the ladder is skipped (no scrolling the window center). **RCS forwards mouse *motion*, not teleports**, so `util/mouse_utils` no longer jumps the cursor with a single `mouse.position=` set: `move_cursor_to_screen`/`scroll_at_screen` now *glide* (small interpolated steps) and `move_cursor_to_screen` adds a tiny *jiggle* on arrival so RCS tracks the cursor into the live SEM box and registers it there before the wheel fires (otherwise the remote cursor never lands in the box and the wheel hits the wrong panel). Cursor-motion tunables (module consts in `mouse_utils.py`): `ALIGN_FAIL_CURSOR_GLIDE_STEPS` (24), `ALIGN_FAIL_CURSOR_GLIDE_DELAY` (0.012s), `ALIGN_FAIL_CURSOR_JIGGLE_PX` (3; 0=off). Sweep order: OUT arm → return to baseline (wheel symmetry; PM-verified) → IN arm; end mag left as-is. No click/recenter (pure wheel + capture). Trigger verdicts: `_ZOOM_PROBE_VERDICTS` in `monitor/cycle.py`. Real wheel requires `SAFE_MODE=0` (else `[DRY-RUN]`, but capture still runs — a good way to confirm the path executes). Saves `<tag>_rcs_zoom_{out,in}{N}.jpg` (+ `_sembox.jpg` crop, + `_marked.jpg` when rematch) + `<tag>_zoom_ladder.json` next to the capture. **PM-dropdown fallback** (`monitor/cycle.py:_run_pm_dropdown_arms` + `sem_monitor/pm_dropdown.py`; `ALIGN_FAIL_PM_DROPDOWN`, default 1): on some tools the mouse wheel does **not** change magnification — mag is set only via the **'PM' button → dropdown**. `ALIGN_FAIL_ZOOM_METHOD` selects the method: **`auto` is the default** (try one OUT wheel notch first, fall back to the PM dropdown only if the PM reading is unchanged) — the tool whose wheel did not change mag has been **fab-out**, so wheel-first is safe again. `=pm_dropdown` skips the wheel entirely and clicks the PM button on every uncertain alarm (for tools where the wheel does not change mag and RCS translates a wheel event in the live box into a recenter click); `=wheel` forces wheel-only (no dropdown fallback). The dropdown read auto-detects the `Spotting:` bbox coordinate space (crop-pixel vs 0-1 vs 0-1000 normalized) and drops any option whose mapped point falls outside the dropdown crop (prevents stray clicks on the live image); option rows are **re-read on each re-open** so a magnification change that shifts the dropdown can't cause a stale-coordinate misclick. The dropdown is **absolute** (pick a value, no drift, no baseline return): the 'PM' button is located by the **two-stage VLM locator** (`vlm/ui_venus_mai_locator.analyze_window_target`: ui-venus coarse bbox → mai-ui refined point, 15s timeout, cached after the first open — the button is static UI; geometric "left of `pm_box_px`" is only the fallback when the VLM misses), click it, then read the dropdown that opens **right beneath the button** (`dropdown_region_below(button_xy, frame_wh)`) with one PaddleOCR `Spotting:` call → `{value, text, box}` per row, then step in **value-space** (out = next-lower value, in = next-higher) and click each target row, capturing + rematching per rung. **Search order follows the verdict**: `ambiguous` (key found but tied) → zoom **IN first** (pin the point), `not_visible` (low score, key not found) → zoom **OUT first** (widen to find) — empirically zoom-in localizes the align point better once the key is visible. The dropdown is re-opened per selection (selecting closes it); option click-points are cached from the first read (dropdown is PM-anchored, re-opens in place). Saves `<tag>_pm_dropdown_open{N}.jpg` per open — a **verification overlay** drawing the detected PM-number box (cyan), the *derived* PM-button click point (red crosshair; note the cyan box is the number, the button is to its left), and the dropdown crop region (yellow), so the office can confirm the click lands on the actual 'PM' button (calibrate via `ALIGN_FAIL_PM_BTN_GAP_RATIO` if not); `zoom_ladder.json` gains `method` (`wheel`|`pm_dropdown`) and a `pm_dropdown` section (`pm_options`, `spotting_raw`, `selections`, `aborted`). Aborts cleanly (engineer handles) if the PM box can't be found or fewer than 2 options parse. Tunables (module consts): the dropdown crop is anchored on the button point via `ALIGN_FAIL_PM_DD_LEFT_RATIO` (0.04), `ALIGN_FAIL_PM_DD_RIGHT_RATIO` (0.12), `ALIGN_FAIL_PM_DD_TOP_GAP_RATIO` (0.0), `ALIGN_FAIL_PM_DD_DOWN_RATIO` (0.45) — all fractions of the frame except DOWN of frame height; `ALIGN_FAIL_PM_DROPDOWN_OPEN_SETTLE_SEC` (0.5). Geometric-fallback consts (only when the VLM button locate fails): `ALIGN_FAIL_PM_BTN_GAP_RATIO` (0.9), `ALIGN_FAIL_PM_BTN_MIN_GAP` (14). Real clicks require `SAFE_MODE=0` (else `[DRY-RUN]`).

### `poc/workflow_3/vlm/prompts/` prompt builders

Each builder returns a `(system_message, user_message)` tuple and takes image `width`/`height` plus target params.

- `prompt_login_rcs_ui_venus.py` — coarse bbox for Server / UserID / Password / Login / Cancel / Shortcut.
- `prompt_login_rcs_mai_ui.py` — refined click point on the cropped+zoomed region (2-stage locator).
- `prompt_ocr_assist.py` — OCR text extraction.
- `prompt_recipe_monitor_counter.py` — grounds the Recipe Monitor measurement counter (N/M) for engineer-done detection.

### `poc/workflow_3/align/` — align-key engine

Design rule (confirmed 2026-05-25): **OpenCV produces quantitative scores and final coordinates; VLM only identifies regions, explains ambiguous FOVs, and assesses feasibility.** Never let a VLM answer override a low CV score or decide a repeatable stage transition.

- `matching/engine.py` — match engine (the coordinate authority). Ensemble path (`compute_align_key_score_ensemble`: C1/C2/C3 proposer RRF + NCC rerank + MIND self-similarity rerank, Youden-calibrated thresholds 0.6053/0.4727) for paused/static frames; lightweight `compute_align_key_score` for live broad-scan. `MatchPolicy` / `DEFAULT_POLICY` / `STRUCTURE_POLICY`; scale bands `DEFAULT_SCALES` (immutable) and `BROAD_SCALES` (low-mag miniature search).
- `matching/mind_rerank.py` — **modality-aware rerank** on top of the NCC selection inside `compute_align_key_score_ensemble` (ported 2026-07-20~21 from the workflow_2 registration A/B, 67 recipes/334 pts). Branches on `template.key_type` (`is_sem_template`): **OM** = sel order ⊕ MIND(self-similarity) order via RRF (`prod_mind`, d=+0.042 > NCC-only +0.009); **SEM** = ECC(cc) rank **alone**, not RRF-combined (`route_sw` 0.826 > route3 combined 0.820 — ECC dominates SEM so mixing dilutes it). Rank-only in both paths (never emits new coordinates — picks among existing candidates; sub-pixel proven moot by route_sw raw==ref); all-rejected → NCC selection unchanged. Kill switches `ALIGN_FAIL_MIND_RERANK=0` (OM), `ALIGN_FAIL_ECC_RERANK=0` (SEM). Keep constants bit-parity with `poc/workflow_2/registration_lab.py` (the bench measures against this implementation).
- `assets.py` — resolves/loads the `align_images/...` tree (see Active Workstreams).
- `templates.py` — materializes a recipe align image into an `AlignKeyTemplate` (cond-aware via `cond_template`: box-crop + decoupled `align_offset_xy`, gated by `ALIGN_FAIL_COND_BOX_CROP`).
- `ok_button.py` — VLM locator for the Align Fail dialog's OK button (screen-absolute coords; VLM identifies the button region only, never the align coordinate).
- `correction.py` — **primary correction entry** (`correct_align_fail_auto`): `key_visibility_gate` decides primary (reposition best_xy + OK click) vs fallback; `CorrectionOutcome.status` ∈ {corrected, fallback_*, escalated_no_ok, ok_detect_error, no_assets} drives the cube-notify decision in `monitor/notify.py`.
- `diagnostics/feasibility_check.py` (`mark_align_feasibility` → `FeasibilityResult`) — beyond the verdict/`[NON-DISTINCT]` banner it now draws the **2nd-best candidate** (magenta box+"2nd" from `result.candidates[1].xy`, the look-alike that drives the ambiguity) on `_marked.jpg`, and sets `reregister_recommended` (= verdict `ambiguous`, i.e. `second_ratio > reregister tau` — a chronic-ambiguous align key). `_feasibility.json` gains `second_xy`/`reregister_recommended`; `monitor/cycle.py` surfaces the recommendation to `result.notes` + a `reregister_recommended` audit-log line so the engineer sees which recipes need their align key re-registered on a more distinctive region.
- `live_search.py` — two-phase fallback search. Physical conventions: **double-click = recenter on click point, wheel = discrete FOV-centered zoom, template routing by OM/SEM mode.** Phase A broad zoom-out + spiral pan (budget 10); Phase B recenter → zoom-in → confirm. Real equipment is isolated behind the `SEMMonitorController` Protocol (Mac mock in same file; real adapter = `sem_monitor/controller.RCSSEMMonitor`).
- Office calibration **done** (2026-07-07): the former gaps — SEM panel landmarks (`poc/workflow_3/templates/sem_panel_landmarks/`), double-click/wheel↔magnification calibration, `read_mode()` real implementation, zoom/click-coordinate + engineer-done-detection tuning — are calibrated on the office PC. Still open: real-data accuracy/threshold confirmation on office data (진행 중) and the joint evaluation with field engineers (실전 테스트, 2026-07~08); see `docs/project_progress/00_executive_summary.md` §7.

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis for GPU cluster environments. MongoDB for metadata, FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` with `PYTHONPATH=./test`.
