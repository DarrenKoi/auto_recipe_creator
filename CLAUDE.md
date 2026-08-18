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

- **`monitor/`** — the loop. `align_fail_monitor.py` (primary entry: polling + edge-trigger + manifest), `align_fail_monitor_only_check.py` (light "check-only" variant: connect → capture one frame → close, no correction actuation / no recording / no engineer watch). **두 진입점은 목적으로 갈린다 — 시험 성격에 따라 골라 쓴다(2026-08-12):** production 은 `_apply_live_mode_defaults()` 가 `SAFE_MODE=0` + `ALIGN_FAIL_CORRECTION_DRY_RUN=0` 을 진입점에서 못박아(seed_env 보다 **먼저** — 오피스 `workflow_3_config.py` 사본의 `CORRECTION_DRY_RUN=1` 이 조용히 덮는 것을 막는다) env 없이 실클릭으로 뜨고, 시작 시 실운전 배너를 찍는다. 되돌리려면 셸 `SAFE_MODE=1`(setdefault 라 셸 env 가 항상 이기고, `config.py` 의 `dry_run or safe_mode` 이중 게이트도 그대로). check-only 는 **의도적으로 그 기본값을 갖지 않는다** — "안전하게 한 번만 보고 싶다"는 요구를 받는 쪽이라 같은 기본값을 복사하면 안 된다, `cycle.py` (per-alarm WorkflowRunner steps + guaranteed teardown; also the check-only cycle), `recording.py` (always-on RecordingSession), `notify.py` (popup + outcome-based cube notify), `engineer_done_align_adjustment.py` (detects engineer finishing manual align via Recipe Monitor measurement counter N→ stops recording early so teardown closes the tool), `success_gather.py` (non-blocking office glue around `align.consensus_gather`), `alarm_source.py` (office module 2-stage fallback + replay CSV), `integration_loader.py` (office adapter loading logs), `manual_record.py` + `frame_meta.py` (**alarm-free manual recording session**, see below), `share_request.py` (**점유 tool 화면 공유 요청 actuator**, see below).

- **점유 tool 화면 공유 요청** (2026-08-18, `ALIGN_FAIL_SHARE_*`): 점유 `Select` 팝업을 검출만 하고 포기하던 경로를 바꿔, "화면 공유"를 골라 `Request` 를 눌러 관전 세션을 얻고 엔지니어의 수동 align 작업을 녹화한다. `occupied_popup.py` 는 fail-**open** detector 로 그대로 두고, 클릭은 `share_request.py` 의 fail-**closed** actuator 가 한다 (오류 정책이 정반대라 파일을 나눴다). 안전은 env 게이트가 아니라 **확인 게이트** — 좌표는 VLM 이 찍고 그 자리 라벨을 OCR 로 읽어 `share`+`screen` 이 확인될 때만 클릭하며, `control`/`terminat`/`cancel` 이 읽히면 정책과 무관하게 클릭하지 않는다. 점유는 **3-상태** (`rcs/row_occupant.py`: `occupied_by_other`/`free`/`unknown`) 이며 `unknown` 은 보정을 막는 대신 outcome 을 `corrected_unverified` 로 강등해 **cube 가 반드시 나가게** 한다 — `correct_align_fail_auto` 가 open-loop 라, 먹지 않은 클릭을 `corrected` 로 보고하면 알림까지 생략되어 아무도 모르는 미보정이 남기 때문이다. 두 새 status(`view_only_observation`, `corrected_unverified`)는 `_RETRY_LATER_OUTCOME_STATUSES` 로 **`active_tools` 가 아니라 cooldown 재시도**로 가며(점유가 풀리면 실제 보정이 돌아야 한다), `share_max_attempts`(2) 상한이 cube 반복과 커서 독점을 끊는다. `row_occupant` 는 반드시 **자기 crop** 을 쓴다 — `tool_row_verify` 의 strip 을 넓히면 점유자 ID 가 `_looks_like_tool_id` 를 통과해 `unreadable`(lenient 통과)이 `mismatch`(무조건 거부)로 승격되어 정상 행의 클릭이 거부된다. 설계 `docs/superpowers/specs/2026-08-18-occupied-share-request-recording-design.md`, 적대적 검토 `docs/opencode/2026-08-18-occupied-share-request-debate.md`.
- **`rcs/`** — RCS GUI automation: open/login (`login_rcs_common`, `login_rcs_ui_venus_mai`)/tool select+match (`tool_name_match`)/close/screenshot. Tool-row click is coarse→fine 2-VLM (coarse bbox → fine point; **both stages default to `mai-ui`** since 2026-08-07) + a **row confirm gate** (`tool_row_verify`): the two VLMs are *not* independent votes (fine only sees the crop coarse chose), so after the point is picked a **single-row strip** is cropped and OCR'd to confirm the text is the target ID. Policy via `SELECT_TOOL_ROW_CONFIRM` = `lenient` (default; reject only on reading a *different* ID) | `strict` (require confirmation) | `off`. Crop tightness needs all three of `SELECT_TOOL_ROW_VERTICAL_PAD_RATIO` (0.35) / `SELECT_TOOL_ROW_VERTICAL_PAD_MIN_PX` (10) / `SELECT_TOOL_ROW_MIN_CROP_HEIGHT` (56) — lowering only the ratio is a no-op because the two floors dominate (list rows are ~24px). A mis-click now reports `failure_class="wrong_tool_opened"` (was indistinguishable from `rcs_occupied`), closes the stray tool window, and retries after the occupied cooldown. Model choice is benchmarked by `bench_tool_locator.py` (no alarm, no clicking).
- **`align/`** — Align fail correction domain. Flat domain modules + two subpackages:
  - `matching/` — coordinate authority: `engine` (match engine, `AlignKeyTemplate`/`build_template`), `ensemble`.
  - `diagnostics/` — offline review/compare entrypoints (`compare_align_images`, `crosshair_detect`, `search_align_key`, `align_review`, `feasibility_check`, `verify_cond_box_crop`, `test_match_on_captured_frames`).
  - domain: `assets` (reads the `align_images/...` tree), `correction` (primary entry: `correct_align_fail_auto(controller, ...) -> CorrectionOutcome`), `live_search` (fallback + `SEMMonitorController` Protocol + Mac mock), `templates` (recipe align image → `AlignKeyTemplate`, cond-aware), `ok_button` (VLM OK-button locator), `search_pattern` (square-spiral pan primitive), `cond_file`/`cond_template`/`clean_align_image`/`consensus_gather` (cond + consensus helpers).
- **`sem_monitor/`** — `panel_locator.py` (landmark 기반 SEM Monitor panel locator) + `controller.py` (real `RCSSEMMonitor` adapter — double-click recenter / wheel zoom / OK click). **Panel ROI 확보는 2단(2026-08-12)**: `build_rcs_sem_monitor(vlm_client=...)` 가 먼저 `detect_sem_box`(check-only 에서 오피스 검증된 live SEM box 검출)로 ROI 를 잡고, 실패 시에만 landmark 템플릿 매칭으로 폴백한다 — `templates/sem_panel_landmarks/` 는 여전히 비어 있고(캘리브레이션 없음), 이전에는 그 때문에 보정 사이클이 step 6 `panel_not_found` 에서 항상 멈췄다. 같은 검출의 `pm_mode` 가 `mode_hint` 로 주입되어 `read_mode()` 가 OM/SEM 을 화면에서 읽은 값으로 답한다(우선순위 `ALIGN_SEM_MODE_OVERRIDE` > `mode_hint` > `sem_mode_default`). 검증: `test_controller.py` (7/7, VLM/실장비 없이 Mac 실행).
- **`recording_filter/`** — offline, on-demand frame-filter package (NOT in the loop hot path). Turns `RecordingSession` frames into `interaction_timeline.json`; `run_filter` orchestrates, `settings` = `RecordingFilterSettings`. Four stages: **1** `frame_reduce` (cv2 change-detection) → **1.5** `region_gate` (**VLM-free per frame**: demotes changes confined to the live SEM box to `ambient`; live-box location detected once per *layout generation*, so cost scales with generations, not frames) → **2a** `click_detect` (VLM cursor locate + ROI change → click) → **2c** `element_label` (click-point crop → PaddleOCR, VLM fallback → *what* was clicked). Stage 1.5/2c exist for the manual-recording use case (see below) and degrade to no-ops on sidecar-less alarm recordings.
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

# workflow_3 — 엔지니어 수동 조작 녹화 (알람 불필요; office Windows, tool 창을 먼저 열어둘 것)
uv run python poc/workflow_3/monitor/manual_record.py        # 열린 Remote Monitoring 창에 붙어 녹화 (기본 600s)
RECORDING_FILTER_INPUT_DIR=<recording 경로> RECORDING_FILTER_MAX_VLM_CALLS=300 \
  uv run python poc/workflow_3/recording_filter/filter_recording.py   # 녹화 -> interaction_timeline.json
WORKFLOW_EXTRACT_INPUT_DIR=<recording_filter 경로> \
  uv run python poc/workflow_3/workflow_extract/extract_workflow.py   # timeline -> workflow.json + workflow.md

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
uv run pytest poc/workflow_3/rcs/test_row_occupant.py                 # 14 (점유 3-상태 판별)
uv run pytest poc/workflow_3/monitor/test_share_request.py            # 35 (확인 게이트/승낙 대기/클릭 경로)
uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py       # 21 (occupancy->outcome->notify->retry 배선)
uv run python poc/workflow_3/vlm/test_label_verify.py                 # 23/23 (shared point->label OCR verifier)

# tool locator VLM combo bench (office; RCS logged in, List tab visible; no alarm, no clicking)
uv run python poc/workflow_3/rcs/bench_tool_locator.py
BENCH_REPEATS=1 uv run python poc/workflow_3/rcs/bench_tool_locator.py   # smoke first (48 runs); full default = 4 combos x 12 tools x 3 = 144 runs / ~432 VLM calls

# tool WINDOW reader bench (office; a tool already open). buttons arm = no click, no mouse move.
uv run python poc/workflow_3/rcs/bench_tool_window_reader.py
BENCH_CURSOR_ARM=1 SAFE_MODE=0 uv run python poc/workflow_3/rcs/bench_tool_window_reader.py  # + cursor-tracking arm (moves mouse, never clicks)

# recording_filter — offline frame-filter unit tests (pytest-style, 71 tests: Stage 1/1.5/2a/2c + wiring)
uv run pytest poc/workflow_3/recording_filter

# workflow_extract — 그룹핑/렌더 단위 테스트 (VLM 불필요, Mac 실행 가능)
uv run pytest poc/workflow_3/workflow_extract

# monitor — engineer-done + success-gather + manual-record smoke tests (run directly)
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/monitor/test_manual_record.py                    # 47 (EQP 파싱/예산/가림 판정/teardown)

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
- **Registered services**: mai-ui (8002) + paddleocr-vl-1.5 (8004) are the only ones **enabled and served**; ui-venus (8001), ui-tars (8003), got-ocr (8005) are `enabled=False` and not started (2026-08-11). Disabled entries stay in `ALL_VLM_SERVICES` so slugs still resolve client-side, but their blueprints aren't registered and `/health` doesn't advertise them — a call to a disabled slug 404s.
- **Health endpoint**: `GET /api/vlm_serve/health`.
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.

### `poc/workflow_3/vlm/flask_vlm.py` — client config hub

Defines `ALL_VLM_SERVICES` (a `list[VLMServiceEntry]`) plus `DEFAULT_*` service/model constants. Two connection modes:

- **`proxy`** — Flask-routed UI/OCR models: `mai-ui-8b` (**primary grounding model — all VLM defaults, 2026-08-07**), `paddleocr-vl-1.5` (OCR assist). `ui-venus-1.5-8b` / `got-ocr` remain in the client's `ALL_VLM_SERVICES` (slugs resolve, URLs build) but are **no longer served** — the `*_SERVICE` rollback env vars now point at a dead port until the model is restarted server-side.
- **`direct`** — company LLM gateway (`http://common.llm.skhynix.com/v1`): `Kimi-K2.5`, `Qwen3-VL-30B-Instruct`.

Helpers: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`. Per-model debug dirs live under `debug_images/<model-slug>/` (slug via `resolve_debug_model_name()` in `poc/workflow_3/__init__.py`).

Run/step tuning lives in `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings` in `runner/workflow_config.py`): retry budget, settle/poll timings, verify service (`paddleocr-vl-1.5`), `service_fallback_order` (`mai-ui` → `ui-venus`), plus loop fields (poll/recording/watch intervals, correction toggles, alarm source). Build it with `load_workflow3_settings()` (env overrides applied; legacy `ALIGN_FAIL_*` env names preserved).

- **Local config (`workflow_3_config.py`, edit-often scratch — distinct from `config.py`):** `config.py` is the authoritative **schema/reader** (defines `Workflow3Settings` defaults + the `ALIGN_FAIL_*`/`SAFE_MODE` env names it reads); `workflow_3_config.py` is a **gitignored convenience front-end** of plain constants (copy from `workflow_3_config.example.py`) that `workflow_3_config_loader.seed_env()` bridges into env *before* `load_workflow3_settings()` runs — so you set toggles in one file instead of a long `ALIGN_FAIL_X=… uv run …` line. One-way flow: `workflow_3_config.py` constants → `seed_env()` (`os.environ`) → `config.py` reads env. **Precedence: real shell env > `workflow_3_config.py` > `config.py` defaults** (seed is setdefault; the loader prints which config values were ignored because env already set them). It can only set vars `config.py` already reads — it never adds a setting, and deleting it just falls back to `config.py` defaults (a malformed scratch file warns + falls back, doesn't crash). `seed_env()` is called in both monitors' `__main__` (`align_fail_monitor.py`, `align_fail_monitor_only_check.py`). Same pattern as workflow_2's `golden_eval_config.py`. (`ALIGN_IMAGES_DIR` is read at package import, *before* `seed_env()`, so it must come from real env or its default — not controllable here.)

**VLM 모델 통일 (2026-08-07):** every VLM default is now **`mai-ui`** — the project goal is to retire `ui-venus`. Switched in two steps: the 2-stage locator (`vlm/ui_venus_mai_locator.py` `DEFAULT_COARSE_SERVICE`/`DEFAULT_REFINE_SERVICE`, commit `64ef936`) and then every single-call service (`sem_box`/`ok_button`/`occupied_popup`/`engineer_done`/3e `abort_button`, `d0b0a8a`). Office-verified with `SAFE_MODE=0`: login / View→List tabs / select tool / screenshot / close tool, both benches (`bench_tool_locator`, `bench_tool_window_reader` acc=1.000), and a replay check-only cycle (SEM box + PM box/modality correct). Still unexercised: OK button, occupied popup, engineer-done counter, 3e abort — each needs its situation to occur. **Rollback** is per-scope, no code edit needed: `VLM_LOCATOR_COMBO="ui-venus>mai-ui"` for the locator, `ALIGN_FAIL_{SEM_BOX,OCCUPIED_POPUP,ENGINEER_DONE_VLM}_SERVICE` / `ALIGN_OK_BUTTON_VLM_SERVICE` / `MEAS_FAIL_ABORT_BUTTON_SERVICE` per service (ui-venus stays registered in `ALL_VLM_SERVICES`). Note `VLM_LOCATOR_COMBO` is read at call time and `rcs/` standalone scripts never call `seed_env()`, so for those it must come from real shell env, not `workflow_3_config.py`.

**Replay dry-run without a real alarm** (the only way to exercise in-tool VLM paths on demand): copy `poc/workflow_3/monitor/replay_fixture.example.csv`, set `EQP_ID`/`RECIPE_ID`, then `ALIGN_FAIL_ALARM_SOURCE=replay` + `ALIGN_FAIL_REPLAY_CSV=<path>`. `ALID` must be `9006`; rows are emitted on the **first poll only** (then empty, so the edge-trigger release path runs too).

**엔지니어 수동 조작 녹화 (2026-08-10, `MANUAL_RECORD_*` env namespace):** 알람과 무관한 별도 진입점. 엔지니어와 "지금부터 녹화하겠다"고 약속한 뒤 **이미 열려 있는** Remote Monitoring 창에 붙어 수동 작업을 녹화한다 — 접속(tool 더블클릭)은 하지 않는다. 목적은 모방 학습/절차 분석용 원천 데이터 확보이며, 지금 단계의 산출물은 자동화가 아니라 **"의미 있는 데이터가 나오는가"에 대한 판단 근거**다. 설계/계획: `docs/superpowers/{specs,plans}/2026-08-10-manual-recording-session*.md`.

- **런처** `monitor/manual_record.py` — 창 제목 `"Remote Monitoring System - <EQP>"` 에서 EQP 를 뽑아 `align_images/<EQP>/_manual/<tag>/recording/` 에 적재. 창이 2개 이상이면 목록만 출력하고 종료한다(`MANUAL_RECORD_EQP_ID` 로 지정; **부분 일치가 모호하면 임의 선택하지 않고 거부** — 엉뚱한 장비를 10분 녹화하느니 다시 실행하는 편이 낫다). `RecordingSession` 은 **감싸기만** 하고 동작을 바꾸지 않는다(`capture_fn` 주입점). 상한: `MANUAL_RECORD_MAX_SEC` (600, 실질 상한) / `MAX_FRAMES` (기본은 `max_sec/poll_sec x 1.25` 파생, 15000) / `MAX_DISK_MB` (4000) — 뒤 둘은 백스톱이며, **샘플링 주기에서 파생**되므로 poll 을 올려도 실질 상한보다 먼저 걸리지 않는다(고정 4000 이던 시절 0.05s 로 바꾸자 10분 세션이 ~3분에 `frame_budget` 으로 끊겼다). 예산 판정은 `RecordingSession` 이 프레임을 쓰는 자리에서 직접 한다. 그 외 `POLL_SEC` (0.05), `JPEG_QUALITY` (85; 알람 녹화는 종전대로 95), `META` (1). 정지 사유는 manifest 에 `user_interrupt`/`max_sec`/`window_gone`/`frame_budget`/`disk_budget`/`watch_error` 로 남고, **어느 경로로 끝나도 teardown 은 완료된다**.
- **사이드카** `monitor/frame_meta.py` → `frame_meta.jsonl` (프레임당 1줄: 창 rect, 전면 창 제목, 가림 여부, 로컬 커서 좌표). `capture_window` 가 창 핸들이 아니라 **창 rect 의 mss 스크린 그랩**이라 다른 앱이 위에 뜨면 그 앱이 찍히므로, 가림을 프레임 단위로 기록해 분석에서 걸러낸다. 가림 판정은 창 영역 5점에서 `WindowFromPoint` → **`GetAncestor(.., GA_ROOT)` 로 정규화 후** 우리 창인지 비교(정규화를 빼면 자식 컨트롤 HWND 가 잡혀 **전 프레임이 `full` 로 오판**된다). 커서는 `GetCursorPos` 폴링이며 **입력 후킹이 아니다 — 키 입력은 기록하지 않는다**. 기록 실패는 1회 경고 후 영구 비활성화(초당 5회 호출이라 콘솔 범람 방지).
- **Stage 2a 사이드카 커서 + Stage 2b 타이핑** (2026-08-11) — 사이드카에 커서가 있으면
  Stage 2a 가 VLM 커서 탐지를 건너뛴다(`cursor_source` 필드로 구분; 알람 녹화는 사이드카가
  없어 기존 VLM 경로 그대로). Stage 2b 는 **커서 정지 + 국소 반복 변화**로 타이핑 구간을 찾아
  구간 시작/끝 OCR 2콜로 값을 복원하고, before == after 면 캐럿 깜빡임으로 보고 버린다.
  `MANUAL_RECORD_*` 가 아니라 `RECORDING_FILTER_TYPING_*` 네임스페이스다.
- **분석 접합** — 사이드카와 프레임은 **`t_sec` 최근접**으로 조인한다(캡처 순번과 저장 seq 는 어긋난다: 변화 없는 샘플은 저장되지 않음). 조인 상한 `META_MAX_JOIN_GAP_SEC` (10.0) 를 넘으면 meta 없음으로 취급 — 사이드카가 중간에 죽어도 낡은 rect/커서에 영구히 조인되지 않는다. **화면→프레임 커서 변환은 반드시 frame/rect 배율 보정**을 거친다(오피스 125/150% 배율에서 단순 뺄셈은 어긋나 라이브 박스 좌·상단 20% 구간의 실제 조작이 `ambient` 로 버려진다; `util/window_utils.image_point_to_screen` 과 같은 규약). 사이드카가 없는 기존 알람 녹화는 게이트가 전량 통과로 degrade 한다(실패 아님).
- **타임라인 스키마** (`interaction_timeline.json`) — `element` / `element_source` (`ocr`|`vlm`|`none`) / `target_kind` (`ui_control`|`live_image`|`unknown`) / `region` / `generation` / `occlusion`. `target_kind` 는 **A 장비 → B 장비 이식 가능성** 표시다: 같은 RCS exe 라 라벨은 재탐색 가능하지만 좌표는 창 위치가 달라 무의미하고, 라이브 영상 위 조작은 CV 재해석이 필요하다. `element_source` 를 따로 두는 이유는 OCR 로 읽은 라벨과 VLM 이 서술한 라벨의 신뢰 수준이 다르기 때문(이식성 판단 시 `ocr` 만 신뢰하는 식으로 필터 가능).
- **첫 오피스 실행 시 주의** — Stage 2a 는 `max_vlm_calls` 기본 0(무제한)이라 10분 세션이 수백~수천 콜이 될 수 있다. 첫 회는 `RECORDING_FILTER_MAX_VLM_CALLS=300` 으로 상한을 걸 것(잘린 양은 `summary.json` 의 `truncated`/`skipped_due_to_cap` 에 정직하게 보고된다). 확인 포인트 3가지: manifest 의 `sampled_count/경과시간` (실측 샘플링 주기, 목표 ~20/s), `region_map_gen0.jpg` 의 시안 박스가 실제 라이브 SEM 영역과 맞는지(**틀리면 이후 게이팅 전부 무효 — 여기서 멈출 것**), `summary.json` 의 `gate_passed/total_change_events` (90%+ 제거면 정상, 0% 면 사이드카 조인 의심). 전량 폐기 시 `run_filter` 는 성공이 아닌 상태를 반환한다.

런타임 env 플래그 레퍼런스(반자동 보정 게이트, foreground takeover, SEM-box/PM mode 검출,
occupied popup, 실패경로 쿨다운, zoom ladder + PM dropdown)는 `workflow3-env-flags` 스킬에
있다 - 기본값/튜너블/롤백 스위치가 필요할 때 불러 쓴다.

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
- `correction.py` — **primary correction entry** (`correct_align_fail_auto`): `key_visibility_gate` decides primary (reposition best_xy + OK click) vs fallback; `CorrectionOutcome.status` ∈ {corrected, **awaiting_engineer_ok**, fallback_*, escalated_ambiguous_key, escalated_no_ok, ok_detect_error, no_assets} drives the cube-notify decision in `monitor/notify.py`. **반자동 모드** (`CorrectionConfig.ok_click_enabled=False`; 운영 루프 기본값): reposition 더블클릭까지만 자동으로 하고 OK 는 누르지 않은 채 `awaiting_engineer_ok` 로 끝낸다. 이 상태값이 따로 있는 이유는 `notify_correction_outcome` 이 `corrected` 면 cube 를 생략하기 때문 — `require_ok_button=False` 로 OK 만 건너뛰면 `corrected` 가 반환되어 "OK 눌러달라"는 알림이 조용히 사라진다(회귀 방지 테스트: `test_correction.py:test_awaiting_engineer_ok*`). `corrected` 가 아니므로 엔지니어 watch 도 계속 돌아 OK 를 누르는 장면까지 녹화된다.
- `diagnostics/feasibility_check.py` (`mark_align_feasibility` → `FeasibilityResult`) — beyond the verdict/`[NON-DISTINCT]` banner it now draws the **2nd-best candidate** (magenta box+"2nd" from `result.candidates[1].xy`, the look-alike that drives the ambiguity) on `_marked.jpg`, and sets `reregister_recommended` (= verdict `ambiguous`, i.e. `second_ratio > reregister tau` — a chronic-ambiguous align key). `_feasibility.json` gains `second_xy`/`reregister_recommended`; `monitor/cycle.py` surfaces the recommendation to `result.notes` + a `reregister_recommended` audit-log line so the engineer sees which recipes need their align key re-registered on a more distinctive region.
- `live_search.py` — two-phase fallback search. Physical conventions: **double-click = recenter on click point, wheel = discrete FOV-centered zoom, template routing by OM/SEM mode.** Phase A broad zoom-out + spiral pan (budget 10); Phase B recenter → zoom-in → confirm. Real equipment is isolated behind the `SEMMonitorController` Protocol (Mac mock in same file; real adapter = `sem_monitor/controller.RCSSEMMonitor`).
- Office calibration **done** (2026-07-07): the former gaps — SEM panel landmarks (`poc/workflow_3/templates/sem_panel_landmarks/`), double-click/wheel↔magnification calibration, `read_mode()` real implementation, zoom/click-coordinate + engineer-done-detection tuning — are calibrated on the office PC. Still open: real-data accuracy/threshold confirmation on office data (진행 중) and the joint evaluation with field engineers (실전 테스트, 2026-07~08); see `docs/project_progress/00_executive_summary.md` §7.

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis for GPU cluster environments. MongoDB for metadata, FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` with `PYTHONPATH=./test`.

## Agent skills

### Issue tracker

Issues are tracked as markdown files under `docs/issues/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + ADRs (root `docs/adr/` and per-workflow `poc/workflow_*/docs/study/adr/`). See `docs/agents/domain.md`.
