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

기능별 상세·경과·근거는 `poc/workflow_3/docs/features/` (아래 `features/...`) 에 있다. 아래에는 깨면 조용히 망가지는 계약만 남긴다 - 그 기능을 고치기 전에 해당 문서를 끝까지 읽을 것.

- **`monitor/`** — the loop. `align_fail_monitor.py` (primary entry: polling + edge-trigger + manifest) vs `align_fail_monitor_only_check.py` (connect → 1 frame → close; no correction / recording / engineer watch) - **시험 목적으로 골라 쓴다.** production 은 `_apply_live_mode_defaults()` 가 `SAFE_MODE=0` + `ALIGN_FAIL_CORRECTION_DRY_RUN=0` 을 seed_env **보다 먼저** 못박아(오피스 `workflow_3_config.py` 사본의 `CORRECTION_DRY_RUN=1` 이 조용히 덮는 것을 막는다) 실클릭으로 뜨고 실운전 배너를 찍는다. 되돌리기는 셸 `SAFE_MODE=1`. check-only 는 **의도적으로 그 기본값을 갖지 않는다**("안전하게 한 번만 보기" 용도 - 복사하지 말 것). 그 밖: `cycle.py` (per-alarm WorkflowRunner steps + guaranteed teardown; check-only cycle 도), `recording.py` (always-on RecordingSession), `notify.py` (popup + outcome-based cube notify), `engineer_done_align_adjustment.py` (Recipe Monitor counter N→ 로 엔지니어 수동 align 종료 감지 → 녹화 조기 종료), `success_gather.py` (`align.consensus_gather` 비차단 office glue), `alarm_source.py` (office module 2-stage fallback + replay CSV), `integration_loader.py` (office adapter loading logs).
- **알람 피드 = 이벤트 로그** (2026-09-17, `features/alarm_feed_active_check.md`): MES 피드는 해제된 알람 row 도 같은 UTC9 로 계속 돌려준다 - row 가 보인다 != 알람이 살아 있다. 세 루프(`align_fail_monitor`/`_only_check`/3e)가 `AlarmFeedCursor.take`(알람 식별 `(EQP_ID, UTC9)`, 한 번만)를 쓰고 `process_fail_rows` 는 UTC9 오래된 순으로 돈다. 그 결과 **cooldown 재시도는 일어나지 않는다**(사용자 결정: 끈 채 명시, 필요하면 피드가 아니라 자체 기억으로) - 다른 곳의 'cooldown 재시도' 는 이 기준으로 읽을 것. 첫 클릭 전 `ok_button.probe_align_dialog`(`Wait Input` 제목 + `align` 문구 둘 다)로 다이얼로그를 재확인해 `align_fail_cleared`/`align_fail_unconfirmed` 로 가른다(롤백 `ALIGN_FAIL_ACTIVE_CHECK=0`; `manual_align_correction.py` 는 항상 끔). **다이얼로그 검출 오피스 미검증** - 첫 cleared 는 `active_check_<n>/screen_no_dialog.jpg` 로 대조.
- **접속 전 List 점유 게이트** (2026-09-15 오피스 확인, `poc/workflow_3/check_tool_occupancy.py`, `features/list_occupancy_gate.md`): 더블클릭 **전에** List 행의 Connection User 를 읽어 `free` 일 때만 접속(`occupied_by_other`→`rcs_occupied`, `unknown`→`rcs_occupancy_unknown`). 계약 **"VLM 은 좌표, PaddleOCR 은 판독"**. `manual_align_correction.py`(`attach_open_tool=True`)는 거치지 않는다. 단독 점검(클릭 없음): 파일 상단 `ACTION_TARGET_TOOL_NAME` 고치고 `uv run python -m poc.workflow_3.check_tool_occupancy`.
- **점유 tool 화면 공유 요청** (2026-08-18, `share_request.py`, `ALIGN_FAIL_SHARE_*`, `features/occupied_share_request.md`): 점유 `Select` 팝업에서 "화면 공유"→`Request` 로 관전 세션을 얻어 엔지니어 작업을 녹화한다. detector(`occupied_popup.py`)는 fail-**open**, actuator 는 fail-**closed** 확인 게이트(OCR 로 `share`+`screen` 확인, `control`/`terminat`/`cancel` 이 읽히면 클릭 안 함) - 오류 정책이 반대라 파일이 갈렸다. 점유 `unknown` 은 outcome 을 `corrected_unverified` 로 강등해 **cube 가 반드시 나가게** 한다(open-loop 보정을 `corrected` 로 보고하면 알림이 생략된다). `rcs/row_occupant` 는 **자기 crop** 을 쓴다 - `tool_row_verify` strip 을 넓혀 공유하면 정상 행 클릭이 `mismatch` 로 거부된다.
- **RCS 부재 시 자동 복구** (2026-08-19, `monitor/rcs_recovery.py`, 기본 on, 롤백 `ALIGN_FAIL_RCS_RECOVERY=0`, `features/rcs_recovery.md`): ① 프로세스가 이미 있으면 재실행하지 않는다(psutil 부재의 '모름'=`None`=보류) ② 복구 로그인은 tool 에 접속하지 않는다(`resolve_login_tool_name`: `""`=접속 안 함 / `None`=env) ③ 창 없는 좀비는 모르면 `windowed`(죽이지 않음), 자동 종료는 opt-in `ALIGN_FAIL_RCS_KILL_STALE=1`. **실장비 미검증.**
- **Recovery Episode 수집** (2026-08-30, `ALIGN_FAIL_EPISODE_COLLECT=1`, **기본 off**, `features/recovery_episode.md`, 스펙 `docs/issues/align-fail-recovery-playbook/`): ALID=9006 active interval 하나 = Episode 하나. `monitor/recovery_episode.py` 가 이벤트 폴더 루트에 `recovery_episode.json` 을 첫 GUI step 전에 쓰고, attempt 폴더에 Guard(`guard_readings.py`)/Verification(`measurement_verification.py`, unknown-only stub)/`numerator_reads.jsonl` 을 남기며, 닫을 때 `poc/workflow_4/playbook` 이 Outcome 을 파생해 `[DIGEST] episode ...` 를 찍는다. 계약: Guard 는 **정확히 셋**이고 관측 실패는 `None`(unknown)이지 `False` 가 아니다 / identity 는 uuid4, tag 는 위치일 뿐 / 저장 경로는 Episode-relative 만 / attempt 폴더 분리가 tag 충돌(두 테이크 섞임) 수정이다(별도 수정 금지). off 면 산출물은 종전과 동일.
- **알람 무관 진입점** (엔지니어가 tool 창을 먼저 연다; 명령은 Running Modules):
  - `monitor/manual_record.py` + `frame_meta.py` - 수동 조작 녹화(아래 Architecture Notes, `features/manual_recording.md`).
  - `monitor/demonstration_rcs_control.py` (`DEMO_RCS_*`, `features/demonstration_rcs_control.md`) - RCS 실행→로그인→장비별 [접속→창 안 조작→닫기] 시연. 원격 입력 상수(`PRE_CLICK_SETTLE`/`CLICK_HOLD`/`ALT_SETTLE`/`SHIFT_SETTLE`)는 줄이지 않는다, 문구에 Shift 기호 금지, 메뉴 형제 이름을 `forbidden` 에 두지 말 것, OK/Print 는 어떤 흐름에서도 누르지 않는다.
  - `monitor/manual_click_button.py` + `button_registry.py` + `manual_screen_inventory.py` (`features/manual_click_button.md`) - 버튼 하나 확인 후 클릭. 새 버튼 = 레지스트리 한 줄, 클릭 배선은 시연의 `build_click_kit`/`locate_with_reveal` 재사용(포크 금지), inventory 는 관찰용(클릭 승인에 쓰지 않음). file_manager 만 오피스 확인.
  - `monitor/manual_cursor_sync_check.py` (`features/manual_cursor_sync_check.md`) - 로컬/장비 커서 drift 점검, 이동만 하고 클릭 없음. 축별 허용치 `|dx|<=50`/`|dy|<=15` 는 오피스 실측값(bbox 중심 편향), 원격 커서 지연 때문에 범위 밖 판독은 재판독한다.
  - `monitor/make_demo_video.py` / `make_demo_video_combined.py` (`DEMO_VIDEO_*`/`DEMO_COMBINED_*`, 오프라인, `features/demo_video.md`) - recording → mp4. 녹화는 tool 창 rect 라 터미널/접속 구간이 없어 로그 패널 합성과 `ALIGN_FAIL_RECORD_PRELUDE=1`(전체 화면 → `recording/prelude/`, 하위 폴더인 이유 = `recording_filter` 비재귀 glob 회피)로 보완. combined 는 `make_demo_video` 함수를 import 해 쓴다(**포크 금지** - 시간축 규약이 갈린다).
- **`rcs/`** — RCS GUI automation: open/login (`login_rcs_common`, `login_rcs_ui_venus_mai`)/tool select+match (`tool_name_match`)/close/screenshot. Tool-row click is coarse→fine 2-VLM (coarse bbox → fine point; **both stages default to `mai-ui`** since 2026-08-07) + a **row confirm gate** (`tool_row_verify`): the two VLMs are *not* independent votes (fine only sees the crop coarse chose), so after the point is picked a **single-row strip** is cropped and OCR'd to confirm the text is the target ID. Policy via `SELECT_TOOL_ROW_CONFIRM` = `lenient` (default; reject only on reading a *different* ID) | `strict` (require confirmation) | `off`. Crop tightness needs all three of `SELECT_TOOL_ROW_VERTICAL_PAD_RATIO` (0.35) / `SELECT_TOOL_ROW_VERTICAL_PAD_MIN_PX` (10) / `SELECT_TOOL_ROW_MIN_CROP_HEIGHT` (56) — lowering only the ratio is a no-op because the two floors dominate (list rows are ~24px). A mis-click now reports `failure_class="wrong_tool_opened"` (was indistinguishable from `rcs_occupied`), closes the stray tool window, and retries after the occupied cooldown. Model choice is benchmarked by `bench_tool_locator.py` (no alarm, no clicking).
- **`align/`** — Align fail correction domain. Flat domain modules + two subpackages:
  - `matching/` — coordinate authority: `engine` (match engine, `AlignKeyTemplate`/`build_template`), `ensemble`.
  - `diagnostics/` — offline review/compare entrypoints (`compare_align_images`, `crosshair_detect`, `search_align_key`, `align_review`, `feasibility_check`, `verify_cond_box_crop`, `test_match_on_captured_frames`).
  - domain: `assets` (reads the `align_images/...` tree), `correction` (primary entry: `correct_align_fail_auto(controller, ...) -> CorrectionOutcome`), `live_search` (legacy fallback + `SEMMonitorController` Protocol + Mac mock), `grid_search` (**2026-08-28 search-around 재설계, 기본 경로**: PM 드롭다운 절대 배율 zoom-out + 2R 박스 FOV 격자 sweep(collect-then-chase) + phase-correlation odometry; 배율 변경은 Protocol 이 아니라 주입 함수 `MagnificationControl` — cycle.py `_PMDropdownSelector` 가 채우고 Mac 은 mock. 등록 배율은 cond.txt `Magnification`. 판독 실패/등록 배율 없음이면 legacy 로 degrade. sweep 은 이동 중 클릭 프레임도 매기고(경계에 걸친 key) 매긴 프레임 overlay 를 `grid_frames/` 에 남긴다. OM 은 PM 판독 후 휠로 recipe 와 같은 단(104/210)에 맞춰 찾는다. OM spiral 은 SEM 과 따로 8 셀(`ALIGN_FAIL_SEARCH_OM_PAN_BUDGET`; 저배율 한 칸이 wafer 위에서 커서 줄였다, SEM 반경 30µm/예산 10 은 그대로). env `ALIGN_FAIL_SEARCH_*`, 롤백 `ALIGN_FAIL_SEARCH_MODE=legacy`. 스펙 `docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md`, 물리 `docs/study/hitachi_mag_fov_pixel_260828.md`; **오피스 실장비 미검증**), `templates` (recipe align image → `AlignKeyTemplate`, cond-aware), `ok_button` (VLM OK-button locator), `search_pattern` (square-spiral pan primitive), `cond_file`/`cond_template`/`clean_align_image`/`consensus_gather` (cond + consensus helpers).
- **`sem_monitor/`** — `panel_locator.py` (landmark 기반 SEM Monitor panel locator) + `controller.py` (real `RCSSEMMonitor` adapter — double-click recenter / wheel zoom / OK click). **Panel ROI 확보는 2단(2026-08-12)**: `build_rcs_sem_monitor(vlm_client=...)` 가 먼저 `detect_sem_box`(check-only 에서 오피스 검증된 live SEM box 검출)로 ROI 를 잡고, 실패 시에만 landmark 템플릿 매칭으로 폴백한다 — `templates/sem_panel_landmarks/` 는 여전히 비어 있고(캘리브레이션 없음), 이전에는 그 때문에 보정 사이클이 step 6 `panel_not_found` 에서 항상 멈췄다. 같은 검출의 `pm_mode` 가 `mode_hint` 로 주입되어 `read_mode()` 가 OM/SEM 을 화면에서 읽은 값으로 답한다(우선순위 `ALIGN_SEM_MODE_OVERRIDE` > `mode_hint` > `sem_mode_default`). 검증: `test_controller.py` (7/7, VLM/실장비 없이 Mac 실행).
- **`recording_filter/`** — offline, on-demand frame-filter package (NOT in the loop hot path). Turns `RecordingSession` frames into `interaction_timeline.json`; `run_filter` orchestrates, `settings` = `RecordingFilterSettings`. Four stages: **1** `frame_reduce` (cv2 change-detection) → **1.5** `region_gate` (**VLM-free per frame**: demotes changes confined to the live SEM box to `ambient`; live-box location detected once per *layout generation*, so cost scales with generations, not frames) → **2a** `click_detect` (VLM cursor locate + ROI change → click) → **2c** `element_label` (click-point crop → PaddleOCR, VLM fallback → *what* was clicked). Stage 1.5/2c exist for the manual-recording use case (see below) and degrade to no-ops on sidecar-less alarm recordings.
- **`vlm/`** — Flask VLM client/config/prompts (`flask_vlm`, `vlm_client`, `ui_venus_mai_locator`, `ocr_spotting`). **`runner/`** — WorkflowRunner/step types/settings. **`util/`** — shared helpers. Top-level: `config.py` (`Workflow3Settings`), `logger.py` (audit trail), `debug_artifacts.py` (debug-file saver, no per-save console spam).

**Extension:** `poc/workflow_3e/` adds new MES-alarm jobs *on top of* workflow_3 without editing its core (imports workflow_3 one-way). First job: **measurement-fail abort** (MES fires a consecutive-fail threshold alarm → connect + abort the running measurement). Runs via a **unified supervisor** (`poc/workflow_3e/monitor.py`) that polls MES once and dispatches align rows to workflow_3's `process_fail_rows` and abort rows to workflow_3e's `process_abort_rows` — one process, so the single RCS cursor stays serialized (no lock; abort "can queue"). Ships **notify-only** behind a double gate (`SAFE_MODE=0` **and** `MEAS_FAIL_ABORT_DRY_RUN=0` to actually click). `MEAS_FAIL_*` env namespace (not `ALIGN_FAIL_*`). See `poc/workflow_3e/README.md` + spec/plan under `poc/workflow_3/docs/superpowers/`.

**State-machine layer:** `poc/workflow_4/` (2026-08-28) is a self-contained hand-rolled FSM framework (`framework/`: graph + validate, bounded engine with failure_class→fallback routing / per-node + global retry budgets / abort polling, `RunState` JSON, mermaid + self-contained HTML live view). It never imports workflow_3 except inside `adapters/`. **What touches production today is only the read-only mirror** (`adapters/workflow3_cycle.py`, `ALIGN_FAIL_GRAPH_VIEW=1`, default off): `cycle.py` hands it the same step list it gives the runner and `context["run_dir"]`, and it renders the runner journal as `workflow_graph.html` next to the journal. The engine is demo-only — it is **not** a second production runner and `WorkflowRunner` is not to be given routing; its first real consumer is the align-correction sub-flow nested inside the `run_correction` executor (ADR `poc/workflow_4/docs/study/adr/0003-*.md`, debate `docs/opencode/2026-08-28-workflow4-engine-vs-runner-debate.md`). **`playbook/`** (2026-08-30) 은 그 옆에 새로 선 **순수 도메인 계층**이다 - workflow_3 를 import 하지 않고 plain data 만 받으며, 첫 조각인 `outcome.py` 가 Verification 우선순위(primary Measurement, unknown 일 때만 분자 fallback)와 Recovery Outcome 파생의 유일한 소유자다. Tests: `uv run pytest poc/workflow_4/`.

**Frozen:** `poc/workflow_1/` keeps only the CCTV/DVR path + early experiments (no active work; still the `align_images` data root).

**Active offline CV bench:** `poc/workflow_2/` is *not* frozen — it is the eval / A-B / tuning harness where matching, ensemble, threshold, and consensus changes are validated against golden sets, then ported into `workflow_3/align`. It imports the engine from `poc.workflow_3.align` (never the reverse) and forks it bit-parity for experiments via `ensemble_lab.py`; golden drivers are `golden_localization_eval_cond.py` (rcp localization), `golden_consensus_eval_cond.py` (consensus A/B), and `golden_combined_eval_cond.py` (**production routed pipeline** — consensus-if-eligible else rcp, reusing both drivers; 3 axes: (A) consensus scaling by `cons_pool_n`, (B) rcp-only arm = `edge_ncc`/lab testbed, (C) routed overall; prints a one-line `[DIGEST]` + `digest.txt` to relay results without re-typing the console). **Current transition:** prove a CV change in workflow_2 → port only the verified change into workflow_3; primary build focus is workflow_3 (the real-time loop).

- **Bench config (shared, no env/CLI):** the 3 golden drivers read `poc/workflow_2/golden_eval_config.py` (gitignored edit-often scratch; copy from `golden_eval_config.example.py`). `golden_eval_config_loader.seed_env()` bridges its constants into env at each driver's top (before `gce`'s import-time `CONSENSUS_MIN_S` read); real env still wins. Constants: `GOLDEN_ROOT` (align_images eval root), `HISTORY_ROOT` (consensus pool root), `LAB_MODE` (`""`|`edge_ncc`), `MIN_S`.
- **Consensus history pool:** lives in a **separate root keyed by `<class>/<recipe>` only (eqp-independent — same recipe shares one pool across tools)**: `<HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg` (+ `.<img>/cond.txt`), the same format `office_success_downloader` writes. Production `align/assets.py` is untouched; `gce._history_images` reads this root directly. `_consensus_template_ab` is **history-first + LOO fallback**: history pool ≥ `min_s` → consensus from that disjoint pool (eval on `from_msr` S, no leakage, no LOO); else the byte-identical `from_msr` leave-one-out path. Office collects **class·recipe·modality-wise ~8–10 most-recent S (rolling, S only)**.

The filesystem contract (office MES writes, `align` reads):

```
align_images/<eqp_id>/<class>/<recipe>/
├─ align_img_from_rcp/      IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe-registered align key (office MES)
├─ align_img_from_msr/      S*/E*                             # measurement trajectory (E = fail) (office MES)
└─ captured_img_from_rcs/   <tag>/…                           # LEGACY (2026-09-17 전) - 새 captures/recording 은 이벤트 폴더
```

- **알람 1건 = 이벤트 폴더 1개** (2026-09-17, `util/event_dir.py`, `features/event_folder.md`): `<EVENTS_DIR>/<eqp_id>-<tag>/[attempt_<n>/]` 에 `console.log` / 로그 사본 / `runs/` / `debug_images/` / `recording/` / `event.json` / Episode JSON 이 모인다(`EVENTS_DIR` 기본 = `ALIGN_IMAGES_DIR` 의 형제 `align_fail_events/`, override `ALIGN_FAIL_EVENTS_DIR`). **새 debug 저장 지점은 `debug_root()`/`runs_root()` 를 호출 시점에 쓸 것** - import 시점에 `DEBUG_IMAGE_DIR / "x"` 를 굳히면 이벤트 폴더를 못 따라간다. take 식별은 `cycle.take_dir_for` 와 `recovery_episode.episode_root_for` 두 곳뿐.

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

# workflow_3 — tool 창 커서 동기화 점검 (알람 불필요; office Windows, tool 창을 먼저 열어둘 것). 이동만, 클릭 없음
uv run python poc/workflow_3/monitor/manual_cursor_sync_check.py   # [DIGEST] verdict=synced|drifted|unknown

# workflow_3 — 열린 tool 창의 버튼 클릭 (기본 File Manager; 가려져 있으면 폴백으로 덮은 창 제목줄을 Alt+click). 대상은 파일 상단 상수
uv run python poc/workflow_3/monitor/manual_click_button.py

# workflow_3 — 레지스트리 기준 현재 화면 표 (클릭 없음; 등록 버튼 label_seen/not_seen + 앞 창 제목)
uv run python poc/workflow_3/monitor/manual_screen_inventory.py

# workflow_3 — live 이미지 모드 변경 (OM / OM-D / SEM; 2026-09-22 오피스 3모드 확인). 열린 tool 창의 Image 드롭다운. 대상은 파일 상단 TARGET_MODE
uv run python poc/workflow_3/monitor/manual_image_mode_change.py
SAFE_MODE=1 uv run python poc/workflow_3/monitor/manual_image_mode_change.py  # 리허설(현재 모드 판독까지)

# workflow_3 — RCS 자동 조작 시연 (알람 불필요; office Windows). 실클릭이 기본
uv run python poc/workflow_3/monitor/demonstration_rcs_control.py
DEMO_RCS_TOOL_IDS="MCD019,MCDC10" DEMO_RCS_DWELL_SEC=10 \
  uv run python poc/workflow_3/monitor/demonstration_rcs_control.py
SAFE_MODE=1 uv run python poc/workflow_3/monitor/demonstration_rcs_control.py  # 리허설(클릭 차단)

# workflow_3 — 녹화 프레임 -> 시연 mp4 (오프라인; Mac/dev PC 에서도 실행 가능)
uv run python poc/workflow_3/monitor/make_demo_video.py      # 최근 recording 폴더 자동 선택
DEMO_VIDEO_INPUT_DIR=<recording 경로> DEMO_VIDEO_SPEED=2 \
  uv run python poc/workflow_3/monitor/make_demo_video.py    # 구간/배속 지정
DEMO_VIDEO_SEGMENTS="0-30,120-260" \
  uv run python poc/workflow_3/monitor/make_demo_video.py    # 필요 없는 구간 잘라내기

# workflow_3 — 여러 회차(trial) 녹화를 자막과 함께 한 편으로 (오프라인)
# 인자는 파일 상단 상수(ROOT/INPUT_DIRS/LABELS/...)를 고쳐 쓴다. 셸 env 는 1회성 override.
uv run python poc/workflow_3/monitor/make_demo_video_combined.py

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

`runner/workflow_runner.py` is a library, not an entry point: `WorkflowRunner` runs a `list[WorkflowStep]` sequentially and `ConditionChecker` evaluates step pre/post conditions; runs are journaled under the active event folder's `runs/` (outside a cycle: `poc/workflow_3/logs/workflow_runs/`). The per-alarm cycle (`monitor/cycle.py`) is built on it; cleanup (stop recording / close tool / popup backstop) is guaranteed by `try/finally`, not steps.

## Testing

```bash
# 전체 (Mac 에서 VLM/오피스 없이 돈다 - 1329 passed / ~60s, 2026-09-26). 아래 목록은 파일별 커버리지 설명
uv run pytest poc/workflow_3 poc/workflow_4

# align engine — synthetic smoke tests
uv run python poc/workflow_3/align/matching/test_engine.py
uv run python poc/workflow_3/align/test_correction.py                 # incl. error paths
uv run python poc/workflow_3/align/matching/test_engine_ensemble.py
uv run python poc/workflow_3/align/matching/test_ensemble.py
uv run python poc/workflow_3/align/diagnostics/test_match_on_captured_frames.py  # needs office capture fixtures
uv run python poc/workflow_3/rcs/test_tool_name_match.py
uv run python poc/workflow_3/rcs/test_tool_row_verify.py              # row confirm gate + crop tightness
uv run pytest poc/workflow_3/align/test_grid_search.py                # search-around: zoom-out 단 선택/격자/odometry/추격 confirm/degrade/cycle 주입 + transit 프레임/추격 중복 제거/프레임 저장/OM 휠 단
uv run pytest poc/workflow_3/align/test_corner_key_recovery.py         # 모서리 key hint/되돌리기 + footprint 보폭 + 게이트 통일 + 즉시 추격 + 탐색 후 reposition/OK 연결
uv run pytest poc/workflow_3/align/matching/test_polarity_record.py    # 극성 반전(OM<->OM-D) best_ncc 부호 기록 + 반전 시 sel<=0.5·chamfer 상한(match 임계 도달 불가)
uv run pytest poc/workflow_3/align/test_fallback_kill_switch.py       # fallback kill switch + pan 예산 10 이 streak 에 안 잘림
uv run pytest poc/workflow_3/rcs/test_row_occupant.py                 # 점유 3-상태 판별
uv run pytest poc/workflow_3/monitor/test_share_request.py            # 확인 게이트/승낙 대기/클릭 경로
uv run pytest poc/workflow_3/monitor/test_alarm_event_queue.py        # 이벤트 로그 피드: 긴 사이클 중 알람/중복/같은 tool 재발/시작 시 과거 무시 + UTC9 순 처리 + cleared/unconfirmed 알림·watch
uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py       # occupancy->outcome->notify->retry 배선 + CORRECT_WHEN_OCCUPIED on-branch
uv run pytest poc/workflow_3/test_check_tool_occupancy.py             # 접속 전 List 점유 게이트: 행/헤더 로케이트 + PaddleOCR Control 접두 판정
uv run pytest poc/workflow_3/monitor/test_rcs_recovery.py             # RCS 재실행 중복 가드 + 복구 로그인이 tool 안 여는 계약 + 창 없는 좀비 판정
uv run pytest poc/workflow_3/monitor/test_recovery_episode.py         # Episode 식별/attempt 폴더/재시작 재개/Outcome+digest
uv run pytest poc/workflow_3/monitor/test_guard_readings.py          # Guard 3종 3상태 + attempt 기록
uv run pytest poc/workflow_3/monitor/test_measurement_verification.py # Verification record + unknown-only stub
uv run pytest poc/workflow_3/monitor/test_numerator_records.py       # 분자 per-read 기록/판정 분류
uv run pytest poc/workflow_3/monitor/test_frame_meta_recorder.py     # 알람 녹화 사이드카 + manifest additive
uv run pytest poc/workflow_3/monitor/test_cycle_timing.py            # logs/align_fail_timing.csv 소요 시간 행: 보정/알람->보정/사이클, UTC9 로컬 해석
uv run pytest poc/workflow_3/monitor/test_prelude_recording.py        # 접속 구간 화면 녹화 게이트/저장 위치/인계
uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py  # 시연 흐름 + 확인 게이트 + 클릭/대문자 입력 + Alt+click 가림 해제
uv run pytest poc/workflow_3/monitor/test_make_demo_video.py          # prelude 시간축 접합 + 편집 구간 + letterbox
uv run pytest poc/workflow_3/monitor/test_make_demo_video_combined.py # 회차 정렬/번호/시간축 리셋/공통 캔버스
uv run python poc/workflow_3/vlm/test_label_verify.py                 # shared point->label OCR verifier

# tool locator VLM combo bench (office; RCS logged in, List tab visible; no alarm, no clicking)
uv run python poc/workflow_3/rcs/bench_tool_locator.py
BENCH_REPEATS=1 uv run python poc/workflow_3/rcs/bench_tool_locator.py   # smoke first (48 runs); full default = 4 combos x 12 tools x 3 = 144 runs / ~432 VLM calls

# tool WINDOW reader bench (office; a tool already open). buttons arm = no click, no mouse move.
uv run python poc/workflow_3/rcs/bench_tool_window_reader.py
BENCH_CURSOR_ARM=1 SAFE_MODE=0 uv run python poc/workflow_3/rcs/bench_tool_window_reader.py  # + cursor-tracking arm (moves mouse, never clicks)

# recording_filter — offline frame-filter unit tests (pytest-style: Stage 1/1.5/2a/2c + wiring)
uv run pytest poc/workflow_3/recording_filter

# workflow_extract — 그룹핑/렌더 단위 테스트 (VLM 불필요, Mac 실행 가능)
uv run pytest poc/workflow_3/workflow_extract

# monitor — engineer-done + success-gather + manual-record smoke tests (run directly)
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/monitor/test_manual_record.py                    # EQP 파싱/예산/가림 판정/teardown
uv run pytest poc/workflow_3/monitor/test_button_registry.py         # 레지스트리 resolve/공통 템플릿/crop->전체 탐색/inventory/whole_word/poll
uv run pytest poc/workflow_3/monitor/test_manual_cursor_sync_check.py  # 커서 동기화 판정: 축별 허용치/다수결/안 따라옴/미검출
uv run pytest poc/workflow_3/monitor/test_manual_image_mode_change.py # image mode: OM vs OM-D 단어 판정/이미 그 모드/행 불일치 시 미클릭+목록 닫기

# Video frame parser unit tests
uv run pytest test/video_frame_parser/tests/

# vlm_input_control integration (safe mode by default; toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules. **예외: `flask_api/vlm_serve/*.py` 의
  route-template stub 들은 영어 한 줄 docstring 을 유지한다** (기존 5개 파일의 선례,
  사용자 판정 2026-09-03). 새 서비스를 추가할 때도 형제 파일과 같은 영어 한 줄로 맞출 것 -
  이 패키지만 섞이면 일관성이 더 나빠진다. 그 밖의 모든 모듈은 한국어 docstring 이다.
- **No `__future__` imports by default**: do not add `from __future__ import annotations` (or any `__future__` import) unless explicitly asked.
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/workflow_3/logger.py` uses Python `logging` with `RotatingFileHandler` for the audit trail (`poc/workflow_3/logs/vlm_calls.log` for VLM calls, `work2.log` for general events). Avoid em-dash (U+2014) inside `print()` strings — the office console is cp949 and cannot encode it (docstrings are fine).
- **Absolute imports** within `poc/`: use `from poc.workflow_3.xxx import ...`; legacy packages import from workflow_3, never the reverse.
- **`__all__` in `__init__.py` is optional**: only add it when it provides clear value for a curated package API.
- **Image format convention**: save debug screenshots locally as **JPEG**; convert to **WebP** (quality=90) when sending to VLM APIs to cut payload size without hurting accuracy.
- **Safe mode**: interactive modules respect `SAFE_MODE` (blocks real mouse/keyboard output). `action_enabled`/`typing_enabled` default to the inverse of `SAFE_MODE` in `WorkflowSettings`. CV correction has a second gate: real reposition/OK clicks require `SAFE_MODE=0` **and** `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.
- **No CLI arguments**: do not use `argparse` or flags. Configuration comes from `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings`), `vlm/flask_vlm.py` constants, or environment variables. Scripts must run with just `uv run python <script>.py`.
- **진입점 상단 상수 블록이 "인자"다** (2026-08-31): 시나리오 하나 = 진입점 `.py` 하나이므로, 그 실행의 knob 은 **실행하는 파일 맨 위**에 산다. 목적별 설정 폴더를 새로 만들지 않는다 - 폴더는 knob 개수를 안 줄이면서 공유 knob 사본을 N개로 늘리고(`_apply_live_mode_defaults` 가 오피스 사본의 `CORRECTION_DRY_RUN=1` 을 막으려고 생긴 것과 같은 사고), gitignored 사본을 Claude 없는 오피스 PC 에서 N개 관리하게 만든다. 두 가지 모양이 있다 - ① **모듈이 자기 env 를 직접 읽으면 상수를 기본값 인자로**(`_env_float("DEMO_RCS_DWELL_SEC", DWELL_SEC)`; `demonstration_rcs_control`/`manual_record`/`make_demo_video`) ② **다른 모듈이 읽으면**(`config.py` 의 `load_workflow3_settings`) `util/env_utils.seed_env_from_constants(globals(), _CONST_TO_ENV, label=...)` 로 시딩(`align_fail_monitor`/`_only_check`/`make_demo_video_combined`). 어느 쪽이든 `setdefault` 라 **셸 env > 파일 상수 > 코드 기본값**이고, 셸 env 에 밀려 무시된 상수는 반드시 콘솔에 찍힌다(사본 파일에 없는 자기고발 장치 - 이것 때문에 파일 상수가 폴더 사본보다 안전하다). `align_fail_monitor` 의 시딩 순서는 `_apply_live_mode_defaults` → 상수 블록 → `workflow_3_config.py` 이며, 상수 블록이 오피스 사본보다 앞서는 이유는 이 파일만 git 에 추적되어 리뷰를 거치기 때문이다. **`.env` 는 남긴다 - 비밀값 전용**(`ACTION_LOGIN_PASSWORD`); `rcs/` standalone 의 `load_dotenv()` 는 그 경로라 상수 블록으로 옮기지 않는다. 규칙은 "`.env`=비밀값, 상수 블록=동작". `align_fail_monitor` 는 **구현이 끝난 기능을 켜고 시작**하되(사용자 결정), 되돌릴 수 없거나 남에게 영향을 주는 넷(`BLOCK_INPUT`/`RCS_KILL_STALE`/`ACCESS_GRANT`/`CORRECT_WHEN_OCCUPIED`)은 `[위험]` 주석과 함께 off 다. `OK_CLICK` 은 2026-09-15 부터 **1**(완전 자동 제어) - List 점유 게이트가 오피스에서 확인된 뒤 반자동 계약을 끝냈다. 0 으로 내리면 `awaiting_engineer_ok` 반자동으로 돌아간다(그때만 cube 알림이 매 알람마다 나간다). 계약 검증은 `util/test_env_utils.py` (10) - 특히 상수 표의 env 이름이 실제 reader 가 읽는 이름 집합에 있는지 대조한다(오타는 조용히 아무 일도 안 하므로).

## Development Workflow

Development is **mixed macOS + Windows**:

- On **macOS**, Claude Code cannot see or drive the actual RCS application. Windows-only paths (RCS, pywinauto, pynput mouse/keyboard) are edited on Mac, pushed via git, pulled at the office, and run there; debugging relies on the user reporting console output and debug screenshots in `poc/workflow_3/debug_images/` (per-model subdirs).
- On **Windows** (office machine), Claude Code runs directly and can execute the automation scripts itself.

Pure-CV and synthetic-data work in `workflow_3/align` (e.g. `diagnostics/compare_align_images.py`, `matching/test_engine.py`) and the replay-source loop dry-run run and are verified on any dev machine without RCS.

## Architecture Notes

### Flask Proxy VLM Architecture

VLM calls route through a Flask proxy at the company server, which provides unified health discovery and per-service routing.

- **Service registry (server side)**: `flask_api/vlm_serve/config.py`, one `VLMServiceEntry` dataclass per model.
- **Registered services**: mai-ui (8002), paddleocr-vl-1.5 (8004), qwen3.8-27b (8006) are enabled and served. **mai-ui-2b was discarded 2026-09-05** — unused; its route stub, registry entry, blueprint wiring and deploy env file are gone (workflow_3's client-side bench references remain). **ui-venus / ui-tars / got-ocr are gone as of 2026-09-03** — weights deleted from the server, so their registry entries, route stubs, deploy env files and start scripts were all removed on both server and client. A call to those slugs now fails at slug resolution (`get_service_by_slug` returns `None`), not at the proxy. Reviving one needs the checkpoint re-imported first; git history is the restore path (the route stub is a 13-line `service_template` copy).
- **Health endpoint**: `GET /api/vlm_serve/health`.
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.
- **서버측 문서는 별도 저장소로 나갔다** (2026-09-04): `docs/setup_vlms/` 12편은 `../llm_serving/docs/`
  에 있다 (기동 절차, 용량 산정, knob 개념, prefill/decode·TTFT·멀티유저). **`deploy_vlms/` 와
  `flask_api/` 는 여기 그대로 남는다** - `web_main.py` 가 `flask_api` 를 import 하고 GPU 서버가 이
  체크아웃을 배포하기 때문이다. 따라서 두 사본이 갈릴 수 있다: **여기가 라이브고 `llm_serving` 은
  분리를 준비하는 자리다.** 서빙 코드를 고치면 어느 쪽인지 밝힐 것.
  참고: `test/flask_api/` 의 서버측 3파일 중 `test_vlm_serve.py` 는 2026-09-03 모델 제거 이후
  기대값이 낡아 실패한다(`ui-venus`/`ui-tars`/`got-ocr` 를 아직 기대). 고친 판이 `llm_serving/tests/`
  에 있다.

### `flask_api/model_upload/` — 모델 가중치 청크 업로드 (2026-08-22)

사내 private cloud 의 Flask 서버로 openweight 모델을 올리는 경로. code-server 웹
드래그앤드롭이 1GB 근처에서 깨지던 것을 대체한다 - 요청 하나에 파일 하나를 싣지 않고
청크로 쪼갠다. 3계층: `store.py` (HTTP 를 모르는 파일시스템/상태 계층 - 이어받기·무결성
로직이 전부 여기 산다), `routes.py` (blueprint, store 주입식), `config.py` (env + 배선).
클라이언트는 `deploy_vlms/scripts/upload_model.py` (stdlib + requests 만, 서버 코드를
import 하지 않는다). 계약 셋 — ① **committed offset 은 기록보다 `.part` 실제 크기가
우선**(상태만 남고 파일이 없으면 seek 이 0 으로 채운 구멍을 만든다) ② **청크 sha256 이
안 맞으면 offset 을 전진시키지 않는다**(소켓이 예외 없이 일찍 닫히는 짧은 바디가 조용히
구멍을 만드는 것을 막는 유일한 방어선) ③ **완료 시 전체 파일 재해싱**(청크 검증은
네트워크만 덮고 디스크/조립 손상은 못 잡는다). staging 은 반드시 목적지 루트 안쪽 -
`os.replace` 가 원자적이려면 같은 파일시스템이어야 한다. 앞단 프록시의
`client_max_body_size` 가 청크보다 작으면 Flask 에 닿기도 전에 413 이 나므로
`/health` 가 서버 상한을 알리고 클라이언트가 청크를 자동으로 줄인다. 운영 문서
`deploy_vlms/UPLOAD.md`. 테스트 50개 전부 Mac 에서 실장비/실서버 없이 돈다
(`uv run pytest flask_api/model_upload deploy_vlms/scripts`) - 마지막 3개는 로컬에
진짜 werkzeug 서버를 띄워 클라이언트<->서버 계약을 왕복 검증한다.

### `poc/workflow_3/vlm/flask_vlm.py` — client config hub

Defines `ALL_VLM_SERVICES` (a `list[VLMServiceEntry]`) plus `DEFAULT_*` service/model constants. Two connection modes:

- **`proxy`** — Flask-routed UI/OCR models: `mai-ui-8b` (**primary grounding model — all VLM defaults, 2026-08-07**), `paddleocr-vl-1.5` (OCR assist), `qwen3.8-27b` (general reasoning, not a grounding/OCR default). `ui-venus` / `ui-tars` / `got-ocr` were **removed from the client registry too** (2026-09-03) — the `*_SERVICE` rollback env vars can no longer name them.
- **`direct`** — company LLM gateway (`http://common.llm.skhynix.com/v1`): `Kimi-K2.5`, `Qwen3-VL-30B-Instruct`.

Helpers: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`. Per-model debug dirs live under `debug_images/<model-slug>/` (slug via `resolve_debug_model_name()` in `poc/workflow_3/__init__.py`).

Run/step tuning lives in `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings` in `runner/workflow_config.py`): retry budget, settle/poll timings, verify service (`paddleocr-vl-1.5`), `service_fallback_order` (`mai-ui` alone — no fallback since 2026-09-03), plus loop fields (poll/recording/watch intervals, correction toggles, alarm source). Build it with `load_workflow3_settings()` (env overrides applied; legacy `ALIGN_FAIL_*` env names preserved).

- **Local config (`workflow_3_config.py`, edit-often scratch — distinct from `config.py`):** `config.py` is the authoritative **schema/reader** (defines `Workflow3Settings` defaults + the `ALIGN_FAIL_*`/`SAFE_MODE` env names it reads); `workflow_3_config.py` is a **gitignored convenience front-end** of plain constants (copy from `workflow_3_config.example.py`) that `workflow_3_config_loader.seed_env()` bridges into env *before* `load_workflow3_settings()` runs — so you set toggles in one file instead of a long `ALIGN_FAIL_X=… uv run …` line. One-way flow: `workflow_3_config.py` constants → `seed_env()` (`os.environ`) → `config.py` reads env. **Precedence: real shell env > `workflow_3_config.py` > `config.py` defaults** (seed is setdefault; the loader prints which config values were ignored because env already set them). It can only set vars `config.py` already reads — it never adds a setting, and deleting it just falls back to `config.py` defaults (a malformed scratch file warns + falls back, doesn't crash). `seed_env()` is called in both monitors' `__main__` (`align_fail_monitor.py`, `align_fail_monitor_only_check.py`). Same pattern as workflow_2's `golden_eval_config.py`. (`ALIGN_IMAGES_DIR` is read at package import, *before* `seed_env()`, so it must come from real env or its default — not controllable here.)

**VLM 모델 통일 (2026-08-07):** every VLM default is now **`mai-ui`** — the project goal is to retire `ui-venus`. Switched in two steps: the 2-stage locator (`vlm/ui_venus_mai_locator.py` `DEFAULT_COARSE_SERVICE`/`DEFAULT_REFINE_SERVICE`, commit `64ef936`) and then every single-call service (`sem_box`/`ok_button`/`occupied_popup`/`engineer_done`/3e `abort_button`, `d0b0a8a`). Office-verified with `SAFE_MODE=0`: login / View→List tabs / select tool / screenshot / close tool, both benches (`bench_tool_locator`, `bench_tool_window_reader` acc=1.000), and a replay check-only cycle (SEM box + PM box/modality correct). Still unexercised: OK button, occupied popup, engineer-done counter, 3e abort — each needs its situation to occur. **그 롤백 경로는 2026-09-03 에 닫혔다** — ui-venus 가중치를 삭제했으므로 `VLM_LOCATOR_COMBO="ui-venus>mai-ui"` 같은 복귀는 불가능하다. per-service env (`ALIGN_FAIL_{SEM_BOX,OCCUPIED_POPUP,ENGINEER_DONE_VLM}_SERVICE` / `ALIGN_OK_BUTTON_VLM_SERVICE` / `MEAS_FAIL_ABORT_BUTTON_SERVICE`) 는 그대로 있지만 고를 수 있는 값은 살아 있는 slug 뿐이다. Note `VLM_LOCATOR_COMBO` is read at call time and `rcs/` standalone scripts never call `seed_env()`, so for those it must come from real shell env, not `workflow_3_config.py`.

**Replay dry-run without a real alarm** (the only way to exercise in-tool VLM paths on demand): copy `poc/workflow_3/monitor/replay_fixture.example.csv`, set `EQP_ID`/`RECIPE_ID`, then `ALIGN_FAIL_ALARM_SOURCE=replay` + `ALIGN_FAIL_REPLAY_CSV=<path>`. `ALID` must be `9006`; rows are emitted on the **first poll only** (then empty, so the edge-trigger release path runs too).

**엔지니어 수동 조작 녹화** (`monitor/manual_record.py` + `monitor/frame_meta.py`, `MANUAL_RECORD_*`): 알람과 무관한 진입점. **이미 열린** Remote Monitoring 창에 붙어 엔지니어 수동 작업을 `align_images/<EQP>/_manual/<tag>/recording/` 에 녹화한다(접속 안 함, `RecordingSession` 은 감싸기만). 창이 여러 개거나 EQP 부분 일치가 모호하면 임의 선택하지 않고 거부. 상한 `MAX_SEC`(600, 실질) / `MAX_FRAMES`·`MAX_DISK_MB`(백스톱, **poll 주기에서 파생**). 사이드카 `frame_meta.jsonl`(창 rect/가림/로컬 커서 - 키 입력은 기록 안 함)은 프레임과 **`t_sec` 최근접**으로 조인하고, 화면->프레임 커서 변환은 **frame/rect 배율 보정** 필수, 가림 판정은 `GetAncestor(GA_ROOT)` 정규화 필수. 사이드카 없는 알람 녹화는 게이트가 전량 통과로 degrade. 첫 오피스 필터 실행은 `RECORDING_FILTER_MAX_VLM_CALLS=300`, `region_map_gen0.jpg` 의 라이브 박스가 틀리면 거기서 멈출 것. 타임라인 스키마/Stage 2a·2b/확인 포인트는 `poc/workflow_3/docs/features/manual_recording.md`.

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
- `correction.py` — **primary correction entry** (`correct_align_fail_auto`): `key_visibility_gate` decides primary (reposition best_xy + OK click) vs fallback; `CorrectionOutcome.status` ∈ {corrected, **awaiting_engineer_ok**, fallback_*, escalated_ambiguous_key, escalated_reposition_unconverged, escalated_no_ok, ok_detect_error, no_assets} drives the cube-notify decision in `monitor/notify.py`. **Reposition 은 closed-loop** (2026-09-17): 클릭 뒤 재캡처·재매칭해 align point 가 FOV 중심 tol(`ALIGN_FAIL_REPOSITION_TOL_RATIO` 0.01 x frame 폭) 안에 올 때까지 최대 `ALIGN_FAIL_REPOSITION_REFINE_MAX`(3) 번 더 누른다 - key 가 사라지면 search-around, 잔차가 안 줄면(클릭 미반영/닮은 이웃 점프) OK 없이 `escalated_reposition_unconverged`. 0 = 종전 1회 클릭 롤백. 게이트의 `engineer_review`(비유일 후보)도 fallback 이 켜져 있으면 search-around 로 판별한다. **반자동 모드** (`CorrectionConfig.ok_click_enabled=False`; 운영 루프 기본값): reposition 더블클릭까지만 자동으로 하고 OK 는 누르지 않은 채 `awaiting_engineer_ok` 로 끝낸다. 이 상태값이 따로 있는 이유는 `notify_correction_outcome` 이 `corrected` 면 cube 를 생략하기 때문 — `require_ok_button=False` 로 OK 만 건너뛰면 `corrected` 가 반환되어 "OK 눌러달라"는 알림이 조용히 사라진다(회귀 방지 테스트: `test_correction.py:test_awaiting_engineer_ok*`). `corrected` 가 아니므로 엔지니어 watch 도 계속 돌아 OK 를 누르는 장면까지 녹화된다.
- **모서리 key 복구 + 탐색 재정비** (2026-09-19, Codex 협의): 근본 원인은 matcher 전 단계가 valid-mode (`cv2.matchTemplate`)라 **template 창이 프레임에 통째로 들어가야만 점수가 난다**는 것 - 검출 footprint 는 1 FOV 가 아니라 (프레임 - template)이다. 그래서 ① `partial_hint.py`: 가장자리에 걸친 key 조각(zero-pad DT + validity mask, 보이는 edge 만 평균)을 찾아 중심으로 recenter(최대 `ALIGN_FAIL_PARTIAL_HINT_MOVES`=2, 헛 hint 는 되돌림). **이동 단서일 뿐 accept 가 아니다** - 엔진 밖에 둔 이유는 calibrated 임계와 workflow_2 lab bit-parity ② `grid_search` 보폭 = `footprint_stride`(1-FOV 보폭은 대각선 0.2~0.8 FOV 자리를 어느 프레임에서도 온전히 못 봤다) ③ **게이트는 하나**: 탐색의 sweep 정지/confirm 은 correction 이 넘기는 `accept_fn`(= primary 의 `key_visibility_gate`) - 종전 match-only 는 primary 가 받는 adjust+distinctive key(오피스 실측 ~0.53)를 중심에 두고도 지나쳤다 ④ 탐색 매칭은 primary 와 같은 호출(원본 template, scale = base x 배율비; 사전 리샘플 금지) ⑤ zoom-out 에서 `match` 후보는 sweep 을 끝내지 않고 즉시 추격(adjust 로 쫓으면 배경 잡음이 예산을 다 쓴다; 점수순 몫 1회 예약) ⑥ 탐색 `match` -> `correct_align_fail` 1회 재진입(탐색 off, 배율비 band) -> closed-loop reposition + OK. 롤백 `ALIGN_FAIL_SEARCH_CONTINUE=0`(종전 `fallback_match`). legacy 탐색/OM PM 판독 실패는 배율을 몰라 재진입하지 않는다. 회귀 `test_corner_key_recovery.py`. **오피스 실장비 미검증**(부분 겹침 임계는 잠정치).
- `diagnostics/feasibility_check.py` (`mark_align_feasibility` → `FeasibilityResult`) — beyond the verdict/`[NON-DISTINCT]` banner it now draws the **2nd-best candidate** (magenta box+"2nd" from `result.candidates[1].xy`, the look-alike that drives the ambiguity) on `_marked.jpg`, and sets `reregister_recommended` (= verdict `ambiguous`, i.e. `second_ratio > reregister tau` — a chronic-ambiguous align key). `_feasibility.json` gains `second_xy`/`reregister_recommended`; `monitor/cycle.py` surfaces the recommendation to `result.notes` + a `reregister_recommended` audit-log line so the engineer sees which recipes need their align key re-registered on a more distinctive region.
- `live_search.py` — two-phase fallback search. Physical conventions: **double-click = recenter on click point, wheel = discrete FOV-centered zoom, template routing by OM/SEM mode.** Phase A broad zoom-out + spiral pan (budget 10); Phase B recenter → zoom-in → confirm. Real equipment is isolated behind the `SEMMonitorController` Protocol (Mac mock in same file; real adapter = `sem_monitor/controller.RCSSEMMonitor`).
- Office calibration (2026-07-07): double-click/wheel↔magnification, `read_mode()` real implementation, zoom/click-coordinate + engineer-done-detection tuning. **SEM panel landmarks 는 미캘리브레이션**(`poc/workflow_3/templates/sem_panel_landmarks/` 는 `.gitkeep` 뿐 - panel ROI 는 `detect_sem_box` 가 잡는다, `sem_monitor/` 항목 참조). Still open: real-data accuracy/threshold confirmation on office data (진행 중) and the joint evaluation with field engineers (실전 테스트, 2026-07~08); see `docs/project_progress/00_executive_summary.md` §7.

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis for GPU cluster environments. MongoDB for metadata, FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` with `PYTHONPATH=./test`.

## Agent skills

### Issue tracker

Issues are tracked as markdown files under `docs/issues/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + ADRs (root `docs/adr/` and per-workflow `poc/workflow_*/docs/study/adr/`). See `docs/agents/domain.md`.
