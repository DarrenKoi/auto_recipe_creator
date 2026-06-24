# Add a measurement-abort job to the production loop

**Date:** 2026-06-25
**Status:** Approved (design)
**Scope:** `poc/workflow_3/` production loop (`monitor/align_fail_monitor.py`, `monitor/cycle.py`, `monitor/alarm_source.py`, `config.py`)

## Problem

The production loop today handles exactly one kind of MES alarm: **align fail** (ALID=9006). It polls the CD-SEM alarm feed, edge-triggers on new `EQP_ID`s, and runs a per-alarm cycle that connects to the tool and attempts CV correction.

A second failure class is now in scope. When a recipe is *measuring* (alignment succeeded) but points keep **failing measurement** — for any reason (align drift, wrong recipe, wafer variation) — the run produces low-credibility data. The fab wants the run **aborted** once failures continue past a threshold (e.g. ~20 consecutive fails of a 100-point recipe), rather than letting it burn the whole wafer.

The question this design answers: **should measurement-fail handling be a separate process, or share the existing loop?**

## Key constraint that decides the structure

The system drives **one** RCS client through **one** OS cursor (pywinauto + pynput). The align cycle (`run_alarm_cycle`) physically monopolizes that cursor from connect → correct → engineer-watch → close. Therefore:

- **Actuation must be serialized.** Two jobs can never drive the GUI at the same instant. This is a hardware/UI fact, not a code-structure choice.
- **Detection is cheap and already unified.** The align trigger is one MES query (`source.poll()` → filter ALID). A measurement-fail trigger is the *same query, a different ALID*.

Confirmed with the user:
- **Abort is an RCS GUI action** (locate + click a Stop/Abort control, confirm dialog) — it contends with the align cycle for the cursor.
- **Abort can queue** — it is acceptable for an abort to wait for an in-flight align cycle to finish. Consecutive-fail is a slow-building condition; minutes of latency are tolerable. No preemption required.
- **MES fires a threshold alarm** when the consecutive-fail count is reached — the system does **not** accumulate per-measurement results itself. The detector is therefore **stateless**, an exact mirror of the align-fail edge-trigger.

## Decision: one process, one loop, two job types sharing one serialized GUI

Separate OS processes are rejected: both processes would fight for the same single cursor (forcing an inter-process lock anyway) and would lose the shared in-memory dedup state (`active_tools`). With "abort can queue," the existing **blocking** cycle model *is* the serialization — no threads, no locks, no job queue are needed.

```
monitor_loop()                              # single while loop (existing)
├─ alarms = source.poll()                   # one MES query per tick
├─ filter_align_fail(alarms)  → align rows → run_alarm_cycle   (existing, blocking)
└─ filter_measurement_fail(..) → abort rows → run_abort_cycle  (new, blocking)
```

Whichever job is detected first in a tick acts first; the other waits for the current cycle's tail or the next tick. The new job reuses the align cycle's RCS-readiness, connect, window-wait, occupied-popup handling, and teardown verbatim. The only genuinely new automation surface is **one UI control** (the Stop/Abort button + its confirm dialog).

### Staged enablement (safety)

Auto-aborting a production measurement run is **destructive and outward-facing** — far higher stakes than nudging a cursor. It ships behind a **double gate** identical to CV correction (`SAFE_MODE=0` **and** an explicit dry-run flag = 0), and **defaults to notify-only**: detect the alarm → connect → capture evidence → locate the Abort button → **cube-notify the engineer**, but do **not** click until validated at the office. Dry-run exercises the entire path (including the VLM locate) and only gates the final click, mirroring how correction was validated.

## Changes

### 1. Detector — `monitor/alarm_source.py` (git-tracked)
- Generalize `AlarmSource` to carry a second, **optional** filter: add `filter_measurement_fail(rows)` backed by `_meas_filter_fn`. When the office module does not provide a measurement filter, the method returns an empty result and the abort job self-disables (graceful degradation, same philosophy as missing office adapters).
- Office contract: `office_align_fail_alarm` (gitignored, office-only) gains a `filter_measurement_fail(rows)` function alongside `get_cdsem_alarms` / `filter_align_fail`. `load_office_module` declares it as an **optional** attr (job disabled with a warning if absent — does not break the align path).
- Replay: add `_replay_filter_measurement_fail(rows)` keying on `MEAS_FAIL_ALID` so dev-PC dry-runs can exercise the new path from a fixture CSV.
- `load_alarm_source` wires both filters into `AlarmSource` for office and replay.

### 2. Config — `config.py` (git-tracked), new `MEAS_FAIL_*` namespace
New `Workflow3Settings` fields (env-overridable via `load_workflow3_settings`):
- `meas_fail_abort_enabled: bool = True` — master toggle for the whole job (detect + notify). Env `MEAS_FAIL_ABORT_ENABLED`.
- `meas_fail_alid: str = ""` — the threshold-alarm ALID. **Office-confirmed unknown**; empty default disables the replay filter with a clear warning. Env `MEAS_FAIL_ALID`.
- `abort_action_dry_run: bool = True` — second actuation gate. The real Abort click fires only when `SAFE_MODE=0` **and** this is `0`. Env `MEAS_FAIL_ABORT_DRY_RUN` (forced `True` whenever `SAFE_MODE=1`, mirroring `correction_dry_run`).
- `abort_button_vlm_service: str = "ui-venus"` — route_slug for the button locator (not a model name). Env `MEAS_FAIL_ABORT_BUTTON_SERVICE`.
- Reused (no new fields): `rich_notify_enabled` (cube), `occupied_retry_cooldown_sec` (occupied cooldown), `connect_*` / `rcs_window_max_trials` (connect+window), `alert_close_timeout_sec` (teardown).

### 3. Abort-button locator — new `monitor/abort_button.py` (git-tracked)
- Mirrors `align/ok_button.py` exactly in shape (system+user prompt, `_frame_to_webp_b64`, strict-JSON parse via `bbox_to_pixels` → `bbox_center`). **VLM identifies the button region only** — it never decides *whether* to abort (MES already did). Design rule preserved.
- `locate_abort_button(*, frame_bgr, client) -> tuple[int, int] | None` — finds the Stop / Abort / 중지 / 정지 measurement-control button; explicitly must NOT return Pause/Cancel-dialog/close-window buttons.
- `locate_abort_confirm(*, frame_bgr, client) -> tuple[int, int] | None` — finds the Yes/확인/OK on the "abort this run?" confirmation dialog. (May delegate to `align.ok_button.locate_ok_button`, which already targets commit/OK/확인 buttons; kept as a thin named wrapper so the abort path reads clearly and can diverge if the confirm dialog differs.)

### 4. Abort cycle — `monitor/cycle.py` (git-tracked)
- `build_abort_steps(eqp_id)` — reuses `ensure_rcs_ready`, `close_alert_popup`, `connect_tool`, `wait_tool_window` step defs, then `capture_before` (reuses `_exec_capture_screen` for evidence) and a new `abort_measurement` step.
- `_exec_abort_measurement(step, context, settings)` — the only new executor:
  1. captures the live tool window;
  2. `locate_abort_button(...)` → screen coords (DPI-corrected via the existing `image_point_to_screen`/`click_at_screen` path);
  3. if `SAFE_MODE=0` **and** `abort_action_dry_run=0`: `click_at_screen(...)`, settle, capture the confirm dialog, `locate_abort_confirm(...)`, click to confirm; else log `[DRY-RUN]` with the located coords and capture, no click.
  4. records a structured outcome (`aborted` / `abort_dry_run` / `abort_button_not_found` / `abort_error`) onto the cycle result.
- `run_abort_cycle(eqp_id, recipe_id, settings, *, tag=None) -> CycleResult` — sibling of `run_alarm_cycle` on the same `WorkflowRunner` + guaranteed-`finally` teardown (close tool, close alert backstop). Sends a cube notification (`notify_abort_outcome`) summarizing the abort result. **No recording session and no engineer-watch** (the abort is the action; nothing to watch for).

### 5. Notify — `monitor/notify.py` (git-tracked)
- `notify_abort_outcome(eqp_id, recipe_id, outcome, *, capture_path="", enabled=True)` — cube rich notification mirroring `notify_correction_outcome`: summarizes the consecutive-fail abort (status, captured evidence path) so the engineer is informed whether the run was auto-aborted or needs manual intervention. In notify-only (dry-run) mode this is the primary output.

### 6. Loop integration — `monitor/align_fail_monitor.py` (git-tracked)
- In `monitor_loop`, reuse the already-fetched `alarms`: compute `meas_fails = source.filter_measurement_fail(alarms)`, window-filter it with the existing `filter_rows_within_window`, and dispatch via a new `process_abort_rows(meas_fails, aborted_tools, settings, abort_cooldown)`.
- `process_abort_rows` mirrors `process_fail_rows`' edge-trigger/dedup (`aborted_tools` set + occupied cooldown) but is thinner: no popup/gather/correction — it calls `run_abort_cycle` and appends an abort manifest row (`measurement_abort_cycles.csv`, reusing `CycleResult` columns).
- Startup banner notes whether the abort job is on and whether it is notify-only vs armed.

### 7. Docs — `poc/workflow_3/README.md`, `CLAUDE.md`
- README: document the second job in the loop description + the `MEAS_FAIL_*` env table + the office checklist item (provide `filter_measurement_fail` + confirm `MEAS_FAIL_ALID` + calibrate the Abort button; arm only after dry-run validation).
- CLAUDE.md: note the loop now handles two MES alarm classes (align fail + measurement-fail abort) sharing one serialized GUI, and the abort double-gate.

## Out of scope / unchanged
- The align-fail path (`run_alarm_cycle`, correction, feasibility, consensus, recording, engineer-watch) — untouched. The abort job is additive.
- Per-measurement result tracking / streak counting — **MES owns this** (threshold alarm). If MES later cannot provide a threshold alarm, a stateful `MeasurementFailTracker` would be a follow-up spec; explicitly not built now.
- Preemption / priority between jobs — not built ("abort can queue").
- The exact Abort-button label/location and confirm-dialog flow — office calibration, like SEM-panel landmarks; the VLM locator absorbs label variation, but real-tool verification is required before arming.

## Risk & mitigation
- **`MEAS_FAIL_ALID` unknown / office filter absent.** Job self-disables with a warning; align path unaffected. The ALID + office `filter_measurement_fail` are the one office-side input needed to activate.
- **Wrong button clicked (aborts wrong thing / clicks live image).** Double-gated to dry-run by default; dry-run logs and captures the located coords for office verification before any click is enabled. Locator prompt explicitly excludes Pause/Cancel/close. Same coarse→fine→confirm discipline as other locators.
- **Abort fires on a transient.** MES owns the threshold (N consecutive); the system only actuates what MES already decided. No independent abort judgment in CV/VLM.
- **Both alarms for one tool.** Independent dedup sets; near-mutually-exclusive in practice (align fail ⇒ not measuring). If both fire, the two cycles run serially — acceptable.
- **Connecting to an actively-measuring tool.** Connect opens the Remote Monitoring view (non-actuating); only the gated click is intrusive. Same connect path already validated by the align job.

## Acceptance
- Dev PC: `MEAS_FAIL_ALARM` replay fixture flows through `filter_measurement_fail` → `process_abort_rows` → `run_abort_cycle` in dry-run, producing a captured frame, a `[DRY-RUN]` located-button log, an abort manifest row, and a cube-notify call — without any click. Verified by self-tests + a replay dry-run.
- The align-fail path is byte-for-byte unchanged in behavior (existing self-tests still pass; align call sites untouched).
- Office: with `filter_measurement_fail` provided and `MEAS_FAIL_ALID` set, the job detects the alarm and notifies; arming the click is a separate, explicit `SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0` step after dry-run verification.
