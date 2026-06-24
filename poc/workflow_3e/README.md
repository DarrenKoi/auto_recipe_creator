# workflow_3e — alarm-job extension for workflow_3

`workflow_3e` adds new MES-alarm-driven jobs **on top of** `workflow_3` without editing its
core. It imports `workflow_3` one-way (extension → core, never the reverse) and runs through a
single **unified supervisor** so the one RCS cursor stays serialized.

First (and so far only) extension: **measurement-fail abort** — when MES fires a
consecutive-fail threshold alarm, connect to the tool and abort the running measurement.

## Why a separate package

Each new alarm job would otherwise pile more `ALIGN_FAIL_*` flags into `workflow_3/config.py`
and more branches into `cycle.py` / `align_fail_monitor.py`. Keeping jobs in `workflow_3e`
isolates that growth. The structural constraint is unchanged: there is one `RcsMainHD.exe` and
one cursor, so **actuation is serialized** — `workflow_3e` runs in the *same process* as the
align job (one supervisor loop), which serializes the GUI for free with no lock. This is safe
because aborting "can queue" (no preemption needed; MES owns the failure-streak counting).

## Run

```bash
# Office: unified loop (align fail + measurement-fail abort), one process
uv run python poc/workflow_3e/monitor.py

# Dev PC dry-run: replay both alarm classes without RCS
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  MEAS_FAIL_ALID=<alid> uv run python poc/workflow_3e/monitor.py
```

`monitor.py` **supersedes** `align_fail_monitor.py` (it runs the align job *and* the abort
job). `align_fail_monitor.py` still runs standalone (align-only) for backward compatibility.

## Modules

| Module | Role |
|---|---|
| `monitor.py` | unified supervisor: polls MES once, dispatches align rows → `workflow_3.process_fail_rows`, abort rows → `process_abort_rows` |
| `detector.py` | `filter_measurement_fail(rows, alid)` — stateless filter on the raw alarm DataFrame (MES owns the streak) |
| `dispatch.py` | `process_abort_rows` (edge-trigger + occupied cooldown) + `append_abort_manifest` (`measurement_abort_cycles.csv`) |
| `abort_cycle.py` | `run_abort_cycle` — reuses `workflow_3`'s connect/window/capture/teardown executors; only `_exec_abort_measurement` is new |
| `abort_button.py` | VLM locator for the Stop/Abort button + confirm dialog (mirrors `align/ok_button.py`; VLM finds the region only) |
| `notify.py` | `notify_abort_outcome` — cube summary via `workflow_3`'s office adapter |
| `config.py` | `Workflow3eSettings(Workflow3Settings)` + `load_workflow3e_settings()` — adds the `MEAS_FAIL_*` fields |

## Environment (`MEAS_FAIL_*`)

| Env | Default | Meaning |
|---|---|---|
| `MEAS_FAIL_ABORT_ENABLED` | `1` | master toggle for the abort job (detect + notify) |
| `MEAS_FAIL_ALID` | `""` | the threshold-alarm ALID — **office-confirmed**; empty = job detects nothing |
| `MEAS_FAIL_ABORT_DRY_RUN` | `1` | actuation gate. Real abort click needs `SAFE_MODE=0` **and** this `=0` |
| `MEAS_FAIL_ABORT_BUTTON_SERVICE` | `ui-venus` | VLM route_slug for the Stop/Abort locator |

All `ALIGN_FAIL_*` env (poll interval, window, correction gates, occupied cooldown, …) still
apply — `Workflow3eSettings` extends `Workflow3Settings`.

## Safety: ships notify-only

Aborting a production run is destructive, so it is **double-gated** (same as CV correction) and
**defaults to notify-only**: detect → connect → capture evidence → locate the Abort button →
cube-notify the engineer, but **do not click**. Dry-run exercises the whole path (including the
VLM locate) and gates only the final click. Arm it (`SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0`)
only after verifying the located button at the office.

## Office activation checklist

1. Provide `filter_measurement_fail(rows)` in `office_align_fail_alarm` **or** set `MEAS_FAIL_ALID`
   (the supervisor filters the raw alarm feed by ALID, so `MEAS_FAIL_ALID` alone is enough).
2. Confirm the measurement-fail threshold **ALID** with MES.
3. Run notify-only first; verify the `_rcs.jpg` evidence capture + the `[DRY-RUN] Abort 버튼 검출`
   coordinate land on the real Stop/Abort button.
4. Arm with `SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0` once verified.

## Tests (dev PC, no RCS)

```bash
uv run python poc/workflow_3e/test_config.py        # 4/4
uv run python poc/workflow_3e/test_detector.py      # 4/4
uv run python poc/workflow_3e/test_abort_button.py  # 3/3
uv run python poc/workflow_3e/test_abort_cycle.py   # 2/2
uv run python poc/workflow_3e/test_dispatch.py      # 3/3
```

Spec + plan: `poc/workflow_3/docs/superpowers/specs/2026-06-25-measurement-abort-job-design.md`
(see the Revision section) and `.../plans/2026-06-25-measurement-abort-job.md`.
