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

## Office contract — how you feed in the alarm

You implement the **detection** side (count consecutive fails, decide the threshold); `workflow_3e`
does the **action** side. There are two ways to deliver the alarm; pick one:

**Path A (recommended) — dedicated provider.** Copy `temp_office_meas_many_fails.py` →
`office_meas_many_fails.py` (gitignored at the office) and implement **one function**:

```python
def get_measurement_fail_alarms() -> pandas.DataFrame:
    """지금 '연속 N회 측정 실패' 임계를 넘긴 tool 들의 표준 알람 rows (없으면 빈 DataFrame)."""
```

- **Input:** none — it queries MES itself. Called every poll (`ALIGN_FAIL_POLL_SEC`).
- **Output:** a DataFrame in the **same schema as align-fail alarms** (`ALARM_COLUMNS`). Build each
  row with the provided `build_measurement_fail_alarm_row(...)` helper so the format is guaranteed.
- **Stateless-safe:** keep returning the same `EQP_ID` row while the tool is over threshold —
  `workflow_3e` edge-triggers (acts once) and re-arms when the row disappears.

Row schema (REQUIRED = consumed; rest = logged/notify):

| Column | Req | Meaning |
|---|---|---|
| `EQP_ID` | ✅ | tool id — connect target + dedup key |
| `ALID` | ✅ | alarm id (`MEAS_FAIL_ALID`, distinct from align's 9006) |
| `UTC9` | ✅ | `"%Y-%m-%d %H:%M:%S"` detection time — recency window + capture folder tag |
| `RECIPE_ID` |  | `"<class>/<recipe>"` — capture/asset path (else `_unregistered`) |
| `ALARM_NAME` |  | human label — **put the fail count here** (e.g. `"...Fail (20/100)"`); it reaches the engineer cube |
| `OPERATION_DESC`, `LOT_TYPE_CD`, `TIMESTAMP` |  | extra context |

**Path B — ride the shared feed.** If the consecutive-fail alarm is already a native MES alarm in
`get_cdsem_alarms()` with its own ALID, skip the provider and just set `MEAS_FAIL_ALID=<that ALID>`.
The supervisor filters the raw feed by it.

The supervisor prefers Path A if `office_meas_many_fails.py` is present, else falls back to Path B.

### Sending the "info"

The failure detail (`ALARM_NAME`, e.g. the `20/100` count) is threaded into the cube notification
automatically — it shows up as `"Measurement Consecutive Fail (20/100) - <status>"`. By default the
abort cube reuses the existing `office_rich_notify.send_cube_align_fail_info` adapter; add a
dedicated `send_cube_meas_fail_info(...)` only if you want a measurement-specific cube format (the
template notes the optional signature).

## Activation checklist

1. Implement `office_meas_many_fails.get_measurement_fail_alarms()` (Path A) **or** set `MEAS_FAIL_ALID` (Path B).
2. Confirm the measurement-fail threshold value (e.g. 20 consecutive) and ALID with MES.
3. Run notify-only first; verify the `_rcs.jpg` evidence capture + the `[DRY-RUN] Abort 버튼 검출`
   coordinate land on the real Stop/Abort button, and that the cube shows the right fail count.
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
