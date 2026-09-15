"""align_fail_timing.csv 행 계산 테스트 (RCS/office 없이 실행).

`uv run pytest poc/workflow_3/monitor/test_cycle_timing.py`
"""

from datetime import datetime

from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult


def _local(hms: str) -> float:
    return datetime.strptime(f"2026-09-15 {hms}", "%Y-%m-%d %H:%M:%S").timestamp()


def _row(info, cycle) -> dict:
    return dict(zip(afm.CYCLE_TIMING_COLUMNS, afm.build_timing_row(info, cycle), strict=True))


def test_full_cycle_durations_use_local_alarm_time():
    cycle = CycleResult(
        eqp_id="EQP1", recipe_id="CLS/RCP", tag="t", outcome_status="corrected",
        started_at=_local("10:01:00"), finished_at=_local("10:03:00"),
        correction_started_at=_local("10:01:30"), correction_finished_at=_local("10:02:10"),
    )
    row = _row({"utc9": "2026-09-15 10:00:00"}, cycle)

    assert row["correction_sec"] == "40.0"
    # naive UTC9 를 UTC 로 해석하면 KST 에서 9시간(32400s) 어긋난다.
    assert row["alarm_to_correction_sec"] == "130.0"
    assert row["cycle_sec"] == "120.0"
    assert row["correction_finished_at"] == "2026-09-15 10:02:10"


def test_correction_not_reached_leaves_blanks():
    cycle = CycleResult(
        eqp_id="EQP1", recipe_id="", tag="t", failure_class="rcs_occupied",
        started_at=_local("10:01:00"), finished_at=_local("10:01:45"),
    )
    row = _row({"utc9": "not-a-time"}, cycle)

    assert row["correction_started_at"] == ""
    assert row["correction_sec"] == ""
    assert row["alarm_to_correction_sec"] == ""
    assert row["cycle_sec"] == "45.0"
