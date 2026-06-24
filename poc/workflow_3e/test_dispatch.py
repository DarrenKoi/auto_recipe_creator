"""process_abort_rows edge-trigger / 게이트 self-test (run_abort_cycle 모킹).

- meas_fail_abort_enabled=False 면 사이클 안 돈다.
- 새 EQP_ID 마다 1회만 abort 사이클(중복 알람 무시), 해제되면 재처리 가능.

    uv run python poc/workflow_3e/test_dispatch.py
"""

import dataclasses

import pandas as pd

from poc.workflow_3.monitor.cycle import CycleResult
from poc.workflow_3e import dispatch
from poc.workflow_3e.config import load_workflow3e_settings


def _settings(**over):
    return dataclasses.replace(load_workflow3e_settings(), **over)


def _rows(*eqps):
    return pd.DataFrame(
        [{"EQP_ID": e, "ALID": "9012", "UTC9": "", "RECIPE_ID": "CLS/RCP"} for e in eqps]
    )


def test_disabled_skips():
    calls = []
    orig = dispatch.run_abort_cycle
    dispatch.run_abort_cycle = lambda *a, **k: calls.append(a) or CycleResult("E", "R", "t")
    try:
        n = dispatch.process_abort_rows(
            _rows("EQP1"), set(), _settings(meas_fail_abort_enabled=False), {}
        )
    finally:
        dispatch.run_abort_cycle = orig
    ok = n == 0 and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] disabled_skips: handled={n} calls={len(calls)}")
    return ok


def test_edge_trigger_once_per_tool():
    calls = []
    orig = dispatch.run_abort_cycle
    dispatch.run_abort_cycle = lambda eqp, rcp, st, **k: (calls.append(eqp) or CycleResult(eqp, rcp, "t"))
    active: set = set()
    try:
        s = _settings(meas_fail_abort_enabled=True, cycle_enabled=True)
        n1 = dispatch.process_abort_rows(_rows("EQP1"), active, s, {})
        n2 = dispatch.process_abort_rows(_rows("EQP1"), active, s, {})  # 중복 - 무시
    finally:
        dispatch.run_abort_cycle = orig
    ok = n1 == 1 and n2 == 0 and calls == ["EQP1"]
    print(f"[{'PASS' if ok else 'FAIL'}] edge_trigger_once_per_tool: n1={n1} n2={n2} calls={calls}")
    return ok


def test_clear_allows_reprocess():
    """알람이 사라졌다 다시 오면 재처리된다."""
    calls = []
    orig = dispatch.run_abort_cycle
    dispatch.run_abort_cycle = lambda eqp, rcp, st, **k: (calls.append(eqp) or CycleResult(eqp, rcp, "t"))
    active: set = set()
    try:
        s = _settings(meas_fail_abort_enabled=True, cycle_enabled=True)
        dispatch.process_abort_rows(_rows("EQP1"), active, s, {})
        dispatch.process_abort_rows(_rows(), active, s, {})       # 해제 -> active 비움
        n3 = dispatch.process_abort_rows(_rows("EQP1"), active, s, {})  # 재처리
    finally:
        dispatch.run_abort_cycle = orig
    ok = n3 == 1 and calls == ["EQP1", "EQP1"]
    print(f"[{'PASS' if ok else 'FAIL'}] clear_allows_reprocess: n3={n3} calls={calls}")
    return ok


def main():
    print("[INFO] process_abort_rows self-test 시작")
    results = [test_disabled_skips(), test_edge_trigger_once_per_tool(), test_clear_allows_reprocess()]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
