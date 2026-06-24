"""run_abort_cycle dry-run self-test (RCS 불필요).

RCS 모듈이 없는 개발 PC 에서 run_abort_cycle 이 rcs_unavailable 로 안전 종료하고
cube notify 를 1회 호출하는지, CycleResult 형태가 맞는지 검증한다.

    uv run python poc/workflow_3e/test_abort_cycle.py
"""

from poc.workflow_3.monitor.cycle import CycleResult
from poc.workflow_3e import abort_cycle as ac
from poc.workflow_3e.config import load_workflow3e_settings


def test_rcs_unavailable_safe_exit():
    """RCS 모듈 비활성이면 rcs_unavailable 로 끝나고 notify 가 1회 불린다."""
    calls = []
    orig_notify = ac.notify_abort_outcome
    orig_avail = ac.RCS_MODULES_AVAILABLE
    ac.notify_abort_outcome = lambda *a, **k: calls.append((a, k))
    ac.RCS_MODULES_AVAILABLE = False
    try:
        result = ac.run_abort_cycle("EQP1", "CLS/RCP", load_workflow3e_settings(), tag="t0")
    finally:
        ac.notify_abort_outcome = orig_notify
        ac.RCS_MODULES_AVAILABLE = orig_avail
    ok = (isinstance(result, CycleResult) and result.run_status == "rcs_unavailable"
          and len(calls) == 1 and result.eqp_id == "EQP1")
    print(f"[{'PASS' if ok else 'FAIL'}] rcs_unavailable_safe_exit: "
          f"status={result.run_status} notify_calls={len(calls)}")
    return ok


def test_build_abort_steps_shape():
    """abort step 시퀀스가 접속 -> 캡처 -> abort 순서를 갖는다."""
    steps = ac.build_abort_steps("EQP1")
    ids = [s.step_id for s in steps]
    ok = ids == [
        "ensure_rcs_ready", "close_alert_popup", "connect_tool",
        "wait_tool_window", "capture_screen", "abort_measurement",
    ]
    print(f"[{'PASS' if ok else 'FAIL'}] build_abort_steps_shape: ids={ids}")
    return ok


def main():
    print("[INFO] run_abort_cycle self-test 시작")
    results = [test_rcs_unavailable_safe_exit(), test_build_abort_steps_shape()]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
