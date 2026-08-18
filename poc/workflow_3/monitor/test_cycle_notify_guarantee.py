"""run_alarm_cycle 알림 보장 self-test — 알람 1건당 cube 정확히 1회.

결과-후-알림 정책(감지 즉시 알리지 않고 접속->판정->보정까지 끝낸 뒤 통보)에서는
사이클이 예외로 죽으면 엔지니어가 그 알람에 대해 아무 통보도 못 받는다. 그 침묵을
막는 것이 여기 검증 대상이다.

CLAUDE.md 규칙: argparse 미사용, [OK] print, Mac 에서 그대로 실행(RCS 불필요).
    uv run python poc/workflow_3/monitor/test_cycle_notify_guarantee.py
"""

from types import SimpleNamespace

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import cycle as cyc
from poc.workflow_3.monitor import notify as ntf


def _swap(state, module, name, value):
    state.setdefault(module, {})[name] = getattr(module, name)
    setattr(module, name, value)


def _restore(state):
    for module, saved in state.items():
        for name, value in saved.items():
            setattr(module, name, value)


class _FakeRun:
    status = "completed"
    run_dir = ""
    step_results: list = []


def _stub_cycle(state, *, run_impl):
    """run_alarm_cycle 이 RCS 없이 돌도록 최소 의존만 대역으로 바꾼다.

    cube 발송은 CycleNotifier 를 거쳐 notify 모듈로 나가므로, 관측점은 cycle 이
    아니라 notify 쪽 sink 다(사이클이 어떤 경로로 부르든 총 발송 수가 잡힌다).
    """
    calls = []
    _swap(state, cyc, "RCS_MODULES_AVAILABLE", True)
    _swap(state, ntf, "notify_correction_outcome",
          lambda *a, **k: calls.append((a, k)))
    _swap(state, ntf, "send_progress_notify", lambda *a, **k: None)
    _swap(state, cyc, "close_alert_window", lambda *a, **k: True)
    _swap(state, cyc, "log_work2_event", lambda **k: None)
    _swap(state, cyc, "WorkflowRunner",
          lambda *a, **k: SimpleNamespace(run=run_impl))
    return calls


def test_cycle_notifies_once_on_normal_completion():
    """정상 종료: 본문에서 1회 발송, finally 재호출은 중복을 만들지 않는다."""
    state = {}
    calls = _stub_cycle(state, run_impl=lambda *a, **k: _FakeRun())
    try:
        settings = load_workflow3_settings()
        result = cyc.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="t1")
    finally:
        _restore(state)
    assert len(calls) == 1, calls
    assert result.run_status == "completed", result.run_status
    print("[OK] test_cycle_notifies_once_on_normal_completion")


def test_cycle_notifies_when_runner_raises():
    """예외 종료: 본문이 발송에 도달 못 해도 finally 가 반드시 통보한다."""
    def _boom(*a, **k):
        raise RuntimeError("step executor 폭발")

    state = {}
    calls = _stub_cycle(state, run_impl=_boom)
    try:
        settings = load_workflow3_settings()
        result = cyc.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="t2")
    finally:
        _restore(state)
    assert len(calls) == 1, calls
    # outcome 은 None - "자동 보정 미수행, 직접 확인 필요" 요약이 나가야 한다.
    assert calls[0][0][2] is None, calls
    assert result.run_status == "error", result.run_status
    print("[OK] test_cycle_notifies_when_runner_raises")


def test_cycle_notifies_when_teardown_also_raises():
    """teardown 까지 깨져도 알림은 이미 나가 있어야 한다(알림이 teardown 뒤가 아님)."""
    def _boom(*a, **k):
        raise RuntimeError("step executor 폭발")

    state = {}
    calls = _stub_cycle(state, run_impl=_boom)
    _swap(state, cyc, "run_teardown",
          lambda *a, **k: (_ for _ in ()).throw(RuntimeError("teardown 폭발")))
    try:
        settings = load_workflow3_settings()
        try:
            cyc.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="t3")
        except RuntimeError:
            pass  # teardown 예외는 이 테스트의 관심사가 아니다.
    finally:
        _restore(state)
    assert len(calls) == 1, calls
    print("[OK] test_cycle_notifies_when_teardown_also_raises")


def test_cycle_reports_failed_stage_to_cube():
    """step 실패로 중단되면 어느 단계에서 멈췄는지가 알림에 실려야 한다."""
    class _FailedRun:
        status = "aborted"
        run_dir = ""
        step_results = [
            SimpleNamespace(status="success", step_id="ensure_rcs_ready",
                            failure_class=""),
            SimpleNamespace(status="failed", step_id="wait_tool_window",
                            failure_class="rcs_occupied_select"),
        ]

    state = {}
    calls = _stub_cycle(state, run_impl=lambda *a, **k: _FailedRun())
    try:
        settings = load_workflow3_settings()
        result = cyc.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="t4")
    finally:
        _restore(state)
    assert len(calls) == 1, calls
    kwargs = calls[0][1]
    assert kwargs.get("failed_step") == "wait_tool_window", kwargs
    assert kwargs.get("failure_class") == "rcs_occupied_select", kwargs
    assert result.failed_step == "wait_tool_window", result.failed_step
    print("[OK] test_cycle_reports_failed_stage_to_cube")


def main() -> int:
    print("[INFO] run_alarm_cycle 알림 보장 self-test 시작")
    tests = [
        test_cycle_notifies_once_on_normal_completion,
        test_cycle_notifies_when_runner_raises,
        test_cycle_notifies_when_teardown_also_raises,
        test_cycle_reports_failed_stage_to_cube,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print(f"[INFO] {len(tests) - failed}/{len(tests)} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
