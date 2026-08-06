"""guarded teardown 헬퍼 + 세 사이클의 teardown 순서 불변식 테스트.

RCS/Windows 없이 도는 단위 테스트다. 사이클 전체는 Mac 에서 RCS_MODULES_AVAILABLE
이 False 라 조기 반환하므로, teardown 목록을 만드는 함수(_teardown_steps 계열)만
직접 호출해 순서를 검사한다.

`uv run python poc/workflow_3/monitor/test_teardown.py` 로 직접 실행.
"""

from poc.workflow_3.monitor.teardown import run_teardown


def test_raising_step_does_not_block_later_steps():
    """한 단계가 던져도 뒤 단계는 반드시 실행된다 - teardown 의 핵심 계약."""
    calls = []

    def _boom():
        calls.append("boom")
        raise RuntimeError("terminate failed")

    failures = run_teardown([
        ("first", lambda: calls.append("first")),
        ("boom", _boom),
        ("last", lambda: calls.append("last")),
    ])
    assert calls == ["first", "boom", "last"], calls
    assert [n for n, _ in failures] == ["boom"], failures
    print("[OK] test_raising_step_does_not_block_later_steps")


def test_failures_returned_in_order_with_names():
    """실패는 (이름, 오류문자열) 로, 실행 순서대로 반환된다."""
    failures = run_teardown([
        ("a", lambda: (_ for _ in ()).throw(ValueError("va"))),
        ("b", lambda: None),
        ("c", lambda: (_ for _ in ()).throw(KeyError("kc"))),
    ])
    assert [n for n, _ in failures] == ["a", "c"], failures
    assert "va" in failures[0][1], failures[0][1]
    assert "ValueError" in failures[0][1], failures[0][1]
    print("[OK] test_failures_returned_in_order_with_names")


def test_helper_never_raises():
    """모든 단계가 던져도 헬퍼 자체는 절대 예외를 올리지 않는다."""
    failures = run_teardown([
        ("x", lambda: (_ for _ in ()).throw(RuntimeError("x"))),
        ("y", lambda: (_ for _ in ()).throw(RuntimeError("y"))),
    ], label="unit")
    assert len(failures) == 2, failures
    print("[OK] test_helper_never_raises")


def test_empty_list_is_noop():
    assert run_teardown([]) == []
    print("[OK] test_empty_list_is_noop")


def test_alarm_cycle_teardown_unblocks_input_first():
    """run_alarm_cycle 의 teardown 첫 단계는 반드시 입력 해제여야 한다.

    뒤 단계(녹화 중지/tool 닫기/팝업)가 전부 실패해도 엔지니어의 물리 입력은
    풀려 있어야 한다. 이것이 F1(check-only 입력 잠금) 계열의 회귀 테스트다.
    """
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.cycle import CycleResult, _teardown_steps

    settings = load_workflow3_settings()
    result = CycleResult(eqp_id="EQP1", recipe_id="C/R", tag="t")
    steps = _teardown_steps(
        "EQP1", {}, result, settings, input_blocked=True, recording=None
    )
    assert steps[0][0] == "input_unblock", [n for n, _ in steps]
    names = [n for n, _ in steps]
    assert names == ["input_unblock", "recording_stop", "close_tool", "close_alert"], names
    print("[OK] test_alarm_cycle_teardown_unblocks_input_first")


def test_check_only_teardown_unblocks_input_first():
    """F1 회귀 테스트 - check-only teardown 이 입력 해제를 마지막에 두면 안 된다.

    기존 순서는 close_tool -> close_alert_window(미보호) -> block_input(False) 라,
    close_alert_window 가 던지면 엔지니어 입력이 잠긴 채 남았다.
    """
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.cycle import _check_teardown_steps

    settings = load_workflow3_settings()
    steps = _check_teardown_steps("EQP1", {}, settings, input_blocked=True)
    names = [n for n, _ in steps]
    assert names[0] == "input_unblock", names
    assert names == ["input_unblock", "close_tool", "close_alert"], names
    print("[OK] test_check_only_teardown_unblocks_input_first")


def test_check_only_teardown_survives_failing_close_alert():
    """close_alert 가 던져도 앞선 입력 해제는 이미 실행됐고, 헬퍼는 안 던진다."""
    from poc.workflow_3.monitor.teardown import run_teardown

    calls = []
    failures = run_teardown([
        ("input_unblock", lambda: calls.append("unblock")),
        ("close_tool", lambda: calls.append("close_tool")),
        ("close_alert", lambda: (_ for _ in ()).throw(OSError("popup gone"))),
    ])
    assert calls == ["unblock", "close_tool"], calls
    assert [n for n, _ in failures] == ["close_alert"], failures
    print("[OK] test_check_only_teardown_survives_failing_close_alert")


def test_abort_cycle_teardown_unblocks_input_first():
    """workflow_3e abort 사이클도 같은 순서 규약을 따른다(F4 - 복제된 형태 방지)."""
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3e.abort_cycle import _abort_teardown_steps

    settings = load_workflow3_settings()
    steps = _abort_teardown_steps("EQP1", {}, settings, input_blocked=True)
    names = [n for n, _ in steps]
    assert names == ["input_unblock", "close_tool", "close_alert"], names
    print("[OK] test_abort_cycle_teardown_unblocks_input_first")


if __name__ == "__main__":
    test_raising_step_does_not_block_later_steps()
    test_failures_returned_in_order_with_names()
    test_helper_never_raises()
    test_empty_list_is_noop()
    test_alarm_cycle_teardown_unblocks_input_first()
    test_check_only_teardown_unblocks_input_first()
    test_check_only_teardown_survives_failing_close_alert()
    test_abort_cycle_teardown_unblocks_input_first()
    print("\n[OK] teardown 헬퍼 테스트 통과")
