"""실패 cooldown + tool 별 가드 스모크 테스트 (F2/F5).

RCS/office 없이 도는 단위 테스트다. process_fail_rows 의 의존성을 stub 으로
바꾸고 CycleResult 를 직접 만들어 분기를 검사한다.

`uv run python poc/workflow_3/monitor/test_failure_cooldown.py` 로 직접 실행.
"""

import time

from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult


def _swap(state, module, name, fn):
    """module.name 을 fn 으로 교체하고 원복용으로 저장."""
    state[(module, name)] = getattr(module, name)
    setattr(module, name, fn)


def _restore(state):
    for (module, name), orig in state.items():
        setattr(module, name, orig)


def _stub_deps(state, module, cycle_fn, cycle_attr="run_alarm_cycle"):
    _swap(state, module, "append_alarm_record", lambda *a, **k: None)
    _swap(state, module, "notify_align_fail_popup", lambda *a, **k: None)
    _swap(state, module, "gather_success_async", lambda *a, **k: None)
    _swap(state, module, "gather_rcp_msr", lambda *a, **k: None)
    _swap(state, module, "append_cycle_manifest", lambda *a, **k: None)
    _swap(state, module, cycle_attr, cycle_fn)


def _cycle_returning(**fields):
    def _fake(eqp_id, recipe_id, settings, tag=None):
        r = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "")
        for k, v in fields.items():
            setattr(r, k, v)
        return r
    return _fake


def test_error_cycle_registers_cooldown_and_skips_active():
    """run_status='error' 면 cooldown 등록 + active 미등록(재시도는 만료 후)."""
    state = {}
    _stub_deps(state, afm, _cycle_returning(run_status="error"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP1", "recipe_id": "C/R"}]

        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP1" not in active, active
        assert "EQP1" in cooldown, cooldown

        # cooldown 중에는 재처리하지 않는다.
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 0, n

        # 만료되면 다시 시도한다.
        cooldown["EQP1"] = time.time() - 1
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 1, n
    finally:
        _restore(state)
    print("[OK] test_error_cycle_registers_cooldown_and_skips_active")


def test_aborted_cycle_with_failed_step_registers_cooldown():
    """runner 중단(failed_step 세팅)도 실패로 보고 cooldown 을 건다."""
    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="aborted", failed_step="wait_tool_window"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        afm.process_fail_rows(
            [{"eqp_id": "EQP2", "recipe_id": "C/R"}], active, settings, cooldown)
        assert "EQP2" in cooldown and "EQP2" not in active
    finally:
        _restore(state)
    print("[OK] test_aborted_cycle_with_failed_step_registers_cooldown")


def test_correction_fallback_does_not_register_cooldown():
    """정상 수행 + fallback outcome 은 실패가 아니다 - 과잉 트리거 방지.

    _exec_run_correction 은 outcome.status 와 무관하게 success 를 반환하므로
    (cycle.py:468-475) fallback 은 run_status='completed' 로 온다.
    """
    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="completed", outcome_status="fallback_live_search"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        afm.process_fail_rows(
            [{"eqp_id": "EQP3", "recipe_id": "C/R"}], active, settings, cooldown)
        assert "EQP3" in active, active
        assert "EQP3" not in cooldown, cooldown
    finally:
        _restore(state)
    print("[OK] test_correction_fallback_does_not_register_cooldown")


def test_raising_tool_does_not_skip_remaining_tools():
    """tool 1대가 던져도 같은 poll 의 나머지 tool 은 처리된다(F5) + 던진 쪽은 cooldown."""
    state = {}
    seen = []

    def _cycle(eqp_id, recipe_id, settings, tag=None):
        seen.append(eqp_id)
        if eqp_id == "EQP_BAD":
            raise RuntimeError("boom")
        return CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "",
                           run_status="completed")

    _stub_deps(state, afm, _cycle)
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP_BAD", "recipe_id": "C/R"},
                 {"eqp_id": "EQP_OK", "recipe_id": "C/R"}]
        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP_OK" in seen, seen
        assert "EQP_OK" in active, active
        assert "EQP_BAD" in cooldown, cooldown
        assert "EQP_BAD" not in active, active
    finally:
        _restore(state)
    print("[OK] test_raising_tool_does_not_skip_remaining_tools")


if __name__ == "__main__":
    test_error_cycle_registers_cooldown_and_skips_active()
    test_aborted_cycle_with_failed_step_registers_cooldown()
    test_correction_fallback_does_not_register_cooldown()
    test_raising_tool_does_not_skip_remaining_tools()
    print("\n[OK] 실패 cooldown / tool 가드 테스트 통과")
