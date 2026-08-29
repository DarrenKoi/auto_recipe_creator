"""실패 cooldown + tool 별 가드 스모크 테스트 (F2/F5).

RCS/office 없이 도는 단위 테스트다. process_fail_rows 의 의존성을 stub 으로
바꾸고 CycleResult 를 직접 만들어 분기를 검사한다.

`uv run python poc/workflow_3/monitor/test_failure_cooldown.py` 로 직접 실행.
"""

import dataclasses
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
    """process_fail_rows 의존성을 no-op 로 교체한다.

    check-only 모니터는 production 에 없는 send_detection_notify_async 를 추가로
    호출하므로, 모듈에 있는 이름만 골라 교체해 두 모니터에 공용으로 쓴다.
    """
    for name in ("append_alarm_record", "notify_align_fail_popup",
                 "send_detection_notify_async", "gather_success_async",
                 "gather_rcp_msr", "append_cycle_manifest"):
        if hasattr(module, name):
            _swap(state, module, name, lambda *a, **k: None)
    _swap(state, module, cycle_attr, cycle_fn)


def _cycle_returning(**fields):
    # **_kwargs: Episode 수집이 켜지면 run_alarm_cycle 이 attempt_seq 도 받는다.
    # 이 fake 는 사이클 인자를 검사하지 않으므로 조용히 흘려보낸다.
    def _fake(eqp_id, recipe_id, settings, tag=None, **_kwargs):
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


def test_occupied_cycle_uses_occupied_duration_not_failure_duration():
    """점유(occupied) 사이클은 두 분기를 모두 만족한다 - 분기 순서가 정답을 가른다.

    occupied 사이클은 failure_class='rcs_occupied_select' 도 세팅되고 failed_step 도
    비어있지 않게 만들 수 있다(_cycle_failed 도 True). process_fail_rows 의
    if/elif 순서가 바뀌면(occupied 체크가 뒤로 가면) 조용히 failure_retry_cooldown_sec
    가 적용된다 - 두 설정 기본값이 둘 다 300.0 이라 기존 테스트(cooldown 등록 여부만
    확인)는 이 차이를 못 잡는다. 여기서는 두 값을 뚜렷이 다르게 오버라이드해 실제로
    OCCUPIED 값이 쓰였는지 확정한다.
    """
    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="aborted",
        failed_step="wait_tool_window",
        failure_class="rcs_occupied_select",
    ))
    try:
        base_settings = afm.load_workflow3_settings()
        occupied_sec = 1000.0
        failure_sec = 50.0
        settings = dataclasses.replace(
            base_settings,
            occupied_retry_cooldown_sec=occupied_sec,
            failure_retry_cooldown_sec=failure_sec,
        )
        active, cooldown = set(), {}
        before = time.time()
        afm.process_fail_rows(
            [{"eqp_id": "EQP_OCC", "recipe_id": "C/R"}], active, settings, cooldown)
        after = time.time()

        assert "EQP_OCC" in cooldown and "EQP_OCC" not in active, (cooldown, active)
        expiry = cooldown["EQP_OCC"]
        # occupied_sec(1000) 기준 넓은 구간 - failure_sec(50) 기준 구간과 겹치지 않는다.
        assert before + occupied_sec - 5 <= expiry <= after + occupied_sec + 5, (
            expiry, before, after)
        assert not (before + failure_sec - 5 <= expiry <= after + failure_sec + 5), (
            "failure_retry_cooldown_sec 가 적용됐다 - 분기 순서가 뒤바뀐 회귀", expiry)
    finally:
        _restore(state)
    print("[OK] test_occupied_cycle_uses_occupied_duration_not_failure_duration")


def test_check_only_monitor_registers_failure_cooldown():
    """check-only 모니터도 같은 규약 - 실패 tool 은 cooldown, 나머지는 계속(F2/F5)."""
    from poc.workflow_3.monitor import align_fail_monitor_only_check as afmc

    state = {}
    seen = []

    def _cycle(eqp_id, recipe_id, settings, tag=None):
        seen.append(eqp_id)
        if eqp_id == "EQP_BAD":
            raise RuntimeError("boom")
        return CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "",
                           run_status="completed")

    _stub_deps(state, afmc, _cycle, cycle_attr="run_check_only_cycle")
    try:
        settings = afmc.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP_BAD", "recipe_id": "C/R"},
                 {"eqp_id": "EQP_OK", "recipe_id": "C/R"}]
        afmc.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP_OK" in seen and "EQP_OK" in active, (seen, active)
        assert "EQP_BAD" in cooldown and "EQP_BAD" not in active, (cooldown, active)
    finally:
        _restore(state)
    print("[OK] test_check_only_monitor_registers_failure_cooldown")


if __name__ == "__main__":
    test_error_cycle_registers_cooldown_and_skips_active()
    test_aborted_cycle_with_failed_step_registers_cooldown()
    test_correction_fallback_does_not_register_cooldown()
    test_raising_tool_does_not_skip_remaining_tools()
    test_occupied_cycle_uses_occupied_duration_not_failure_duration()
    test_check_only_monitor_registers_failure_cooldown()
    print("\n[OK] 실패 cooldown / tool 가드 테스트 통과")
