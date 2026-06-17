"""점유 'select' 팝업 검출 + abort 인터리브 + cooldown 재시도 스모크 테스트.

VLM/Windows 없이 도는 단위 테스트(의존성 stub):
  * _is_select_title / detect_select_popup — 제목 검출 + VLM 확인 + 폴백 분기.
  * wait_for_remote_monitoring_window(abort_check=...) — 팝업 감지 시 첫 시도에 조기 중단.
  * process_fail_rows — 점유(rcs_occupied_select)면 active 미등록 + cooldown, 만료 후 재시도.

`uv run python poc/workflow_3/monitor/test_occupied_popup.py` 로 직접 실행.
"""

import time
from types import SimpleNamespace

from PIL import Image

from poc.workflow_3.monitor import occupied_popup as op
from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult
from poc.workflow_3.rcs import login_rcs_common as lrc


def _swap(state, module, name, fn):
    """module.name 을 fn 으로 교체하고 원복용으로 저장."""
    state[(module, name)] = getattr(module, name)
    setattr(module, name, fn)


def _restore(state):
    for (module, name), orig in state.items():
        setattr(module, name, orig)


def _fake_vlm(is_popup: bool):
    text = '{"is_select_popup": %s, "options_seen": []}' % ("true" if is_popup else "false")
    return SimpleNamespace(
        chat_with_image_b64=lambda **kw: SimpleNamespace(text=text)
    )


# ------------------------------------------------------------------
# 1) _is_select_title.
# ------------------------------------------------------------------


def test_is_select_title():
    assert op._is_select_title("select")
    assert op._is_select_title("SELECT")
    assert op._is_select_title("  Select  ")
    assert not op._is_select_title("Remote Monitoring System - EQP1")
    assert not op._is_select_title("")
    print("[OK] test_is_select_title")


# ------------------------------------------------------------------
# 2) detect_select_popup 분기.
# ------------------------------------------------------------------


def test_detect_no_select_title():
    state = {}
    _swap(state, op, "collect_window_rows", lambda: [SimpleNamespace(title="Remote Monitoring System")])
    try:
        assert op.detect_select_popup(_fake_vlm(True)) is False   # 제목 없음 → VLM 호출 전 False.
    finally:
        _restore(state)
    print("[OK] test_detect_no_select_title")


def test_detect_title_no_vlm():
    state = {}
    _swap(state, op, "collect_window_rows", lambda: [SimpleNamespace(title="select")])
    try:
        assert op.detect_select_popup(None) is True   # 제목만으로 점유 판단.
    finally:
        _restore(state)
    print("[OK] test_detect_title_no_vlm")


def test_detect_vlm_confirms():
    state = {}
    _swap(state, op, "collect_window_rows", lambda: [SimpleNamespace(title="select")])
    _swap(state, op, "find_window_by_title_prefix", lambda *a, **k: object())
    _swap(state, op, "capture_window", lambda win: Image.new("RGB", (80, 40)))
    try:
        assert op.detect_select_popup(_fake_vlm(True)) is True
    finally:
        _restore(state)
    print("[OK] test_detect_vlm_confirms")


def test_detect_vlm_rejects():
    state = {}
    _swap(state, op, "collect_window_rows", lambda: [SimpleNamespace(title="select")])
    _swap(state, op, "find_window_by_title_prefix", lambda *a, **k: object())
    _swap(state, op, "capture_window", lambda win: Image.new("RGB", (80, 40)))
    try:
        # VLM 이 점유 팝업 아님으로 판단 → 오검출로 보고 접속 계속(False).
        assert op.detect_select_popup(_fake_vlm(False)) is False
    finally:
        _restore(state)
    print("[OK] test_detect_vlm_rejects")


def test_detect_vlm_unavailable_fallback():
    state = {}
    _swap(state, op, "collect_window_rows", lambda: [SimpleNamespace(title="select")])
    # 창 핸들 못 얻음 → confirm None → 제목만으로 점유 판단(True).
    _swap(state, op, "find_window_by_title_prefix", lambda *a, **k: None)
    try:
        assert op.detect_select_popup(_fake_vlm(True)) is True
    finally:
        _restore(state)
    print("[OK] test_detect_vlm_unavailable_fallback")


# ------------------------------------------------------------------
# 3) abort_check 인터리브 — 팝업 감지 시 첫 시도에 조기 중단.
# ------------------------------------------------------------------


def test_wait_abort_check_short_circuits():
    state = {}
    find_calls = {"n": 0}

    def fake_find(tool_name):
        find_calls["n"] += 1
        return None, "", ""

    _swap(state, lrc, "find_remote_monitoring_window", fake_find)
    try:
        window, title, backend = lrc.wait_for_remote_monitoring_window(
            "EQP1", timeout_sec=5.0, poll_interval_sec=0.01, max_attempts=3,
            abort_check=lambda: True,
        )
    finally:
        _restore(state)
    assert window is None
    # abort_check 가 find 보다 먼저 검사되므로 find 는 호출되지 않아야 한다(조기 중단).
    assert find_calls["n"] == 0, find_calls
    print("[OK] test_wait_abort_check_short_circuits")


# ------------------------------------------------------------------
# 4) process_fail_rows — 점유 cooldown / 재시도.
# ------------------------------------------------------------------


def _stub_process_deps(state, failure_class):
    _swap(state, afm, "append_alarm_record", lambda *a, **k: None)
    _swap(state, afm, "notify_align_fail_popup", lambda *a, **k: None)
    _swap(state, afm, "gather_success_async", lambda *a, **k: None)
    _swap(state, afm, "gather_rcp_msr", lambda *a, **k: None)
    _swap(state, afm, "append_cycle_manifest", lambda *a, **k: None)

    def fake_cycle(eqp_id, recipe_id, settings, tag=None):
        r = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "")
        r.failure_class = failure_class
        return r

    _swap(state, afm, "run_alarm_cycle", fake_cycle)


def test_process_fail_rows_occupied_cooldown():
    state = {}
    _stub_process_deps(state, failure_class="rcs_occupied_select")
    try:
        settings = afm.load_workflow3_settings()
        active: set = set()
        cooldown: dict = {}
        fails = [{"eqp_id": "EQP1", "recipe_id": "C/R"}]

        # 1차: 점유 → active 미등록 + cooldown 등록.
        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP1" not in active, active
        assert "EQP1" in cooldown, cooldown

        # 2차: 같은 알람, cooldown 중 → 건너뜀(신규 처리 0).
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 0 and "EQP1" not in active

        # cooldown 만료 → 재시도(다시 점유라 또 cooldown, active 미등록).
        cooldown["EQP1"] = time.time() - 1
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 1, n
        assert "EQP1" not in active and "EQP1" in cooldown
    finally:
        _restore(state)
    print("[OK] test_process_fail_rows_occupied_cooldown")


def test_process_fail_rows_success_marks_active():
    """점유가 아니면(예: corrected/'') active 등록 + cooldown 미등록(기존 동작)."""
    state = {}
    _stub_process_deps(state, failure_class="")
    try:
        settings = afm.load_workflow3_settings()
        active: set = set()
        cooldown: dict = {}
        fails = [{"eqp_id": "EQP2", "recipe_id": "C/R"}]
        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP2" in active and "EQP2" not in cooldown
    finally:
        _restore(state)
    print("[OK] test_process_fail_rows_success_marks_active")


if __name__ == "__main__":
    test_is_select_title()
    test_detect_no_select_title()
    test_detect_title_no_vlm()
    test_detect_vlm_confirms()
    test_detect_vlm_rejects()
    test_detect_vlm_unavailable_fallback()
    test_wait_abort_check_short_circuits()
    test_process_fail_rows_occupied_cooldown()
    test_process_fail_rows_success_marks_active()
    print("\n=== occupied_popup: 9/9 통과 ===")
