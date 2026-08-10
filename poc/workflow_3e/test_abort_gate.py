"""abort 라벨 확인 게이트가 실제로 클릭을 막는지 검증한다 (VLM/Windows 불필요).

`abort_button.py` 의 순수 판정 로직은 `test_abort_button.py` 가 본다. 여기서 보는 것은
**배선**이다 - 게이트가 호출부에 실제로 연결돼 있어서, 확인되지 않은 좌표면 클릭이
일어나지 않는가. 순수 로직만 맞고 배선이 빠지면 게이트는 장식이다.

    uv run python poc/workflow_3e/test_abort_gate.py
"""

from dataclasses import replace
from types import SimpleNamespace

from PIL import Image

from poc.workflow_3e import abort_button as ab
from poc.workflow_3e import abort_cycle as ac
from poc.workflow_3e.config import load_workflow3e_settings


def _swap(state, module, name, fn):
    state[(module, name)] = getattr(module, name)
    setattr(module, name, fn)


def _restore(state):
    for (module, name), orig in state.items():
        setattr(module, name, orig)


def _armed_settings():
    """실제 클릭이 무장된 설정(이중 게이트 통과) - 라벨 게이트만 남긴다."""
    return replace(
        load_workflow3e_settings(), action_enabled=True, abort_action_dry_run=False
    )


def _run_gate(verdict, policy, clicks, *, confirm_xy=None, confirm_verdict=None):
    """스텁을 걸고 _exec_abort_measurement 를 1회 실행한다. (context, result) 반환.

    confirm_xy 를 주면 abort 클릭 뒤 확인 다이얼로그가 뜬 상황을 재현한다. 이때 라벨 확인은
    expected_labels 로 어느 게이트인지 구분해 confirm_verdict 를 돌려준다.
    """
    state = {}
    _swap(state, ac, "capture_window", lambda win: Image.new("RGB", (1000, 500)))
    _swap(state, ac, "locate_abort_button", lambda **kw: (500, 400))
    _swap(state, ac, "locate_abort_confirm", lambda **kw: confirm_xy)

    def fake_verify(image, point_xy, expected_labels, **kw):
        if expected_labels is ab.CONFIRM_BUTTON_LABELS:
            return confirm_verdict
        return verdict

    _swap(state, ac, "verify_button_label_at_point", fake_verify)
    _swap(state, ac, "load_abort_label_policy", lambda: policy)
    _swap(
        state, ac, "click_at_screen",
        lambda point, label, **kw: clicks.append((label, point["x"], point["y"])),
    )

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    _swap(state, ac, "_make_abort_vlm_client", lambda settings: _FakeClient())

    step = SimpleNamespace(step_id="abort_measurement", step_type="action")
    context = {"eqp_id": "EQP1", "tag": "T", "tool_window": object()}
    try:
        result = ac._exec_abort_measurement(step, context, _armed_settings())
    finally:
        _restore(state)
    return context, result


def test_mismatch_blocks_click():
    """다른 버튼을 읽었으면(Pause 등) 절대 누르지 않는다 - 가장 중요한 케이스."""
    clicks = []
    verdict = ab.classify_button_tokens(["Pause"], ab.ABORT_BUTTON_LABELS)
    context, result = _run_gate(verdict, ab.ABORT_LABEL_POLICY_STRICT, clicks)
    ok = (
        clicks == []
        and context["abort_outcome"] == "abort_label_unconfirmed"
        and result.status == "failed"
    )
    print(f"[{'PASS' if ok else 'FAIL'}] mismatch_blocks_click: clicks={clicks} outcome={context.get('abort_outcome')}")
    return ok


def test_unreadable_blocks_click_under_strict():
    clicks = []
    verdict = ab.classify_button_tokens([], ab.ABORT_BUTTON_LABELS)
    context, _ = _run_gate(verdict, ab.ABORT_LABEL_POLICY_STRICT, clicks)
    ok = clicks == [] and context["abort_outcome"] == "abort_label_unconfirmed"
    print(f"[{'PASS' if ok else 'FAIL'}] unreadable_blocks_click_under_strict: clicks={clicks}")
    return ok


def test_ocr_unavailable_blocks_click_under_strict():
    """OCR 자체를 못 돌린 경우(verdict None)도 strict 에선 클릭 금지."""
    clicks = []
    context, _ = _run_gate(None, ab.ABORT_LABEL_POLICY_STRICT, clicks)
    ok = clicks == [] and context["abort_outcome"] == "abort_label_unconfirmed"
    print(f"[{'PASS' if ok else 'FAIL'}] ocr_unavailable_blocks_click_under_strict: clicks={clicks}")
    return ok


def test_confirmed_allows_click():
    clicks = []
    verdict = ab.classify_button_tokens(["Stop"], ab.ABORT_BUTTON_LABELS)
    context, result = _run_gate(verdict, ab.ABORT_LABEL_POLICY_STRICT, clicks)
    ok = (
        [c[0] for c in clicks] == ["abort_button"]
        and context["abort_outcome"] == "aborted"
        and result.status == "success"
    )
    print(f"[{'PASS' if ok else 'FAIL'}] confirmed_allows_click: clicks={clicks}")
    return ok


def test_policy_off_allows_click_without_verification():
    """롤백 경로 - 게이트를 끄면 확인 없이 기존 동작 그대로."""
    clicks = []
    verdict = ab.classify_button_tokens(["Pause"], ab.ABORT_BUTTON_LABELS)
    context, _ = _run_gate(verdict, ab.ABORT_LABEL_POLICY_OFF, clicks)
    ok = [c[0] for c in clicks] == ["abort_button"] and context["abort_outcome"] == "aborted"
    print(f"[{'PASS' if ok else 'FAIL'}] policy_off_allows_click_without_verification: clicks={clicks}")
    return ok


def test_confirm_dialog_mismatch_blocks_confirm_click():
    """확인 다이얼로그의 Yes/확인 도 같은 게이트를 통과해야 한다.

    여기서 막히면 Stop 은 이미 눌렸고 다이얼로그가 열린 채 남는다 - No/취소 를 잘못 누르는
    것보다 낫다(엔지니어가 마무리).
    """
    clicks = []
    ok_verdict = ab.classify_button_tokens(["Stop"], ab.ABORT_BUTTON_LABELS)
    bad_confirm = ab.classify_button_tokens(["Cancel"], ab.CONFIRM_BUTTON_LABELS)
    context, _ = _run_gate(
        ok_verdict, ab.ABORT_LABEL_POLICY_STRICT, clicks,
        confirm_xy=(600, 300), confirm_verdict=bad_confirm,
    )
    ok = (
        [c[0] for c in clicks] == ["abort_button"]
        and context.get("abort_confirm_verdict") == "mismatch"
    )
    print(f"[{'PASS' if ok else 'FAIL'}] confirm_dialog_mismatch_blocks_confirm_click: clicks={clicks}")
    return ok


def test_confirm_dialog_confirmed_allows_confirm_click():
    clicks = []
    ok_verdict = ab.classify_button_tokens(["Stop"], ab.ABORT_BUTTON_LABELS)
    good_confirm = ab.classify_button_tokens(["확인"], ab.CONFIRM_BUTTON_LABELS)
    _, _ = _run_gate(
        ok_verdict, ab.ABORT_LABEL_POLICY_STRICT, clicks,
        confirm_xy=(600, 300), confirm_verdict=good_confirm,
    )
    ok = [c[0] for c in clicks] == ["abort_button", "abort_confirm"]
    print(f"[{'PASS' if ok else 'FAIL'}] confirm_dialog_confirmed_allows_confirm_click: clicks={clicks}")
    return ok


def main():
    print("[INFO] abort 라벨 게이트 배선 테스트 시작")
    results = [
        test_mismatch_blocks_click(),
        test_unreadable_blocks_click_under_strict(),
        test_ocr_unavailable_blocks_click_under_strict(),
        test_confirmed_allows_click(),
        test_policy_off_allows_click_without_verification(),
        test_confirm_dialog_mismatch_blocks_confirm_click(),
        test_confirm_dialog_confirmed_allows_confirm_click(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
