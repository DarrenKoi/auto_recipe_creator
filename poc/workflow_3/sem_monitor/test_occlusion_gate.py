"""캡처 직전 가림 대기 게이트 self-test — 실장비/VLM 없이 Mac 에서 돈다.

접속 요청("Information") 팝업이 3초간 tool 창을 가리는 동안 캡처하면 팝업 픽셀이
프레임에 들어오고, 보정은 그 프레임에서 좌표를 뽑는다. 이 테스트는 게이트가
(1) 가림이 걷힐 때까지 기다렸다가 캡처하고 (2) 판정 불가/미주입이면 무해하게
비켜서며 (3) 예산을 넘겨도 예외 대신 캡처로 진행하는지를 확인한다.

    uv run python poc/workflow_3/sem_monitor/test_occlusion_gate.py
"""

import numpy as np

from poc.workflow_3.sem_monitor import controller as ctrl


class _FakePanel:
    model_id = "fake"
    panel_roi = (0, 0, 40, 30)
    confidence = 0.9


def _monitor(states, **kwargs):
    """states 를 순서대로 뱉는 가림 판정자를 단 controller 를 만든다."""
    seq = list(states)

    def _probe():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return (
        ctrl.RCSSEMMonitor(
            object(), _FakePanel(), occlusion_fn=_probe, **kwargs
        ),
        seq,
    )


def _with_capture(fn):
    """capture_window 를 상수 프레임 대역으로 바꾼다."""
    orig = ctrl.capture_window
    ctrl.capture_window = lambda _win: np.zeros((30, 40, 3), dtype=np.uint8)
    try:
        return fn()
    finally:
        ctrl.capture_window = orig


def test_waits_until_occlusion_clears() -> bool:
    """가림(full) 이 걷히면 그때 캡처한다 - 팝업 프레임으로 좌표를 뽑지 않는다."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 2.0
    try:
        mon, seq = _monitor(["full", "full", "none"])
        state = mon._wait_unoccluded()
        ok = state == "none" and len(seq) == 1
        print(f"[{'PASS' if ok else 'FAIL'}] 가림 해소까지 대기: state={state}")
        return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_unknown_does_not_block() -> bool:
    """판정 불가(unknown)는 가림이 아니다 - Mac/조회 실패에서 게이트가 no-op."""
    mon, _ = _monitor(["unknown"])
    state = mon._wait_unoccluded()
    ok = state == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] unknown 은 통과: state={state}")
    return ok


def test_no_probe_is_noop() -> bool:
    """occlusion_fn 미주입이면 기존 호출부/mock 은 아무 영향이 없다."""
    mon = ctrl.RCSSEMMonitor(object(), _FakePanel())
    ok = mon.occlusion_fn is None and mon._wait_unoccluded() == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] 미주입이면 게이트 no-op")
    return ok


def test_budget_exhausted_still_captures() -> bool:
    """영구 가림에서도 예외 대신 캡처로 진행한다(작업 표시줄 등과 구분 불가)."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.05
    try:
        mon, _ = _monitor(["full"])
        frame = _with_capture(mon.capture_screen)
        ok = frame.shape == (30, 40)
        print(f"[{'PASS' if ok else 'FAIL'}] 예산 소진 후에도 캡처 진행: shape={frame.shape}")
        return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_probe_exception_does_not_break_capture() -> bool:
    """판정자가 던져도 캡처는 계속된다 - 감시는 보조 장치다."""
    def _boom():
        raise RuntimeError("probe down")

    mon = ctrl.RCSSEMMonitor(object(), _FakePanel(), occlusion_fn=_boom)
    frame = _with_capture(mon.capture_screen)
    ok = frame.shape == (30, 40)
    print(f"[{'PASS' if ok else 'FAIL'}] 판정 예외에도 캡처 진행")
    return ok


def test_wait_sec_zero_disables() -> bool:
    """ALIGN_SEM_OCCLUSION_WAIT_SEC=0 롤백 스위치 - 판정자를 아예 안 부른다."""
    orig = ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_WAIT_SEC = 0.0
    calls = []
    try:
        mon = ctrl.RCSSEMMonitor(
            object(), _FakePanel(),
            occlusion_fn=lambda: calls.append(1) or "full",
        )
        state = mon._wait_unoccluded()
        ok = state == "unknown" and not calls
        print(f"[{'PASS' if ok else 'FAIL'}] WAIT_SEC=0 이면 감시 끔: calls={len(calls)}")
        return ok
    finally:
        ctrl.OCCLUSION_WAIT_SEC = orig


def main() -> int:
    tests = [
        test_waits_until_occlusion_clears,
        test_unknown_does_not_block,
        test_no_probe_is_noop,
        test_budget_exhausted_still_captures,
        test_probe_exception_does_not_break_capture,
        test_wait_sec_zero_disables,
    ]
    results = [t() for t in tests]
    passed = sum(1 for r in results if r)
    print(f"\n[INFO] 가림 게이트: {passed}/{len(results)} 통과")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
