"""캡처 직전 가림 대기 게이트 self-test — 실장비/VLM 없이 Mac 에서 돈다.

접속 요청("Information") 팝업이 3초간 tool 창을 가리는 동안 캡처하면 팝업 픽셀이
프레임에 들어오고, 보정은 그 프레임에서 좌표를 뽑는다. 이 테스트는 게이트가
(1) 가림이 걷힐 때까지 기다렸다가 캡처하고 (2) 판정 불가/미주입이면 무해하게
비켜서며 (3) 예산을 넘겨도 예외 대신 캡처로 진행하는지를 확인한다.

    uv run python poc/workflow_3/sem_monitor/test_occlusion_gate.py
"""

import time
from pathlib import Path

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


def test_budget_exhausted_fails_loud_when_live() -> bool:
    """실행 중(action_enabled)이면 예산 소진 시 캡처하지 않고 크게 실패한다.

    매칭 점수는 backstop 이 아니다 - 낮은 점수는 fallback_search 로 가고 그 경로는
    스테이지를 실제로 움직인다. '못 보면 가만히' 가 아니라 '못 보면 움직인다'가
    되므로 오염 프레임을 좌표 근거로 쓰면 안 된다(codex 리뷰 2026-09-16).
    """
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.05
    try:
        mon, _ = _monitor(["full"], action_enabled=True)
        try:
            _with_capture(mon.capture_screen)
            ok = False
            print("[FAIL] 영구 가림인데 캡처가 통과했다")
        except RuntimeError as exc:
            ok = "가림" in str(exc)
            print(f"[{'PASS' if ok else 'FAIL'}] 실행 중 영구 가림은 예외: {exc}")
        return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_budget_exhausted_permissive_in_dry_run() -> bool:
    """dry-run/SAFE_MODE 는 좌표 로그만 남기므로 종전대로 캡처한다."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.05
    try:
        mon, _ = _monitor(["full"], action_enabled=False)
        frame = _with_capture(mon.capture_screen)
        ok = frame.shape == (30, 40)
        print(f"[{'PASS' if ok else 'FAIL'}] dry-run 은 캡처 진행: shape={frame.shape}")
        return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_panel_discovery_blocked_when_occluded() -> bool:
    """panel 탐색 캡처도 같은 게이트를 지난다 - 캐시되는 ROI/modality 를 오염시키지 않게."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.05
    captured = []
    orig_capture = ctrl.capture_window
    ctrl.capture_window = lambda _w: captured.append(1) or np.zeros((30, 40, 3), np.uint8)
    reasons: list = []
    try:
        built = ctrl.build_rcs_sem_monitor(
            object(), occlusion_fn=lambda: "full", reason_sink=reasons,
        )
        ok = built is None and not captured and reasons == ["occluded_full"]
        print(
            f"[{'PASS' if ok else 'FAIL'}] 가림 중 panel 탐색 차단: "
            f"built={built}, captures={len(captured)}, reasons={reasons}"
        )
        return ok
    finally:
        ctrl.capture_window = orig_capture
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


def test_heartbeat_reports_clear_state() -> bool:
    """가림이 없어도 heartbeat 로 판정값을 찍는다 - 침묵이 두 뜻이 되지 않게."""
    orig = ctrl.OCCLUSION_LOG_SEC
    ctrl.OCCLUSION_LOG_SEC = 0.02
    lines = []
    import builtins

    real_print = builtins.print
    builtins.print = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
    try:
        mon, _ = _monitor(["none"])
        mon._wait_unoccluded()          # 첫 호출 - 상태 변화라 찍힌다
        first = len([l for l in lines if "가림 판정=" in l])
        mon._wait_unoccluded()          # 곧바로 다시 - throttle 로 안 찍힌다
        second = len([l for l in lines if "가림 판정=" in l])
        time.sleep(0.03)
        mon._wait_unoccluded()          # heartbeat 간격 경과 - 다시 찍힌다
        third = len([l for l in lines if "가림 판정=" in l])
    finally:
        builtins.print = real_print
        ctrl.OCCLUSION_LOG_SEC = orig
    ok = first == 1 and second == 1 and third == 2
    print(f"[{'PASS' if ok else 'FAIL'}] heartbeat: 변화={first}, throttle={second}, 재출력={third}")
    return ok


def test_unknown_after_occlusion_is_not_clearance() -> bool:
    """full -> unknown 은 해소가 아니다 - 판정기를 잃은 것과 화면이 깨끗한 것은 다르다."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.06
    try:
        mon, _ = _monitor(["full", "unknown"], action_enabled=True)
        try:
            _with_capture(mon.capture_screen)
            print("[FAIL] full -> unknown 인데 캡처가 통과했다")
            return False
        except RuntimeError as exc:
            ok = "full" in str(exc)
            print(f"[{'PASS' if ok else 'FAIL'}] full -> unknown 은 계속 막힘: {exc}")
            return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_probe_exception_after_occlusion_is_not_clearance() -> bool:
    """가림을 본 뒤 판정자가 던지면 통과가 아니다(예외 = 증거 상실)."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.06
    calls = []

    def _probe():
        calls.append(1)
        if len(calls) == 1:
            return "full"
        raise RuntimeError("probe lost")

    try:
        mon = ctrl.RCSSEMMonitor(
            object(), _FakePanel(), occlusion_fn=_probe, action_enabled=True
        )
        try:
            _with_capture(mon.capture_screen)
            print("[FAIL] 가림 후 판정 예외인데 캡처가 통과했다")
            return False
        except RuntimeError as exc:
            ok = "가림" in str(exc)
            print(f"[{'PASS' if ok else 'FAIL'}] 가림 후 판정 예외도 막힘")
            return ok
    finally:
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def test_landmark_fallback_reprobes() -> bool:
    """VLM 실패 후 landmark 캡처 전에 다시 본다 - VLM 왕복 중에 뜬 팝업을 놓치지 않게."""
    orig_poll, orig_wait = ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC
    ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = 0.01, 0.05
    states = ["none", "full", "full", "full", "full", "full", "full", "full"]
    captured = []
    orig_capture = ctrl.capture_window
    ctrl.capture_window = lambda _w: captured.append(1) or np.zeros((30, 40, 3), np.uint8)
    reasons: list = []
    try:
        built = ctrl.build_rcs_sem_monitor(
            object(),
            occlusion_fn=lambda: states.pop(0) if len(states) > 1 else states[0],
            reason_sink=reasons,
            landmarks_dir=Path("/nonexistent"),
        )
        # landmark 디렉터리가 없으면 landmark_missing 으로 먼저 끝난다 - 그 경로는
        # 캡처를 안 하므로, 여기서 확인할 것은 '첫 게이트를 통과했다' 와
        # '가림 상태로 캡처하지 않았다' 두 가지다.
        ok = built is None and not captured
        print(f"[{'PASS' if ok else 'FAIL'}] landmark 폴백 재검사: captures={len(captured)}, reasons={reasons}")
        return ok
    finally:
        ctrl.capture_window = orig_capture
        ctrl.OCCLUSION_POLL_SEC, ctrl.OCCLUSION_WAIT_SEC = orig_poll, orig_wait


def main() -> int:
    tests = [
        test_waits_until_occlusion_clears,
        test_unknown_does_not_block,
        test_no_probe_is_noop,
        test_budget_exhausted_fails_loud_when_live,
        test_budget_exhausted_permissive_in_dry_run,
        test_panel_discovery_blocked_when_occluded,
        test_probe_exception_does_not_break_capture,
        test_wait_sec_zero_disables,
        test_heartbeat_reports_clear_state,
        test_unknown_after_occlusion_is_not_clearance,
        test_probe_exception_after_occlusion_is_not_clearance,
        test_landmark_fallback_reprobes,
    ]
    results = [t() for t in tests]
    passed = sum(1 for r in results if r)
    print(f"\n[INFO] 가림 게이트: {passed}/{len(results)} 통과")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
