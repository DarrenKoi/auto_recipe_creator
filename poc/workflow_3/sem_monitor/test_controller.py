"""RCSSEMMonitor 조립 self-test — live SEM box 경로 / landmark 폴백 / modality 우선순위.

실장비·VLM 없이 Mac 에서 그대로 돈다(capture_window 와 detect_sem_box 를 대역으로 교체).
CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print.

    uv run python poc/workflow_3/sem_monitor/test_controller.py
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poc.workflow_3.sem_monitor import controller as ctrl
from poc.workflow_3.sem_monitor import sem_box_detect as sbd


@dataclass
class _FakeDetection:
    """detect_sem_box 반환값 중 build 경로가 읽는 필드만 흉내낸다."""

    detected: bool
    bbox_px: dict | None
    pm_mode: str | None
    confidence: float | None = 0.9


class _Patched:
    """capture_window / detect_sem_box 를 컨텍스트 동안 대역으로 바꾼다."""

    def __init__(self, detection):
        self.detection = detection
        self.calls: list[dict] = []

    def __enter__(self):
        self._orig_capture = ctrl.capture_window
        self._orig_detect = sbd.detect_sem_box
        ctrl.capture_window = lambda _win: np.zeros((600, 800, 3), dtype=np.uint8)

        def _fake_detect(_image, _client, **kwargs):
            self.calls.append(kwargs)
            if isinstance(self.detection, Exception):
                raise self.detection
            return self.detection

        sbd.detect_sem_box = _fake_detect
        return self

    def __exit__(self, *_exc):
        ctrl.capture_window = self._orig_capture
        sbd.detect_sem_box = self._orig_detect
        return False


# 존재하지 않는 landmark 디렉터리 — VLM 실패 시 폴백이 None 으로 끝나는지 보기 위함.
_EMPTY_LANDMARKS = Path("/nonexistent/sem_panel_landmarks")


def test_vlm_box_becomes_panel_roi() -> bool:
    """live SEM box(l,t,r,b) → panel_roi(x,y,w,h) 변환 + pm_mode 주입."""
    detection = _FakeDetection(
        detected=True,
        bbox_px={"left": 100, "top": 50, "right": 420, "bottom": 290},
        pm_mode="OM",
    )
    with _Patched(detection):
        monitor = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, pm_two_stage=True,
            vlm_client=object(),
        )
    ok = (
        monitor is not None
        and monitor.panel.panel_roi == (100, 50, 320, 240)
        and monitor.panel.model_id == "vlm_live_box"
        and monitor.read_mode() == "OM"
    )
    roi = monitor.panel.panel_roi if monitor else None
    print(f"[{'PASS' if ok else 'FAIL'}] vlm_box_roi: roi={roi} mode={monitor.read_mode() if monitor else '-'}")
    return ok


def test_two_stage_flag_forwarded() -> bool:
    """pm_two_stage 가 detect_sem_box 의 two_stage 로 전달되는지(PM 재판독 옵션 유실 방지)."""
    detection = _FakeDetection(True, {"left": 0, "top": 0, "right": 10, "bottom": 10}, "SEM")
    with _Patched(detection) as patched:
        ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), pm_two_stage=True,
        )
    ok = bool(patched.calls) and patched.calls[0].get("two_stage") is True
    print(f"[{'PASS' if ok else 'FAIL'}] two_stage_forwarded: calls={patched.calls}")
    return ok


def test_undetected_falls_back_and_fails_clean() -> bool:
    """VLM 미검출 → landmark 폴백 → landmark 도 없으면 None(panel_not_found 로 이어짐)."""
    detection = _FakeDetection(detected=False, bbox_px=None, pm_mode=None)
    with _Patched(detection):
        monitor = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(),
        )
    ok = monitor is None
    print(f"[{'PASS' if ok else 'FAIL'}] undetected_fallback: monitor={monitor}")
    return ok


def test_vlm_exception_falls_back() -> bool:
    """detect_sem_box 예외가 사이클을 죽이지 않고 폴백 경로로 내려가는지."""
    with _Patched(RuntimeError("flask down")):
        monitor = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(),
        )
    ok = monitor is None  # 폴백도 비어 있으므로 None, 단 예외는 새어 나오지 않아야 한다.
    print(f"[{'PASS' if ok else 'FAIL'}] vlm_exception_fallback: monitor={monitor}")
    return ok


def test_degenerate_bbox_rejected() -> bool:
    """right<=left 같은 이상 박스는 채택하지 않는다(0 크기 ROI 로 capture 가 터지는 것 방지)."""
    detection = _FakeDetection(True, {"left": 200, "top": 50, "right": 200, "bottom": 290}, "SEM")
    with _Patched(detection):
        monitor = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(),
        )
    ok = monitor is None
    print(f"[{'PASS' if ok else 'FAIL'}] degenerate_bbox: monitor={monitor}")
    return ok


def test_mode_priority() -> bool:
    """read_mode 우선순위: env override > mode_hint > mode_default."""
    detection = _FakeDetection(True, {"left": 0, "top": 0, "right": 10, "bottom": 10}, "OM")
    with _Patched(detection):
        monitor = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), mode_default="SEM",
        )
    hint_wins = monitor.read_mode() == "OM"

    os.environ["ALIGN_SEM_MODE_OVERRIDE"] = "sem"
    try:
        override_wins = monitor.read_mode() == "SEM"  # 대소문자 정규화 포함.
    finally:
        del os.environ["ALIGN_SEM_MODE_OVERRIDE"]

    # pm_mode 판독 실패(None) 면 mode_default 로 떨어진다.
    detection_nomode = _FakeDetection(True, {"left": 0, "top": 0, "right": 10, "bottom": 10}, None)
    with _Patched(detection_nomode):
        bare = ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), mode_default="SEM",
        )
    default_used = bare.mode_hint is None and bare.read_mode() == "SEM"

    ok = hint_wins and override_wins and default_used
    print(
        f"[{'PASS' if ok else 'FAIL'}] mode_priority: hint={hint_wins} "
        f"override={override_wins} default={default_used}"
    )
    return ok


def test_no_vlm_client_uses_landmarks_only() -> bool:
    """vlm_client 미주입이면 detect_sem_box 를 아예 부르지 않는다(개발 PC 경로)."""
    detection = _FakeDetection(True, {"left": 0, "top": 0, "right": 10, "bottom": 10}, "SEM")
    with _Patched(detection) as patched:
        monitor = ctrl.build_rcs_sem_monitor(object(), landmarks_dir=_EMPTY_LANDMARKS)
    ok = monitor is None and patched.calls == []
    print(f"[{'PASS' if ok else 'FAIL'}] no_vlm_client: calls={patched.calls} monitor={monitor}")
    return ok


def test_reason_sink_names_the_specific_failure() -> bool:
    """panel_not_found 의 4가지 원인이 서로 구분되어야 한다.

    이 값이 step 저널의 error_message 가 된다. 정적 문자열 하나로 뭉개면
    "VLM 클라이언트가 안 만들어졌다" 와 "VLM 이 보고 SEM box 가 없었다" 를
    사후에 구분할 수 없다 - 원인도 고칠 곳도 전혀 다른 두 버그다.
    """
    cases = []

    # (1) vlm_client 미주입 - 지금은 조용히 None 을 돌려주는 유일한 경로.
    with _Patched(_FakeDetection(True, {"left": 0, "top": 0, "right": 9, "bottom": 9}, "SEM")):
        sink = []
        ctrl.build_rcs_sem_monitor(object(), landmarks_dir=_EMPTY_LANDMARKS, reason_sink=sink)
        cases.append(("vlm_client_missing", sink))

    # (2) detect_sem_box 예외.
    with _Patched(RuntimeError("flask down")):
        sink = []
        ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), reason_sink=sink
        )
        cases.append(("sem_box_detect_error", sink))

    # (3) VLM 이 보긴 했는데 박스가 없다 - 점유 view-only 화면에서 실제로 나올 수 있다.
    with _Patched(_FakeDetection(detected=False, bbox_px=None, pm_mode=None)):
        sink = []
        ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), reason_sink=sink
        )
        cases.append(("sem_box_not_detected", sink))

    # (4) 박스는 왔는데 크기가 이상하다.
    with _Patched(_FakeDetection(True, {"left": 50, "top": 50, "right": 50, "bottom": 10}, "SEM")):
        sink = []
        ctrl.build_rcs_sem_monitor(
            object(), landmarks_dir=_EMPTY_LANDMARKS, vlm_client=object(), reason_sink=sink
        )
        cases.append(("sem_box_degenerate", sink))

    ok = all(sink and expected in sink[0] for expected, sink in cases)
    # landmark 도 비었다는 사실이 함께 남아야 "폴백이 있었는데 실패" 와 구분된다.
    ok = ok and all(sink and "landmark_missing" in sink[0] for _, sink in cases)
    print(f"[{'PASS' if ok else 'FAIL'}] reason_sink: {[s[0] if s else None for _, s in cases]}")
    return ok


def main() -> int:
    print("[INFO] RCSSEMMonitor 조립 self-test 시작")
    results = [
        test_vlm_box_becomes_panel_roi(),
        test_two_stage_flag_forwarded(),
        test_undetected_falls_back_and_fails_clean(),
        test_vlm_exception_falls_back(),
        test_degenerate_bbox_rejected(),
        test_mode_priority(),
        test_no_vlm_client_uses_landmarks_only(),
        test_reason_sink_names_the_specific_failure(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
