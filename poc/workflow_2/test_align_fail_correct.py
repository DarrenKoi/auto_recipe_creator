"""align_fail_correct 합성 self-test — VLM/실장비 없이 primary/fallback 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_2/test_align_fail_correct.py
"""

import numpy as np

from poc.workflow_1.util.json_utils import bbox_1000_to_pixels, bbox_center
from poc.workflow_2.align_fail_correct import (
    CorrectionConfig,
    _make_primary_demo,
    correct_align_fail,
    key_visibility_gate,
)
from poc.workflow_2.align_key_matcher import AlignKeyMatchResult
from poc.workflow_2.live_align_search import LiveSearchConfig
from poc.workflow_2.test_align_key_match import make_wafer_background
from poc.workflow_2.vlm_ok_button_box import locate_ok_button


class _FakeController:
    """고정 frame/screen 을 돌려주고 actuation 호출을 기록하는 테스트용 controller."""

    def __init__(self, frame: np.ndarray, screen: np.ndarray, *, mode: str = "SEM") -> None:
        self.frame = frame
        self.screen = screen
        self.mode = mode
        self.move_calls: list[tuple[int, int]] = []
        self.zoom_calls: list[int] = []
        self.screen_clicks: list[tuple[int, int]] = []

    def capture(self) -> np.ndarray:
        return self.frame

    def capture_screen(self) -> np.ndarray:
        return self.screen

    def read_mode(self) -> str:
        return self.mode

    def move_to_point(self, fov_x: int, fov_y: int) -> None:
        self.move_calls.append((int(fov_x), int(fov_y)))

    def zoom(self, direction: int) -> None:
        self.zoom_calls.append(int(direction))

    def click_screen(self, screen_x: int, screen_y: int) -> None:
        self.screen_clicks.append((int(screen_x), int(screen_y)))


def _dummy_result(decision: str, *, orb: float = 0.0, scale: float = 1.0) -> AlignKeyMatchResult:
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    return AlignKeyMatchResult(
        score=0.5,
        chamfer_score=0.5,
        orb_inlier_ratio=orb,
        best_xy=(2, 2),
        best_scale=scale,
        decision=decision,
        debug_overlay=overlay,
    )


def test_gate() -> bool:
    """match→True; adjust→orb 보강 있을 때만 True; low/tiny-scale→False."""
    checks = {
        "match(orb=0)": key_visibility_gate(_dummy_result("match")) is True,
        "adjust(orb>0)": key_visibility_gate(_dummy_result("adjust", orb=0.3)) is True,
        "adjust(orb=0)": key_visibility_gate(_dummy_result("adjust", orb=0.0)) is False,
        "low": key_visibility_gate(_dummy_result("low")) is False,
        "match(tiny-scale)": key_visibility_gate(_dummy_result("match", scale=0.3)) is False,
    }
    ok = all(checks.values())
    print(f"[{'PASS' if ok else 'FAIL'}] gate: {checks}")
    return ok


def test_primary_path() -> bool:
    """key 가 보이는 프레임 → corrected/primary, move_to_point·click_screen 각각 1회."""
    monitor, templates = _make_primary_demo()
    frame = monitor.capture()
    screen = monitor.capture_screen()
    fake = _FakeController(frame, screen, mode="SEM")

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,  # actuation 호출 횟수를 검증하려면 dry_run=False.
        config=CorrectionConfig(require_ok_button=True),
    )
    ok = (
        outcome.status == "corrected"
        and outcome.path == "primary"
        and outcome.key_decision in ("match", "adjust")
        and len(fake.move_calls) == 1
        and len(fake.screen_clicks) == 1
        and outcome.ok_screen_xy == (690, 560)
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] primary: status={outcome.status} path={outcome.path} "
        f"moves={len(fake.move_calls)} clicks={len(fake.screen_clicks)} ok_xy={outcome.ok_screen_xy}"
    )
    return ok


def test_fallback_path() -> bool:
    """key 가 없는 featureless 프레임 → gate low → fallback(live_align_search) 위임."""
    monitor, templates = _make_primary_demo()
    featureless = make_wafer_background(frame_size=(512, 768))  # key 없음.
    screen = np.zeros((600, 800), dtype=np.uint8)
    fake = _FakeController(featureless, screen, mode="SEM")

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,
        fallback_config=LiveSearchConfig(pan_budget=3, initial_zoom_out_steps=1),
    )
    ok = (
        outcome.path == "fallback"
        and outcome.status.startswith("fallback_")
        and outcome.fallback is not None
        and len(fake.screen_clicks) == 0  # fallback 에선 OK 클릭 없음.
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] fallback: status={outcome.status} path={outcome.path} "
        f"clicks={len(fake.screen_clicks)}"
    )
    return ok


def test_ok_locator_mapping() -> bool:
    """fake client 의 relative_1000 bbox → 예상 screen 픽셀 중심으로 매핑."""
    frame = np.zeros((600, 800), dtype=np.uint8)
    bbox_1000 = {"left": 800, "top": 880, "right": 920, "bottom": 960}

    class _FakeResp:
        text = (
            '{"ok_button_visible": true, "coord_system": "relative_1000", '
            '"ok_button_bbox": {"left": 800, "top": 880, "right": 920, "bottom": 960}, '
            '"confidence": 0.9}'
        )

    class _FakeClient:
        def chat_with_image_b64(self, **_kwargs):
            return _FakeResp()

    got = locate_ok_button(frame_bgr=frame, client=_FakeClient())
    expected_center = bbox_center(bbox_1000_to_pixels(bbox_1000, 800, 600))
    expected = (expected_center["x"], expected_center["y"])
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] ok_locator mapping: got={got} expected={expected}")
    return ok


def main() -> int:
    print("[INFO] align_fail_correct self-test 시작")
    results = [
        test_gate(),
        test_primary_path(),
        test_fallback_path(),
        test_ok_locator_mapping(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
