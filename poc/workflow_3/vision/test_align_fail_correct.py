"""align_fail_correct 합성 self-test — VLM/실장비 없이 primary/fallback 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_2/test_align_fail_correct.py
"""

import numpy as np

from poc.workflow_3.util.json_utils import bbox_center, bbox_to_pixels
from poc.workflow_3.vision.align_fail_correct import (
    PAUSED_SCALES,
    CorrectionConfig,
    CorrectionOutcome,
    _make_primary_demo,
    _with_key_ambiguity,
    correct_align_fail,
    key_visibility_gate,
)
from poc.workflow_3.vision.align_key_matcher import (
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    compute_align_key_score_ensemble,
)
from poc.workflow_3.vision.live_align_search import LiveSearchConfig, route_template
from poc.workflow_3.vision.vlm_ok_button_box import locate_ok_button


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


def _dummy_result(decision: str, *, orb: float = 0.0, scale: float = 1.0,
                  distinctive: bool = True) -> AlignKeyMatchResult:
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    return AlignKeyMatchResult(
        score=0.5,
        chamfer_score=0.5,
        orb_inlier_ratio=orb,
        best_xy=(2, 2),
        best_scale=scale,
        decision=decision,
        debug_overlay=overlay,
        distinctive=distinctive,
    )


def test_gate() -> bool:
    """match→True; adjust→distinctive 일 때만 True(ensemble: orb 폐지); low/tiny-scale→False."""
    checks = {
        "match": key_visibility_gate(_dummy_result("match")) is True,
        "adjust(distinctive)": key_visibility_gate(_dummy_result("adjust", distinctive=True)) is True,
        "adjust(not distinctive)": key_visibility_gate(_dummy_result("adjust", distinctive=False)) is False,
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
    """key 없음 → gate low → fallback 위임. stateful mock 으로 실제 pan/zoom 전이를 exercise(#7)."""
    # key_in_view=False → featureless. _MockSEMMonitor 는 stateful(pan 하면 capture 변화).
    monitor, templates = _make_primary_demo(key_in_view=False)

    outcome = correct_align_fail(
        monitor,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,
        fallback_config=LiveSearchConfig(pan_budget=6, initial_zoom_out_steps=1),
    )
    ok = (
        outcome.path == "fallback"
        and outcome.status.startswith("fallback_")
        and outcome.fallback is not None
        and outcome.fallback.pan_count > 0  # 정지 프레임이 아니라 실제 pan 전이를 돌았다.
        and len(monitor.screen_clicks) == 0  # fallback 에선 OK 클릭 없음.
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] fallback: status={outcome.status} path={outcome.path} "
        f"pan_count={outcome.fallback.pan_count if outcome.fallback else '-'} "
        f"clicks={len(monitor.screen_clicks)}"
    )
    return ok


def test_fallback_notify() -> bool:
    """notify_fn 이 fallback escalation 으로 그대로 전달·발화되는지 검증(#6).

    순수 검정(edge 없는) 프레임은 매 iteration score≈0(low)이라 low_streak 가 단조 증가 →
    low_streak_limit 에서 escalation + notify 가 결정적으로 발생한다.
    """
    monitor, templates = _make_primary_demo(key_in_view=True)
    black = np.zeros((512, 768), dtype=np.uint8)  # edge 없음 → 항상 low.
    screen = np.zeros((600, 800), dtype=np.uint8)
    fake = _FakeController(black, screen, mode="SEM")

    notified: list = []

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,
        notify_fn=lambda state, recent: notified.append(state),
        fallback_config=LiveSearchConfig(
            pan_budget=20, low_streak_limit=3, initial_zoom_out_steps=1
        ),
    )
    ok = (
        outcome.status == "fallback_escalated"
        and len(notified) >= 1  # notify_fn 이 escalation 으로 전달·발화됨.
        and len(fake.screen_clicks) == 0
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] fallback_notify: status={outcome.status} "
        f"notified={len(notified)}"
    )
    return ok


def test_ok_detect_error() -> bool:
    """OK locator 가 예외를 던지면 'ok_detect_error'(+error 기록)로 surface, escalate 와 구분(#4)."""
    monitor, templates = _make_primary_demo(key_in_view=True)

    def _boom(_screen):
        raise RuntimeError("VLM 연결 실패")

    outcome = correct_align_fail(
        monitor,
        templates,
        ok_locator=_boom,
        dry_run=False,
    )
    ok = (
        outcome.status == "ok_detect_error"
        and outcome.path == "primary"
        and outcome.error is not None
        and "RuntimeError" in outcome.error
        and outcome.ok_screen_xy is None
        and len(monitor.screen_clicks) == 0  # 에러면 OK 클릭 안 함.
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] ok_detect_error: status={outcome.status} "
        f"error={outcome.error!r}"
    )
    return ok


def _fake_client(payload_json: str):
    class _FakeResp:
        text = payload_json

    class _FakeClient:
        def chat_with_image_b64(self, **_kwargs):
            return _FakeResp()

    return _FakeClient()


def test_ok_locator_mapping() -> bool:
    """relative_1000 / pixel 두 coord_system 을 각각 올바른 screen 픽셀 중심으로 매핑."""
    frame = np.zeros((600, 800), dtype=np.uint8)  # (h, w) = (600, 800)

    # 1) relative_1000 — 정규화 좌표.
    rel_bbox = {"left": 800, "top": 880, "right": 920, "bottom": 960}
    got_rel = locate_ok_button(
        frame_bgr=frame,
        client=_fake_client(
            '{"ok_button_visible": true, "coord_system": "relative_1000", '
            '"ok_button_bbox": {"left": 800, "top": 880, "right": 920, "bottom": 960}, '
            '"confidence": 0.9}'
        ),
    )
    exp_rel = bbox_center(bbox_to_pixels(rel_bbox, 800, 600, "relative_1000"))
    ok_rel = got_rel == (exp_rel["x"], exp_rel["y"])

    # 2) pixel — 모델이 절대 픽셀로 응답(fix #2: /1000 로 잘못 스케일하지 않아야).
    px_bbox = {"left": 640, "top": 540, "right": 740, "bottom": 580}
    got_px = locate_ok_button(
        frame_bgr=frame,
        client=_fake_client(
            '{"ok_button_visible": true, "coord_system": "pixel", '
            '"ok_button_bbox": {"left": 640, "top": 540, "right": 740, "bottom": 580}, '
            '"confidence": 0.9}'
        ),
    )
    exp_px = bbox_center(bbox_to_pixels(px_bbox, 800, 600, "pixel"))
    # pixel 경로는 ~(689, 559) 근처여야 한다(상단 1/10 이 아니라 실제 버튼 위치).
    ok_px = got_px == (exp_px["x"], exp_px["y"]) and got_px[0] > 600 and got_px[1] > 500

    ok = ok_rel and ok_px
    print(
        f"[{'PASS' if ok else 'FAIL'}] ok_locator mapping: "
        f"rel got={got_rel} exp=({exp_rel['x']},{exp_rel['y']}) | "
        f"pixel got={got_px} exp=({exp_px['x']},{exp_px['y']})"
    )
    return ok


def test_outcome_ambiguity_defaults() -> bool:
    """신규 모호도 필드는 전부 기본값 보유 → 기존 positional 생성부(no_assets 등) 무변경 통과."""
    o = CorrectionOutcome("no_assets", "primary", "low", None, None, None)
    ok = o.second_ratio is None and o.score_gap is None and o.distinctive is True
    print(
        f"[{'PASS' if ok else 'FAIL'}] outcome_ambiguity_defaults: "
        f"second_ratio={o.second_ratio} score_gap={o.score_gap} distinctive={o.distinctive}"
    )
    return ok


def test_with_key_ambiguity_stamps() -> bool:
    """_with_key_ambiguity 가 result 의 second_ratio/score_gap/distinctive 를 stamp(기존 필드 보존)."""
    base = CorrectionOutcome("corrected", "primary", "match", (1, 2), (3, 4), None)
    result = _dummy_result("match", distinctive=False)
    result.second_ratio = 0.991
    result.score_gap = 0.005
    stamped = _with_key_ambiguity(base, result)
    ok = (
        stamped.second_ratio == 0.991
        and stamped.score_gap == 0.005
        and stamped.distinctive is False
        # 기존 필드 보존.
        and stamped.status == "corrected"
        and stamped.best_xy == (1, 2)
        and stamped.ok_screen_xy == (3, 4)
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] with_key_ambiguity_stamps: "
        f"second_ratio={stamped.second_ratio} distinctive={stamped.distinctive} status={stamped.status}"
    )
    return ok


def test_primary_path_stamps_ambiguity() -> bool:
    """primary 경로 outcome 이 matcher 가 실제로 낸 모호도 값을 그대로 싣는다(독립 재계산과 일치)."""
    monitor, templates = _make_primary_demo()
    frame = monitor.capture()
    screen = monitor.capture_screen()
    fake = _FakeController(frame, screen, mode="SEM")

    # 동일 입력으로 matcher 를 독립 재계산해 기대값을 얻는다(ensemble 은 결정적).
    template = route_template(templates, "SEM")
    expected = compute_align_key_score_ensemble(
        template, frame, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY
    )

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,
        config=CorrectionConfig(require_ok_button=True),
    )
    ok = (
        outcome.status == "corrected"
        and outcome.second_ratio == expected.second_ratio  # None==None 또는 float 정확 일치.
        and outcome.score_gap == expected.score_gap
        and outcome.distinctive == expected.distinctive
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] primary_stamps_ambiguity: "
        f"second_ratio={outcome.second_ratio} (exp={expected.second_ratio}) "
        f"distinctive={outcome.distinctive}"
    )
    return ok


def main() -> int:
    print("[INFO] align_fail_correct self-test 시작")
    results = [
        test_gate(),
        test_primary_path(),
        test_fallback_path(),
        test_fallback_notify(),
        test_ok_detect_error(),
        test_ok_locator_mapping(),
        test_outcome_ambiguity_defaults(),
        test_with_key_ambiguity_stamps(),
        test_primary_path_stamps_ambiguity(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
