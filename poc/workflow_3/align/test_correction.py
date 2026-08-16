"""align_fail_correct 합성 self-test — VLM/실장비 없이 primary/fallback 경로 검증.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/align/test_correction.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3.util.json_utils import bbox_center, bbox_to_pixels
from poc.workflow_3.align.correction import (
    PAUSED_SCALES,
    CorrectionConfig,
    CorrectionOutcome,
    _make_primary_demo,
    _with_key_ambiguity,
    correct_align_fail,
    key_visibility_gate,
)
from poc.workflow_3.align.cond_template import cond_align_offset
from poc.workflow_3.align.templates import _load_template
from poc.workflow_3.align.matching.engine import (
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    build_template,
    compute_align_key_score_ensemble,
)
from poc.workflow_3.align.live_search import LiveSearchConfig, clamp_to_fov, route_template
from poc.workflow_3.align.ok_button import locate_ok_button


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
                  distinctive: bool = True,
                  second_ratio: float | None = None) -> AlignKeyMatchResult:
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
        second_ratio=second_ratio,
    )


def test_gate() -> bool:
    """router: 임계 미지정 시 기존 bool 과 1:1(act/fallback_search), 임계 지정 + 만성 모호 시 engineer_review.

    Tier 0.1 — key_visibility_gate 가 bool → route intent 로 승격. reregister_ratio_threshold
    None(기본)이면 과거 동작 보존: True→"act", False→"fallback_search". 임계가 주어지고
    key 는 present(match/adjust+distinctive)이나 second_ratio>임계(만성 모호, AUC0.91 미스예측)면
    auto-act 대신 "engineer_review" — 평평한 surface 에서 확신 오정렬을 막는다.
    """
    R = key_visibility_gate
    checks = {
        # --- 임계 None: 기존 bool 의미 보존 (act↔과거 True, fallback_search↔과거 False) ---
        "match→act": R(_dummy_result("match")) == "act",
        "adjust(distinctive)→act": R(_dummy_result("adjust", distinctive=True)) == "act",
        "adjust(not distinctive)→fallback": R(_dummy_result("adjust", distinctive=False)) == "fallback_search",
        "low→fallback": R(_dummy_result("low")) == "fallback_search",
        "match(tiny-scale)→fallback": R(_dummy_result("match", scale=0.3)) == "fallback_search",
        # --- 임계 지정: present + 만성 모호(second_ratio>tau) → engineer_review ---
        "match+ambiguous→review": R(
            _dummy_result("match", second_ratio=0.99), reregister_ratio_threshold=0.98
        ) == "engineer_review",
        "adjust(distinctive)+ambiguous→review": R(
            _dummy_result("adjust", distinctive=True, second_ratio=0.99),
            reregister_ratio_threshold=0.98,
        ) == "engineer_review",
        # --- 임계 지정이라도 모호도 낮으면 act (정상 보정 경로 유지) ---
        "match+distinct→act": R(
            _dummy_result("match", second_ratio=0.80), reregister_ratio_threshold=0.98
        ) == "act",
        # --- not-present 면 임계와 무관하게 fallback (키가 없으면 탐색이지 review 아님) ---
        "low+threshold→fallback": R(
            _dummy_result("low", second_ratio=0.99), reregister_ratio_threshold=0.98
        ) == "fallback_search",
        # --- second_ratio None(후보 1개)이면 모호 판정 불가 → act ---
        "match+none_ratio→act": R(
            _dummy_result("match", second_ratio=None), reregister_ratio_threshold=0.98
        ) == "act",
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


def test_awaiting_engineer_ok() -> bool:
    """ok_click_enabled=False → reposition 은 하되 OK 는 안 누르고 awaiting_engineer_ok.

    corrected 로 끝나면 notify 가 cube 를 생략해 "OK 눌러달라"는 알림이 사라지므로,
    status 가 corrected 와 구분되는지까지 검증한다(침묵 회귀 방지).
    """
    monitor, templates = _make_primary_demo()
    fake = _FakeController(monitor.capture(), monitor.capture_screen(), mode="SEM")

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,
        config=CorrectionConfig(ok_click_enabled=False),
    )
    ok = (
        outcome.status == "awaiting_engineer_ok"
        and outcome.path == "primary"
        and len(fake.move_calls) == 1  # reposition 은 수행.
        and len(fake.screen_clicks) == 0  # OK 는 누르지 않음.
        and outcome.ok_screen_xy == (690, 560)  # 위치는 근거로 기록.
        and outcome.best_xy is not None
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] awaiting_ok: status={outcome.status} "
        f"moves={len(fake.move_calls)} clicks={len(fake.screen_clicks)} ok_xy={outcome.ok_screen_xy}"
    )
    return ok


def test_awaiting_engineer_ok_without_locator() -> bool:
    """OK 를 못 찾아도 반자동 모드는 escalated_no_ok 가 아니라 awaiting_engineer_ok.

    OK 클릭이 우리 일이 아닌 모드에서 'OK 미검출'은 실패가 아니다 - reposition 결과는
    유효하므로 엔지니어에게 같은 행동(확인 후 OK)을 요청한다.
    """
    monitor, templates = _make_primary_demo()
    fake = _FakeController(monitor.capture(), monitor.capture_screen(), mode="SEM")

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: None,
        dry_run=False,
        config=CorrectionConfig(ok_click_enabled=False, require_ok_button=True),
    )
    ok = (
        outcome.status == "awaiting_engineer_ok"
        and outcome.ok_screen_xy is None
        and len(fake.screen_clicks) == 0
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] awaiting_ok(no locator): status={outcome.status} "
        f"ok_xy={outcome.ok_screen_xy}"
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


def test_engineer_review_route() -> bool:
    """present 하나 만성 모호(second_ratio>tau) → primary 보류: escalated_ambiguous_key, 무액션·fallback 미진입.

    데모는 key 가 보이는(present) 입력이라 임계 None 이면 corrected 로 끝난다(test_primary_path).
    여기서는 실제 second_ratio 바로 아래로 reregister 임계를 낮춰 '만성 모호' 를 결정적으로
    강제하고, 게이트가 act 대신 engineer_review 로 라우팅해 actuation 을 막는지 검증한다.
    """
    monitor, templates = _make_primary_demo(key_in_view=True)
    frame = monitor.capture()
    screen = monitor.capture_screen()
    fake = _FakeController(frame, screen, mode="SEM")

    # 동일 입력으로 matcher 를 독립 재계산해 실제 second_ratio 를 얻는다(ensemble 결정적).
    template = route_template(templates, "SEM")
    expected = compute_align_key_score_ensemble(
        template, frame, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY
    )
    if expected.second_ratio is None:
        print("[PASS] engineer_review_route: demo second_ratio None(후보 1개) - 검증 스킵")
        return True
    thr = max(0.0, expected.second_ratio - 0.01)  # 실제값 바로 아래 → second_ratio>thr 로 모호 강제.

    outcome = correct_align_fail(
        fake,
        templates,
        ok_locator=lambda _s: (690, 560),
        dry_run=False,  # actuation 이 '일어나지 않아야' 함을 호출 횟수로 검증.
        config=CorrectionConfig(reregister_ratio_threshold=thr),
    )
    ok = (
        outcome.status == "escalated_ambiguous_key"
        and outcome.path == "primary"
        and outcome.key_decision in ("match", "adjust")
        and len(fake.move_calls) == 0       # reposition(더블클릭) 안 함.
        and len(fake.screen_clicks) == 0    # OK 클릭 안 함.
        and outcome.fallback is None        # fallback 탐색에도 진입 안 함(키는 보임).
        and outcome.second_ratio == expected.second_ratio  # 모호도 stamp 는 그대로 실린다.
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] engineer_review_route: status={outcome.status} "
        f"moves={len(fake.move_calls)} clicks={len(fake.screen_clicks)} "
        f"fallback={outcome.fallback} sr={outcome.second_ratio}"
    )
    return ok


def test_load_template_branches() -> bool:
    """_load_template 3분기: cond box -> box-crop+offset, cond 없음 -> center-crop+offset(0), flag off -> whole."""
    gray = np.full((512, 512), 110, dtype=np.uint8)
    cv2.rectangle(gray, (150, 200), (250, 300), 255, 1)  # box px (150,200)-(250,300).

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        img = root / "IMAP0001.png"
        cv2.imwrite(str(img), gray)
        # cond sidecar: .<name>/cond.txt; box idx6..9 = cursor(px x10), crosshair idx4,5 = -1.
        cond_dir = root / ".IMAP0001.png"
        cond_dir.mkdir()
        (cond_dir / "cond.txt").write_text(
            "Scope\tOM\nPixel\t512,512\n!Cursor_info\t0,0,0,0,-1,-1,1500,2000,2500,3000\n",
            encoding="utf-8",
        )
        box_ltrb = (1500, 2000, 2500, 3000)
        exp_offset = cond_align_offset(box_ltrb, gray.shape)  # off-center -> 비-(0,0).

        # 1) cond_box_crop=True + cond 존재 -> box-crop + decoupled offset.
        t_box = _load_template(img, recipe_id="R", key_type="om", cond_box_crop=True)
        box_ok = (
            t_box.align_offset_xy == exp_offset
            and exp_offset != (0, 0)
            and t_box.raw_image.shape == (96, 96)  # inset=2 on 100x100 cond box.
        )

        # 2) cond_box_crop=True + cond 없음(sidecar 미생성) -> center-area crop + offset(0).
        img2 = root / "IMAP0002.png"
        cv2.imwrite(str(img2), gray)
        t_center = _load_template(img2, recipe_id="R", key_type="sem", cond_box_crop=True)
        center_ok = (
            t_center.align_offset_xy == (0, 0)
            and t_center.raw_image.shape != (512, 512)
        )

        # 4) cond_box_crop=True + 작은 centered box -> check_cond_box "warn" 이지만 여전히 box-crop.
        img3 = root / "IMAP0003.png"
        cv2.imwrite(str(img3), gray)
        cond_dir3 = root / ".IMAP0003.png"
        cond_dir3.mkdir()
        # 22px centered box -> inner=22-2*CROP_INSET_PX(2)=18 in [16,24) -> warn:box:small. cursor=px*10.
        (cond_dir3 / "cond.txt").write_text(
            "Scope\tOM\nPixel\t512,512\n!Cursor_info\t0,0,0,0,-1,-1,2450,2450,2670,2670\n",
            encoding="utf-8",
        )
        t_warn = _load_template(img3, recipe_id="R", key_type="om", cond_box_crop=True)
        warn_ok = (
            t_warn.align_offset_xy == (0, 0)         # box 중심 == 이미지 중심.
            and t_warn.raw_image.shape == (18, 18)   # 22px box, inset 2 -> 18px crop (box-crop, not center).
        )

        # 3) cond_box_crop=False -> whole-template(구 동작) + offset(0).
        t_whole = _load_template(img, recipe_id="R", key_type="om", cond_box_crop=False)
        whole_ok = (
            t_whole.align_offset_xy == (0, 0)
            and t_whole.raw_image.shape == (512, 512)
        )

    ok = box_ok and center_ok and warn_ok and whole_ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] load_template_branches: "
        f"box(off={t_box.align_offset_xy},shape={t_box.raw_image.shape}) "
        f"center(off={t_center.align_offset_xy},shape={t_center.raw_image.shape}) "
        f"warn(off={t_warn.align_offset_xy},shape={t_warn.raw_image.shape}) "
        f"whole(shape={t_whole.raw_image.shape})"
    )
    return ok


def test_offset_applied_to_reposition() -> bool:
    """reposition 타깃 == clamp(best_xy + round(offset x best_scale)). scale=1·scale!=1·offset0."""
    import poc.workflow_3.align.correction as afc

    frame = np.zeros((600, 800, 3), dtype=np.uint8)  # fw=800, fh=600.
    screen = np.zeros((600, 800), dtype=np.uint8)
    margin = CorrectionConfig().click_margin_ratio

    def _controlled(best_xy, best_scale):
        return AlignKeyMatchResult(
            score=0.9, chamfer_score=0.9, orb_inlier_ratio=0.0,
            best_xy=best_xy, best_scale=best_scale, decision="match",
            debug_overlay=np.zeros((4, 4, 3), dtype=np.uint8), distinctive=True,
        )

    def _run(offset, best_xy, best_scale):
        tpl = build_template(np.full((32, 32), 120, dtype=np.uint8),
                             recipe_id="R", version="v0", key_type="sem",
                             align_offset_xy=offset)
        fake = _FakeController(frame, screen, mode="SEM")
        orig = afc.compute_align_key_score_ensemble
        afc.compute_align_key_score_ensemble = lambda *a, **k: _controlled(best_xy, best_scale)
        try:
            correct_align_fail(fake, {"SEM": tpl},
                               ok_locator=lambda _s: (10, 10), dry_run=False)
        finally:
            afc.compute_align_key_score_ensemble = orig
        return fake.move_calls

    moves1 = _run((40, -30), (400, 300), 1.0)   # (400,300)+(40,-30) = (440,270).
    exp1 = clamp_to_fov(440, 270, 800, 600, margin)
    moves2 = _run((40, -30), (400, 300), 2.0)   # +round((40,-30)*2) = (480,240).
    exp2 = clamp_to_fov(480, 240, 800, 600, margin)
    moves0 = _run((0, 0), (400, 300), 1.0)      # offset0 -> best_xy (regression guard).
    exp0 = clamp_to_fov(400, 300, 800, 600, margin)

    ok = moves1 == [exp1] and moves2 == [exp2] and moves0 == [exp0]
    print(
        f"[{'PASS' if ok else 'FAIL'}] offset_applied: "
        f"scale1={moves1}(exp{exp1}) scale2={moves2}(exp{exp2}) zero={moves0}(exp{exp0})"
    )
    return ok


def test_scale_pinned_flag_in_history() -> bool:
    """best_scale 가 PAUSED_SCALES 양끝 밴드에 박히면 history 에 scale_pinned=True.

    scale band(0.7~1.4)가 live box/template 실제 비율을 못 덮으면 매칭이 끝 밴드에
    고정된다 — 오피스 검증에서 이 신호로 밴드 커버리지를 판정한다(mis-scale 리스크).
    """
    import poc.workflow_3.align.correction as afc
    from poc.workflow_3.align.correction import PAUSED_SCALES

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    screen = np.zeros((600, 800), dtype=np.uint8)

    def _controlled(best_scale):
        return AlignKeyMatchResult(
            score=0.9, chamfer_score=0.9, orb_inlier_ratio=0.0,
            best_xy=(400, 300), best_scale=best_scale, decision="match",
            debug_overlay=np.zeros((4, 4, 3), dtype=np.uint8), distinctive=True,
        )

    def _run(best_scale):
        tpl = build_template(np.full((32, 32), 120, dtype=np.uint8),
                             recipe_id="R", version="v0", key_type="sem")
        fake = _FakeController(frame, screen, mode="SEM")
        orig = afc.compute_align_key_score_ensemble
        afc.compute_align_key_score_ensemble = lambda *a, **k: _controlled(best_scale)
        try:
            out = correct_align_fail(fake, {"SEM": tpl},
                                     ok_locator=lambda _s: (10, 10), dry_run=True)
        finally:
            afc.compute_align_key_score_ensemble = orig
        return out.history[0]

    hi = _run(max(PAUSED_SCALES))    # 1.4 — 상단 끝 밴드.
    lo = _run(min(PAUSED_SCALES))    # 0.7 — 하단 끝 밴드.
    mid = _run(1.0)                  # 내부 밴드.

    ok = (hi.get("scale_pinned") is True and lo.get("scale_pinned") is True
          and mid.get("scale_pinned") is False)
    print(
        f"[{'PASS' if ok else 'FAIL'}] scale_pinned_flag: "
        f"hi={hi.get('scale_pinned')} lo={lo.get('scale_pinned')} mid={mid.get('scale_pinned')}"
    )
    return ok


def test_correct_auto_uses_resolver() -> bool:
    """correct_align_fail_auto 가 resolve_templates 를 호출해 라우팅 dict 을 받는다."""
    import poc.workflow_3.align.correction as corr

    called = {"resolve": 0}

    class _A:
        eqp_id = "E1"; class_name = "c"; recipe_name = "r"; recipe_dir = "/x"

    def _resolve(assets, **kw):
        called["resolve"] += 1
        called["kw"] = kw
        from poc.workflow_3.align.matching.test_engine import make_synthetic_template
        from poc.workflow_3.align.matching.engine import build_template
        t = build_template(make_synthetic_template(key_type="box"),
                           recipe_id="r", version="v", key_type="sem")
        return {"SEM": t}

    orig_assets = corr.resolve_assets_auto
    orig_resolve = getattr(corr, "resolve_templates", None)
    corr.resolve_assets_auto = lambda **k: _A()
    corr.resolve_templates = _resolve
    try:
        monitor, _ = corr._make_primary_demo(key_in_view=True)
        # eqp_id="" (자동선택)로 호출 — resolver 는 assets.eqp_id("E1")를 받아야 한다(리뷰 가드).
        out = corr.correct_align_fail_auto(monitor, dry_run=True,
                                           eqp_id="", recipe_name="c/r")
        assert called["resolve"] == 1
        assert called["kw"]["eqp_id"] == "E1"        # assets.eqp_id 가 forwarded (빈 문자열 아님)
        assert called["kw"]["cond_box_crop"] is not None  # config pass-through 존재
        assert out.status in ("corrected", "escalated_no_ok", "ok_detect_error",
                              "fallback_corrected", "fallback_escalated",
                              "escalated_ambiguous_key")
    finally:
        corr.resolve_assets_auto = orig_assets
        if orig_resolve is not None:
            corr.resolve_templates = orig_resolve
        else:
            del corr.resolve_templates

    ok = called["resolve"] == 1
    print(
        f"[{'PASS' if ok else 'FAIL'}] correct_auto_uses_resolver: "
        f"resolve_calls={called['resolve']} status={out.status}"
    )
    return ok


def main() -> int:
    print("[INFO] align_fail_correct self-test 시작")
    results = [
        test_gate(),
        test_primary_path(),
        test_awaiting_engineer_ok(),
        test_awaiting_engineer_ok_without_locator(),
        test_fallback_path(),
        test_fallback_notify(),
        test_ok_detect_error(),
        test_ok_locator_mapping(),
        test_outcome_ambiguity_defaults(),
        test_with_key_ambiguity_stamps(),
        test_primary_path_stamps_ambiguity(),
        test_engineer_review_route(),
        test_load_template_branches(),
        test_offset_applied_to_reposition(),
        test_scale_pinned_flag_in_history(),
        test_correct_auto_uses_resolver(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
