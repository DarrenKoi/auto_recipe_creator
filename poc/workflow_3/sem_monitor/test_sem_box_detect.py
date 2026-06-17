"""sem_box_detect + feasibility 통합(모드 선택/좌표 shift/overlay) 스모크 테스트.

VLM 없이 도는 단위 테스트:
  * pm_text_to_mode  — PM 텍스트 → OM/SEM/None 매핑(사용자 규칙 104/210→OM, K→SEM).
  * CV 함수(grey_frame_mask / snap_box_to_edges / sharpness_in_box / bbox_px_to_1000).
  * mark_align_feasibility — detect_sem_box 를 모킹해 (a) PM=OM 이면 OM template 선택 +
    box 원점만큼 align point 풀프레임 환산, (b) 검출 실패 시 전체 창 매칭 폴백을 검증.

`uv run python poc/workflow_3/sem_monitor/test_sem_box_detect.py` 로 직접 실행.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from PIL import Image

from poc.workflow_3.sem_monitor import sem_box_detect as sbd
from poc.workflow_3.sem_monitor.sem_box_detect import (
    SemBoxDetection,
    bbox_px_to_1000,
    crop_pm_region,
    grey_frame_mask,
    pm_text_to_mode,
    sharpness_in_box,
    snap_box_to_edges,
)
from poc.workflow_3.align.diagnostics import feasibility_check as fc


# ------------------------------------------------------------------
# 1) pm_text_to_mode.
# ------------------------------------------------------------------


def test_pm_text_to_mode():
    cases = {
        "104": "OM",
        "210": "OM",
        " 104 ": "OM",      # whitespace 무시.
        "30K": "SEM",
        "5k": "SEM",        # 소문자 k.
        "100K": "SEM",
        "20 K": "SEM",      # 숫자-K 사이 공백.
        "10 k": "SEM",      # 공백 + 소문자.
        "210K": "SEM",      # 숫자+K 는 OM 값이어도 SEM(진짜 고배율 readout).
        "": None,
        "   ": None,
        "abc": None,
        "999": None,        # OM 값 집합 밖.
        "PARK": None,       # 글자-only K → SEM 오판 금지.
        "OK": None,         # 글자-only K → None.
        "PM104": None,      # 라벨 누수(글자 섞임) → 거부, 안전 폴백.
        "104.5": None,      # 순수 숫자 아님 → None.
    }
    for text, expected in cases.items():
        got = pm_text_to_mode(text)
        assert got == expected, f"pm_text_to_mode({text!r})={got!r} != {expected!r}"
    assert pm_text_to_mode(None) is None
    # 비문자열 입력(VLM 이 숫자/리스트 반환) 방어 — 크래시 없이 None.
    assert pm_text_to_mode(104) is None
    assert pm_text_to_mode(["2", "1", "0"]) is None
    assert pm_text_to_mode(True) is None
    print("[OK] test_pm_text_to_mode")


# ------------------------------------------------------------------
# 2) CV 함수.
# ------------------------------------------------------------------


def test_grey_frame_mask():
    bgr = np.zeros((4, 3, 3), dtype=np.uint8)
    bgr[0, :] = (180, 180, 180)   # 밝은 무채색 회색 → 1.
    bgr[1, :] = (180, 60, 60)     # 유채색 → 0.
    bgr[2, :] = (20, 20, 20)      # 어두움 → 0.
    bgr[3, :] = (185, 178, 188)   # band 안 + 작은 채도 → 1.
    mask = grey_frame_mask(bgr)
    assert mask[0].all() and not mask[1].any() and not mask[2].any() and mask[3].all()
    print("[OK] test_grey_frame_mask")


def test_snap_box_to_edges():
    # 회색 테두리 사각형(120,120,120 위에 (180) 프레임). VLM bbox 를 약간 빗나가게 주면
    # band 안 회색 직선으로 snap 되어 실제 테두리(20,20,80,60)에 가까워져야 한다.
    gray = np.full((80, 100), 120, dtype=np.uint8)
    true_box = {"left": 20, "top": 20, "right": 80, "bottom": 60}
    gray[true_box["top"], true_box["left"]:true_box["right"]] = 180
    gray[true_box["bottom"], true_box["left"]:true_box["right"]] = 180
    gray[true_box["top"]:true_box["bottom"], true_box["left"]] = 180
    gray[true_box["top"]:true_box["bottom"], true_box["right"]] = 180
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = grey_frame_mask(bgr)
    vlm_box = {"left": 23, "top": 23, "right": 77, "bottom": 57}   # 안쪽으로 빗나감.
    snapped = snap_box_to_edges(gray, mask, vlm_box)
    for side in ("left", "top", "right", "bottom"):
        assert abs(snapped[side] - true_box[side]) <= 2, f"{side}: {snapped[side]} vs {true_box[side]}"
    print("[OK] test_snap_box_to_edges")


def test_sharpness_in_box():
    flat = np.full((40, 40), 128, dtype=np.uint8)
    rng = np.random.RandomState(0)
    noisy = rng.randint(0, 256, size=(40, 40)).astype(np.uint8)
    box = {"left": 0, "top": 0, "right": 40, "bottom": 40}
    assert sharpness_in_box(flat, box) < 1.0
    assert sharpness_in_box(noisy, box) > 100.0
    print("[OK] test_sharpness_in_box")


def test_bbox_px_to_1000():
    box = {"left": 50, "top": 25, "right": 150, "bottom": 75}
    out = bbox_px_to_1000(box, width=200, height=100)
    assert out == {"left": 250, "top": 250, "right": 750, "bottom": 750}, out
    print("[OK] test_bbox_px_to_1000")


# ------------------------------------------------------------------
# 2b) PM 2단계(crop + OCR) — locate(단일 호출)→crop→OCR.
# ------------------------------------------------------------------


def test_crop_pm_region():
    img = Image.new("RGB", (300, 200))
    box = {"left": 100, "top": 100, "right": 140, "bottom": 120}   # 40x20.
    crop = crop_pm_region(img, box, pad_ratio=0.5)                 # pad 20x10.
    # 좌우 각 20px, 상하 각 10px 패딩 → 80x40, 프레임 안.
    assert crop is not None and crop.size == (80, 40), crop.size
    # 프레임 경계에서 잘리는 경우(clamp).
    edge = crop_pm_region(img, {"left": 0, "top": 0, "right": 10, "bottom": 10}, pad_ratio=0.5)
    assert edge is not None and edge.size[0] <= 300 and edge.size[1] <= 200
    assert crop_pm_region(img, None) is None
    print("[OK] test_crop_pm_region")


def _install_detect_stubs(monkey_state, *, payload, panel_bbox, ocr_text):
    """detect_sem_box 의 VLM/OCR 의존성을 stub 으로 교체."""
    def fake_run(*, image_b64, width, height, client):
        return payload, panel_bbox

    def fake_ocr(crop, ocr_client):
        return ocr_text

    for name, fn in [("_run_sem_box_detection", fake_run), ("ocr_pm_crop", fake_ocr)]:
        monkey_state[name] = getattr(sbd, name)
        setattr(sbd, name, fn)


def _restore_sbd(monkey_state):
    for name, orig in monkey_state.items():
        setattr(sbd, name, orig)


def test_detect_two_stage_ocr_override():
    """two_stage: inline=30K(SEM) 이지만 OCR=104 → OCR 우선(OM) + crop 디버그 저장."""
    payload = {
        "panel_visible": False, "pm_box_text": "30K",
        "pm_box_bbox": {"left": 100, "top": 100, "right": 200, "bottom": 150},
        "mode_label": None, "confidence": 0.5,
    }
    state = {}
    _install_detect_stubs(state, payload=payload, panel_bbox=None, ocr_text="104")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            crop_path = Path(tmp) / "pm_crop.jpg"
            detect = sbd.detect_sem_box(
                Image.new("RGB", (300, 200)), client="dummy",
                ocr_client="dummy", two_stage=True, pm_crop_debug_path=crop_path,
            )
            assert crop_path.exists(), "PM crop 디버그 저장 안 됨"
    finally:
        _restore_sbd(state)

    assert detect.pm_text == "104", detect.pm_text
    assert detect.pm_mode == "OM", detect.pm_mode
    assert detect.pm_text_source == "ocr_crop", detect.pm_text_source
    assert detect.pm_box_px is not None
    print("[OK] test_detect_two_stage_ocr_override")


def test_detect_two_stage_ocr_empty_fallback():
    """two_stage 이지만 OCR 빈 결과 → inline(30K=SEM) 으로 폴백."""
    payload = {
        "panel_visible": False, "pm_box_text": "30K",
        "pm_box_bbox": {"left": 100, "top": 100, "right": 200, "bottom": 150},
        "mode_label": None, "confidence": 0.5,
    }
    state = {}
    _install_detect_stubs(state, payload=payload, panel_bbox=None, ocr_text=None)
    try:
        detect = sbd.detect_sem_box(
            Image.new("RGB", (300, 200)), client="dummy",
            ocr_client="dummy", two_stage=True,
        )
    finally:
        _restore_sbd(state)

    assert detect.pm_text == "30K", detect.pm_text
    assert detect.pm_mode == "SEM", detect.pm_mode
    assert detect.pm_text_source == "inline_vlm", detect.pm_text_source
    print("[OK] test_detect_two_stage_ocr_empty_fallback")


# ------------------------------------------------------------------
# 3) feasibility 통합 — 모드 선택 + 좌표 shift + overlay.
# ------------------------------------------------------------------


class _FakeTemplate:
    def __init__(self, name, offset):
        self.name = name
        self.align_offset_xy = offset
        self.raw_image = np.zeros((20, 30), dtype=np.uint8)   # (th, tw).


def _install_feasibility_stubs(monkey_state, *, detect, best_xy, best_scale, calls, scores=None):
    """feasibility_check 의 무거운 의존성을 결정적 stub 으로 교체. (원복용 dict 채움)"""
    scores = scores or {"OM": 0.9, "SEM": 0.3}

    def fake_resolve(**kwargs):
        return SimpleNamespace()   # truthy.

    om_tpl = _FakeTemplate("OM", (5, -5))
    sem_tpl = _FakeTemplate("SEM", (0, 0))

    def fake_build(assets, cond_box_crop=True):
        return {"OM": om_tpl, "SEM": sem_tpl}

    def fake_detect(image, client, **kwargs):
        return detect

    def fake_compute(template, gray, scales=None, policy=None):
        calls.append(template.name)
        return SimpleNamespace(
            score=scores[template.name],
            decision="match",
            second_ratio=0.5,
            best_scale=best_scale,
            best_xy=best_xy,
            distinctive=True,
        )

    def fake_count(eqp_id, recipe_id):
        return (0, 0)

    for name, fn in [
        ("resolve_assets_auto", fake_resolve),
        ("build_templates_from_assets", fake_build),
        ("detect_sem_box", fake_detect),
        ("compute_align_key_score_ensemble", fake_compute),
        ("count_staged_events", fake_count),
    ]:
        monkey_state[name] = getattr(fc, name)
        setattr(fc, name, fn)
    return om_tpl, sem_tpl


def _restore(monkey_state):
    for name, orig in monkey_state.items():
        setattr(fc, name, orig)


def _write_frame(path, w=300, h=200):
    cv2.imwrite(str(path), np.full((h, w, 3), 180, dtype=np.uint8))


def test_feasibility_pm_mode_and_shift():
    """PM=OM → OM template 선택 + box 원점만큼 align point 풀프레임 환산."""
    box = {"left": 100, "top": 50, "right": 250, "bottom": 150}
    detect = SemBoxDetection(
        detected=True, width=300, height=200, bbox_px=box, bbox_1000=None,
        sharpness=120.0, blurry=False, mode_label="OM", confidence=0.9,
        vlm_bbox_px=box, pm_text="104", pm_mode="OM",
    )
    state, calls = {}, []
    _install_feasibility_stubs(state, detect=detect, best_xy=(30, 40), best_scale=1.0, calls=calls)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "cap_rcs.jpg"
            _write_frame(frame)
            feas = fc.mark_align_feasibility(
                frame, eqp_id="EQP", recipe_id="C/R", vlm_client="dummy",
            )
    finally:
        _restore(state)

    assert sorted(calls) == ["OM", "SEM"], f"가드용으로 두 template 모두 매칭: {calls}"
    assert feas.mode_source == "PM(104)", feas.mode_source   # OM 점수 충분 → PM 유지.
    assert feas.pm_mode == "OM"
    assert feas.sem_box_bbox == (100, 50, 250, 150), feas.sem_box_bbox
    # match_xy = best_xy + origin = (30+100, 40+50) = (130, 90).
    # align_xy = match_xy + offset*scale = (130+5, 90-5) = (135, 85).
    assert feas.align_xy == (135, 85), feas.align_xy
    assert feas.frame_wh == (300, 200), feas.frame_wh   # 풀프레임 좌표계 유지.
    print("[OK] test_feasibility_pm_mode_and_shift")


def test_feasibility_pm_override_when_far_worse():
    """PM=OM 오독이라도 SEM 이 margin 이상 높으면 점수 승자(SEM)로 폴백."""
    box = {"left": 100, "top": 50, "right": 250, "bottom": 150}
    detect = SemBoxDetection(
        detected=True, width=300, height=200, bbox_px=box, bbox_1000=None,
        sharpness=120.0, blurry=False, mode_label="OM", confidence=0.9,
        vlm_bbox_px=box, pm_text="104", pm_mode="OM",
    )
    state, calls = {}, []
    # PM 은 OM 이라 하지만 SEM 이 0.95 vs OM 0.30 → margin(0.15) 초과 → SEM 채택.
    _install_feasibility_stubs(
        state, detect=detect, best_xy=(30, 40), best_scale=1.0, calls=calls,
        scores={"OM": 0.30, "SEM": 0.95},
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "cap_rcs.jpg"
            _write_frame(frame)
            feas = fc.mark_align_feasibility(
                frame, eqp_id="EQP", recipe_id="C/R", vlm_client="dummy",
            )
    finally:
        _restore(state)

    assert feas.modality == "SEM", feas.modality
    assert feas.mode_source == "score_override(PM=104)", feas.mode_source
    print("[OK] test_feasibility_pm_override_when_far_worse")


def test_feasibility_fallback_when_not_detected():
    """SEM box 검출 실패 → 전체 창 매칭 폴백(try-both, origin (0,0))."""
    detect = SemBoxDetection(
        detected=False, width=300, height=200, bbox_px=None, bbox_1000=None,
        sharpness=None, blurry=False, mode_label=None, confidence=0.1,
        vlm_bbox_px=None, pm_text=None, pm_mode=None,
    )
    state, calls = {}, []
    _install_feasibility_stubs(state, detect=detect, best_xy=(10, 20), best_scale=1.0, calls=calls)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "cap_rcs.jpg"
            _write_frame(frame)
            feas = fc.mark_align_feasibility(
                frame, eqp_id="EQP", recipe_id="C/R", vlm_client="dummy",
            )
    finally:
        _restore(state)

    assert sorted(calls) == ["OM", "SEM"], f"폴백은 두 template 모두 매칭: {calls}"
    assert feas.mode_source == "score", feas.mode_source
    assert feas.sem_box_bbox is None
    # origin (0,0) → align_xy = best_xy + offset. OM 이 점수 높아 선택(offset 5,-5).
    assert feas.align_xy == (15, 15), feas.align_xy
    print("[OK] test_feasibility_fallback_when_not_detected")


if __name__ == "__main__":
    test_pm_text_to_mode()
    test_grey_frame_mask()
    test_snap_box_to_edges()
    test_sharpness_in_box()
    test_bbox_px_to_1000()
    test_crop_pm_region()
    test_detect_two_stage_ocr_override()
    test_detect_two_stage_ocr_empty_fallback()
    test_feasibility_pm_mode_and_shift()
    test_feasibility_pm_override_when_far_worse()
    test_feasibility_fallback_when_not_detected()
    print("\n=== sem_box_detect: 11/11 통과 ===")
