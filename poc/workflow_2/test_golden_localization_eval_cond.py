"""golden_localization_eval_cond 의 cond 전용 순수 헬퍼 테스트.

검증 대상(검출 무관, cond.txt 만으로 결정되는 부품):
  - cond_align_offset : align point(이미지 중심) - box 중심. crop 과 분리(decoupled).
  - cond_offset_norm  : offset 크기를 이미지 *대각선* 으로 정규화(crop 무관 척도).
  - check_cond_box    : box 가 template 으로 쓸만한지 가드(ok/warn/skip + 사유).
  - cond_template_crop: stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 한 crop
                        (crop 중심 == box 중심 → offset 과 일관).
"""

import cv2
import numpy as np

from poc.workflow_3.align.cond_file import CondInfo
import poc.workflow_2.golden_localization_eval_cond as glec


def _cond(box_ltrb, crosshair_xy=None):
    return CondInfo(scope="OM", pixel=(512, 512),
                    box_ltrb=box_ltrb, crosshair_xy=crosshair_xy)


# --- cond_align_offset (decoupled: image_center - box_center) ---

def test_centered_box_has_zero_offset():
    # 512px 이미지, box 중심 = (256,256) → offset (0,0).
    # cursor frame ×10: (2060,2060,3060,3060)/10 = (206,206,306,306), center (256,256).
    off = glec.cond_align_offset((2060, 2060, 3060, 3060), (512, 512))
    assert off == (0, 0)


def test_offcenter_box_offset_is_image_center_minus_box_center():
    # box 중심 (200,256) → offset (256-200, 256-256) = (56, 0).
    # cursor: l,r 중심 2000 → (1500,2560,2500,2560)? 대신 정사각 박스로.
    # center x=200 → l=150,r=250 (px) → cursor (1500,2060,2500,3060) center (200,256).
    off = glec.cond_align_offset((1500, 2060, 2500, 3060), (512, 512))
    assert off == (56, 0)


# --- cond_offset_norm (normalized by image diagonal) ---

def test_offset_norm_zero_for_centered_box():
    assert glec.cond_offset_norm((2060, 2060, 3060, 3060), (512, 512)) == 0.0


def test_offset_norm_uses_image_diagonal():
    # offset (56,0), diag = hypot(512,512) ≈ 724.08 → 56/724.08 ≈ 0.0773.
    onorm = glec.cond_offset_norm((1500, 2060, 2500, 3060), (512, 512))
    assert abs(onorm - 56.0 / float(np.hypot(512, 512))) < 1e-6


# --- check_cond_box (guardrails) ---

def test_check_ok_for_normal_centered_box():
    status, reason, onorm = glec.check_cond_box((2060, 2060, 3060, 3060), (512, 512))
    assert status == "ok"
    assert onorm == 0.0


def test_check_skip_for_tiny_box():
    # box 10px → inset 후 내부 < MIN_INNER_PX → skip.
    # 중심 256: (2510,2510,2610,2610)/10 = (251,251,261,261) → 10x10px.
    status, reason, _ = glec.check_cond_box((2510, 2510, 2610, 2610), (512, 512))
    assert status == "skip"
    assert reason == "box:too_small"


def test_check_skip_for_far_offcenter_box():
    # box 중심을 이미지 좌상단 쪽으로 크게 이동 → offset_norm > OFFSET_SKIP.
    # center (40,40): (150,150,650,650)/10 = (15,15,65,65) center (40,40).
    # offset (216,216), diag 724 → 0.422 > 0.38.
    status, reason, onorm = glec.check_cond_box((150, 150, 650, 650), (512, 512))
    assert status == "skip"
    assert reason == "offset:too_far"
    assert onorm > glec.OFFSET_SKIP


def test_check_skip_for_out_of_bounds_box():
    # 이미지(512) 밖으로 나가는 box.
    status, reason, _ = glec.check_cond_box((4800, 2060, 5600, 3060), (512, 512))
    assert status == "skip"
    assert reason == "box:out_of_bounds"


def test_check_warn_for_moderately_offcenter_box():
    # offset_norm 이 WARN(0.25)~SKIP(0.38) 사이 → warn. center (130,256):
    # (1050,2060,2550,3060)/10 = (105,206,255,306) center (180,256). offset(76,0)/724=0.105 → 너무 작음.
    # 더 멀리: center (60,256) → offset 196 /724 = 0.27. box 150px wide 충분.
    # (350,2060,1550,3060)/10=(35,206,155,306) center(95,256) offset(161)/724=0.222 → warn 미달.
    # center (52,256): (40,2060,1000,3060)/10=(4,206,100,306) center(52,256) offset(204)/724=0.282 → warn.
    status, reason, onorm = glec.check_cond_box((40, 2060, 1000, 3060), (512, 512))
    assert status == "warn"
    assert glec.OFFSET_WARN < onorm <= glec.OFFSET_SKIP


# --- cond_template_crop (inpaint stroke -> symmetric inset interior) ---

def _gray_with_box_stroke(box_xywh, bg=110, stroke=255):
    img = np.full((512, 512), bg, dtype=np.uint8)
    x, y, w, h = box_xywh
    cv2.rectangle(img, (x, y), (x + w, y + h), stroke, 1)
    return img


def test_crop_center_equals_box_center():
    # box (x,y,w,h) = (156,156,200,200) → center (256,256). cursor ×10.
    box_ltrb = (1560, 1560, 3560, 3560)
    gray = _gray_with_box_stroke((156, 156, 200, 200))
    _, (x0, y0, w, h) = glec.cond_template_crop(gray, _cond(box_ltrb))
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    assert abs(cx - 256) <= 0.5 and abs(cy - 256) <= 0.5


def test_crop_is_inset_inside_box():
    # 대칭 inset 만큼 box 보다 작아야 한다(stroke·smear 회피).
    box_ltrb = (1560, 1560, 3560, 3560)   # 200px box
    gray = _gray_with_box_stroke((156, 156, 200, 200))
    _, (x0, y0, w, h) = glec.cond_template_crop(gray, _cond(box_ltrb))
    assert w == 200 - 2 * glec.CROP_INSET_PX
    assert h == 200 - 2 * glec.CROP_INSET_PX


def test_crop_excludes_bright_stroke():
    # inpaint + inset 후 crop 안에 밝은 stroke(255) 가 남지 않아야 한다.
    box_ltrb = (1560, 1560, 3560, 3560)
    gray = _gray_with_box_stroke((156, 156, 200, 200), bg=110, stroke=255)
    crop, _ = glec.cond_template_crop(gray, _cond(box_ltrb))
    assert int(crop.max()) < 200   # 순수 배경(110±) 만 남음.


def test_crop_does_not_inpaint_crosshair_in_rcp_template():
    # rcp box template 은 box stroke 만 지운다. crosshair 가 cond 에 있어도 box 내부의
    # *실제 내용*(crosshair 선이 가로지르는 픽셀)을 inpaint 로 지우면 안 된다(매칭 신호 보존).
    box_ltrb = (1560, 1560, 3560, 3560)              # box (156,156,200,200)
    gray = _gray_with_box_stroke((156, 156, 200, 200), bg=110, stroke=255)
    cv2.line(gray, (160, 256), (352, 256), 200, 1)   # 내부 가로 줄(=실제 내용 대용) 밝기 200
    cond = _cond(box_ltrb, crosshair_xy=(2560, 2560))  # 중심(256,256) crosshair 존재
    crop, _ = glec.cond_template_crop(gray, cond)
    assert int(crop.max()) >= 190   # 줄이 보존됨(=crosshair 제거 안 함).


# --- _offset_diag_cond (대각선 정규화 척도로 자체 진단; gle._offset_diag 의 0.20 short-side 와 분리) ---

def test_offset_diag_cond_uses_diagonal_tol():
    recs = [{"recipe": "a", "mod": "om", "offset_norm": 0.30},
            {"recipe": "b", "mod": "sem", "offset_norm": 0.10}]
    d = glec._offset_diag_cond(recs)
    assert d["tol"] == glec.OFFSET_WARN
    assert d["n_assumption_sensitive"] == 1   # 0.30 > 0.25 만 민감, 0.10 은 아님.


def test_offset_diag_cond_empty():
    d = glec._offset_diag_cond([])
    assert d["n"] == 0


# --- lever_verdict (measure-first 결정: proposer-wall vs reranker-alive) ---

def test_lever_verdict_no_data():
    v = glec.lever_verdict({"n": 0})
    assert v["verdict"] == "no_data"


def test_lever_verdict_proposer_wall_when_topk_near_old_ceiling():
    # gt_in_topk 가 옛 천장(0.594) 근처 → 진실이 후보에 자주 없음 → proposer 가 벽.
    v = glec.lever_verdict(
        {"n": 100, "rank1_hit_rate": 0.40, "gt_in_topk_rate": 0.60, "topk_not_rank1_rate": 0.20})
    assert v["verdict"] == "proposer_wall"


def test_lever_verdict_reranker_alive_when_topk_high_and_gap_present():
    # gt_in_topk 높고(진실이 후보에 있음) rank1 와 격차 큼 → 재정렬로 이득 가능.
    v = glec.lever_verdict(
        {"n": 100, "rank1_hit_rate": 0.55, "gt_in_topk_rate": 0.85, "topk_not_rank1_rate": 0.25})
    assert v["verdict"] == "reranker_alive"


def test_lever_verdict_near_ceiling_when_topk_high_but_no_gap():
    # gt_in_topk 높지만 거의 rank1 → 남은 미스는 proposer(gt<1) 몫, reranker 여지 작음.
    v = glec.lever_verdict(
        {"n": 100, "rank1_hit_rate": 0.82, "gt_in_topk_rate": 0.85, "topk_not_rank1_rate": 0.03})
    assert v["verdict"] == "near_ceiling"


# --- _route_modality (msr frame 을 어느 modality rcp 로 매칭할지; race 제거의 핵심) ---

def _mcond(text):
    from poc.workflow_3.align.cond_file import parse_cond
    return parse_cond(text)


def test_route_uses_msr_om_key_when_both_available():
    cond = _mcond("!OM_Brightness\t128\nMagnification\t104\n")
    assert glec._route_modality(cond, {"om", "sem"}) == "om"


def test_route_uses_msr_sem_key_when_both_available():
    cond = _mcond("Accelerating_voltage\t1000\nMagnification\t50000\n")
    assert glec._route_modality(cond, {"om", "sem"}) == "sem"


def test_route_falls_back_to_single_recipe_modality_when_msr_unknown():
    # msr 미상(모호 배율) + recipe 가 om rcp 만 보유 → om 으로 폴백.
    cond = _mcond("Magnification\t300\n")
    assert glec._route_modality(cond, {"om"}) == "om"


def test_route_none_when_msr_unknown_and_recipe_dual():
    # msr 미상 + recipe 가 om·sem 둘 다 → 모호 → None(skip, 틀린-modality 측정 차단).
    cond = _mcond("Magnification\t300\n")
    assert glec._route_modality(cond, {"om", "sem"}) is None


def test_route_none_when_inferred_modality_has_no_rcp():
    # msr 는 sem 인데 recipe 에 sem rcp template 이 없음 → 틀린 om 으로 매칭 금지 → None.
    cond = _mcond("Accelerating_voltage\t1000\n")
    assert glec._route_modality(cond, {"om"}) is None


def test_route_single_modality_when_cond_none():
    # cond 자체가 None 이어도 recipe 가 단일 modality 면 그걸로 폴백.
    assert glec._route_modality(None, {"sem"}) == "sem"


# --- _combine_2up (rcp | msr 한 장 결합 패널; 눈으로 추출→매칭→찍기 추적용) ---

def _bgr(h, w, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_combine_2up_returns_bgr_uint8():
    out = glec._combine_2up(_bgr(120, 200), _bgr(120, 200),
                            rcp_label="RCP", msr_label="MSR")
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8


def test_combine_2up_height_is_common_and_width_holds_both():
    # 높이 다른 두 패널 → 공통 높이로 맞추고 가로로 붙임(둘 다 들어갈 너비).
    rcp, msr = _bgr(100, 160), _bgr(140, 220)
    out = glec._combine_2up(rcp, msr, rcp_label="RCP", msr_label="MSR")
    # 헤더 band 가 위에 붙으므로 공통 높이는 더 큰 패널(헤더 포함) 기준.
    assert out.shape[0] == max(100, 140) + glec._PANEL_HEADER_PX
    # 좌(rcp)+구분선+우(msr) → 각 입력 너비 합 이상.
    assert out.shape[1] >= 160 + 220


def test_combine_2up_rcp_none_returns_msr_only_width():
    # rcp 가 없으면(해당 modality box template 부재) msr 패널만, 너비 보존.
    msr = _bgr(120, 240)
    out = glec._combine_2up(None, msr, rcp_label="RCP", msr_label="MSR")
    assert out.shape[1] == 240   # 헤더는 높이만 늘리고 너비는 보존.
    assert out.shape[0] == 120 + glec._PANEL_HEADER_PX


def test_combine_2up_has_separator_column():
    # 두 패널 사이 구분선(어두운 띠)이 한 번 들어간다 → 너비에 sep 만큼 더해짐.
    rcp, msr = _bgr(120, 100), _bgr(120, 100)
    out = glec._combine_2up(rcp, msr, rcp_label="R", msr_label="M")
    # 같은 높이(헤더 동일) → resize 없이 100+sep+100.
    assert out.shape[1] == 100 + glec._PANEL_SEP_PX + 100


def test_matcher_for_eval_toggle(monkeypatch):
    """ALIGN_USE_ENSEMBLE 토글: 참이면 ensemble 매처, 아니면 baseline(eval 전용)."""
    import poc.workflow_2.golden_localization_eval as gle
    from poc.workflow_3.align.matching import engine as akm

    monkeypatch.delenv("ALIGN_ENSEMBLE_LAB_MODE", raising=False)
    monkeypatch.delenv("ALIGN_USE_ENSEMBLE", raising=False)
    assert gle._matcher_for_eval() is akm.compute_align_key_score        # 기본 baseline.
    monkeypatch.setenv("ALIGN_USE_ENSEMBLE", "1")
    assert gle._matcher_for_eval() is akm.compute_align_key_score_ensemble
    monkeypatch.setenv("ALIGN_USE_ENSEMBLE", "true")
    assert gle._matcher_for_eval() is akm.compute_align_key_score_ensemble
    monkeypatch.setenv("ALIGN_USE_ENSEMBLE", "0")
    assert gle._matcher_for_eval() is akm.compute_align_key_score         # off → baseline.


def test_matcher_for_eval_lab_mode_uses_lab_wrapper(monkeypatch):
    """ALIGN_ENSEMBLE_LAB_MODE 는 production ensemble 대신 workflow_2 lab wrapper 를 고른다."""
    import poc.workflow_2.golden_localization_eval as gle

    monkeypatch.setenv("ALIGN_ENSEMBLE_LAB_MODE", "edge_ncc")
    monkeypatch.delenv("ALIGN_LAB_ENSEMBLE_CHANNELS", raising=False)
    monkeypatch.delenv("ENSEMBLE_CHANNELS", raising=False)
    matcher = gle._matcher_for_eval()
    assert matcher.__name__ == "_compute_align_key_score_lab_for_eval"
    assert gle._lab_channels_for_eval() == ("canny", "scharr", "orient", "edge_ncc")


def test_matcher_for_eval_lab_channels_env(monkeypatch):
    """ALIGN_LAB_ENSEMBLE_CHANNELS 가 있으면 lab mode 기본 채널보다 우선한다."""
    import poc.workflow_2.golden_localization_eval as gle

    monkeypatch.setenv("ALIGN_ENSEMBLE_LAB_MODE", "1")
    monkeypatch.setenv("ALIGN_LAB_ENSEMBLE_CHANNELS", "canny,c4")
    assert gle._lab_channels_for_eval() == ("canny", "edge_ncc")


# --- Tier 1.1: miss-distance bin 분류 (순수) ---

def test_displacement_bin_boundaries():
    # frame 512x512 → center (256,256), short=512. norm = |GT-center|/512.
    fhw = (512, 512)
    assert glec.displacement_bin((296, 256), fhw) == "near"      # 40/512=0.078
    assert glec.displacement_bin((333, 256), fhw) == "mid"       # 77/512=0.150
    assert glec.displacement_bin((386, 256), fhw) == "far"       # 130/512=0.254
    assert glec.displacement_bin((456, 256), fhw) == "veryfar"   # 200/512=0.391


def test_rescue_bin_boundaries():
    tol = glec.gle.GT_TOL_NORM   # center dist_norm 을 tol 배수로 줘 경계 검증.
    assert glec.rescue_bin(0.5 * tol) == "hit"
    assert glec.rescue_bin(1.5 * tol) == "near"
    assert glec.rescue_bin(3.0 * tol) == "far"
    assert glec.rescue_bin(5.0 * tol) == "veryfar"


# --- Tier 1.1: bin × arm 집계 ---

def _loc(in_topk, hit, dist_norm=0.0):
    """합성 _localize 결과(집계가 쓰는 키만)."""
    return {"in_topk": in_topk, "hit": hit, "dist_norm": dist_norm,
            "topk_rank": 1 if hit else (2 if in_topk else None),
            "align_xy": (0, 0), "mod": "om", "score": 0.5}


def _row(center, box, gt_xy, frame_hw, label="S"):
    cells = {}
    if center is not None:
        cells["center__inpaint"] = center
    if box is not None:
        cells["box__inpaint"] = box
    return {"label": label, "crosshair_xy": gt_xy, "frame_hw": frame_hw, "cells": cells}


def test_binned_report_displacement_aggregates():
    fhw = (512, 512)   # 두 행 모두 near (norm<0.10).
    rows = [
        _row(_loc(in_topk=True, hit=True),  _loc(in_topk=True, hit=False), (296, 256), fhw),
        _row(_loc(in_topk=False, hit=False), _loc(in_topk=True, hit=True),  (300, 256), fhw),
    ]
    near = glec._binned_localization_report(rows)["by_displacement"]["near"]
    assert near["center"] == {"n": 2, "gt_in_topk": 0.5, "rank1": 0.5}
    assert near["box"] == {"n": 2, "gt_in_topk": 1.0, "rank1": 0.5}


def test_binned_report_rescue_uses_center_distnorm():
    tol = glec.gle.GT_TOL_NORM
    fhw = (512, 512)
    rows = [
        _row(_loc(True, True, dist_norm=0.5 * tol), _loc(True, False), (256, 256), fhw),  # hit bin
        _row(_loc(False, False, dist_norm=3.0 * tol), _loc(True, True), (256, 256), fhw), # far bin
    ]
    by_miss = glec._binned_localization_report(rows)["by_center_miss"]
    assert by_miss["hit"]["box"] == {"n": 1, "gt_in_topk": 1.0, "rank1": 0.0}
    assert by_miss["far"]["box"] == {"n": 1, "gt_in_topk": 1.0, "rank1": 1.0}


def test_binned_report_skips_missing_box_and_nonS():
    fhw = (512, 512)
    rows = [
        _row(_loc(True, True), None, (296, 256), fhw),            # box 결손 → box 분모서 제외.
        _row(_loc(True, True), _loc(True, True), (296, 256), fhw, label="E"),  # 비-S → 무시.
    ]
    near = glec._binned_localization_report(rows)["by_displacement"]["near"]
    assert near["center"]["n"] == 1
    assert near["box"] == {"n": 0, "gt_in_topk": None, "rank1": None}


# --- Tier 1.1: ensemble 매처 hardcode (setdefault escape hatch) ---
# akm 는 기존 test_matcher_for_eval_toggle 과 동일 경로로 — _matcher_for_eval 반환과 is-동일.

def test_apply_matcher_default_forces_ensemble(monkeypatch):
    import poc.workflow_2.golden_localization_eval as gle
    from poc.workflow_3.align.matching import engine as akm
    monkeypatch.delenv("ALIGN_USE_ENSEMBLE", raising=False)
    glec._apply_matcher_default()                       # 미설정 → ensemble 로 채움.
    assert gle._matcher_for_eval() is akm.compute_align_key_score_ensemble


def test_apply_matcher_default_respects_explicit_off(monkeypatch):
    import poc.workflow_2.golden_localization_eval as gle
    from poc.workflow_3.align.matching import engine as akm
    monkeypatch.setenv("ALIGN_USE_ENSEMBLE", "0")       # 명시적 0 → 유지(escape hatch).
    glec._apply_matcher_default()
    assert gle._matcher_for_eval() is akm.compute_align_key_score
