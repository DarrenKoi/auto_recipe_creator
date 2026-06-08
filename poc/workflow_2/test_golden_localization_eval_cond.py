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

from poc.workflow_2.cond_file import CondInfo
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
