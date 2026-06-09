"""inpaint 잔상(ghost) 정량화 — 정답 없는(self-supervised) 측정.

주석(흰 box / crosshair)이 있던 자리를 cond 좌표로 알 수 있으므로:
  - footprint : 주석 선 + 약간의 halo 여유. '청소돼야 하는 자리'.
  - bg_ring   : footprint 바깥의 배경 띠. '정상 웨이퍼' 참조.
청소가 잘 됐으면 footprint 밝기 ≈ bg_ring 밝기 → 잔상점수≈0. 흰 선이 남아 있으면
footprint 가 더 밝아 점수가 커진다. 정답(주석 없는 원본)이 필요 없다.

평가용 footprint/bg_ring 은 **테스트하는 마스크 파라미터와 무관하게 고정**(EVAL_*)
이어야 여러 (thickness·dilate·radius) 조합을 공정하게 비교할 수 있다.
"""

import cv2
import numpy as np

from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, build_removal_mask

# 평가용(파라미터 독립) 고정 마진. 실제 청소 마스크와 분리한다.
EVAL_CORE = 1        # 선 코어 두께
EVAL_PAD = 3         # footprint 가 선 밖으로 덮는 halo 여유
EVAL_BG_GAP = 3      # footprint 와 bg_ring 사이 완충(겹침 방지)
EVAL_BG_WIDTH = 6    # bg_ring 띠 폭


def build_eval_masks(
    shape_hw,
    *,
    box_ltrb=None,
    crosshair_xy=None,
    oversample=OVERSAMPLE,
):
    """(footprint, bg_ring) 0/255 마스크를 만든다. 두 영역은 서로 겹치지 않는다."""
    common = dict(box_ltrb=box_ltrb, crosshair_xy=crosshair_xy,
                  oversample=oversample, thickness=EVAL_CORE)
    footprint = build_removal_mask(shape_hw, dilate=EVAL_PAD, **common)
    inner_guard = build_removal_mask(shape_hw, dilate=EVAL_PAD + EVAL_BG_GAP, **common)
    outer = build_removal_mask(
        shape_hw, dilate=EVAL_PAD + EVAL_BG_GAP + EVAL_BG_WIDTH, **common)
    bg_ring = cv2.subtract(outer, inner_guard)     # 완충 밖 ~ outer 사이 띠
    return footprint, bg_ring


def ghost_residual(gray, footprint, bg_ring):
    """footprint 와 bg_ring 평균 밝기 차의 절댓값. 클수록 잔상이 많이 남음."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    fp = footprint > 0
    bg = bg_ring > 0
    if not fp.any() or not bg.any():
        return 0.0
    return float(abs(gray[fp].mean() - gray[bg].mean()))


def mask_area_fraction(mask):
    """마스크가 차지하는 비율(0~1). 클수록 실제 내용을 많이 파괴(=collateral)."""
    return float(np.count_nonzero(mask)) / float(mask.size)
