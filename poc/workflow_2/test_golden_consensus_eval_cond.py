"""consensus 재등록(cond) 순수 헬퍼 테스트.

핵심 불변:
  - crosshair(=align point) 중심으로 crop 을 정렬해야 median 이 또렷한 align-key 가 된다
    (msr 이미지 중심은 align point 가 아님 — 웨이퍼마다 다름).
  - cond 로 crosshair 를 *지운 뒤* crop → 중앙 distractor 없는 깨끗한 consensus 재료.
"""

import cv2
import numpy as np

from poc.workflow_2.cond_file import CondInfo
import poc.workflow_2.golden_consensus_eval_cond as gce


def _cond(crosshair_xy, box_ltrb=None, scope="OM"):
    return CondInfo(scope=scope, pixel=(512, 512),
                    box_ltrb=box_ltrb, crosshair_xy=crosshair_xy)


# --- _cond_crosshair_xy ---

def test_crosshair_xy_converts_cursor_to_image_px():
    # cursor (2000,2560)/10 = (200,256).
    assert gce._cond_crosshair_xy(_cond((2000, 2560))) == (200, 256)


def test_crosshair_xy_none_when_absent():
    assert gce._cond_crosshair_xy(_cond(None)) is None
    assert gce._cond_crosshair_xy(None) is None


# --- _cond_consensus_crop ---

def test_consensus_crop_has_requested_size():
    gray = np.full((512, 512), 110, np.uint8)
    crop = gce._cond_consensus_crop(gray, _cond((2000, 2560)), (64, 64))
    assert crop is not None and crop.shape == (64, 64)


def test_consensus_crop_is_centered_on_crosshair():
    # crosshair (200,256). 그 좌상단(190,246)에 선과 무관한 밝은 점 → crop 중심 부근에 와야.
    gray = np.full((512, 512), 110, np.uint8)
    gray[246, 190] = 200
    crop = gce._cond_consensus_crop(gray, _cond((2000, 2560)), (64, 64))
    # 점은 crosshair 기준 (-10,-10) → 64 crop 중심(32,32) 기준 (22,22).
    assert int(crop[22, 22]) >= 180


def test_consensus_crop_removes_crosshair():
    # crosshair 선(255)을 그려도 crop 안에서 inpaint 로 사라져야(중앙 distractor 제거).
    gray = np.full((512, 512), 110, np.uint8)
    cv2.line(gray, (200, 0), (200, 511), 255, 1)   # 세로 crosshair
    cv2.line(gray, (0, 256), (511, 256), 255, 1)   # 가로 crosshair
    crop = gce._cond_consensus_crop(gray, _cond((2000, 2560)), (64, 64))
    assert int(crop.max()) < 200   # 밝은 선이 남지 않음.


def test_consensus_crop_none_without_crosshair():
    gray = np.full((512, 512), 110, np.uint8)
    assert gce._cond_consensus_crop(gray, _cond(None), (64, 64)) is None
