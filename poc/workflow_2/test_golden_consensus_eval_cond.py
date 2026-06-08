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


# --- co-registration (sub-pixel 정렬로 median blur 감소) ---

def test_align_to_ref_undoes_known_shift():
    rng = np.random.default_rng(1)
    ref = np.clip(110 + rng.integers(-30, 30, (64, 64)), 0, 255).astype(np.uint8)
    shifted = cv2.warpAffine(ref, np.float32([[1, 0, 3], [0, 1, 2]]), (64, 64),
                             borderMode=cv2.BORDER_REPLICATE)
    aligned = gce._align_to_ref(shifted, ref)
    inner = (slice(8, 56), slice(8, 56))
    err_aligned = np.mean(np.abs(aligned[inner].astype(int) - ref[inner].astype(int)))
    err_shifted = np.mean(np.abs(shifted[inner].astype(int) - ref[inner].astype(int)))
    assert err_aligned < err_shifted   # 정렬 후 ref 와 더 가까워야.


def test_coregister_aligns_crops_so_median_matches_truth():
    # jitter 준 crop 들을 정렬하면 (1) 서로 겹쳐 cross-crop std 가 급감하고
    # (2) median 이 원본(base, 정답 template)에 더 가까워진다(=consensus 또렷).
    rng = np.random.default_rng(2)
    base = np.full((64, 64), 110, np.uint8)
    cv2.rectangle(base, (20, 20), (44, 44), 230, 2)
    cv2.circle(base, (32, 32), 6, 40, -1)
    crops = []
    for _ in range(6):
        dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        crops.append(cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]),
                                    (64, 64), borderMode=cv2.BORDER_REPLICATE))
    aligned = gce.coregister_crops(crops)
    inner = (slice(10, 54), slice(10, 54))

    raw_std = float(np.std(np.stack(crops).astype(np.float32), 0)[inner].mean())
    al_std = float(np.std(np.stack(aligned).astype(np.float32), 0)[inner].mean())
    assert al_std < raw_std * 0.5          # 정렬로 crop 들이 겹친다(불일치 급감).

    raw_med = np.median(np.stack(crops), 0).astype(np.uint8)
    al_med = np.median(np.stack(aligned), 0).astype(np.uint8)
    b = base[inner].astype(int)
    mse_raw = float(np.mean((raw_med[inner].astype(int) - b) ** 2))
    mse_al = float(np.mean((al_med[inner].astype(int) - b) ** 2))
    assert mse_al < mse_raw                 # 정렬 median 이 정답 template 에 더 가깝다.


def test_coregister_passthrough_when_too_few():
    c = [np.zeros((8, 8), np.uint8)]
    out = gce.coregister_crops(c)
    assert len(out) == 1 and out[0] is c[0]


# --- scope label(충실 3분류) vs routing modality(OMDF→OM) ---

def test_scope_label_faithful_three_way():
    assert gce._scope_label(_cond((1, 1), scope="SEM")) == "sem"
    assert gce._scope_label(_cond((1, 1), scope="OM")) == "om"
    # "OM" 부분일치로 OMDF 를 삼키지 않아야(진단 가시성).
    assert gce._scope_label(_cond((1, 1), scope="OMDF")) == "omdf"
    assert gce._scope_label(_cond((1, 1), scope=None)) is None
    assert gce._scope_label(None) is None


def test_modality_routes_omdf_into_om():
    # OMDF 는 OM 의 한 종류(OM+darkfield) → routing 은 om. SEM 만 별도.
    assert gce._modality_of(_cond((1, 1), scope="OMDF")) == "om"
    assert gce._modality_of(_cond((1, 1), scope="OM")) == "om"
    assert gce._modality_of(_cond((1, 1), scope="SEM")) == "sem"


def test_modality_missing_is_none_not_om():
    # Scope 없으면 침묵 om 금지 → None (호출부가 skip/카운트).
    assert gce._modality_of(_cond((1, 1), scope=None)) is None
    assert gce._modality_of(None) is None


# --- _resolve_mod: msr scope 우선, 없으면 recipe rcp modality 폴백 ---

def test_resolve_mod_prefers_cond_scope():
    assert gce._resolve_mod(_cond((1, 1), scope="SEM"), "om") == "sem"


def test_resolve_mod_falls_back_to_recipe_when_scope_missing():
    # msr cond 에 Scope 없으면 recipe 의 rcp modality 로 frame 을 살린다(과거처럼 drop X).
    assert gce._resolve_mod(_cond((1, 1), scope=None), "om") == "om"


def test_resolve_mod_none_when_no_info_anywhere():
    assert gce._resolve_mod(_cond((1, 1), scope=None), None) is None
