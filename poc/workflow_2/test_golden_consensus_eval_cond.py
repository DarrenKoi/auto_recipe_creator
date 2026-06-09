"""consensus 재등록(cond) 순수 헬퍼 테스트.

핵심 불변:
  - crosshair(=align point) 중심으로 crop 을 정렬해야 median 이 또렷한 align-key 가 된다
    (msr 이미지 중심은 align point 가 아님 — 웨이퍼마다 다름).
  - cond 로 crosshair 를 *지운 뒤* crop → 중앙 distractor 없는 깨끗한 consensus 재료.
"""

from types import SimpleNamespace

import cv2
import numpy as np

from poc.workflow_2.cond_file import CondInfo
import poc.workflow_2.golden_consensus_eval_cond as gce


def _cond(crosshair_xy, box_ltrb=None, scope="OM", raw=None):
    return CondInfo(scope=scope, pixel=(512, 512),
                    box_ltrb=box_ltrb, crosshair_xy=crosshair_xy, raw=raw or {})


def _msr_cond(raw):
    """msr cond 모사: Scope 없음(=None), crosshair 있음, raw 키로 modality 구분."""
    return CondInfo(scope=None, pixel=(512, 512), crosshair_xy=(2000, 2560), raw=raw)


# --- _recipe_key: by_recipe dict 고유 키 (recipe_id=leaf 만이면 충돌→데이터 유실) ---

def test_recipe_key_unique_across_eqp_class():
    # recipe_id(leaf)가 같아도 eqp/class 가 다르면 키가 달라야(probe: 298 dir→276 고유, 충돌).
    a = SimpleNamespace(eqp_id="EQP1", class_name="CLS", recipe_id="RCP")
    b = SimpleNamespace(eqp_id="EQP2", class_name="CLS", recipe_id="RCP")
    assert gce._recipe_key(a) != gce._recipe_key(b)


def test_recipe_key_includes_eqp_class_recipe():
    a = SimpleNamespace(eqp_id="E", class_name="C", recipe_id="R")
    assert gce._recipe_key(a) == "E/C/R"


# --- _floor_min_s: LOO 바닥 3 보정 (len(others)>=2 가드상 fm>=3 이어야 점이 난다) ---

def test_floor_min_s_raises_below_three():
    # 2 이하는 무의미(같은 결과) → 3 으로 올린다(조용한 no-op 방지).
    assert gce._floor_min_s(2) == 3
    assert gce._floor_min_s(1) == 3
    assert gce._floor_min_s(0) == 3


def test_floor_min_s_keeps_three_and_above():
    assert gce._floor_min_s(3) == 3
    assert gce._floor_min_s(4) == 4
    assert gce._floor_min_s(10) == 10


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


# --- _resolve_mod: msr 키/배율 추론, 없으면 recipe rcp modality 폴백 (Scope 안 봄) ---

def test_resolve_mod_falls_back_to_recipe_when_scope_missing():
    # msr cond 에 Scope 없으면 recipe 의 rcp modality 로 frame 을 살린다(과거처럼 drop X).
    assert gce._resolve_mod(_cond((1, 1), scope=None), "om") == "om"


def test_resolve_mod_none_when_no_info_anywhere():
    assert gce._resolve_mod(_msr_cond({}), None) is None


# --- _msr_modality: msr cond 엔 Scope 없음 → 키/배율로 추론 (사용자 규칙 2026-06-08) ---

def test_msr_modality_om_brightness_key():
    assert gce._msr_modality(_msr_cond({"om_brightness": [""], "magnification": ["104"]})) == "om"


def test_msr_modality_accel_voltage_key():
    assert gce._msr_modality(
        _msr_cond({"accelerating_voltage": ["1000"], "magnification": ["20000"]})) == "sem"


def test_msr_modality_low_mag_is_om():
    assert gce._msr_modality(_msr_cond({"magnification": ["150"]})) == "om"


def test_msr_modality_high_mag_is_sem():
    assert gce._msr_modality(_msr_cond({"magnification": ["20000"]})) == "sem"


def test_msr_modality_ambiguous_mag_is_none():
    assert gce._msr_modality(_msr_cond({"magnification": ["300"]})) is None


def test_msr_modality_key_beats_magnification():
    # 키 존재가 1순위 — OM_Brightness 면 mag 가 SEM-range 여도 om.
    assert gce._msr_modality(
        _msr_cond({"om_brightness": [""], "magnification": ["20000"]})) == "om"


def test_resolve_mod_uses_msr_inference_when_scope_absent():
    # Scope 없고 recipe_mod 도 None(om+sem 둘 다 있는 recipe)이어도 msr 키로 해결 → frame 산다.
    assert gce._resolve_mod(_msr_cond({"accelerating_voltage": ["1"]}), None) == "sem"
    assert gce._resolve_mod(_msr_cond({"om_brightness": [""]}), None) == "om"


def test_msr_modality_from_parsed_real_like_cond():
    from poc.workflow_2.cond_file import parse_cond
    text = ("# Observation condition\nMagnification           104000\n"
            "Accelerating_voltage    1000\nPixel                   512,512\n"
            "!Cursor_info            -1,-1,-1,-1,2097,2561,-1,-1,-1,-1\n")
    cond = parse_cond(text)
    assert cond.scope is None and cond.crosshair_xy == (2097, 2561)
    assert gce._msr_modality(cond) == "sem"


# --- _precrop_drop_reason: 누락 사유 분류(coverage 손실 가시화, code-review [4]/[5]) ---

def test_drop_reason_missing_cond():
    assert gce._precrop_drop_reason(None, (1, 1), "om", True) == "missing_cond"


def test_drop_reason_missing_crosshair():
    assert gce._precrop_drop_reason(_cond(None), None, "om", True) == "missing_crosshair"


def test_drop_reason_missing_modality():
    assert gce._precrop_drop_reason(_cond((1, 1)), (1, 1), None, True) == "missing_modality"


def test_drop_reason_no_template():
    assert gce._precrop_drop_reason(_cond((1, 1)), (1, 1), "om", False) == "no_template"


def test_drop_reason_kept_is_none():
    assert gce._precrop_drop_reason(_cond((1, 1)), (1, 1), "om", True) is None


# --- _miss_dist_distribution: consensus miss 거리 분포(ensemble 적용 여지 진단) ---
from poc.workflow_2.align_similarity import _miss_dist_distribution as _mdd


def test_miss_dist_empty():
    d = _mdd([])
    assert d["n"] == 0 and d["bins"] is None and d["median"] is None


def test_miss_dist_bin_boundaries_left_closed():
    # tol=0.2 → near[0.20-0.30) mid[0.30-0.40) far[0.40-0.60) veryfar[>=0.60].
    # 경계값(0.30/0.40/0.60)은 *상위* bin 으로(left-closed) — 부동소수점 오분류 없어야.
    b = _mdd([0.30, 0.40, 0.60])["bins"]
    assert b["near[0.20-0.30)"] == 0
    assert b["mid[0.30-0.40)"] == 1
    assert b["far[0.40-0.60)"] == 1
    assert b["veryfar[>=0.60]"] == 1


def test_miss_dist_just_inside_boundaries():
    b = _mdd([0.299, 0.399, 0.599])["bins"]
    assert b["near[0.20-0.30)"] == 1 and b["mid[0.30-0.40)"] == 1 and b["far[0.40-0.60)"] == 1


def test_miss_dist_counts_and_stats():
    d = _mdd([0.21, 0.25, 0.29, 0.33, 0.45, 0.70, 0.95])
    assert d["n"] == 7
    assert d["bins"]["near[0.20-0.30)"] == 3      # 0.21, 0.25, 0.29
    assert d["bins"]["veryfar[>=0.60]"] == 2      # 0.70, 0.95
    assert sum(d["bins"].values()) == 7           # 모든 miss 가 정확히 한 bin
    assert d["max"] == 0.95 and d["median"] == 0.33
