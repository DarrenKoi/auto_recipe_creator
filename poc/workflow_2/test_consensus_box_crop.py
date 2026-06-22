"""box-crop align_offset 매핑 테스트.

box template 의 match 중심(box center)에 align_offset 을 가산하여 align point 를 계산하는
_gt_in_topk 행동 검증.
"""

from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poc.workflow_2 import align_similarity as alsim
from poc.workflow_3.align.matching.engine import build_template, AlignKeyCandidate


def _dummy_template(r=10):
    """간단한 더미 template."""
    tpl_img = np.full((2 * r, 2 * r), 100, np.uint8)
    return tpl_img


def test_gt_in_topk_applies_align_offset_to_reach_align_point():
    """offset 이 있는 box template: match 중심(box center)에 offset 을 더해야 truth(align point)와 일치.

    mock candidate (128,128)을 만들고, offset=(20,-10) scale=1.0 이면
    align point = (128,128) + (20,-10) = (148,118) 로 truth 에 부딪혀야 함.
    """
    frame = np.full((256, 256), 50, np.uint8)
    tpl = build_template(_dummy_template(r=10), recipe_id="R", version="box", key_type="sem")
    tpl.align_offset_xy = (20, -10)   # align point = box center + (20,-10) = (148,118)
    truth_align_point = (148, 118)

    # mock _propose_topk 을 덮어서 (128,128) 에 후보를 반환
    mock_cand = AlignKeyCandidate(score=0.9, chamfer_score=0.9, xy=(128, 128),
                                   scale=1.0, template_size=(20, 20))
    with mock.patch('poc.workflow_2.align_similarity._propose_topk') as mock_propose:
        mock_propose.return_value = [mock_cand]
        out = alsim._gt_in_topk(frame, truth_align_point, {"sem": tpl},
                                scales=(1.0,), topk=8)

    assert out is not None
    assert out["in_topk"] is True, "offset 적용 시 box-center match 가 align point 로 매핑돼 truth 히트해야 함"


def test_gt_in_topk_zero_offset_unchanged():
    """offset (0,0)(center template): align point == match center — 기존 동작 유지(회귀 가드).

    candidate (128,128) 이 그대로 truth (128,128) 와 매칭돼야 함.
    """
    frame = np.full((256, 256), 50, np.uint8)
    tpl = build_template(_dummy_template(r=10), recipe_id="R", version="center", key_type="sem")
    # align_offset_xy 기본 (0,0). truth = (128,128).
    truth_xy = (128, 128)

    mock_cand = AlignKeyCandidate(score=0.9, chamfer_score=0.9, xy=(128, 128),
                                   scale=1.0, template_size=(20, 20))
    with mock.patch('poc.workflow_2.align_similarity._propose_topk') as mock_propose:
        mock_propose.return_value = [mock_cand]
        out = alsim._gt_in_topk(frame, truth_xy, {"sem": tpl}, scales=(1.0,), topk=8)

    assert out is not None
    assert out["in_topk"] is True


from poc.workflow_2 import golden_consensus_eval_cond as gce
from poc.workflow_3.align.matching.engine import AlignKeyTemplate


def _stack_crops(marker_val, size=40, n=4):
    crops = []
    for _ in range(n):
        c = np.full((size, size), 30, np.uint8)
        c[10:30, 10:30] = marker_val
        crops.append(c)
    return crops


def test_consensus_box_arm_reports_fixed_denominator_per_modality(monkeypatch, tmp_path):
    """box arm 이 켜지면 per-(modality, arm) recall + 고정 분모 카운트를 반환한다."""
    # _gt_in_topk 을 결정적 스텁으로: center 는 항상 hit, box 는 항상 no-candidate(None 후보 모사).
    calls = {"center": 0, "box": 0}

    def _fake_gt(gray, xy, tpls, *, scales=None, topk=None, tol_short=None):
        (mod, tpl), = tpls.items()
        ver = getattr(tpl, "version", "")
        if "box" in ver:
            calls["box"] += 1
            return None              # box: 후보 0개 -> miss(skip 아님)여야 함
        calls["center"] += 1
        return {"topk_rank": 1, "in_topk": True, "n_cand": 8,
                "best_cand_dist_norm": 0.0, "modality": mod, "cand_xys": [],
                "peak_ratio": 1.0, "cand_scores": [], "cand_ncc": None}

    monkeypatch.setattr(alsim, "_gt_in_topk", _fake_gt)

    # 한 recipe·SEM·4 S 프레임. center crop + box crop + box_tpls 제공.
    fm = [{"path": f"S{i}", "xy": (50, 50), "mod": "sem",
           "crop": _stack_crops(220)[0], "crop_box": _stack_crops(200)[0]} for i in range(4)]
    box_tpl = build_template(np.full((20, 20), 200, np.uint8),
                             recipe_id="R", version="sem_box", key_type="sem")
    by_recipe = {"R": {"s_frames": fm, "e_paths": [],
                       "rcp_tpls": {}, "box_tpls": {"sem": (box_tpl, (5, 5))},
                       "history_crops": {}}}

    res = alsim._consensus_template_ab(by_recipe, min_s=3, out_dir=str(tmp_path),
                                       box_crop=True,
                                       frame_loader=lambda f: np.full((100, 100), 30, np.uint8))
    ab = res["box_crop_ab"]["per_modality"]["sem"]
    assert ab["center"]["n_eval"] == ab["box"]["n_eval"], "두 arm 의 분모가 같아야(고정 분모)"
    assert ab["box"]["n_no_candidate"] == ab["box"]["n_eval"], "box 후보 0개는 miss 로 세야(분모 유지)"
    assert ab["box"]["recall"] == 0.0
    assert ab["center"]["recall"] == 1.0


def test_consensus_box_arm_off_by_default(tmp_path):
    """box_crop=False(기본) -> box_crop_ab 없음(기존 동작 불변)."""
    fm = [{"path": f"S{i}", "xy": (50, 50), "mod": "sem",
           "crop": _stack_crops(220)[0]} for i in range(4)]
    by_recipe = {"R": {"s_frames": fm, "e_paths": [], "rcp_tpls": {}, "history_crops": {}}}
    res = alsim._consensus_template_ab(by_recipe, min_s=3, out_dir=str(tmp_path),
                                       frame_loader=lambda f: np.full((100, 100), 30, np.uint8))
    assert res is None or "box_crop_ab" not in (res or {})


def test_box_crop_is_centered_on_box_region_offset_from_crosshair():
    """box crop = crosshair - offset 중심·box size. crosshair 중심 center crop 과 다른 영역."""
    # 합성: full frame 에 box marker 를 crosshair 에서 offset 만큼 떨어뜨려 둔다.
    size = 300
    gray = np.full((size, size), 30, np.uint8)
    crosshair = (200, 150)
    offset = (40, -20)               # offset = img_center - box_center → box center = crosshair - offset
    box_cx, box_cy = crosshair[0] - offset[0], crosshair[1] - offset[1]   # (160,170)
    gray[box_cy - 12:box_cy + 12, box_cx - 12:box_cx + 12] = 220          # box 영역 마커
    box_tpl = build_template(np.full((24, 24), 220, np.uint8),
                             recipe_id="R", version="box", key_type="sem")
    crop = gce._box_consensus_crop(gray, crosshair, offset, box_tpl)
    assert crop is not None
    assert crop.shape == (24, 24)
    # box 마커를 담았으면 평균이 밝다(잘못된 영역이면 어둡다).
    assert crop.mean() > 150, "box crop 이 box 영역(밝은 마커)을 담아야 함"


def test_history_crops_box_feeds_box_arm_in_history_path(monkeypatch, tmp_path):
    """I2 회귀 가드: history_crops_box 가 채워지면 history 경로에서 box arm 이 실제 hit 를 낸다.

    history 경로(use_history=True)에서 box arm pool 은 data["history_crops_box"][mod] 로
    채워진다. 이전 버전은 history_crops_box 를 쓰기 않아 pool 이 항상 비어 n_no_cand 가
    n_eval 과 같았다. 이 테스트는 history_crops_box 가 min_s 이상 채워진 경우 box arm 에서
    n_no_cand < n_eval (= 실제 hit 발생)임을 보장한다.

    setup: SEM 1개 eval frame, history_crops >=min_s (history 경로 진입),
           history_crops_box >=2 (box pool 충분), _gt_in_topk 스텁 = box arm 항상 hit.
    """
    calls = {"center": 0, "box": 0}

    def _fake_gt(gray, xy, tpls, *, scales=None, topk=None, tol_short=None):
        (mod, tpl), = tpls.items()
        ver = getattr(tpl, "version", "")
        if "box" in ver:
            calls["box"] += 1
            return {"topk_rank": 1, "in_topk": True, "n_cand": 8,
                    "best_cand_dist_norm": 0.0, "modality": mod, "cand_xys": [],
                    "peak_ratio": 1.0, "cand_scores": [], "cand_ncc": None}
        calls["center"] += 1
        return {"topk_rank": 1, "in_topk": True, "n_cand": 8,
                "best_cand_dist_norm": 0.0, "modality": mod, "cand_xys": [],
                "peak_ratio": 1.0, "cand_scores": [], "cand_ncc": None}

    monkeypatch.setattr(alsim, "_gt_in_topk", _fake_gt)

    # eval: SEM 1장(history 경로이므로 LOO pool 은 쓰지 않는다).
    fm = [{"path": "S0", "xy": (50, 50), "mod": "sem",
           "crop": _stack_crops(220)[0], "crop_box": _stack_crops(200)[0]}]
    box_tpl = build_template(np.full((20, 20), 200, np.uint8),
                             recipe_id="R", version="sem_box", key_type="sem")
    # history_crops: SEM 4장 — use_history=True 조건 충족(min_s=3).
    history_center = _stack_crops(220, n=4)
    # history_crops_box: SEM 4장 — box arm pool 충분.
    history_box = _stack_crops(200, n=4)
    by_recipe = {
        "R": {
            "s_frames": fm,
            "e_paths": [],
            "rcp_tpls": {},
            "box_tpls": {"sem": (box_tpl, (5, 5))},
            "history_crops": {"sem": history_center},
            "history_crops_box": {"sem": history_box},
        }
    }

    res = alsim._consensus_template_ab(by_recipe, min_s=3, out_dir=str(tmp_path),
                                       box_crop=True,
                                       frame_loader=lambda f: np.full((100, 100), 30, np.uint8))
    assert res is not None, "결과 없음 — recipe 가 min_s 조건을 통과해야 함"
    ab = res["box_crop_ab"]["per_modality"]["sem"]
    assert ab["box"]["n_eval"] > 0, "history 경로에서 box arm 분모가 0"
    assert ab["box"]["n_no_candidate"] < ab["box"]["n_eval"], (
        "history_crops_box 가 채워졌으므로 box arm 이 최소 1개 이상 hit 를 내야 함 "
        "(n_no_cand < n_eval). history_crops_box 미설정이면 이 조건은 항상 실패한다.")
    assert ab["box"]["recall"] > 0.0, "box arm recall > 0 이어야(history pool 활용)"


def test_box_crop_digest_formats_per_modality_delta():
    """_format_box_crop_digest: per-modality center-vs-box delta + 카운트 표기를 검증한다.

    SEM: box-center = 0.88-0.71 = +0.17, OM: 0.90-0.91 = -0.01.
    줄에 modality 이름·delta·n_eval·no_cand 가 있어야 함.
    """
    per_mod = {
        "sem": {"center": {"recall": 0.71, "rank1": 0.71, "n_eval": 100, "n_hit": 71, "n_no_candidate": 0},
                "box": {"recall": 0.88, "rank1": 0.85, "n_eval": 100, "n_hit": 88, "n_no_candidate": 3}},
        "om": {"center": {"recall": 0.91, "rank1": 0.90, "n_eval": 50, "n_hit": 46, "n_no_candidate": 0},
               "box": {"recall": 0.90, "rank1": 0.89, "n_eval": 50, "n_hit": 45, "n_no_candidate": 1}},
    }
    lines = gce._format_box_crop_digest(per_mod)
    joined = "\n".join(lines)
    assert "sem" in joined.lower() and "om" in joined.lower()
    assert "+0.17" in joined or "+0.170" in joined, "SEM box-center delta(+0.17) 표기"
    assert "n_eval" in joined and "box_no_cand" in joined


def test_gt_in_topk_tol_short_overrides_hit_tolerance(monkeypatch):
    """tol_short 가 주어지면 hit 판정 tolerance 를 그 reference 로 정규화한다.

    center vs box arm 비교 공정성: box template 은 크기가 달라 per-tpl short(=tol)이 달라지면
    같은 align point 를 두 arm 이 다른 픽셀 tolerance 로 채점한다. tol_short 로 양 arm 의
    tolerance 를 일치시켜 apples-to-apples 로 만든다. (GT_TOL_NORM=0.20)
    """
    class _Cand:
        def __init__(self):
            self.xy = (60, 50)   # truth (50,50) 에서 10 px 떨어짐
            self.score = 0.9
            self.scale = 1.0

    monkeypatch.setattr(alsim, "_propose_topk", lambda *a, **k: [_Cand()])
    tpl = build_template(np.full((20, 20), 200, np.uint8),
                         recipe_id="R", version="t", key_type="sem")
    frame = np.zeros((200, 200), np.uint8)
    truth = (50, 50)

    # per-tpl short=20 → tol=0.20*20=4 px; dist 10 px > 4 → miss.
    miss = alsim._gt_in_topk(frame, truth, {"sem": tpl}, scales=(1.0,), topk=8)
    assert miss is not None and miss["in_topk"] is False

    # tol_short=100 → tol=0.20*100=20 px; dist 10 px < 20 → hit.
    hit = alsim._gt_in_topk(frame, truth, {"sem": tpl}, scales=(1.0,), topk=8, tol_short=100)
    assert hit is not None and hit["in_topk"] is True
