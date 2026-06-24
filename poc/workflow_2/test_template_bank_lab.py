"""Template-bank matcher bench fork 테스트 (합성 데이터, Mac). workflow_3 미수정 fork."""
import os

import numpy as np

import poc.workflow_2.golden_eval_config_loader as cfg


def test_seed_env_bridges_tbank_defaults(monkeypatch):
    for k in ("TBANK_HEATMAP", "TBANK_RRF", "TBANK_PEAK_NMS_FRAC",
              "TBANK_CLUSTER_TOL_FRAC", "TBANK_RRF_K"):
        monkeypatch.delenv(k, raising=False)
    cfg.seed_env()
    assert os.environ["TBANK_HEATMAP"] == "1"
    assert os.environ["TBANK_RRF"] == "1"
    assert os.environ["TBANK_PEAK_NMS_FRAC"] == "0.5"
    assert os.environ["TBANK_CLUSTER_TOL_FRAC"] == "0.1"
    assert os.environ["TBANK_RRF_K"] == "60"


def _mark_crop(seed, size=64):
    """저텍스처 배경 + seed 위치 고유 마크가 있는 합성 gray crop."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size) * 30 + 20).astype(np.uint8)
    cx = cy = size // 2
    img[cy - 6:cy + 6, cx - 1:cx + 1] = 230   # 십자 마크(고유 구조).
    img[cy - 1:cy + 1, cx - 6:cx + 6] = 230
    return img


def test_bank_build_keeps_members_individual():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    assert len(bank) == 4                     # median 으로 합치지 않고 N 개 유지.
    assert all(hasattr(m, "edge_map") for m in bank)
    assert all(m.key_type == "om" for m in bank)


def test_bank_build_respects_min_s():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(2)]
    assert tb.bank_build(crops, recipe_id="r", modality="om", min_s=3) == []
