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
