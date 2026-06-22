"""reregister 리포트 순수 헬퍼 + config 브리지 테스트."""
import os
from poc.workflow_2 import golden_eval_config_loader as cfg


def test_seed_env_bridges_reregister_defaults():
    # 기존 값 격리
    for k in ("REREGISTER_BOX_SUGGEST", "REREGISTER_TOPN"):
        os.environ.pop(k, None)
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "1"
    assert os.environ["REREGISTER_TOPN"] == "0"


def test_seed_env_respects_existing_reregister(monkeypatch):
    monkeypatch.setenv("REREGISTER_BOX_SUGGEST", "0")
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "0"  # OS env 우선
