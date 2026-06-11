from poc.workflow_3.config import load_workflow3_settings


def test_consensus_defaults():
    s = load_workflow3_settings()
    assert s.consensus_enabled is True
    assert s.consensus_min_s == 4
    assert s.gather_max_events == 8
    assert s.consensus_sync_timeout_sec == 8.0
    assert s.consensus_refresh_ttl_sec == 21600


if __name__ == "__main__":
    test_consensus_defaults(); print("1/1 pass")
