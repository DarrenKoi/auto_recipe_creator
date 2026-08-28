"""offline_demo 시나리오 실행 검증 — happy path 는 tmp dir 에서 완료+산출물 확인."""

from pathlib import Path

from poc.workflow_4.demo.offline_demo import run_scenario
from poc.workflow_4.framework.run_state import RunStatus


def test_happy_scenario_completes_with_artifacts(tmp_path: Path):
    state, persist_dir = run_scenario("happy", persist_root=tmp_path)
    assert state.status is RunStatus.COMPLETED
    assert persist_dir.is_dir()
    assert (persist_dir / "run_state.json").is_file()
    assert (persist_dir / "workflow_graph.md").is_file()


def test_fallback_scenario_records_fallback_event(tmp_path: Path):
    state, persist_dir = run_scenario("fallback", persist_root=tmp_path)
    assert state.status is RunStatus.COMPLETED
    events = [r.event for r in state.history]
    assert "fallback" in events
    assert persist_dir.is_dir()


def test_escalate_scenario_hits_global_budget(tmp_path: Path):
    state, persist_dir = run_scenario("escalate", persist_root=tmp_path)
    assert state.status is RunStatus.ESCALATED
    assert state.failure_class == "global_budget_exhausted"
    assert persist_dir.is_dir()