"""워크플로 엔진 동작 테스트 — happy/fallback/retry/escalate/abort/persist."""

import json

import pytest

from poc.workflow_4.framework.engine import EngineConfig, NodeOutcome, WorkflowEngine
from poc.workflow_4.framework.run_state import RunState, RunStatus
from poc.workflow_4.framework.state_graph import NodeKind, WorkflowGraph, WorkflowNode


def _simple_graph() -> WorkflowGraph:
    graph = WorkflowGraph(name="simple", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="b"))
    graph.add_node(WorkflowNode("b", "B", default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    return graph


def _ok_handler(node_id, context, run_state):
    return NodeOutcome(status="success")


def _scripted(specs):
    """spec 을 하나씩 소비하는 클로저 핸들러 (소진 후 success)."""
    queue = list(specs)
    pos = 0

    def handler(node_id, context, run_state):
        nonlocal pos
        if pos < len(queue):
            spec = queue[pos]
            pos += 1
            return NodeOutcome(**spec)
        return NodeOutcome(status="success")

    return handler


def test_happy_path_completes(tmp_path):
    engine = WorkflowEngine(
        _simple_graph(),
        {"a": _ok_handler, "b": _ok_handler},
        EngineConfig(persist_dir=tmp_path),
    )
    state = engine.run()
    assert state.status is RunStatus.COMPLETED
    assert state.current_node == "done"
    assert state.history[-1].event == "success"


def test_invalid_graph_raises_value_error(tmp_path):
    graph = WorkflowGraph(name="bad", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="missing"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    engine = WorkflowEngine(graph, {"a": _ok_handler}, EngineConfig(persist_dir=tmp_path))
    with pytest.raises(ValueError, match="missing"):
        engine.run()


def test_missing_handler_escalates(tmp_path):
    engine = WorkflowEngine(
        _simple_graph(),
        {"a": _ok_handler},  # b 핸들러 없음
        EngineConfig(persist_dir=tmp_path),
    )
    state = engine.run()
    assert state.status is RunStatus.ESCALATED
    assert state.failure_class == "handler_missing"


def _fallback_graph() -> WorkflowGraph:
    graph = WorkflowGraph(name="fb", entry_node_id="a")
    graph.add_node(
        WorkflowNode("a", "A", default_next="b", failure_routes={"boom": "fb_node"})
    )
    graph.add_node(
        WorkflowNode("fb_node", "FB", kind=NodeKind.FALLBACK, max_retries=2,
                     default_next="b")
    )
    graph.add_node(WorkflowNode("b", "B", default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    return graph


def test_failure_routes_to_fallback_and_returns(tmp_path):
    handlers = {
        "a": _scripted([{"status": "failed", "failure_class": "boom"}]),
        "fb_node": _ok_handler,
        "b": _ok_handler,
    }
    engine = WorkflowEngine(_fallback_graph(), handlers, EngineConfig(persist_dir=tmp_path))
    state = engine.run()
    assert state.status is RunStatus.COMPLETED
    assert state.current_node == "done"
    events = [r.event for r in state.history]
    assert "fallback" in events


def _retry_graph() -> WorkflowGraph:
    graph = WorkflowGraph(name="retry", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", max_retries=2, default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    return graph


def test_retry_budget_per_node_respected(tmp_path):
    handlers = {
        "a": _scripted(
            [
                {"status": "failed", "failure_class": "flaky"},
                {"status": "failed", "failure_class": "flaky"},
                {"status": "success"},
            ]
        )
    }
    engine = WorkflowEngine(_retry_graph(), handlers, EngineConfig(persist_dir=tmp_path))
    state = engine.run()
    assert state.status is RunStatus.COMPLETED
    retries = [r for r in state.history if r.event == "retry"]
    assert len(retries) == 2


def test_retry_budget_exhaustion_escalates(tmp_path):
    handlers = {
        "a": _scripted(
            [
                {"status": "failed", "failure_class": "flaky"},
                {"status": "failed", "failure_class": "flaky"},
                {"status": "failed", "failure_class": "flaky"},
            ]
        )
    }
    engine = WorkflowEngine(_retry_graph(), handlers, EngineConfig(persist_dir=tmp_path))
    state = engine.run()
    assert state.status is RunStatus.ESCALATED
    assert state.failure_class == "flaky"
    assert state.history[-1].event == "escalate"


def test_global_budget_exhaustion_escalates(tmp_path):
    handlers = {
        "a": _scripted([{"status": "failed", "failure_class": "flaky"}]),
    }
    engine = WorkflowEngine(
        _retry_graph(), handlers, EngineConfig(persist_dir=tmp_path, global_retry_budget=1)
    )
    state = engine.run()
    assert state.status is RunStatus.ESCALATED
    assert state.failure_class == "global_budget_exhausted"


def test_abort_check_stops_with_aborted(tmp_path):
    def abort_check() -> bool:
        return True

    engine = WorkflowEngine(
        _simple_graph(),
        {"a": _ok_handler, "b": _ok_handler},
        EngineConfig(persist_dir=tmp_path, abort_check=abort_check),
    )
    state = engine.run()
    assert state.status is RunStatus.ABORTED
    assert state.history[-1].event == "abort"


def test_run_state_json_written_and_round_trips(tmp_path):
    engine = WorkflowEngine(
        _simple_graph(),
        {"a": _ok_handler, "b": _ok_handler},
        EngineConfig(persist_dir=tmp_path),
    )
    state = engine.run()

    json_path = tmp_path / "run_state.json"
    assert json_path.is_file()
    restored = RunState.from_json_dict(json.loads(json_path.read_text(encoding="utf-8")))
    assert restored.run_id == state.run_id
    assert restored.graph_name == "simple"
    assert restored.status is RunStatus.COMPLETED
    assert restored.current_node == "done"
    assert restored.finished_at is not None
    assert [r.event for r in restored.history] == [r.event for r in state.history]

def test_abort_during_retry_cooldown_is_immediate(tmp_path):
    """cooldown sleep 중에도 abort_check 를 본다 - 긴급 단축키가 cooldown 만큼 늦으면 안 된다."""
    calls = {"n": 0}

    def abort_check() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2  # 첫 폴(노드 실행 전)은 통과, cooldown 첫 폴에서 abort

    graph = WorkflowGraph(name="cd", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", max_retries=3, retry_cooldown_sec=5.0, default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    handlers = {"a": _scripted([{"status": "failed", "failure_class": "flaky"}])}
    engine = WorkflowEngine(
        graph, handlers, EngineConfig(persist_dir=tmp_path, abort_check=abort_check)
    )
    import time as _t
    t0 = _t.monotonic()
    state = engine.run()
    assert _t.monotonic() - t0 < 2.0  # 5s cooldown 을 기다리지 않는다
    assert state.status is RunStatus.ABORTED


def test_handlers_receive_abort_check_in_context(tmp_path):
    seen = {}

    def handler(node_id, context, run_state):
        seen["abort_check"] = context.get("abort_check")
        return NodeOutcome(status="success")

    check = lambda: False  # noqa: E731
    engine = WorkflowEngine(
        _simple_graph(), {"a": handler, "b": handler},
        EngineConfig(persist_dir=tmp_path, abort_check=check),
    )
    engine.run()
    assert seen["abort_check"] is check


def test_fallback_visit_does_not_consume_fallback_node_retry(tmp_path):
    """fallback 노드로 라우팅되는 것과 그 노드가 실패하는 것은 다른 예산이다."""
    graph = WorkflowGraph(name="fb2", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="b", failure_routes={"boom": "fb"}))
    graph.add_node(WorkflowNode("fb", "FB", kind=NodeKind.FALLBACK, max_retries=1,
                                retry_cooldown_sec=0.0, default_next="b"))
    graph.add_node(WorkflowNode("b", "B", default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    handlers = {
        "a": _scripted([{"status": "failed", "failure_class": "boom"}]),
        "fb": _scripted([{"status": "failed", "failure_class": "flaky"}]),  # 1회 실패 후 성공
        "b": _ok_handler,
    }
    engine = WorkflowEngine(graph, handlers, EngineConfig(persist_dir=tmp_path))
    state = engine.run()
    assert state.status is RunStatus.COMPLETED  # 예전엔 방문 1 + 실패 1 = 2 > 1 로 escalate
    assert state.fallback_visits == {"fb": 1}
    assert state.node_retries == {"a": 1, "fb": 1}
