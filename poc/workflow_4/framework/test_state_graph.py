"""state_graph.validate() 동작 테스트 — 미지 target / default_next 누락 / terminal 미도달."""

from poc.workflow_4.framework.state_graph import NodeKind, WorkflowGraph, WorkflowNode


def test_validate_catches_empty_graph():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    problems = graph.validate()
    assert problems
    assert any("no nodes" in p for p in problems)


def test_validate_catches_unknown_default_target():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="missing"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    problems = graph.validate()
    assert any("missing" in p for p in problems)


def test_validate_catches_unknown_failure_route_target():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(
        WorkflowNode("a", "A", failure_routes={"boom": "nope"}, default_next="done")
    )
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    problems = graph.validate()
    assert any("nope" in p for p in problems)


def test_validate_catches_missing_default_next_on_non_terminal():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A"))  # non-terminal, default_next 없음
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    problems = graph.validate()
    assert any("default_next" in p for p in problems)


def test_validate_catches_unreachable_terminal():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="b"))
    graph.add_node(WorkflowNode("b", "B", default_next="a"))  # terminal 우회 사이클
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    problems = graph.validate()
    assert any("terminal" in p for p in problems)


def test_validate_ok_graph_passes():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    assert graph.validate() == []


def test_add_node_rejects_duplicate():
    graph = WorkflowGraph(name="g", entry_node_id="a")
    graph.add_node(WorkflowNode("a", "A", default_next="done"))
    graph.add_node(WorkflowNode("done", "Done", kind=NodeKind.TERMINAL))
    try:
        graph.add_node(WorkflowNode("a", "duplicate"))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected ValueError")