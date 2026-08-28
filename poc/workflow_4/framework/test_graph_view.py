"""그래프 뷰 렌더링 테스트 — mermaid / ascii / md snapshot / HTML live view."""

from poc.workflow_4.framework.graph_view import (
    render_ascii,
    render_html,
    render_mermaid,
    write_graph_html,
    write_graph_snapshot,
)
from poc.workflow_4.framework.run_state import RunState, TransitionRecord
from poc.workflow_4.framework.state_graph import NodeKind, WorkflowGraph, WorkflowNode


def _graph() -> WorkflowGraph:
    graph = WorkflowGraph(name="view_graph", entry_node_id="a")
    graph.add_node(
        WorkflowNode("a", "노드 A", default_next="b", failure_routes={"boom": "c"})
    )
    graph.add_node(WorkflowNode("b", "노드 B", default_next="done"))
    graph.add_node(
        WorkflowNode("c", "폴백 C", kind=NodeKind.FALLBACK, default_next="a")
    )
    graph.add_node(WorkflowNode("done", "완료", kind=NodeKind.TERMINAL))
    return graph


def _state() -> RunState:
    return RunState(run_id="r1", graph_name="view_graph", current_node="b")


def test_render_mermaid_contains_required_parts():
    out = render_mermaid(_graph(), _state())
    assert "stateDiagram-v2" in out
    assert "classDef active fill:#ffd54f,stroke:#f57f17" in out
    assert "class b active" in out
    assert "a --> b: success" in out
    assert "a --> c: boom" in out


def test_render_ascii_marks_current_node():
    out = render_ascii(_graph(), _state())
    assert "* b" in out
    assert "(attempt 1/3)" in out  # max_retries=3 기본값


def test_write_graph_snapshot_writes_files(tmp_path):
    state = _state()
    state.history.append(
        TransitionRecord(
            seq=1,
            ts="2026-08-28T10:00:00",
            from_node="a",
            to_node="b",
            event="success",
            attempt=1,
        )
    )
    path = write_graph_snapshot(tmp_path, _graph(), state)
    assert path.is_file()
    assert path.name == "workflow_graph.md"
    md = path.read_text(encoding="utf-8")
    assert "stateDiagram-v2" in md
    assert "## History" in md
    assert "a" in md and "b" in md
    # HTML 도 같은 호출로 함께 써진다 (live view).
    assert (tmp_path / "workflow_graph.html").is_file()


def test_render_html_contains_required_parts():
    out = render_html(_graph(), _state())
    assert "mermaid.initialize" in out
    assert "startOnLoad:true" in out
    assert "classDef active fill:#ffd54f,stroke:#f57f17" in out
    assert "class b active" in out
    assert "stateDiagram-v2" in out
    assert '<meta http-equiv="refresh" content="1">' in out
    assert "<table>" in out
    assert "view_graph" in out


def test_render_html_history_rows():
    state = _state()
    state.history.append(
        TransitionRecord(
            seq=1,
            ts="2026-08-28T10:00:00",
            from_node="a",
            to_node="b",
            event="success",
            failure_class=None,
            attempt=1,
        )
    )
    out = render_html(_graph(), state)
    assert "<td>a</td>" in out
    assert "<td>b</td>" in out
    assert "<td>success</td>" in out


def test_render_html_custom_refresh():
    out = render_html(_graph(), _state(), refresh_sec=3)
    assert '<meta http-equiv="refresh" content="3">' in out


def test_write_graph_html_creates_file(tmp_path):
    path = write_graph_html(tmp_path, _graph(), _state())
    assert path.is_file()
    assert path.name == "workflow_graph.html"
    content = path.read_text(encoding="utf-8")
    assert "mermaid.initialize" in content
    assert "stateDiagram-v2" in content