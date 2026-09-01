"""live_dir 고정 사본 계약: 알람마다 run_dir 이 바뀌어도 같은 경로에 갱신된다."""

import json
from pathlib import Path

from poc.workflow_4.adapters.workflow3_cycle import (
    CycleGraphMirror,
    build_step_chain_graph,
)


def _write_journal(run_dir: Path, step_id: str, status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": status,
                "started_at": "2026-09-02T00:00:00",
                "step_results": [{"step_id": step_id, "status": "success"}],
            }
        ),
        encoding="utf-8",
    )


def test_live_copy_follows_new_run_dir(tmp_path):
    graph = build_step_chain_graph("cycle", [("s1", "step 1"), ("s2", "step 2")])
    live = tmp_path / "_live"
    run_dir = {"cur": tmp_path / "run_a"}

    mirror = CycleGraphMirror(
        graph, run_dir_fn=lambda: run_dir["cur"], live_dir=live
    )

    _write_journal(run_dir["cur"], "s1", "running")
    assert mirror.poll_once() is not None
    first = (live / "workflow_graph.html").read_text(encoding="utf-8")
    assert (run_dir["cur"] / "workflow_graph.html").is_file()

    # 다음 알람: run_dir 이 통째로 바뀐다. 라이브 경로는 그대로여야 한다.
    run_dir["cur"] = tmp_path / "run_b"
    _write_journal(run_dir["cur"], "s2", "completed")
    assert mirror.poll_once() is not None

    assert (run_dir["cur"] / "workflow_graph.html").is_file()
    assert (live / "workflow_graph.html").read_text(encoding="utf-8") != first
    # 탭이 폴링하는 state.js 도 같이 갱신된다(리로드 없이 반영되는 근거).
    assert (live / "workflow_graph_state.js").is_file()


def test_live_dir_none_is_noop(tmp_path):
    graph = build_step_chain_graph("cycle", [("s1", "step 1")])
    run_dir = tmp_path / "run_a"
    mirror = CycleGraphMirror(graph, run_dir_fn=lambda: run_dir)
    _write_journal(run_dir, "s1", "completed")
    assert mirror.poll_once() is not None
    assert list(tmp_path.iterdir()) == [run_dir]
