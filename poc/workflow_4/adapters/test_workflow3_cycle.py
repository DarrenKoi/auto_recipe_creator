"""cycle3 mirror adapter 테스트 — 그래프 스펙 + 저널 미러링 (offline, tmp_path).

가짜 저널 JSON 으로 mirror 동작을 검증하고, 마지막 테스트 하나만 workflow_3 의
실제 producer(`build_cycle_steps`/`build_check_steps`)를 import 해 그래프가 실제
step 목록으로 만들어지는지 확인한다(손으로 적은 step 목록은 production 과
어긋나도 테스트가 통과하기 때문).
"""

import json
import time
from pathlib import Path

import pytest

from poc.workflow_4.adapters.workflow3_cycle import (
    CycleGraphMirror,
    build_step_chain_graph,
)
from poc.workflow_4.framework.run_state import RunStatus
from poc.workflow_4.framework.state_graph import NodeKind

# 테스트 픽스처 — production 순서와 같지만 여기서 갈라져도 mirror 로직 검증엔 무관.
STEPS = [
    ("ensure_rcs_ready", "RCS 준비"),
    ("close_alert_popup", "팝업 닫기"),
    ("connect_tool", "tool 접속"),
    ("wait_tool_window", "창 대기"),
    ("start_recording", "녹화"),
    ("locate_sem_panel", "SEM panel"),
    ("run_correction", "보정"),
]
STEP_IDS = [s[0] for s in STEPS]


def _graph():
    return build_step_chain_graph("cycle3_test", STEPS)


def _step_file(
    step_id: str,
    status: str = "success",
    failure_class: str | None = None,
    attempt_count: int = 1,
    error_message: str | None = None,
) -> dict:
    """workflow_3 StepResult.to_dict() 형태의 가짜 저널."""
    return {
        "step_id": step_id,
        "status": status,
        "failure_class": failure_class,
        "attempt_count": attempt_count,
        "strategy_used": "align_fail_cycle",
        "vlm_service_used": "",
        "detected_point": None,
        "detected_bbox": None,
        "screen_point": None,
        "verification_result": None,
        "before_screenshot": None,
        "after_screenshot": None,
        "error_message": error_message,
        "elapsed_ms": 10.0,
        "timestamp": "2026-08-28T10:00:00",
        "safe_mode": True,
    }


def _run_state_json(
    run_id: str,
    status: str,
    step_results: list[dict],
    current_step_index: int = -1,
    finished_at: str | None = None,
) -> dict:
    """workflow_3 WorkflowRun.to_dict() 형태의 가짜 저널."""
    return {
        "run_id": run_id,
        "workflow_name": "align_fail_cycle_EQP1",
        "status": status,
        "started_at": "2026-08-28T10:00:00",
        "finished_at": finished_at,
        "current_step_index": current_step_index,
        "total_retries_used": 0,
        "retry_budget_remaining": 20,
        "settings_snapshot": {},
        "interrupts_encountered": [],
        "step_results": step_results,
        "run_dir": "",
    }


def _write_journal(run_dir: Path, payload: dict) -> None:
    (run_dir / "run_state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_step(run_dir: Path, step_id: str, **kwargs) -> None:
    (run_dir / f"step_{step_id}.json").write_text(
        json.dumps(_step_file(step_id, **kwargs), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- 그래프 스펙


def test_step_chain_graph_validates_and_chains():
    graph = _graph()
    assert graph.validate() == []
    for prev, nxt in zip(STEP_IDS, STEP_IDS[1:] + ["succeeded"]):
        assert graph.nodes[prev].default_next == nxt
    for step_id in STEP_IDS:
        assert graph.nodes[step_id].failure_routes == {"failed": "aborted"}
    for term in ("succeeded", "aborted", "teardown"):
        assert graph.nodes[term].kind is NodeKind.TERMINAL


def test_step_chain_graph_rejects_empty():
    with pytest.raises(ValueError):
        build_step_chain_graph("empty", [])


# ---------------------------------------------------------------- 미러 동작


def test_mirror_advances_and_writes_snapshots(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mirror = CycleGraphMirror(_graph(), run_dir_fn=lambda: run_dir, poll_sec=0.05)

    # 1) running, step 1 완료 → in-flight = close_alert_popup.
    _write_journal(run_dir, _run_state_json("r1", "running", [], -1))
    _write_step(run_dir, "ensure_rcs_ready")
    state = mirror.poll_once()
    assert state is not None
    assert state.status is RunStatus.RUNNING
    assert state.current_node == "close_alert_popup"
    assert (run_dir / "workflow_graph.md").is_file()
    assert (run_dir / "workflow_graph.html").is_file()
    assert "mermaid.initialize" in (run_dir / "workflow_graph.html").read_text(
        encoding="utf-8"
    )
    assert not list(run_dir.glob("*.tmp"))  # 원자적 쓰기의 임시 파일이 남지 않는다

    # 2) 변경 없음 → 스냅샷 재작성 없음(시그니처 동일).
    md_mtime = (run_dir / "workflow_graph.md").stat().st_mtime
    time.sleep(0.01)
    state2 = mirror.poll_once()
    assert state2 is not None and state2.status is RunStatus.RUNNING
    assert (run_dir / "workflow_graph.md").stat().st_mtime == md_mtime

    # 3) aborted 로 전환 → ABORTED + failure_class + current=aborted.
    _write_journal(
        run_dir,
        _run_state_json(
            "r1",
            "aborted",
            [
                _step_file("ensure_rcs_ready"),
                _step_file("connect_tool", "failed", "connect_error"),
            ],
            current_step_index=2,
            finished_at="2026-08-28T10:00:05",
        ),
    )
    state3 = mirror.poll_once()
    assert state3 is not None
    assert state3.status is RunStatus.ABORTED
    assert state3.current_node == "aborted"
    assert state3.failure_class == "connect_error"
    assert state3.history[-1].event == "abort"

    mirror.stop(final=True)
    assert (run_dir / "workflow_graph.md").is_file()
    assert (run_dir / "workflow_graph.html").is_file()


def test_mirror_completed_maps_to_succeeded(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mirror = CycleGraphMirror(_graph(), run_dir_fn=lambda: run_dir, poll_sec=0.05)
    results = [_step_file(step_id) for step_id in STEP_IDS]
    _write_journal(
        run_dir,
        _run_state_json(
            "r2", "completed", results,
            current_step_index=len(STEP_IDS) - 1,
            finished_at="2026-08-28T10:00:10",
        ),
    )
    state = mirror.poll_once()
    assert state is not None
    assert state.status is RunStatus.COMPLETED
    assert state.current_node == "succeeded"
    assert len(state.history) == len(STEP_IDS)
    mirror.stop(final=False)


def test_mirror_waits_for_run_dir_then_follows_it(tmp_path):
    """run() 전에 start 해도 run_dir 이 생길 때까지 폴만 건너뛰고, 생기면 그 폴더를 쓴다."""
    context: dict = {}
    mirror = CycleGraphMirror(_graph(), run_dir_fn=lambda: context.get("run_dir"), poll_sec=0.05)
    assert mirror.poll_once() is None  # 아직 run_dir 없음

    actual = tmp_path / "111111_111111_align_fail_cycle_EQP9"
    actual.mkdir()
    _write_journal(actual, _run_state_json("r9", "running", [], -1))
    _write_step(actual, "ensure_rcs_ready")
    context["run_dir"] = actual  # runner.run() 이 하는 일
    state = mirror.poll_once()
    assert state is not None
    assert state.status is RunStatus.RUNNING
    assert state.current_node == "close_alert_popup"
    assert (actual / "workflow_graph.md").is_file()
    assert (actual / "workflow_graph.html").is_file()
    mirror.stop(final=False)


def test_mirror_start_stop_thread_writes_final(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_journal(run_dir, _run_state_json("r3", "running", [], -1))
    _write_step(run_dir, "ensure_rcs_ready")

    mirror = CycleGraphMirror(_graph(), run_dir_fn=lambda: run_dir, poll_sec=0.02)
    mirror.start()
    time.sleep(0.3)  # 폴링 스레드가 몇 번 돌도록 대기
    mirror.stop(final=True)
    assert (run_dir / "workflow_graph.md").is_file()
    assert (run_dir / "workflow_graph.html").is_file()
    assert mirror._thread is None  # 스레드 종료 확인
    # stop 은 idempotent.
    mirror.stop(final=True)


# ------------------------------------------------- 실제 producer 와의 접합


def test_real_wf3_steps_build_valid_graphs():
    """cycle.py 의 실제 step 목록으로 그래프가 만들어지고 validate 를 통과한다."""
    cycle = pytest.importorskip("poc.workflow_3.monitor.cycle")
    for builder in (cycle.build_cycle_steps, cycle.build_check_steps):
        steps = builder("EQP1")
        graph = build_step_chain_graph("x", [(s.step_id, s.target_description) for s in steps])
        assert graph.validate() == []
        assert graph.entry_node_id == steps[0].step_id
        assert graph.nodes[steps[-1].step_id].default_next == "succeeded"
