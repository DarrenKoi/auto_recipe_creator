"""cycle3 mirror 데모 — 가짜 저널을 몇 초에 걸쳐 만들어 live view 를 확인한다.

workflow_3 runner 없이 run_dir 저널(run_state.json + step_<id>.json)을 직접
합성해 CycleGraphMirror 가 workflow_graph.md/.html 스냅샷을 쓰는 과정을 보여준다.
offline / 안전 / workflow_3 import 없음. SAFE_MODE 와 무관하게 화면을 건드리지
않는다.

실행:
    uv run python poc/workflow_4/adapters/run_cycle3_mirror_demo.py

산출물: `poc/workflow_4/debug_images/demo_runs/mirror_demo_<ts>/`
  - workflow_graph.md   (mermaid + history 테이블)
  - workflow_graph.html (self-contained live view — 브라우저로 직접 열기)
"""

import json
import time
from pathlib import Path

from poc.workflow_4 import DEBUG_IMAGE_DIR
from poc.workflow_4.adapters.workflow3_cycle import (
    CycleGraphMirror,
    build_step_chain_graph,
)

# production build_cycle_steps() 의 step 순서를 흉내 낸 데모 픽스처 (import 없이 offline).
DEMO_STEPS = [
    ("ensure_rcs_ready", "RCS 준비(접속 전)"),
    ("close_alert_popup", "감지 팝업 닫기(접속 전)"),
    ("connect_tool", "tool 접속(List 탭 더블클릭)"),
    ("wait_tool_window", "tool 접속(Remote Monitoring 창 대기)"),
    ("start_recording", "녹화 시작"),
    ("locate_sem_panel", "SEM panel 인식(보정 준비)"),
    ("run_correction", "align 보정"),
]
CYCLE3_STEP_IDS = [s[0] for s in DEMO_STEPS]


def _step_result(
    step_id: str,
    status: str = "success",
    failure_class: str | None = None,
    attempt_count: int = 1,
    error_message: str | None = None,
) -> dict:
    """workflow_3 StepResult.to_dict() 와 동일한 형태의 가짜 저널 dict."""
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
        "elapsed_ms": 120.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "safe_mode": True,
    }


def _run_state_json(
    run_id: str,
    status: str,
    step_results: list[dict],
    started_at: str,
    current_step_index: int = -1,
    finished_at: str | None = None,
) -> dict:
    """workflow_3 WorkflowRun.to_dict() 와 동일한 형태의 가짜 저널 dict."""
    return {
        "run_id": run_id,
        "workflow_name": "align_fail_cycle_demo",
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "current_step_index": current_step_index,
        "total_retries_used": 0,
        "retry_budget_remaining": 0,
        "settings_snapshot": {},
        "interrupts_encountered": [],
        "step_results": step_results,
        "run_dir": "",
    }


def main() -> None:
    run_id = time.strftime("mirror_demo_%y%m%d_%H%M%S")
    run_dir: Path = DEBUG_IMAGE_DIR / "demo_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[INFO] cycle3 mirror demo - run_dir={run_dir}")

    mirror = CycleGraphMirror(
        build_step_chain_graph("cycle3_align_fail_demo", DEMO_STEPS),
        run_dir_fn=lambda: run_dir,
        poll_sec=0.2,
        refresh_sec=1,
    )
    mirror.start()

    def _write_state(status, results, index, finished_at=None):
        (run_dir / "run_state.json").write_text(
            json.dumps(
                _run_state_json(run_id, status, results, started_at, index, finished_at),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    _write_state("running", [], -1)
    time.sleep(1.0)

    results: list[dict] = []
    for i, step_id in enumerate(CYCLE3_STEP_IDS):
        time.sleep(1.0)
        result = _step_result(step_id)
        results.append(result)
        (run_dir / f"step_{step_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_state("running", list(results), i)
        print(f"[INFO] journal: {step_id} complete")

    time.sleep(1.0)
    _write_state(
        "completed",
        list(results),
        len(CYCLE3_STEP_IDS) - 1,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    time.sleep(1.0)
    mirror.stop(final=True)

    print("[INFO] cycle3 mirror demo done")
    print(f"[INFO]   md   : {run_dir / 'workflow_graph.md'}")
    print(f"[INFO]   html : {run_dir / 'workflow_graph.html'}")


if __name__ == "__main__":
    main()