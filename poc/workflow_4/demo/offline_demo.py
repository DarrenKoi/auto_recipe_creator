"""workflow_4 오프라인 데모 — 가짜 align-fail 복구 그래프로 프레임워크 동작 시연.

GUI / VLM / Windows 의존 없이 순수 파이썬 클로저 핸들러로 세 시나리오를 돌린다:

- happy     : 전 노드 성공 경로 → COMPLETED
- fallback  : locate_sem_panel 1회 실패(panel_not_found) → zoom_probe fallback → 성공
- escalate  : observe_alarm 반복 실패 → global retry budget 소진 → ESCALATED

화면을 건드릴 코드가 없으므로 SAFE_MODE 게이트도 없다. 산출물은 각각
`debug_images/demo_runs/<scenario>_<ts>/` 에 run_state.json + workflow_graph.md 로
남는다.

실행:
    uv run python poc/workflow_4/demo/offline_demo.py

시나리오 선택은 아래 상수(또는 1회성 env `WF4_DEMO_SCENARIO`)로 한다.
"""

import os
import time
from pathlib import Path

# 어느 시나리오를 돌릴지: "happy" | "fallback" | "escalate" | "all".
DEMO_SCENARIO = "all"

from poc.workflow_4 import DEBUG_IMAGE_DIR
from poc.workflow_4.framework.engine import (
    EngineConfig,
    NodeHandler,
    NodeOutcome,
    WorkflowEngine,
)
from poc.workflow_4.framework.graph_view import print_ascii
from poc.workflow_4.framework.run_state import RunState, RunStatus
from poc.workflow_4.framework.state_graph import (
    NodeKind,
    WorkflowGraph,
    WorkflowNode,
)


def build_align_recovery_graph() -> WorkflowGraph:
    """align_recovery 그래프 — entry observe_alarm, terminal succeeded."""
    graph = WorkflowGraph(name="align_recovery", entry_node_id="observe_alarm")
    graph.add_node(
        WorkflowNode("observe_alarm", "RCS 알람 관찰", max_retries=3,
                     default_next="connect_tool")
    )
    graph.add_node(
        WorkflowNode("connect_tool", "장비 접속", default_next="locate_sem_panel")
    )
    graph.add_node(
        WorkflowNode(
            "locate_sem_panel",
            "SEM 패널 위치 확인",
            max_retries=2,
            retry_cooldown_sec=0.0,
            failure_routes={"panel_not_found": "zoom_probe"},
            default_next="correct_align",
        )
    )
    graph.add_node(
        WorkflowNode(
            "zoom_probe",
            "zoom 탐색으로 패널 재탐색",
            kind=NodeKind.FALLBACK,
            max_retries=2,
            default_next="locate_sem_panel",
        )
    )
    graph.add_node(
        WorkflowNode(
            "correct_align",
            "align key 보정",
            max_retries=2,
            failure_routes={"key_not_visible": "fallback_search"},
            default_next="verify_result",
        )
    )
    graph.add_node(
        WorkflowNode(
            "fallback_search",
            "fallback 탐색으로 key 재탐색",
            kind=NodeKind.FALLBACK,
            max_retries=2,
            default_next="correct_align",
        )
    )
    graph.add_node(
        WorkflowNode(
            "verify_result",
            "보정 결과 검증",
            max_retries=2,
            retry_cooldown_sec=0.0,
            default_next="succeeded",
        )
    )
    graph.add_node(
        WorkflowNode("succeeded", "성공 - 워크플로 종료", kind=NodeKind.TERMINAL)
    )
    return graph


_ALIGN_RECOVERY_NODE_IDS = [
    "observe_alarm",
    "connect_tool",
    "locate_sem_panel",
    "zoom_probe",
    "correct_align",
    "fallback_search",
    "verify_result",
    "succeeded",
]

# node_id -> scripted failure specs. script 는 실패만 기술하고, 소진 후에는
# 기본 success 를 돌려준다 (한 번 실패하고 성공하는 fallback 경로 시연).
_SCENARIO_SCRIPTS: dict[str, dict[str, list[dict]]] = {
    "happy": {},
    "fallback": {
        "locate_sem_panel": [
            {"status": "failed", "failure_class": "panel_not_found"},
        ],
    },
    "escalate": {
        "observe_alarm": [
            {"status": "failed", "failure_class": "observe_error"},
            {"status": "failed", "failure_class": "observe_error"},
            {"status": "failed", "failure_class": "observe_error"},
        ],
    },
}

_SCENARIO_ENGINE_OVERRIDES: dict[str, dict] = {
    # budget 3: observe_alarm 이 3번째 실패에서 global_budget_exhausted 로 상승.
    "escalate": {"global_retry_budget": 3},
}


def make_scripted_handler(specs: list[dict]) -> NodeHandler:
    """주어진 outcome spec 을 하나씩 소비하는 클로저 핸들러.

    spec 이 소진되면 기본 success 를 돌려준다 (script 는 실패만 기술).
    """
    queue = list(specs)
    pos = 0

    def handler(node_id: str, context: dict, run_state: RunState) -> NodeOutcome:
        nonlocal pos
        if pos < len(queue):
            spec = queue[pos]
            pos += 1
            return NodeOutcome(**spec)
        return NodeOutcome(status="success")

    return handler


def build_scenario_handlers(scenario_name: str) -> dict[str, NodeHandler]:
    """시나리오의 모든 노드에 scripted 핸들러를 만들어 반환한다."""
    script = _SCENARIO_SCRIPTS.get(scenario_name, {})
    return {
        node_id: make_scripted_handler(script.get(node_id, []))
        for node_id in _ALIGN_RECOVERY_NODE_IDS
    }


def run_scenario(
    scenario_name: str,
    persist_root: Path | None = None,
    engine_overrides: dict | None = None,
) -> tuple[RunState, Path]:
    """시나리오 하나를 실행하고 (RunState, persist_dir) 를 반환한다."""
    graph = build_align_recovery_graph()
    handlers = build_scenario_handlers(scenario_name)
    timestamp = time.strftime("%y%m%d_%H%M%S")
    persist_dir = (
        persist_root or (DEBUG_IMAGE_DIR / "demo_runs")
    ) / f"{scenario_name}_{timestamp}"

    overrides = dict(_SCENARIO_ENGINE_OVERRIDES.get(scenario_name, {}))
    overrides.update(engine_overrides or {})
    config = EngineConfig(persist_dir=persist_dir, **overrides)
    engine = WorkflowEngine(graph, handlers, config=config)

    print(
        f"[INFO] [WF4] === scenario '{scenario_name}' start "
        f"(persist_dir={persist_dir}) ==="
    )
    state = engine.run()
    print_ascii(graph, state)
    print(
        f"[INFO] [WF4] === scenario '{scenario_name}' done: "
        f"status={state.status.value} current={state.current_node} "
        f"failure_class={state.failure_class} ==="
    )
    print(f"[INFO] [WF4]     artifact: {persist_dir / 'run_state.json'}")
    print(f"[INFO] [WF4]     artifact: {persist_dir / 'workflow_graph.md'}")
    return state, persist_dir


def main() -> None:
    """DEMO_SCENARIO(env WF4_DEMO_SCENARIO 가 우선)로 시나리오를 골라 데모를 실행한다."""
    scenarios = ["happy", "fallback", "escalate"]
    choice = os.environ.get("WF4_DEMO_SCENARIO", "").strip() or DEMO_SCENARIO
    print(f"[INFO] [WF4] workflow_4 offline demo (scenario={choice!r})")
    if choice != "all":
        if choice not in scenarios:
            print(f"[ERROR] [WF4] scenario {choice!r} 는 {scenarios} 또는 'all' 중 하나여야 합니다.")
            return
        scenarios = [choice]
    for scenario_name in scenarios:
        run_scenario(scenario_name)
    print(
        "[INFO] [WF4] demo complete. "
        "See poc/workflow_4/debug_images/demo_runs/ for artifacts."
    )


if __name__ == "__main__":
    main()