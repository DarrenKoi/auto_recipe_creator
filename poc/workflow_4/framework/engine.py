"""워크플로 엔진 — 유계(bounded) 상태 머신 실행 루프.

성공/스킵 라우팅, 실패 유형별 fallback 라우팅, in-place retry, global retry
budget, abort_check 폴링을 처리한다. 모든 전이마다 run_state.json 과
workflow_graph.md(live view) 를 persist_dir 에 덮어쓴다.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from poc.workflow_4 import DEBUG_IMAGE_DIR
from poc.workflow_4.framework.graph_view import write_graph_snapshot, write_text_atomic
from poc.workflow_4.framework.run_state import (
    RunState,
    RunStatus,
    TransitionRecord,
    now_iso,
)
from poc.workflow_4.framework.state_graph import NodeKind, WorkflowGraph

# 안전장치: 성공 간선 사이클이 terminal 을 우회하면 이 상한에서 멈춘다.
# 정상 경로는 전이 수가 수십 개 이하이므로 1000 은 절대 안 걸린다.
MAX_TRANSITIONS = 1000


@dataclass
class NodeOutcome:
    """노드 핸들러가 돌려주는 실행 결과."""

    status: str  # "success" | "failed" | "skipped"
    failure_class: str | None = None
    note: str | None = None
    goto: str | None = None  # 성공 시 명시적 next-node override


NodeHandler = Callable[[str, dict, RunState], NodeOutcome]


@dataclass
class EngineConfig:
    """엔진 실행 파라미터."""

    global_retry_budget: int = 20  # sum(node_retries) 상한
    persist_dir: Path | None = None  # None 이면 DEBUG_IMAGE_DIR / run_<run_id>
    pause_on_escalate: bool = True  # ESCALATED 시 상태 기록 후 멈춘다 (v1 기본/유일 동작)
    abort_check: Callable[[], bool] | None = None  # True → ABORTED (노드 실행 전 + cooldown 중 폴링)
    abort_poll_sec: float = 0.1  # cooldown sleep 을 이 간격으로 쪼개 abort_check 를 본다


class WorkflowEngine:
    """워크플로 그래프를 따라 유계 실행하는 상태 머신 엔진."""

    def __init__(
        self,
        graph: WorkflowGraph,
        handlers: dict[str, NodeHandler],
        config: EngineConfig | None = None,
    ):
        self.graph = graph
        self.handlers = dict(handlers)
        self.config = config or EngineConfig()
        self._persist_dir: Path | None = None

    # ------------------------------------------------------------------ run

    def run(
        self,
        context: dict | None = None,
        run_id: str | None = None,
    ) -> RunState:
        """그래프 검증 후 entry 노드부터 실행한다. 최종 RunState 를 반환."""
        problems = self.graph.validate()
        if problems:
            raise ValueError(
                f"workflow graph {self.graph.name!r} validation failed: "
                + "; ".join(problems)
            )

        resolved_run_id = run_id or time.strftime("wf4_%y%m%d_%H%M%S")
        self._persist_dir = self.config.persist_dir or (
            DEBUG_IMAGE_DIR / f"run_{resolved_run_id}"
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        ctx = dict(context or {})
        ctx.setdefault("run_id", resolved_run_id)
        ctx.setdefault("graph_name", self.graph.name)
        ctx.setdefault("persist_dir", self._persist_dir)
        # 핸들러가 노드 **안에서**(pan 루프 등) 같은 latch 를 볼 수 있게 넘긴다 -
        # 엔진은 노드 사이에서만 폴링하므로 긴 노드의 중단은 핸들러 몫이다.
        ctx.setdefault("abort_check", self.config.abort_check)

        run_state = RunState(
            run_id=resolved_run_id,
            graph_name=self.graph.name,
            current_node=self.graph.entry_node_id,
            status=RunStatus.RUNNING,
        )
        # 첫 전이 전에도 live view 가 존재하도록 초기 스냅샷을 쓴다.
        self._persist(run_state)

        transitions = 0
        while run_state.status is RunStatus.RUNNING:
            node = self.graph.nodes[run_state.current_node]

            # abort_check 는 각 노드 실행 전에 폴링한다.
            if self.config.abort_check is not None and self.config.abort_check():
                self._abort(run_state, node)
                break

            if transitions >= MAX_TRANSITIONS:
                self._escalate(
                    run_state,
                    node,
                    "step_budget_exhausted",
                    f"safety transition cap {MAX_TRANSITIONS} exceeded",
                )
                break

            handler = self.handlers.get(node.node_id)
            if handler is None:
                if node.kind is NodeKind.TERMINAL:
                    # 핸들러 없는 terminal 도달 → 그대로 완료.
                    self._complete(
                        run_state, node, note="terminal reached (no handler)"
                    )
                else:
                    self._escalate(
                        run_state,
                        node,
                        "handler_missing",
                        f"no handler registered for node {node.node_id!r}",
                    )
                break

            try:
                outcome = handler(node.node_id, ctx, run_state)
            except Exception as exc:  # 핸들러 예외는 failed 로 흡수 (무한 루프 방지)
                outcome = NodeOutcome(
                    status="failed", failure_class="handler_exception", note=str(exc)
                )
            if not isinstance(outcome, NodeOutcome):
                outcome = NodeOutcome(
                    status="failed",
                    failure_class="invalid_outcome",
                    note=f"handler returned {type(outcome).__name__}",
                )

            if node.kind is NodeKind.TERMINAL:
                self._handle_terminal(run_state, node, outcome)
                break

            if outcome.status in ("success", "skipped"):
                self._advance(run_state, node, outcome)
            elif outcome.status == "failed":
                if self._handle_failure(run_state, node, outcome):
                    break
            else:
                invalid = NodeOutcome(
                    status="failed",
                    failure_class="invalid_outcome",
                    note=f"unknown status {outcome.status!r}",
                )
                if self._handle_failure(run_state, node, invalid):
                    break

            transitions += 1

        # finished_at 이 반영된 최종 상태를 한 번 더 기록한다.
        self._persist(run_state)
        return run_state

    # ----------------------------------------------------------- 라우팅 헬퍼

    def _advance(self, run_state: RunState, node, outcome: NodeOutcome) -> None:
        """성공/스킵 라우팅 — goto > default_next 우선."""
        event = outcome.status
        target_id = outcome.goto or node.default_next
        if target_id is None:
            self._escalate(
                run_state,
                node,
                "no_default_next",
                f"node {node.node_id!r} succeeded but has no default_next",
            )
            return
        if target_id not in self.graph.nodes:
            self._escalate(
                run_state,
                node,
                "unknown_target",
                f"target {target_id!r} not in graph",
            )
            return
        self._on_transition(
            run_state,
            self._make_record(
                run_state,
                from_node=node.node_id,
                to_node=target_id,
                event=event,
                attempt=run_state.attempt,
                note=outcome.note,
            ),
        )
        run_state.current_node = target_id
        run_state.attempt = 1  # 새 노드 진입 시 per-node attempt 카운터 리셋.

    def _handle_failure(
        self, run_state: RunState, node, outcome: NodeOutcome
    ) -> bool:
        """실패 라우팅. True 면 루프 종료(escalate), False 면 계속 진행."""
        node_id = node.node_id
        failure_class = outcome.failure_class
        run_state.node_retries[node_id] = (
            run_state.node_retries.get(node_id, 0) + 1
        )

        # global retry budget 먼저 확인.
        if sum(run_state.node_retries.values()) >= self.config.global_retry_budget:
            self._escalate(
                run_state,
                node,
                "global_budget_exhausted",
                f"global retry budget {self.config.global_retry_budget} exhausted",
            )
            return True

        # 1. failure_class 에 대한 fallback route 가 있으면 그쪽으로.
        if failure_class and failure_class in node.failure_routes:
            target_id = node.failure_routes[failure_class]
            target = self.graph.nodes[target_id]
            # 방문 횟수는 node_retries(실패 횟수)와 **다른** 카운터다 - 같이 세면
            # fallback 노드가 라우팅되는 순간 자기 retry 예산 1을 잃는다.
            run_state.fallback_visits[target_id] = (
                run_state.fallback_visits.get(target_id, 0) + 1
            )
            if run_state.fallback_visits[target_id] > target.max_retries:
                self._escalate(
                    run_state,
                    node,
                    failure_class,
                    f"fallback node {target_id!r} visited more than "
                    f"{target.max_retries} times",
                )
                return True
            self._on_transition(
                run_state,
                self._make_record(
                    run_state,
                    from_node=node_id,
                    to_node=target_id,
                    event="fallback",
                    failure_class=failure_class,
                    attempt=run_state.attempt,
                    note=outcome.note,
                ),
            )
            run_state.current_node = target_id
            run_state.attempt = 1
            return False

        # 2. 아니면 같은 노드에서 in-place retry.
        if run_state.node_retries[node_id] <= node.max_retries:
            if self._cooldown_aborted(node.retry_cooldown_sec):
                self._abort(run_state, node)
                return True
            run_state.attempt += 1
            self._on_transition(
                run_state,
                self._make_record(
                    run_state,
                    from_node=node_id,
                    to_node=node_id,
                    event="retry",
                    failure_class=failure_class,
                    attempt=run_state.attempt,
                    note=outcome.note,
                ),
            )
            return False

        # 3. retry 예산 소진 → escalate.
        self._escalate(
            run_state,
            node,
            failure_class,
            f"node retry budget exhausted (max_retries={node.max_retries})",
        )
        return True

    def _cooldown_aborted(self, cooldown_sec: float) -> bool:
        """cooldown 동안 잠들되 abort_check 를 계속 본다. True 면 중단 요청."""
        deadline = time.monotonic() + max(0.0, cooldown_sec)
        check = self.config.abort_check
        while True:
            if check is not None and check():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, max(0.01, self.config.abort_poll_sec)))

    def _handle_terminal(
        self, run_state: RunState, node, outcome: NodeOutcome
    ) -> None:
        """terminal 노드의 outcome 은 기록만 하고 COMPLETED 로 끝난다."""
        event = outcome.status if outcome.status in ("success", "skipped", "failed") else "success"
        run_state.status = RunStatus.COMPLETED
        run_state.finished_at = now_iso()
        if outcome.failure_class:
            run_state.failure_class = outcome.failure_class
        run_state.note = outcome.note or f"terminal {node.node_id!r} outcome recorded"
        self._on_transition(
            run_state,
            self._make_record(
                run_state,
                from_node=node.node_id,
                to_node=node.node_id,
                event=event,
                failure_class=outcome.failure_class,
                attempt=run_state.attempt,
                note=outcome.note,
            ),
        )

    # ------------------------------------------------------- 상태 전이 기록

    def _abort(self, run_state: RunState, node) -> None:
        """abort_check 가 True 인 경우 — ABORTED 로 종료."""
        run_state.status = RunStatus.ABORTED
        run_state.finished_at = now_iso()
        run_state.note = "abort_check triggered"
        self._on_transition(
            run_state,
            self._make_record(
                run_state,
                from_node=node.node_id,
                to_node=node.node_id,
                event="abort",
                attempt=run_state.attempt,
                note="abort_check triggered",
            ),
        )

    def _complete(self, run_state: RunState, node, note: str | None = None) -> None:
        """핸들러 없는 terminal 도달 — COMPLETED."""
        run_state.status = RunStatus.COMPLETED
        run_state.finished_at = now_iso()
        run_state.note = note
        self._on_transition(
            run_state,
            self._make_record(
                run_state,
                from_node=node.node_id,
                to_node=node.node_id,
                event="success",
                attempt=run_state.attempt,
                note=note,
            ),
        )

    def _escalate(
        self,
        run_state: RunState,
        node,
        failure_class: str | None,
        note: str,
    ) -> None:
        """ESCALATED 로 종료. v1 은 resume 이 없어 항상 여기서 멈춘다."""
        if not self.config.pause_on_escalate:
            print(
                "[WARNING] pause_on_escalate=False 는 v1 에서 미지원입니다. "
                "ESCALATED 로 종료합니다."
            )
        run_state.status = RunStatus.ESCALATED
        run_state.failure_class = failure_class
        run_state.finished_at = now_iso()
        run_state.note = note
        self._on_transition(
            run_state,
            self._make_record(
                run_state,
                from_node=node.node_id,
                to_node=node.node_id,
                event="escalate",
                failure_class=failure_class,
                attempt=run_state.attempt,
                note=note,
            ),
        )

    def _make_record(
        self,
        run_state: RunState,
        *,
        from_node: str,
        to_node: str,
        event: str,
        failure_class: str | None = None,
        attempt: int | None = None,
        note: str | None = None,
    ) -> TransitionRecord:
        return TransitionRecord(
            seq=len(run_state.history) + 1,
            ts=now_iso(),
            from_node=from_node,
            to_node=to_node,
            event=event,
            failure_class=failure_class,
            attempt=attempt if attempt is not None else run_state.attempt,
            note=note,
        )

    def _on_transition(
        self, run_state: RunState, record: TransitionRecord
    ) -> None:
        """모든 전이 뒤 호출 — 로그 출력 + run_state.json + graph snapshot."""
        run_state.history.append(record)
        print(
            f"[INFO] {self.graph.name} {record.from_node} -> "
            f"{record.to_node} ({record.event}, {record.attempt})"
        )
        self._persist(run_state)

    # ------------------------------------------------------------- persistence

    def _persist(self, run_state: RunState) -> None:
        """run_state.json + workflow_graph.md 를 persist_dir 에 덮어쓴다."""
        persist_dir = self._persist_dir
        if persist_dir is None:
            return
        persist_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            persist_dir / "run_state.json",
            json.dumps(run_state.to_json_dict(), ensure_ascii=False, indent=2),
        )
        write_graph_snapshot(persist_dir, self.graph, run_state)