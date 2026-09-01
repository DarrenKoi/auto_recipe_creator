"""cycle3(workflow_3 알람 사이클) → workflow_4 mirror adapter.

`run_alarm_cycle` / `run_check_only_cycle` 가 runner(WorkflowRunner)로 step 을 순차
실행하며 남기는 저널(run_dir/run_state.json + step_<id>.json)을 **읽기 전용**으로
폴링해 workflow_4 `RunState` + 그래프 스냅샷(mermaid .md + HTML live view)으로
미러링한다. workflow_3 쪽 파일은 절대 건드리지 않는다.

저널 타이밍 (workflow_3/runner/workflow_runner.py 기준):
  * run 시작      : run_state.json (status="running", current_step_index=-1, step_results=[])
  * step 끝마다    : step_<id>.json + run_state.json (step_results 누적,
                     current_step_index = 방금 끝난 step 의 인덱스)
  * run 끝        : run_state.json (status="completed" | "aborted" | ..., finished_at)

따라서 status="running" 일 때 **in-flight 노드** = step_results 개수 번째 step
(= 마지막 완료 step + 1) 이다. runner 는 step 이 "failed"/그 외 성공 아님으로
끝나면 즉시 run.status 를 aborted 로 바꾸고 멈춘다. 현재 노드 매핑:
completed → succeeded terminal, aborted/error → aborted terminal(마지막
failure_class 보존), running → step_results 다음 step.

그래프는 호출부가 넘긴 **실제 step 목록**(`build_cycle_steps()` 의 step_id +
target_description)으로 만든다 - 여기에 step 이름/실패 class 표를 복사해 두면
production 과 어긋난다(실제로 존재하지 않는 failure_class 가 표에 있었다). runner 는
어떤 failure_class 든 즉시 aborted 로 끝내므로 실패 간선은 step 마다 `failed ->
aborted` 하나이고, 실제 class 는 history 의 failure_class 열에 그대로 보인다.

run_dir 은 runner 가 run() 시작 시 만들어 `context["run_dir"]` 에 넣는다. mirror 는
`run_dir_fn` 으로 그 값을 **폴링마다** 읽으므로 run() 전에 start 해도 되고, 경로를
예측하거나 glob 으로 찾을 필요가 없다(예측+glob 은 첫 폴이 runner 의 mkdir 보다
빠르면 직전 run 폴더에 영구히 붙는 경쟁이 있었다).

Bounded: 폴링 루프는 `threading.Event.wait(poll_sec)` 로만 잠들고, `stop()` 이
event 를 세우면 즉시 빠져나온다. thread 타이머는 쓰지 않는다.
"""

import json
import platform
import threading
from pathlib import Path
from typing import Callable, Iterable

from poc.workflow_4.framework.graph_view import (
    open_graph_view,
    write_graph_snapshot,
)
from poc.workflow_4.framework.run_state import (
    RunState,
    RunStatus,
    TransitionRecord,
    now_iso,
)
from poc.workflow_4.framework.state_graph import (
    NodeKind,
    WorkflowGraph,
    WorkflowNode,
)

# (step_id, 설명) 쌍. 호출부는 WorkflowStep 에서 (step_id, target_description) 을 뽑아 넘긴다.
StepSpec = tuple[str, str]


def build_step_chain_graph(name: str, steps: Iterable[StepSpec]) -> WorkflowGraph:
    """step 목록을 success chain 으로 연결한 미러 그래프를 만든다.

    마지막 step → succeeded(terminal), 각 step 의 `failed` → aborted(terminal),
    그리고 teardown 은 runner 의 finally 가 항상 실행하므로 succeeded/aborted
    양쪽에서 도달하는 terminal 로 모델링한다(저널에는 teardown 이 없다 — 그래프
    표시용 노드다).
    """
    specs = list(steps)
    if not specs:
        raise ValueError("mirror graph needs at least one step")
    graph = WorkflowGraph(name=name, entry_node_id=specs[0][0])
    for i, (step_id, description) in enumerate(specs):
        default_next = specs[i + 1][0] if i + 1 < len(specs) else "succeeded"
        graph.add_node(
            WorkflowNode(
                step_id,
                description or step_id,
                max_retries=1,  # cycle step 은 runner 에서 1회 실행(실패 시 abort)
                failure_routes={"failed": "aborted"},
                default_next=default_next,
            )
        )
    graph.add_node(
        WorkflowNode(
            "succeeded", "성공 - 사이클 완료",
            kind=NodeKind.TERMINAL, default_next="teardown",
        )
    )
    graph.add_node(
        WorkflowNode(
            "aborted", "실패 - 사이클 중단",
            kind=NodeKind.TERMINAL, default_next="teardown",
        )
    )
    graph.add_node(
        WorkflowNode(
            "teardown", "teardown (runner finally - 항상 실행)",
            kind=NodeKind.TERMINAL,
        )
    )
    return graph


class CycleGraphMirror:
    """cycle3 저널을 폴링해 workflow_4 RunState + 그래프 스냅샷으로 미러링한다.

    Parameters:
      graph       — build_step_chain_graph() 결과.
      run_dir_fn  — 저널 디렉터리를 돌려주는 콜러블. 아직 없으면 None (그 폴은
                    건너뛴다). production 은 `lambda: context.get("run_dir")`.
      persist_dir — 스냅샷(.md/.html)을 쓸 곳. None 이면 run_dir 과 동일(저널 옆).
      poll_sec    — 폴링 주기(초, 하한 0.05).
      refresh_sec — HTML meta refresh 주기(초, 하한 1).
      autoopen    — 첫 스냅샷 후 HTML 을 기본 브라우저로 연다. **Windows 에서만**
                    동작한다(오피스 PC 전용 편의 - Mac dry-run 에서 브라우저가
                    튀어나오지 않게).

    스레드: `start()` 로 daemon 폴링 스레드를 띄우고 `stop(final=True)` 로 멈춘다.
    stop 은 idempotent. start 하지 않고 `poll_once()` 만 호출해 동기적으로도 쓸
    수 있다(테스트/데모).
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        run_dir_fn: Callable[[], "Path | str | None"],
        persist_dir: Path | str | None = None,
        *,
        live_dir: Path | str | None = None,
        poll_sec: float = 0.5,
        refresh_sec: int = 1,
        autoopen: bool = False,
    ):
        self.graph = graph
        self._run_dir_fn = run_dir_fn
        self._persist_dir = Path(persist_dir) if persist_dir is not None else None
        # 알람마다 바뀌는 run_dir 과 **별개**인 고정 경로. 엔지니어가 이 경로의
        # HTML 탭을 한 번 열어두면 이후 사이클은 state.js 폴링으로 갱신되므로
        # 창을 새로 띄울 필요가 없다(autoopen 의 전면 탈취를 대체한다).
        self._live_dir = Path(live_dir) if live_dir is not None else None
        self.poll_sec = max(0.05, float(poll_sec))
        self.refresh_sec = max(1, int(refresh_sec))
        self.autoopen = bool(autoopen) and platform.system() == "Windows"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_signature = None
        self._latest: tuple[Path, RunState] | None = None
        self._opened = False
        # terminal 이 아닌 노드 = 실행 step 목록(삽입 순서 유지).
        self._step_ids = [
            n.node_id for n in graph.nodes.values() if n.kind is not NodeKind.TERMINAL
        ]

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """daemon 폴링 스레드를 띄운다 (이미 떠 있으면 no-op)."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._poll_loop, name="wf4_cycle_mirror", daemon=True
            )
            self._thread.start()

    def stop(self, final: bool = True) -> None:
        """스레드를 멈추고, final=True 면 마지막 스냅샷을 쓴다 (idempotent, 예외 없음)."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.poll_sec * 3))
            self._thread = None
            if thread.is_alive():
                # 아직 쓰는 중일 수 있다 - 같은 .tmp 를 두 스레드가 쓰지 않게 최종 폴은 생략.
                print("[WARNING] cycle mirror thread 가 join 시간 안에 안 끝나 최종 스냅샷 생략")
                return
        if not final:
            return
        try:
            self.poll_once()
        except Exception as exc:
            print(f"[WARNING] cycle mirror final poll error: {exc}")
        with self._lock:
            latest = self._latest
        if latest is not None:
            try:
                self._write_snapshots(*latest)
            except Exception as exc:  # 호출부의 finally 안에서 불린다 - 절대 던지지 않는다
                print(f"[WARNING] cycle mirror final snapshot error: {exc}")

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_sec):
            try:
                self.poll_once()
            except Exception as exc:
                print(f"[WARNING] cycle mirror poll error: {exc}")

    # --------------------------------------------------------------- polling

    def poll_once(self) -> RunState | None:
        """저널을 한 번 읽고, 바뀌었으면 스냅샷을 쓴다. 미러링 RunState 반환."""
        raw_dir = self._run_dir_fn()
        if not raw_dir:
            return None
        run_dir = Path(raw_dir)
        journal = self._read_journal(run_dir)
        if journal is None:
            return None
        state = self._build_run_state(journal, run_dir)
        signature = self._signature(state)
        persist_dir = self._persist_dir or run_dir
        with self._lock:
            changed = signature != self._last_signature
            self._last_signature = signature
            self._latest = (persist_dir, state)
        if changed:
            self._write_snapshots(persist_dir, state)
        return state

    # -------------------------------------------------------------- journal

    def _read_journal(self, run_dir: Path) -> dict | None:
        """run_state.json + step_<id>.json 을 읽어 dict 로 합친다.

        run_state.json 이 쓰기 중(JSON 파싱 실패)이면 None → 이번 폴은 건너뛴다.
        runner 는 step 파일을 run_state.json 보다 먼저 쓰므로 같은 step 은 파일
        내용을 우선한다.
        """
        state_path = run_dir / "run_state.json"
        if not state_path.is_file():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        results = list(data.get("step_results") or [])
        by_id = {r.get("step_id"): r for r in results if isinstance(r, dict)}
        for step_id in self._step_ids:
            step_path = run_dir / f"step_{step_id}.json"
            if step_path.is_file():
                try:
                    by_id[step_id] = json.loads(step_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        ordered = [by_id[sid] for sid in self._step_ids if sid in by_id]
        return {"status": str(data.get("status") or "running"), "step_results": ordered, "raw": data}

    def _build_run_state(self, journal: dict, run_dir: Path) -> RunState:
        """저널 dict 를 workflow_4 RunState 로 매핑한다."""
        raw = journal["raw"]
        status = journal["status"]
        results = journal["step_results"]
        step_ids = self._step_ids
        run_id = str(raw.get("run_id") or run_dir.name)

        if status == "completed":
            wf4_status = RunStatus.COMPLETED
        elif status in ("aborted", "error"):
            wf4_status = RunStatus.ABORTED
        elif status == "running":
            wf4_status = RunStatus.RUNNING
        else:
            # runner 가 남길 수 있는 그 외 종료 상태("escalated" 등)는 escalate 로 본다.
            wf4_status = RunStatus.ESCALATED

        history: list[TransitionRecord] = []
        failure_class: str | None = None
        prev_node = self.graph.entry_node_id
        for seq, result in enumerate(results, start=1):
            step_id = str(result.get("step_id") or "")
            st = str(result.get("status") or "success")
            event = st if st in ("success", "skipped", "failed") else "failed"
            if st == "failed":
                failure_class = result.get("failure_class") or failure_class
            history.append(
                TransitionRecord(
                    seq=seq,
                    ts=str(result.get("timestamp") or now_iso()),
                    from_node=prev_node,
                    to_node=step_id,
                    event=event,
                    failure_class=result.get("failure_class"),
                    attempt=int(result.get("attempt_count") or 1),
                    note=result.get("error_message"),
                )
            )
            prev_node = step_id

        # 현재 노드 = in-flight step (status=running) 또는 terminal.
        if wf4_status is RunStatus.COMPLETED:
            current_node = "succeeded"
        elif wf4_status in (RunStatus.ABORTED, RunStatus.ESCALATED):
            current_node = "aborted"
        else:  # RUNNING
            n_done = len(results)
            current_node = step_ids[n_done] if n_done < len(step_ids) else "succeeded"

        # aborted 로 끝났으면 실패 step → aborted 전이를 history 에 닫아준다.
        # 저널에는 없는 **합성 레코드** — runner 의 중단을 그래프에 반영하는 미러링
        # 해석이며, node_retries/실패 정보는 전부 저널에서 가져온다.
        if wf4_status in (RunStatus.ABORTED, RunStatus.ESCALATED) and results:
            last_failed = next(
                (r for r in reversed(results) if str(r.get("status")) == "failed"),
                None,
            )
            if last_failed is not None:
                history.append(
                    TransitionRecord(
                        seq=len(history) + 1,
                        ts=str(raw.get("finished_at") or now_iso()),
                        from_node=str(last_failed.get("step_id") or prev_node),
                        to_node="aborted",
                        event="abort",
                        failure_class=failure_class,
                        attempt=int(last_failed.get("attempt_count") or 1),
                        note="runner aborted cycle (mirror 합성 레코드)",
                    )
                )

        node_retries = {
            str(r.get("step_id")): int(r.get("attempt_count") or 1) - 1
            for r in results
            if int(r.get("attempt_count") or 1) > 1
        }

        return RunState(
            run_id=run_id,
            graph_name=self.graph.name,
            status=wf4_status,
            current_node=current_node,
            attempt=1,
            node_retries=node_retries,
            history=history,
            started_at=str(raw.get("started_at") or now_iso()),
            finished_at=raw.get("finished_at"),
            failure_class=failure_class,
            note=None,
        )

    @staticmethod
    def _signature(state: RunState) -> tuple:
        """변화 감지용 시그니처."""
        return (
            state.status.value,
            state.current_node,
            state.finished_at,
            tuple(
                (r.from_node, r.to_node, r.event, r.failure_class)
                for r in state.history
            ),
        )

    # ------------------------------------------------------------ snapshots

    def _write_snapshots(self, persist_dir: Path, run_state: RunState) -> None:
        """persist_dir(+ live_dir) 에 workflow_graph.md/.html 을 덮어쓴다."""
        write_graph_snapshot(
            persist_dir, self.graph, run_state, refresh_sec=self.refresh_sec
        )
        if self._live_dir is not None and self._live_dir != persist_dir:
            # 라이브 사본 실패는 사이클 실패가 아니다 - run_dir 스냅샷이 정본.
            try:
                write_graph_snapshot(
                    self._live_dir, self.graph, run_state, refresh_sec=self.refresh_sec
                )
            except Exception as exc:
                print(f"[WARNING] live graph 사본 쓰기 실패: {exc}")
        if self.autoopen and not self._opened:
            self._opened = True
            open_graph_view(persist_dir / "workflow_graph.html")


__all__ = ["CycleGraphMirror", "StepSpec", "build_step_chain_graph"]
