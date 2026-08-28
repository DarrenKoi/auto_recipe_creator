# workflow_4 — 상태 머신 워크플로 프레임워크

`observe → decide → act → verify` 개념 위에서 **워크플로가 어디에 있는지, 어디로 갈
수 있는지, 실패 시 어떻게 retry/fallback 되는지, live graph 를 어떻게 시각화하는지**
추적하는 상태 머신 프레임워크 레이어다. GUI/VLM/Windows 의존이 전혀 없는
self-contained 패키지이며, 외부 workflow 라이브러리 의존 없이 hand-rolled FSM 으로
구현했다(0 신규 deps). 시각화는 mermaid/ascii 문자열 생성뿐이다.

workflow_4 는 `poc.workflow_1/2/3` 을 import 하지 않는다 (workflow_3 의 철학을 따름).

## 지금 무엇이 production 에 닿아 있나 (2026-08-28)

- **닿아 있는 것: 미러(live graph view)뿐이다.** workflow_3 사이클은 그대로
  `WorkflowRunner` 로 돌고, `CycleGraphMirror` 가 그 저널을 읽어 HTML 로 보여준다.
- **엔진(`WorkflowEngine`)은 아직 데모만 돌린다.** production 의 두 번째 runner 가
  되지 않는다 - 첫 실소비처는 `run_correction` step **안에서** 도는 align 보정
  sub-flow(보정 -> live_search fallback -> zoom ladder, 지금은 중첩 파이썬 제어
  흐름)이고, 그때도 runner/teardown/알림은 그대로 둔다. 근거와 세 가지 전제 조건은
  `docs/study/adr/0003-engine-first-consumer-nested-in-run-correction.md`.

## 패키지 구조

| 경로 | 내용 |
|---|---|
| `framework/state_graph.py` | `NodeKind` / `WorkflowNode` / `WorkflowGraph` + `validate()` |
| `framework/run_state.py` | `RunStatus` / `TransitionRecord` / `RunState` + 명시적 JSON 직렬화 |
| `framework/engine.py` | `NodeOutcome` / `NodeHandler` / `EngineConfig` / `WorkflowEngine` (유계 실행 루프) |
| `framework/graph_view.py` | `render_mermaid` / `render_ascii` / `write_graph_snapshot`(.md+.html, 원자적 쓰기) / `render_html` / `open_graph_view` |
| `framework/assets/mermaid.min.js` | vendored mermaid 라이브러리(HTML 에 인라인 임베드, git 추적, 3.6MB) |
| `adapters/` | workflow_3 과의 유일한 import 경계 — `CycleGraphMirror`(cycle3 저널 미러) |
| `demo/offline_demo.py` | 가짜 align-fail 복구 그래프 데모 (happy / fallback / escalate) |

의존 방향: `engine → {graph_view, run_state, state_graph}`, `graph_view → {run_state,
state_graph}`. `demo → framework`. 역방향 import 금지. 예외: `adapters/` 만이
workflow_4 → poc.workflow_3 import 가 허용되는 경계다(현재는 저널 JSON 을 직접 읽어
실제 wf3 import 없음; 테스트 하나만 실제 producer 를 import 해 접합을 확인한다).

## 핵심 개념

- **노드(WorkflowNode)**: `default_next`(성공 이동), `failure_routes`(failure_class →
  fallback 노드 라우팅 테이블), `max_retries`(in-place retry 예산), `kind`(NORMAL /
  FALLBACK / TERMINAL).
- **엔진(WorkflowEngine)**: 핸들러 `(node_id, context, run_state) -> NodeOutcome`
  를 호출하며 실패를 분류해 라우팅한다. 모든 전이마다 `run_state.json` 과
  `workflow_graph.md/.html` 을 persist_dir 에 덮어쓴다(live view).
- **유계성**: global retry budget(`global_retry_budget`), 노드별 `max_retries`,
  fallback 재방문 상한, `MAX_TRANSITIONS=1000` 안전장치, `abort_check()` 폴링으로
  무한 루프가 구조적으로 불가능하다.
- **abort 는 두 곳에서 본다**: 엔진이 노드 사이 + retry cooldown 중(`abort_poll_sec`
  간격으로 쪼개서) 폴링하고, 같은 콜러블을 `context["abort_check"]` 로 핸들러에도
  넘긴다 - pan 루프처럼 긴 노드의 중단은 핸들러가 그걸 보고 해야 한다(엔진은 노드
  안을 끊을 수 없다).
- **실패 라우팅 순서**: ① failure_routes 매칭 → fallback 이동(재방문 상한 초과 시
  escalate) ② 같은 노드 in-place retry ③ retry 예산 소진 → ESCALATED.
- **예산 의미(주의)**: `node_retries` 는 노드별 **실패 횟수**, `fallback_visits` 는
  fallback 노드로 **라우팅된 횟수**로 서로 다른 카운터다(같이 세면 fallback 노드가
  라우팅되는 순간 retry 예산 1을 잃는다 - 2026-08-28 리뷰에서 잡힘). fallback 노드의
  `max_retries` 는 두 예산 모두의 상한이다. `global_retry_budget` 은 `node_retries`
  합의 상한이고 `>=` 비교라 budget=N 이면 N 번째 실패에서 escalate 한다.

## 데모 실행 (오프라인, 안전)

```bash
uv run python poc/workflow_4/demo/offline_demo.py
```

GUI/VLM/Windows 의존 없이 순수 파이썬 클로저 핸들러로 세 시나리오를 돌린다:
`happy`(전 성공), `fallback`(panel_not_found 1회 → zoom_probe fallback → 성공),
`escalate`(global retry budget 소진 → ESCALATED). 화면을 건드릴 코드가 없다.
시나리오 선택은 파일 상단 `DEMO_SCENARIO`(1회성 override 는 env `WF4_DEMO_SCENARIO`).

산출물은 각 시나리오별 `poc/workflow_4/debug_images/demo_runs/<scenario>_<ts>/` 에
`run_state.json` + `workflow_graph.md` + `workflow_graph.html` 로 남는다.

## 라이브 그래프 보기 (HTML — 아무 브라우저, 오프라인)

`render_html` 은 vendored `assets/mermaid.min.js` 를 **인라인 임베드**해 단일
self-contained HTML 을 만든다 — IDE/VS Code 없이 Edge/Chrome 등 아무 브라우저로
열리고, `<meta http-equiv="refresh">` 로 자동 새로고침되며, mermaid 그래프(현재
노드 `active` 노란색 강조) + history 테이블(seq/ts/from/to/event/failure_class/
attempt)이 한 화면에 보인다.

```bash
# 데모 산출물 HTML 열기 (Windows 는 os.startfile, 그 외 webbrowser)
uv run python -c "from poc.workflow_4.framework.graph_view import open_graph_view; import glob; open_graph_view(sorted(glob.glob('poc/workflow_4/debug_images/demo_runs/happy_*/workflow_graph.html'))[-1])"
```

- 스냅샷: `write_graph_snapshot` 은 `.md` 와 `.html` 둘 다 덮어쓴다(overwrite-only,
  tmp+`os.replace` 라 브라우저가 새로고침 중에 반쯤 쓰인 파일을 읽지 않는다).
- asset 이 없으면 CDN fallback + 배너(오프라인에서는 vendor 필요).
- `open_graph_view(html_path)` 은 Windows(`os.startfile`) / 그 외(`webbrowser`)로
  열고 절대 예외를 던지지 않는다.

## workflow_3 사이클 미러 (live graph view on office Windows)

workflow_3 의 알람 사이클(`run_alarm_cycle`/`run_check_only_cycle`)은 자체
WorkflowRunner 로 그대로 돈다. 옵트인 어댑터 `CycleGraphMirror`
(`poc/workflow_4/adapters/workflow3_cycle.py`)가 runner 의 **step 저널을 읽기
전용**으로 폴링해 사이클 진행을 workflow_4 `RunState` + 그래프 스냅샷으로
미러링한다 — workflow_3 실행/teardown/알림 로직은 전혀 건드리지 않는다.

오피스 Windows 에서 켜기 (기본 off — 안 켜면 동작 byte-identical):

```bash
ALIGN_FAIL_GRAPH_VIEW=1 uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

또는 `workflow_3_config.py` 에서 `graph_view_enabled = True`. `ALIGN_FAIL_GRAPH_AUTOOPEN`
(기본 1) 은 Windows 에서 첫 스냅샷 후 `workflow_graph.html` 을 기본 브라우저로 자동
연다(다른 OS 에서는 무시 - Mac dry-run 에서 브라우저가 튀어나오지 않게).

- 산출물 위치: `poc/workflow_3/logs/workflow_runs/<run_id>_align_fail_cycle_<eqp>/`
  안에 `workflow_graph.md` + `workflow_graph.html` 이 저널 옆에 생긴다.
- 그래프는 `cycle.py` 가 runner 에 넘기는 **바로 그 step 목록**
  (`build_cycle_steps()` 7 step / `build_check_steps()` 5 step 의 step_id +
  target_description)으로 만든다 - 어댑터 안에 step 이름/실패 class 표를 복사해
  두지 않는다(복사본은 어긋난다: 존재하지 않는 `rcs_recovery_failed` 가 표에 있었다).
  runner 는 어떤 failure_class 든 즉시 aborted 로 끝내므로 실패 간선은 step 마다
  `failed -> aborted` 하나이고 실제 class 는 history 열에 그대로 보인다.
- run_dir 은 runner.run() 이 `context["run_dir"]` 에 넣는 값을 mirror 가 폴링마다
  읽는다(`run_dir_fn`). run() 전에 시작해도 되고 경로를 예측하지 않는다.
- in-flight 노드 = 저널의 `step_results` 개수 번째 step. 실패 시 `aborted` terminal
  로 점프하고 마지막 `failure_class` 를 보존한다.
- 켜도 workflow_3 은 workflow_4 에 하드 의존하지 않는다 — import 실패 시 경고 1회
  후 자동 비활성.
- 오프라인 데모: `uv run python poc/workflow_4/adapters/run_cycle3_mirror_demo.py`
  (가짜 저널을 만들어 몇 초에 걸쳐 live view 가 채워지는 과정을 보여줌).

## 테스트 실행

```bash
uv run pytest poc/workflow_4/
```

- `framework/test_state_graph.py` — validate 가 미지 target / default_next 누락 /
  terminal 미도달을 잡는다.
- `framework/test_engine.py` — happy path 완료, fallback 라우팅, 노드별 retry 예산,
  global budget escalate, abort(노드 전 + cooldown 중), 핸들러 context 의 abort_check,
  `run_state.json` round-trip.
- `framework/test_graph_view.py` — mermaid 에 `stateDiagram-v2`/active classDef/
  현재 노드 포함, snapshot(.md+.html) 생성, `render_html` 구성요소.
- `adapters/test_workflow3_cycle.py` — step chain 그래프 스펙, mirror 저널 폴링
  (in-flight 진행 / aborted 매핑 / run_dir 대기 후 추적 / 스레드 start-stop), 그리고
  **실제 `build_cycle_steps`/`build_check_steps` 로 그래프가 validate 를 통과하는지**.
- `demo/test_offline_demo.py` — happy 시나리오가 tmp dir 에서 COMPLETED + 산출물 생성.

## env 플래그

workflow_4 자체 env 는 데모 시나리오 선택(`WF4_DEMO_SCENARIO`) 하나뿐이다. 미러의
on/off 와 자동 열기는 workflow_3 쪽 env(`ALIGN_FAIL_GRAPH_VIEW`,
`ALIGN_FAIL_GRAPH_AUTOOPEN`)다.

## 설계 근거

외부 workflow 라이브러리(LangGraph / transitions / python-statemachine /
pydantic-graph / Burr)를 쓰지 않고 hand-rolled FSM 을 선택했다. 상세는
`docs/study/adr/0001-hand-rolled-fsm-not-langgraph.md` 참조. workflow_3 사이클을
엔진으로 옮기지 않고 **읽기 전용 미러**를 택한 이유와, Windows 에서 보이는
self-contained HTML 뷰의 선택 근거는 `docs/study/adr/0002-cycle3-mirror-and-windows-html-view.md`,
엔진의 첫 실소비처를 어디에 둘지(runner 를 고치지 않고 `run_correction` 안에 중첩)는
`docs/study/adr/0003-engine-first-consumer-nested-in-run-correction.md` 참조.
