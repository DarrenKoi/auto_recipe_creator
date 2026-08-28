# workflow_4 엔진 vs workflow_3 runner — opencode oc-discuss 기록

- Run date: 2026-08-28
- Skill: oc-discuss
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: workflow_4 `WorkflowEngine` 이 두 번째 production runner 가 되어야 하는가,
  아니면 wf3 `WorkflowRunner` 에 `failure_routes` 를 얹고 wf4 는 시각화 층으로 남는가
- Elapsed: round1 67s · round2 71s · Session: ses_fb9ef3187ffezn3T6DA060rKpC

## Claude 의 처음 입장

wf4 엔진은 두 번째 production runner 가 되면 안 된다. runner 가 이미 저널/teardown/
조건/예산을 가지므로 부족한 건 비선형 라우팅뿐이고, 그건 `WorkflowStep` 에
`failure_routes`/`next` + index-jump 루프(~50-80줄)로 충분하다. wf4 는 저널 위의
시각화 층으로 두고 엔진은 지우거나 데모로 얼린다.

반증 조건으로 "runner 저널이 노드 재방문을 표현할 수 없다면" 을 스스로 적었다.

## What the model said — round 1

> **Verdict: the conclusion (engine is not a production runner today) is right; the position's own remedy is wrong, and one of its premises is factually false.**
>
> **Objection 1 (worst) — the remedy contradicts the position's risk logic.** "Add `failure_routes`/`next` + an index-jump loop to WorkflowRunner" means mutating the production runner's core loop — the exact surface whose regressions (teardown, notify, cooldown) are only discoverable at the office, days later, per your own constraint. Freezing a demo-only engine costs zero production risk; the proposed ~60-line graft costs real risk now, to avoid a hypothetical future schema cost. It's risk-now vs. risk-never.
>
> **Objection 2 — the 50-80 line estimate is false, and your own falsifier already fired.** `step_<id>.json` is keyed by `step_id` (workflow_runner.py:319), so a revisit overwrites prior history — the journal cannot represent a node visited twice, which you yourself named as falsifying. Fixing that *is* a production journal schema change, breaking the very tooling compatibility you're defending (mirror, demo-video, office engineers). Add revisit caps, global budget across revisits, and cycle detection: that's the engine reimplemented as an imperative jump loop — minus `graph.validate()`, minus `MAX_TRANSITIONS` boundedness — inside the most battle-tested function in the repo. You get a second FSM either way; the choice is a validated declarative graph vs. an implicit unvalidated one.
>
> **Objection 3 — factual error: WorkflowRunner does not own teardown.** The guaranteed teardown ordering and finally-guaranteed notification live in `cycle.py`/`teardown.py` *around* the runner, not in it (the runner's `run()` has no teardown at all). Those contracts survive regardless of which executor runs steps, so "second engine must replicate teardown" is a misattribution. Worse: `_exec_run_correction` returns success regardless of correction outcome (test_failure_cooldown.py:95) — the runner's StepResult schema doesn't even carry the failure classification routing needs. The runner today *cannot* route what it never learns.
>
> **Objection 4 (minor)** — "zero production consumers" at t=0 of a deliberately untracked package is the adoption-start state by definition, not evidence.
>
> **Strongest objection I rejected:** "engine adoption now, migrate the cycle." Wrong because dual journals and a Windows-only regression surface are real costs with no current consumer; ADR 0002's mirror is the correct integration.
>
> **Corrected position:** don't mutate WorkflowRunner either. When the correction sub-flow needs graph semantics, run it as a wf4 engine graph *inside* the `run_correction` executor or at cycle level — teardown/notify stay in cycle.py, the journal schema stays intact.

## Claude 의 round 2 — 전부 양보, 정정된 입장을 공격하라고 요청

Objection 1~3 양보(2 는 스스로 적은 반증 조건이 그대로 맞았고, 3 은
`test_failure_cooldown.py` docstring 으로 확인). 정정된 입장(엔진을 `run_correction`
실행기 안에 중첩)에 대해 (a) runner `timeout_sec` 와 엔진 cooldown/예산의 충돌,
(b) 한 run_dir 에 두 저널 스키마, (c) abort latch 가 노드 사이에서만 보이는 문제를
공격하라고 요청.

## What the model said — round 2

> **(a) Real, but latent.** The runner never enforces `timeout_sec` — `run()` calls the executor synchronously and the field is declarative metadata (workflow_runner.py:220). So today it's cosmetic. The concrete failure is scale, not the field: `global_retry_budget=20` × 15s VLM timeouts (cycle.py:2159) × `retry_cooldown_sec` sleeps = a run_correction "step" that runs 5–10 minutes inside structures (recording session, occupied-popup assumptions) sized for the old duration profile. Fix: derive the nested `EngineConfig` from a wall-clock deadline, not transition counts. Not a blocker to the position.
>
> **(b) Position holds; emitting StepResult is premature coupling.** `StepResult` fields (`strategy_used`, `detected_bbox`, `vlm_service_used`) are meaningless for graph transitions, and making the framework emit them breaks the wf4-imports-nothing rule for zero benefit — two viewers, one per scope, is coherent. One concrete cost you must fix: both write `workflow_graph.html` into one run tree, and the outer one is auto-opened by `ALIGN_FAIL_GRAPH_AUTOOPEN`. During a long correction the office engineer stares at "run_correction active" while the real action lives in `correction_graph/workflow_graph.html`. Distinct filename + a link from outer to inner solves it.
>
> **(c) Genuine defect — must fix before any office run.** The contract in wf3 is "hotkey → mouse back immediately": typing stops mid-string, `wait_for_remote_monitoring_window` early-outs, watch loops exit (cycle.py:1194). The nested engine breaks it twice: `abort_check` polls only between nodes, and `time.sleep(retry_cooldown_sec)` (engine.py:263) ignores the latch entirely. Concrete case: ctrl+alt+q during a pan-loop node buys you another full node plus an unbroken cooldown sleep of screen movement. Fix is small and doesn't sink the position: put the latch in `ctx` (handlers already receive it), wire the same latch into `EngineConfig.abort_check`, chunk the cooldown sleep.
>
> Position holds; (c) is a required amendment, (a) a design note, (b) a filename fix.

## Verdict

### Agreed

- 엔진은 오늘 production runner 가 아니고, 앞으로도 사이클 runner 를 대체하지 않는다.
- `WorkflowRunner` 는 고치지 않는다. 비선형 라우팅이 필요해지면 `run_correction`
  실행기 안에서 엔진 그래프를 돌린다(teardown/알림/저널 스키마 불변).
- 엔진이 runner 의 StepResult 형태를 내보내는 건 조기 결합 — 범위마다 뷰 하나.
- (c) 는 지금 고친다: cooldown 을 쪼개 abort 폴링, `context["abort_check"]` 전달.
- (a)(b) 는 중첩 실행 스펙의 수용 기준으로 적어 둔다(ADR 0003).

### Disputed

없음. 2 라운드에서 수렴.

### I was wrong

- runner 저널이 재방문을 표현 못 한다는 건 내가 적은 반증 조건이었는데 확인도
  않고 "50-80줄" 이라고 했다. `step_<id>.json` 키 하나로 무너진다.
- teardown 을 runner 소유로 잘못 붙였다. `cycle.py` 의 finally 다.
- "오피스에서만 드러나는 회귀" 를 제약으로 적어 놓고, 바로 그 표면(runner 루프)을
  건드리는 안을 냈다.

## Judgement

- accepted: Objection 1/2/3, round 2 (a)(b)(c).
- declined: 없음.

## Follow-up

- 엔진 수정(abort-aware cooldown, `context["abort_check"]`) + 테스트 2개: 같은 커밋.
- ADR: `poc/workflow_4/docs/study/adr/0003-engine-first-consumer-nested-in-run-correction.md`.
