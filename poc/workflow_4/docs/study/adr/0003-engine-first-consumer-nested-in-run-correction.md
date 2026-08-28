---
status: accepted
---

# 엔진은 두 번째 production runner 가 아니다 — 첫 소비처는 `run_correction` 안의 보정 sub-flow

## 결정

- `WorkflowRunner`(workflow_3)는 **고치지 않는다**. `failure_routes`/index-jump 를
  runner 에 붙이는 안은 기각.
- `WorkflowEngine`(workflow_4)은 production 사이클의 runner 를 대체하지 않는다.
  첫 실소비처는 align 보정 sub-flow(`correct_align_fail_auto` -> live_search fallback
  -> zoom ladder; 지금은 `_exec_run_correction` 안의 중첩 파이썬 제어 흐름)이며,
  그때 엔진은 **`run_correction` step 실행기 안에서** 돈다. teardown/알림/cooldown
  은 `cycle.py` 에 그대로 남고, runner 저널 스키마도 그대로다.
- 그때까지 엔진은 데모 전용이다. workflow_4 가 production 에 닿는 곳은 미러뿐.

## 맥락 / 이유

2026-08-28 opencode 토론(`docs/opencode/2026-08-28-workflow4-engine-vs-runner-debate.md`)
에서 "runner 에 라우팅을 얹고 엔진은 지우자" 는 처음 입장이 세 가지로 무너졌다.

1. **risk-now vs risk-never.** runner 루프는 회귀가 오피스에서만, 며칠 뒤에 드러나는
   표면이다. 데모 전용 엔진을 얼려 두는 비용은 0 인데, 가상의 스키마 비용을 피하려
   그 표면을 지금 건드리는 건 거꾸로다.
2. **runner 저널은 재방문을 표현할 수 없다.** `step_<id>.json` 이 step_id 로 키가
   잡혀(`workflow_runner.py:319`) 같은 노드를 두 번 지나면 앞 기록이 덮인다.
   fallback 라우팅을 runner 에 넣으면 저널 스키마 변경이 강제되고, 그 스키마를
   읽는 도구(미러, 데모 영상, 오피스 엔지니어)가 전부 깨진다 - 지키려던 것을 깨는 셈.
3. **teardown 은 runner 소유가 아니다.** 보장된 teardown 순서와 알림 게이트는
   runner 를 **감싸는** `cycle.py`/`teardown.py` 에 있다. 어느 실행기가 step 을
   돌리든 그 계약은 유지된다. 게다가 `_exec_run_correction` 은 outcome 과 무관하게
   success 를 돌려주므로(`test_failure_cooldown.py` 참조) runner 는 라우팅에 필요한
   실패 분류를 애초에 받지 못한다.

## 중첩 실행의 전제 조건 (토론 2라운드)

- **(필수) abort latch 관통.** wf3 의 계약은 "단축키 -> 즉시 마우스 반환" 이다.
  엔진은 노드 사이에서만 abort 를 보던 데다 cooldown `time.sleep` 이 latch 를
  무시했다. 2026-08-28 수정: cooldown 을 `abort_poll_sec` 로 쪼개 폴링하고, 같은
  콜러블을 `context["abort_check"]` 로 핸들러에 넘긴다. **긴 노드(pan 루프) 안의
  중단은 핸들러가 그걸 보고 해야 한다** - 엔진은 노드 안을 끊을 수 없다.
- **(설계 노트) 예산은 전이 수가 아니라 벽시계로.** runner 의 `timeout_sec` 는
  선언일 뿐 강제되지 않는다. `global_retry_budget=20` x VLM 타임아웃 15s x cooldown
  이면 한 step 이 5~10 분이 되어 녹화 세션/점유 가정이 깨진다. 중첩 `EngineConfig`
  는 데드라인에서 역산할 것.
- **(파일명) 뷰 두 개가 한 run_dir 에.** 바깥 미러가 `workflow_graph.html` 을
  자동으로 여는데 실제 진행은 `correction_graph/` 아래 안쪽 뷰에 있다. 안쪽은 다른
  파일명 + 바깥 뷰에서 링크.
- **(기각) 엔진이 runner 의 StepResult 형태를 내보내는 것.** `strategy_used`,
  `detected_bbox` 같은 필드가 그래프 전이에 무의미하고, 프레임워크의 0-import 규칙만
  깬다. 범위마다 뷰 하나씩이 맞다.

## 결과

- workflow_3 쪽 변경은 미러 훅 하나로 유지된다(`_maybe_start_graph_mirror`).
- 엔진 API 변경: `EngineConfig.abort_poll_sec`, `context["abort_check"]`.
- 보정 sub-flow 를 엔진 그래프로 옮기는 작업은 별도 스펙으로 한다. 위 전제 조건
  세 개가 그 스펙의 수용 기준이다.
