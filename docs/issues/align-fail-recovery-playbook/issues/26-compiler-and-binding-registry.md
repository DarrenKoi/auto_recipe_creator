# 26 — Compiler and binding registry

Type: task
Status: ready-for-agent
Blocked by: 24
Spec: [spec.md](../spec.md) (Evaluator and compiler; Binding registry and shadow mode)

## What to build

승인된 Playbook 버전의 **구조만** workflow_4 `WorkflowGraph` 로 컴파일한다: `select_rule`, rule 당
`act`/`verify`, 별도 id 와 별도 visit counter 의 fallback act/verify, terminal `recovered`/`handoff`/
`aborted`. 컴파일 전 어휘, observation contract, binding 버전, 참조 kind, 그래프 validity 를 검증한다.
의미(Guard 평가, rule 선택, Verification, Outcome)는 노드 handler 가 24 번 evaluator 를 호출하는 것이지
그래프 edge 에 복제되지 않는다.

모든 노드는 `max_retries=0` 이고 closed failure class 8개 + engine-native 4개(`handler_exception`,
`invalid_outcome`, `step_budget_exhausted`, `global_budget_exhausted`) 전부에 명시 route 를 가진다 - 기존
엔진은 route 없는 실패를 제자리 재시도하므로 빠진 route 는 컴파일 실패다. deadline 은 엔진 필드가 아니라
기존 abort callable 과 합성한 하나의 `abort_check` 이며, 엔진은 노드 사이에서만 폴링하므로 긴 바인딩은
같은 callable 을 받아 스스로 폴링한다. 엔진 소스는 바뀌지 않는다.

workflow_3 쪽에는 Guard observer / Action handler / Verification reader 와 capability 를 기술하는 버전
붙은 registry v1 을 둔다. shadow 용 Action 바인딩은 logger 뿐이라 마우스/키보드 함수에 닿을 수 없다.
R6 는 이 registry 데이터로 판정한다.

## Acceptance criteria

- [ ] 승인되지 않은 버전은 컴파일이 거부된다.
- [ ] 출력 그래프가 합의된 노드 모양이고 fallback 노드는 별도 id/counter 이며 기존 graph validate 를 통과한다.
- [ ] 모든 노드 `max_retries=0` + 12 class 전부 route; route 하나가 빠지면 컴파일 실패.
- [ ] Verification unknown → handoff, 읽힌 failure → 선언된 fallback 만, 아니면 handoff; terminal 이름은 Outcome 을 바꾸지 않는다.
- [ ] 합성 `abort_check` 를 폴링하는 fake 긴 바인딩이 wall-clock deadline 에 멈춘다; 엔진 소스 무변경을 테스트가 확인한다.
- [ ] registry 가 버전을 stamp 하고 R6 가 빠진 binding 을 잡는다.
- [ ] spec 테스트 27, 29, 30 을 덮는다.
