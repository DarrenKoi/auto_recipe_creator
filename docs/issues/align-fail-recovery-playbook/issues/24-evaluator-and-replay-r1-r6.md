# 24 — Evaluator and offline replay R1–R6

Type: task
Status: ready-for-agent
Blocked by: 22
Spec: [spec.md](../spec.md) (Evaluator and compiler; Offline replay and promotion gate)

## What to build

17 번의 순수 evaluator 를 확장해 3상태 Guard 평가, 설치 capability 적용성, 유일 rule 선택, closed failure
class(`guard_unknown`, `no_match`, `multiple_matches`, `not_applicable`, `precondition_failed`,
`verification_unknown`, `verification_failed`, `handler_exception`), `scope.required_capabilities` 도출을
한 곳에 둔다. 선택은 required Guard 가 전부 `true` 인 rule 이 정확히 하나일 때만 성공한다.

offline replay 는 frozen Episode 상태(정본 + 기록된 annotation cut) 에 같은 evaluator 를 돌리는 consistency
검사다. R1 경로 재현, R2 Guard 하나씩 `unknown` 변이 → 선택 없음(reason/evidence 없는 non-unknown 은
`unknown` 취급 + 플래그), R3 rule 당 qualified recovered ≥ 1 + 열린 질문 0(0 이면 rule 제거), R4
`unresolved` 비어 있음 + 8 signature 전수에서 일치 ≤ 1, R5 observation contract 일치, R6 binding 완전성.
`replay_report.json` 은 (playbook_version, evidence cut) 당 불변, `evidence_kind="consistency"`, supporting
Episode 하나뿐인 rule 은 `circular=true`, hash/cut 불일치면 실행 거부. GUI 바인딩은 호출하지 않는다.
no-arg replay 명령과 `[DIGEST] replay playbook=<ver> cut=<...> episodes=N rules=M pass|fail reasons=<R1..R6>`.

## Acceptance criteria

- [ ] 선택이 unique / no-match / multiple-match / guard-unknown / not-applicable 을 각각 class 로 낸다.
- [ ] R1–R6 각각 실패 케이스 테스트가 있고 R4 는 8 signature 를 전수 평가한다.
- [ ] `scope.required_capabilities` 는 도출값이며 문서의 hand-declared 값은 무시되거나 거부된다.
- [ ] replay_report 가 결정론적이고, `consistency` 와 `circular` 를 표시하며, hash 불일치에 거부한다.
- [ ] replay 경로에서 GUI/바인딩 호출이 0 이다.
- [ ] spec 테스트 20, 22–28 을 덮는다.
