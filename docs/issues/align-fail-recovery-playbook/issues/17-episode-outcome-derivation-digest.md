# 17 — Episode Outcome derivation and the `[DIGEST] episode` line

Type: task
Status: ready-for-agent
Blocked by: 13, 15, 16
Spec: [spec.md](../spec.md) (Structured Verification and Outcome; Evaluator and compiler)

## What to build

workflow_4 의 Playbook 도메인 패키지에 **첫 순수 evaluator 조각** 을 만든다: Guard reading, Measurement
record, numerator 기록, clearance, abort, handoff 기록을 plain data 로 받아 Outcome ∈ {`recovered`,
`escalated`, `aborted`, `unknown`} 을 낸다. workflow_3 를 import 하지 않는다. 24 번이 이 evaluator 를
Guard 평가/rule 선택으로 확장하므로, Verification 우선순위와 Outcome 파생은 여기가 유일한 소유자다.

per-alarm cycle 은 Episode 를 닫을 때 이 함수로 final Outcome 을 `recovery_episode.json` 에 쓰고, 오피스에서
집으로 복사할 한 줄 `[DIGEST] episode …` 를 찍는다. 이 digest 가 18 번 오피스 gate 의 산출물이다.

## Acceptance criteria

- [ ] primary Measurement `success` → recovered; `failure` → recovered 아님; `unknown` 일 때만 numerator fallback 을 보고, fallback 은 strictly increasing 연속만 success 다.
- [ ] clearance, OK click, `corrected`, runner 완료, cursor idle, 창 닫힘, probable-close 는 단독으로 recovered 를 만들지 못한다.
- [ ] 실패/unknown attempt 뒤 qualified recovered attempt 가 오면 Episode Outcome 은 recovered 이고 attempt 이력은 보존된다; 명시 handoff 기록 → escalated; abort latch → aborted; 그 외 unknown.
- [ ] `handoff` 라는 노드/상태 이름만으로는 escalated 가 되지 않는다.
- [ ] digest 는 한 줄이며 최소 episode id(축약), EQP, recipe, attempts 수, outcome, Guard 3값, verification 경로(primary/fallback/unknown), complete|incomplete(reason) 를 담는다.
- [ ] spec 테스트 8(의미 부분), 9, 21 을 덮는다.
