# 21 — `annotations.jsonl`, terminal reviewer, and question derivation

Type: task
Status: ready-for-agent
Blocked by: 20
Spec: [spec.md](../spec.md) (Review Packet, annotations, and approval records)

## What to build

Episode 마다 append-only `annotations.jsonl` 을 둔다. 라인은 `annotation_id`, role, actor identity, kind
∈ {`guard_reading`, `action_meaning`, `verification_reading`, `rationale`}, target ref, value, evidence ref,
superseded id, rationale, timestamp. actor identity 는 리포 설정 `RECOVERY_REVIEW_ACTOR` 에서 오고 없으면
append 를 거부한다. 근거 없는 값 변경은 `rationale` 로 강등되고, 충돌하는 live annotation 은 `unresolved`
이며 시각 순서로 풀리지 않는다. approver 역할은 rationale 만 쓸 수 있다.

effective annotation 축약(cut 까지의 live 값, supersede 적용, `recovery_actors` 파생)은 workflow_4 의 순수
함수다. `verification_reading` annotation 이 15 번의 Measurement record 를 `source=annotation` 으로 채운다
- 첫 qualified Episode 의 primary Verification 출처다.

질문은 다섯 경우에만 파생된다: unknown Guard, `unclassified` Action, inferred Action, 성공 Verification
없음, annotation 충돌. 근거 있는 답이 붙으면 effective 상태에서 사라진다. no-arg 터미널 reviewer 가
최신 eligible Episode 를 발견해(모호하면 후보 나열 후 종료) 열린 질문만 보여 주고 답을 append 한다.

## Acceptance criteria

- [ ] `RECOVERY_REVIEW_ACTOR` 미설정 시 append 가 거부되고 파일은 무변경이다.
- [ ] 근거 없는 값 변경은 `rationale` 로 append 되고 effective 관측이 바뀌지 않는다.
- [ ] 유효한 supersede 는 이력을 지우지 않고 effective 를 바꾼다; 충돌 live annotation 은 `unresolved` 다.
- [ ] approver 의 관측 편집은 거부되고 actor 의 의미 편집은 수용된다.
- [ ] `recovery_actors` 가 live guard/action/verification annotation 에서 파생된다; annotation 없는 Episode 는 actor 가 없다.
- [ ] `verification_reading` annotation 이 Measurement record 를 채우고 17 번 Outcome 파생이 그것을 primary 로 쓴다.
- [ ] 질문은 다섯 경우에만 나오고 해소 후 사라진다.
- [ ] reviewer 는 argparse 없이 설정/발견으로 돌고, 모호 발견 시 아무 파일도 바꾸지 않는다.
- [ ] spec 테스트 7(annotation 부분), 13(질문 부분), 14, 15, 16(역할/actor 부분)을 덮는다.
