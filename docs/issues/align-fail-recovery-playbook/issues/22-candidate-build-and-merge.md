# 22 — Candidate build and merge

Type: task
Status: ready-for-agent
Blocked by: 21
Spec: [spec.md](../spec.md) (Candidate building and merge)

## What to build

qualified rule-application segment(pre-Action Guard reading → actor 확인 Action 열 → Verification + Outcome)
에서 candidate `recovery_playbook.json` 을 만든다. semantic merge key 는 capability-compatible scope,
Guard kind/값(`unknown` 은 literal), 어휘로 bin 된 parameter 를 포함한 ordered Action kind, Verification
contract, Outcome 이다. 좌표, 원 pan offset, OCR 라벨, EQP/recipe, timestamp, 배열 위치, 기존 rule id 는
키에서 빠지고 provenance 에만 남는다. 읽은 OM/SEM mode 도 v1 키가 아니다.

canonical key 를 정렬한 뒤 순차 rule id 를 부여해 입력 순서에 불변이고, provenance 는 (episode_id,
attempt_seq, event_seq 범위) set 이라 중복 입력은 no-op 다. 같은 signature 의 다른 성공 Action 은
`unresolved`, Guard 차이 + 다른 성공 Action 만 분기다. 비-recovered/incomplete Episode 는 제약이나
반례일 뿐 supporting 이 아니다. `fallback_rule_id` 는 v1 에서 null 이다. 내용이 바뀌면 새 불변 버전이고
문서 content-hash 필드는 없다. no-arg build 명령과 `[DIGEST] candidate …` 한 줄을 낸다.

## Acceptance criteria

- [ ] Episode 셔플, provenance 중복, rule 셔플, 기존 rule id 에 결과가 불변이다.
- [ ] `unknown` 은 wildcard 로 동작하지 않고, EQP/recipe/mode 는 분기를 만들지 않으며, 같은 signature 충돌은 `unresolved` 에 남는다.
- [ ] 비-recovered 와 incomplete Episode 는 supporting provenance 가 되지 않는다.
- [ ] Playbook 문서에 `schema_version`, `playbook_id`, `playbook_version`, lifecycle/approval 참조, derived scope, selection policy, `action_vocabulary_version`, `observation_contract`, `rules`, `unresolved` 가 있고 frame/좌표/raw event/복사 Episode 데이터는 없다.
- [ ] no-arg build 가 설정/발견으로 돌고 한 줄 digest 를 낸다.
- [ ] spec 테스트 17, 18, 19 를 덮는다.
