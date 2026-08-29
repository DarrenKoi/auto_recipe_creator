# 25 — Approval record and gate

Type: task
Status: ready-for-agent
Blocked by: 23, 24
Spec: [spec.md](../spec.md) (Review Packet, annotations, and approval records; Offline replay and promotion gate)

## What to build

승인은 annotation 과 분리된 append-only record 다: `playbook_version`, freeze 시각, supporting Episode 마다
정본 sha256 + annotation-id cut, replay_report 경로, `circular` 플래그, 승인 전 digest 를 본 (EQP, recipe)
strata, approver identity, `approval_site`. `approved` 는 다섯 조건이 전부 참일 때만이다: 모든 supporting
Review Packet 열린 질문 0, `unresolved` 없음, replay pass, approver 가 설정된 owner 이고
`approval_site="office"` 선언, owner identity 가 모든 supporting Episode 의 `recovery_actors` 에 없음.
아니면 `needs_evidence` 또는 `rejected`. identity 와 site 는 선언값이고 소프트웨어가 검증하지 않는다.
no-arg approve 명령과 digest 한 줄.

## Acceptance criteria

- [ ] 다섯 조건 중 하나라도 빠지면 `approved` 가 아니다(각 조건의 negative 테스트).
- [ ] owner 만 actor 인 Episode 는 supporting 이 될 수 없다.
- [ ] record 는 annotation 파일과 분리돼 있고 approver 는 annotation 관측값을 바꿀 수 없다.
- [ ] 승인 후 Episode 파일이나 annotation 이 바뀌면 24 번 replay 가 hash/cut 불일치로 거부한다.
- [ ] 모호 발견 시 Episode/annotation/Playbook/replay/approval 파일 어느 것도 바뀌지 않는다.
- [ ] spec 테스트 16(승인 부분), 33(부분)을 덮는다.
