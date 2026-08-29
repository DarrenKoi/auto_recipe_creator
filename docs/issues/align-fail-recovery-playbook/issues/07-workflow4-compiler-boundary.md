# `workflow_4` compiler 경계를 결정한다

Type: grilling
Mode: HITL
Status: open
Blocked by: 03, 06

## Question

승인된 `recovery_playbook.json` 의 어떤 부분을 `workflow_4` `WorkflowGraph` / handler 계약으로 변환하고,
어떤 책임을 Playbook 평가기나 기존 `workflow_3` 안에 남겨야 하는가?

새 production runner 를 만들지 않고, 기존 teardown / 알림 / cooldown 계약을 보존하며, 3상태 Guard 와
`unknown` escalation, Action handler 찾기, Verification 실패 분류, provenance 연결을 어느 경계에 둘지 결정한다.
