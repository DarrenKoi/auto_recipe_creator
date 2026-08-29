# Offline replay 와 candidate 승격 gate 를 결정한다

Type: grilling
Mode: HITL
Status: open
Blocked by: 01, 03, 04, 05

## Question

Recovery Playbook candidate 를 shadow / dry-run 용으로 승인하기 전에 어떤 offline replay 결과와
인간 확인이 필요한가?

모든 근거 Trace 에서 같은 Guard / Action / Verification 경로를 재현하는지, `unknown` 을 `false`로
오판하지 않는지, 성공 주장에 `recovered` 근거가 있는지, 충돌과 `unresolved` 경로가 없는지,
적용 범위가 근거보다 넓지 않은지를 검사하는 승격 계약을 확정한다. production 자동 클릭 승격은
범위 밖이다.
