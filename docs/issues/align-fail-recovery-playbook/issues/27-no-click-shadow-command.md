# 27 — No-click shadow command

Type: task
Status: ready-for-agent
Blocked by: 25, 26
Spec: [spec.md](../spec.md) (Binding registry and shadow mode; Operational entrypoints and compatibility)

## What to build

production 알람 모니터와 별개인 standalone no-arg workflow_3 명령이다. 설정/발견으로 active Episode 와
tool 문맥을 정하고(모호하면 후보를 나열하고 무변경 종료), pre-Action Guard 를 읽어 rule 선택 예측을
기록한 뒤, 엔지니어의 수동 복구와 Verification 을 관찰한다. Action 노드는 logger 바인딩이라 마우스/키보드
함수 호출이 0 이다 - `SAFE_MODE` 와 무관하다.

미승인 candidate 의 진단 shadow 는 개발자용 prediction report 와 digest 만 남기고 attempt 를 만들지
않는다. 승인된 버전의 shadow 는 `execution_mode="shadow"` attempt 를 쓰되 관측된 Action 과 Verification
을 예측 rule 이 아니라 Recovery Actor 에 귀속한다. digest 는 네 항목뿐이다: (EQP, recipe) 별 Guard
unknown 율, 선택 class 분포, 예측 vs 확인 Action 열, Verification reader 와 actor 판독의 일치. 단일
정확도 숫자는 없고, 승인 전 digest 를 본 적 없는 strata 를 분리 표시한다. 기존 teardown/notify/
cooldown/abort 계약은 무변경이다.

## Acceptance criteria

- [ ] 진단 shadow 는 GUI Action 함수 호출 0, supporting attempt 없음, 4항목 digest.
- [ ] 승인 shadow 는 shadow attempt 를 기록하고 Action/Verification 을 actor 에 귀속하며 GUI Action 함수 호출 0.
- [ ] 마우스/키보드 hook 계측 테스트가 어떤 모드에서도 호출 0 을 확인한다.
- [ ] 모호 발견 시 어떤 파일도 바뀌지 않는다.
- [ ] digest 에 aggregate accuracy 숫자가 없고 unseen strata 가 구분된다.
- [ ] 기존 monitor/cycle 테스트가 무변경으로 통과한다.
- [ ] spec 테스트 31, 32, 33 을 덮는다.
