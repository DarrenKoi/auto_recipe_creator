# 최소 Recovery Playbook 계약을 프로토타입한다

Type: prototype
Mode: HITL
Status: resolved
Blocked by: 01, 02, 08

## Question

[`current observation semantics`](01-current-observation-semantics.md)와
[`representative Recovery Trace corpus`](02-representative-trace-corpus.md)에 기대어,
`recovery_playbook.json` 의 가장 작은 검토 가능 계약은 어떤 모양이어야 하는가?

프로토타입은 적용 범위, 3상태 Recovery Guard, Recovery Action 의미와 parameter, Verification,
Outcome / fallback, Trace provenance, candidate / approval, `unresolved` 를 표현하되 원본 Trace 데이터를
복제하지 않아야 한다.

## Prototype

[최소 Recovery Playbook 계약 logic prototype](../prototypes/minimal-recovery-playbook-contract.prototype.html)

브라우저에서 파일을 직접 열고 free-play 또는 guided walkthrough를 누른다. 이 throwaway artifact가
확인할 질문은 “원본 Trace를 복제하지 않는 하나의 rule이 3상태 Guard, 도메인 Action, 우선순위
Verification, Outcome, provenance, lifecycle, `unresolved`를 충분히 표현하는가?”이다.

## Answer

`recovery_playbook.json`은 실행 그래프나 원본 Trace 사본이 아니라 **버전이 고정된 Recovery rule
집합**이다. 최소 최상위 계약은 다음 필드만 가진다.

- `schema_version`, `playbook_id`, 불변 `playbook_version`
- 정확한 `playbook_version`에 대한 `candidate` / `approved` lifecycle과 approval 기록
- 장비 ID 목록이 아닌 필요한 관측·제어 capability를 표현하는 `scope`
- 암묵적 priority 없이 rule 선택 실패를 정의하는 `rule_selection`
- `rules`, 해결되지 않은 충돌을 보존하는 `unresolved`

각 rule은 `rule_id`, 3상태 `guards`, 도메인 의미와 parameter를 가진 `actions`, primary/fallback
`verification`, Outcome과 다음 처리를 정하는 `transitions`, 선택적인 `fallback_rule_id`, 원본을
가리키는 `provenance`만 가진다. 화면 좌표, frame, 원본 event payload는 복제하지 않고
`recovery_episode.json`의 Trace/event reference로 연결한다.

다음 불변식을 적용한다.

1. 모든 필수 Guard가 `true`인 유일한 rule만 자동 선택할 수 있다. `unknown`, 일치 rule 없음,
   둘 이상 일치는 Action을 선택하지 않고 Outcome `unknown`으로 보존한 뒤 escalation한다.
2. 선언 순서나 숨은 priority로 rule 충돌을 해소하지 않는다. 관측 가능한 Guard 차이가 생기기
   전까지 충돌은 `unresolved`다.
3. fallback Action은 inline으로 만들지 않는다. 별도 rule이 자체 Guard와 qualified Trace
   provenance를 가질 때만 `fallback_rule_id`로 참조한다. 현재 값은 `null`이다.
4. Verification은 Measurement 정상화를 primary로, Measurement를 읽을 수 없을 때의 분자 엄격
   증가를 fallback으로 사용한다. 둘 다 판정할 수 없으면 Outcome은 `unknown`이다.
5. 승인 상태는 playbook 전체의 정확한 `playbook_version`에만 적용한다. scope, rule,
   provenance가 바뀌면 새 candidate 버전을 만들며, 현 단계에는 content hash를 추가하지 않는다.

[logic prototype](../prototypes/minimal-recovery-playbook-contract.prototype.html)은 위 계약의 primary
성공, Guard unknown, Verification fallback, 다중 rule 충돌을 실행해 볼 수 있다. 성공 Recovery
Trace가 아직 0건이므로 구조만 검토할 수 있으며, 이 예시는 승인 가능한 Playbook이나 production
실행 그래프가 아니다. `workflow_4` 변환은 승인된 계약이 생긴 뒤 별도 단계에서 다룬다.
