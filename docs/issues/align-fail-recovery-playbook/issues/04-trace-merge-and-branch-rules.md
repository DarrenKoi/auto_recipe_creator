# Recovery Trace 병합과 분기 생성 규칙을 결정한다

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 03

## Question

[`minimal Recovery Playbook contract`](03-minimal-playbook-contract.md) 안에서 여러 Recovery Trace 를
어떻게 정렬·병합해야 관측하지 않은 Guard 를 만들지 않고 provenance 를 보존할 수 있는가?

도메인 행동 동일성, 시간·step alignment, 대조 Trace 에서만 분기를 생성하는 규칙, annotation
을 근거로 받아들이는 경계, 같은 Guard 의 다른 성공 Action 을 `unresolved` 로 보존하는 방식을
확정한다.

## Answer

### 비교와 정렬 단위

병합 단위는 Episode 전체나 `workflow.json`의 개별 click이 아니라 **한 번의 rule 적용 근거
구간**이다. 한 구간은 필수 Guard 관측에서 시작해 순서화된 Recovery Action과
Verification/Outcome으로 끝나며, `episode_id`, `attempt_seq`, `event_seq` 범위로 원본에 연결한다.
retry 또는 새로운 Guard 평가는 별도 구간이다.

Trace끼리는 timestamp나 평평한 step 번호가 아니라 `Guard -> Action -> Verification` 의미 역할과
`event_seq` 순서로 정렬한다. 누락되거나 반복된 Action을 다른 Trace에 맞추려고 압축하지 않고
차이로 보존한다. timestamp는 정렬 identity가 아니라 provenance다.

### 병합 규칙

다음 semantic merge key가 모두 같을 때만 같은 rule에 provenance reference를 추가한다.

1. 호환되는 capability 기반 scope
2. Guard 종류와 관측값
3. 순서가 같은 Action 종류와 도메인 parameter
4. Verification 계약과 Recovery Outcome

`unknown`은 wildcard가 아니다. 화면 좌표, OCR label, EQP/recipe ID, timestamp는 merge key에서
제외하고 원본 provenance에만 둔다. `matched_recipe_box_center`처럼 같은 도메인 역할로 정규화된
parameter는 같게 보지만, 이동 방향이나 탐색 방식처럼 handler 동작을 바꾸는 parameter는 별도
의미로 유지한다. 정규화 여부가 불명확하면 병합하지 않고 `unresolved`로 둔다.

provenance는 `episode_id / attempt_seq / event_seq 범위`를 identity로 하는 집합처럼 취급한다.
같은 근거의 반복 입력은 no-op이고 입력 순서, rule 배열 순서, `rule_id`가 병합 결과를 바꾸지
않아야 한다.

`recovered`가 아닌 `aborted`, `escalated`, `unknown` Trace는 성공 Action이나 fallback rule의
supporting provenance가 아니다. scope를 좁히거나 Guard/Verification 결측과 충돌을 드러내는
제약·반증 근거로만 보존한다.

### 분기와 충돌

호환되는 scope와 공통 경로를 가진 qualified 근거 구간에서 **Action 전에 관측된 Guard 차이**가
서로 다른 Action과 각 Action의 `recovered` Verification에 대응할 때만 별도 rule을 만든다.
분기점은 최초 Guard 차이다. 임의의 최소 Trace 개수는 두지 않지만 각 경로는 자체 qualified
근거를 가져야 한다.

Guard signature가 같은데 의미가 다른 Action이 모두 성공했다면 다수결, 빈도, 선언 순서,
암묵적 priority로 고르지 않는다. 더 정밀한 관측 가능 Guard가 생길 때까지 `unresolved` 충돌로
연결하고 자동 실행은 escalation한다.

특정 EQP/recipe에서만 차이가 보이더라도 ID별 branch를 만들지 않는다. 차이를 설명하는 관측
가능 capability 또는 환경 Guard를 찾으면 scope를 좁히거나 Guard로 분리하고, 찾지 못하면
`unresolved`로 유지한다.

### Annotation 경계

엔지니어 annotation은 근거 frame/event를 가리킬 때만 다음에 사용할 수 있다.

- 관측 가능한 Guard 값의 판독 또는 정정
- 실제 수행된 Action의 도메인 의미 분류
- 판독 가능한 Verification 설명
- 행동 선택 이유의 rationale 기록

annotation만으로 관측되지 않은 Guard, counterfactual Action의 성공, 엔지니어 의도를 Guard로
만들 수 없고 근거 없이 `unknown`을 `true`/`false`로 바꿀 수 없다. 원본 Trace와 기존 annotation은
수정하지 않는다. 정정은 새 `annotation_id`가 대상 evidence와 `supersedes`를 가리키는 append-only
기록이다. annotation끼리 충돌하거나 근거가 판독 불가능하면 최신값을 자동 채택하지 않고
`unknown`/`unresolved`로 둔다.

annotation만 있는 대안은 `unresolved` 안의 branch 제안으로 남긴다. qualified 성공 Trace가
생기기 전에는 실행 가능한 `rules` 또는 `fallback_rule_id`로 승격하지 않으며, 이를 위해 별도
`evidence_status` 필드를 추가하지 않는다.

### `unresolved` 해소와 근거 무효화

`unresolved`는 다음 근거 중 하나가 생길 때만 새 candidate Playbook 버전에서 해소한다.

- 새 qualified Trace가 Action 전의 관측 가능한 Guard 차이를 입증해 rule을 분리한다.
- 의미 검토에서 두 Action과 parameter가 같은 도메인 행동으로 판명돼 병합한다.
- 오분류된 evidence가 정정되어 충돌 근거에서 제외된다.

성공 사례의 수가 더 많다는 이유만으로는 충돌을 해소하지 않는다. supporting Trace가 나중에
`incomplete` 또는 오분류로 판명돼도 기존 불변 Playbook 버전을 수정하지 않는다. 새 candidate에서
reference를 제외하고 병합을 다시 계산한다. qualified 근거가 0건이 된 rule은 실행 가능한
`rules`에서 제거하고 필요하면 `unresolved` 제안으로 남기며, 원본 Trace는 보존한다.

현재 성공 Recovery Trace가 0건이므로 위 규칙은 앞으로 수집할 근거에 적용할 정규화 계약이다.
실제 branch나 승인 가능한 rule이 존재한다고 주장하지 않는다.
