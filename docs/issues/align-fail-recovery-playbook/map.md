# Align Fail Recovery Trace 정규화와 조건부 Playbook 지도

## Destination

Align Fail 후 엔지니어 Recovery Trace 를 인과관계를 날조하지 않는 조건부 Recovery
Playbook 으로 정규화하는 구현 가능한 명세에 필요한 모든 결정을 확정한다.

## Notes

- 이 지도는 계획만 다룬다. 코드 구현과 live GUI 조작은 하지 않는다.
- 각 세션은 `matt-skills-curated:wayfinder`, `grilling`, `domain-modeling` 지침을 따른다.
- 정본 용어는 [`CONTEXT.md`](../../../CONTEXT.md)를 따른다.
- Recovery Trace 는 실행 절차가 아니라 근거고, Recovery Playbook 은 여러 Trace 를 통합한
  모델이다.
- Recovery Guard 는 관측 가능한 상태만 쓰며 `true` / `false` / `unknown` 을 반환한다.
  `unknown` 은 다른 행동 분기가 아니라 escalation 근거다.
- Trace 하나는 선형 후보만 입증한다. 조건 분기는 대조 Trace 또는 엔지니어의 명시적
  annotation 이 있을 때만 생성한다.
- 녹화 좌표를 복제하지 않고 Recovery Action 을 도메인 행동으로 정규화한다. 원본 이벤트와
  프레임은 provenance 로 보존한다.
- 병합은 `EQP_ID` / `RECIPE_ID` 가 아니라 관측 상태, 행동 의미, 검증 결과를 기준으로
  한다. 적용 범위는 ID 목록보다 필요 능력과 관측 환경으로 표현한다.
- `recovery_playbook.json` 이 검토 가능한 정본이다. 기존 `workflow.json` 은 단일 Trace 근거로
  남기고, 승인된 Playbook 만 `workflow_4` 실행 그래프로 변환한다.
- 검토는 녹화 후 모호한 지점만 다룬고, 서로 다른 성공 행동은 다수결로 덩지 않는다.
  관측 차이를 찾지 못하면 `unresolved` 로 보존하고 escalation 한다.
- Trace 하나로 candidate 는 만들 수 있지만, shadow/dry-run 승인은 엔지니어 확인,
  모든 근거 Trace 의 offline replay, 성공 근거, 충돌 부재를 요구한다. 임의의 최소 Trace
  개수는 두지 않는다.
- 개발 루프 제약(2026-08-29 사용자): 모든 코드는 집(Mac)에서 작성하고 오피스는 **실행과 확인만**
  한다. 오피스에서 돌아오는 것은 텍스트(콘솔 digest, JSON 요약, 점수)뿐이며 이미지·원본은
  반입되지 않는다. 따라서 이 지도의 모든 gate·검사·shadow 산출물은 인자 없는 한 명령으로 오피스에서
  돌고, 사람이 복사해 집으로 옮길 수 있는 **한 줄 digest** 를 내야 한다. align key 평가 때 점수
  digest 를 받아 코드를 고친 이력이 이 방식의 선례다.
- **지도 완료 (2026-08-29).** 열린 티켓이 없고 fog 도 없다. 다음은 지도 밖이다: 위 결정 아홉 개를
  구현 명세(`to-spec`)로 옮긴다 - collector 의 Episode 정본·id·stamp 보강(08/06), `annotations.jsonl`
  과 검토 묶음 렌더러(05), `poc/workflow_4/playbook/` 의 evaluator·compiler·어휘 JSON(07/09),
  workflow_3 의 분류 단계·바인딩·shadow 모드·replay 도구와 digest(06/07/09). 05 의 논리 프로토타입은
  `prototypes/` 에 남는다.
- 현재 [`workflow extraction 설계`](../../../poc/workflow_3/docs/superpowers/specs/2026-08-11-workflow-extraction-design.md)는
  평평한 의미 step 목록까지만 제공하고, [`workflow_4`](../../../poc/workflow_4/README.md) 엔진은
  현재 demo 전용이며 `failure_class` 라우팅만 가진다.

## Decisions so far

<!-- 닫힌 티켓의 해답을 한 줄로 요약하고, 상세는 티켓을 링크한다. -->

- [현재 관측 신호의 Recovery 의미를 확정한다](issues/01-current-observation-semantics.md) — matcher·점유·관측 가능성만 Guard로 평가할 수 있고, action/correction status는 provenance이며, 현재 durable 성공 Verification은 Recipe Monitor 분자 증가뿐이다.
- [대표 Recovery Trace corpus 를 선택한다](issues/02-representative-trace-corpus.md) — 아직 성공 Recovery Episode가 없어 corpus는 0건이며, synthetic/demo/pre-action abort를 대체 근거로 쓰지 않고 첫 검증 가능 성공 Trace부터 수집한다.
- [부족한 Recovery 관측 신호의 최소 보강 경계를 결정한다](issues/08-minimal-observation-gap-contract.md) — Measurement를 primary, 분자를 fallback Verification으로 두고 세 종류의 3상태 Guard와 Episode/attempt/event 연결, 첫 qualified Trace의 필수 evidence bundle을 확정했다.
- [최소 Recovery Playbook 계약을 프로토타입한다](issues/03-minimal-playbook-contract.md) — Playbook을 불변 버전의 rule 집합으로 두고, 유일한 3상태 Guard 일치만 자동 선택하며 충돌·무근거 fallback은 `unresolved`로 escalation하는 최소 계약을 확정했다.
- [Recovery Trace 병합과 분기 생성 규칙을 결정한다](issues/04-trace-merge-and-branch-rules.md) — rule 적용 근거 구간을 semantic key로 순서 독립 병합하고, 관측 Guard로 입증된 성공 경로만 분기하며 충돌·annotation-only 대안은 `unresolved`로 보존한다.
- [엔지니어 검토와 annotation 흐름을 프로토타입한다](issues/05-engineer-review-prototype.md) — 검토 묶음은 Episode·rule·annotation에서 파생하는 view이고, 질문은 unknown/미분류/추론/미검증/충돌에만 파생되며, annotation은 근거 필수 append-only(근거 없으면 rationale 강등), 승인은 별도 record로 버전에 고정되고 열린 질문 0건일 때만 `approved`다.
- [Offline replay 와 candidate 승격 gate 를 결정한다](issues/06-offline-replay-approval-gate.md) — replay는 같은 evaluator의 consistency 검사(R1~R6, `circular` 표시)이지 correctness가 아니고, 승인은 owner가 오피스에서 frame을 보고 자기 승인 없이 cut+sha256으로 근거를 고정하며, shadow는 승인 전엔 진단 전용·승인 후에도 provenance가 아니며 digest 네 항목만 stratum 조건으로 인용한다.
- [`workflow_4` compiler 경계를 결정한다](issues/07-workflow4-compiler-boundary.md) — 구조만 컴파일하고 의미(Guard 평가·rule 선택·Verification·Outcome·scope 도출)는 replay와 같은 순수 evaluator에 두며, 그래프는 `select_rule`→`act:R`→`verify:R`(+별도 id의 fallback 노드)로 failure_class만 라우팅하고, 바인딩·teardown·id 부여는 workflow_3에 남고, shadow는 `execution_mode`로 구분되어 provenance가 아니다.
- [Recovery Action 어휘와 step 분류 계약을 결정한다](issues/09-recovery-action-vocabulary-and-classification.md) — 닫힌 어휘 v1 다섯 kind(그 외 `unclassified`)를 버전 붙은 JSON 데이터로 두고, 분류기는 (step·전후 frame·시퀀스 위치)만의 순수 함수로 Outcome을 입력에 넣지 않으며(다음 step이 OK면 reposition 제안, PM diff 3값), 좌표를 방출하지 않고 수행자 확인을 거친 kind만 병합 입력이다.

## Not yet specified

현재 없음.

## Out of scope

- 실장비에서 Recovery Action 을 클릭하는 production 실행과 rollout gate.
- 이 지도를 풀는 동안 `workflow_3` / `workflow_4` 실행 코드를 변경하는 일.
- 조작 녹화를 바로 복제하는 end-to-end imitation learning 또는 VLM 자율 정책.
- Align Fail Recovery 밖의 일반 recipe 생성 자동화.
