# `workflow_4` compiler 경계를 결정한다

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 03, 06

## Question

승인된 `recovery_playbook.json` 의 어떤 부분을 `workflow_4` `WorkflowGraph` / handler 계약으로 변환하고,
어떤 책임을 Playbook 평가기나 기존 `workflow_3` 안에 남겨야 하는가?

새 production runner 를 만들지 않고, 기존 teardown / 알림 / cooldown 계약을 보존하며, 3상태 Guard 와
`unknown` escalation, Action handler 찾기, Verification 실패 분류, provenance 연결을 어느 경계에 둘지 결정한다.

## Answer

사용자 위임(2026-08-29)에 따라 추천안을 채택하되
[opencode 토론](../../../opencode/2026-08-29-recovery-playbook-workflow4-compiler-debate.md)으로
압박했다. 세 라운드에서 여섯 가지가 정정됐다.

### 경계 원칙: 구조만 컴파일하고 의미는 컴파일하지 않는다

Playbook 의 의미는 전부 **순수 evaluator 하나**에 둔다. [승격 gate](06-offline-replay-approval-gate.md)의
offline replay 와 live 실행이 같은 모듈을 부른다 - 그래프 간선이 선택 논리를 다시 적으면 replay 가
live 를 증명하지 못한다.

| 책임 | 어디에 | 비고 |
|---|---|---|
| Guard 3상태 평가, 유일 rule 선택, `unknown`/no-match/multi-match 처리 | evaluator (`poc/workflow_4/playbook/`, 순수) | 설치 capability set 을 주입받아 `not_applicable` 판정 |
| Verification primary -> fallback 우선순위, **Recovery Outcome 도출** | evaluator | [최소 계약](03-minimal-playbook-contract.md)의 transitions 표를 소유 |
| `scope.required_capabilities` 도출 | evaluator | 손 선언 금지 (06) |
| playbook 버전 -> `WorkflowGraph` | compiler (같은 subpackage, 순수) | 참조 kind 의 바인딩이 없으면 compile 거부 |
| Guard observer / Action handler / Verification reader 바인딩 registry + `bindings_version` | workflow_3 (`monitor/` 안의 바인딩 모듈) | shadow 는 같은 그래프에 로거 바인딩 |
| teardown, 알림, cooldown, abort latch, recording, Episode/attempt id 부여 | workflow_3 `cycle.py` | ADR 0003 - `WorkflowRunner` 와 저널 스키마 불변 |

evaluator + compiler 를 workflow_4 에 두는 이유: playbook 은 프레임워크의 입력 형식이고, replay 는
evaluator 와 compiler 가 **같이 버전**되어야 의미가 있다. 별도 패키지는 세 번째 의존 방향만 만든다.

### 그래프 모양

승인된 playbook 버전 하나가 그래프 하나다. 그래프는 [`run_correction` 실행기 안](../../../../poc/workflow_4/docs/study/adr/0003-engine-first-consumer-nested-in-run-correction.md)에서
돈다.

- `select_rule` (NORMAL, `max_retries=0`) - evaluator 를 live 관측에 호출해 `selected_rule` 을
  context 에 둔다. `guard_unknown` / `no_match` / `multiple_matches` / `not_applicable` 은 모두
  terminal `handoff` 로 라우팅하고 class 는 run_state 에 보존한다. 재평가는 retry 가 아니다.
- rule R 마다 `act:R` (NORMAL) - 바인딩 registry 로 R 의 Action 을 순서대로 실행하며 각 Action 의
  선언된 precondition(예: `ok_control_available`)을 검사한다. `precondition_failed` /
  `handler_exception` -> `handoff`. `default_next` 는 `verify:R`.
- `verify:R` (NORMAL) - evaluator 의 Verification 논리를 호출한다. 성공 -> terminal `recovered`;
  `verification_unknown` -> `handoff` (항상); `verification_failed` (읽혔고 부정) -> F 가 선언돼
  있을 때만 `fallback_act:F`.
- `fallback_act:F` (FALLBACK) -> `fallback_verify:F` - **별도 노드 id, 자기 retry/visit 카운터**.
  F 의 Guard 는 `fallback_act:F` handler 안에서 평가한다(전부 `true` 아니면 `handoff`).
  `fallback_verify:F` 는 F 자신의 Verification 계약을 적용한다. 같은 rule 이 select 후보이면서
  fallback 대상이어도 노드를 공유하지 않는다. `fallback_rule_id` 가 `null` 인 지금 compiler 는
  fallback 노드를 내지 않는다 - 모양만 정한다.
- terminal: `recovered`, `handoff`, `aborted`.

**terminal 노드는 Outcome 이 아니다.** run_state 에 evaluator 가 쓰는 `outcome` 필드(`recovered` /
`escalated` / `aborted` / `unknown`)가 따로 있다. `handoff` 는 escalation 이라는 *행동*이고 Outcome 은
03 의 표를 따른다(`guard_unknown`, `verification_unknown` 등은 `unknown`; `escalated` 는
[관측 의미](01-current-observation-semantics.md)대로 명시적 인계가 확인될 때만).

failure_class 는 닫힌 집합이다: `guard_unknown`, `no_match`, `multiple_matches`, `not_applicable`,
`precondition_failed`, `verification_unknown`, `verification_failed`, `handler_exception`.

### Provenance 와 shadow

run_state 는 `playbook_version`, evaluator 버전, `bindings_version`, `execution_mode`
(`shadow` / `live`), `episode_id`, `attempt_seq`, `rule_id` 를 가진다. Episode/attempt id 는
[관측 계약](08-minimal-observation-gap-contract.md)대로 cycle 이 준다.

- live attempt 는 새 Recovery Trace attempt 다(supporting 후보).
- shadow attempt 는 **엔지니어 수동 복구의 Trace** + prediction record 다. verify 가 읽은 결과는
  엔지니어 행동에 귀속되며, rule Action 의 supporting provenance 가 되지 않는다.
- shadow digest 는 06 의 네 항목이다. 그중 (c) 는 "다르다" boolean 이 아니라 엔지니어의 관측된 행동을
  **정규화된 도메인 Action 열**(`unclassified` 를 1급 값으로)로 싣는다 - 집에서 이미지 없이 04 병합을
  할 수 있는 유일한 형태다. 그 분류 계약은
  [Recovery Action 어휘와 step 분류 계약](09-recovery-action-vocabulary-and-classification.md)에서 정한다.

### ADR 0003 의 전제 = 수용 기준

abort latch 를 `context["abort_check"]` 로 관통, 예산은 벽시계 데드라인에서 역산, 안쪽 그래프 HTML 은
바깥 미러와 다른 파일명 + 링크. 새 결정이 아니라 이 티켓의 수용 기준이다.

### 검증

Mac: evaluator/compiler 는 순수라 단위 테스트와 replay 로 검증한다. 오피스: shadow digest 네 항목이
텍스트로 돌아온다. 그래프 자체의 첫 오피스 실행은 승인된 playbook 이 생긴 뒤이고, 그때도 클릭은 없다.

