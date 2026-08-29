# 엔지니어 검토와 annotation 흐름을 프로토타입한다

Type: prototype
Mode: HITL
Status: resolved
Blocked by: 03, 04

## Question

엔지니어가 모든 클릭을 다시 설명하지 않고도 Recovery Action 의 의미, 모호한 Guard,
분기 이유, Recovery Outcome, 적용 범위를 확인할 수 있는 가장 작은 검토 산출물은 무엇인가?

대표 전·후 프레임, 정규화 step, 시스템이 묻는 최소 질문, 행동 수행자의 설명과 Playbook
승인자의 승인을 따로 남기는 방식을 거친 아티팩트로 만들어 같이 검토한다.

## Prototype

[엔지니어 검토와 annotation 흐름 logic prototype](../prototypes/engineer-review-annotation-flow.prototype.html)

브라우저에서 파일을 직접 열고 free-play 또는 guided walkthrough(5개)를 누른다. 이 throwaway
artifact가 확인한 질문은 "하나의 검토 묶음이 Guard·step 의미·분기 이유·Outcome·적용 범위를
보여 주고, 시스템이 묻는 최소 질문에 append-only annotation 으로만 답하며, 승인이 annotation 과
분리된 기록으로 남는가?"이다. 순수 모듈 `ReviewModel`(`effective` / `questions` / `blockers` /
`reduce`)은 DOM 을 모르며 그대로 들어낼 수 있다.

## Answer

사용자 위임(2026-08-29: "선택이 필요할 땐 추천안을 지지")에 따라 아래는 추천안을 그대로
채택한 결정이다.

### 검토 산출물의 단위

검토 묶음(Review Packet)은 저장하는 정본이 아니라 **파생 view** 다. Episode 정본
`recovery_episode.json` + 그 Episode 를 근거로 연결한 candidate Playbook rule + Recovery
Annotation 기록에서 매번 다시 만든다. 한 묶음은 Episode 하나 x rule 하나를 다룬다.

저장 정본은 셋뿐이다.

1. `recovery_episode.json` - 수집기가 쓰는 Episode 정본. 검토 중 수정하지 않는다.
2. `annotations.jsonl` - Episode 폴더 안의 append-only 사람 기록.
3. Playbook lifecycle 의 approval record - 승인자가 쓰며 정확한 `playbook_version` 에 고정된다.

### 검토 묶음이 보여 주는 것

순서대로 다음만 보여 준다. 클릭 좌표, OCR 원문, 원본 event payload 는 보여 주지 않고
provenance 참조로만 연결한다.

1. **Episode 헤더** - EQP / recipe / alarm 시각 / `attempt_seq`, candidate `playbook_version` 과
   `rule_ref`, **분기 이유**(이 rule 을 형제 rule 과 가른 최초 Guard 차이; 대조 Trace 가 없으면
   "단일 rule"), **적용 범위**(capability 목록). 범위가 근거보다 넓은지의 판단은 검토 묶음이
   아니라 승격 gate 의 책임이다.
2. **Action 전 Recovery Guard 3종** - 판정(`true`/`false`/`unknown`/`unresolved`), 판정 이유,
   근거 frame, 출처(packet 또는 annotation id).
3. **정규화 step** - 대표 **전·후 frame**, 도메인 의미, `추론`/`확인됨` 배지, 접힌 raw event 수,
   출처. 대표 frame 은 새로 고르지 않고 `change_events.json` 에서 파생한다: 전 = step 의 첫
   구성 event 의 `prev_frame_path`, 후 = 마지막 구성 event 의 `frame_path`.
4. **Verification** primary(Measurement) / fallback(분자) 와 **Recovery Outcome**.
5. **시스템이 묻는 질문** 목록.

### 시스템이 묻는 최소 질문

질문은 저장하지 않고 실효값에서 **파생**한다. 그래서 답한 질문이 남아 있는 불일치가 생기지
않는다. 질문이 생기는 조건은 다음뿐이다.

- 필수 Guard 가 `unknown` -> "Action 전 상태를 보여 주는 frame 이 있는가"
- step 의 도메인 의미가 미분류 -> "어떤 Recovery Action 인가"
- step 이 `inferred=true` 이고 확인 annotation 이 없음 -> "추론 X 가 맞는가" (한 번의 확인)
- Outcome 이 `recovered` 가 아님 -> "Measurement 정상화 frame, 없으면 분자 증가 frame"
- 같은 대상에 충돌하는 annotation -> "정정(supersedes)하라"

OCR 이나 직접 관측으로 확인된 step 과 `true`/`false` 로 읽힌 Guard 는 묻지 않는다. 가상
Episode(클릭 8건, step 3개)에서 질문은 4건이다. 질문의 답변자는 항상 행동 수행자다.

### Recovery Annotation 기록

`annotations.jsonl` 한 줄은 `annotation_id`, `role`, `by`, `kind`, `target_ref`, `value`,
`evidence_ref`, `supersedes`, `rationale`, `at` 를 가진다.

- `kind` 는 `guard_reading` / `action_meaning` / `verification_reading` / `rationale` 넷뿐이다.
- `target_ref` 는 Episode 내부 참조다: `guard:<kind>`, `step:<seq>`, `verification:primary|fallback`.
- 값을 바꾸는 세 kind 는 `evidence_ref`(frame 또는 event 범위)가 필수다. 근거가 없으면
  **거부하지 않고 `rationale` 로 강등해 기록**한다 - 발언은 보존하고 관측값은 바꾸지 않는다.
- 정정은 `supersedes` 를 가진 새 줄이다. 삭제와 수정은 없다.
- 같은 대상에 서로 다른 값의 살아 있는 annotation 이 둘 이상이면 실효값은 `unresolved` 다.
  최신값을 자동 채택하지 않는다([병합 규칙](04-trace-merge-and-branch-rules.md)과 일치).
- 관측·의미 판독은 `role=actor` 만 할 수 있다. 승인자는 `rationale` 만 남길 수 있다.

### 승인 기록

annotation 과 **분리**된 approval record 다([최소 계약](03-minimal-playbook-contract.md)의
lifecycle 필드): `decision` (`approved` / `needs_evidence` / `rejected`), 고정된
`playbook_version`, `approver`, `rationale`, 당시 열린 질문의 `open_questions`.

- `approved` 는 열린 질문 0건, `unresolved` 0건일 때만 가능하다. 아니면 `needs_evidence` 또는
  `rejected` 만 남길 수 있다.
- 승인은 candidate 버전 전체에 대한 것이므로 그 버전이 근거로 삼는 **모든** Episode 의 검토
  묶음이 질문 0건이어야 한다. 프로토타입은 Episode 1건만 보여 준다.
- 수행자와 승인자가 동일인일 수 있는지는 검토 묶음이 강제하지 않는다. 두 record 모두 `by` 를
  남기고, 정책은 [승격 gate](06-offline-replay-approval-gate.md) 가 정한다.

### 용어

`CONTEXT.md` 에 **Recovery Annotation**, **검토 묶음 (Review Packet)**, **행동 수행자 (Recovery
Actor)**, **Playbook 승인자 (Playbook Approver)** 를 추가했다.

### 검증

프로토타입의 순수 모듈을 node 로 돌려 확인했다: 초기 질문 4건(OK step 은 묻지 않음), 최소
검토 완주 시 질문 0건 + `recovered` + 버전 고정 승인, 근거 없는 판독은 rationale 강등 + 값
불변 + 승인 차단, 의미 충돌은 `unresolved` + 승인 차단, supersedes 정정 후 해소(기록은 3건
그대로), 승인자 관측 판독 거부, 분자 fallback 으로 `recovered`.

