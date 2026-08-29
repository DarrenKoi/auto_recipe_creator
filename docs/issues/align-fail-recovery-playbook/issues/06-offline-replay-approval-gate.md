# Offline replay 와 candidate 승격 gate 를 결정한다

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 01, 03, 04, 05

## Question

Recovery Playbook candidate 를 shadow / dry-run 용으로 승인하기 전에 어떤 offline replay 결과와
인간 확인이 필요한가?

모든 근거 Trace 에서 같은 Guard / Action / Verification 경로를 재현하는지, `unknown` 을 `false`로
오판하지 않는지, 성공 주장에 `recovered` 근거가 있는지, 충돌과 `unresolved` 경로가 없는지,
적용 범위가 근거보다 넓지 않은지를 검사하는 승격 계약을 확정한다. production 자동 클릭 승격은
범위 밖이다.

## Answer

사용자 위임(2026-08-29)에 따라 추천안을 채택하되, 애매한 지점은
[opencode 토론](../../../opencode/2026-08-29-recovery-playbook-promotion-gate-debate.md)으로
판가름했다. 처음 입장의 네 가지가 토론에서 무너져 아래는 정정된 계약이다.

### 승격이 뜻하는 것

`candidate` -> `approved` 는 `approved_for="shadow"` 뿐이다. shadow 는 오피스에서 live Guard
평가 + rule 선택 + Action 은 **로그만**(클릭 없음) + Verification reader 는 엔지니어 수동 복구의
결과를 읽는 실행이다. production 클릭 승격은 범위 밖이다.

승인 없는 candidate 도 shadow 를 돌릴 수 있다 - **진단 전용**이다. 산출물은 개발자용 digest 뿐이며
attempt 로 기록되지 않고 corpus 근거가 아니다. 승인된 버전의 shadow 만 `execution_mode="shadow"`
attempt 로 기록되며, 그것도 rule 의 Action 을 뒷받침하는 supporting provenance 는 절대 아니다
([compiler 경계](07-workflow4-compiler-boundary.md)와 공유).

### Offline replay 는 consistency 검사다 (correctness 가 아니다)

replay 는 live 가 쓸 **같은** 순수 Playbook evaluator 를 supporting Episode 의 실효 관측(Episode
정본 + annotation cut 까지의 [Recovery Annotation](05-engineer-review-prototype.md))에 돌리는
결정론 검사다. GUI 가 없고 Mac 에서도 돈다. 산출물 `replay_report.json` 은 (playbook_version,
evidence_cut) 단위로 쓰고, 승인이 참조한 뒤에는 덮어쓰지 않는다.

| 검사 | 내용 | 실패 시 |
|---|---|---|
| R1 경로 재현 | Episode 마다 유일 rule = provenance rule, Action 열(종류 + 도메인 parameter) 일치, Verification 경로(primary/fallback) 일치 | candidate 전체 실패 |
| R2 unknown 안전 | 모든 rule 이 durable Guard 3종을 선언; Episode 마다 Guard 하나씩 `unknown` 으로 변이 -> 선택 없음 + escalation; `evidence_ref`/`reason` 없는 `true`/`false` 는 `unknown` 취급 + 플래그 | candidate 전체 실패 |
| R3 성공 근거 | rule 마다 supporting Episode >= 1, Outcome `recovered` + verification `evidence_ref`, 검토 묶음 열린 질문 0. 비-`recovered` Episode 는 제약 근거만 | qualified 0 인 rule 제거 |
| R4 충돌 없음 | `unresolved` 비어 있음; 2^3 Guard signature 전수 검사에서 rule 일치 <= 1 | candidate 전체 실패 |
| R5 contract 일치 | 모든 supporting Episode 의 `observation_contract` == playbook 의 것 | candidate 전체 실패 |
| R6 바인딩 완전성 | playbook 이 참조하는 guard / action / verification kind 전부에 바인딩이 있음 | compile 거부 |

- `scope.required_capabilities` 는 손으로 선언하지 않는다. evaluator 가 rule 이 참조하는
  Guard/Verification 에서 고정 표로 **도출**한다. "범위가 근거보다 넓은가" 는 사전 검사로는 R5 까지만
  가능하고, 도달 범위는 shadow 에서 **측정**한다.
- replay_report 는 `evidence_kind="consistency"` 를 달고, supporting Episode 가 하나뿐인 rule 은
  `circular=true` 다. 두 값은 승인 record 로 승계된다. **replay 통과는 correctness 근거가 아니며,
  뒤의 production gate 는 replay_report 를 인용할 수 없다.** correctness 근거는 05 의 inferred step
  인간 확인과 아래 shadow digest 뿐이다.
- 콘솔 한 줄: `[DIGEST] replay playbook=<ver> cut=<...> episodes=N rules=M pass|fail
  reasons=<R1..R6>`.

### 인간 확인

1. supporting Episode 전부의 검토 묶음이 열린 질문 0건이고 inferred step 전부를 행동 수행자가
   확인했다.
2. 승인자는 **playbook owner(개발자)가 오피스에서** frame 을 보며 승인한다. 집에서는 승인할 수 없다
   (frame 이 없다). 집으로 오는 것은 replay digest 와 approval record 뿐이다.
3. 자기 승인 금지: owner 가 행동 수행자인 Episode 는 supporting provenance 가 될 수 없고 진단/제약
   근거만 된다. 결과적으로 owner 의 시험 복구는 corpus 를 키우지 않는다 - 명시적 비용이다.
4. approval record 가 고정하는 것: `playbook_version`, freeze 시각, supporting Episode 마다
   `recovery_episode.json` 의 sha256 + annotation-id cut, `replay_report` 경로, `circular` 플래그,
   승인 전에 digest 가 집으로 relay 된 (EQP, recipe) strata 목록. hash 가 어긋나면 replay 는 돌지
   않는다. 03 의 "playbook 문서에 content hash 없음" 은 유지된다(근거 파일 hash 는 별개).

### Shadow 가 증거로 셀 수 있는 것

shadow digest 는 네 항목뿐이고 정확도 단일 숫자는 없다.

- (a) Guard 별 `unknown` 율, (EQP, recipe) 별
- (b) 선택 class 분포 - unique / no_match / multiple / guard_unknown
- (c) unique 선택에서 rule 의 Action 열과 엔지니어 실제 행동의 agreement / difference. difference 는
  miss 가 아니라 [병합 규칙](04-trace-merge-and-branch-rules.md)의 입력(충돌 또는 분기 후보)이다.
- (d) verification reader 가 엔지니어 복구의 실제 결과와 일치했는가

(a)(b)(d) 는 시스템(관측 + reader)의 correctness 신호이고 (c) 는 corpus 입력이다. 일반화 근거는
**freeze 이후** Episode 이면서 **승인 전 digest 를 본 적 없는** (EQP, recipe) stratum 에서만 인용한다
- 시간 분할만으로는 같은 장비·레시피의 반복 재발(twin)이 부풀리는 것을 막지 못한다.

### Episode 정본에 추가할 필드

[관측 계약](08-minimal-observation-gap-contract.md)의 `recovery_episode.json` 에 collector 가
`observation_contract`, `execution_mode`, `bindings_version` 을 stamp 한다.

### 검증

현재 성공 Episode 가 0건이라 gate 는 공허하다. 첫 오피스 실행에서 replay digest 한 줄과 shadow
digest 네 항목이 집으로 오는 것이 이 티켓의 실행 확인이다. 위 계약은 인자 없는 한 명령으로 오피스에서
돌고 텍스트만 되돌아오는 개발 루프를 전제로 한다.

