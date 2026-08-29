# Recovery Playbook → workflow_4 compiler 경계 — opencode oc-discuss 기록

- Run date: 2026-08-29
- Skill: oc-discuss
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: `docs/issues/align-fail-recovery-playbook/issues/07-workflow4-compiler-boundary.md` —
  승인된 `recovery_playbook.json` 의 어느 부분을 `WorkflowGraph`/handler 로 변환하고 무엇을
  evaluator / workflow_3 에 남길지
- Elapsed: round1 90s · round2 39s · round3 20s · Session: ses_fb520b277ffe7yhuuWyPdG1Nj4

## Claude 의 처음 입장

C1 **구조만 컴파일, 의미는 컴파일하지 않는다** — Guard 평가·유일 rule 선택·unknown/no-match/
multi-match·Verification 우선순위·Outcome 도출은 replay 와 같은 순수 evaluator 하나에 둔다.
C2 그래프 = `select_rule` → rule 마다 `act:R` → `verify:R` → terminal; failure_class 로만 라우팅,
`verification_failed` 만 fallback 으로, `verification_unknown` 은 항상 escalate. C3 evaluator +
compiler 는 `poc/workflow_4/playbook/`(순수), action/guard/verification 바인딩은 workflow_3, 그래프는
ADR 0003 대로 `_exec_run_correction` 안에서. C4 shadow = 같은 그래프 + 로거 바인딩, verify 는 실관측.
C5 run_state 에 playbook_version/rule_id/episode_id/attempt_seq → 실행마다 새 Trace attempt.
C6 ADR 0003 의 전제 3개는 수용 기준.

## What the model said — round 1

> **Verdict: the position is mostly right (C1/C3 are sound), but C4/C5 as written contain a corpus-poisoning bug, and C2 leaves the fallback path unverified.**
>
> **1. Shadow mode poisons the evidence corpus (C4+C5, worst).** C4 keeps `verify:*` reading real observations while `act:*` only logs. During shadow runs the engineer is manually recovering the tool — so `verify` will read *the human's* post-action state, the evaluator will derive success, and the run terminates `recovered`. C5 then declares every shadow run "a new Recovery Trace attempt" with `playbook_version` + `rule_id` provenance — but provenance has **no execution-mode field**, so this is indistinguishable from live qualified success evidence for a rule whose actions never executed. Ticket 04 explicitly gates rules/fallbacks on qualified `recovered` provenance; ticket 06's replay gate cannot catch this, because replay re-runs the same evaluator over the same trace and reproduces the same false `recovered`. Cost: the first approved playbook version — and every branch derived from it — rests on fabricated success evidence. Fix is cheap (mode in provenance, shadow attempts excluded from supporting corpus), but as stated the position fails its own falsifier (4).
>
> **2. The fallback path is never verified (C2).** C2 wires `verification_failed` → `act:<fallback_rule_id>` and gives the fallback node guard evaluation, but never states `default_next` to a `verify:<fallback_rule_id>` with its own verification contract (ticket 03 invariant 4 applies to fallback rules too). Either fallback actions are unverified — so "one attempt = one trace" produces a segment with actions but no Verification — or the spec is incomplete. Also, a fallback rule is simultaneously a `select_rule` candidate: `act:R` has two identities sharing one node's `node_retries`/`fallback_visits`/global budget counters, so a primary-path failure at `act:R` silently trims a legitimate later fallback routing and escalates. Cost: falsifier (1) half-landed — the routing expresses it, but budgets conflate the two identities.
>
> **3. Outcome vocabulary mismatch.** Terminals are {recovered, escalated, aborted} but ticket 08 requires Verification-unknown episodes to end Outcome `unknown`, preserved, never converted. C2 routes `verification_unknown` → escalated "always"; anyone reading run_state's terminal as Outcome violates the unknown-preservation invariant. Derivation must be explicitly evaluator-owned, not implied.
>
> **Strongest objection I rejected:** C3's placement of the evaluator in workflow_4 breaking its domain-free stance. Rejected because the playbook *is* workflow_4's declared input format, a separate package would add a third dependency direction, and replay needs evaluator+compiler versioned together.

## Claude 의 round 2 — 세 반론 전부 양보, 정정된 입장을 공격하라고 요청

```text
Here is my response to each of your objections.

Objection 1 (shadow poisons the corpus) - CONCEDED. Provenance and run_state gain
execution_mode in {shadow, live}. A shadow attempt is never supporting provenance for a
rule's actions. A shadow run is recorded as two things: (a) a Recovery Trace of the
ENGINEER's manual recovery (the recording + verification readings are attributed to the
human's actions), and (b) a prediction record {guard values, selected rule or escalation
class, later human action} used only for prediction-match statistics. Additionally
(from the parallel ticket-06 debate): shadow of an UNAPPROVED candidate is diagnostic
only - a text digest for the developer, not recorded as an attempt at all.

Objection 2 (fallback unverified; node identity conflation) - CONCEDED on both halves.
A fallback rule F compiles to DISTINCT node ids fallback_act:F -> fallback_verify:F with
their own retry/visit counters; F's own guards are evaluated inside fallback_act:F's
handler (all true, else escalate); fallback_verify:F applies F's own verification
contract (03 invariant 4). A select_rule candidate and a fallback target never share a
node even when they are the same rule. Because fallback_rule_id is null today, the
compiler emits no fallback nodes until one exists - the shape is decided, not built.

Objection 3 (terminal != Outcome) - CONCEDED. Recovery Outcome is evaluator-owned per
03's transitions table and written to run_state as its own field in {recovered,
escalated, aborted, unknown}; nothing derives Outcome from the terminal node id.
verification_unknown -> terminal handoff (escalation as an action) with outcome unknown;
guard_unknown / no_match / multiple_matches -> terminal handoff with outcome unknown;
escalated is only written when explicit engineer handoff is confirmed (ticket 01).

Where I conceded, treat it as settled - do not re-argue it. Press, with a concrete failure
case, on: (i) is "prediction vs what the human then did" a sound correctness signal when
the human never sees the prediction and may recover by a different but equally valid
route? (ii) does anything in C3 ownership (evaluator + compiler versioned together in
poc/workflow_4/playbook/, bindings in workflow_3, graph nested in _exec_run_correction)
break the home-writes / office-runs-and-relays-text loop? (iii) C1 "structure not
semantics": name a playbook semantic that would be forced into graph edges anyway.
Raise a new objection only if it is genuinely stronger. Under 300 words.
```

## What the model said — round 2

> **(i) Prediction-match is a selection signal misused as a rule signal.** Concrete: playbook selects R (pan-left); engineer, never seeing the prediction, recovers via zoom ladder. The prediction record scores a "miss" against R — but per ticket 04, same-guard + different-successful-action is an *unresolved conflict*, not refutation. Aggregated over the n≈1–5 shadow episodes that will ever exist, the statistic is noise, yet as the only quantitative number in the text digest it will acquire gate authority. Prediction-match can legitimately score only the escalation classes (did selection fail when it should have?); rule-level "prediction accuracy" will systematically punish valid alternatives.
>
> **(ii) Binding drift breaks the replay gate's attribution.** C3 versions evaluator+compiler together — good — but the verification *readers* live in workflow_3, unversioned relative to the playbook. Concrete: home replay runs ticket-06 against synthetic bindings and derives `recovered`; office runs the same approved version against a real Measurement reader whose parsing drifted, derives `unknown`. The relayed digest shows divergence but cannot attribute it to playbook semantics vs binding semantics, because run_state provenance pins `playbook_version` but not a binding/reader version. Cost: the approval gate blesses candidates that fail only at office, and the failure gets debugged as a playbook bug. Fix is pinning a binding registry version into run_state — cheap, but currently absent.
>
> **(iii) Rule-level scope applicability has no home.** `scope` is capability-based (03). Concrete: an office install lacking the Measurement-reader capability compiles and starts fine — `select_rule` succeeds (guards evaluable), per-action preconditions pass — then `act:R` dies as `handler_exception` when the unbound kind is looked up. That misfiles "rule not applicable on this installation" as an execution failure, and no failure_class in C5's closed set names it. Either the evaluator takes an injected capability set (pure, fine — say so), or scope semantics silently leak into the registry's KeyError.

## Claude 의 round 3 — 세 반론 전부 양보, shadow digest 완전성만 압박 요청

```text
All three conceded.

(i) Prediction-match is dropped as a rule score. The shadow digest reports exactly four
things, none of them an accuracy number: (a) Guard evaluability per tool/recipe - the
unknown rate of each of the 3 Guards; (b) selection class distribution - unique / no_match
/ multiple / guard_unknown; (c) for unique selections, AGREEMENT vs DIFFERENCE between the
rule's action sequence and what the engineer then did - a DIFFERENCE is filed as new Trace
evidence for ticket-04 merging (potential unresolved conflict or branch), never as a miss;
(d) verification-reader agreement - whether the Measurement/numerator readers produced the
outcome the engineer's recovery actually had. Only (a), (b) and (d) are correctness signals
for the SYSTEM (observation + readers); (c) is corpus input.

(ii) run_state and every digest line pin bindings_version (the workflow_3 binding registry
version: guard observers, action handlers, verification readers) next to playbook_version
and evaluator version, so a home-vs-office divergence is attributable.

(iii) The evaluator takes an injected installation capability set (pure). The compiler
refuses to compile a playbook whose referenced guard/action/verification kinds lack a
binding, and select_rule returns failure_class not_applicable when a rule's derived
required capabilities are not installed. handler_exception is reserved for genuine crashes.

Press only: is the four-item shadow digest complete for the home/office loop, or is there a
signal the developer at home needs that it cannot carry? Under 150 words. If nothing, say
converged.
```

## What the model said — round 3

> Not complete. (c) is a boolean the loop cannot act on. A DIFFERENCE is filed as ticket-04 evidence, but merging needs the *semantic content* of the divergent human path — action kinds with normalized parameters, in merge-key form (ticket 04's Guard→Action→Verification roles). Office-side classification of human recovery steps is exactly the unsolved problem (today it's engineer annotation, ticket 05); if the digest carries only "DIFFERENCE: rule R vs engineer," every DIFFERENCE becomes "go inspect office artifacts" — which the no-images/no-raw-data relay forbids. The corpus-input claim is hollow unless the digest includes a compact, machine-classified action sequence (or an explicit `unclassified_action` marker that routes to annotation rather than merging). Cost: shadow accumulates unactionable DIFFERENCE flags; ticket-04's first real conflicts sit unmerged until someone manually reconstructs the path.
>
> Fix: (c) must emit the engineer's observed action sequence in normalized domain terms, with `unclassified` as a first-class value — that classification contract is a new ticket, and it's the loop's real missing piece.

## Verdict

### Agreed

- **구조만 컴파일, 의미는 컴파일하지 않는다.** Guard 평가·유일 rule 선택·unknown 처리·Verification
  우선순위·Outcome 도출·scope 도출은 replay 와 같은 순수 evaluator 하나에 둔다. evaluator + compiler
  는 `poc/workflow_4/playbook/` 에 함께 버전된다(모델도 별도 패키지 안을 기각).
- 그래프는 `select_rule` -> `act:R` -> `verify:R` -> terminal 이고 failure_class 로만 라우팅한다.
  fallback 은 별도 노드 id(`fallback_act:F` -> `fallback_verify:F`, 자기 카운터, F 의 Guard 는 handler
  안에서 평가)이며 fallback_rule_id 가 null 인 지금은 내지 않는다.
- terminal 노드 != Outcome. Outcome 은 evaluator 가 run_state 의 별도 필드로 쓴다.
- shadow 는 `execution_mode` 로 구분하고 rule Action 의 supporting provenance 가 아니다. shadow 의
  verify 결과는 엔지니어 행동에 귀속된다. prediction-match 는 rule 점수로 쓰지 않는다.
- run_state 와 digest 는 `bindings_version` 을 `playbook_version`/evaluator 버전 옆에 박는다.
- evaluator 는 설치 capability set 을 주입받아 `not_applicable` 을 낸다. compiler 는 바인딩 없는
  kind 를 참조하는 playbook 을 거부한다. `handler_exception` 은 진짜 crash 전용.
- shadow digest (c) 는 boolean 이 아니라 **정규화된 도메인 Action 열**(`unclassified` 1급 값)이어야
  하고, 그 분류 계약은 지도에 없는 결정이다 -> 새 티켓.

### Disputed

- 없음.

### I was wrong

- shadow 의 verify 가 실관측을 읽으면 `recovered` 가 rule 에 귀속되어 corpus 를 오염시킨다는 점을
  놓쳤다(스스로 적은 반증 조건 4 가 그대로 맞았다).
- fallback 을 `verify` 없이 `act` 로만 적었고, 같은 rule 이 select 후보와 fallback 대상으로 한 노드의
  예산을 공유하는 문제를 못 봤다.
- `verification_unknown -> escalated` 로 terminal 을 Outcome 처럼 썼다. 08 의 unknown 보존 불변식 위반.
- prediction-match 를 rule 의 정확도 신호로 제안했다. 04 에 따르면 다른 성공 경로는 충돌 입력이다.
- 바인딩 버전과 설치 capability 를 provenance 에서 빠뜨렸다 - 집/오피스 divergence 귀속이 안 된다.
- digest (c) 를 boolean 으로 두면 집에서 행동할 수 없다는 점 - 개발 루프 제약을 스스로 적고도 놓쳤다.

## Judgement

전부 accepted - 07 티켓 Answer 에 반영. 마지막 반론은 티켓 09(Recovery Action 어휘와 step 분류 계약)로
졸업.

## Follow-up

`docs/issues/align-fail-recovery-playbook/issues/07-workflow4-compiler-boundary.md` Answer,
`issues/09-recovery-action-vocabulary-and-classification.md` 신설.
