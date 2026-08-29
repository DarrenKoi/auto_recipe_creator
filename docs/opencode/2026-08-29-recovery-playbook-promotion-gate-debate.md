# Recovery Playbook 승격 gate (offline replay) — opencode oc-discuss 기록

- Run date: 2026-08-29
- Skill: oc-discuss
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: `docs/issues/align-fail-recovery-playbook/issues/06-offline-replay-approval-gate.md` —
  candidate `recovery_playbook.json` 을 shadow/dry-run 용 `approved` 로 올리기 전의 offline replay
  결과와 인간 확인 계약
- Elapsed: round1 89s · round2 39s · round3 22s · Session: ses_fb520bce6ffe5HeGyIGWw5mkKs

## Claude 의 처음 입장

G1 replay = live 와 같은 순수 evaluator 를 supporting Episode 의 실효 관측(05 의 annotation 반영)에
돌리는 결정론 검사. G2 Episode 마다 같은 rule·같은 action 열·같은 Verification 경로 재현(단일
Episode rule 의 순환성은 인정). G3 unknown 안전성(모든 rule 이 durable Guard 3종 선언, Guard 하나씩
unknown 으로 바꿔 escalation 확인, 근거 없는 true/false 는 unknown 취급). G4 rule 당 recovered
Episode ≥1, 검토 묶음 열린 질문 0. G5 unresolved 비어 있음 + 2^3 Guard signature 전수 검사로 rule
겹침 없음. G6 capability scope 검사(rule guards 커버, observation_contract 일치, capability ⊇ 함의).
G7 승인자가 replay_report 를 보고 정확한 버전에 승인, **자기 승인은 shadow 한정 허용**(self_review
기록). G8 승인이 주는 것은 shadow(관측·선택·로그, 클릭 없음)뿐. G9 0건이라 gate 는 현재 공허.

반증 조건으로 (1) replay 가 오정규화 rule 을 못 가른다, (2) shadow 가 자기 승인을 위험하게 하는
일을 한다, (3) G6 가 동어반복이다, (4) 어딘가 unknown 이 false 로 조용히 취급된다 를 적었다.

## What the model said — round 1

> **Objection 1 (worst): Replay proves consistency, not correctness — and the circularity is worse than G2 admits.** Falsifier (1) is true. Replay compares the evaluator's output against the *normalized steps the rule was merged from*, produced by the same upstream normalizer. A systematic normalization error (e.g., R1 `double_click` inference mislabeling, per 01's `inferred=true` fields) appears identically in rule and Episode, so replay reproduces it perfectly — for multi-Episode rules too. Multi-Episode replay only catches *variance* across episodes, never shared bias; only hand-edited rules get real scrutiny. Cost: the first real approvals — at the system's least-tested moment (G9) — will be single-Episode, maximally circular rules carrying a `replay_report.json` that reads as "path reproduced" evidence. That false confidence is exactly what a future production gate will cite.
>
> **Objection 2: Annotations are a free variable controlled by the self-approver.** Replay runs over *effective* observations = episode + append-only annotations (G1). An actor-approver can iterate: adjust guard readings/step classifications with evidence_refs until replay passes. G3(c) only defends the *missing*-evidence case, not evidence gaming. Combined with G7 self-approval, every link in the chain — annotations, review answers, approval — is one person. Shadow executing no clicks doesn't bound the damage: the *record* shadow produces (falsifier 2) becomes the de facto evidence corpus for later production promotion.
>
> **Objection 3: The approval references mutable evidence.** Effective observations drift post-approval via new annotations; re-running replay overwrites `replay_report.json`. G7(b) references the report by path, and 03 explicitly declined content hashes — so an approved version's evidentiary basis silently changes while approval stays attached. Not tamper-evident; violates the immutability discipline the map applies to everything else.
>
> **Objection 4: G6 is near-vacuous (falsifier 3 confirmed).** (a) restates G3(a). (b) checks a field no producer writes yet. (c) requires a Guard→capability implication map that no ticket defines — 04's "Guard-split mechanism" is actually "keep it `unresolved` until a new observable Guard appears," i.e., there is no mechanism. As written, G6 passes while scope is arbitrarily wide.
>
> The overall shape (unknown≠false everywhere, no synthetic episodes, shadow-only grant) is sound; objections 1–2 are structural, not marginal.

## Claude 의 round 2 — 네 반론 전부 양보, 정정된 입장을 공격하라고 요청

```text
Here is my response to each of your objections.

Objection 1 (replay circularity) - CONCEDED. Replay is a consistency/regression check
only. replay_report.json carries evidence_kind="consistency" and circular=true for any rule
with exactly one supporting Episode; the approval record copies that flag forward.
Correctness evidence for a rule comes from two other places: (i) the review packet (05)
forces the actor to confirm every inferred=true step against its frames - a human check
independent of the normalizer - and a rule cannot be approved with an unconfirmed inferred
step; (ii) shadow runs on NEW episodes, where the playbook's prediction (selected rule or
escalation class + guard values) is compared with what the engineer then actually did.
That prediction-vs-human comparison is the real correctness test and is the only thing a
later production gate may cite - never replay_report.

Objection 2 (self-approver controls annotations) - CONCEDED. Self-approval is forbidden
even for shadow: if approver == actor of any supporting Episode the version stays
`candidate`. Further, under the home/office loop the developer at home cannot see frames,
so the approver must be an office engineer with frame access; the developer never
approves and receives only the digest + approval record. Shadow is split in two: an
UNAPPROVED candidate may run in shadow at the office as DIAGNOSTIC ONLY - output is a
digest for the developer, never corpus evidence. Only shadow runs of an APPROVED version
are recorded as attempts, with execution_mode="shadow", and even those are never
supporting provenance for the rule's actions - they feed prediction-match statistics only.

Objection 3 (approval references mutable evidence) - CONCEDED. Approval pins an evidence
cut: for every supporting Episode, the last annotation_id at approval time (append-only
ids make the cut stable). replay_report is written per (playbook_version, evidence_cut)
and is never overwritten once an approval references it. Annotations after the cut affect
only the next candidate version. No content hash (03's decision stands): append-only +
id cut is tamper-evident enough for a shadow-only grant.

Objection 4 (G6 vacuous) - CONCEDED. G6 is replaced: scope.required_capabilities is
DERIVED by the evaluator from the Guards/Verification a rule consults via a fixed table in
the playbook schema, never hand-declared; the collector stamps observation_contract into
recovery_episode.json (an addition to 08's episode file); the check is that every
supporting Episode's contract equals the playbook's. The gate claims no reach beyond
evidence. Reach is MEASURED in shadow: a per-tool/recipe digest of guard evaluability and
prediction match, and that digest is what any later scope widening must cite.

Where I conceded, treat it as settled - do not re-argue it. Press, with a concrete failure
case, on: (i) is the diagnostic-vs-evidence shadow split coherent, or does it leak?
(ii) office-side approver with frames / home-side developer with digests only - what
breaks? (iii) evidence cut by annotation_id without a hash. Raise a new objection only if
it is genuinely stronger than what you already gave. Under 300 words.
```

## What the model said — round 2

> **(i) The split leaks through corpus reuse.** Diagnostic shadow runs consume NEW live episodes at the office, and nothing bars those same episodes from later entering the corpus as supporting evidence. The developer iterates rules at home against diagnostic digests of episode E (guard values, selected rule, prediction-miss class), ships a new candidate, and E is then collected as a "qualified" supporting Episode for that very rule. Replay reproduces it trivially — the rule was tuned to it via the digest channel. Prediction-match statistics become in-sample. Fix must be explicit: any episode touched by a diagnostic run is quarantined from supporting-provenance for rules whose diagnostic output the developer saw.
>
> **(ii) Approval deadlocks or rubber-stamps.** With self-approval forbidden, the approver must be an office engineer who is not the actor of ANY supporting Episode. Your team: engineers act, developer approves — but the developer can no longer approve (no frames). If the acting engineers are one or two people, the only eligible approver is an engineer who never acted — who must also own the playbook schema semantics they didn't write and can't discuss with the developer in real time (office has no Claude; dev is asleep at home). Concrete failure: a guard reading is ambiguous, the packet question requires contract interpretation, nobody at the office owns the contract — approval either stalls across day/night cycles or an unqualified engineer signs it to unblock. The gate needs a named approver role that actually exists on the office roster.
>
> **(iii) The cut pins annotations, not the episode.** `recovery_episode.json` is collector-written and immutable only by convention. If a collector rerun or bug rewrites it (guard values, paths) after approval, the annotation cut stays stable while the thing being cut changes; replay rerun diverges while the approval record still "pins" it. Pin the episode file's last-write identity (even just byte length + mtime in the approval record) or declare the file frozen-by-gate.

## Claude 의 round 3 — (i) 반박(격리는 교착), (ii) 부분 양보(owner 가 오피스에서 승인), (iii) 양보

```text
Point by point.

(i) Corpus leak - REBUTTED, with one concession. Your quarantine ("any episode touched by a
diagnostic run is barred from supporting provenance") deadlocks: at the office every real
episode will run with the diagnostic shadow on, so the first N episodes - the entire early
corpus - would all be quarantined and no rule could ever gain supporting evidence. The
distinction that actually matters is supporting provenance vs correctness evidence. A rule
being merged FROM episode E and then citing E as supporting provenance is not a leak - that
is what merging (ticket 04) is; replay is already conceded to be consistency-only. What must
be out-of-sample is the prediction-match statistic. Concession: it is out-of-sample BY
CONSTRUCTION through a time split - a version V may count prediction-match only from
episodes whose alarm time is after V's freeze time, and the approval record stores that
freeze time. Any episode the developer saw a digest of before freezing V is, by definition,
before the freeze and cannot count toward V's correctness statistic.

(ii) Approver deadlock - PARTIALLY CONCEDED; I over-corrected in round 2. The developer is
physically at the office on office days (the home/office loop is about where code is
WRITTEN, not where the developer is). Named role: approver = the playbook owner (the
developer) AT THE OFFICE with frame access; nobody approves from home. Self-approval stays
forbidden, with the consequence made explicit: episodes where the owner was the actor are
diagnostic/constraint evidence only, never supporting provenance - consistent with the
playbook modelling the ENGINEERS' practice. Contract interpretation questions are the
owner's, and the owner is present when approving.

(iii) Episode file not pinned - CONCEDED. The approval record pins each supporting
episode's recovery_episode.json by sha256 plus the annotation-id cut; replay refuses to run
if a pinned hash mismatches. Ticket 03's "no content hash" decision was about the playbook
document, not evidence files, and stands.

Press (i) only: give a concrete case where the freeze-time split still lets a tuned-to-E
episode count as correctness evidence for the version tuned to it, or drop it. Confirm or
deny that (ii)'s named role resolves the deadlock. Under 200 words.
```

## What the model said — round 3

> **(i) Concrete case — correlated recurrence.** Episode E1 (tool A, recipe R) alarms Monday; developer sees its diagnostic digest, tunes rule, freezes V Monday night. Tuesday, the same align fail recurs on tool A/recipe R — episode E2, alarm time after freeze, so it counts toward V's prediction-match. But E2 is a near-duplicate of E1: same failure mode, same screen state, likely same corrective action. The rule was tuned on E1's digest and is now "validated out-of-sample" on E2. With recurring failures on a handful of recipes, V's correctness statistic can be built almost entirely from post-freeze twins of pre-freeze tuning episodes. The split blocks temporal leakage but not serial correlation; the statistic will systematically overstate generalization, and a later production gate citing it inherits that inflation. Minimum fix: stratify prediction-match by (EQP, recipe, failure class) and refuse to cite any stratum whose evidence is dominated by recurrences of pre-freeze episodes.
>
> **(ii) Confirmed.** Owner-at-office resolves the deadlock: frame access present, contract interpretation owned, no overnight stall. Residual cost, not incoherence: the owner's own test recoveries are permanently diagnostic-only, further shrinking the already-zero qualified corpus — worth stating in the ticket, but the role itself is sound.

## Verdict

### Agreed

- replay 는 **consistency 검사**다. 단일 Episode rule 은 순환이며 `circular=true` 로 표시해 승인
  record 까지 승계한다. correctness 근거는 05 의 inferred step 인간 확인과 shadow 에서 온다.
- 자기 승인 금지. 승인자는 playbook owner(개발자)가 **오피스에서** frame 을 보고 한다. 집에서는
  digest 와 approval record 만 받는다. owner 가 actor 인 Episode 는 진단/제약 근거만 된다(corpus 가
  그만큼 늦게 자라는 비용을 티켓에 명시).
- 승인 record 는 annotation-id cut 과 `recovery_episode.json` sha256 으로 근거를 고정한다.
  replay_report 는 (version, cut) 단위로 쓰고 승인이 참조하면 덮어쓰지 않는다. 03 의 "playbook 문서
  content hash 없음" 결정은 유지된다.
- G6 는 폐기. scope 는 evaluator 가 참조 Guard/Verification 에서 **도출**하고, collector 가
  `observation_contract` 를 Episode 에 stamp 하며, 도달 범위는 shadow 에서 **측정**한다.
- 승인 없는 candidate 의 shadow 는 진단 전용(digest 만, attempt 기록 없음). 승인된 버전의 shadow 도
  rule action 의 supporting provenance 는 아니다.
- 일반화 근거는 freeze 이후 Episode 이면서 승인 전 digest 를 본 적 없는 (EQP, recipe) stratum 에서만
  인용한다 - 시간 분할만으로는 반복 재발 twin 을 못 막는다.

### Disputed

- 없음. (i) 의 격리 제안은 교착 근거로 기각되었고 모델도 시간 분할 + stratum 조건으로 옮겨 왔다.

### I was wrong

- G2 를 "경로 재현 근거" 로 적은 것. replay 로는 정규화기의 공통 편향을 잡을 수 없다.
- shadow 한정 자기 승인 허용. shadow 기록이 사실상 corpus 가 되므로 위험이 0 이 아니다.
- 승인이 경로만 참조하도록 한 것(가변 근거). cut + 파일 hash 가 필요했다.
- G6 전체. (a) 는 G3(a) 중복, (b) 는 아무도 안 쓰는 필드, (c) 는 정의되지 않은 표.
- round 2 에서 "개발자는 승인하지 않는다" 로 과잉 정정한 것. 개발자는 오피스에 있을 때 frame 을 본다.
- "freeze 시각 분할이면 out-of-sample" - 반복 재발 twin 에는 불충분했다.

## Judgement

전부 accepted - 06 티켓 Answer 에 반영. 경로 재현 대신 consistency 라벨, 자기 승인 금지 + owner
오피스 승인, cut + sha256 고정, scope 도출, 진단/증거 shadow 분리, stratum 조건.

## Follow-up

`docs/issues/align-fail-recovery-playbook/issues/06-offline-replay-approval-gate.md` Answer.
