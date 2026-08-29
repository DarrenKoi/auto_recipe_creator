# Recovery Action 어휘와 step 분류 계약 — opencode oc-discuss 기록

- Run date: 2026-08-29
- Skill: oc-discuss
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: `docs/issues/align-fail-recovery-playbook/issues/09-recovery-action-vocabulary-and-classification.md` —
  Playbook 이 참조할 Recovery Action 닫힌 어휘와, `workflow_extract` step + 전·후 frame 기계 판독을
  그 어휘로 바꾸는 분류 계약
- Elapsed: round1 104s · round2 31s · Session: ses_fb516f33dffe1AqtNpukanA2RF

## Claude 의 처음 입장

V1 닫힌 어휘 v1 = `reposition_to_align_key` / `pan_fov` / `set_magnification` / `confirm_align` /
`handoff_to_engineer`; RCS 수준 동작과 비-PM 타이핑·드롭다운·미지 클릭은 어휘가 아니라
`unclassified`. V2 분류기는 workflow_3 `workflow_extract` 의 순수 후속 단계로, R1 더블클릭은
**before frame** 의 matcher 로 reposition/pan 을 가르고, OK 라벨 클릭은 confirm_align, PM 배율
before/after 차이는 set_magnification. V3 기계 분류는 제안이며 05 의 확인/분류 질문을 거친다;
"기타" 는 어휘 확장 제안(`unresolved`)이고 확장은 schema_version bump. V4 parameter 에 화면 px
없음. V5 step 당 digest 한 줄. V6 어휘 표는 `poc/workflow_4/playbook/` 에 두고 workflow_3 이
soft import.

## What the model said — round 1

> **Objection 1 (worst): V2's flagship rule is structurally anti-correlated with its own success condition.** The matcher's confident zone is zoomed-in, scale~1.0 (terminal guard: scale ≥ 0.6 + ORB, live_search.py). Recovery traces exist *because* the key wasn't matchable — so on exactly the before-frames the classifier must read, the matcher returns unknown → `unclassified`. And when it does fire, a pan near a spurious `best_xy` is mislabeled `reposition_to_align_key`, binding to `move_to_point(best_xy)` — a fabricated recenter that poisons ticket-04 semantic-key merging. Falsifier (1) isn't a tail risk; it's the expected day-one outcome, and with 0 Episodes, home-only text digests, and no images executable at home, the tolerance can't even be calibrated. The position states the rule but no confidence gate or calibration plan. Cost: digest action column collapses to `unclassified`, defeating the very motivation cited in ticket 09.
>
> **Objection 2: `set_magnification` is one kind over two behaviorally different bindings, and the trace cannot choose between them.** Wheel zoom leaves no event (recording constraint); the common real mag change is wheel, caught only by PM OCR before/after diff — but `parse_pm_magnification` is deliberately conservative (None on noise), so the diff is three-valued (same/diff/unknown), not binary, and the rule as written has no unknown branch. Even when the diff fires, OCR may yield no readable `mag` value. Worse: dropdown absolute set and wheel steps produce different FOV trajectories (wheel preserves FOV center, live_search.py:14); since the mechanism is unobservable, replay (ticket 06) will flag "engineer did differently" artifacts of *binding choice*, not real divergence. Falsifier (2), plus a contract hole the position doesn't list.
>
> **Objection 3: V6 mischaracterizes the existing seam.** The "soft import" is an opt-in, default-off, auto-disabling mirror (config.py:367, cycle.py:1285) whose real data flow is workflow_4 *reading* workflow_3's journal (workflow3_cycle.py: "read-only"). Putting the vocabulary import inside workflow_extract makes a fail-safe optional dependency load-bearing: on import failure, does extraction degrade to all-unclassified (silently changing digests) or crash? Unspecified. `poc/workflow_4/playbook/` doesn't exist yet; workflow_4 is demo-only (map.md:36). No cycle exists — the real risk is the unstated co-versioning contract: a v2 vocab bump must atomically update workflow_4 data and the workflow_3 classifier, and v1 playbooks must stay interpretable.
>
> **Considered and rejected:** falsifier (3), "other" swamping. The append-only `unresolved` proposal path is a genuine pressure valve; premature vocab growth is the bigger danger with 0 Episodes.

## Claude 의 round 2 — (1) 부분 양보(시퀀스 역할 + after frame), (2) unknown 분기 양보·궤적 반박, (3) 양보(어휘는 데이터)

```text
Here is my response to each of your objections.

Objection 1 (matcher rule anti-correlated with success) - PARTIALLY CONCEDED. Conceded:
the before-frame matcher is the wrong signal on exactly the frames that matter, and no
calibration plan was stated. Corrected rule: a live-box double_click is classified by its
ROLE IN THE SEQUENCE first - the last live-box double_click before a confirm_align (OK) in an
Episode whose Verification later read recovered is proposed as reposition_to_align_key;
corroboration is the matcher on the AFTER frame (post-recenter, key centered, the most
matchable state, judged with the existing Youden-calibrated ensemble thresholds), never the
before frame. Earlier live-box double_clicks in the same Episode are proposed pan_fov only
when followed by further navigation; anything else is unclassified. The classifier never
emits coordinates: the parameter is the symbolic matched_recipe_box_center and the live
binding computes best_xy at run time, so a mislabel cannot bind a fabricated point.
Rebutted: "poisons ticket-04 merging" - machine classification is a proposal; per ticket 05
an inferred step is a merge input only after the actor confirms it against the frames.
Calibration plan for the home/office loop: the classifier's office digest emits per step
(proposed kind, after-frame matcher score, confirmed kind); the mismatch table is what the
home side tunes on. Thresholds start at the ensemble's calibrated values.

Objection 2 (set_magnification) - CONCEDED on the missing unknown branch: the PM diff is
same / diff / unknown. same -> no action; diff with a readable mag -> set_magnification(mag);
diff without a readable value, or unknown -> unclassified with hint "magnification_change".
Rebutted on trajectories: both wheel and PM dropdown change electron-optical magnification
about the current FOV center (no stage move; live_search.py:14 says wheel is FOV-centered
and the PM dropdown is the same optics), so mechanism affects step granularity only. The
action is domain-level "bring magnification to X"; mechanism is the binding's choice, and
replay compares kind+params, so binding choice cannot register as engineer divergence.

Objection 3 (V6 seam) - CONCEDED. The vocabulary is DATA, not a Python import: a versioned
JSON table (recovery_actions.v1.json) checked in under poc/workflow_4/playbook/ and loaded
by path by both the evaluator and the workflow_3 classifier. Missing/unreadable table is a
HARD failure of classification (never a silent all-unclassified digest). Playbooks and
classifier output both stamp action_vocabulary_version; a bump adds a new file side by side,
old versions stay readable, and a classifier loaded with version N emits unclassified for
kinds outside N. The mirror's opt-in soft import is unrelated and untouched.

Where I conceded, treat it as settled. Press, with a concrete failure case, only on the
corrected classification rule (sequence role + after-frame corroboration). Raise a new
objection only if genuinely stronger. Under 250 words; if nothing lands, say converged.
```

## What the model said — round 2

> **Press: the corrected rule is no longer the V2 contract, and its outcome-dependence smuggles causality back in.**
>
> Concrete failure: two Episodes contain byte-identical action sequences — pan, pan, recenter-double_click, OK. Episode A's Verification later reads `recovered`; Episode B's reads `not_recovered` (or stays `unknown` pending Recipe Monitor count). Under the corrected rule the *same* double_click is proposed `reposition_to_align_key` in A and `pan_fov`/`unclassified` in B. The action column — the merge key for ticket 04 — now encodes the Episode's *outcome*, not the actor's behavior. Merging A and B yields "this action is a reposition only when recovery succeeds": a fabricated causal branch, produced by the classifier rather than by a Guard-verified observational difference, which is exactly what map.md forbids ("인과관계를 날조하지 않는") and what ticket 04 routes to `unresolved`.
>
> Second-order cost: the classifier is now a function of Verification, which per ticket 08 is fallback-derived and *arrives late* (Recipe Monitor molecule count). So the digest can't be emitted at extraction time; it must be revised when Verification lands, and if an engineer later corrects a Verification annotation, previously confirmed classifications silently flip — breaking the append-only/immutable-version assumptions of tickets 05/06 (replay consistency: same trace, two runs, different action columns).
>
> If you want outcome-gating, it belongs in the *proposal's confidence* or the merge gate, not in the kind assignment. The kind must be a function of (step, frames, sequence position) only; "was the reposition the successful one" is Verification's job.
>
> Nothing else lands — objection on trajectories and the V6 JSON-table fix are sound.

## Verdict

### Agreed

- 닫힌 어휘 v1(`reposition_to_align_key` / `pan_fov` / `set_magnification` / `confirm_align` /
  `handoff_to_engineer`)과 "그 외는 `unclassified`". "기타" 는 `unresolved` 어휘 제안이며 어휘
  확장은 schema bump - 0건 상태에서는 조기 확장이 더 큰 위험(모델이 스스로 기각).
- 분류기는 **(step, 전·후 frame 기계 판독, 시퀀스 위치)만의 순수 함수**다. Outcome/Verification 은
  입력이 아니다 - kind 가 결과를 부호화하면 04 가 `unresolved` 로 보낼 것을 분류기가 인과 분기로
  날조하고, 늦게 오는 Verification 이 이미 확인된 분류를 뒤집어 append-only 를 깬다.
- reposition 제안 = live-box double_click 의 **다음 step 이 confirm_align 인 시퀀스 위치** +
  after-frame matcher 의 corroboration(calibrated threshold). before-frame matcher 는 실패 프레임에서
  구조적으로 안 잡히므로 쓰지 않는다. 분류기는 좌표를 방출하지 않는다.
- PM 배율 diff 는 same / diff / unknown 3값이고 unknown 분기가 있어야 한다. 배율 변경 메커니즘
  (wheel vs dropdown)은 바인딩의 선택이라 replay divergence 가 아니다.
- 어휘는 Python import 가 아니라 **버전 붙은 JSON 데이터**(`recovery_actions.v1.json`)이며 없으면
  hard fail. 미러의 opt-in soft import 와 무관.
- 기계 분류는 제안이고 05 의 수행자 확인을 거친 kind 만 병합 입력이다. 캘리브레이션은 오피스 digest 의
  (제안 kind, matcher 점수, 확인 kind) 불일치 표로 집에서 한다.

### Disputed

- 없음. round 2 에서 모델이 "그 외엔 걸리는 것 없음" 으로 수렴.

### I was wrong

- before-frame matcher 로 reposition/pan 을 가르려 한 것 - 회복 Trace 는 key 가 안 잡혀서 생긴다.
- round 2 정정안에서 Episode Outcome 을 kind 판정 입력에 넣은 것 - 인과 날조 + 불변성 파괴.
- 어휘 표를 soft import 로 가져오려 한 것 - fail-safe 옵션 의존을 load-bearing 으로 만든다.
- set_magnification 에 unknown 분기를 빠뜨린 것.

## Judgement

전부 accepted - 09 티켓 Answer 에 반영.

## Follow-up

`docs/issues/align-fail-recovery-playbook/issues/09-recovery-action-vocabulary-and-classification.md` Answer.
