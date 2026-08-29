# Align Fail Recovery Playbook 구현 명세

Type: spec
Status: ready-for-agent

## Problem Statement

현재 시스템은 Align Fail 알람, 화면 녹화, 평평한 조작 step, matcher 판단, 일부 완료 신호를
각각 남기지만, 한 Recovery Episode 안에서 Guard, Action, Verification, Outcome을 이어 주는
정본이 없다. 따라서 여러 Recovery Trace를 비교해도 어떤 사전 상태에서 어떤 Recovery Action이
실제로 성공했는지 안전하게 병합할 수 없고, 관측 실패를 `false`로 오인하거나 실행 상태를
`recovered`로 과장할 위험이 있다.

또한 candidate Recovery Playbook을 만들더라도 행동 수행자의 판독, 승인자의 승인, 동일 evaluator를
사용한 offline replay, 클릭 없는 shadow 관측을 연결하는 승격 경로가 없다. 현재 검증 가능한 성공
Recovery Episode가 0건이므로, synthetic/demo 자료나 `corrected`/`completed` 상태를 성공 근거로
대체해서는 안 된다.

## Solution

기존 workflow_3 수집·추출 경로에 최소한의 Episode 식별자와 구조화된 관측 산출물을 더하고,
workflow_4에 GUI 의존이 없는 Recovery Playbook 도메인 계층을 만든다. 이 계층은 같은 순수
evaluator를 candidate 생성, offline replay, shadow rule 선택에 재사용하고, 승인된 Playbook의
구조만 기존 `WorkflowGraph`로 컴파일한다.

구현 범위는 다음 흐름으로 끝난다.

```text
ALID=9006 active interval
  -> recovery_episode.json + raw evidence
  -> normalized and actor-confirmed Recovery Actions
  -> derived Review Packets + append-only annotations
  -> immutable candidate recovery_playbook.json
  -> R1-R6 replay + office approval
  -> no-click shadow observation + text digest
```

Production Recovery Action 실행은 포함하지 않는다. candidate는 승인 전에도 진단용 shadow로 평가할
수 있지만 corpus 근거가 되지 않으며, 승인된 shadow도 엔지니어가 실제로 한 Action의 provenance만
남긴다.

## User Stories

1. As an Align Fail operator, I want one opaque `episode_id` for one active ALID=9006 interval, so that retries and evidence are not mistaken for separate failures.
2. As an Align Fail operator, I want a cleared alarm followed by a new alarm to receive a new `episode_id`, so that repeated failures remain distinct.
3. As a developer, I want each retry to have a monotonic `attempt_seq`, so that attempt-local failures and eventual recovery can be reconstructed.
4. As a developer, I want Guard, Action, and Verification records to have monotonic `event_seq` values, so that ordering does not depend on filenames or wall-clock precision.
5. As a recovery actor, I want raw JPEGs captured from before the Action through post-Action Verification, so that every semantic claim can point to visible evidence.
6. As a recovery actor, I want frame metadata captured for automatic alarm-cycle recordings as well as manual recordings, so that window geometry, occlusion, and cursor observability are reviewable.
7. As a playbook approver, I want missing or unreadable observations stored as `unknown`, so that lack of evidence never becomes a false negative or an alternate branch.
8. As a playbook approver, I want screen observability, occupancy/control, and SEM mode plus key visibility/uniqueness stored as the only Episode-level Recovery Guards, so that speculative state does not enter rule selection.
9. As an operator, I want OK-control availability checked as an Action precondition, so that a missing control does not get confused with Episode state or recovery failure.
10. As a recovery actor, I want current matcher decisions and correction results retained as provenance, so that useful evidence is not discarded even when it cannot prove recovery.
11. As a playbook approver, I want Measurement normalization or measurement resumption to be the primary Recovery Verification, so that visual quality is authoritative.
12. As a playbook approver, I want strict Recipe Monitor numerator increase used only when Measurement is unreadable, so that a counter increase cannot override observed Measurement failure.
13. As a playbook approver, I want alarm clearance recorded as supporting evidence only, so that disappearance of an alarm cannot independently produce `recovered`.
14. As an operator, I want cursor idle, window closure, OK click, `corrected`, and runner completion excluded from success Verification, so that workflow progress is not reported as equipment recovery.
15. As a developer, I want every required evidence reference to be relative to its Episode root, so that Episode folders can move without breaking provenance.
16. As a developer, I want incomplete Episodes preserved and reason-labeled, so that collection failures are visible rather than silently deleted.
17. As a developer, I want the existing flat workflow extraction preserved as Trace evidence, so that the new domain layer does not rewrite or hide the original inference.
18. As a recovery actor, I want extracted steps classified into a small closed Recovery Action vocabulary, so that different recordings can be compared by domain meaning.
19. As a recovery actor, I want `unclassified` to remain a first-class result, so that unfamiliar controls are reviewed instead of guessed.
20. As a recovery actor, I want machine classifications marked as inferred until I confirm them, so that only confirmed Action meaning can support a Playbook rule.
21. As a recovery actor, I want the classifier to ignore Outcome and Verification, so that later success cannot retroactively change what Action was observed.
22. As an operator, I want classified Actions to contain symbolic targets and normalized parameters but no replay coordinates, so that a bad classification cannot become an unsafe click point.
23. As a developer, I want the Action vocabulary stored as versioned JSON and loaded strictly, so that old Playbooks remain interpretable after vocabulary changes.
24. As a recovery actor, I want one Review Packet per Episode and candidate rule, so that I only review evidence relevant to one proposed rule application.
25. As a recovery actor, I want Review Packets to show representative before/after frames, Guard readings, normalized steps, Verification, Outcome, branch reason, and scope, so that I can judge the proposed semantics without reading raw payloads.
26. As a recovery actor, I want questions generated only for unknown Guards, unclassified or inferred Actions, missing successful Verification, and annotation conflicts, so that review effort stays bounded.
27. As a recovery actor, I want evidence-backed answers appended to `annotations.jsonl`, so that the original Episode and previous opinions are never rewritten.
28. As a recovery actor, I want evidence-free claims preserved as rationale rather than discarded, so that context remains available without changing observed state.
29. As a recovery actor, I want corrections to supersede prior annotations explicitly, so that the audit trail remains intact.
30. As a playbook approver, I want approval records separate from Recovery Annotations, so that observation reading and lifecycle authority are not conflated.
31. As a playbook approver, I want approval blocked while any Review Packet has an open question or unresolved conflict, so that incomplete review cannot become an approved version.
32. As a developer, I want candidate rules merged only when capability scope, Guard values, Action sequence, Verification contract, and Outcome match, so that different causal paths are not collapsed by frequency.
33. As a developer, I want merge results independent of input order, existing rule IDs, and array order, so that repeated builds are deterministic.
34. As a developer, I want `unknown` excluded from wildcard matching, so that an unobserved Guard cannot broaden a rule.
35. As a playbook approver, I want conflicting successful Actions under the same Guard signature kept in `unresolved`, so that majority vote or declaration order cannot choose an unsafe Action.
36. As a playbook approver, I want non-recovered Episodes usable only as constraints or counter-evidence, so that unsuccessful Actions cannot become supporting provenance.
37. As a developer, I want the evaluator to be the single owner of Guard evaluation, unique rule selection, Verification priority, Outcome derivation, and required capability derivation, so that replay and live shadow cannot drift.
38. As a developer, I want the compiler to translate only approved Playbook structure into workflow_4 nodes, so that graph edges do not duplicate Playbook semantics.
39. As an operator, I want every `unknown`, no-match, multi-match, not-applicable, failed precondition, failed/unknown Verification, and handler exception to route to handoff, so that automated selection fails closed.
40. As a developer, I want fallback Actions represented by distinct nodes and visit counters, so that a fallback cannot borrow the primary rule's retry history.
41. As a developer, I want compilation to fail when a referenced Guard, Action, or Verification binding is absent, so that incomplete installations cannot start shadow evaluation.
42. As a playbook approver, I want offline replay to run the same evaluator against every frozen supporting Episode, so that approval checks the actual semantics that shadow will use.
43. As a playbook approver, I want replay to prove consistency rather than claim correctness, so that circular single-Trace evidence remains visibly limited.
44. As a playbook approver, I want each approved evidence cut fixed by Episode hash and annotation cut, so that later file edits invalidate replay instead of silently changing its meaning.
45. As a playbook owner, I want self-approval prohibited when I was the recovery actor, so that my own test recovery cannot qualify its own Playbook.
46. As an office operator, I want each replay, review, approval, and shadow flow runnable by a no-argument command using repository configuration and artifact discovery, so that office execution does not require editing source or constructing CLI flags.
47. As a home developer, I want each office command to emit a single copyable digest line, so that results can return from the office without images or raw evidence.
48. As an office operator, I want candidate shadow mode to perform no clicks and create no supporting attempt, so that diagnosis cannot be mistaken for evidence.
49. As an office operator, I want approved shadow mode to log predicted Actions and observe the engineer's actual recovery, so that rule selection and Verification readers can be measured safely.
50. As a developer, I want shadow digests stratified by EQP and recipe and separated by unseen strata, so that repeated twins do not masquerade as generalization.
51. As an operator, I want the current workflow_3 teardown, notification, cooldown, and abort-latch behavior preserved, so that Playbook work cannot weaken existing safety guarantees.
52. As a maintainer, I want old recordings, `workflow.json`, runner journals, and current monitor behavior to remain readable when Playbook features are disabled, so that rollout is additive and reversible.

## Implementation Decisions

### Delivery boundary and rollout order

- Implement in eight tracer-bullet increments: Episode collection, structured Verification, Action classification, review/annotation, candidate build, evaluator/replay, compiler/bindings, then shadow orchestration.
- Each increment must produce a usable artifact through the highest existing seam before the next increment starts. Do not first build a complete parallel framework and wire it at the end.
- **Office gate after increment 2.** Increments 1-2 are pushed and run at the office before increment 3 starts. The returned digest and files of one real Episode (any Outcome; `incomplete` is allowed) become the producer fixtures for increments 3-8. Hand-written consumer-shaped fixtures for `recovery_episode.json`, the capture manifest, and the Verification decision record are not accepted once a real one exists; synthetic fixtures stay acceptable only for the pure workflow_4 Playbook layer, which receives plain validated data and never parses workflow_3 files itself.
- The final deliverable is **approved-for-shadow only**. No configuration value or entrypoint may make compiled Recovery Actions click the production UI in this spec.
- Unapproved candidates may run diagnostic shadow evaluation, but their results are developer diagnostics only. Approved shadow attempts remain non-supporting provenance for the predicted Action.
- Use Python standard-library dataclasses, enums, JSON/JSONL, `uuid`, `hashlib`, and atomic replace. Do not add a schema, workflow, web, or database dependency.

### Ownership and dependency direction

- workflow_3 owns alarm lifecycle, Episode/attempt identity, capture, concrete Guard observers, concrete Verification readers, Action classification inputs, binding registries, teardown, and the standalone shadow command.
- workflow_4 owns versioned Playbook data validation, candidate merge, effective-annotation reduction, Review Packet derivation, pure evaluation, replay checks, capability derivation, and graph compilation.
- The workflow_4 Playbook domain package must not import workflow_3. Concrete values and binding capability descriptions are passed in as plain validated data.
- The existing workflow_4 framework remains generic. Add Playbook behavior beside it rather than adding Recovery-specific fields to generic node or engine types unless a failing compiler/engine test proves the generic API cannot express the accepted graph.
- The existing workflow_3 `WorkflowRunner` remains unchanged. The current alarm cycle continues to own teardown, notifications, cooldown, recording, and abort handling.

### Recovery Episode lifecycle and persistence

- The alarm monitor creates or resumes one Episode for one ALID=9006 active interval. Its stable alarm fingerprint contains the equipment, alarm code, recipe, and original alarm timestamp; the opaque identity is a UUID and is never reconstructed from a path or timestamp.
- Cooldown retries for the same active alarm reuse the Episode and increment `attempt_seq`. Alarm clearance persists a clearance event and closes that Episode. Any later recurrence creates a new Episode even when equipment and recipe are unchanged.
- The Episode root is the existing per-alarm capture directory `captured_img_from_rcs/<tag>/` (or `_unregistered/<tag>/` when the alarm carries no recipe). `<tag>` is already derived from the alarm's original UTC9 at second resolution, so one tag is one active interval; the tag is a location, never the identity. `recovery_episode.json` lives at that root.
- Each attempt writes under `<episode_root>/attempt_<attempt_seq>/`: its `recording/` (including any prelude sub-folder), capture frames, matcher/feasibility artifacts, numerator decision records, and Verification decision records. This replaces the current shared `<tag>/recording/`, where cooldown retries interleave two takes in one folder. The runner journal stays outside the Episode root and is referenced by run id, not path; it is provenance, not required evidence. `make_demo_video` discovers recordings recursively and needs no change; the `recording_filter` discovery glob gains the `attempt_*` depth; manual recordings keep `_manual/<tag>/recording/`.
- The monitor writes the initial Episode file before starting GUI work. A process restart may resume a still-collecting Episode only when the full alarm fingerprint matches; otherwise it marks the prior Episode incomplete and starts a new one. On its first poll after start the monitor also scans the capture tree for open Episodes whose fingerprint is absent from the alarm list and closes them as `incomplete` with reason `alarm_gone_during_restart`. The eqp-to-Episode map is in-memory and is rebuilt from disk only by this scan.
- `execution_mode` is one of `live` (the production alarm monitor; the attempt records its settings snapshot reference so `safe_mode`/dry-run state stays visible as provenance) or `shadow` (approved shadow attempts). The check-only monitor and diagnostic candidate shadow create no Episode and no attempt.
- `recovery_episode.json` stores schema and observation-contract versions, `episode_id`, alarm attributes and fingerprint, `execution_mode`, `bindings_version`, ordered attempts, evidence references, derived `recovery_actors`, final Outcome, and `complete`/`incomplete` with reasons. It points to existing capture, runner, matcher, classification, and Verification artifacts rather than copying their payloads.
- Episode-relative paths are the only stored artifact paths. Path traversal outside the Episode root is rejected when loading.
- `attempt_seq` starts at 1. Semantic Guard/Action/Verification events inside an attempt receive increasing `event_seq`; wall-clock timestamps remain provenance and never serve as identity.
- Episode writes are atomic. Collection may update a still-open Episode, but a hash-fixed approval cut is immutable; any later evidence or annotation creates a new candidate evidence cut.
- Missing raw evidence marks the Episode `incomplete` but never deletes it. Derived files may be regenerated; raw JPEG and capture metadata are not regenerated or synthesized.

### Capture and durable Recovery Guards

- Reuse one shared metadata capture wrapper for both manual and automatic alarm-cycle `RecordingSession` callers. Do not fork or replace the recorder.
- The wrapper records frame time, window geometry, foreground title, occlusion, cursor position, and Episode/attempt identity without turning cursor position into an Action or intent claim.
- The capture manifest is extended additively with `episode_id`, `attempt_seq`, and capture completeness. Existing consumers that ignore the new fields continue to work.
- Persist exactly three Episode-level Guard groups: screen observability/occlusion, occupancy/control availability, and SEM mode plus align-key visibility/uniqueness. Every reading contains `true`/`false`/`unknown`, a reason, observation time, and evidence reference.
- The third Guard is `true` only when the mode (OM/SEM) was read, the template for that mode matched, and the key was unique. The read mode itself is stored in the reading's detail and provenance, not as a Guard value, so in v1 mode is **not** part of the rule signature: an OM recovery and a SEM recovery with the same three-Guard signature but different Actions land in `unresolved` (fail-closed, accepted). Promoting mode into the signature is a new observation-contract version and changes R4 from eight to sixteen signatures.
- Existing matcher and occupancy outputs are adapted into these Guard readings. Read, parse, stale-sidecar, asset, mode, capture, or matcher failures produce `unknown`; a safety-oriented `candidate` result is not converted to `true`.
- OK-control availability is stored and evaluated as the precondition of `confirm_align`, not as an Episode-level Guard.
- Additional Guard kinds require a new observed Trace that changes Action choice. They are not added pre-emptively.

### Structured Verification and Outcome

- Define the Measurement Verification as a three-state decision record (`success`/`failure`/`unknown` with reason, baseline reference, and post-Action frame reference). Two sources may fill it: an automatic reader, or an actor `verification_reading` annotation pointing at the frames. The annotation is the accepted primary source for the first qualified Episodes; it is not a weaker tier.
- The automatic reader ships as an `unknown`-only stub. The current Assist CV (`sem_monitor/assist_score.py`) counts whole-panel row bands and does not separate the Measurement column from the Addressing columns; a column-aware reader cannot be built or verified without office frames, so it is an office calibration gate, not a home increment. Until calibrated, the stub reuses the Assist panel locator/crop only to persist the crop as evidence.
- Primary success requires readable post-baseline Measurement normalization or measurement resumption, normal rows, no relevant red failure, and a referenced post-Action frame. Unreadable layout, blank Measurement, stale baseline, or crop failure is `unknown`.
- Only when primary Measurement is `unknown` may the evaluator use the existing strictly increasing numerator sequence as fallback. OCR miss, equal/decreasing values, relocalization, or insufficient reads cannot produce success. The numerator decision records that the engineer-done detector currently writes only under a debug directory are written into the attempt directory whenever Episode collection is enabled, because the fallback reads them; the detector's boolean return is not used, since it conflates `false` with `unknown`.
- Alarm clearance is persisted and may support the Episode narrative, but it never independently changes Outcome to `recovered`.
- Cursor idle, window closure, probable-close evidence, correction status, OK click, and runner completion remain provenance only.
- Outcome is derived at Episode level as `recovered`, `escalated`, `aborted`, or `unknown`. A failed or unknown attempt does not overwrite a later qualified recovered attempt, while explicit abort or handoff evidence remains visible in attempt history.
- Reaching a graph node named `handoff` does not itself set Outcome to `escalated`; escalation requires a persisted explicit handoff. Guard or Verification uncertainty remains Outcome `unknown`, and an observed abort latch or unrecoverable cancellation produces `aborted`.

### Recovery Action vocabulary and classification

- Store Action vocabulary v1 in `recovery_actions.v1.json`. It contains kind, parameter schema, required preconditions, and required binding capabilities for `reposition_to_align_key`, `pan_fov`, `set_magnification`, `confirm_align`, and `handoff_to_engineer`.
- The v1 parameter contracts are fixed: `reposition_to_align_key` uses the symbolic target `matched_recipe_box_center`; `pan_fov` uses `direction` (one of eight compass points) and `magnitude` (`small`/`medium`/`large`, binned from the live-box ratio offset by thresholds stored in the vocabulary file), with the raw `(dx, dy)` ratio kept only in provenance; `set_magnification` uses one positive absolute `mag`; `confirm_align` and `handoff_to_engineer` have no parameters. `confirm_align` requires `ok_control_available`. Continuous values never enter a parameter contract, because the merge key compares parameters by equality and a raw offset would make every pan rule single-Episode and `circular`.
- Loading a missing, malformed, duplicate, or unsupported vocabulary hard-fails classification and compilation. A new vocabulary version is a new adjacent JSON file; old versions stay readable.
- Extend workflow extraction with one pure classifier that consumes a grouped step, its representative before/after machine readings, and sequence context. It returns vocabulary version, Action kind or `unclassified`, normalized parameters, source, evidence reference, hint, and `inferred`.
- Apply the resolved first-match rules: live-image double-click followed by confirmed align becomes `reposition_to_align_key`; navigation without confirm becomes `pan_fov`; a readable PM before/after difference becomes `set_magnification`; an OK/confirm control click becomes `confirm_align`; everything else is `unclassified`.
- Classification runs in two deterministic passes: first classify control-click and PM-difference steps that do not depend on neighbors; then classify live-image double-clicks using the next semantic step. Reposition may be corroborated only by the after-frame matcher crossing its calibrated ensemble threshold; the before frame does not prove reposition intent.
- PM difference is `same`, `diff`, or `unknown`. A changed but unreadable value remains `unclassified` with a magnification-change hint.
- The classifier does not receive Verification or Outcome and never emits screen coordinates. Existing flat steps and raw-event references remain unchanged.
- Machine classifications are proposals. Only an actor-confirmed effective Action can support candidate merge. The digest line format is `<seq> <kind>(<params>) src=<rule|cv|ocr|annotation> inferred=<0|1> t=<sec>`.

### Review Packet, annotations, and approval records

- Build the Review Packet as a deterministic derived model from one Episode, one candidate rule, and the effective annotation cut. Do not persist it as another source of truth.
- Render the model as self-contained local HTML using existing offline rendering patterns. The HTML is read-only and links or embeds Episode-relative evidence; it does not write canonical files.
- Provide a no-argument terminal reviewer that opens or reports the newest eligible packet, presents only derived open questions, and appends answers to the Episode's `annotations.jsonl`. Repository configuration may select a different Episode; do not add `argparse` flags.
- Each annotation line contains `annotation_id`, role, actor identity, kind, target reference, value, evidence reference, superseded annotation, rationale, and timestamp. Kinds are limited to `guard_reading`, `action_meaning`, `verification_reading`, and `rationale`.
- Actor identity comes from repository configuration (`RECOVERY_REVIEW_ACTOR`, a free identity string) at the moment the terminal reviewer appends a line; the reviewer refuses to append without it. An Episode's `recovery_actors` is the derived set of actor identities on its live `guard_reading`/`action_meaning`/`verification_reading` annotations. The automatic cycle cannot know who stood at the tool, so an Episode with no such annotation has no actor and cannot be supporting. Identity and approval site (`approval_site="office"`) are declared values recorded for audit; the software does not verify them.
- A value-changing annotation without evidence is appended as `rationale` and does not alter effective state. Conflicting live annotations produce `unresolved`; timestamp order never resolves a conflict automatically.
- Only the Recovery Actor may change Guard, Action meaning, or Verification readings. The Playbook Approver may add rationale but not observational readings.
- Approval is a separate append-only record tied to an exact `playbook_version`, Episode hashes, annotation cuts, replay report, freeze time, circular flags, and pre-approval digest strata.
- `approved` is accepted only when every supporting Review Packet has zero open questions, no unresolved entries exist, replay passes, the approver is the configured Playbook owner with `approval_site="office"` declared, and the owner's identity is absent from `recovery_actors` of every supporting Episode. Other decisions are `needs_evidence` or `rejected`.

### Candidate building and merge

- Build candidate rules from qualified rule-application segments that start with pre-Action Guard readings and end with ordered confirmed Actions plus Verification and Outcome.
- A Playbook version contains `schema_version`, `playbook_id`, immutable `playbook_version`, lifecycle and approval references, derived capability scope, explicit rule-selection policy, `action_vocabulary_version`, `observation_contract`, `rules`, and `unresolved`.
- Each rule contains `rule_id`, three-state Guards, ordered Actions, primary/fallback Verification, transitions, nullable `fallback_rule_id`, and provenance references. Frames, screen coordinates, raw event payloads, and copied Episode data are forbidden.
- The semantic merge key is capability-compatible scope, Guard kinds and values, ordered Action kinds and domain parameters (as binned by the vocabulary), Verification contract, and Outcome. `unknown` is a literal value, never a wildcard. The read OM/SEM mode is not a key component in v1 (see Guards).
- Exclude coordinates, raw pan offsets, OCR labels, equipment/recipe IDs, timestamps, array positions, and existing rule IDs from the merge key. Preserve them only in provenance.
- Treat provenance as a set keyed by `episode_id`, `attempt_seq`, and `event_seq` range. Duplicate input is a no-op.
- Sort canonical semantic keys before assigning stable sequential rule IDs, so input order cannot change output.
- Create a separate branch only when qualified recovered evidence shows a pre-Action Guard difference paired with a different successful Action. Same-signature Action conflicts stay in `unresolved`.
- Non-recovered Episodes may narrow scope or expose conflicts, but cannot support a successful rule or fallback.
- A fallback references another independently qualified rule. No inline fallback Action is generated, and v1 keeps `fallback_rule_id=null` until evidence supports one.
- Every build emits a new immutable candidate version when scope, rules, provenance, unresolved state, or observation/vocabulary contract changes. The Playbook itself has no document content-hash field.

### Evaluator and compiler

- Expose one pure evaluator as the only owner of three-state Guard evaluation, installed-capability applicability, unique rule selection, Verification primary/fallback priority, Outcome derivation, failure classification, and required-capability derivation.
- Selection succeeds only when exactly one rule has all required Guards `true`. Any required `unknown`, no match, multiple matches, or missing capability selects no Action and returns its explicit closed failure class.
- The closed failure classes are `guard_unknown`, `no_match`, `multiple_matches`, `not_applicable`, `precondition_failed`, `verification_unknown`, `verification_failed`, and `handler_exception`.
- Compile only an approved Playbook version. The compiler validates vocabulary, observation contract, binding version, referenced kinds, and graph validity before returning a graph.
- The compiled graph contains `select_rule`, per-rule `act` and `verify` nodes, optional distinct fallback Action/Verification nodes, and `recovered`, `handoff`, and `aborted` terminals.
- Every compiled node has `max_retries=0` and an explicit `failure_routes` entry for every closed failure class plus the engine-native classes (`handler_exception`, `invalid_outcome`, `step_budget_exhausted`, `global_budget_exhausted`). The generic engine retries a failure in place when no route matches; an unrouted class would re-evaluate the same frame or re-run an Action, so routing is the compiler's responsibility and compile-time validation rejects a gap.
- Terminal node names are control-flow results, not Recovery Outcome. Evaluator-derived Outcome remains a separate run-state value.
- Primary Verification success reaches recovered. Verification `unknown` always hands off. Readable Verification failure reaches a fallback only when an independently qualified fallback rule is declared; otherwise it hands off.
- The existing abort callable is composed with a wall-clock deadline into the single `abort_check` handed to the engine (`abort_check = outer_abort() or now > deadline`); the generic engine gains no deadline field. The engine polls `abort_check` only between nodes and during retry cooldown, so every potentially long binding receives the same callable and polls it itself. Any future live integration must keep its graph artifacts separate from the outer cycle mirror and link them rather than overwriting it.

### Offline replay and promotion gate

- Replay evaluates the frozen effective Episode state: `recovery_episode.json` plus annotations only through the recorded annotation-ID cut. It uses the same evaluator and compiler versions as shadow and never invokes GUI bindings.
- R1 requires the unique selected rule, confirmed Action sequence, and primary/fallback Verification path to reproduce every supporting provenance segment.
- R2 requires all three durable Guards on every rule, mutates each Guard to `unknown` in turn, and requires no Action plus handoff. A non-unknown reading without reason and evidence is treated as `unknown` and flagged.
- R3 requires at least one qualified `recovered` Episode with successful Verification evidence and zero Review Packet questions per rule. Rules with zero qualified support are removed before the candidate can pass.
- R4 requires empty `unresolved` and at most one matching rule for all eight three-Guard signatures.
- R5 requires every supporting Episode to use the Playbook's exact observation contract.
- R6 requires complete installed bindings for every referenced Guard, Action, and Verification kind and rejects compilation otherwise.
- `scope.required_capabilities` is derived by the evaluator from referenced Guard and Verification kinds. It is never hand-declared in the Playbook.
- `replay_report.json` is immutable per Playbook version and evidence cut, uses `evidence_kind="consistency"`, and marks any rule with one supporting Episode as `circular=true`. Replay passing is not correctness evidence and cannot be cited by a later production-click gate.
- Replay refuses to run when any frozen Episode hash or annotation cut differs from the approval input.

### Binding registry and shadow mode

- workflow_3 exposes one versioned registry describing Guard observers, Recovery Action handlers, Verification readers, and their capabilities. Registry completeness is data visible to replay/compiler, not a second copy of Playbook semantics.
- Reuse existing matcher, SEM controller, magnification control, OK locator, Measurement reader, numerator reader, and handoff seams. Do not copy service or GUI logic into workflow_4.
- Live Action bindings may be registered and unit-tested in safe mode, but this spec exposes no live execution entrypoint. Shadow uses logger bindings for Action nodes and therefore cannot invoke mouse or keyboard functions.
- Implement shadow first as a standalone no-argument workflow_3 command, separate from the production alarm monitor. It discovers the configured active Episode/tool context, records pre-Action observations and a prediction, then observes the Recovery Actor's manual work and Verification.
- Diagnostic candidate shadow writes only a developer prediction report and digest. Approved shadow writes an `execution_mode="shadow"` attempt, but attributes observed Actions and Verification to the Recovery Actor, never to the predicted rule.
- Shadow digest contains only Guard unknown rates by equipment/recipe, selection-class distribution, predicted-versus-confirmed Action sequences, and Verification-reader agreement with the actor-reviewed result. There is no aggregate accuracy number.
- Generalization claims may use only post-freeze Episodes from equipment/recipe strata whose digest was not seen before approval.

### Operational entrypoints and compatibility

- Provide no-argument entrypoints for Episode post-processing, Review Packet rendering/annotation, candidate build, replay, approval, and shadow. Inputs come from `.env`/repository configuration or newest unambiguous eligible-artifact discovery.
- Ambiguous discovery fails with a list of candidates and no mutation. Missing evidence fails closed with one reason-labeled digest.
- Replay emits `[DIGEST] replay playbook=<ver> cut=<...> episodes=N rules=M pass|fail reasons=<R1..R6>`.
- Each office-only command emits one final `[DIGEST]` line suitable for copy/paste to the home environment; verbose logs and JSON reports remain local.
- Existing feature flags default Playbook collection and shadow behavior off until their focused tests pass. Enabling Episode collection is additive; enabling shadow never enables correction clicks.
- Existing recordings and workflow extraction without Episode metadata continue to work as legacy, non-qualified evidence. They may be reviewed, but cannot pass the observation-contract check without recollection. The one legacy code change is the `recording_filter` discovery glob gaining the `attempt_*` depth; old `<tag>/recording/` folders stay discoverable.

## Testing Decisions

- Good tests assert externally visible contracts: persisted JSON/JSONL, selected rule or failure class, graph shape, digest text, and absence of GUI calls. They do not assert private helper call order or duplicate every dataclass field in isolated tests.
- Prefer the highest existing seam for each tracer bullet: alarm-row processing for Episode identity, recording session output for capture metadata, workflow extraction output for Action classification, candidate build/replay for Playbook semantics, compiler plus `WorkflowEngine` for graph behavior, and the standalone command for no-click shadow.
- Reuse existing temporary-directory patterns, injected capture/read/click functions, synthetic images, fake alarm rows, fake binding registries, and workflow_4 graph validation. Do not require Windows for pure contracts.

Required automated coverage:

1. One active alarm with cooldown retries produces one Episode, increasing attempts, distinct `attempt_<n>/recording/` folders, and no timestamp/path-derived identity.
2. Alarm clearance persists a clearance event; a later recurrence creates a different Episode.
3. A restart resumes only an exact active-alarm fingerprint, marks mismatches incomplete, and closes an open Episode whose alarm is gone as `incomplete(alarm_gone_during_restart)`.
4. Automatic recording writes frame metadata and Episode/attempt fields while legacy consumers still read the manifest.
5. Capture or sidecar failure preserves the Episode and produces Guard `unknown`, never `false` or `true`.
6. The three durable Guard groups serialize with reason, observation time, and Episode-relative evidence.
7. Measurement success, Measurement failure, and unreadable Measurement produce distinct structured decisions, whether filled by an actor `verification_reading` annotation or by a reader; the shipped reader stub yields only `unknown` and still persists its crop.
8. Numerator fallback is considered only for Measurement `unknown`, reads the decision records from the attempt directory rather than the detector's boolean, and OCR miss, equal/decrease, and reground reset cannot recover.
9. Alarm clear, cursor idle, window close, correction status, runner completion, and probable-close evidence cannot independently produce `recovered`.
10. Every Action-classification rule has one positive and one adjacent negative example, including next-step confirm context and PM `same`/`diff`/`unknown`.
11. Classification receives no Outcome, emits no coordinates, and returns `unclassified` for unsupported or unreadable steps; `pan_fov` emits binned `direction`/`magnitude` only, two offsets in the same bin merge, and the raw offset survives only in provenance.
12. Missing or malformed vocabulary fails classification and compilation; old vocabulary versions remain readable.
13. Review questions appear only for the five resolved ambiguity classes and disappear from effective state after evidence-backed resolution.
14. Evidence-free value changes are appended as rationale and leave effective observation unchanged.
15. Conflicting annotations yield `unresolved`; a valid superseding annotation resolves effective state without deleting history.
16. Approver observation edits are rejected, actor semantic edits are accepted, approval remains separate, an append without a configured actor identity is refused, and `recovery_actors` derives from live annotations so an Episode whose only actor is the owner cannot support approval.
17. Candidate merge is invariant under shuffled Episodes, duplicated provenance, shuffled rules, and pre-existing rule IDs.
18. `unknown` never behaves as a wildcard, equipment/recipe IDs and the read OM/SEM mode never create a branch, and same-signature Action conflicts stay unresolved.
19. Non-recovered and incomplete Episodes cannot become supporting provenance.
20. Evaluator selection covers unique, no-match, multiple-match, Guard-unknown, and not-applicable outcomes.
21. Verification covers primary success, fallback success only after primary unknown, readable failure, and total unknown.
22. R1 reproduces supporting rule/Action/Verification paths for every Episode.
23. R2 mutates each durable Guard to `unknown` and proves no Action is selected.
24. R3 removes zero-qualified rules and requires recovered evidence plus zero Review Packet questions.
25. R4 exhaustively evaluates all eight three-Guard signatures and rejects any multi-match.
26. R5 rejects observation-contract mismatch.
27. R6 rejects missing Guard, Action, or Verification bindings before graph execution.
28. Replay reports `evidence_kind="consistency"`, flags single-Trace rules as circular, is deterministic for the same evidence cut, and refuses a hash mismatch.
29. Compiler output has the accepted node shape, distinct fallback IDs/counters, `max_retries=0` on every node, a route for every closed and engine-native failure class, and passes generic graph validation; a missing route fails compilation.
30. The engine receives the composed abort callable; a fake long-running binding that polls it stops at the wall-clock deadline, and the generic engine source is unchanged.
31. Candidate shadow calls zero GUI Action functions, writes no supporting attempt, and emits the four-part digest.
32. Approved shadow records a shadow attempt, attributes actual Actions to the actor, and still calls zero GUI Action functions.
33. Ambiguous no-argument artifact discovery fails without modifying Episode, annotation, Playbook, replay, or approval files.
34. All persisted artifact references reject absolute paths and parent-directory escape.
35. Existing focused workflow_3 recording/filter/extraction/monitor tests and the full workflow_4 suite continue to pass, and `recording_filter` discovery finds both `<tag>/recording/` and `<tag>/attempt_<n>/recording/`.

Office acceptance is deliberately narrower than production validation:

- Run increments 1-2 first: confirm one real Episode (any Outcome) lands under `captured_img_from_rcs/<tag>/` with `attempt_<n>/` folders and copy its digest home before increment 3 begins.
- Run each office entrypoint without CLI arguments and confirm its final single-line digest can be copied home.
- Calibrate the column-aware Measurement reader only against office frames; until then the actor `verification_reading` annotation is the primary Verification source.
- Inspect one Review Packet against its referenced frames and confirm an appended annotation changes only effective derived state.
- On the first real successful Episode, confirm the required evidence bundle, primary/fallback decision, actor identity, and Episode hash before calling it qualified.
- Run diagnostic shadow with action hooks instrumented and confirm no mouse or keyboard operation occurs.
- Do not approve shadow until a non-owner Recovery Actor supplies the first qualified recovered Episode and R1-R6 pass.

## Out of Scope

- Production execution of compiled Recovery Actions or any rollout gate that enables automatic clicks.
- End-to-end imitation learning, autonomous VLM policy selection, or replay of recorded screen coordinates.
- New Recovery Guard kinds not demonstrated by a Trace that changes Action selection.
- A database, remote review service, multi-user web application, or external workflow dependency.
- Copying office screenshots or raw Episode artifacts to the home environment.
- Treating synthetic matcher data, workflow_4 demos, pre-action aborts, `corrected`, or `completed` as successful Recovery Episodes.
- Replacing workflow_3 runner, teardown, notification, cooldown, recording, or abort contracts.
- Align Fail behavior outside ALID=9006 and general recipe-creation automation.

## Further Notes

- The nine decision tickets are resolved; this spec is ready for implementation decomposition.
- `ready-for-agent` means the implementation boundary is decided. It does **not** mean a Playbook can already be approved: the current qualified recovered corpus remains 0.
- The first implementation stops after increments 1-2 (Episode collection + structured Verification) land and one real Episode digest has come back from the office; that Episode's files are the fixtures for increments 3-8. Synthetic contract fixtures remain acceptable only for the pure workflow_4 Playbook layer. The first real Episode is an office evidence gate, not a reason to add fake production data.
- The Review Packet HTML renderer has no consumer before office approval; within increment 4 it is built last and may slip until the first approval is in sight, as long as the terminal reviewer and `annotations.jsonl` land.
- Attempt-scoped recording folders also close the known tag-collision defect (cooldown retries interleaving two takes in one `recording/`); do not fix that defect separately.
- The static Review Packet plus terminal append flow is intentional. Add a local review server only if office use proves that opening evidence and appending annotations as two local steps is materially inadequate.
