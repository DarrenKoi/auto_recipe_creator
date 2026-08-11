---
type: Workflow Guide
title: Align-Fail Detection and Correction
description: End-to-end behavior, domain vocabulary, matching policy, data contracts, safety gates, failure handling, and extension jobs for production CD-SEM align-fail automation.
resource: "poc/workflow_3/monitor/align_fail_monitor.py"
tags: [align-fail, cd-sem, workflow, computer-vision, safety]
---

# Align-fail workflow

## Business goal and trigger

An **Align Fail** means a CD-SEM tool cannot align the current wafer view to the recipe's registered align key. Alarm `ALID=9006` triggers the production workflow. The objective is not merely to classify the screen: it is to recover the correct target point safely, preserve evidence for manual handling, and avoid disrupting another engineer's RCS session.

The registered align key has separate OM and SEM references. An engineer-drawn box marks the distinctive pattern; its center is the intended target. The live crosshair marks the current, possibly wrong, attempt. A double-click recenters the live view on the chosen point; a separate dialog OK click confirms progression.

## Alarm lifecycle

`monitor/align_fail_monitor.py` performs edge-triggered dispatch by `EQP_ID`:

1. Poll office or replay alarm source.
2. Filter Align Fail rows within the configured time window.
3. Collapse duplicate rows per tool.
4. Process only tools not already active or in occupied cooldown.
5. Start asynchronous success-image gather and synchronous RCP gather.
6. Run the alarm cycle and append a one-row manifest.
7. Re-arm only after the alarm row disappears.

If RCS reports the tool is occupied through a `select` popup, the workflow does not choose share/terminate options. It records an occupied failure, leaves the tool out of the active set, and retries after cooldown. This behavior is part of the safety model described in the [operations runbook](../operations/runbook.md#safety-gates).

## Correction decision

`align/correction.py:correct_align_fail_auto()` is the domain entrypoint. It resolves assets/templates, captures the paused live frame, and invokes the production ensemble matcher. The result contains the best candidate, scale, decision, top candidates, and ambiguity evidence.

A visibility/distinctiveness gate separates:

- **Primary path:** key is visible and sufficiently trustworthy; reposition to the selected candidate and locate/click OK when actuation is armed.
- **Fallback/review path:** key is absent, weak, or ambiguous; attempt bounded live search/zoom support or escalate to the engineer while recording continues.

`align/live_search.py` isolates equipment movement behind `SEMMonitorController`: double-click recenter, wheel zoom, mode-aware templates, broad spiral search, then zoom-in confirmation. The check-only monitor can run zoom-ladder/PM-dropdown probes and save marked evidence without entering the production correction path.

## Matching and modality routing

`align/matching/engine.py` is coordinate authority. The paused-frame ensemble pipeline performs proposer fusion, structural rescoring, NCC selection, then a modality-aware verifier:

- **OM:** fuse baseline/NCC selection order with MIND self-similarity order using reciprocal-rank fusion.
- **SEM:** use ECC correlation order alone. Bench results showed ECC dominated this modality and RRF combination diluted it.

Both paths only reorder existing candidates. Out-of-frame, flat, low-score, or non-convergent candidates are rejected; if all candidates are rejected, selection falls back to the prior NCC result. Kill switches are `ALIGN_FAIL_MIND_RERANK=0` and `ALIGN_FAIL_ECC_RERANK=0`.

The policy came from [workflow 2 registration evaluation](research-pipelines.md#cv-evaluation-and-porting), where multiple pseudo-arms were compared before the final OM/MIND and SEM/ECC split was ported. Keep constants and gates in `workflow_2/registration_lab.py` and `workflow_3/align/matching/mind_rerank.py` aligned; they are duplicated rather than shared.

## Templates and consensus

`align/templates.py` builds modality-aware `AlignKeyTemplate` objects from registered recipe images. When `cond.txt` is available, `cond_template.py` can crop to the engineer's box while preserving a decoupled target offset. `ALIGN_FAIL_COND_BOX_CROP=0` rolls back to the whole-template path.

Production prefers a trusted consensus template derived from recent successful S images, then falls back per modality to RCP when cache data is absent, insufficient, blurry, delayed, or invalid. `ALIGN_FAIL_CONSENSUS=0` is the master rollback. The minimum is clamped to at least three samples; current default is four. A cold cache receives one bounded wait, then proceeds with RCP while background gathering may warm a later cycle.

## Data contracts

### MES/runtime asset tree

```text
<ALIGN_IMAGES_DIR>/<eqp_id>/<class>/<recipe>/
├── align_img_from_rcp/       IMAP0001.* (OM), IMAP0002.* (SEM)
├── align_img_from_msr/       S* success / E* failure trajectory
└── captured_img_from_rcs/    captures, overlays, recordings
```

`RECIPE_ID` has `<class>/<recipe>` form. Runtime correction consumes RCP plus consensus; MSR is explicitly offline-bench-only. `cond.txt` is stored in a hidden sidecar directory `.<image-name>/cond.txt`; cursor coordinates may be 10x oversampled. Do not flatten or omit hidden sidecars during data migration.

Current source defaults `ALIGN_IMAGES_DIR` to `poc/workflow_3/align_images`. Office MES must write there or the environment must point to the actual MES root before importing workflow 3.

### Consensus cache

```text
<ALIGN_CONSENSUS_CACHE_DIR>/<class>/<recipe>/events/<event_id>/S*.jpeg
```

The absence of `<eqp_id>` is intentional: equal recipes share successful history across tools. Do not merge this derived cache with the equipment-keyed MES tree. The [architecture overview](../architecture/overview.md#data-flow) explains ownership.

### Alarm rows

Core fields are `EQP_ID`, `ALID`, `UTC9`, and usually `RECIPE_ID`; names/operation/lot fields supply logs and notifications. A missing recipe does not prevent connection/recording, but it prevents recipe-based correction and routes artifacts under `_unregistered`.

## Outcomes and engineer handoff

`CorrectionOutcome.status` drives notification behavior: corrected, fallback variants, no assets, OK-locator error, or escalation. Recording captures automatic and manual operations using change detection plus heartbeat frames. Optional engineer-done detection grounds and OCRs the Recipe Monitor counter, then ends watch early only after the configured count is observed consistently.

The cycle emits audit logs, a CSV manifest, per-step journals, captures/overlays, and recording frames. Use [operations](../operations/runbook.md#artifacts-and-observability) to locate them and [testing guidance](../operations/testing.md) before modifying an outcome or gate.

## Re-registration feedback

Ambiguous keys can be flagged for human re-registration when the second candidate is too close to the best candidate. Separately, the workflow 2 registration bench creates a recipe-level proposer-miss worklist. These signals are advisory outputs, not an automatic registration queue. A proposer miss means the truth never reached top-K and a reranker cannot fix it; a rank error means the truth was present and verifier changes may help.

## Measurement-fail abort extension

`poc/workflow_3e` adds a consecutive-measurement-fail abort job while preserving [the core architecture](../architecture/overview.md#extension-architecture). `monitor.py` is the unified production supervisor: it polls once, dispatches align rows first, then abort rows, keeping the single RCS cursor serialized.

Abort flow is connect -> capture evidence -> locate Abort/Stop -> optionally click -> notify -> guaranteed teardown. It ships notify-only. Real abort requires both `SAFE_MODE=0` and `MEAS_FAIL_ABORT_DRY_RUN=0`; an empty `MEAS_FAIL_ALID` detects nothing. The provider may repeatedly return an active row because `dispatch.py` performs edge triggering and re-arms when the row disappears.

A crucial limitation is that the abort path's final state is not a strong measurement-stopped postcondition: if no confirmation control is found after the destructive click, the cycle may still report the attempted abort. Office validation must therefore review evidence and actual equipment behavior before arming.
