---
type: Architecture Overview
title: Repository Architecture
description: System boundaries, dependency direction, runtime orchestration, data flow, and recent architectural evolution for the auto recipe creator monorepo.
resource: "poc/workflow_3"
tags: [architecture, workflow-3, rcs, computer-vision]
---

# Repository architecture

## Product shape

The repository combines three related layers:

1. **Equipment automation** — [the align-fail workflow](../workflows/align-fail.md) in `poc/workflow_3`, plus additive jobs in `poc/workflow_3e`.
2. **Model-backed perception services** — [VLM integrations](integrations.md) in `flask_api/vlm_serve`, `deploy_vlms`, and workflow-specific clients/prompts.
3. **Offline research and knowledge extraction** — [research pipelines](../workflows/research-pipelines.md) in `poc/workflow_2` and `side_projects/document_extraction`.

The root Flask application (`web_main.py`) is infrastructure, not the equipment-loop entrypoint. It registers the VLM API tree and a placeholder GPU-dashboard blueprint; office automation runs as Python scripts under `poc/workflow_3/monitor` or `poc/workflow_3e`.

## Production package boundaries

`poc/workflow_3` is a layered package:

```text
monitor
  -> align, rcs, sem_monitor, runner, vlm, util
align / rcs / sem_monitor
  -> vlm, runner, util as needed
util
  -> leaf helpers
```

- `monitor/` owns polling, edge triggering, per-alarm orchestration, recording, notification, and office adapter loading.
- `runner/` supplies declarative `WorkflowStep`, condition, result, and journal structures. `WorkflowRunner` is fail-fast; domain executors perform most practical retry behavior.
- `rcs/` owns login, tool selection, window handling, screenshots, and teardown.
- `align/` owns assets, templates, matching, correction, consensus, and fallback search. `align/matching/engine.py` is coordinate authority.
- `sem_monitor/` translates screen/panel coordinates into real double-click, wheel, and OK-button operations.
- `vlm/` owns service selection, API clients, and prompt builders.
- `recording_filter/` is an offline postprocessor and is not on the live loop's hot path.

The architectural constraint is one-way migration: workflow 3 never imports workflow 1/2. The [source map](../source-map.md) identifies the legacy and bench boundaries.

## Alarm-cycle control flow

`monitor/align_fail_monitor.py` polls an office or replay source, filters recent Align Fail rows, collapses them by equipment, and edge-triggers only new tools. A tool remains active until its alarm disappears. An occupied RCS `select` popup is not touched; instead, the tool enters a bounded cooldown and may be retried.

`monitor/cycle.py` builds these steps:

```text
ensure RCS ready
-> close alert popup
-> connect tool
-> wait for Remote Monitoring window
-> start screenshot recording
-> locate SEM panel / create controller
-> run CV correction
```

Cleanup is intentionally outside the declarative steps. Recording stop, notification backstops, input release, and tool close are guarded by outer `try/finally` logic so teardown still runs when a step fails. The [operations runbook](../operations/runbook.md) explains the resulting artifacts and failure diagnosis.

## Perception and actuation split

The system separates qualitative perception from quantitative action:

- VLM/OCR locates UI controls, SEM regions, PM values, or Recipe Monitor counters.
- Classical CV proposes and scores align-key locations.
- The correction layer applies a visibility/distinctiveness gate before choosing immediate correction or live search/manual escalation.
- Rerankers may select only among existing candidates; they do not produce a new coordinate.
- Real input is controlled by settings and independent dry-run gates.

This split lets [the CV bench](../workflows/research-pipelines.md#cv-evaluation-and-porting) measure changes offline while the live loop retains explicit safety boundaries.

## Data flow

The primary MES/runtime tree is:

```text
align_images/<eqp_id>/<class>/<recipe>/
├── align_img_from_rcp/       # IMAP0001 OM, IMAP0002 SEM registered references
├── align_img_from_msr/       # S/E trajectory; offline bench only at runtime
└── captured_img_from_rcs/    # event captures and recording frames
```

`poc/workflow_3/__init__.py` currently defaults this root to `poc/workflow_3/align_images`, overridable by `ALIGN_IMAGES_DIR`. `align/assets.py` is the central reader. Production correction consumes RCP plus optional consensus, not `align_img_from_msr`; offline fetching is explicit through `monitor/fetch_msr_offline.py`.

Consensus uses a separate recipe-keyed cache:

```text
<ALIGN_CONSENSUS_CACHE_DIR>/<class>/<recipe>/events/<event_id>/S*.jpeg
```

It deliberately omits equipment ID so successful examples for the same recipe are shared across tools. `align/consensus_gather.py` centralizes path construction, and `monitor/success_gather.py` deduplicates in-flight work by recipe. Reintroducing equipment into this path silently fragments the pool and defeats the office writer contract.

## Extension architecture

`poc/workflow_3e` demonstrates the preferred extension style. It imports workflow 3, subclasses settings, reuses RCS/runner executors, and adds independent detection/dispatch/manifest modules. Its unified supervisor polls once, processes align rows, then measurement-abort rows in the same process. This preserves the single-cursor serialization constraint without a cross-process lock.

New alarm jobs should follow this direction rather than adding unrelated branches and flags to `workflow_3/monitor/cycle.py`. See the [align workflow extension section](../workflows/align-fail.md#measurement-fail-abort-extension).

## Evolution visible in git

Recent history shows a deliberate research-to-production progression:

- Workflow 1's GUI automation and workflow 2's CV correction were consolidated into workflow 3 as the production package.
- Consensus and registration were developed in the workflow 2 golden drivers and selectively ported.
- The July rerank sequence tested multiple pseudo-arms before choosing modality-aware routing: OM fuses baseline/NCC selection with MIND ranks; SEM switches to ECC rank alone because combining it diluted the strongest signal.
- Registration evaluation then separated **proposer misses** from **rank errors**, producing a recipe-level re-registration worklist rather than assuming every failure is fixable by another verifier.
- Workflow 3e was added as an extension package to keep new alarm jobs from destabilizing the align core.

These decisions are encoded in current source and tests, not only commit messages. For changes, use the [testing matrix](../operations/testing.md#change-to-check-matrix) and preserve bench/production parity where algorithms are duplicated.
