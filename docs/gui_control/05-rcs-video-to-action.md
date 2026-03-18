# RCS Video-To-Action

This document merges the RCS video overview, implementation guide, and task breakdown into one roadmap.

## 1. Core Decision

Do not treat raw AVI files as immediate training input.

The first useful asset is:

`video -> manual-control episode -> step trajectory -> retrieval memory`

That is more realistic than direct end-to-end retraining and fits the current repository structure better.

## 2. Target Output

Each useful step should eventually carry at least:

- `pre_frame`
- `post_frame`
- `action_type`
- `action_args`
- `target_bbox`
- `target_text`
- `effect_summary`
- `task_objective`
- `success`
- `confidence`

The objective is searchable trajectory memory, not raw replay only.

## 3. Recommended Pipeline

1. inventory all AVI assets
2. pick a representative gold set
3. mine manual-control episodes
4. detect and track the white cursor
5. extract click, double-click, drag, scroll, and `type_candidate`
6. ground targets with VLM and OCR
7. summarize local objective and effect
8. store trajectory records
9. retrieve similar records during online execution
10. verify post-action success with new screenshots

## 4. Why Video-Only Is Hard

Known constraints:

- most recorded time may be auto-running and not useful
- keyboard events are not directly observable
- text entry often has to be reconstructed from OCR diff
- cursor shape changes can break simple tracking

That is why the cursor and local UI change signals are the first-class features.

## 5. Work Package Compression

The old task list can be read as five practical phases:

### Phase A: Data And Sampling

- inventory table
- representative gold set

### Phase B: Episode And Cursor Extraction

- manual-control episode mining
- white cursor detection
- cursor tracking and recovery

### Phase C: Event Building

- click / double-click / drag / scroll extraction
- OCR-diff based `type_candidate`
- step segmentation

### Phase D: Grounding And Memory

- target grounding with VLM/OCR
- local objective generation
- trajectory storage and retrieval schema

### Phase E: Online Planner And Verification

- next-action planning
- post-action verification
- human-in-the-loop rules
- evaluation protocol

## 6. Repo Mapping

- `test/video_frame_parser/`: offline frame and episode extraction
- `test/vlm_input_control/`: retrieval context and older prompt-building logic
- `poc/work2/`: online state recognition, action planning, and verification

This is an additive architecture, not a replacement of `poc/work2`.

## 7. Evaluation Priorities

Measure:

- cursor recall and position error
- click/double-click extraction quality
- OCR-diff quality for typed values
- target grounding accuracy
- planner success rate
- end-to-end step verification accuracy

## 8. Practical Scope Rule

Keep the first version narrow:

- video-only assumption
- white cursor assumption
- recipe-related GUI actions only
- no full keyboard replay reconstruction

That scope is large enough to be useful and small enough to implement.
