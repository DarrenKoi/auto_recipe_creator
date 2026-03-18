# Grounding And Hybrid Automation Patterns

This document merges the old automation strategy, Microsoft vision-tool survey, hybrid pattern notes, and the broader engineering GUI automation research into one operational guide.

## 1. Default Loop

The default loop for this repo is:

1. observe the current screenshot
2. decide the next action with the smallest sufficient toolset
3. act through mouse or keyboard control
4. verify the result with a new screenshot

Anything more optimistic than that becomes fragile in RCS-like software.

## 2. Role Split By Tool

| Tool | Best role | Avoid using it as |
|------|-----------|-------------------|
| `UI-Venus` | full-screen GUI grounding and semantic target selection | exact text authority |
| `UI-TARS` | alternate primary model, multi-step action reasoning | drop-in OCR replacement |
| `MAI-UI` | zoom-in grounding sidecar for small crops | always-on full-screen primary |
| `PaddleOCR-VL-1.5` | text, spotting, layout, table reading | primary click planner |
| `GOT-OCR-2.0-hf` | hard crop OCR fallback | general GUI agent |
| `OmniParser V2` | SoM-style parser, interactable boxes, icon captions | standalone workflow planner |
| `pywinauto` / `uiautomation` | object control on accessible widgets | custom-rendered UI parser |
| `pynput` | final action execution | state understanding |

## 3. Recommended Hybrid Patterns

### 3.1 Accessible-Control First

Use when the target is a normal editable field or a reliable dialog control.

Flow:

1. try object lookup
2. use vision only if object lookup fails or the control is ambiguous
3. still verify after action

### 3.2 Full-Screen Grounding -> Crop Retry -> OCR Refinement

Use when the control is visible but small, crowded, or text-dependent.

Flow:

1. `UI-Venus` or `UI-TARS` on the full screenshot
2. crop around the predicted area
3. rerun grounding on the crop, usually with `MAI-UI` or the same primary model
4. use OCR on the crop if text precision matters
5. merge the evidence and click only after verification

This is the most important pattern for dense engineering screens.

### 3.3 OmniParser / SoM Sidecar

Use when the UI contains many small interactable regions or icon-only controls.

Flow:

1. run OmniParser
2. turn parsed boxes into compact hints or a SoM overlay
3. let the primary VLM decide which marked element matters
4. convert the chosen region back to actual pixels for execution

Practical note:

- OmniParser is attractive because it adds boxes and interactability, not because it replaces every model in the stack.
- The YOLO detection component carries AGPL-3.0 implications and must be reviewed before production distribution.

### 3.4 Try-Catch Fallback Chain

A sensible escalation order is:

1. object API
2. structured parser or crop retry
3. full VLM reasoning
4. human review if confidence remains low

Do not reverse this order unless the screen is known to be fully inaccessible.

## 4. State Management

The repo should prefer industrial patterns over open-ended ReAct loops whenever the workflow is known.

### 4.1 Prefer State Machines For Known Workflows

RCS recipe work is closer to:

- `login_screen`
- `main_menu`
- `recipe_editor`
- `parameter_dialog`

than to an unrestricted desktop agent problem.

State-machine execution is cheaper, easier to verify, and easier to make safe.

### 4.2 Use Plan-Then-Execute For Stable Procedures

If the recipe flow is known, plan the stage order once and use VLMs mainly for:

- state recognition
- target grounding
- anomaly detection
- post-action verification

### 4.3 Keep History Small And Useful

Useful history:

- recent screenshots
- current state label
- last action
- unresolved ambiguity notes

Unhelpful history:

- long raw OCR dumps
- full prompt transcripts from many prior turns
- repeated copies of unchanged policy text

## 5. Reliability Patterns

### 5.1 Cache Layout, Not Just Coordinates

If a screen layout is stable, cache:

- screen hash
- target region
- target label
- last successful method

Revalidate after any significant layout change.

### 5.2 Separate Decision From Execution

The model should not directly "own" the mouse. It should produce a candidate action that can be checked against:

- current state
- allowed action types
- danger zones
- recent screen change

### 5.3 Human-In-The-Loop For Final Approval

The closer the action gets to recipe values, measurement execution, or probe movement, the stronger the manual confirmation requirement should be.

## 6. Practical Guidance For This Repo

- Use `poc/work2/flask_vlm.py` and `poc/work2/vlm_client.py` as the client contract.
- Use `poc/work2/connection_check.py` before debugging automation logic.
- Use `poc/work2/login_benchmark.py` for head-to-head service comparison.
- Treat OCR hints as auxiliary context, not as a replacement for pixel-based grounding.
- Keep debug artifacts first-class: source JPEG, sent WebP, raw response, overlay image, and call log.
