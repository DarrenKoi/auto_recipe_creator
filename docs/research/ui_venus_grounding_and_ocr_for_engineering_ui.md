# UI-Venus Grounding + OCR for Complex Engineering Screenshots

Date: 2026-03-18

## Purpose

This note focuses on one practical question for this repository:

- How to use `UI-Venus-1.5` effectively for screenshot grounding
- How to make it work better on dense, high-resolution, engineering-style desktop UIs
- How to combine it with OCR models so final click coordinates are more precise

This note is narrower than the existing pipeline memo in [`paddleocr_vl_ui_venus_pipeline_research.md`](./paddleocr_vl_ui_venus_pipeline_research.md). The emphasis here is prompt shape, retry strategy, crop strategy, and OCR-assisted coordinate refinement.

The guidance below is based on official model cards, official repositories, official benchmark dataset cards, and official PaddleOCR documentation checked on 2026-03-18.

## Executive Summary

- Treat `UI-Venus` as the primary **selector / grounder**, not as the final authority for exact text.
- For complex desktop screenshots, the default pattern should be:
  1. full-screen `UI-Venus` pass
  2. crop / zoom retry around the predicted area
  3. OCR-assisted refinement on the crop
  4. post-action verification screenshot
- Use the official `UI-Venus` grounding format for single-element localization. In practice, one element per call is safer than asking for many targets at once.
- When location matters, prefer OCR modes that return coordinates or spotting output. For `PaddleOCR-VL-1.5`, `Spotting:` is more relevant than `OCR:` when the goal is precise grounding.
- For native screenshots, keep OCR preprocessing mostly off. Enable orientation or unwarping only for rotated, photographed, warped, or degraded inputs.
- In engineering UIs, the hard cases are usually not "what text is present?" but "which of several tiny, similar controls is the actionable one?" That is exactly where `UI-Venus` should stay in the loop even if OCR is available.

## Why Engineering UIs Are Hard

`ScreenSpot-Pro` is explicitly designed for professional, high-resolution GUI grounding. Its dataset card highlights the same failure modes we care about here:

- 1,581 high-resolution screenshots
- 23 professional applications
- categories including Development, Creative, CAD and Engineering, Scientific and Analytical, Office, and OS-common UIs
- the most common resolution is `2560x1440`
- target elements occupy only `0.07%` of screenshot area on average

That is close to the operating regime of many engineering desktops: small controls, dense toolbars, property grids, repeated widgets, multi-pane layouts, and mixed icon-text affordances.

`UI-Venus-1.5` is strong on those benchmarks, but the public Hugging Face evaluation page for the current `8B` family also shows that difficulty varies sharply by application:

- `Autocad Windows`: `52.9`
- `Blender Windows`: `60.6`
- `Inventor Windows`: `68.6`
- `Eviews Windows`: `92.0`

Interpretation:

- `UI-Venus` is already a strong base model for screenshot-only grounding.
- But engineering and CAD-style interfaces remain materially harder than ordinary office-style screens.
- So "one full-screen call and trust the first point" is not a strong enough policy for this repo.

## What UI-Venus Is Best At

Based on the official `UI-Venus-1.5` model card and repo:

- screenshot-only GUI grounding
- cross-platform UI understanding
- locating text, icon, button, tab, menu, and panel targets from natural-language instructions
- reasoning over professional UI layouts
- refusal-aware grounding settings such as `VenusBench-GD`

This makes `UI-Venus` a good fit for:

- deciding which visible control the instruction refers to
- disambiguating between repeated controls
- resolving icon-only elements that OCR cannot read
- choosing the correct clickable surface instead of just reading nearby text

## Where UI-Venus Alone Is Not Enough

The same sources imply the limits:

- very small numeric text
- dense parameter tables
- exact string fidelity
- tiny labels inside high-resolution screenshots
- fields whose meaning depends on nearby text more than on the control shape itself

Practical conclusion:

- `UI-Venus` should choose **what** to click
- OCR should help verify **which text / which row / which value** is involved
- the final click point should sometimes be a merge of both

This role split is partly an inference from the official task framing and benchmark focus, not a direct quote from one source.

## Best Prompting Pattern for UI-Venus Grounding

### 1. Use the official single-target grounding format

The current repo prompt in [`poc/work2/prompts/prompt_login_rcs_ui_venus.py`](../../poc/work2/prompts/prompt_login_rcs_ui_venus.py) already follows the official style:

```text
Output the center point of the position corresponding to the following instruction: {instruction}. The output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1].
```

Recommended rule:

- one screenshot
- one target
- one point
- refusal allowed with `[-1,-1]`

### 2. Prefer precise natural-language anchors over generic names

Weak:

```text
the input box
```

Better:

```text
the editable text field to the right of the visible label 'User ID' in the login dialog
```

Better for engineering UIs:

```text
the numeric input field in the 'Exposure' row inside the parameter grid on the right panel
```

Useful anchor types:

- visible text label
- row or column role
- tab or group box name
- left/right/above/below relation
- icon semantics
- state words such as selected, checked, disabled, active
- panel name or dialog name

### 3. Describe the clickable surface, not just the concept

If the user must click the field, ask for the field.
If the user must click the button, ask for the button.
If the user only needs the label text, ask for the label text.

Bad:

```text
the password area
```

Better:

```text
the editable password field in the third form row
```

### 4. Avoid overloaded prompts

Avoid mixing these in one grounding call:

- multiple targets
- planning steps
- OCR requests
- justification requests
- long JSON schemas

For grounding, shorter is usually better. The official format suggests that the model is tuned for direct element-point mapping, not for long chain-of-thought style instructions.

### 5. Use explicit infeasibility

Always keep the `[-1,-1]` refusal path. In complex screens, "not visible" is a valid and useful outcome.

That is especially important for:

- hidden tabs
- collapsed accordions
- controls outside the current scroll viewport
- dialogs that are expected but not actually open

## Prompt Patterns That Work Better on Engineering Screens

### Repeated toolbar icons

```text
the magnifier toolbar button in the top horizontal toolbar, immediately to the left of the hand-pan icon
```

### Parameter grid row

```text
the editable numeric field in the row labeled 'Threshold' inside the parameter table on the lower-right panel
```

### Tab among many similar tabs

```text
the tab labeled 'Recipe' in the main tab strip near the top of the application window
```

### Small dialog button in a crowded modal

```text
the 'Apply' button at the bottom-right of the visible settings dialog, not the main window behind it
```

### Icon-only target

```text
the gear-shaped settings button in the title area of the right-side panel
```

## The Best Retry Pattern: Full Screen -> Zoom -> OCR Refinement

For dense desktop UIs, the best practical pattern is not a single pass.

### Pass 1. Full-screen UI-Venus

Use `UI-Venus` on the raw full screenshot first.

Purpose:

- find the correct region
- decide the likely control family
- avoid OCR-first confusion when multiple similar labels exist

### Pass 2. Crop / zoom around the prediction

If the target is small, crowded, repeated, or near a dense grid, crop around the first predicted point and ask again.

Why this matters:

- `ScreenSpot-Pro` targets are extremely small on average
- `RegionFocus` reports `28%+` performance gains on `ScreenSpot-Pro` by dynamically zooming relevant regions for strong GUI agents

A practical crop policy:

- create a square crop centered on the first prediction
- try multiple crop scales, for example `20%`, `35%`, `50%` of the short image side
- remap the refined crop coordinate back to full-image pixels

This is an inference from the benchmark and `RegionFocus` results, but it aligns well with the current repo structure and the public `+ZoomIn` style evaluation patterns used in GUI grounding research.

### Pass 3. OCR-assisted coordinate refinement

Run OCR on the crop, not only on the full screen, when:

- the target is text-bearing
- the target is adjacent to a text label
- the target lives in a table or parameter grid
- the first grounding point is near several similar elements

## Which OCR Model to Use for Which Job

### `PaddleOCR-VL-1.5`

Best when you need:

- OCR plus layout understanding
- spotting under difficult conditions
- table-heavy or structured engineering screens
- multilingual or irregular-shaped text

Important official usage detail:

- the model supports task keywords such as `OCR:`, `Spotting:`, `Table Recognition:`, `Chart Recognition:`, and `Seal Recognition:`
- the official model card explicitly recommends image upscaling for `Spotting:` on smaller inputs and uses a larger pixel budget for spotting than for generic OCR

Practical rule:

- use `OCR:` when you only need text content
- use `Spotting:` when you need text plus location
- use `Table Recognition:` when the screen is essentially a grid or report

### General PaddleOCR / PP-OCR pipeline

Best when you need:

- fast text boxes
- word boxes
- character boxes
- deterministic coordinate output

Official OCR docs show:

- `rec_boxes` for rectangle coordinates
- `return_word_box=True` for word or character-level boxes
- optional document orientation, text-line orientation, and unwarping modules

For precise click grounding, those coordinate-bearing outputs are often more useful than plain text alone.

### `GOT-OCR-2.0-hf`

Best as a sidecar or fallback when you need:

- crop-based OCR reread
- interactive OCR on a specific region
- recognition inside a user-specified box

Its official model card explicitly supports interactive OCR on a specific region using a box.

That makes it a good fallback for "I already know the approximate area, now reread exactly this patch."

## OCR + UI-Venus Fusion Patterns That Actually Help

### Pattern A. Text button or tab

Use when the target itself is text-bearing.

Steps:

1. `UI-Venus` picks the semantic target region.
2. OCR on the crop finds the matching text box.
3. Final click point is the OCR text box center, or a small expansion around it.

Examples:

- text buttons
- menu items
- tab labels
- list entries

### Pattern B. Input field next to a label

Use when OCR can read the label, but the clickable surface is not the label.

Steps:

1. OCR finds the label box, for example `Threshold`, `User ID`, `Server`.
2. `UI-Venus` identifies the interactive field associated with that label.
3. Final click point is the `UI-Venus` point, but constrained by the OCR row anchor.

Good rule:

- OCR determines the correct row
- `UI-Venus` determines the correct control surface

This is usually better than clicking the OCR label center directly.

### Pattern C. Dense parameter grid

Use when multiple similar values appear in the same pane.

Steps:

1. OCR or `PaddleOCR-VL Spotting` extracts text anchors for row labels and current values.
2. Use those anchors to isolate the target row or cell neighborhood.
3. Run `UI-Venus` again only on that localized crop.
4. Click the refined point.

This is the most important pattern for engineering UI automation.

### Pattern D. Icon-only target with nearby text

Use when the icon is not readable by OCR, but nearby text can reduce ambiguity.

Steps:

1. OCR identifies nearby labels or group box names.
2. The prompt to `UI-Venus` includes those anchors.
3. `UI-Venus` returns the icon point.

Example:

```text
the small dropdown icon in the 'Acquisition Settings' group, to the right of the label 'Mode'
```

### Pattern E. OCR as verifier, not selector

Do not dump a large raw OCR transcript into the `UI-Venus` prompt by default.

Better:

- keep `UI-Venus` prompt screenshot-centric
- run OCR as a sidecar
- merge results in code

Reason:

- `UI-Venus` is trained and benchmarked as a screenshot-grounding model
- huge OCR text dumps can add noise
- the most useful OCR contribution is usually compact structure: matched label, box coordinates, row membership, confidence

This is an inference from the official task framing and from practical GUI-agent design, not a direct model-card statement.

## A Good Merge Contract for Precise Grounding

When this repo merges model outputs, the merged result should keep evidence, not just the final point.

Recommended shape:

```json
{
  "instruction": "the editable numeric field in the row labeled 'Threshold'",
  "image_width": 2560,
  "image_height": 1440,
  "ui_venus_fullscreen_point": {"x": 1784, "y": 622},
  "crop_box": {"x1": 1520, "y1": 420, "x2": 2048, "y2": 920},
  "ui_venus_crop_point": {"x": 1812, "y": 618},
  "ocr_anchor_text": "Threshold",
  "ocr_anchor_box": {"x1": 1603, "y1": 602, "x2": 1702, "y2": 628},
  "final_click_point": {"x": 1812, "y": 618},
  "final_strategy": "ui_venus_crop_constrained_by_ocr_row",
  "verification_required": true
}
```

This makes debugging much easier than saving only one final point.

## Confidence Without Native Confidence Scores

`UI-Venus` grounding output is point-only, so confidence must be estimated indirectly.

Recommended proxy signals:

- full-screen point and crop point agree closely
- OCR finds the expected nearby label
- the point lands inside the expected panel or dialog
- repeated `UI-Venus` calls on slightly different crops are consistent
- post-click screenshot shows the expected state change

Bad signs:

- first and second pass land on different control families
- OCR cannot find any related anchor text
- the point lands on empty whitespace
- the predicted point is near several repeated controls with no distinguishing evidence

## When to Turn OCR Preprocessing On

Official PaddleOCR docs support:

- document orientation classification
- text-line orientation classification
- text image unwarping / rectification

Recommended rule:

- native application screenshot: usually keep them `off`
- photographed screen, remote-camera capture, skewed or warped image: turn them `on`

For this repo's normal Windows screenshot capture flow, native screenshots are the baseline, so heavy OCR preprocessing should be exception logic, not the default.

## Recommended Repo-Level Changes for `poc/work2`

### 1. Keep `ui-venus` as the primary grounding model

This already matches the shared defaults in [`poc/work2/flask_vlm.py`](../../poc/work2/flask_vlm.py).

### 2. Change OCR prompt selection from always-`OCR:` to task-based branching

Current prompt builder:

- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py)

Recommended branching:

- `OCR:` for plain text extraction
- `Spotting:` for text-plus-location grounding support
- `Table Recognition:` for dense parameter tables or report-style panes

### 3. Add crop retry after the first `UI-Venus` point

This is likely the highest-value improvement for complex engineering screens.

### 4. Save full evidence for each pass

Keep:

- full screenshot JPEG
- VLM input WebP
- crop JPEG and WebP
- OCR raw JSON
- merged grounding JSON
- overlay image

That fits the current repo debugging style well.

### 5. Verify after acting

For engineering automation, never trust the click alone.

After click or type:

- capture again
- confirm focus / state / value change
- only then continue

## Recommended Default Pipeline

For this repo, the best default grounding pipeline is:

1. Capture native screenshot.
2. Run `UI-Venus` with the official single-target prompt.
3. If the target is small, repeated, text-adjacent, or in a grid, generate a zoom crop around the first point.
4. Run `UI-Venus` again on the crop.
5. Run OCR on that crop:
   - `Spotting:` if location matters
   - `OCR:` if exact text readback is the main need
   - `Table Recognition:` for table-like panes
6. Merge:
   - use OCR to identify the correct row / label / text box
   - use `UI-Venus` to choose the clickable control surface
7. Click.
8. Capture again and verify the resulting state.

## Bottom Line

For complex engineering screenshots, the most effective way to use `UI-Venus` is not to ask it to do everything alone.

The best practical split is:

- `UI-Venus` for semantic grounding and clickable-surface selection
- OCR for text evidence, row anchoring, and precise text localization
- crop / zoom retry as the bridge between them

If this repo wants materially better grounding on dense RCS-style screens, the first improvement to implement is:

- `UI-Venus` single-target full-screen grounding
- followed by crop retry
- followed by `PaddleOCR-VL Spotting` or PP-OCR coordinate refinement on the crop

That is the highest-signal path supported by the current public sources.

## Sources

Primary sources used for this note:

1. UI-Venus-1.5 model card:
   https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
2. UI-Venus official repository:
   https://github.com/inclusionAI/UI-Venus
3. UI-Venus-1.5 official project page:
   https://ui-venus.github.io/UI-Venus-1.5/
4. ScreenSpot-Pro dataset card:
   https://huggingface.co/datasets/Voxel51/ScreenSpot-Pro
5. PaddleOCR-VL-1.5 model card:
   https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
6. PaddleOCR-VL official docs:
   https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
7. PaddleOCR OCR pipeline docs:
   https://www.paddleocr.ai/v3.0.1/en/version3.x/pipeline_usage/OCR.html
8. PaddleOCR document preprocessing docs:
   https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/doc_preprocessor.html
9. PaddleOCR text image unwarping docs:
   https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_image_unwarping.html
10. PaddleOCR text-line orientation docs:
    https://paddlepaddle.github.io/PaddleOCR/v3.0.1/en/version3.x/module_usage/textline_orientation_classification.html
11. GOT-OCR-2.0-hf model card:
    https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
12. RegionFocus paper / official repo:
    https://openaccess.thecvf.com/content/ICCV2025/html/Luo_Visual_Test-time_Scaling_for_GUI_Agent_Grounding_ICCV_2025_paper.html
    https://github.com/tiangeluo/RegionFocus
