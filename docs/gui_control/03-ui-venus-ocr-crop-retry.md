# UI-Venus, OCR, And Crop-Retry

This document consolidates the repository's guidance on `UI-Venus`, OCR sidecars, and crop-retry grounding for dense engineering UIs.

## 1. Start Order Depends On The Task

### 1.1 Grounding-Heavy Tasks

Examples:

- find the login button
- pick a tab
- click the input field next to a label

Recommended order:

1. `UI-Venus` full-screen pass
2. crop retry if needed
3. OCR only when text disambiguation is required

### 1.2 Extraction-Heavy Tasks

Examples:

- read parameter values
- extract a table row
- verify an exact numeric field

Recommended order:

1. `PaddleOCR-VL-1.5` or OCR pipeline first
2. normalize text plus coordinates
3. use `UI-Venus` to decide semantic role or clickable surface

## 2. Prompting Rules For UI-Venus

Treat `UI-Venus` as a single-target grounder.

Recommended prompt shape:

- one screenshot
- one target
- one point
- explicit refusal path such as `[-1,-1]`

Anchor the request with:

- visible label text
- row/column relation
- panel or dialog name
- left/right/above/below relation
- state words such as active, selected, checked

Better:

- "the editable text field to the right of the visible label 'User ID'"
- "the numeric input field in the 'Exposure' row inside the right parameter panel"

Avoid:

- multiple targets in one grounding call
- long JSON schemas when only a point is needed
- mixing planning, OCR, and grounding in one prompt

## 3. OCR Mode Selection

| Need | Best mode |
|------|-----------|
| broad text extraction | `OCR:` |
| text plus location | `Spotting:` |
| grid/table structure | `Table Recognition:` |
| hard crop reread | `GOT-OCR-2.0-hf` |

Practical rule:

- use `Spotting:` when the final click depends on text coordinates
- use `OCR:` when only the content matters
- use `Table Recognition:` for report-like or parameter-grid screens

## 4. When Crop Retry Should Fire

Use crop retry if one or more of these are true:

- model confidence is below `0.6`
- target minimum side is below `40px`
- target area ratio is below `0.003`
- more than `3` similar neighbors exist within about `80px`
- the first pass lands in a dense toolbar or parameter grid

Practical crop policy:

- center the crop on the first predicted point
- start around `20%` to `30%` of the short image side
- retry one or two larger windows if context is insufficient
- remap crop coordinates back to original pixels before execution

Common failures to guard against:

- forgetting the crop offset during remap
- mixing `relative_1000` and pixel coordinates
- using the wrong crop width/height during back-conversion

## 5. Merge Rules

### 5.1 Text Buttons Or Tabs

- if OCR finds the exact target text with a strong box, prefer the OCR box center
- reject the merge if the crop-grounded point is far outside that box neighborhood

### 5.2 Input Field Next To A Label

- use OCR to find the correct row or label anchor
- use `UI-Venus` for the final clickable field point
- reject if the field point does not align with the anchored row

### 5.3 No Reliable OCR Anchor

- fall back to the crop-grounded `UI-Venus` point
- mark the strategy explicitly in the debug JSON

### 5.4 Conflicting Evidence

- do not click
- mark the result unresolved
- capture artifacts for review

## 6. Current Repo Anchors

The main files to align with this strategy are:

- `poc/work2/login_rcs_ui_venus.py`
- `poc/work2/login_rcs_ui_venus_rev2.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/pipeline_ocr.py`
- `poc/work2/prompts/prompt_login_rcs_ui_venus.py`
- `poc/work2/prompts/prompt_ocr_assist.py`
- `poc/work2/util/image_utils.py`

## 7. Recommended Implementation Sequence

1. keep the full-screen `UI-Venus` pass
2. add crop-region helpers and coordinate remap helpers
3. add OCR task branching in `prompt_ocr_assist.py`
4. store a compact OCR hint instead of raw OCR dumps
5. add merge rules for button targets and labeled inputs
6. save evidence artifacts before enabling real clicks

## 8. Debug Artifact Standard

For each target, keep:

- source screenshot as JPEG
- sent payload as WebP if applicable
- raw model outputs
- crop image
- overlay image with full-screen point, crop point, and final point
- final decision JSON including strategy name

This is the fastest way to improve grounding quality without guessing.
