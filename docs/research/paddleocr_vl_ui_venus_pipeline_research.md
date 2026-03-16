# PaddleOCR-VL-1.5 + UI-Venus Pipeline Research (2026-03-17)

## Purpose

This note documents how `PaddleOCR-VL-1.5` and `UI-Venus` should be combined in this repo, especially around `poc/work2`, to extract information from images and screenshots more reliably.

The main questions are:

1. What is `PaddleOCR-VL-1.5` good at?
2. What is `UI-Venus` good at?
3. What is the best practical pipeline when both are available?

This research was written on `2026-03-17` and only uses primary sources: official repos, official model cards, official docs, and official technical reports.

## Summary

- For **extraction-heavy** tasks, `PaddleOCR-VL-1.5 -> UI-Venus` is the better default order.
- For **grounding-heavy** tasks, `UI-Venus -> PaddleOCR-VL-1.5` is the better default order.
- `PaddleOCR-VL-1.5` is stronger at exact text, layout, spotting, and structured visual-text parsing.
- `UI-Venus` is stronger at screenshot-based UI understanding, target grounding, and selecting the relevant control or region.
- The useful pattern is not "dump all OCR text into the VLM." The useful pattern is "build OCR anchors first, then let UI-Venus decide which anchors matter."
- PaddleOCR preprocessing should usually be conditional. It matters much more for camera photos, remote-screen photos, distortion, and rotated inputs than for clean native screenshots.
- If coordinates matter, `Spotting:` or coordinate-returning PaddleOCR flows are often better than using `OCR:` alone.

## What PaddleOCR-VL-1.5 Can Do

Official model-card material positions `PaddleOCR-VL-1.5` as an OCR VLM for document parsing, text spotting, and complex element understanding.

Its published strengths include:

- Support for `109` languages
- Support for text, tables, formulas, charts, seals, and spotting
- Irregular-shaped localization
- Robustness claims for skew, warping, screen photography, and illumination variation
- Both page-level parsing and element-level recognition

In practical terms, this is not just a basic text reader. It is closer to a structured parsing model that can read where content is, what kind of content it is, and how it is organized.

Important task-level implications:

1. `OCR:` is useful for broad text extraction.
2. `Spotting:` is more useful when text location and recognition both matter.
3. `Table Recognition:` matters for report-like, grid-like, and parameter-table screens.
4. The model is explicitly framed for more difficult capture conditions than perfectly clean screenshots.

## Additional Strengths From the Wider PaddleOCR Stack

Official PaddleOCR docs also matter here, not just the `PaddleOCR-VL-1.5` model card.

The broader PaddleOCR stack provides:

- OCR pipelines that return text content together with text position coordinates
- Support for single-character coordinates in newer PP-OCR flows
- Optional document orientation classification
- Optional text image unwarping
- Optional text-line orientation classification

That means the full design space is larger than "`UI-Venus` plus one OCR model call."

A practical split is:

- Fast coordinate-and-text pass: general PaddleOCR or PP-OCR
- Rich parsing and hard cases: `PaddleOCR-VL-1.5`

This repo currently exposes `paddleocr-vl-1.5` through the proxy path. A lighter local PaddleOCR pass could be added later if latency becomes important.

## Limits of PaddleOCR-VL-1.5

Its task framing also makes the limits fairly clear.

1. It is primarily an OCR, parsing, and spotting model.
2. Its official task examples are based on task keywords such as `OCR:`, `Spotting:`, `Table Recognition:`, `Formula Recognition:`, and `Chart Recognition:`.
3. The model card explicitly points users toward the official PaddleOCR method for faster and more complete page-level parsing.

Practical reading of that:

- `PaddleOCR-VL-1.5` is not the best primary planner for instruction-based GUI target selection.
- It is better used as an extraction engine or OCR sidecar that supplies text evidence and layout evidence.

That conclusion is an inference from the official task framing and examples, not a direct quoted claim.

## What UI-Venus Can Do

Official `UI-Venus` materials position it as a screenshot-driven GUI agent.

The published strengths are:

- GUI grounding from screenshots
- Navigation across mobile, desktop, and web interfaces from screenshots
- Strong benchmark positioning on tasks such as `ScreenSpot-Pro`, `ScreenSpot-v2`, `OSWorld-G`, and `AndroidWorld`
- Visual-only reasoning emphasis
- Public benchmark reporting for `+ZoomIn` variants, which supports crop-based retry patterns

For repo-level use, this means:

1. It is good at deciding which UI object matches an instruction.
2. It can reason over icons, tabs, panels, menus, buttons, and controls as UI elements, not just as text.
3. It matches the screenshot-centric design already used in `poc/work2`.

## Limits of UI-Venus

The official framing is about grounding and navigation, not dense OCR authority.

That leads to a practical limit:

- Small numeric values, long codes, dense parameter tables, multiline text blocks, and report-like panels are usually better handled by `PaddleOCR-VL-1.5` when exact text fidelity matters.
- `UI-Venus` is better at deciding what is important and where to act than at serving as the final source of truth for exact extracted strings.

This is an inference from the benchmark focus and task framing of the official materials.

## Recommended Role Split

The cleanest role split is:

| Role | Better fit | Why |
|------|------------|-----|
| Exact text extraction | `PaddleOCR-VL-1.5` | OCR, spotting, and parsing specialization |
| Text anchor generation with coordinates | `PaddleOCR-VL-1.5` or general PaddleOCR | Text polygons, boxes, spotting, coordinate output |
| Table, formula, chart, and structured region reading | `PaddleOCR-VL-1.5` | Directly aligned with published tasks |
| Instruction-based UI target selection | `UI-Venus` | Grounding and navigation specialization |
| Icon, tab, button, menu, and panel interpretation | `UI-Venus` | Screenshot-only GUI reasoning |
| Ambiguous crop resolution | `UI-Venus` plus OCR retry | Semantic reasoning plus text evidence |

## Best Pipeline for Extraction-Heavy Work

If the main goal is to extract information from an image well, this is the best default pattern.

### Recommended Flow

1. **Input triage**
   Decide whether the image is a clean screenshot or a photo/distorted capture.
2. **Conditional PaddleOCR preprocessing**
   Enable orientation correction, unwarping, or text-line orientation only when the input quality calls for it.
3. **OCR first pass**
   Use general PaddleOCR when fast coordinates are enough.
   Use `PaddleOCR-VL-1.5` when the content is dense, structured, multilingual, or visually difficult.
4. **OCR normalization**
   Convert raw OCR output into compact JSON with fields like `text`, `bbox` or `polygon`, `score`, `reading_order`, and `block_type`.
5. **UI-Venus semantic pass**
   Give `UI-Venus` the screenshot plus the compact OCR anchors and ask it to identify which regions or values are relevant.
6. **Crop-based escalation**
   For low-confidence or dense areas, rerun both models on zoomed crops.
7. **Merge**
   Prefer OCR for exact strings.
   Prefer `UI-Venus` for semantic selection and field-role interpretation.
8. **Verify**
   If the two disagree, keep the field unresolved or trigger a second pass.

### Why This Order Works

- Text extraction failures are often "the text was not read correctly," which OCR models address better.
- UI extraction failures are often "the text was read, but the wrong field was chosen," which grounding models address better.
- So for extraction-heavy work, the stable pattern is: OCR creates evidence, then `UI-Venus` resolves meaning.

## Best Pipeline for Grounding-Heavy Work

If the main goal is to find buttons, tabs, fields, or click targets, the order should be reversed.

### Recommended Flow

1. `UI-Venus` full-screen pass
2. OCR only for low-confidence or text-dependent cases
3. Crop around the likely label or target cluster
4. Combine OCR text with `UI-Venus` grounding before deciding the final target
5. Re-check the next screenshot after the action

### Typical Cases

- "Click the Save button"
- "Find the recipe name input"
- "Select the View or List tab"
- "Find the input box next to this small text label"

These are grounding-dominant problems, so `UI-Venus` should lead.

## `OCR:` vs `Spotting:`

The current `poc/work2/prompts/ocr_assist.py` uses `OCR:` only.

That is a reasonable default, but it is not always the best option.

Recommended branching:

- Read broad text content: `OCR:`
- Read text together with location: `Spotting:`
- Read table structure: `Table Recognition:`
- Read chart content: `Chart Recognition:`

So even without changing the wider architecture, adding task-keyword branching would improve the quality of the OCR stage.

## When to Enable Preprocessing

Official PaddleOCR preprocessing docs cover:

- Orientation classification
- Geometric distortion correction or unwarping

General OCR flows also support text-line orientation classification.

Practical rule set:

- Native screenshot: preprocessing usually `off`
- Monitor photo, phone capture, remote-screen photo, warped image: preprocessing usually `on`
- Frequent rotation issues: orientation `on`
- Perspective or bending distortion: unwarping `on`

Preprocessing should be a conditional stage, not an always-on default.

## Recommended Operating Modes for This Repo

### Option A: Minimal change to the current repo shape

1. Use `UI-Venus` as the default GUI-analysis model
2. Use `PaddleOCR-VL-1.5` as the text and structure sidecar
3. For extraction tasks, structure OCR results before sending them onward
4. Pass compact OCR hints to `UI-Venus`, not a raw OCR dump
5. Trigger crop retries only on low-confidence regions

This fits the current `poc/work2/flask_vlm.py` design well.

### Option B: Better extraction latency and precision

1. Add a fast general PaddleOCR pass
2. Escalate to `PaddleOCR-VL-1.5` only for dense, complex, multilingual, or ambiguous regions
3. Use `UI-Venus` as the final semantic resolver

This is probably the best long-term design if extraction quality becomes a main focus, but that lighter PaddleOCR path does not exist in the repo yet.

## What Not To Do

1. Do not force `PaddleOCR-VL-1.5` to be the primary click-grounding model.
2. Do not force `UI-Venus` to be the authority for exact extracted text.
3. Do not dump the full raw OCR text into the prompt without structure.
4. Do not rely on a single full-screen pass for hard cases; use crop-based retries.
5. Do not enable all document preprocessing by default for clean screenshots.

## Final Recommendation

For the current repo, the strongest practical recommendation is:

- If the task is **information extraction**, start with `PaddleOCR-VL-1.5`, then use `UI-Venus`.
- If the task is **UI target finding or action grounding**, start with `UI-Venus`, then use `PaddleOCR-VL-1.5`.
- Treat `PaddleOCR-VL-1.5` as the text and structure evidence engine.
- Treat `UI-Venus` as the UI meaning and grounding engine.
- Connect them with compact OCR JSON hints, not raw plain text.
- Use crop-based escalation for small text, dense panels, and low-confidence regions.

The best pattern is not "pick one main model and attach the other as a generic add-on." The best pattern is a dual pipeline where the order changes with the task.

## Immediate Follow-Up Items

1. Add task-keyword branching in `poc/work2/prompts/ocr_assist.py` instead of hard-coding `OCR:`
2. Add an OCR normalization helper that returns `text`, `coords`, `score`, and `block_type`
3. Standardize a compact OCR-hint format for `UI-Venus` prompts
4. Add low-confidence crop-retry rules
5. Add conditional preprocessing based on screenshot-vs-photo style input

## Sources

- PaddleOCR GitHub: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR OCR pipeline docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>
- PaddleOCR document preprocessing docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/doc_preprocessor.html>
- PaddleOCR text line orientation docs: <https://www.paddleocr.ai/main/en/version3.x/module_usage/textline_orientation_classification.html>
- PaddleOCR-VL-1.5 model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>
- PaddleOCR-VL-1.5 technical report: <https://huggingface.co/papers/2601.21957>
- vLLM PaddleOCR-VL recipe: <https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- UI-Venus GitHub: <https://github.com/inclusionAI/UI-Venus>
- UI-Venus-1.5-8B model card: <https://huggingface.co/inclusionAI/UI-Venus-1.5-8B>
- UI-Venus-1.5 technical report: <https://huggingface.co/papers/2602.09082>
