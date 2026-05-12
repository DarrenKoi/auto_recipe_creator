# Research Notes

## Question

Can the current company-available models extract useful information from screenshots of PowerPoint, PDF, and Excel content when the underlying files cannot be parsed directly because they are DRM protected?

Working answer: yes, but only for visible information. The strongest design is a hybrid extraction pipeline where OCR/document parsing models provide evidence, GUI/layout models interpret visual structure, and a large VLM performs final synthesis on top of extracted evidence.

## Model Capability Notes

### PaddleOCR-VL-1.5

PaddleOCR-VL-1.5 is the best first-pass model for this side project because it is document-focused rather than GUI-action-focused.

Relevant capabilities from the official documentation and model card:

- OCR and page-level document parsing
- Table recognition
- Formula recognition
- Chart recognition
- Text spotting
- Seal recognition
- Robustness cases that include screen photography, illumination variation, skew, scanning, and warping

Sources:

- PaddleOCR-VL-1.5 docs: <https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html>
- PaddleOCR-VL-1.5 model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>

Recommended role:

- Extract raw text and reading order.
- Parse visible tables and formulas.
- Identify chart text, legends, axes, and visible data labels.
- Produce the first evidence layer before any high-level VLM interpretation.

### UI-Venus-1.5-8B

UI-Venus is a GUI agent model family built for GUI grounding, navigation, and visual understanding across real-world application screens. The model card documents OpenAI-compatible serving through vLLM and describes strong GUI grounding performance.

Source:

- UI-Venus-1.5-8B model card: <https://huggingface.co/inclusionAI/UI-Venus-1.5-8B>

Recommended role:

- Detect high-level regions in a full screenshot.
- Identify whether the screenshot is a slide, PDF page, spreadsheet, dialog, or mixed screen.
- Label visual blocks such as title, subtitle, body, table, chart, legend, toolbar, footnote, and page number.
- Provide crop boxes for refinement.

### MAI-UI-8B

MAI-UI is also a GUI-focused model family. The 8B model card documents OpenAI-compatible serving and grounding performance across GUI benchmarks.

Source:

- MAI-UI-8B model card: <https://huggingface.co/Tongyi-MAI/MAI-UI-8B>

Recommended role:

- Refine crops from dense layouts.
- Inspect small chart legends, Excel table headers, footers, and labels.
- Cross-check UI-Venus region proposals when a region is visually ambiguous.

### Kimi-K2.5

The repo already defines a direct company API model entry for `kimi-k2.5` in `poc/work2/flask_vlm.py`. Its response time is expected to be slower, so it should not run on every crop.

Recommended role:

- Merge OCR and visual evidence into a final structured answer.
- Resolve conflicts between OCR output and layout model interpretation.
- Summarize slide intent and chart meaning.
- Normalize extracted tables into consistent schemas.
- Produce final Markdown and JSON with confidence notes.

Use it only after cheaper passes have collected evidence, or when the extraction confidence is low.

## Why A Hybrid Pipeline Is Needed

No single model should own the whole task:

- OCR models are strongest at visible text and document elements, but may miss high-level business meaning.
- GUI grounding models are useful for region detection and visual hierarchy, but they are not the most reliable source of exact text.
- Large VLMs are valuable for synthesis, but slow and more likely to hallucinate if asked to read dense screenshots directly without OCR evidence.

The pipeline should therefore separate:

- Evidence extraction
- Region detection
- Crop refinement
- Final reasoning

## Data Types And Expected Difficulty

| Source type | Easy cases | Hard cases |
| --- | --- | --- |
| PowerPoint | large title/body text, simple charts, visible tables | dense slides, small footnotes, screenshots embedded inside slides |
| PDF | text-heavy pages, numbered sections, visible tables | scanned low-resolution pages, multi-column reading order, equations |
| Excel | visible grids, headers, simple formulas shown in cells | hidden rows/columns, wide sheets, tiny values, merged cells, filters |

## Safety Boundary

This project must stay on the screen-observation side:

- Allowed: user-captured screenshots of content the user can view.
- Allowed: OCR and VLM analysis of visible pixels.
- Not allowed: DRM removal, protected file parsing, hidden-content extraction, credential bypass, or automation intended to defeat access controls.

## Research Hypotheses

1. PaddleOCR-VL alone will recover most visible text, but will not always produce useful business-level summaries.
2. UI-Venus will improve region routing for slides, spreadsheets, and mixed UI screenshots.
3. MAI-UI will improve small-region extraction when used on crops, not full screenshots.
4. Kimi-K2.5 will improve final document summaries and conflict resolution, but should be gated by confidence or used as a final synthesis step because of latency.
5. The best measurable improvement will come from crop retry and evidence merging, not from switching one model for another.
