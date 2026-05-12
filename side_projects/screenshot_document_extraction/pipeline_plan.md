# Pipeline Plan

## Goal

Build a screenshot-only extraction workflow that recovers as much visible information as possible from PowerPoint, PDF, and Excel screenshots.

The workflow should produce both machine-readable JSON and human-readable Markdown. Its main downstream purpose is a RAG database, so each extracted item must keep enough context, provenance, and confidence to be retrieved safely later. It should reuse the existing `poc/work2` VLM service definitions and client patterns.

## Inputs

Minimum input per item:

- Screenshot image path
- Optional source type hint: `powerpoint`, `pdf`, `excel`, or `unknown`
- Optional user goal: for example, `summarize`, `extract_table`, `extract_chart`, or `full_extract`
- Optional document/session metadata: title, owner, source system, capture date, topic, and confidentiality label

Images should be kept as local JPEG artifacts and sent to VLM APIs as WebP payloads when possible, matching the current repo convention.

## Output Contract

The final output should use this conceptual schema:

```json
{
  "source_image": "path/to/screenshot.jpg",
  "source_type": "powerpoint|pdf|excel|unknown",
  "document_id": "",
  "screenshot_index": 1,
  "overall_confidence": 0.0,
  "summary_markdown": "",
  "regions": [
    {
      "region_id": "r001",
      "type": "title|body|table|chart|formula|footer|legend|other",
      "bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},
      "text": "",
      "surrounding_context": "",
      "confidence": 0.0,
      "model_sources": ["paddleocr-vl-1.5", "ui-venus"]
    }
  ],
  "tables": [],
  "charts": [],
  "formulas": [],
  "rag_chunks": [],
  "unresolved": []
}
```

This is a planning contract, not a committed Python type yet. Implementation can start with dictionaries and later move to dataclasses if the schema stabilizes.

## Stage 1: Preprocess Screenshot

Actions:

- Load image with Pillow.
- Normalize orientation if needed.
- Save a local JPEG copy for debug review.
- Convert to WebP quality 90 for VLM payload.
- Record width, height, file size, and source type hint.

Notes:

- Do not upscale blindly. Use crop-specific zoom only after region detection.
- Preserve the original screenshot for audit and comparison.

## Stage 2: First-Pass OCR And Document Parsing

Model:

- `paddleocr-vl-1.5`

Prompt style:

- Use the model's OCR/document task style first.
- Ask for visible text, reading order, tables, charts, formulas, and region hints.

Expected result:

- Raw visible text
- Candidate table blocks
- Candidate chart text
- Candidate formula text
- Reading-order notes

Failure handling:

- If output is too terse, retry with a narrower prompt such as table-only or chart-only.
- If dense text is missed, wait for Stage 4 crop retry rather than repeatedly sending the same full screenshot.

## Stage 3: Layout And Visual Region Detection

Model:

- `ui-venus`

Prompt style:

- Ask for visible document/screen regions and bounding boxes.
- Ask the model to identify the screenshot type and visual hierarchy.
- Require JSON-like output where possible.

Expected result:

- Screenshot type classification
- Region list with approximate bounding boxes
- Visual roles such as title, body, chart, legend, table, toolbar, footer
- Crop candidates for small or dense regions

Failure handling:

- If bounding boxes are coarse, use them only as crop candidates.
- Do not trust exact text from this pass when OCR evidence disagrees.

## Stage 4: Crop Refinement

Models:

- `mai-ui` for visual grounding and small-region interpretation
- `paddleocr-vl-1.5` for OCR-heavy crops

Crop targets:

- Dense Excel ranges
- Small chart legends
- Axis labels
- Footnotes
- Slide body text under 16 px visual height
- PDF tables and formulas

Rules:

- Expand crop boxes with small margins before sending.
- Store crop metadata with parent screenshot coordinates.
- Prefer crop retry over full-image retry for low-confidence text.

Expected result:

- Better text recall for small regions
- Refined table headers and cells
- Refined chart labels
- Correction candidates for OCR mistakes

## Stage 5: Evidence Merge

Actions:

- Normalize region coordinates.
- Deduplicate repeated OCR text.
- Attach OCR text to layout regions.
- Merge table candidates by overlap.
- Mark conflicts instead of silently choosing when model outputs disagree.

Conflict examples:

- OCR sees a number but the chart summary uses a different number.
- UI-Venus labels a region as a table but OCR returns paragraph text.
- Two crops return different table headers.

Resolution:

- Prefer exact OCR for text.
- Prefer layout models for region type and bounding boxes.
- Prefer Kimi-K2.5 for semantic conflict resolution when exact evidence is available.

## Stage 6: Large VLM Synthesis

Model:

- `kimi-k2.5`

Use only when:

- The user asks for a high-quality final summary.
- The screenshot has charts or complex tables.
- Confidence is low after OCR and crop refinement.
- Multiple screenshots need a merged narrative.

Input to Kimi-K2.5:

- Original screenshot if latency budget allows
- OCR text
- Region list
- Crop-level extracted text
- Table/chart candidates
- Explicit unresolved conflicts

Expected result:

- Final Markdown summary
- Final JSON cleanup
- Human-review checklist
- Confidence explanation

Important rule:

- Do not ask Kimi-K2.5 to invent missing data. The prompt must say that unavailable or unreadable fields should be marked as unknown.

## Stage 7: Human Review Loop

Create a review packet for each screenshot:

- Original screenshot path
- Marked crop paths if available
- Extracted Markdown
- Extracted JSON
- Unresolved fields
- Low-confidence regions

The review loop should be lightweight. A human should be able to correct the final JSON or mark the screenshot as not extractable.

## Stage 8: RAG Chunk Generation

Goal:

- Convert extraction evidence into records that can be embedded, searched, filtered, and cited.

Chunk types:

- `region_text`: title/body/footer text from one visual region
- `table_summary`: table title, headers, important rows, and notes
- `table_row`: row-level chunk when each row has independent meaning
- `chart_summary`: chart title, axes, legends, visible values, and inferred trend
- `formula`: formula text plus nearby label/context
- `document_summary`: screenshot-level or session-level summary

Each chunk should include:

- `chunk_id`
- `document_id`
- `source_image`
- `screenshot_index`
- `source_type`
- `region_id`
- `region_type`
- `bbox`
- `content`
- `context_before`
- `context_after`
- `parent_heading`
- `model_sources`
- `confidence`
- `review_status`

Rules:

- Preserve table and chart context rather than embedding isolated cell text.
- Keep bbox and source image paths so retrieved answers can cite visible evidence.
- Mark low-confidence chunks as searchable but not trusted for direct answer generation unless the user accepts uncertain evidence.
- Keep original OCR text and cleaned content separately when they differ.

## Implementation Shape For A Later Code Pass

Recommended package location:

- `side_projects/screenshot_document_extraction/`

Potential files:

- `extract_screenshot.py`
- `prompts.py`
- `models.py`
- `merge.py`
- `schemas.py`
- `rag_chunks.py`

Reuse:

- `poc.work2.vlm_client.Work2VLMClient`
- `poc.work2.flask_vlm` service registry
- `poc.work2.util.image_utils.encode_image_webp`

Do not add CLI-heavy argument parsing for the first version. Prefer `.env` and in-code defaults, matching the repo's current operational-script style.

The RAG database design details live in [rag_db_plan.md](./rag_db_plan.md).
