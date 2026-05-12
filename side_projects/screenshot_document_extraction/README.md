# Screenshot Document Extraction Side Project

## Purpose

This side project studies whether the current company-served VLM/OCR models can recover useful information from screenshots of DRM-protected documents.

The target inputs are screenshots of:

- PowerPoint slides
- PDF pages
- Excel sheets

The project is screenshot-only. It does not parse protected files directly, remove DRM, bypass access controls, or automate around document protection. The goal is to extract information that is already visible on screen.

## Available Model Roles

The current repo already has a usable model-service foundation in `poc/work2`:

- `poc/work2/flask_vlm.py` defines shared service slugs, model names, and endpoint mappings.
- `poc/work2/vlm_client.py` provides an OpenAI-compatible image client.
- `poc/work2/connection_check.py` verifies the Flask proxy and per-service `/v1/models` readiness.

For this side project, use the models as a tiered system:

| Model | Service slug | Primary role |
| --- | --- | --- |
| PaddleOCR-VL-1.5 | `paddleocr-vl-1.5` | OCR, reading order, tables, formulas, charts, document parsing |
| UI-Venus-1.5-8B | `ui-venus` | whole-screenshot layout and UI/document visual understanding |
| MAI-UI-8B | `mai-ui` | crop-level refinement for dense or small regions |
| Kimi-K2.5 | `kimi-k2.5` | slow high-quality synthesis, ambiguity resolution, and final reasoning |

## Expected Outputs

For each screenshot, the extraction pipeline should produce:

- Raw OCR text
- Detected regions such as title, body, table, chart, formula, footer, legend, and notes
- Structured tables when visible cell boundaries and text are recoverable
- Chart summaries based on visible labels, axes, legends, and trends
- A final Markdown summary for human reading
- A final JSON payload with confidence, source regions, and unresolved fields

For a group of screenshots, the pipeline should produce:

- A merged outline
- Key facts
- Table inventory
- Chart inventory
- Low-confidence review checklist

## Practical Limits

Screenshots do not contain the original Office/PDF object structure. The pipeline can only infer from visible pixels. Expected weak points are:

- Tiny text
- Blurry or compressed screenshots
- Hidden Excel rows or columns
- Truncated cells
- Overlapping labels
- Charts without visible numeric labels
- Speaker notes or comments not visible in the screenshot

The large VLM should reduce reasoning errors, but it cannot recover information that is not visible.

## Recommended First Experiment

1. Run `uv run python poc/work2/connection_check.py` to verify available services.
2. Collect a small local screenshot set:
   - 3 PowerPoint screenshots
   - 3 PDF page screenshots
   - 3 Excel screenshots
3. Run the extraction manually or with a minimal script:
   - `paddleocr-vl-1.5` first
   - `ui-venus` second
   - crop retry with `mai-ui` or `paddleocr-vl-1.5`
   - final merge with `kimi-k2.5` only when needed
4. Score the results using `benchmark_plan.md`.

## Related Documents

- [research_notes.md](./research_notes.md)
- [pipeline_plan.md](./pipeline_plan.md)
- [benchmark_plan.md](./benchmark_plan.md)
