# Benchmark Plan

## Purpose

Measure whether the hybrid model pipeline actually improves extraction from screenshots of PowerPoint, PDF, and Excel content.

The benchmark should compare:

- `paddleocr-vl-1.5` only
- `paddleocr-vl-1.5` plus `ui-venus`
- hybrid crop refinement with `mai-ui`
- full pipeline with `kimi-k2.5` synthesis

## Test Set

Start with 9 screenshots:

| ID | Type | Required content |
| --- | --- | --- |
| ppt_001 | PowerPoint | title and large body text |
| ppt_002 | PowerPoint | dense technical slide |
| ppt_003 | PowerPoint | chart with legend and axis labels |
| pdf_001 | PDF | text-heavy page |
| pdf_002 | PDF | table-heavy page |
| pdf_003 | PDF | low-quality or scanned-looking page |
| xls_001 | Excel | simple visible grid |
| xls_002 | Excel | wide dense table |
| xls_003 | Excel | chart or filtered sheet view |

Optional later cases:

- Korean and English mixed content
- formulas visible in cells
- merged cells
- multi-column PDF layout
- slide screenshot containing another screenshot

## Ground Truth

For each screenshot, create a small manual answer file with:

- Visible title
- Important body text
- Table headers and selected rows
- Chart title, axis labels, legend labels, and visible numeric labels
- Expected summary
- Regions that are intentionally unreadable

Ground truth should only include information visible in the screenshot.

## Metrics

### Text Recall

Measure whether important visible text appears in the extraction.

Score:

- 1.0: all important text recovered
- 0.7: most important text recovered, minor omissions
- 0.4: some useful text recovered, many omissions
- 0.0: mostly failed

### Table Accuracy

Measure:

- Header correctness
- Cell text correctness
- Row/column alignment
- Missing or extra columns

Score each table from 0.0 to 1.0.

### Chart Understanding

Measure:

- Chart title
- Axis labels
- Legend labels
- Visible numeric labels
- Correct trend or comparison summary

Do not penalize the model for missing values that are not visible.

### Layout Accuracy

Measure whether extracted regions match the document structure:

- title vs body
- table vs paragraph
- chart vs image
- footer/page number

Approximate boxes are acceptable for planning. Exact pixel-perfect boxes are not required in the first benchmark.

### Hallucination Rate

Count claims that are not supported by visible screenshot evidence.

Examples:

- Invented numbers
- Invented row names
- Chart trend not supported by labels or bars
- Summary includes hidden document context

Lower is better.

### Latency

Record per model call:

- service slug
- image type: full screenshot or crop
- latency ms
- token usage if available
- success or error

This matters because `kimi-k2.5` is expected to be slow.

## Comparison Matrix

For each screenshot, record:

| Pipeline | Text | Table | Chart | Layout | Hallucination | Latency | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OCR only |  |  |  |  |  |  |  |
| OCR + UI-Venus |  |  |  |  |  |  |  |
| OCR + UI-Venus + crops |  |  |  |  |  |  |  |
| Full with Kimi-K2.5 |  |  |  |  |  |  |  |

## Acceptance Criteria

The side project is worth implementing further if:

- PowerPoint and PDF text recall reaches at least 0.7 on most screenshots.
- Excel simple-grid extraction reaches at least 0.7 table accuracy.
- The full pipeline reduces hallucinations compared with direct large-VLM reading.
- Crop refinement improves dense-table or small-label cases.
- Kimi-K2.5 improves summary quality enough to justify its latency on complex pages.

## Failure Categories

Use these labels in benchmark notes:

- `text_too_small`
- `image_blurry`
- `reading_order_wrong`
- `table_structure_wrong`
- `chart_labels_missing`
- `hallucinated_value`
- `model_timeout`
- `unparseable_screenshot`

## Next Step After Benchmark

If results are promising, implement a minimal extraction script that:

1. Reads screenshots from an input folder.
2. Runs the tiered model pipeline.
3. Writes one JSON and one Markdown file per screenshot.
4. Writes a session-level summary.
5. Logs model latency and failures.
