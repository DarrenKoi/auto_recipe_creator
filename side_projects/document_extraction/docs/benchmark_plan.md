# Benchmark 계획

## 목적

Hybrid model pipeline이 PowerPoint, PDF, Excel screenshot에서 실제로 extraction 품질을 개선하는지 측정합니다.

이 프로젝트의 목적은 RAG database이므로, benchmark는 extraction accuracy뿐 아니라 retrieval usefulness도 측정해야 합니다. 즉 chunk가 나중에 citation과 함께 질문에 답할 수 있을 만큼 context를 보존하는지 확인해야 합니다.

Benchmark는 다음 pipeline을 비교합니다.

- `paddleocr-vl-1.5` only
- `paddleocr-vl-1.5` plus `ui-venus`
- `mai-ui`를 사용한 hybrid crop refinement
- `kimi-k2.6` synthesis를 포함한 full pipeline

## Test Set

우선 9개 screenshot으로 시작합니다.

| ID | Type | Required content |
| --- | --- | --- |
| ppt_001 | PowerPoint | title과 큰 body text |
| ppt_002 | PowerPoint | dense technical slide |
| ppt_003 | PowerPoint | legend와 axis label이 있는 chart |
| pdf_001 | PDF | text-heavy page |
| pdf_002 | PDF | table-heavy page |
| pdf_003 | PDF | low-quality 또는 scanned-looking page |
| xls_001 | Excel | simple visible grid |
| xls_002 | Excel | wide dense table |
| xls_003 | Excel | chart 또는 filtered sheet view |

Optional later cases:

- Korean과 English mixed content
- Cell에 보이는 formula
- Merged cell
- Multi-column PDF layout
- 다른 screenshot이 포함된 slide screenshot

## Ground Truth

각 screenshot에 대해 작은 manual answer file을 만듭니다.

- Visible title
- Important body text
- Table header와 selected row
- Chart title, axis label, legend label, visible numeric label
- Expected summary
- 의도적으로 unreadable한 region

Ground truth에는 screenshot에 보이는 정보만 포함합니다.

## Metrics

### Text Recall

중요한 visible text가 extraction에 나타나는지 측정합니다.

Score:

- 1.0: 중요한 text를 모두 복원
- 0.7: 대부분의 중요한 text를 복원, minor omission 있음
- 0.4: 일부 유용한 text는 복원했지만 omission이 많음
- 0.0: 대부분 실패

### Table Accuracy

다음을 측정합니다.

- Header correctness
- Cell text correctness
- Row/column alignment
- Missing 또는 extra column

각 table을 0.0부터 1.0까지 scoring합니다.

### Chart Understanding

다음을 측정합니다.

- Chart title
- Axis label
- Legend label
- Visible numeric label
- Correct trend 또는 comparison summary

보이지 않는 value를 놓친 것은 penalize하지 않습니다.

### Layout Accuracy

Extracted region이 document structure와 맞는지 측정합니다.

- title vs body
- table vs paragraph
- chart vs image
- footer/page number

첫 benchmark에서는 approximate box를 허용합니다. Pixel-perfect box는 필요하지 않습니다.

### Hallucination Rate

Visible screenshot evidence로 뒷받침되지 않는 claim을 셉니다.

Examples:

- 만들어낸 number
- 만들어낸 row name
- Label 또는 bar로 뒷받침되지 않는 chart trend
- Hidden document context를 포함한 summary

낮을수록 좋습니다.

### Latency

Model call별로 기록합니다.

- service slug
- image type: full screenshot 또는 crop
- latency ms
- token usage가 있으면 포함
- success 또는 error

`kimi-k2.6`는 느릴 것으로 예상되므로 latency 기록이 중요합니다.

### RAG Readiness

Extracted output이 retrieval에 적합한지 측정합니다.

- Content chunk에 source image, screenshot order, region type, bbox가 있습니다.
- Table과 chart가 surrounding label과 heading을 유지합니다.
- Chunk text가 embedding search에 충분히 구체적입니다.
- Low-confidence evidence가 명확히 labeling되어 있습니다.
- Retrieved chunk만으로도 original model이 missing context를 추측하지 않고 answer를 지원할 수 있습니다.

각 screenshot을 0.0부터 1.0까지 scoring합니다.

## Comparison Matrix

각 screenshot마다 다음을 기록합니다.

| Pipeline | Text | Table | Chart | Layout | RAG readiness | Hallucination | Latency | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OCR only |  |  |  |  |  |  |  |  |
| OCR + UI-Venus |  |  |  |  |  |  |  |  |
| OCR + UI-Venus + crops |  |  |  |  |  |  |  |  |
| Full with Kimi-K2.6 |  |  |  |  |  |  |  |  |

## Acceptance Criteria

Side project를 더 구현할 가치가 있으려면 다음을 만족해야 합니다.

- 대부분의 screenshot에서 PowerPoint와 PDF text recall이 0.7 이상입니다.
- Excel simple-grid extraction의 table accuracy가 0.7 이상입니다.
- Full pipeline이 direct large-VLM reading보다 hallucination을 줄입니다.
- Crop refinement가 dense-table 또는 small-label case를 개선합니다.
- Kimi-K2.6가 complex page에서 latency를 감수할 만큼 summary quality를 개선합니다.
- 대부분의 screenshot에서 RAG readiness가 0.7 이상입니다.
- Retrieved chunk가 screenshot과 region을 cite할 수 있을 만큼 source metadata를 포함합니다.

## Failure Categories

Benchmark note에는 다음 label을 사용합니다.

- `text_too_small`
- `image_blurry`
- `reading_order_wrong`
- `table_structure_wrong`
- `chart_labels_missing`
- `hallucinated_value`
- `model_timeout`
- `unparseable_screenshot`
- `context_missing_for_rag`
- `chunk_too_broad`
- `chunk_too_fragmented`

## Benchmark 이후 다음 단계

결과가 유망하면 minimal extraction script를 구현합니다.

1. Input folder에서 screenshot을 읽습니다.
2. Tiered model pipeline을 실행합니다.
3. Screenshot별 JSON과 Markdown file을 씁니다.
4. `rag_db_plan.md` 기준의 RAG chunk record를 씁니다.
5. Session-level summary를 씁니다.
6. Model latency와 failure를 log합니다.
