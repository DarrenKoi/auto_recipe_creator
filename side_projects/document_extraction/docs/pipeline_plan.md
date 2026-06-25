# Pipeline 계획

## 목표

PowerPoint, PDF, Excel screenshot에서 화면에 보이는 정보를 최대한 복원하는 screenshot-only extraction workflow를 구축합니다.

Workflow는 machine-readable JSON과 human-readable Markdown을 모두 생성해야 합니다. 주요 downstream 목적은 RAG database이므로, 각 extracted item은 나중에 안전하게 retrieval될 수 있도록 context, provenance, confidence를 충분히 보존해야 합니다. 구현은 기존 `poc/work2` VLM service definition과 client pattern을 재사용합니다.

## 입력

Item별 최소 입력:

- Screenshot image path
- Optional source type hint: `powerpoint`, `pdf`, `excel`, 또는 `unknown`
- Optional user goal: 예를 들어 `summarize`, `extract_table`, `extract_chart`, `full_extract`
- Optional document/session metadata: title, owner, source system, capture date, topic, confidentiality label

Image는 local artifact로 JPEG를 보관하고, VLM API payload는 가능한 경우 WebP로 전송합니다. 이는 현재 repo convention과 맞춥니다.

## Output Contract

Final output은 다음 conceptual schema를 사용합니다.

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

이는 planning contract이며, 아직 고정된 Python type은 아닙니다. 구현은 dictionary로 시작하고 schema가 안정되면 dataclass로 옮길 수 있습니다.

## Stage 1: Screenshot Preprocess

작업:

- Pillow로 image를 load합니다.
- 필요하면 orientation을 normalize합니다.
- Debug review를 위해 local JPEG copy를 저장합니다.
- VLM payload용으로 WebP quality 90으로 변환합니다.
- Width, height, file size, source type hint를 기록합니다.

참고:

- 무조건 upscale하지 않습니다. Region detection 이후 crop-specific zoom만 사용합니다.
- Audit와 비교를 위해 original screenshot을 보존합니다.

## Stage 2: First-Pass OCR And Document Parsing

Model:

- `paddleocr-vl-1.5`

Prompt style:

- 먼저 model의 OCR/document task style을 사용합니다.
- Visible text, reading order, table, chart, formula, region hint를 요청합니다.

Expected result:

- Raw visible text
- Candidate table block
- Candidate chart text
- Candidate formula text
- Reading-order note

Failure handling:

- Output이 너무 짧으면 table-only 또는 chart-only처럼 더 좁은 prompt로 retry합니다.
- Dense text가 누락되면 같은 full screenshot을 반복 전송하지 말고 Stage 4 crop retry를 기다립니다.

## Stage 3: Layout And Visual Region Detection

Model:

- `ui-venus`

Prompt style:

- Visible document/screen region과 bounding box를 요청합니다.
- Screenshot type과 visual hierarchy를 식별하도록 요청합니다.
- 가능하면 JSON-like output을 요구합니다.

Expected result:

- Screenshot type classification
- Approximate bounding box를 포함한 region list
- title, body, chart, legend, table, toolbar, footer 같은 visual role
- 작거나 dense한 region을 위한 crop candidate

Failure handling:

- Bounding box가 coarse하면 crop candidate로만 사용합니다.
- OCR evidence와 충돌할 때 이 pass의 exact text를 신뢰하지 않습니다.

## Stage 4: Crop Refinement

Models:

- Visual grounding과 small-region interpretation에는 `mai-ui`
- OCR-heavy crop에는 `paddleocr-vl-1.5`

Crop targets:

- Dense Excel range
- 작은 chart legend
- Axis label
- Footnote
- Visual height가 16 px 미만인 slide body text
- PDF table과 formula

Rules:

- 전송 전에 crop box에 작은 margin을 더합니다.
- Parent screenshot coordinate와 함께 crop metadata를 저장합니다.
- Low-confidence text에는 full-image retry보다 crop retry를 우선합니다.

Expected result:

- Small region text recall 개선
- Table header와 cell refinement
- Chart label refinement
- OCR mistake correction candidate

## Stage 5: Evidence Merge

작업:

- Region coordinate를 normalize합니다.
- 반복 OCR text를 deduplicate합니다.
- OCR text를 layout region에 attach합니다.
- Overlap 기준으로 table candidate를 merge합니다.
- Model output이 충돌하면 조용히 선택하지 말고 conflict로 표시합니다.

Conflict examples:

- OCR은 특정 number를 읽었는데 chart summary는 다른 number를 사용합니다.
- UI-Venus는 region을 table로 labeling했지만 OCR은 paragraph text를 반환합니다.
- 두 crop이 서로 다른 table header를 반환합니다.

Resolution:

- Text는 exact OCR을 우선합니다.
- Region type과 bounding box는 layout model을 우선합니다.
- 정확한 evidence가 있을 때 semantic conflict resolution은 Kimi-K2.5를 우선합니다.

## Stage 6: Large VLM Synthesis

Model:

- `kimi-k2.5`

Use only when:

- 사용자가 high-quality final summary를 요청한 경우
- Screenshot에 chart 또는 complex table이 있는 경우
- OCR과 crop refinement 이후 confidence가 낮은 경우
- 여러 screenshot을 merged narrative로 묶어야 하는 경우

Input to Kimi-K2.5:

- Latency budget이 허용하면 original screenshot
- OCR text
- Region list
- Crop-level extracted text
- Table/chart candidate
- Explicit unresolved conflict

Expected result:

- Final Markdown summary
- Final JSON cleanup
- Human-review checklist
- Confidence explanation

Important rule:

- Kimi-K2.5가 missing data를 만들어내도록 요청하지 않습니다. Prompt에는 unavailable 또는 unreadable field를 unknown으로 표시하라고 명시해야 합니다.

## Stage 7: Human Review Loop

각 screenshot에 대해 review packet을 만듭니다.

- Original screenshot path
- Marked crop path가 있으면 포함
- Extracted Markdown
- Extracted JSON
- Unresolved field
- Low-confidence region

Review loop는 가볍게 유지합니다. 사람은 final JSON을 수정하거나 screenshot을 not extractable로 표시할 수 있어야 합니다.

## Stage 8: RAG Chunk Generation

Goal:

- Extraction evidence를 embedding, search, filtering, citation이 가능한 record로 변환합니다.

Chunk types:

- `region_text`: 하나의 visual region에서 나온 title/body/footer text
- `table_summary`: table title, header, important row, note
- `table_row`: 각 row가 독립적인 의미를 가질 때의 row-level chunk
- `chart_summary`: chart title, axis, legend, visible value, inferred trend
- `formula`: formula text와 nearby label/context
- `document_summary`: screenshot-level 또는 session-level summary

각 chunk는 다음을 포함해야 합니다.

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

- 고립된 cell text만 embedding하지 말고 table과 chart context를 보존합니다.
- Retrieved answer가 visible evidence를 cite할 수 있도록 bbox와 source image path를 유지합니다.
- Low-confidence chunk는 searchable하게 두되, 사용자가 uncertain evidence를 허용하지 않는 한 direct answer generation에서 trusted evidence로 쓰지 않습니다.
- Original OCR text와 cleaned content가 다르면 둘을 분리해서 보존합니다.

## 이후 Code Pass의 구현 형태

> **구현 상태(2026-06-25): skeleton 완료.** 아래 파일들이
> `side_projects/document_extraction/extraction/` 에 구현되어 있고, 모델 서버 없이
> OFFLINE(dry-run) 으로 e2e 가 도는 것을 스모크 테스트로 검증했다. 자세한 건
> [`../extraction/README.md`](../extraction/README.md) 참고.

Package location (실제):

- `side_projects/document_extraction/extraction/` (캡처 코드와 co-located.
  옛 표기 `screenshot_document_extraction/` 은 README 에 명시된 대로 `document_extraction/` 로 개명됨)

Files (구현됨):

- `extract_screenshot.py` — Stage 1~8 오케스트레이터 (폴더 단위, CLI 인자 없음)
- `prompts.py` — 스테이지별 `(system, user)` 빌더
- `models.py` — `StageRunner` (service slug 별 VLM 호출 + offline 폴백)
- `merge.py` — Stage 5 evidence merge (순수 함수)
- `schemas.py` — 출력 계약 dataclass
- `rag_chunks.py` — Stage 8 chunk 생성 + embedding_text + JSONL writer
- `test_extraction_smoke.py` — OFFLINE e2e 스모크

Reuse (현 production 경로 — docs 의 `poc.work2` 표기는 stale):

- `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient` (service slug 기반)
- `poc.workflow_3.vlm.flask_vlm` service registry
- Kimi service slug 은 `kimi-k2.6` (옛 `kimi-k2.5` 아님)

첫 버전에서는 CLI-heavy argument parsing을 추가하지 않습니다. 현재 repo의 operational-script style에 맞춰 `.env`와 in-code default를 우선합니다.

RAG database design의 상세 내용은 [rag_db_plan.md](./rag_db_plan.md)에 있습니다.
