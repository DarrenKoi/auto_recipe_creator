# RAG Database 계획

## 목표

Screenshot extraction project는 일회성 summary가 아니라 RAG용 database를 구축해야 합니다.

각 extracted record는 다음을 보존해야 합니다.

- Visible content
- Local context
- Document/session context
- Source screenshot path
- Region coordinate
- Model provenance
- Confidence 및 review status

이렇게 해야 나중에 retrieval answer가 visible screenshot evidence까지 trace될 수 있습니다.

## 설계 원칙

주변 context가 중요한 경우 isolated OCR fragment만 embedding하지 않습니다.

나쁜 RAG chunk:

```text
Yield improved 12%.
```

더 나은 RAG chunk:

```text
In slide "Q2 CD-SEM recipe setup improvement", the chart "Manual vs AI-assisted setup time" states that yield improved 12% after the new recipe setup workflow. Source region: chart_summary, screenshot_index=3.
```

두 번째 형태는 더 길지만, heading, chart identity, metric, source를 함께 담기 때문에 retrieval에 훨씬 유용합니다.

## Storage Layers

두 개의 logical layer를 사용합니다.

| Layer | Purpose |
| --- | --- |
| Raw evidence store | 원본 OCR/model output, screenshot path, crop path, bbox, log 보관 |
| Retrieval store | Normalized chunk, metadata, embedding, review status 보관 |

Raw evidence store는 debugging과 reprocessing 용도입니다. Retrieval store는 RAG 용도입니다.

첫 구현에서는 두 layer를 모두 JSONL file로 저장할 수 있습니다. 이후 필요에 따라 retrieval store를 MongoDB plus FAISS, OpenSearch 또는 다른 vector database로 옮길 수 있습니다.

## Document Identity

각 screenshot ingestion session은 다음 ID를 부여해야 합니다.

- `collection_id`: dataset 또는 project group
- `document_id`: logical source document 또는 meeting material
- `screenshot_id`: document/session 안의 개별 image
- `screenshot_index`: session 내부의 stable ordering
- `source_type`: `powerpoint`, `pdf`, `excel`, 또는 `unknown`

Original filename이 민감하면 safe alias를 저장하고, original path는 local private manifest에만 보관합니다.

## Chunk Schema

Conceptual retrieval chunk:

```json
{
  "chunk_id": "doc001_s003_r005",
  "collection_id": "recipe_training_docs",
  "document_id": "doc001",
  "screenshot_id": "doc001_s003",
  "screenshot_index": 3,
  "source_type": "powerpoint",
  "source_image": "captures/doc001/slide_003.jpg",
  "region_id": "r005",
  "region_type": "chart_summary",
  "bbox": {"left": 100, "top": 220, "right": 900, "bottom": 620},
  "parent_heading": "Manual vs AI-assisted setup time",
  "content": "The chart compares manual setup and AI-assisted setup time. The visible trend shows lower setup time for the AI-assisted workflow.",
  "raw_ocr_text": "Manual setup ... AI-assisted ...",
  "context_before": "The slide discusses recipe setup automation benefits.",
  "context_after": "A footer notes that the data is from an internal pilot.",
  "keywords": ["recipe setup", "automation", "setup time"],
  "model_sources": ["paddleocr-vl-1.5", "ui-venus", "kimi-k2.6"],
  "confidence": 0.78,
  "review_status": "needs_review",
  "created_at": "2026-05-13"
}
```

## Chunk Types

Content마다 retrieval behavior가 다르므로 explicit chunk type을 사용합니다.

| Chunk type | Use when | Retrieval note |
| --- | --- | --- |
| `document_summary` | screenshot 또는 document-level summary | broad question에 유용 |
| `region_text` | title/body/footer region | direct fact lookup에 유용 |
| `table_summary` | whole table explanation | table intent와 column meaning에 유용 |
| `table_row` | row가 독립적으로 의미 있을 때 | entity lookup에 유용 |
| `chart_summary` | chart, legend, axis, trend | metric 및 trend question에 유용 |
| `formula` | visible formula 또는 equation | neighboring label과 variable 유지 |
| `unresolved` | low-confidence지만 유용할 수 있는 evidence | searchable하되 answer는 cautious하게 생성 |

## Context 보존 규칙

- 각 chunk를 가장 가까운 visible title 또는 heading에 연결합니다.
- Excel은 visible한 경우 tab name, filter state, header row, nearby section label 같은 sheet-like context를 포함합니다.
- Table은 row-level chunk마다 table title, header, unit, row label을 포함합니다.
- Chart는 chart title, axis label, legend label, visible numeric label을 포함합니다.
- PDF는 visible한 경우 page number와 section heading을 보존합니다.
- PowerPoint는 child chunk에 slide title과 subtitle을 보존합니다.
- Low-confidence evidence를 제거하지 말고 `needs_review`로 표시합니다.

## Embedding Text

Chunk metadata와 content에서 별도의 `embedding_text` 값을 생성합니다.

Recommended template:

```text
Source type: {source_type}
Document: {document_id}
Screenshot index: {screenshot_index}
Heading: {parent_heading}
Region type: {region_type}
Content: {content}
Context before: {context_before}
Context after: {context_after}
Keywords: {keywords}
```

Raw OCR만 embedding하지 않습니다. Raw OCR은 noisy하고 context가 부족할 수 있습니다.

## Retrieval Metadata

Minimum filterable metadata:

- `collection_id`
- `document_id`
- `screenshot_id`
- `screenshot_index`
- `source_type`
- `region_type`
- `confidence`
- `review_status`
- `model_sources`
- `created_at`

Optional metadata:

- `topic`
- `owner`
- `capture_date`
- `confidentiality_label`
- `language`
- `project`

## Retrieval Strategy

Hybrid retrieval을 사용합니다.

- Exact OCR text, title, table header, visible label에 대한 keyword search
- `embedding_text`에 대한 vector search
- Source type, document, review status, confidence 기반 metadata filter
- 복잡한 질문이거나 top chunk들이 충돌할 때만 large model rerank

Answer generation은 다음을 cite해야 합니다.

- `document_id`
- `screenshot_index`
- `region_id`
- `source_image`
- 필요한 경우 `bbox`

## Quality Gates

Trusted retrieval set에 chunk를 넣으려면 다음 조건을 만족해야 합니다.

- `content`가 비어 있지 않습니다.
- `source_image`가 존재합니다.
- `region_type`이 알려져 있습니다.
- `confidence >= 0.7`이거나 사람이 `review_status=approved`로 표시했습니다.
- Table/chart chunk는 screenshot을 열지 않아도 content를 이해할 수 있을 만큼 label을 포함합니다.

Threshold 아래의 chunk도 저장할 수 있지만, lower-trust retrieval tier에 둡니다.

## 첫 구현 계획

1. Evidence merge stage 뒤에 chunk generator를 추가합니다.
2. Raw evidence를 `outputs/raw_evidence/*.json`에 씁니다.
3. Retrieval chunk를 `outputs/rag_chunks/*.jsonl`에 씁니다.
4. `embedding_text`를 생성하되 첫 pass에서는 embedding backend를 필수로 두지 않습니다.
5. JSONL을 읽고 keyword retrieval을 확인하는 simple search smoke test를 추가합니다.

이렇게 하면 FAISS, OpenSearch, MongoDB 또는 다른 database를 선택하기 전에도 첫 버전을 유용하게 사용할 수 있습니다.
