# RAG Database Plan

## Goal

The screenshot extraction project should build a database for RAG, not just produce one-off summaries.

Each extracted record must preserve:

- Visible content
- Local context
- Document/session context
- Source screenshot path
- Region coordinates
- Model provenance
- Confidence and review status

This makes later retrieval answers traceable to visible screenshot evidence.

## Design Principle

Do not embed isolated OCR fragments when the surrounding context matters.

Bad RAG chunk:

```text
Yield improved 12%.
```

Better RAG chunk:

```text
In slide "Q2 CD-SEM recipe setup improvement", the chart "Manual vs AI-assisted setup time" states that yield improved 12% after the new recipe setup workflow. Source region: chart_summary, screenshot_index=3.
```

The second form is longer, but it is much more useful for retrieval because it carries the heading, chart identity, metric, and source.

## Storage Layers

Use two logical layers:

| Layer | Purpose |
| --- | --- |
| Raw evidence store | Keep original OCR/model outputs, screenshot paths, crop paths, bbox, and logs |
| Retrieval store | Keep normalized chunks, metadata, embeddings, and review status |

The raw evidence store is for debugging and reprocessing. The retrieval store is for RAG.

The first implementation can store both layers as JSONL files. Later, the retrieval store can move to MongoDB plus FAISS, OpenSearch, or another vector database depending on the surrounding project needs.

## Document Identity

Each screenshot ingestion session should assign:

- `collection_id`: dataset or project group
- `document_id`: logical source document or meeting material
- `screenshot_id`: one image within the document/session
- `screenshot_index`: stable ordering inside the session
- `source_type`: `powerpoint`, `pdf`, `excel`, or `unknown`

If the original filename is sensitive, store a safe alias and keep the original path only in a local private manifest.

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
  "model_sources": ["paddleocr-vl-1.5", "ui-venus", "kimi-k2.5"],
  "confidence": 0.78,
  "review_status": "needs_review",
  "created_at": "2026-05-13"
}
```

## Chunk Types

Use explicit chunk types because retrieval behavior differs by content:

| Chunk type | Use when | Retrieval note |
| --- | --- | --- |
| `document_summary` | screenshot or document-level summary | good for broad questions |
| `region_text` | title/body/footer region | good for direct fact lookup |
| `table_summary` | whole table explanation | good for table intent and column meaning |
| `table_row` | row is meaningful alone | good for entity lookup |
| `chart_summary` | chart, legend, axes, or trend | good for metric and trend questions |
| `formula` | visible formula or equation | keep neighboring label and variables |
| `unresolved` | low-confidence but potentially useful evidence | searchable but answer cautiously |

## Context Preservation Rules

- Attach each chunk to the nearest visible title or heading.
- For Excel, include sheet-like context if visible, such as tab name, filter state, header row, and nearby section label.
- For tables, include table title, headers, units, and row labels in every row-level chunk.
- For charts, include chart title, axis labels, legend labels, and visible numeric labels.
- For PDFs, preserve page number and section heading when visible.
- For PowerPoint, preserve slide title and subtitle in child chunks.
- Do not remove low-confidence evidence; mark it as `needs_review`.

## Embedding Text

Generate a separate `embedding_text` value from the chunk metadata and content.

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

Do not embed only raw OCR. Raw OCR can be noisy and context-poor.

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

Use hybrid retrieval:

- Keyword search over exact OCR text, titles, table headers, and visible labels.
- Vector search over `embedding_text`.
- Metadata filters for source type, document, review status, and confidence.
- Rerank with a large model only for complex questions or when top chunks conflict.

Answer generation should cite:

- `document_id`
- `screenshot_index`
- `region_id`
- `source_image`
- `bbox` when useful

## Quality Gates

A chunk should be accepted into the trusted retrieval set only when:

- `content` is not empty.
- `source_image` exists.
- `region_type` is known.
- `confidence >= 0.7`, or a human marked `review_status=approved`.
- Table/chart chunks include enough labels to understand the content without opening the screenshot.

Chunks below the threshold can still be stored, but should stay in a lower-trust retrieval tier.

## First Implementation Plan

1. Add a chunk generator after the evidence merge stage.
2. Write raw evidence to `outputs/raw_evidence/*.json`.
3. Write retrieval chunks to `outputs/rag_chunks/*.jsonl`.
4. Generate `embedding_text` but do not require an embedding backend in the first pass.
5. Add a simple search smoke test that reads JSONL and checks keyword retrieval.

This keeps the first version useful even before choosing FAISS, OpenSearch, MongoDB, or another database.
