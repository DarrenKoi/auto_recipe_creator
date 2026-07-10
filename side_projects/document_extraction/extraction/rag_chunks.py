"""Stage 8: RAG chunk 생성 + embedding_text + JSONL writer.

ExtractionResult 의 region/table/chart/formula 를 retrieval-ready chunk 로 바꾼다
(rag_db_plan.md). 각 chunk 는 source provenance(bbox, source_image, index)와
context, confidence, review_status 를 보존해 나중에 citation 이 가능하게 한다.

embedding backend 는 첫 pass 에서 필수가 아니다(rag_db_plan.md). 여기서는
embedding_text 만 생성하고, 실제 embedding 은 후속 단계로 둔다.
"""

import json
from pathlib import Path

from side_projects.document_extraction.extraction.schemas import (
    ExtractionResult,
    RagChunk,
)


# confidence 가 이 값 이상이면 trusted tier 후보 (rag_db_plan.md Quality Gates).
TRUST_CONFIDENCE = 0.7

# 표 1개당 생성할 table_row chunk 상한(폭주 방지). 초과분은 로그로 알림(silent 금지).
MAX_TABLE_ROW_CHUNKS = 50


def build_embedding_text(chunk: RagChunk) -> str:
    """chunk metadata + content 로 embedding 입력 텍스트를 만든다.

    raw OCR 만 임베딩하지 않는다(noisy). heading/region_type/context 를 함께 넣어
    retrieval 친화적으로 만든다(rag_db_plan.md "Embedding Text").
    """
    keywords = ", ".join(chunk.keywords)
    return (
        f"Source type: {chunk.source_type}\n"
        f"Document: {chunk.document_id}\n"
        f"Screenshot index: {chunk.screenshot_index}\n"
        f"Heading: {chunk.parent_heading}\n"
        f"Region type: {chunk.region_type}\n"
        f"Content: {chunk.content}\n"
        f"Context before: {chunk.context_before}\n"
        f"Context after: {chunk.context_after}\n"
        f"Keywords: {keywords}"
    )


def _review_status(confidence: float) -> str:
    return "needs_review" if confidence < TRUST_CONFIDENCE else "approved"


def _new_chunk(result: ExtractionResult, suffix: str, **kwargs) -> RagChunk:
    """공통 식별자/provenance 를 채운 RagChunk 를 만든다."""
    base_id = result.screenshot_id or f"s{result.screenshot_index:03d}"
    chunk = RagChunk(
        chunk_id=f"{base_id}_{suffix}",
        collection_id=result.collection_id,
        document_id=result.document_id,
        screenshot_id=result.screenshot_id,
        screenshot_index=result.screenshot_index,
        source_type=result.source_type,
        source_image=result.source_image,
        created_at=kwargs.pop("created_at", ""),
        **kwargs,
    )
    chunk.review_status = _review_status(chunk.confidence)
    chunk.embedding_text = build_embedding_text(chunk)
    return chunk


def generate_chunks(
    result: ExtractionResult,
    *,
    created_at: str = "",
    chart_crop_lookup: dict | None = None,
) -> list[RagChunk]:
    """ExtractionResult -> RagChunk 목록.

    region_text / table_summary / chart_summary / formula / document_summary 를 생성한다.
    table_row(행 단위) chunk 는 후속 단계로 두고 skeleton 에서는 table_summary 만.

    chart_crop_lookup: {chart_region_id -> crop 경로}. chart_summary chunk 에
    R1 래스터 provenance(crop_path)를 연결한다(3-표상 저장의 R1 고리).
    """
    chart_crop_lookup = chart_crop_lookup or {}
    chunks: list[RagChunk] = []

    # region_text: title/body/footer/legend 등 텍스트 region
    for region in result.regions:
        if not region.text.strip():
            continue
        chunk_type = "region_text"
        chunks.append(
            _new_chunk(
                result,
                region.region_id,
                region_id=region.region_id,
                region_type=chunk_type,
                bbox=region.bbox,
                parent_heading=_nearest_heading(result, region),
                content=region.text.strip(),
                raw_ocr_text=region.text.strip(),
                confidence=region.confidence,
                model_sources=list(region.model_sources),
                created_at=created_at,
            )
        )

    # table_summary
    for table in result.tables:
        header = ", ".join(table.header)
        content = (
            f"Table '{table.title}'. Columns: {header}. "
            f"{len(table.cells)} visible rows."
        ).strip()
        chunks.append(
            _new_chunk(
                result,
                table.region_id,
                region_id=table.region_id,
                region_type="table_summary",
                parent_heading=table.title,
                content=content,
                confidence=table.confidence,
                model_sources=list(table.model_sources),
                created_at=created_at,
            )
        )

    # table_row: 각 row 가 독립적으로 의미 있도록 header 맥락과 함께 chunk 화
    for table in result.tables:
        if not table.header or not table.cells:
            continue
        rows = table.cells
        if len(rows) > MAX_TABLE_ROW_CHUNKS:
            print(
                f"[INFO] table_row 상한 적용: {table.region_id} {len(rows)}행 중 "
                f"{MAX_TABLE_ROW_CHUNKS}행만 chunk 화"
            )
            rows = rows[:MAX_TABLE_ROW_CHUNKS]
        for ridx, row in enumerate(rows):
            pairs = []
            for cidx, val in enumerate(row):
                col = table.header[cidx] if cidx < len(table.header) else f"col{cidx + 1}"
                pairs.append(f"{col}: {val}")
            content = f"Row in table '{table.title}': " + "; ".join(pairs)
            chunks.append(
                _new_chunk(
                    result,
                    f"{table.region_id}_row{ridx + 1}",
                    region_id=table.region_id,
                    region_type="table_row",
                    parent_heading=table.title,
                    content=content,
                    confidence=table.confidence,
                    model_sources=list(table.model_sources),
                    created_at=created_at,
                )
            )

    # chart_summary
    for chart in result.charts:
        content = (
            f"Chart '{chart.title}'. Axes: {', '.join(chart.axis_labels)}. "
            f"Legend: {', '.join(chart.legend_labels)}. "
            f"Trend: {chart.trend_summary}".strip()
        )
        chunks.append(
            _new_chunk(
                result,
                chart.region_id,
                region_id=chart.region_id,
                region_type="chart_summary",
                crop_path=chart_crop_lookup.get(chart.region_id, ""),
                parent_heading=chart.title,
                content=content,
                confidence=chart.confidence,
                model_sources=list(chart.model_sources),
                created_at=created_at,
            )
        )

    # formula
    for formula in result.formulas:
        if not formula.latex.strip():
            continue
        chunks.append(
            _new_chunk(
                result,
                formula.region_id,
                region_id=formula.region_id,
                region_type="formula",
                parent_heading=formula.nearby_label,
                content=formula.latex.strip(),
                confidence=formula.confidence,
                model_sources=list(formula.model_sources),
                created_at=created_at,
            )
        )

    # document_summary: 합성 요약이 있으면 broad-question chunk 로 추가
    if result.summary_markdown.strip():
        chunks.append(
            _new_chunk(
                result,
                "summary",
                region_type="document_summary",
                content=result.summary_markdown.strip(),
                confidence=result.overall_confidence,
                model_sources=list(result.summary_model_sources) or ["unknown"],
                created_at=created_at,
            )
        )

    result.rag_chunks = chunks
    return chunks


def _nearest_heading(result: ExtractionResult, region) -> str:
    """region 위쪽에 가장 가까운 title region 텍스트를 heading 으로 쓴다.

    bbox 기준: title.bottom <= region.top(위에 있음) 중 가로로 겹치는 것을 우선,
    그다음 세로 gap 이 작은 것을 고른다. 위쪽 title 이 없으면 첫 title 로 폴백.
    bbox 가 전부 0(offline stub)이면 자연스럽게 첫 title 로 수렴한다.
    """
    titles = [r for r in result.regions if r.type == "title" and r.text.strip()]
    if not titles:
        return ""

    rb = region.bbox
    above = []
    for t in titles:
        if t.region_id == region.region_id:
            continue  # 자기 자신 제외
        tb = t.bbox
        if tb.bottom <= rb.top:
            overlap = min(tb.right, rb.right) - max(tb.left, rb.left)
            gap = rb.top - tb.bottom
            above.append((0 if overlap > 0 else 1, gap, t))
    if above:
        above.sort(key=lambda x: (x[0], x[1]))
        return above[0][2].text.strip()
    return titles[0].text.strip()


def write_chunks_jsonl(chunks: list[RagChunk], out_path: Path) -> int:
    """RagChunk 목록을 JSONL 로 append 저장하고 기록한 개수를 반환한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("a", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_raw_evidence(result: ExtractionResult, out_path: Path) -> None:
    """raw evidence(전체 ExtractionResult)를 JSON 으로 저장(debug/reprocess 용)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "TRUST_CONFIDENCE",
    "build_embedding_text",
    "generate_chunks",
    "write_chunks_jsonl",
    "write_raw_evidence",
]
