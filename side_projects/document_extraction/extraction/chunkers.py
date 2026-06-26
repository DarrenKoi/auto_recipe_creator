"""C3: PageModel -> RagChunk (질의유형별 chunker, 순수, VLM 불필요).

질의유형 매핑:
- 절차/how-to     -> region_type="procedure"  (Step 시퀀스를 한 chunk 로 묶음)
- 파라미터/스펙   -> "table_summary" + "table_row" (native 표, flatten 금지)
- 에러/알람 코드  -> "error_code"     (code->meaning->action 단위)
- 개념/다이어그램 -> "figure"         (그림 + 근접 caption, source_image=원본)
- 일반 prose      -> "region_text"

chunk-type 은 기존 관례대로 RagChunk.region_type 에 저장한다(rag_chunks.py 와 동일).
build_embedding_text / TRUST_CONFIDENCE / write_chunks_jsonl 은 재사용한다.
section_path 는 context_before 에 실어 embedding_text 에 자연히 포함시킨다.
"""

import re

from side_projects.document_extraction.extraction.schemas import BBox, RagChunk
from side_projects.document_extraction.extraction.rag_chunks import (
    MAX_TABLE_ROW_CHUNKS,
    TRUST_CONFIDENCE,
    build_embedding_text,
)

# 디지털 harvest 는 정확하므로 텍스트/표 chunk 는 높은 confidence.
CONF_TEXT = 0.95
CONF_FIGURE = 0.8   # caption 은 근접 텍스트 추정(정확도 낮음)

# 에러코드 표 판정: header 셀에 이 키워드가 있으면 error_code chunk 로.
ERROR_HEADER_KEYWORDS = {"code", "error", "alarm", "alid", "fault", "alm", "err"}

# 절차 단계 라인: "Step 3:" 또는 "3." / "3)" 로 시작.
_STEP_RE = re.compile(r"^\s*(step\s+\d+\b|\d+[.)]\s)", re.I)


def _step_line_count(text: str) -> int:
    return sum(1 for ln in text.splitlines() if _STEP_RE.match(ln))


def _bbox(coords) -> BBox:
    if not coords or len(coords) < 4:
        return BBox()
    return BBox(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))


def _section_leaf(page) -> str:
    """페이지 섹션 경로의 말단 제목(없으면 빈 문자열). 표/그림 chunk 의 heading 폴백."""
    return page.section_path[-1] if page.section_path else ""


def _make_chunk(doc_id, page, suffix, region_type, content, *, bbox=None,
                parent_heading="", source_image="", confidence=CONF_TEXT,
                keywords=None, raw_text="", context_after=""):
    """공통 provenance 를 채운 RagChunk. section_path 는 context_before 로 보존."""
    chunk = RagChunk(
        chunk_id=f"{doc_id}_p{page.page_no:04d}_{suffix}",
        document_id=doc_id,
        screenshot_id=f"{doc_id}_p{page.page_no:04d}",
        screenshot_index=page.page_no,
        source_type="pdf",
        source_image=source_image or page.render_path,
        region_id=suffix,
        region_type=region_type,
        bbox=bbox or BBox(),
        parent_heading=parent_heading,
        content=content,
        raw_ocr_text=raw_text,
        context_before=" > ".join(page.section_path),
        context_after=context_after,
        keywords=keywords or [],
        confidence=confidence,
        review_status="approved" if confidence >= TRUST_CONFIDENCE else "needs_review",
    )
    chunk.embedding_text = build_embedding_text(chunk)
    return chunk


def chunk_page(page, doc_id) -> list:
    """한 페이지 -> RagChunk 목록. 블록은 한 번만 소비(절차로 묶인 블록은 region_text 제외)."""
    chunks = []
    consumed = set()

    chunks.extend(_procedure_chunks(page, doc_id, consumed))
    chunks.extend(_table_chunks(page, doc_id))
    chunks.extend(_figure_chunks(page, doc_id))
    chunks.extend(_region_text_chunks(page, doc_id, consumed))
    return chunks


def chunk_bundle(bundle) -> list:
    """bundle 전체 -> RagChunk 목록 (structure 가 먼저 assign 되어 있어야 함)."""
    out = []
    for page in bundle.pages:
        out.extend(chunk_page(page, bundle.doc_id))
    return out


def _procedure_chunks(page, doc_id, consumed) -> list:
    """연속한 step 블록을 한 procedure chunk 로. run 의 총 step 라인이 2개 이상일 때만."""
    chunks = []
    run = []  # [(idx, block)]
    order = 1

    def flush():
        nonlocal order
        if not run:
            return
        total = sum(_step_line_count(b.text) for _, b in run)
        if total >= 2:
            text = "\n".join(b.text for _, b in run)
            first = run[0][1]
            chunks.append(_make_chunk(
                doc_id, page, f"proc{order}", "procedure", text,
                bbox=_bbox(first.bbox), parent_heading=first.parent_heading,
                keywords=["procedure"], raw_text=text))
            order += 1
            for idx, _ in run:
                consumed.add(idx)

    for idx, b in enumerate(page.blocks):
        if b.is_heading or not b.text.strip():
            flush(); run = []
            continue
        if _step_line_count(b.text) >= 1:
            run.append((idx, b))
        else:
            flush(); run = []
    flush()
    return chunks


def _is_error_table(header) -> bool:
    return any(str(h).strip().lower() in ERROR_HEADER_KEYWORDS for h in header)


def _row_pairs(header, row) -> str:
    """행을 'Col: val; ...' 로. header 보다 긴 행은 colN 으로 보강(데이터 안 자름)."""
    pairs = []
    for cidx, val in enumerate(row):
        col = header[cidx] if cidx < len(header) else f"col{cidx + 1}"
        pairs.append(f"{col}: {'' if val is None else val}")
    return "; ".join(pairs)


def _table_chunks(page, doc_id) -> list:
    """표 -> error_code(에러표) 또는 table_summary + table_row(일반표)."""
    chunks = []
    tnum = 0
    for table in page.tables:
        rows = table.get("rows") or []
        if not rows:
            continue
        tnum += 1
        header = [("" if c is None else str(c)) for c in rows[0]]
        data = rows[1:]

        if _is_error_table(header):
            for ridx, row in enumerate(data):
                code = "" if not row else ("" if row[0] is None else str(row[0]))
                chunks.append(_make_chunk(
                    doc_id, page, f"t{tnum}_err{ridx + 1}", "error_code",
                    _row_pairs(header, row), bbox=_bbox(table.get("bbox")),
                    parent_heading=_section_leaf(page),
                    keywords=[code] if code else [], raw_text=_row_pairs(header, row)))
            continue

        # 일반 표: 요약 1개 + 행 단위(상한)
        chunks.append(_make_chunk(
            doc_id, page, f"t{tnum}_sum", "table_summary",
            f"Table columns: {', '.join(header)}. {len(data)} rows.",
            bbox=_bbox(table.get("bbox")),
            parent_heading=_section_leaf(page),
            keywords=header))
        capped = data
        if len(data) > MAX_TABLE_ROW_CHUNKS:
            print(f"[INFO] table_row 상한 적용: p{page.page_no} t{tnum} "
                  f"{len(data)}행 중 {MAX_TABLE_ROW_CHUNKS}행만 chunk 화")
            capped = data[:MAX_TABLE_ROW_CHUNKS]
        for ridx, row in enumerate(capped):
            chunks.append(_make_chunk(
                doc_id, page, f"t{tnum}_row{ridx + 1}", "table_row",
                _row_pairs(header, row), bbox=_bbox(table.get("bbox")),
                parent_heading=_section_leaf(page),
                raw_text=_row_pairs(header, row)))
    return chunks


def _nearest_block_text(page, fbbox) -> str:
    """figure bbox 와 세로 중심이 가장 가까운 텍스트 블록의 text(근접 caption 추정).
    실제 caption 품질은 C4(VLM)에서 보강한다 - 여기선 휴리스틱."""
    if not fbbox or len(fbbox) < 4 or not page.blocks:
        return ""
    fcy = (fbbox[1] + fbbox[3]) / 2
    best, bestd = "", None
    for b in page.blocks:
        if b.is_heading or not b.bbox or not b.text.strip():
            continue
        bcy = (b.bbox[1] + b.bbox[3]) / 2
        d = abs(bcy - fcy)
        if bestd is None or d < bestd:
            best, bestd = b.text.strip(), d
    return best


def _figure_chunks(page, doc_id) -> list:
    """그림 1개 = 1 figure chunk. source_image=원본 바이트, caption=근접 텍스트."""
    chunks = []
    for fnum, fig in enumerate(page.figures, start=1):
        fbbox = (fig.get("bboxes_on_page") or [None])[0]
        caption = _nearest_block_text(page, fbbox)
        content = caption or f"Figure on page {page.page_no}"
        chunks.append(_make_chunk(
            doc_id, page, f"fig{fnum}", "figure", content,
            bbox=_bbox(fbbox), source_image=fig.get("path", ""),
            parent_heading=_section_leaf(page),
            confidence=CONF_FIGURE, keywords=["figure"]))
    return chunks


def _region_text_chunks(page, doc_id, consumed) -> list:
    """heading/소비됨 아닌 본문 블록 -> region_text."""
    chunks = []
    order = 1
    for idx, b in enumerate(page.blocks):
        if idx in consumed or b.is_heading or not b.text.strip():
            continue
        chunks.append(_make_chunk(
            doc_id, page, f"r{order}", "region_text", b.text.strip(),
            bbox=_bbox(b.bbox), parent_heading=b.parent_heading, raw_text=b.text.strip()))
        order += 1
    return chunks


__all__ = ["CONF_FIGURE", "CONF_TEXT", "ERROR_HEADER_KEYWORDS",
           "chunk_bundle", "chunk_page"]
