"""Stage 5: evidence merge.

OCR(Stage 2) + layout(Stage 3) + crop(Stage 4) 의 raw 출력을 받아
ExtractionResult 의 regions/tables/charts/formulas 로 정규화한다.

규칙(pipeline_plan.md Stage 5):
    - region 좌표 정규화, 반복 OCR 텍스트 dedup
    - OCR 텍스트를 layout region 에 attach
    - 모델 출력이 충돌하면 조용히 고르지 말고 conflict 로 표시
    - 텍스트는 exact OCR 우선, region type/bbox 는 layout 모델 우선

이 단계는 *순수 함수* 라 VLM 서버 없이 단위 테스트가 된다.
"""

from side_projects.document_extraction.extraction.schemas import (
    BBox,
    Chart,
    ExtractionResult,
    Formula,
    Region,
    Table,
)


def _bbox_from_dict(raw: dict | None) -> BBox:
    if not isinstance(raw, dict):
        return BBox()
    def _int(key: str) -> int:
        try:
            return int(raw.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0
    return BBox(_int("left"), _int("top"), _int("right"), _int("bottom"))


def _normalize_region_type(raw_type: str) -> str:
    from side_projects.document_extraction.extraction.schemas import REGION_TYPES

    value = (raw_type or "other").strip().lower()
    return value if value in REGION_TYPES else "other"


def merge_evidence(
    *,
    source_image: str,
    ocr: dict,
    layout: dict,
    screenshot_index: int = 1,
    document_id: str = "",
    collection_id: str = "",
    screenshot_id: str = "",
) -> ExtractionResult:
    """OCR + layout evidence 를 ExtractionResult 로 병합한다.

    crop refinement 결과는 호출부에서 ocr/layout 에 미리 반영했다고 가정한다
    (skeleton 단계에서는 crop 을 별도 인자로 받지 않는다).
    """
    source_type = (layout.get("source_type") or "unknown").strip().lower()
    result = ExtractionResult(
        source_image=source_image,
        source_type=source_type if source_type else "unknown",
        document_id=document_id,
        collection_id=collection_id,
        screenshot_id=screenshot_id,
        screenshot_index=screenshot_index,
    )

    raw_text = (ocr.get("raw_text") or "").strip()

    # --- regions: layout 의 bbox/type 권위 + OCR raw_text attach -------------
    layout_regions = layout.get("regions") or []
    for idx, raw_region in enumerate(layout_regions):
        if not isinstance(raw_region, dict):
            continue
        region = Region(
            region_id=f"r{idx + 1:03d}",
            type=_normalize_region_type(raw_region.get("type", "other")),
            bbox=_bbox_from_dict(raw_region.get("bbox")),
            model_sources=["ui-venus"],
        )
        # 단일 region 이면 OCR 전체 텍스트를 붙인다(가장 단순한 attach 규칙).
        if len(layout_regions) == 1:
            region.text = raw_text
            if raw_text:
                region.model_sources.append("paddleocr-vl-1.5")
        result.regions.append(region)

    # layout 이 비었으면 전체를 하나의 body region 으로 둔다.
    if not result.regions and raw_text:
        result.regions.append(
            Region(
                region_id="r001",
                type="body",
                text=raw_text,
                model_sources=["paddleocr-vl-1.5"],
            )
        )

    # --- tables / charts / formulas: OCR 출력에서 정규화 --------------------
    for tidx, raw_table in enumerate(ocr.get("tables") or []):
        if not isinstance(raw_table, dict):
            continue
        result.tables.append(
            Table(
                region_id=f"t{tidx + 1:03d}",
                title=(raw_table.get("title") or "").strip(),
                header=[str(h) for h in (raw_table.get("header") or [])],
                cells=[[str(c) for c in row] for row in (raw_table.get("rows") or [])],
                model_sources=["paddleocr-vl-1.5"],
            )
        )

    for cidx, raw_chart in enumerate(ocr.get("charts") or []):
        if not isinstance(raw_chart, dict):
            continue
        result.charts.append(
            Chart(
                region_id=f"c{cidx + 1:03d}",
                title=(raw_chart.get("title") or "").strip(),
                axis_labels=[str(a) for a in (raw_chart.get("axis_labels") or [])],
                legend_labels=[str(l) for l in (raw_chart.get("legend_labels") or [])],
                visible_values=[str(v) for v in (raw_chart.get("visible_values") or [])],
                trend_summary=(raw_chart.get("trend_summary") or "").strip(),
                model_sources=["paddleocr-vl-1.5"],
            )
        )

    for fidx, raw_formula in enumerate(ocr.get("formulas") or []):
        if not isinstance(raw_formula, dict):
            continue
        result.formulas.append(
            Formula(
                region_id=f"f{fidx + 1:03d}",
                latex=(raw_formula.get("latex") or "").strip(),
                nearby_label=(raw_formula.get("nearby_label") or "").strip(),
                model_sources=["paddleocr-vl-1.5"],
            )
        )

    # --- 충돌 표시: layout 이 table 이라는데 OCR 표가 하나도 없으면 conflict --
    layout_table_regions = [r for r in result.regions if r.type == "table"]
    if layout_table_regions and not result.tables:
        for region in layout_table_regions:
            region.conflicts.append(
                "layout marked region as table but OCR returned no table structure"
            )
            result.unresolved.append(
                f"{region.region_id}: table region without parsed table"
            )

    return result


__all__ = ["merge_evidence"]
