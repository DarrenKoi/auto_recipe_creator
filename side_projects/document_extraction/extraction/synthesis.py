"""Stage 6 대체: 결정론적 무(無)-LLM 합성.

kimi(또는 어떤 VLM)도 호출하지 않고, merge 된 evidence 만으로 summary_markdown,
overall_confidence, unresolved 를 조립한다. 기본 경로가 모델 0콜로 e2e 완주하도록
하는 것이 목적("no more kimi" 정신). 사용자가 고품질 합성을 원하면 호출부에서
StageRunner.run_synthesis(kimi-k2.6) 로 갈아끼울 수 있다.

규칙(pipeline_plan.md Stage 6 important rule 계승): 보이지 않는 값을 만들지 않는다.
여기서는 *추출된 evidence 만* 문장으로 엮으므로 창작 위험이 없다.
"""

from side_projects.document_extraction.extraction.schemas import ExtractionResult


def _title_text(result: ExtractionResult) -> str:
    """첫 title region 텍스트(없으면 빈 문자열)."""
    for region in result.regions:
        if region.type == "title" and region.text.strip():
            return region.text.strip()
    return ""


def _confidence(result: ExtractionResult) -> float:
    """evidence 신뢰도 집계.

    region/table/chart/formula 의 confidence 평균을 쓰되, 모두 0(미측정)이면
    '얼마나 많은 종류의 evidence 가 잡혔는지'를 약한 proxy 로 사용한다(0.0~0.6).
    값 창작이 아니라 'evidence 충실도' 추정이므로 0.7(trust gate) 밑으로 캡한다.
    """
    confs = [r.confidence for r in result.regions if r.confidence > 0]
    confs += [t.confidence for t in result.tables if t.confidence > 0]
    confs += [c.confidence for c in result.charts if c.confidence > 0]
    confs += [f.confidence for f in result.formulas if f.confidence > 0]
    if confs:
        return round(sum(confs) / len(confs), 3)

    # 신뢰도 미측정(offline stub 등): evidence 다양성 기반 약한 추정.
    has_text = any(r.text.strip() for r in result.regions)
    kinds = sum(
        [has_text, bool(result.tables), bool(result.charts), bool(result.formulas)]
    )
    return round(min(0.6, 0.15 * kinds), 3)


def synthesize_deterministic(result: ExtractionResult) -> dict:
    """evidence -> {summary_markdown, overall_confidence, unresolved}. 모델 미사용."""
    lines: list[str] = []

    title = _title_text(result)
    if title:
        lines.append(f"# {title}")

    # 본문: title 이 아닌 텍스트 region 을 bullet 로
    body_texts = [
        r.text.strip()
        for r in result.regions
        if r.type != "title" and r.text.strip()
    ]
    if body_texts:
        lines.append("")
        for text in body_texts:
            # 너무 길면 한 줄로 collapse
            lines.append(f"- {' '.join(text.split())}")

    # 표 요약
    if result.tables:
        lines.append("")
        lines.append("## Tables")
        for table in result.tables:
            cols = ", ".join(table.header) if table.header else "(no header)"
            name = table.title or table.region_id
            lines.append(f"- **{name}** — columns: {cols}; {len(table.cells)} rows")

    # 차트 요약(보이는 라벨/추세만)
    if result.charts:
        lines.append("")
        lines.append("## Charts")
        for chart in result.charts:
            name = chart.title or chart.region_id
            bits = []
            if chart.legend_labels:
                bits.append("legend: " + ", ".join(chart.legend_labels))
            if chart.trend_summary:
                bits.append("trend: " + chart.trend_summary)
            suffix = ("; " + "; ".join(bits)) if bits else ""
            lines.append(f"- **{name}**{suffix}")

    # 수식
    if result.formulas:
        lines.append("")
        lines.append("## Formulas")
        for formula in result.formulas:
            if formula.latex.strip():
                lines.append(f"- `{formula.latex.strip()}`")

    summary_markdown = "\n".join(lines).strip()

    # unresolved: region.conflicts 가 merge 의 coarse 한 항목보다 구체적이므로,
    # conflict 가 있는 region 은 merge 가 넣은 동일-region 항목(예: 'r001: ...')을
    # 제거하고 상세 conflict 로 대체한다(같은 충돌의 이중 보고 방지).
    conflict_region_ids = {r.region_id for r in result.regions if r.conflicts}
    unresolved = [
        u
        for u in result.unresolved
        if not any(u.startswith(f"{rid}:") for rid in conflict_region_ids)
    ]
    for region in result.regions:
        for conflict in region.conflicts:
            unresolved.append(f"{region.region_id}: {conflict}")

    return {
        "summary_markdown": summary_markdown,
        "overall_confidence": _confidence(result),
        "unresolved": unresolved,
    }


__all__ = ["synthesize_deterministic"]
