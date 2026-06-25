"""스크린샷 단위 채점 + 파이프라인 비교 매트릭스 + acceptance 판정.

benchmark_plan.md 의 Comparison Matrix / Acceptance Criteria 를 구현한다.
"""

from dataclasses import asdict, dataclass, field

from side_projects.document_extraction.benchmark import metrics
from side_projects.document_extraction.benchmark.ground_truth import GroundTruth


@dataclass
class ScreenshotScore:
    """스크린샷 1장 채점 결과."""

    screenshot_id: str
    source_type: str = "unknown"
    text_recall: float = 0.0
    table_accuracy: float = 0.0
    chart_understanding: float = 0.0
    layout_accuracy: float = 0.0
    rag_readiness: float = 0.0
    hallucination_rate: float = 0.0  # 낮을수록 좋음
    latency: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def score_screenshot(
    extraction: dict, gt: GroundTruth
) -> ScreenshotScore:
    """추출 산출물 1장을 GT 와 대조해 모든 메트릭을 매긴다."""
    tr, tr_d = metrics.text_recall(extraction, gt)
    ta, ta_d = metrics.table_accuracy(extraction, gt)
    cu, cu_d = metrics.chart_understanding(extraction, gt)
    la, la_d = metrics.layout_accuracy(extraction, gt)
    rr, rr_d = metrics.rag_readiness(extraction, gt)
    hr, hr_d = metrics.hallucination_rate(extraction, gt)
    lat = metrics.latency_summary(extraction)

    return ScreenshotScore(
        screenshot_id=gt.screenshot_id,
        source_type=gt.source_type,
        text_recall=round(tr, 3),
        table_accuracy=round(ta, 3),
        chart_understanding=round(cu, 3),
        layout_accuracy=round(la, 3),
        rag_readiness=round(rr, 3),
        hallucination_rate=round(hr, 3),
        latency=lat,
        detail={
            "text_recall": tr_d,
            "table_accuracy": ta_d,
            "chart_understanding": cu_d,
            "layout_accuracy": la_d,
            "rag_readiness": rr_d,
            "hallucination": hr_d,
        },
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[ScreenshotScore]) -> dict:
    """여러 스크린샷 점수를 평균낸 집계."""
    if not scores:
        return {}
    return {
        "n": len(scores),
        "text_recall": round(_mean([s.text_recall for s in scores]), 3),
        "table_accuracy": round(_mean([s.table_accuracy for s in scores]), 3),
        "chart_understanding": round(_mean([s.chart_understanding for s in scores]), 3),
        "layout_accuracy": round(_mean([s.layout_accuracy for s in scores]), 3),
        "rag_readiness": round(_mean([s.rag_readiness for s in scores]), 3),
        "hallucination_rate": round(_mean([s.hallucination_rate for s in scores]), 3),
        "total_latency_ms": round(
            sum(float(s.latency.get("total_ms") or 0.0) for s in scores), 1
        ),
    }


def comparison_matrix(pipeline_scores: dict) -> str:
    """파이프라인별 집계를 benchmark_plan.md 의 Markdown 비교 표로 렌더한다.

    pipeline_scores: {pipeline_name: [ScreenshotScore, ...]}
    """
    header = (
        "| Pipeline | Text | Table | Chart | Layout | RAG | Hallucination | Latency(ms) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for name, scores in pipeline_scores.items():
        agg = aggregate(scores)
        lines.append(
            f"| {name} | {agg.get('text_recall', 0)} | {agg.get('table_accuracy', 0)} "
            f"| {agg.get('chart_understanding', 0)} | {agg.get('layout_accuracy', 0)} "
            f"| {agg.get('rag_readiness', 0)} | {agg.get('hallucination_rate', 0)} "
            f"| {agg.get('total_latency_ms', 0)} |"
        )
    return "\n".join(lines)


# benchmark_plan.md "Acceptance Criteria" 임계
ACCEPTANCE = {
    "ppt_pdf_text_recall_min": 0.7,
    "excel_table_accuracy_min": 0.7,
    "rag_readiness_min": 0.7,
}


def check_acceptance(scores: list[ScreenshotScore]) -> dict:
    """acceptance criteria 통과 여부를 판정한다(점수 집합 1개 = 1개 파이프라인)."""
    ppt_pdf = [s for s in scores if s.source_type in {"powerpoint", "pdf"}]
    excel = [s for s in scores if s.source_type == "excel"]

    ppt_pdf_recall = _mean([s.text_recall for s in ppt_pdf]) if ppt_pdf else None
    excel_table = _mean([s.table_accuracy for s in excel]) if excel else None
    rag = _mean([s.rag_readiness for s in scores]) if scores else None

    results = {
        "ppt_pdf_text_recall": None if ppt_pdf_recall is None else round(ppt_pdf_recall, 3),
        "excel_table_accuracy": None if excel_table is None else round(excel_table, 3),
        "rag_readiness": None if rag is None else round(rag, 3),
        "passed": {},
    }
    # None = 해당 source 카테고리가 벤치셋에 없음 -> n/a (all_passed 에서 제외).
    results["passed"]["ppt_pdf_text_recall"] = (
        None if ppt_pdf_recall is None
        else ppt_pdf_recall >= ACCEPTANCE["ppt_pdf_text_recall_min"]
    )
    results["passed"]["excel_table_accuracy"] = (
        None if excel_table is None
        else excel_table >= ACCEPTANCE["excel_table_accuracy_min"]
    )
    results["passed"]["rag_readiness"] = (
        None if rag is None else rag >= ACCEPTANCE["rag_readiness_min"]
    )
    applicable = [v for v in results["passed"].values() if v is not None]
    results["all_passed"] = bool(applicable) and all(applicable)
    return results


__all__ = [
    "ACCEPTANCE",
    "ScreenshotScore",
    "aggregate",
    "check_acceptance",
    "comparison_matrix",
    "score_screenshot",
]
