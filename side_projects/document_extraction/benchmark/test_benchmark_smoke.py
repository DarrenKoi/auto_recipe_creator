"""벤치 채점 하네스 스모크 테스트 (순수 함수, 서버 불필요).

합성 ground-truth + 합성 추출 산출물로 각 메트릭이 합리적인 점수를 내는지,
완벽 추출은 만점/0 hallucination, 빈 추출은 0점이 나오는지 검증한다.

실행:
    uv run python -m side_projects.document_extraction.benchmark.test_benchmark_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.benchmark import scorer
from side_projects.document_extraction.benchmark.ground_truth import GroundTruth
from side_projects.document_extraction.benchmark.run_benchmark import run_benchmark


def _gt_dict(sid: str = "ppt_001") -> dict:
    return {
        "screenshot_id": sid,
        "source_type": "powerpoint",
        "title": "Q2 CD-SEM recipe setup improvement",
        "important_texts": ["recipe setup automation", "12 percent yield"],
        "tables": [
            {"title": "Setup time", "header": ["mode", "min"],
             "rows": [["manual", "30"], ["AI", "18"]]}
        ],
        "charts": [
            {"title": "Manual vs AI setup time", "axis_labels": ["mode", "minutes"],
             "legend_labels": ["manual", "AI"], "visible_values": ["30", "18"],
             "trend": "AI assisted is lower"}
        ],
        "region_types": ["title", "body", "table", "chart"],
        "expected_summary_keywords": ["setup time", "automation"],
        "unreadable": ["tiny footnote"],
    }


def _perfect_extraction(sid: str = "ppt_001") -> dict:
    return {
        "source_image": f"captures/{sid}.webp",
        "source_type": "powerpoint",
        "screenshot_index": 1,
        "summary_markdown": (
            "Q2 CD-SEM recipe setup automation reduced setup time. "
            "Manual 30 min vs AI 18 min."
        ),
        "overall_confidence": 0.85,
        "regions": [
            {"type": "title", "text": "Q2 CD-SEM recipe setup improvement"},
            {"type": "body", "text": "recipe setup automation improved 12 percent yield"},
            {"type": "table", "text": ""},
            {"type": "chart", "text": ""},
        ],
        "tables": [
            {"title": "Setup time", "header": ["mode", "min"],
             "cells": [["manual", "30"], ["AI", "18"]]}
        ],
        "charts": [
            {"title": "Manual vs AI setup time", "axis_labels": ["mode", "minutes"],
             "legend_labels": ["manual", "AI"], "visible_values": ["30", "18"],
             "trend_summary": "AI assisted is lower"}
        ],
        "formulas": [],
        "rag_chunks": [
            {"chunk_id": f"{sid}_r001", "source_image": f"captures/{sid}.webp",
             "screenshot_index": 1, "region_type": "region_text",
             "content": "Q2 CD-SEM recipe setup automation improved yield by 12 percent",
             "parent_heading": "Q2 CD-SEM recipe setup improvement",
             "confidence": 0.85, "review_status": "approved"},
            {"chunk_id": f"{sid}_t001", "source_image": f"captures/{sid}.webp",
             "screenshot_index": 1, "region_type": "table_summary",
             "content": "Table Setup time. Columns: mode, min. 2 visible rows.",
             "parent_heading": "Setup time",
             "confidence": 0.8, "review_status": "approved"},
        ],
        "stage_log": [
            {"stage": "ocr", "service": "paddleocr-vl-1.5", "mode": "online", "latency_ms": 1200},
            {"stage": "synthesis", "service": "kimi-k2.6", "mode": "online", "latency_ms": 8000},
        ],
    }


def test_perfect_scores_high() -> None:
    gt = GroundTruth.from_dict(_gt_dict())
    score = scorer.score_screenshot(_perfect_extraction(), gt)
    assert score.text_recall >= 0.99, score.text_recall
    assert score.table_accuracy >= 0.99, score.table_accuracy
    assert score.chart_understanding >= 0.99, score.chart_understanding
    assert score.layout_accuracy >= 0.99, score.layout_accuracy
    assert score.rag_readiness >= 0.99, score.rag_readiness
    assert score.hallucination_rate == 0.0, score.hallucination_rate
    assert score.latency["online_calls"] == 2
    print("[PASS] test_perfect_scores_high")


def test_empty_extraction_low() -> None:
    gt = GroundTruth.from_dict(_gt_dict())
    empty = {"source_image": "x.webp", "regions": [], "tables": [], "charts": [],
             "rag_chunks": [], "summary_markdown": ""}
    score = scorer.score_screenshot(empty, gt)
    assert score.text_recall == 0.0
    assert score.table_accuracy == 0.0
    assert score.chart_understanding == 0.0
    assert score.layout_accuracy == 0.0
    assert score.rag_readiness == 0.0
    print("[PASS] test_empty_extraction_low")


def test_hallucination_detected() -> None:
    gt = GroundTruth.from_dict(_gt_dict())
    ext = _perfect_extraction()
    # GT 에 없는 숫자 99 를 요약에 끼워넣음 -> hallucination 으로 잡혀야 함
    ext["summary_markdown"] += " Mysterious gain of 99 units appeared."
    score = scorer.score_screenshot(ext, gt)
    assert score.hallucination_rate > 0.0, "fabricated number 99 가 잡혀야 함"
    assert "99" in score.detail["hallucination"]["hallucinated"]
    print("[PASS] test_hallucination_detected")


def test_acceptance_and_matrix() -> None:
    gt = GroundTruth.from_dict(_gt_dict())
    scores = [scorer.score_screenshot(_perfect_extraction(), gt)]
    acc = scorer.check_acceptance(scores)
    assert acc["passed"]["ppt_pdf_text_recall"] is True
    assert acc["passed"]["rag_readiness"] is True
    matrix = scorer.comparison_matrix({"Full pipeline": scores})
    assert "| Pipeline |" in matrix and "Full pipeline" in matrix
    print("[PASS] test_acceptance_and_matrix")


def test_run_benchmark_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp)
        (bench / "ground_truth").mkdir()
        (bench / "extractions" / "ocr_only").mkdir(parents=True)
        (bench / "extractions" / "full").mkdir(parents=True)

        gt = _gt_dict("ppt_001")
        (bench / "ground_truth" / "ppt_001.json").write_text(
            json.dumps(gt, ensure_ascii=False), encoding="utf-8"
        )
        # full = 완벽, ocr_only = 표/차트 없는 빈약한 추출
        (bench / "extractions" / "full" / "ppt_001.json").write_text(
            json.dumps(_perfect_extraction("ppt_001"), ensure_ascii=False), encoding="utf-8"
        )
        weak = {"source_image": "x.webp", "screenshot_index": 1,
                "regions": [{"type": "body", "text": "recipe setup automation"}],
                "tables": [], "charts": [], "rag_chunks": [], "summary_markdown": ""}
        (bench / "extractions" / "ocr_only" / "ppt_001.json").write_text(
            json.dumps(weak, ensure_ascii=False), encoding="utf-8"
        )

        out = run_benchmark(bench)
        assert "full" in out["scores"] and "ocr_only" in out["scores"]
        # full 이 ocr_only 보다 table/chart 에서 우월해야 함
        full = out["scores"]["full"][0]
        weak_s = out["scores"]["ocr_only"][0]
        assert full.table_accuracy > weak_s.table_accuracy
        assert full.chart_understanding > weak_s.chart_understanding
        # 결과 파일 생성 확인
        assert (bench / "results" / "comparison_matrix.md").exists()
        assert (bench / "results" / "acceptance.json").exists()
    print("[PASS] test_run_benchmark_end_to_end")


def main() -> int:
    test_perfect_scores_high()
    test_empty_extraction_low()
    test_hallucination_detected()
    test_acceptance_and_matrix()
    test_run_benchmark_end_to_end()
    print("\n[INFO] 모든 벤치 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
