"""추출 벤치마크 채점 하네스 (benchmark_plan.md).

모델이 만든 추출 산출물(extraction/ 의 ExtractionResult JSON)을 ground-truth
JSON 과 대조해 채점한다. 모든 채점 로직은 *순수 함수* 라 VLM 서버 없이 집에서
단위 테스트가 된다(실제 스크린샷/모델 호출은 사내 PC 에서, 채점은 어디서나).

메트릭(benchmark_plan.md):
    Text Recall / Table Accuracy / Chart Understanding / Layout Accuracy /
    Hallucination Rate / Latency / RAG Readiness

흐름:
    ground_truth.GroundTruth   <- 사람이 만든 정답 파일
    metrics.*                  <- (extraction, gt) -> 0..1 점수 + 근거
    scorer.score_screenshot    <- 한 장 채점 -> ScreenshotScore
    scorer.comparison_matrix   <- 파이프라인 x 스크린샷 비교 표(Markdown)
    run_benchmark              <- 폴더 단위 실행(엔트리, CLI 인자 없음)
"""

from side_projects.document_extraction.benchmark.ground_truth import GroundTruth
from side_projects.document_extraction.benchmark.scorer import (
    ScreenshotScore,
    comparison_matrix,
    score_screenshot,
)

__all__ = [
    "GroundTruth",
    "ScreenshotScore",
    "comparison_matrix",
    "score_screenshot",
]
