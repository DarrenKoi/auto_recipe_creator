"""
RAG Evaluation Framework

RAG 시스템 성능 평가 프레임워크
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from test.vlm_input_control.screen_capture import ScreenCapture
from test.vlm_input_control.vlm_screen_analysis import (
    VLMScreenAnalyzer, VLMProvider
)
from test.vlm_input_control.rag_context_manager import create_rag_context_manager
from test.vlm_input_control.rag_prompt_builder import create_rag_prompt_builder
from test.video_frame_parser.db_handler import DatabaseHandler
from test.video_frame_parser.config import DatabaseConfig


@dataclass
class EvaluationCase:
    """평가 케이스"""
    case_id: str
    screenshot_path: str
    expected_state: str
    query_text: Optional[str] = None
    description: str = ""


@dataclass
class EvaluationResult:
    """평가 결과"""
    case_id: str
    # Without RAG
    no_rag_state: str
    no_rag_confidence: float
    no_rag_time_ms: float
    no_rag_correct: bool
    # With RAG
    rag_state: str
    rag_confidence: float
    rag_time_ms: float
    rag_correct: bool
    # Retrieval 품질
    retrieval_time_ms: float
    num_similar_frames: int
    num_actions: int
    num_errors: int


@dataclass
class EvaluationSummary:
    """평가 요약"""
    total_cases: int
    # Accuracy
    no_rag_accuracy: float
    rag_accuracy: float
    accuracy_improvement: float
    # Latency
    no_rag_avg_latency_ms: float
    rag_avg_latency_ms: float
    latency_overhead_ms: float
    latency_overhead_pct: float
    # Retrieval
    avg_retrieval_time_ms: float
    avg_similar_frames: float
    avg_actions: float
    avg_errors: float


class RAGEvaluator:
    """RAG 평가기"""

    def __init__(
        self,
        vlm_provider: VLMProvider,
        api_url: str,
        api_key: Optional[str] = None,
        db_config: Optional[DatabaseConfig] = None
    ):
        """
        Args:
            vlm_provider: VLM 제공자
            api_url: VLM API URL
            api_key: API 키
            db_config: 데이터베이스 설정
        """
        self.vlm_provider = vlm_provider
        self.api_url = api_url
        self.api_key = api_key

        # DatabaseHandler 초기화
        config = db_config or DatabaseConfig()
        self.db = DatabaseHandler(config)
        self.db.initialize()
        self.db.init_text_faiss_index()

        # RAG 컴포넌트
        self.rag_manager = create_rag_context_manager(
            db_handler=self.db,
            use_text_search=True,
            use_visual_search=True
        )

        self.rag_prompt_builder = create_rag_prompt_builder(
            max_similar_frames=3
        )

        # VLM Analyzer (Without RAG)
        self.vlm_no_rag = VLMScreenAnalyzer(
            provider=vlm_provider,
            api_base_url=api_url,
            api_key=api_key,
            use_rag=False
        )

        # VLM Analyzer (With RAG)
        self.vlm_with_rag = VLMScreenAnalyzer(
            provider=vlm_provider,
            api_base_url=api_url,
            api_key=api_key,
            rag_manager=self.rag_manager,
            rag_prompt_builder=self.rag_prompt_builder,
            use_rag=True
        )

        print(f"[INFO] RAGEvaluator 초기화 완료")

    def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        """
        단일 케이스 평가

        Args:
            case: 평가 케이스

        Returns:
            EvaluationResult
        """
        print(f"\n[INFO] 평가 중: {case.case_id}")
        print(f"[INFO] 설명: {case.description}")

        # 이미지 로드
        with open(case.screenshot_path, 'rb') as f:
            image_data = f.read()

        # 1. Without RAG
        print("[INFO] Without RAG 분석 중...")
        start_time = time.time()
        result_no_rag = self.vlm_no_rag.analyze_screen(image_data)
        no_rag_time = (time.time() - start_time) * 1000

        # 2. With RAG
        print("[INFO] With RAG 분석 중...")
        retrieval_start = time.time()

        # RAG 컨텍스트 검색
        rag_context = self.rag_manager.retrieve_context(
            current_screen=image_data,
            query_text=case.query_text,
            top_k=3
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        # VLM 분석
        start_time = time.time()
        result_with_rag = self.vlm_with_rag.analyze_screen(
            image_data, query_text=case.query_text
        )
        rag_time = (time.time() - start_time) * 1000

        # 3. 결과 평가
        no_rag_correct = (
            result_no_rag and
            result_no_rag.state_id == case.expected_state
        )

        rag_correct = (
            result_with_rag and
            result_with_rag.state_id == case.expected_state
        )

        result = EvaluationResult(
            case_id=case.case_id,
            no_rag_state=result_no_rag.state_id if result_no_rag else "error",
            no_rag_confidence=result_no_rag.confidence if result_no_rag else 0.0,
            no_rag_time_ms=no_rag_time,
            no_rag_correct=no_rag_correct,
            rag_state=result_with_rag.state_id if result_with_rag else "error",
            rag_confidence=result_with_rag.confidence if result_with_rag else 0.0,
            rag_time_ms=rag_time,
            rag_correct=rag_correct,
            retrieval_time_ms=retrieval_time,
            num_similar_frames=len(rag_context.similar_frames),
            num_actions=len(rag_context.action_sequences),
            num_errors=len(rag_context.error_patterns)
        )

        # 결과 출력
        print(f"\n결과:")
        print(f"  Without RAG: {result.no_rag_state} "
              f"(신뢰도: {result.no_rag_confidence:.2f}, "
              f"{'✓ 정답' if no_rag_correct else '✗ 오답'})")
        print(f"  With RAG: {result.rag_state} "
              f"(신뢰도: {result.rag_confidence:.2f}, "
              f"{'✓ 정답' if rag_correct else '✗ 오답'})")
        print(f"  RAG 검색: {retrieval_time:.1f}ms "
              f"({result.num_similar_frames}개 프레임, "
              f"{result.num_actions}개 작업)")

        return result

    def evaluate_all(
        self,
        cases: List[EvaluationCase]
    ) -> Tuple[List[EvaluationResult], EvaluationSummary]:
        """
        전체 케이스 평가

        Args:
            cases: 평가 케이스 리스트

        Returns:
            (results, summary) 튜플
        """
        print(f"\n{'='*60}")
        print(f"RAG 시스템 평가 시작 ({len(cases)}개 케이스)")
        print(f"{'='*60}")

        results = []

        for i, case in enumerate(cases, 1):
            print(f"\n진행률: {i}/{len(cases)}")
            result = self.evaluate_case(case)
            results.append(result)

        # 요약 생성
        summary = self._compute_summary(results)

        return results, summary

    def _compute_summary(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationSummary:
        """평가 요약 계산"""
        total = len(results)

        # Accuracy
        no_rag_correct = sum(1 for r in results if r.no_rag_correct)
        rag_correct = sum(1 for r in results if r.rag_correct)

        no_rag_accuracy = no_rag_correct / total if total > 0 else 0.0
        rag_accuracy = rag_correct / total if total > 0 else 0.0
        accuracy_improvement = rag_accuracy - no_rag_accuracy

        # Latency
        no_rag_avg_latency = sum(r.no_rag_time_ms for r in results) / total
        rag_avg_latency = sum(r.rag_time_ms for r in results) / total
        latency_overhead = rag_avg_latency - no_rag_avg_latency
        latency_overhead_pct = (latency_overhead / no_rag_avg_latency) * 100

        # Retrieval
        avg_retrieval_time = sum(r.retrieval_time_ms for r in results) / total
        avg_similar_frames = sum(r.num_similar_frames for r in results) / total
        avg_actions = sum(r.num_actions for r in results) / total
        avg_errors = sum(r.num_errors for r in results) / total

        return EvaluationSummary(
            total_cases=total,
            no_rag_accuracy=no_rag_accuracy,
            rag_accuracy=rag_accuracy,
            accuracy_improvement=accuracy_improvement,
            no_rag_avg_latency_ms=no_rag_avg_latency,
            rag_avg_latency_ms=rag_avg_latency,
            latency_overhead_ms=latency_overhead,
            latency_overhead_pct=latency_overhead_pct,
            avg_retrieval_time_ms=avg_retrieval_time,
            avg_similar_frames=avg_similar_frames,
            avg_actions=avg_actions,
            avg_errors=avg_errors
        )

    def close(self):
        """리소스 정리"""
        self.db.close()


def load_evaluation_cases(cases_file: str) -> List[EvaluationCase]:
    """
    평가 케이스 로드

    Args:
        cases_file: 케이스 JSON 파일 경로

    Returns:
        EvaluationCase 리스트
    """
    with open(cases_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cases = []
    for item in data.get("cases", []):
        case = EvaluationCase(
            case_id=item["case_id"],
            screenshot_path=item["screenshot_path"],
            expected_state=item["expected_state"],
            query_text=item.get("query_text"),
            description=item.get("description", "")
        )
        cases.append(case)

    return cases


def print_summary(summary: EvaluationSummary):
    """평가 요약 출력"""
    print(f"\n{'='*60}")
    print("평가 요약")
    print(f"{'='*60}")

    print(f"\n총 케이스 수: {summary.total_cases}")

    print(f"\n[정확도 (Accuracy)]")
    print(f"  Without RAG: {summary.no_rag_accuracy*100:.1f}%")
    print(f"  With RAG: {summary.rag_accuracy*100:.1f}%")
    print(f"  개선: {summary.accuracy_improvement*100:+.1f}%p")

    print(f"\n[지연시간 (Latency)]")
    print(f"  Without RAG: {summary.no_rag_avg_latency_ms:.1f}ms")
    print(f"  With RAG: {summary.rag_avg_latency_ms:.1f}ms")
    print(f"  오버헤드: +{summary.latency_overhead_ms:.1f}ms "
          f"({summary.latency_overhead_pct:+.1f}%)")

    print(f"\n[검색 (Retrieval)]")
    print(f"  평균 검색 시간: {summary.avg_retrieval_time_ms:.1f}ms")
    print(f"  평균 유사 프레임: {summary.avg_similar_frames:.1f}개")
    print(f"  평균 작업 시퀀스: {summary.avg_actions:.1f}개")
    print(f"  평균 에러 패턴: {summary.avg_errors:.1f}개")


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")

    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="평가 케이스 JSON 파일"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="출력 결과 JSON 파일"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="qwen3_vl",
        choices=["qwen3_vl", "kimi_2", "openai_gpt4v"],
        help="VLM 제공자"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="VLM API URL"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API 키"
    )

    parser.add_argument(
        "--db-uri",
        type=str,
        default="mongodb://localhost:27017/",
        help="MongoDB URI"
    )

    args = parser.parse_args()

    # VLM Provider 매핑
    provider_map = {
        "qwen3_vl": VLMProvider.QWEN3_VL,
        "kimi_2": VLMProvider.KIMI_2,
        "openai_gpt4v": VLMProvider.OPENAI_GPT4V
    }

    # 평가 케이스 로드
    cases = load_evaluation_cases(args.cases)
    print(f"[INFO] {len(cases)}개 평가 케이스 로드됨")

    # DB Config
    db_config = DatabaseConfig()
    db_config.mongo_uri = args.db_uri

    # 평가 실행
    try:
        evaluator = RAGEvaluator(
            vlm_provider=provider_map[args.provider],
            api_url=args.api_url,
            api_key=args.api_key,
            db_config=db_config
        )

        results, summary = evaluator.evaluate_all(cases)

        # 요약 출력
        print_summary(summary)

        # 결과 저장
        output_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results]
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] 결과 저장됨: {args.output}")

        evaluator.close()

    except Exception as e:
        print(f"\n[ERROR] 평가 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
