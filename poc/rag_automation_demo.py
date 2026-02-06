"""
RAG Automation Demo

RAG 기반 자동화 데모 (With RAG vs Without RAG 비교)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

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


class RAGAutomationDemo:
    """RAG 자동화 데모"""

    def __init__(
        self,
        vlm_provider: VLMProvider,
        api_url: str,
        api_key: Optional[str] = None,
        db_config: Optional[DatabaseConfig] = None,
        use_rag: bool = False
    ):
        """
        Args:
            vlm_provider: VLM 제공자
            api_url: VLM API URL
            api_key: API 키 (옵션)
            db_config: 데이터베이스 설정 (옵션)
            use_rag: RAG 사용 여부
        """
        self.vlm_provider = vlm_provider
        self.use_rag = use_rag

        # 화면 캡처
        self.screen_capture = ScreenCapture()

        # DatabaseHandler 초기화 (RAG 사용 시)
        self.db = None
        self.rag_manager = None
        self.rag_prompt_builder = None

        if use_rag:
            print("\n[INFO] RAG 시스템 초기화 중...")
            try:
                # DatabaseHandler
                config = db_config or DatabaseConfig()
                self.db = DatabaseHandler(config)
                self.db.initialize()
                self.db.init_text_faiss_index()

                # RAGContextManager
                self.rag_manager = create_rag_context_manager(
                    db_handler=self.db,
                    use_text_search=True,
                    use_visual_search=True
                )

                # RAGPromptBuilder
                self.rag_prompt_builder = create_rag_prompt_builder(
                    max_similar_frames=3
                )

                print("[INFO] RAG 시스템 초기화 완료")
            except Exception as e:
                print(f"[WARNING] RAG 시스템 초기화 실패: {e}")
                print("[INFO] RAG 없이 진행합니다")
                use_rag = False

        # VLM Analyzer
        self.vlm_analyzer = VLMScreenAnalyzer(
            provider=vlm_provider,
            api_base_url=api_url,
            api_key=api_key,
            rag_manager=self.rag_manager,
            rag_prompt_builder=self.rag_prompt_builder,
            use_rag=use_rag
        )

        print(f"\n[INFO] RAG Automation Demo 초기화 완료")
        print(f"[INFO] VLM Provider: {vlm_provider.value}")
        print(f"[INFO] RAG: {'활성화' if use_rag else '비활성화'}")

    def run_screen_analysis(
        self,
        query_text: Optional[str] = None,
        region: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        화면 분석 실행

        Args:
            query_text: 쿼리 텍스트 (RAG 검색용)
            region: 캡처 영역 (x, y, width, height)

        Returns:
            분석 결과 딕셔너리
        """
        print("\n" + "="*60)
        print("화면 분석 시작")
        print("="*60)

        start_time = time.time()

        # 1. 화면 캡처
        print("\n[1] 화면 캡처 중...")
        if region:
            screenshot = self.screen_capture.capture_region(*region)
        else:
            screenshot = self.screen_capture.capture_screen()

        capture_time = time.time() - start_time
        print(f"[INFO] 캡처 완료 (소요 시간: {capture_time*1000:.1f}ms)")

        # 2. VLM 분석
        print("\n[2] VLM 분석 중...")
        analysis_start = time.time()

        result = self.vlm_analyzer.analyze_screen(
            image_data=screenshot,
            query_text=query_text
        )

        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time

        # 3. 결과 출력
        print("\n" + "="*60)
        print("분석 결과")
        print("="*60)

        if result:
            print(f"상태 ID: {result.state_id}")
            print(f"상태 이름: {result.state_name}")
            print(f"신뢰도: {result.confidence:.2f}")
            print(f"설명: {result.description}")

            if result.ui_elements:
                print(f"\nUI 요소 ({len(result.ui_elements)}개):")
                for elem in result.ui_elements[:5]:
                    print(f"  - {elem.get('name')}: {elem.get('type')}")

            if result.suggested_actions:
                print(f"\n제안 작업 ({len(result.suggested_actions)}개):")
                for action in result.suggested_actions[:5]:
                    print(f"  - {action}")
        else:
            print("분석 실패")

        print("\n" + "="*60)
        print("성능 지표")
        print("="*60)
        print(f"화면 캡처: {capture_time*1000:.1f}ms")
        print(f"VLM 분석: {analysis_time*1000:.1f}ms")
        print(f"총 소요 시간: {total_time*1000:.1f}ms")

        return {
            "success": result is not None,
            "result": result,
            "capture_time_ms": capture_time * 1000,
            "analysis_time_ms": analysis_time * 1000,
            "total_time_ms": total_time * 1000
        }

    def run_comparison(
        self,
        query_text: Optional[str] = None,
        region: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        With RAG vs Without RAG 비교 실행

        Args:
            query_text: 쿼리 텍스트
            region: 캡처 영역

        Returns:
            비교 결과 딕셔너리
        """
        print("\n" + "="*60)
        print("RAG 비교 분석")
        print("="*60)

        # Without RAG
        print("\n[TEST 1] RAG 없이 분석...")
        self.vlm_analyzer.use_rag = False
        result_no_rag = self.run_screen_analysis(query_text, region)

        # With RAG
        if self.rag_manager:
            print("\n[TEST 2] RAG 사용하여 분석...")
            self.vlm_analyzer.use_rag = True
            result_with_rag = self.run_screen_analysis(query_text, region)
        else:
            print("\n[WARNING] RAG 시스템을 사용할 수 없습니다")
            result_with_rag = None

        # 비교
        print("\n" + "="*60)
        print("비교 결과")
        print("="*60)

        if result_with_rag:
            print(f"\nWithout RAG:")
            print(f"  - 총 소요 시간: {result_no_rag['total_time_ms']:.1f}ms")
            print(f"  - 신뢰도: {result_no_rag['result'].confidence:.2f}")

            print(f"\nWith RAG:")
            print(f"  - 총 소요 시간: {result_with_rag['total_time_ms']:.1f}ms")
            print(f"  - 신뢰도: {result_with_rag['result'].confidence:.2f}")

            overhead = result_with_rag['total_time_ms'] - result_no_rag['total_time_ms']
            overhead_pct = (overhead / result_no_rag['total_time_ms']) * 100

            print(f"\nRAG 오버헤드:")
            print(f"  - 절대값: +{overhead:.1f}ms")
            print(f"  - 상대값: +{overhead_pct:.1f}%")

        return {
            "without_rag": result_no_rag,
            "with_rag": result_with_rag
        }

    def close(self):
        """리소스 정리"""
        if self.db:
            self.db.close()
            print("[INFO] DatabaseHandler 종료")


def main():
    parser = argparse.ArgumentParser(description="RAG Automation Demo")

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
        "--use-rag",
        action="store_true",
        help="RAG 사용"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="With RAG vs Without RAG 비교 실행"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="쿼리 텍스트"
    )

    parser.add_argument(
        "--region",
        type=str,
        help="캡처 영역 (x,y,width,height)"
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

    # Region 파싱
    region = None
    if args.region:
        try:
            x, y, w, h = map(int, args.region.split(','))
            region = (x, y, w, h)
        except:
            print(f"[ERROR] 잘못된 region 형식: {args.region}")
            sys.exit(1)

    # DB Config
    db_config = DatabaseConfig()
    db_config.mongo_uri = args.db_uri

    # Demo 실행
    try:
        demo = RAGAutomationDemo(
            vlm_provider=provider_map[args.provider],
            api_url=args.api_url,
            api_key=args.api_key,
            db_config=db_config,
            use_rag=args.use_rag
        )

        if args.compare:
            # 비교 모드
            demo.run_comparison(query_text=args.query, region=region)
        else:
            # 단일 분석 모드
            demo.run_screen_analysis(query_text=args.query, region=region)

        demo.close()

    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
