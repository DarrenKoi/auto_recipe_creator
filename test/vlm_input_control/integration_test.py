"""
Integration Test Module

VLM 화면 분석과 마우스/키보드 제어를 통합하여 테스트합니다.
실제 자동화 시나리오를 시뮬레이션하여 전체 파이프라인을 검증합니다.

테스트 케이스:
- TC-01: VM 창 캡처
- TC-02: 상태 매칭 (정상)
- TC-04: 클릭 전달
- NFR-01: 단일 액션 실행 속도 < 200ms
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple
from dataclasses import dataclass

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screen_capture import ScreenCapture, MSS_AVAILABLE
from mouse_control import MouseController, MouseButton, PYNPUT_MOUSE_AVAILABLE
from keyboard_control import KeyboardController, PYNPUT_KEYBOARD_AVAILABLE
from vlm_screen_analysis import VLMScreenAnalyzer, VLMProvider, ScreenAnalysisResult


@dataclass
class TestResult:
    """테스트 결과"""
    test_id: str
    test_name: str
    passed: bool
    message: str
    elapsed_ms: float


class IntegrationTester:
    """통합 테스트 클래스"""

    def __init__(
        self,
        vlm_provider: VLMProvider = VLMProvider.LOCAL,
        vlm_api_url: Optional[str] = None,
        output_dir: str = "./test_outputs"
    ):
        """
        Args:
            vlm_provider: VLM 제공자
            vlm_api_url: VLM API URL
            output_dir: 테스트 출력 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 컴포넌트 초기화
        self.screen_capture = ScreenCapture(output_dir=os.path.join(output_dir, "captures"))
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.vlm_analyzer = VLMScreenAnalyzer(
            provider=vlm_provider,
            api_base_url=vlm_api_url
        )

        self.test_results: list[TestResult] = []

        print("[INFO] IntegrationTester 초기화 완료")
        print(f"  - Screen Capture: {'OK' if MSS_AVAILABLE else 'N/A'}")
        print(f"  - Mouse Control: {'OK' if PYNPUT_MOUSE_AVAILABLE else 'N/A'}")
        print(f"  - Keyboard Control: {'OK' if PYNPUT_KEYBOARD_AVAILABLE else 'N/A'}")
        print(f"  - VLM Analyzer: {vlm_provider.value}")

    def run_all_tests(self, safe_mode: bool = True) -> list[TestResult]:
        """
        모든 테스트를 실행합니다.

        Args:
            safe_mode: True면 실제 입력 동작을 수행하지 않음

        Returns:
            테스트 결과 리스트
        """
        print("\n" + "=" * 70)
        print("Integration Test Suite - VLM + Input Control")
        print("=" * 70)

        if safe_mode:
            print("[MODE] Safe Mode - 실제 마우스/키보드 입력은 비활성화됩니다.")
        else:
            print("[MODE] Live Mode - 실제 마우스/키보드 입력이 수행됩니다!")
            print("[WARNING] 3초 후 테스트가 시작됩니다. 취소하려면 Ctrl+C를 누르세요.")
            time.sleep(3)

        # 테스트 실행
        self.test_tc01_screen_capture()
        self.test_tc02_state_matching()
        self.test_tc04_click_accuracy(safe_mode)
        self.test_nfr01_action_speed(safe_mode)
        self.test_pipeline_capture_analyze_act(safe_mode)

        # 결과 요약
        self._print_summary()

        return self.test_results

    def test_tc01_screen_capture(self):
        """TC-01: VM 창 캡처 테스트"""
        print("\n" + "-" * 50)
        print("[TC-01] VM 창 캡처 테스트")
        print("-" * 50)

        start_time = time.time()

        if not MSS_AVAILABLE:
            result = TestResult(
                test_id="TC-01",
                test_name="VM 창 캡처",
                passed=False,
                message="mss 라이브러리 없음",
                elapsed_ms=0
            )
            self.test_results.append(result)
            print(f"[SKIP] {result.message}")
            return

        # 화면 캡처 수행
        image_data = self.screen_capture.capture_full_screen(save=True)
        elapsed_ms = (time.time() - start_time) * 1000

        # 검증
        passed = image_data is not None and len(image_data) > 0 and elapsed_ms < 50

        if image_data:
            message = f"캡처 성공 - 크기: {len(image_data)} bytes, 지연: {elapsed_ms:.2f}ms"
        else:
            message = "캡처 실패"

        result = TestResult(
            test_id="TC-01",
            test_name="VM 창 캡처",
            passed=passed,
            message=message,
            elapsed_ms=elapsed_ms
        )
        self.test_results.append(result)

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {message}")

        if not passed and elapsed_ms >= 50:
            print(f"  기준: 지연 < 50ms, 실제: {elapsed_ms:.2f}ms")

    def test_tc02_state_matching(self):
        """TC-02: 상태 매칭 테스트"""
        print("\n" + "-" * 50)
        print("[TC-02] 상태 매칭 테스트")
        print("-" * 50)

        start_time = time.time()

        # 화면 캡처
        if MSS_AVAILABLE:
            image_data = self.screen_capture.capture_full_screen(save=False)
        else:
            # Mock 이미지
            image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'

        # VLM 분석
        result = self.vlm_analyzer.analyze_screen(image_data)
        elapsed_ms = (time.time() - start_time) * 1000

        # 검증 (confidence > 0.85)
        passed = result is not None and result.confidence > 0.85

        if result:
            message = f"상태: {result.state_id}, 확신도: {result.confidence:.2f}"
        else:
            message = "분석 실패"

        test_result = TestResult(
            test_id="TC-02",
            test_name="상태 매칭",
            passed=passed,
            message=message,
            elapsed_ms=elapsed_ms
        )
        self.test_results.append(test_result)

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {message}")

        if result and not passed:
            print(f"  기준: confidence > 0.85, 실제: {result.confidence:.2f}")

    def test_tc04_click_accuracy(self, safe_mode: bool = True):
        """TC-04: 클릭 전달 정확도 테스트"""
        print("\n" + "-" * 50)
        print("[TC-04] 클릭 전달 정확도 테스트")
        print("-" * 50)

        if not PYNPUT_MOUSE_AVAILABLE:
            result = TestResult(
                test_id="TC-04",
                test_name="클릭 전달",
                passed=False,
                message="pynput 라이브러리 없음",
                elapsed_ms=0
            )
            self.test_results.append(result)
            print(f"[SKIP] {result.message}")
            return

        start_time = time.time()

        # 테스트 좌표
        target_x, target_y = 100, 100

        # 마우스 이동
        self.mouse.move_to(target_x, target_y)
        time.sleep(0.1)

        # 위치 확인
        actual_x, actual_y = self.mouse.position

        # 정확도 계산 (기준: ±5px)
        error_x = abs(actual_x - target_x)
        error_y = abs(actual_y - target_y)

        elapsed_ms = (time.time() - start_time) * 1000

        passed = error_x <= 5 and error_y <= 5
        message = f"목표: ({target_x}, {target_y}), 실제: ({actual_x}, {actual_y}), 오차: (±{error_x}, ±{error_y})px"

        result = TestResult(
            test_id="TC-04",
            test_name="클릭 전달",
            passed=passed,
            message=message,
            elapsed_ms=elapsed_ms
        )
        self.test_results.append(result)

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {message}")

        # 실제 클릭은 safe_mode에서 수행하지 않음
        if not safe_mode:
            print("  실제 클릭 수행 중...")
            self.mouse.click()

    def test_nfr01_action_speed(self, safe_mode: bool = True):
        """NFR-01: 단일 액션 실행 속도 테스트"""
        print("\n" + "-" * 50)
        print("[NFR-01] 단일 액션 실행 속도 테스트")
        print("-" * 50)

        if not PYNPUT_MOUSE_AVAILABLE:
            result = TestResult(
                test_id="NFR-01",
                test_name="단일 액션 속도",
                passed=False,
                message="pynput 라이브러리 없음",
                elapsed_ms=0
            )
            self.test_results.append(result)
            print(f"[SKIP] {result.message}")
            return

        # 여러 번 측정하여 평균 계산
        measurements = []
        num_trials = 5

        for i in range(num_trials):
            start_time = time.time()

            # 마우스 이동 + 클릭 시뮬레이션
            self.mouse.move_to(150 + i * 10, 150 + i * 10)

            if not safe_mode:
                self.mouse.click()

            elapsed_ms = (time.time() - start_time) * 1000
            measurements.append(elapsed_ms)

        avg_time = sum(measurements) / len(measurements)
        max_time = max(measurements)

        # 기준: < 200ms (VLM 미사용 시)
        passed = avg_time < 200
        message = f"평균: {avg_time:.2f}ms, 최대: {max_time:.2f}ms (기준: < 200ms)"

        result = TestResult(
            test_id="NFR-01",
            test_name="단일 액션 속도",
            passed=passed,
            message=message,
            elapsed_ms=avg_time
        )
        self.test_results.append(result)

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {message}")

    def test_pipeline_capture_analyze_act(self, safe_mode: bool = True):
        """통합 파이프라인 테스트: 캡처 -> 분석 -> 액션"""
        print("\n" + "-" * 50)
        print("[PIPELINE] 통합 파이프라인 테스트")
        print("-" * 50)

        total_start = time.time()

        # Step 1: 화면 캡처
        print("  Step 1: 화면 캡처...")
        capture_start = time.time()

        if MSS_AVAILABLE:
            image_data = self.screen_capture.capture_full_screen(save=True)
        else:
            image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'

        capture_time = (time.time() - capture_start) * 1000
        print(f"    캡처 완료: {capture_time:.2f}ms")

        # Step 2: VLM 분석
        print("  Step 2: VLM 분석...")
        analyze_start = time.time()
        analysis_result = self.vlm_analyzer.analyze_screen(image_data)
        analyze_time = (time.time() - analyze_start) * 1000

        if analysis_result:
            print(f"    분석 완료: {analyze_time:.2f}ms")
            print(f"    상태: {analysis_result.state_name} (confidence: {analysis_result.confidence:.2f})")
        else:
            print(f"    분석 실패")

        # Step 3: 액션 결정 및 실행
        print("  Step 3: 액션 실행...")
        action_start = time.time()

        if analysis_result and analysis_result.ui_elements:
            # 첫 번째 UI 요소 위치로 이동 (시뮬레이션)
            print(f"    발견된 UI 요소: {len(analysis_result.ui_elements)}개")

            if PYNPUT_MOUSE_AVAILABLE:
                # 화면 중앙으로 이동 (예시)
                self.mouse.move_to(200, 200)

                if not safe_mode:
                    self.mouse.click()
                    print("    클릭 수행됨")

        action_time = (time.time() - action_start) * 1000
        print(f"    액션 완료: {action_time:.2f}ms")

        # 전체 시간
        total_time = (time.time() - total_start) * 1000

        # 결과 기록
        passed = image_data is not None and analysis_result is not None
        message = f"캡처: {capture_time:.1f}ms, 분석: {analyze_time:.1f}ms, 액션: {action_time:.1f}ms, 총: {total_time:.1f}ms"

        result = TestResult(
            test_id="PIPELINE",
            test_name="통합 파이프라인",
            passed=passed,
            message=message,
            elapsed_ms=total_time
        )
        self.test_results.append(result)

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} {message}")

    def _print_summary(self):
        """테스트 결과 요약을 출력합니다."""
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)

        passed_count = sum(1 for r in self.test_results if r.passed)
        total_count = len(self.test_results)

        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.test_id}: {result.test_name}")
            print(f"         {result.message}")

        print("-" * 70)
        print(f"Results: {passed_count}/{total_count} tests passed")

        if passed_count == total_count:
            print("\n[SUCCESS] 모든 테스트 통과!")
        else:
            failed = [r for r in self.test_results if not r.passed]
            print(f"\n[ATTENTION] {len(failed)}개 테스트 실패")
            for f in failed:
                print(f"  - {f.test_id}: {f.test_name}")

        print("=" * 70)

    def cleanup(self):
        """리소스를 정리합니다."""
        self.screen_capture.close()
        print("[INFO] 리소스 정리 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="VLM + Input Control Integration Test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="실제 마우스/키보드 입력을 수행합니다 (주의 필요)"
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default=os.environ.get("VLM_API_BASE_URL"),
        help="VLM API URL (예: http://localhost:8000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_outputs",
        help="테스트 출력 디렉토리"
    )

    args = parser.parse_args()

    # 테스터 초기화
    tester = IntegrationTester(
        vlm_provider=VLMProvider.LOCAL,
        vlm_api_url=args.vlm_url,
        output_dir=args.output_dir
    )

    try:
        # 테스트 실행
        results = tester.run_all_tests(safe_mode=not args.live)

        # 종료 코드 결정
        all_passed = all(r.passed for r in results)
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 테스트가 중단되었습니다.")
        sys.exit(130)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
