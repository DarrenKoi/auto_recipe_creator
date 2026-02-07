"""
Home Study Demo - GUI 자동화 학습

Hugging Face 무료 API를 사용하여 집에서 GUI 자동화를 학습합니다.
GPU 없이 동작하며, 회사 API 없이도 테스트할 수 있습니다.

Usage:
    # 환경 설정 (최초 1회)
    export HF_TOKEN="hf_xxxx"

    # 기본 데모 (화면 분석)
    uv run python -m poc.home.demo

    # 특정 모드 실행
    uv run python -m poc.home.demo --mode screen_analysis
    uv run python -m poc.home.demo --mode object_detection
    uv run python -m poc.home.demo --mode ui_elements
    uv run python -m poc.home.demo --mode interactive
"""

import sys
import os
import time
import json
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Optional

# Windows 콘솔 인코딩 문제 해결
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from test.vlm_input_control import ScreenCapture, MouseController, KeyboardController
from poc.home.hf_vlm import HuggingFaceVLM, HFModel, VLMResponse


@dataclass
class DemoMetrics:
    """데모 성능 메트릭"""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: List[float] = field(default_factory=list)

    def add(self, latency_ms: float, success: bool):
        self.total_requests += 1
        self.latencies.append(latency_ms)
        if success:
            self.successful += 1
        else:
            self.failed += 1

    def print_summary(self):
        print("\n" + "=" * 60)
        print("📊 데모 결과 요약")
        print("=" * 60)
        print(f"총 요청 수: {self.total_requests}")
        print(f"성공: {self.successful}")
        print(f"실패: {self.failed}")
        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"평균 응답 시간: {avg:.0f}ms")
            print(f"최소: {min(self.latencies):.0f}ms")
            print(f"최대: {max(self.latencies):.0f}ms")
        print("=" * 60)


class HomeAutomationDemo:
    """집에서 GUI 자동화를 학습하기 위한 데모"""

    def __init__(
        self,
        model: HFModel = HFModel.QWEN2_VL_7B,
        safe_mode: bool = True
    ):
        """
        Args:
            model: 사용할 HuggingFace 모델
            safe_mode: True면 실제 마우스/키보드 제어 안 함
        """
        self.safe_mode = safe_mode
        self.metrics = DemoMetrics()

        # 모듈 초기화
        self.screen = ScreenCapture()
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        print(f"\n[INFO] Home Automation Demo 초기화")
        print(f"[INFO] Safe Mode: {safe_mode}")

        # VLM 초기화
        try:
            self.vlm = HuggingFaceVLM(model=model)
        except ImportError as e:
            print(f"\n[ERROR] {e}")
            print("\n[해결 방법]")
            print("  1. uv sync --extra home")
            print("  2. export HF_TOKEN='your_token'")
            sys.exit(1)

    def _capture_screen(self) -> bytes:
        """화면 캡처 후 PNG 바이트로 반환"""
        # ScreenCapture.capture_full_screen()은 이미 PNG 바이트를 반환
        screenshot_bytes = self.screen.capture_full_screen(save=False)
        return screenshot_bytes

    def demo_screen_analysis(self):
        """
        데모 1: 화면 분석

        현재 화면을 캡처하고 VLM으로 분석합니다.
        """
        print("\n" + "=" * 60)
        print("🔍 데모 1: 화면 분석")
        print("=" * 60)

        print("\n[1/3] 화면 캡처 중...")
        start = time.time()
        image = self._capture_screen()
        capture_time = (time.time() - start) * 1000
        print(f"[INFO] 캡처 완료 ({capture_time:.0f}ms, {len(image)/1024:.1f}KB)")

        print("\n[2/3] VLM 분석 중... (첫 요청은 모델 로딩으로 느릴 수 있음)")
        response = self.vlm.analyze_screen(
            image,
            "이 화면을 분석해주세요. 어떤 애플리케이션이고, 무엇을 하고 있는지 설명해주세요."
        )

        print(f"\n[3/3] 분석 결과 ({response.latency_ms:.0f}ms):")
        print("-" * 60)
        if response.success:
            print(response.content)
        else:
            print(f"[ERROR] {response.error}")
        print("-" * 60)

        self.metrics.add(response.latency_ms, response.success)

    def demo_ui_elements(self):
        """
        데모 2: UI 요소 분석

        화면에서 클릭 가능한 UI 요소를 찾습니다.
        """
        print("\n" + "=" * 60)
        print("🎯 데모 2: UI 요소 분석")
        print("=" * 60)

        print("\n[1/3] 화면 캡처 중...")
        image = self._capture_screen()

        print("\n[2/3] UI 요소 분석 중...")
        response = self.vlm.analyze_ui_elements(image, return_json=True)

        print(f"\n[3/3] 분석 결과 ({response.latency_ms:.0f}ms):")
        print("-" * 60)

        if response.success:
            try:
                # JSON 파싱 시도
                content = response.content
                # JSON 블록 추출
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    content = content[start:end].strip()

                data = json.loads(content)
                print(f"화면 유형: {data.get('screen_type', 'N/A')}")

                elements = data.get('ui_elements', [])
                print(f"\nUI 요소 ({len(elements)}개):")
                for i, elem in enumerate(elements[:10], 1):  # 최대 10개
                    print(f"  {i}. [{elem.get('type', '?')}] {elem.get('name', 'N/A')}")
                    if elem.get('text'):
                        print(f"      텍스트: {elem.get('text')}")

                actions = data.get('possible_actions', [])
                if actions:
                    print(f"\n가능한 액션:")
                    for action in actions[:5]:
                        print(f"  - {action}")

            except json.JSONDecodeError:
                print("[WARN] JSON 파싱 실패, 원본 출력:")
                print(response.content[:500])
        else:
            print(f"[ERROR] {response.error}")

        print("-" * 60)
        self.metrics.add(response.latency_ms, response.success)

    def demo_object_detection(self):
        """
        데모 3: 객체 탐지

        DETR 모델을 사용하여 이미지에서 객체를 탐지합니다.
        (참고: 일반 객체용, UI 요소 탐지에는 제한적)
        """
        print("\n" + "=" * 60)
        print("📦 데모 3: 객체 탐지 (DETR)")
        print("=" * 60)

        print("\n[1/2] 화면 캡처 중...")
        image = self._capture_screen()

        print("\n[2/2] 객체 탐지 중...")
        start = time.time()
        objects = self.vlm.detect_objects(image)
        latency = (time.time() - start) * 1000

        print(f"\n탐지 결과 ({latency:.0f}ms):")
        print("-" * 60)
        if objects:
            for i, obj in enumerate(objects[:15], 1):
                print(f"  {i}. {obj.label} (신뢰도: {obj.score:.2%})")
                print(f"      위치: {obj.bbox}")
        else:
            print("  탐지된 객체 없음 (UI 요소 탐지에는 VLM 분석이 더 효과적)")
        print("-" * 60)

        self.metrics.add(latency, len(objects) > 0)

    def demo_interactive(self):
        """
        데모 4: 대화형 화면 분석

        사용자가 질문을 입력하면 현재 화면에 대해 답변합니다.
        """
        print("\n" + "=" * 60)
        print("💬 데모 4: 대화형 화면 분석")
        print("=" * 60)
        print("\n현재 화면에 대해 질문해보세요.")
        print("'q' 또는 'quit'을 입력하면 종료합니다.\n")

        while True:
            try:
                question = input("질문: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if question.lower() in ['q', 'quit', 'exit']:
                print("종료합니다.")
                break

            if not question:
                continue

            print("\n[INFO] 화면 캡처 및 분석 중...")
            image = self._capture_screen()
            response = self.vlm.analyze_screen(image, question)

            print(f"\n답변 ({response.latency_ms:.0f}ms):")
            print("-" * 40)
            if response.success:
                print(response.content)
            else:
                print(f"[ERROR] {response.error}")
            print("-" * 40 + "\n")

            self.metrics.add(response.latency_ms, response.success)

    def run_all_demos(self):
        """모든 데모 순차 실행"""
        print("\n" + "=" * 60)
        print("🏠 Home Study Demo - GUI 자동화 학습")
        print("=" * 60)
        print("\nHugging Face 무료 API를 사용하여 화면 분석을 테스트합니다.")
        print("첫 번째 요청은 모델 로딩으로 인해 느릴 수 있습니다.\n")

        input("준비되면 Enter를 누르세요...")

        # 데모 실행
        self.demo_screen_analysis()
        self.demo_ui_elements()

        # 최종 요약
        self.metrics.print_summary()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Home GUI Automation Demo")
    parser.add_argument(
        "--mode",
        choices=["all", "screen_analysis", "ui_elements", "object_detection", "interactive"],
        default="all",
        help="실행할 데모 모드"
    )
    parser.add_argument(
        "--model",
        choices=["qwen2_vl_7b", "qwen2_vl_2b", "llava"],
        default="qwen2_vl_7b",
        help="사용할 VLM 모델"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live mode (실제 마우스/키보드 제어)"
    )

    args = parser.parse_args()

    # 모델 매핑
    model_map = {
        "qwen2_vl_7b": HFModel.QWEN2_VL_7B,
        "qwen2_vl_2b": HFModel.QWEN2_VL_2B,
        "llava": HFModel.LLAVA_1_5_7B,
    }

    demo = HomeAutomationDemo(
        model=model_map[args.model],
        safe_mode=not args.live
    )

    # 모드별 실행
    if args.mode == "all":
        demo.run_all_demos()
    elif args.mode == "screen_analysis":
        demo.demo_screen_analysis()
        demo.metrics.print_summary()
    elif args.mode == "ui_elements":
        demo.demo_ui_elements()
        demo.metrics.print_summary()
    elif args.mode == "object_detection":
        demo.demo_object_detection()
        demo.metrics.print_summary()
    elif args.mode == "interactive":
        demo.demo_interactive()
        demo.metrics.print_summary()


if __name__ == "__main__":
    main()
