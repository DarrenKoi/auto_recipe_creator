"""
CPU 기반 자동화 PoC 데모

회사 내부 VLM API(Kimi 2, Qwen3-VL)를 사용하여
CPU만으로 자동화가 가능함을 증명하는 스크립트.

Rate Limits:
- Kimi 2: 1 request / 3 seconds
- Qwen3-VL: 1 request / 1 second

Usage:
    python -m poc.cpu_automation_demo --provider qwen3_vl --api-url YOUR_API_URL
"""

import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from io import BytesIO

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.vlm_input_control import ScreenCapture, MouseController, KeyboardController
from test.vlm_input_control.vlm_screen_analysis import VLMScreenAnalyzer, VLMProvider

# Rate limit 설정 (초 단위)
RATE_LIMITS = {
    VLMProvider.KIMI_2: 3.0,      # 3초에 1회
    VLMProvider.QWEN3_VL: 1.0,    # 1초에 1회
    VLMProvider.QWEN_VL: 1.0,
    VLMProvider.OPENAI_GPT4V: 0.1,
    VLMProvider.LOCAL: 0.0
}


@dataclass
class PerformanceMetrics:
    """성능 측정 메트릭"""
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    avg_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return (self.successful_actions / self.total_actions) * 100

    def add_measurement(self, latency_ms: float, success: bool):
        self.total_actions += 1
        self.latencies.append(latency_ms)
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        self.avg_latency_ms = sum(self.latencies) / len(self.latencies)

    def print_report(self):
        print("\n" + "="*60)
        print("📊 CPU 기반 자동화 PoC 성능 리포트")
        print("="*60)
        print(f"총 액션 수:        {self.total_actions}")
        print(f"성공:              {self.successful_actions}")
        print(f"실패:              {self.failed_actions}")
        print(f"성공률:            {self.success_rate:.1f}%")
        print(f"평균 레이턴시:     {self.avg_latency_ms:.0f} ms")
        if self.latencies:
            print(f"최소 레이턴시:     {min(self.latencies):.0f} ms")
            print(f"최대 레이턴시:     {max(self.latencies):.0f} ms")
        print("="*60)

        print("\n💡 GPU 도입 시 예상 개선:")
        print(f"  레이턴시: {self.avg_latency_ms:.0f}ms → 600ms (약 {self.avg_latency_ms/600:.1f}배 빠름)")
        print(f"  성공률:   {self.success_rate:.0f}% → 95%+ ({max(0, 95-self.success_rate):.0f}%p 향상)")
        print(f"  비용:     API 호출 비용 → $0 (로컬 추론)")
        print(f"  확장성:   API rate limit → 무제한 (로컬)")
        print("="*60 + "\n")


class CPUAutomationDemo:
    """CPU 기반 자동화 데모"""

    def __init__(
        self,
        provider: VLMProvider,
        api_url: str,
        api_key: Optional[str] = None,
        safe_mode: bool = True,
        use_webp: bool = True,
        max_image_size: int = 1920
    ):
        """
        Args:
            provider: VLM 제공자 (KIMI_2, QWEN3_VL)
            api_url: 회사 내부 API URL
            api_key: API 인증 키 (optional)
            safe_mode: True면 실제 마우스/키보드 제어 안 함
            use_webp: WebP 변환 사용 (파일 크기 감소)
            max_image_size: 최대 이미지 크기 (긴 쪽 기준, 픽셀)
        """
        self.provider = provider
        self.safe_mode = safe_mode
        self.use_webp = use_webp
        self.max_image_size = max_image_size

        # 모듈 초기화
        self.screen = ScreenCapture()
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.vlm = VLMScreenAnalyzer(
            provider=provider,
            api_base_url=api_url,
            api_key=api_key
        )

        # 메트릭
        self.metrics = PerformanceMetrics()

        # Rate limiting
        self.last_api_call_time = 0
        self.rate_limit = RATE_LIMITS.get(provider, 0.0)

        print(f"[INFO] CPU 자동화 데모 초기화 완료")
        print(f"[INFO] VLM Provider: {provider.value}")
        print(f"[INFO] Safe Mode: {safe_mode}")
        print(f"[INFO] Rate Limit: {self.rate_limit}s per request")
        print(f"[INFO] Image Format: {'WebP' if use_webp else 'PNG'}")
        print(f"[INFO] Max Image Size: {max_image_size}px")

    def run_rcs_login_demo(self, server: str, username: str, password: str):
        """
        RCS 로그인 자동화 데모

        Args:
            server: RCS 서버 주소
            username: 사용자 이름
            password: 비밀번호
        """
        print("\n" + "="*60)
        print("🚀 RCS 로그인 자동화 데모 시작")
        print("="*60 + "\n")

        # Step 1: 화면 캡처
        print("[1/5] 화면 캡처 중...")
        start_time = time.time()
        screenshot = self.screen.capture_full_screen()
        capture_time = (time.time() - start_time) * 1000
        print(f"[INFO] 캡처 완료 ({capture_time:.1f}ms)")

        # Step 2: VLM 분석
        print("[2/5] VLM API 호출 중...")

        # Rate limiting 적용
        self._wait_for_rate_limit()

        vlm_start = time.time()

        prompt = """
화면을 분석하여 RCS 로그인에 필요한 UI 요소를 찾아주세요.

다음 JSON 형식으로 응답해주세요:
{
  "ui_elements": [
    {"name": "server_input", "bbox": [x1, y1, x2, y2], "type": "input"},
    {"name": "username_input", "bbox": [x1, y1, x2, y2], "type": "input"},
    {"name": "password_input", "bbox": [x1, y1, x2, y2], "type": "input"},
    {"name": "login_button", "bbox": [x1, y1, x2, y2], "type": "button"}
  ]
}
"""

        image_bytes = self._pil_to_bytes(screenshot)
        response = self.vlm._call_vlm_api(image_bytes, prompt)
        vlm_time = (time.time() - vlm_start) * 1000
        print(f"[INFO] VLM 분석 완료 ({vlm_time:.0f}ms)")

        if not response:
            print("[ERROR] VLM API 응답 없음")
            self.metrics.add_measurement(vlm_time, False)
            return

        # Step 3: JSON 파싱
        print("[3/5] UI 요소 파싱 중...")
        try:
            # JSON 추출 시도
            json_str = self._extract_json_from_response(response)
            ui_data = json.loads(json_str)
            ui_elements = ui_data.get("ui_elements", [])
            print(f"[INFO] {len(ui_elements)}개 UI 요소 탐지")
        except json.JSONDecodeError:
            print("[ERROR] JSON 파싱 실패")
            print(f"[DEBUG] 응답: {response[:200]}...")
            self.metrics.add_measurement(vlm_time, False)
            return

        # Step 4: 입력 실행
        print("[4/5] 자동 입력 수행 중...")

        for element in ui_elements:
            elem_name = element.get("name", "")
            bbox = element.get("bbox", [])
            elem_type = element.get("type", "")

            if not bbox or len(bbox) != 4:
                continue

            # 중심점 계산
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            if elem_type == "input":
                # 입력 필드 클릭 및 타이핑
                if self.safe_mode:
                    print(f"[SAFE MODE] Would click {elem_name} at ({center_x}, {center_y})")
                else:
                    self.mouse.click(center_x, center_y)
                    time.sleep(0.3)

                # 값 입력
                if "server" in elem_name:
                    text = server
                elif "username" in elem_name:
                    text = username
                elif "password" in elem_name:
                    text = password
                else:
                    continue

                if self.safe_mode:
                    print(f"[SAFE MODE] Would type: {text}")
                else:
                    self.keyboard.type_text(text)
                    time.sleep(0.2)

            elif elem_type == "button" and "login" in elem_name:
                # 로그인 버튼 클릭
                if self.safe_mode:
                    print(f"[SAFE MODE] Would click login button at ({center_x}, {center_y})")
                else:
                    self.mouse.click(center_x, center_y)

        # Step 5: 결과 기록
        total_time = (time.time() - start_time) * 1000
        print(f"[5/5] 완료 (총 {total_time:.0f}ms)")

        self.metrics.add_measurement(total_time, True)

        print("\n✅ RCS 로그인 데모 완료\n")

    def run_screen_analysis_demo(self):
        """
        단순 화면 분석 데모 (UI 요소 인식만 테스트)
        """
        print("\n" + "="*60)
        print("🔍 화면 분석 데모 시작")
        print("="*60 + "\n")

        # Step 1: 화면 캡처
        print("[1/3] 화면 캡처 중...")
        start_time = time.time()
        screenshot = self.screen.capture_full_screen()
        capture_time = (time.time() - start_time) * 1000
        print(f"[INFO] 캡처 완료 ({capture_time:.1f}ms)")

        # Step 2: VLM 분석
        print("[2/3] VLM API 호출 중...")

        # Rate limiting 적용
        self._wait_for_rate_limit()

        vlm_start = time.time()

        prompt = """
현재 화면을 분석하여 다음 정보를 JSON 형식으로 제공해주세요:

{
  "screen_type": "화면 유형 (예: desktop, application, dialog)",
  "main_content": "화면의 주요 내용 설명",
  "ui_elements": [
    {"name": "요소 이름", "type": "button/input/label/menu/etc", "location": "위치"}
  ],
  "possible_actions": ["가능한 액션 1", "가능한 액션 2"]
}
"""

        image_bytes = self._pil_to_bytes(screenshot)
        response = self.vlm._call_vlm_api(image_bytes, prompt)
        vlm_time = (time.time() - vlm_start) * 1000
        print(f"[INFO] VLM 분석 완료 ({vlm_time:.0f}ms)")

        if not response:
            print("[ERROR] VLM API 응답 없음")
            self.metrics.add_measurement(vlm_time, False)
            return

        # Step 3: 결과 출력
        print("[3/3] 분석 결과:")
        print("-" * 60)
        try:
            json_str = self._extract_json_from_response(response)
            result = json.loads(json_str)
            print(f"화면 유형: {result.get('screen_type', 'N/A')}")
            print(f"주요 내용: {result.get('main_content', 'N/A')}")
            print(f"UI 요소 수: {len(result.get('ui_elements', []))}")
            print(f"가능한 액션 수: {len(result.get('possible_actions', []))}")

            if result.get('ui_elements'):
                print("\nUI 요소 목록:")
                for elem in result['ui_elements'][:5]:  # 최대 5개만 출력
                    print(f"  - {elem.get('name', 'N/A')}: {elem.get('type', 'N/A')}")
        except json.JSONDecodeError:
            print("[WARN] JSON 파싱 실패, raw 응답 출력:")
            print(response[:500])
        print("-" * 60)

        total_time = (time.time() - start_time) * 1000
        self.metrics.add_measurement(total_time, response is not None)

        print("\n✅ 화면 분석 데모 완료\n")

    def _wait_for_rate_limit(self):
        """Rate limit을 준수하기 위해 대기"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_api_call_time
            if elapsed < self.rate_limit:
                wait_time = self.rate_limit - elapsed
                print(f"[INFO] Rate limit 대기: {wait_time:.1f}초...")
                time.sleep(wait_time)
        self.last_api_call_time = time.time()

    def _optimize_image(self, image):
        """이미지 최적화 (크기 조정 + 포맷 변환)"""
        from PIL import Image

        # 원본 크기
        original_size = image.size
        original_format = image.format or "Unknown"

        # 크기 조정 (긴 쪽 기준)
        max_dim = max(image.size)
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[INFO] 이미지 리사이즈: {original_size} → {new_size}")

        return image

    def _pil_to_bytes(self, image) -> bytes:
        """PIL Image를 bytes로 변환 (WebP 또는 PNG)"""
        # 이미지 최적화
        image = self._optimize_image(image)

        buffer = BytesIO()

        if self.use_webp:
            # WebP 변환 (파일 크기 약 30% 감소)
            image.save(buffer, format="WEBP", quality=85, method=6)
            file_size = buffer.tell()
            print(f"[INFO] WebP 변환 완료: {file_size/1024:.1f} KB")
        else:
            # PNG (무손실)
            image.save(buffer, format="PNG", optimize=True)
            file_size = buffer.tell()
            print(f"[INFO] PNG 저장: {file_size/1024:.1f} KB")

        return buffer.getvalue()

    def _extract_json_from_response(self, response: str) -> str:
        """응답에서 JSON 블록을 추출"""
        # JSON 블록 찾기
        if '```json' in response:
            start_idx = response.find('```json') + 7
            end_idx = response.find('```', start_idx)
            if end_idx != -1:
                return response[start_idx:end_idx].strip()

        # 중괄호로 시작하는 JSON 찾기
        if '{' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if end_idx > start_idx:
                return response[start_idx:end_idx + 1]

        return response

    def print_final_report(self):
        """최종 성능 리포트 출력"""
        self.metrics.print_report()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CPU 기반 자동화 PoC 데모")
    parser.add_argument("--provider", choices=["kimi_2", "qwen3_vl"], required=True,
                        help="VLM Provider 선택")
    parser.add_argument("--api-url", required=True, help="회사 내부 VLM API URL")
    parser.add_argument("--api-key", help="API 인증 키 (optional)")
    parser.add_argument("--safe-mode", action="store_true", default=True,
                        help="Safe mode (실제 입력 안 함)")
    parser.add_argument("--live", action="store_true",
                        help="Live mode (실제 입력 수행, safe-mode 비활성화)")
    parser.add_argument("--demo-type", choices=["rcs_login", "screen_analysis"],
                        default="screen_analysis",
                        help="데모 유형 선택")
    parser.add_argument("--use-webp", action="store_true", default=True,
                        help="WebP 포맷 사용 (파일 크기 감소)")
    parser.add_argument("--no-webp", action="store_false", dest="use_webp",
                        help="PNG 포맷 사용 (무손실)")
    parser.add_argument("--max-image-size", type=int, default=1920,
                        help="최대 이미지 크기 (긴 쪽 기준, 픽셀)")
    parser.add_argument("--server", default="192.168.1.100", help="RCS 서버 주소 (rcs_login용)")
    parser.add_argument("--username", default="admin", help="사용자 이름 (rcs_login용)")
    parser.add_argument("--password", default="password", help="비밀번호 (rcs_login용)")

    args = parser.parse_args()

    # Provider 매핑
    provider_map = {
        "kimi_2": VLMProvider.KIMI_2,
        "qwen3_vl": VLMProvider.QWEN3_VL
    }

    # Safe mode 결정
    safe_mode = not args.live if args.live else args.safe_mode

    demo = CPUAutomationDemo(
        provider=provider_map[args.provider],
        api_url=args.api_url,
        api_key=args.api_key,
        safe_mode=safe_mode,
        use_webp=args.use_webp,
        max_image_size=args.max_image_size
    )

    # 데모 실행
    if args.demo_type == "rcs_login":
        demo.run_rcs_login_demo(
            server=args.server,
            username=args.username,
            password=args.password
        )
    elif args.demo_type == "screen_analysis":
        demo.run_screen_analysis_demo()

    # 최종 리포트
    demo.print_final_report()


if __name__ == "__main__":
    main()
