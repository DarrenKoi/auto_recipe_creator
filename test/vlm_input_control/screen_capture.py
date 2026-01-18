"""
Screen Capture Module

mss 라이브러리를 활용한 화면 캡처 기능을 제공합니다.
VM 창이나 전체 화면을 고속으로 캡처할 수 있습니다.

요구사항: FR-01 (VM 창 화면 캡처 및 실시간 모니터링)
테스트 케이스: TC-01 (VM 창 캡처)
"""

import time
import os
from typing import Optional, Tuple
from datetime import datetime

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[WARNING] mss 라이브러리가 설치되지 않았습니다. pip install mss")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] Pillow 라이브러리가 설치되지 않았습니다. pip install Pillow")


class ScreenCapture:
    """화면 캡처 클래스"""

    def __init__(self, output_dir: str = "./captures"):
        """
        Args:
            output_dir: 캡처 이미지 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if MSS_AVAILABLE:
            self.sct = mss.mss()
        else:
            self.sct = None

    def get_monitors(self) -> list:
        """
        사용 가능한 모니터 목록을 반환합니다.

        Returns:
            모니터 정보 리스트 (각 모니터의 좌표 및 크기)
        """
        if not MSS_AVAILABLE or not self.sct:
            print("[ERROR] mss 라이브러리를 사용할 수 없습니다.")
            return []

        monitors = self.sct.monitors
        print(f"[INFO] 감지된 모니터 수: {len(monitors) - 1}")  # 첫 번째는 전체 화면
        for i, mon in enumerate(monitors):
            print(f"  모니터 {i}: {mon}")
        return monitors

    def capture_full_screen(self, save: bool = True) -> Optional[bytes]:
        """
        전체 화면을 캡처합니다.

        Args:
            save: 파일로 저장할지 여부

        Returns:
            PNG 이미지 바이트 데이터 또는 None
        """
        if not MSS_AVAILABLE or not self.sct:
            print("[ERROR] mss 라이브러리를 사용할 수 없습니다.")
            return None

        start_time = time.time()

        # 전체 화면 캡처 (monitor 0은 모든 모니터를 포함)
        screenshot = self.sct.grab(self.sct.monitors[0])

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 전체 화면 캡처 완료 - 크기: {screenshot.width}x{screenshot.height}, 소요시간: {elapsed_ms:.2f}ms")

        # PNG로 변환
        png_data = mss.tools.to_png(screenshot.rgb, screenshot.size)

        if save:
            filename = f"fullscreen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(png_data)
            print(f"[INFO] 저장됨: {filepath}")

        return png_data

    def capture_region(self, x: int, y: int, width: int, height: int, save: bool = True) -> Optional[bytes]:
        """
        특정 영역을 캡처합니다.

        Args:
            x: 시작 X 좌표
            y: 시작 Y 좌표
            width: 캡처 영역 너비
            height: 캡처 영역 높이
            save: 파일로 저장할지 여부

        Returns:
            PNG 이미지 바이트 데이터 또는 None
        """
        if not MSS_AVAILABLE or not self.sct:
            print("[ERROR] mss 라이브러리를 사용할 수 없습니다.")
            return None

        start_time = time.time()

        region = {
            "left": x,
            "top": y,
            "width": width,
            "height": height
        }

        screenshot = self.sct.grab(region)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 영역 캡처 완료 - 위치: ({x}, {y}), 크기: {width}x{height}, 소요시간: {elapsed_ms:.2f}ms")

        png_data = mss.tools.to_png(screenshot.rgb, screenshot.size)

        if save:
            filename = f"region_{x}_{y}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(png_data)
            print(f"[INFO] 저장됨: {filepath}")

        return png_data

    def capture_monitor(self, monitor_index: int = 1, save: bool = True) -> Optional[bytes]:
        """
        특정 모니터를 캡처합니다.

        Args:
            monitor_index: 모니터 인덱스 (1부터 시작)
            save: 파일로 저장할지 여부

        Returns:
            PNG 이미지 바이트 데이터 또는 None
        """
        if not MSS_AVAILABLE or not self.sct:
            print("[ERROR] mss 라이브러리를 사용할 수 없습니다.")
            return None

        monitors = self.sct.monitors
        if monitor_index >= len(monitors):
            print(f"[ERROR] 모니터 인덱스 {monitor_index}가 범위를 벗어났습니다. (최대: {len(monitors) - 1})")
            return None

        start_time = time.time()

        screenshot = self.sct.grab(monitors[monitor_index])

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 모니터 {monitor_index} 캡처 완료 - 크기: {screenshot.width}x{screenshot.height}, 소요시간: {elapsed_ms:.2f}ms")

        png_data = mss.tools.to_png(screenshot.rgb, screenshot.size)

        if save:
            filename = f"monitor{monitor_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(png_data)
            print(f"[INFO] 저장됨: {filepath}")

        return png_data

    def continuous_capture(self, interval_ms: int = 100, duration_seconds: int = 5, region: Optional[dict] = None):
        """
        연속 캡처를 수행합니다 (실시간 모니터링 테스트용).

        Args:
            interval_ms: 캡처 간격 (밀리초)
            duration_seconds: 총 캡처 시간 (초)
            region: 캡처 영역 (None이면 전체 화면)
        """
        if not MSS_AVAILABLE or not self.sct:
            print("[ERROR] mss 라이브러리를 사용할 수 없습니다.")
            return

        capture_count = 0
        start_time = time.time()
        interval_sec = interval_ms / 1000.0

        print(f"[INFO] 연속 캡처 시작 - 간격: {interval_ms}ms, 시간: {duration_seconds}초")

        capture_times = []

        while (time.time() - start_time) < duration_seconds:
            capture_start = time.time()

            if region:
                self.sct.grab(region)
            else:
                self.sct.grab(self.sct.monitors[0])

            capture_time_ms = (time.time() - capture_start) * 1000
            capture_times.append(capture_time_ms)
            capture_count += 1

            # 다음 캡처까지 대기
            elapsed = time.time() - capture_start
            if elapsed < interval_sec:
                time.sleep(interval_sec - elapsed)

        avg_capture_time = sum(capture_times) / len(capture_times) if capture_times else 0
        print(f"[INFO] 연속 캡처 완료 - 총 {capture_count}회, 평균 캡처 시간: {avg_capture_time:.2f}ms")

        # TC-01 기준: 지연 < 50ms
        if avg_capture_time < 50:
            print("[PASS] TC-01: 캡처 지연 시간이 50ms 미만입니다.")
        else:
            print(f"[WARN] TC-01: 캡처 지연 시간이 {avg_capture_time:.2f}ms로 50ms를 초과합니다.")

    def get_image_as_pil(self, png_data: bytes) -> Optional['Image.Image']:
        """
        PNG 바이트 데이터를 PIL Image로 변환합니다.

        Args:
            png_data: PNG 이미지 바이트 데이터

        Returns:
            PIL Image 객체 또는 None
        """
        if not PIL_AVAILABLE:
            print("[ERROR] Pillow 라이브러리를 사용할 수 없습니다.")
            return None

        return Image.open(io.BytesIO(png_data))

    def close(self):
        """리소스를 정리합니다."""
        if self.sct:
            self.sct.close()
            print("[INFO] ScreenCapture 리소스 정리 완료")


def test_screen_capture():
    """화면 캡처 기능 테스트"""
    print("=" * 60)
    print("Screen Capture Test")
    print("=" * 60)

    capture = ScreenCapture(output_dir="./test_captures")

    # 1. 모니터 목록 확인
    print("\n[TEST 1] 모니터 목록 확인")
    monitors = capture.get_monitors()

    # 2. 전체 화면 캡처
    print("\n[TEST 2] 전체 화면 캡처")
    full_screen = capture.capture_full_screen(save=True)
    if full_screen:
        print(f"  캡처된 이미지 크기: {len(full_screen)} bytes")

    # 3. 영역 캡처
    print("\n[TEST 3] 영역 캡처 (100x100 at 0,0)")
    region = capture.capture_region(0, 0, 100, 100, save=True)
    if region:
        print(f"  캡처된 이미지 크기: {len(region)} bytes")

    # 4. 연속 캡처 테스트 (성능 측정)
    print("\n[TEST 4] 연속 캡처 성능 테스트 (1초간)")
    capture.continuous_capture(interval_ms=50, duration_seconds=1)

    capture.close()
    print("\n" + "=" * 60)
    print("Screen Capture Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_screen_capture()
