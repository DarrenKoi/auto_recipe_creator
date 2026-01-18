"""
Mouse Control Module

pynput 라이브러리를 활용한 마우스 제어 기능을 제공합니다.
클릭, 이동, 드래그 등의 마우스 동작을 자동화합니다.

요구사항: FR-03 (마우스/키보드 입력 자동화)
테스트 케이스: TC-04 (클릭 전달)
"""

import time
from typing import Optional, Tuple, Callable
from enum import Enum

try:
    from pynput.mouse import Button, Controller as MouseControllerBase, Listener as MouseListener
    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False
    print("[WARNING] pynput 라이브러리가 설치되지 않았습니다. pip install pynput")


class MouseButton(Enum):
    """마우스 버튼 열거형"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class MouseController:
    """마우스 제어 클래스"""

    def __init__(self):
        if PYNPUT_MOUSE_AVAILABLE:
            self.controller = MouseControllerBase()
        else:
            self.controller = None

        self._listener = None
        self._click_callback = None

    @property
    def position(self) -> Tuple[int, int]:
        """현재 마우스 위치를 반환합니다."""
        if not self.controller:
            return (0, 0)
        return self.controller.position

    def move_to(self, x: int, y: int, duration: float = 0.0):
        """
        마우스를 특정 좌표로 이동합니다.

        Args:
            x: 목표 X 좌표
            y: 목표 Y 좌표
            duration: 이동에 걸리는 시간 (초), 0이면 즉시 이동
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        if duration <= 0:
            self.controller.position = (x, y)
            print(f"[INFO] 마우스 이동: ({x}, {y})")
        else:
            # 부드러운 이동 (선형 보간)
            start_x, start_y = self.position
            steps = int(duration * 60)  # 60fps 기준

            for i in range(steps + 1):
                t = i / steps
                curr_x = int(start_x + (x - start_x) * t)
                curr_y = int(start_y + (y - start_y) * t)
                self.controller.position = (curr_x, curr_y)
                time.sleep(duration / steps)

            print(f"[INFO] 마우스 이동 (부드러움): ({start_x}, {start_y}) -> ({x}, {y})")

    def move_relative(self, dx: int, dy: int):
        """
        마우스를 현재 위치에서 상대적으로 이동합니다.

        Args:
            dx: X축 이동량
            dy: Y축 이동량
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        current_x, current_y = self.position
        new_x = current_x + dx
        new_y = current_y + dy
        self.controller.position = (new_x, new_y)
        print(f"[INFO] 마우스 상대 이동: ({dx}, {dy}) -> 현재 위치: ({new_x}, {new_y})")

    def click(self, button: MouseButton = MouseButton.LEFT, count: int = 1):
        """
        현재 위치에서 마우스 클릭을 수행합니다.

        Args:
            button: 클릭할 버튼 (LEFT, RIGHT, MIDDLE)
            count: 클릭 횟수 (2면 더블클릭)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        pynput_button = self._get_pynput_button(button)
        self.controller.click(pynput_button, count)

        click_type = "더블클릭" if count == 2 else "클릭"
        print(f"[INFO] 마우스 {button.value} {click_type} at {self.position}")

    def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT, count: int = 1):
        """
        특정 좌표에서 마우스 클릭을 수행합니다.

        Args:
            x: 클릭할 X 좌표
            y: 클릭할 Y 좌표
            button: 클릭할 버튼
            count: 클릭 횟수
        """
        start_time = time.time()

        self.move_to(x, y)
        time.sleep(0.01)  # 이동 후 잠시 대기
        self.click(button, count)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 클릭 액션 소요시간: {elapsed_ms:.2f}ms")

        # NFR-01 기준: 단일 액션 < 200ms
        if elapsed_ms < 200:
            print("[PASS] NFR-01: 단일 액션 실행 속도가 200ms 미만입니다.")

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        더블클릭을 수행합니다.

        Args:
            x: 클릭할 X 좌표 (None이면 현재 위치)
            y: 클릭할 Y 좌표 (None이면 현재 위치)
        """
        if x is not None and y is not None:
            self.click_at(x, y, MouseButton.LEFT, count=2)
        else:
            self.click(MouseButton.LEFT, count=2)

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        우클릭을 수행합니다.

        Args:
            x: 클릭할 X 좌표 (None이면 현재 위치)
            y: 클릭할 Y 좌표 (None이면 현재 위치)
        """
        if x is not None and y is not None:
            self.click_at(x, y, MouseButton.RIGHT)
        else:
            self.click(MouseButton.RIGHT)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """
        드래그를 수행합니다.

        Args:
            start_x: 시작 X 좌표
            start_y: 시작 Y 좌표
            end_x: 끝 X 좌표
            end_y: 끝 Y 좌표
            duration: 드래그에 걸리는 시간 (초)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        # 시작 위치로 이동
        self.move_to(start_x, start_y)
        time.sleep(0.05)

        # 마우스 버튼 누르기
        self.controller.press(Button.left)

        # 부드럽게 이동
        steps = int(duration * 60)
        for i in range(steps + 1):
            t = i / steps
            curr_x = int(start_x + (end_x - start_x) * t)
            curr_y = int(start_y + (end_y - start_y) * t)
            self.controller.position = (curr_x, curr_y)
            time.sleep(duration / steps)

        # 마우스 버튼 놓기
        self.controller.release(Button.left)

        print(f"[INFO] 드래그 완료: ({start_x}, {start_y}) -> ({end_x}, {end_y})")

    def scroll(self, dx: int = 0, dy: int = 0):
        """
        마우스 스크롤을 수행합니다.

        Args:
            dx: 수평 스크롤 양 (양수: 오른쪽, 음수: 왼쪽)
            dy: 수직 스크롤 양 (양수: 위, 음수: 아래)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        self.controller.scroll(dx, dy)
        print(f"[INFO] 스크롤: dx={dx}, dy={dy}")

    def press(self, button: MouseButton = MouseButton.LEFT):
        """마우스 버튼을 누릅니다 (놓지 않음)."""
        if not self.controller:
            return
        self.controller.press(self._get_pynput_button(button))
        print(f"[INFO] 마우스 {button.value} 버튼 누름")

    def release(self, button: MouseButton = MouseButton.LEFT):
        """마우스 버튼을 놓습니다."""
        if not self.controller:
            return
        self.controller.release(self._get_pynput_button(button))
        print(f"[INFO] 마우스 {button.value} 버튼 놓음")

    def start_listener(self, on_click: Optional[Callable] = None, on_move: Optional[Callable] = None):
        """
        마우스 이벤트 리스너를 시작합니다.

        Args:
            on_click: 클릭 이벤트 콜백 함수 (x, y, button, pressed)
            on_move: 이동 이벤트 콜백 함수 (x, y)
        """
        if not PYNPUT_MOUSE_AVAILABLE:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        def default_on_click(x, y, button, pressed):
            action = "pressed" if pressed else "released"
            print(f"[LISTENER] Mouse {button} {action} at ({x}, {y})")
            if on_click:
                on_click(x, y, button, pressed)

        def default_on_move(x, y):
            if on_move:
                on_move(x, y)

        self._listener = MouseListener(on_click=default_on_click, on_move=default_on_move)
        self._listener.start()
        print("[INFO] 마우스 리스너 시작됨")

    def stop_listener(self):
        """마우스 이벤트 리스너를 중지합니다."""
        if self._listener:
            self._listener.stop()
            self._listener = None
            print("[INFO] 마우스 리스너 중지됨")

    def _get_pynput_button(self, button: MouseButton):
        """MouseButton을 pynput Button으로 변환합니다."""
        if not PYNPUT_MOUSE_AVAILABLE:
            return None

        mapping = {
            MouseButton.LEFT: Button.left,
            MouseButton.RIGHT: Button.right,
            MouseButton.MIDDLE: Button.middle
        }
        return mapping.get(button, Button.left)


def test_mouse_control():
    """마우스 제어 기능 테스트"""
    print("=" * 60)
    print("Mouse Control Test")
    print("=" * 60)

    if not PYNPUT_MOUSE_AVAILABLE:
        print("[SKIP] pynput 라이브러리가 없어 테스트를 건너뜁니다.")
        return

    mouse = MouseController()

    # 1. 현재 위치 확인
    print("\n[TEST 1] 현재 마우스 위치 확인")
    current_pos = mouse.position
    print(f"  현재 위치: {current_pos}")

    # 2. 마우스 이동 테스트
    print("\n[TEST 2] 마우스 이동 테스트")
    original_pos = mouse.position

    # 테스트 위치로 이동 (화면 중앙 근처)
    test_x, test_y = 100, 100
    mouse.move_to(test_x, test_y)
    time.sleep(0.1)

    new_pos = mouse.position
    print(f"  이동 후 위치: {new_pos}")

    # TC-04 기준: 정확도 ± 5px
    accuracy_x = abs(new_pos[0] - test_x)
    accuracy_y = abs(new_pos[1] - test_y)

    if accuracy_x <= 5 and accuracy_y <= 5:
        print(f"[PASS] TC-04: 이동 정확도 - X 오차: {accuracy_x}px, Y 오차: {accuracy_y}px (기준: ±5px)")
    else:
        print(f"[FAIL] TC-04: 이동 정확도 - X 오차: {accuracy_x}px, Y 오차: {accuracy_y}px (기준: ±5px)")

    # 3. 상대 이동 테스트
    print("\n[TEST 3] 상대 이동 테스트")
    mouse.move_relative(50, 50)
    time.sleep(0.1)
    print(f"  상대 이동 후 위치: {mouse.position}")

    # 4. 부드러운 이동 테스트
    print("\n[TEST 4] 부드러운 이동 테스트")
    mouse.move_to(200, 200, duration=0.3)

    # 5. 클릭 테스트 (실제 클릭은 주석 처리 - 안전을 위해)
    print("\n[TEST 5] 클릭 테스트 (시뮬레이션)")
    print("  [INFO] 실제 클릭은 안전을 위해 비활성화되었습니다.")
    print("  [INFO] 활성화하려면 아래 코드의 주석을 해제하세요.")
    # mouse.click_at(150, 150)
    # mouse.double_click(150, 150)
    # mouse.right_click(150, 150)

    # 6. 스크롤 테스트
    print("\n[TEST 6] 스크롤 테스트 (시뮬레이션)")
    print("  [INFO] 실제 스크롤은 안전을 위해 비활성화되었습니다.")
    # mouse.scroll(dy=-3)  # 아래로 스크롤

    # 원래 위치로 복귀
    mouse.move_to(original_pos[0], original_pos[1])

    print("\n" + "=" * 60)
    print("Mouse Control Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_mouse_control()
