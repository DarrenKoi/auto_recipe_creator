"""
Keyboard Control Module

pynput 라이브러리를 활용한 키보드 제어 기능을 제공합니다.
키 입력, 조합키, 텍스트 타이핑 등의 키보드 동작을 자동화합니다.

요구사항: FR-03 (마우스/키보드 입력 자동화)
"""

import time
from typing import Optional, List, Callable

try:
    from pynput.keyboard import Key, Controller as KeyboardControllerBase, Listener as KeyboardListener
    PYNPUT_KEYBOARD_AVAILABLE = True
except ImportError:
    PYNPUT_KEYBOARD_AVAILABLE = False
    print("[WARNING] pynput 라이브러리가 설치되지 않았습니다. pip install pynput")


# 특수 키 매핑
SPECIAL_KEYS = {
    'enter': Key.enter,
    'return': Key.enter,
    'tab': Key.tab,
    'space': Key.space,
    'backspace': Key.backspace,
    'delete': Key.delete,
    'escape': Key.esc,
    'esc': Key.esc,
    'up': Key.up,
    'down': Key.down,
    'left': Key.left,
    'right': Key.right,
    'home': Key.home,
    'end': Key.end,
    'page_up': Key.page_up,
    'page_down': Key.page_down,
    'insert': Key.insert,
    'f1': Key.f1,
    'f2': Key.f2,
    'f3': Key.f3,
    'f4': Key.f4,
    'f5': Key.f5,
    'f6': Key.f6,
    'f7': Key.f7,
    'f8': Key.f8,
    'f9': Key.f9,
    'f10': Key.f10,
    'f11': Key.f11,
    'f12': Key.f12,
    'ctrl': Key.ctrl,
    'ctrl_l': Key.ctrl_l,
    'ctrl_r': Key.ctrl_r,
    'alt': Key.alt,
    'alt_l': Key.alt_l,
    'alt_r': Key.alt_r,
    'shift': Key.shift,
    'shift_l': Key.shift_l,
    'shift_r': Key.shift_r,
    'caps_lock': Key.caps_lock,
    'num_lock': Key.num_lock,
    'print_screen': Key.print_screen,
    'scroll_lock': Key.scroll_lock,
    'pause': Key.pause,
    'menu': Key.menu,
    'cmd': Key.cmd,
    'win': Key.cmd,  # Windows 키
} if PYNPUT_KEYBOARD_AVAILABLE else {}


class KeyboardController:
    """키보드 제어 클래스"""

    def __init__(self):
        if PYNPUT_KEYBOARD_AVAILABLE:
            self.controller = KeyboardControllerBase()
        else:
            self.controller = None

        self._listener = None

    def press_key(self, key: str):
        """
        단일 키를 누릅니다.

        Args:
            key: 누를 키 (문자 또는 특수키 이름)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        pynput_key = self._get_pynput_key(key)
        self.controller.press(pynput_key)
        print(f"[INFO] 키 누름: {key}")

    def release_key(self, key: str):
        """
        단일 키를 놓습니다.

        Args:
            key: 놓을 키
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        pynput_key = self._get_pynput_key(key)
        self.controller.release(pynput_key)
        print(f"[INFO] 키 놓음: {key}")

    def tap_key(self, key: str, count: int = 1, interval: float = 0.05):
        """
        키를 눌렀다 놓습니다 (탭).

        Args:
            key: 탭할 키
            count: 반복 횟수
            interval: 반복 간 간격 (초)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        pynput_key = self._get_pynput_key(key)

        for i in range(count):
            self.controller.press(pynput_key)
            self.controller.release(pynput_key)
            if i < count - 1:
                time.sleep(interval)

        print(f"[INFO] 키 탭: {key} x{count}")

    def type_text(self, text: str, interval: float = 0.02):
        """
        텍스트를 타이핑합니다.

        Args:
            text: 입력할 텍스트
            interval: 문자 간 간격 (초)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        start_time = time.time()

        for char in text:
            self.controller.type(char)
            if interval > 0:
                time.sleep(interval)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 텍스트 입력 완료: '{text[:20]}{'...' if len(text) > 20 else ''}' ({len(text)}자, {elapsed_ms:.2f}ms)")

    def type_text_instant(self, text: str):
        """
        텍스트를 즉시 입력합니다 (간격 없음).

        Args:
            text: 입력할 텍스트
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        self.controller.type(text)
        print(f"[INFO] 텍스트 즉시 입력: '{text[:20]}{'...' if len(text) > 20 else ''}'")

    def hotkey(self, *keys: str):
        """
        조합키를 입력합니다 (예: Ctrl+C, Alt+F4).

        Args:
            *keys: 조합할 키들 (순서대로 누르고 역순으로 놓음)
        """
        if not self.controller:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        start_time = time.time()

        # 순서대로 키 누르기
        pressed_keys = []
        for key in keys:
            pynput_key = self._get_pynput_key(key)
            self.controller.press(pynput_key)
            pressed_keys.append(pynput_key)
            time.sleep(0.01)

        # 역순으로 키 놓기
        for pynput_key in reversed(pressed_keys):
            self.controller.release(pynput_key)
            time.sleep(0.01)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[INFO] 조합키 입력: {'+'.join(keys)} ({elapsed_ms:.2f}ms)")

        # NFR-01 기준: 단일 액션 < 200ms
        if elapsed_ms < 200:
            print("[PASS] NFR-01: 단일 액션 실행 속도가 200ms 미만입니다.")

    def copy(self):
        """Ctrl+C (복사)"""
        self.hotkey('ctrl', 'c')

    def paste(self):
        """Ctrl+V (붙여넣기)"""
        self.hotkey('ctrl', 'v')

    def cut(self):
        """Ctrl+X (잘라내기)"""
        self.hotkey('ctrl', 'x')

    def select_all(self):
        """Ctrl+A (전체 선택)"""
        self.hotkey('ctrl', 'a')

    def undo(self):
        """Ctrl+Z (실행 취소)"""
        self.hotkey('ctrl', 'z')

    def redo(self):
        """Ctrl+Y (다시 실행)"""
        self.hotkey('ctrl', 'y')

    def save(self):
        """Ctrl+S (저장)"""
        self.hotkey('ctrl', 's')

    def find(self):
        """Ctrl+F (찾기)"""
        self.hotkey('ctrl', 'f')

    def new_tab(self):
        """Ctrl+T (새 탭)"""
        self.hotkey('ctrl', 't')

    def close_tab(self):
        """Ctrl+W (탭 닫기)"""
        self.hotkey('ctrl', 'w')

    def alt_tab(self):
        """Alt+Tab (창 전환)"""
        self.hotkey('alt', 'tab')

    def switch_window(self):
        """Alt+Tab (창 전환) - alt_tab의 별칭"""
        self.alt_tab()

    def enter(self):
        """Enter 키 입력"""
        self.tap_key('enter')

    def escape(self):
        """Escape 키 입력"""
        self.tap_key('escape')

    def tab(self):
        """Tab 키 입력"""
        self.tap_key('tab')

    def backspace(self, count: int = 1):
        """
        Backspace 키 입력

        Args:
            count: 반복 횟수
        """
        self.tap_key('backspace', count=count)

    def delete(self, count: int = 1):
        """
        Delete 키 입력

        Args:
            count: 반복 횟수
        """
        self.tap_key('delete', count=count)

    def arrow_up(self, count: int = 1):
        """위쪽 화살표 키"""
        self.tap_key('up', count=count)

    def arrow_down(self, count: int = 1):
        """아래쪽 화살표 키"""
        self.tap_key('down', count=count)

    def arrow_left(self, count: int = 1):
        """왼쪽 화살표 키"""
        self.tap_key('left', count=count)

    def arrow_right(self, count: int = 1):
        """오른쪽 화살표 키"""
        self.tap_key('right', count=count)

    def function_key(self, number: int):
        """
        펑션 키 (F1-F12) 입력

        Args:
            number: 펑션 키 번호 (1-12)
        """
        if 1 <= number <= 12:
            self.tap_key(f'f{number}')
        else:
            print(f"[ERROR] 유효하지 않은 펑션 키 번호: {number}")

    def start_listener(self, on_press: Optional[Callable] = None, on_release: Optional[Callable] = None):
        """
        키보드 이벤트 리스너를 시작합니다.

        Args:
            on_press: 키 누름 이벤트 콜백 함수
            on_release: 키 놓음 이벤트 콜백 함수
        """
        if not PYNPUT_KEYBOARD_AVAILABLE:
            print("[ERROR] pynput 라이브러리를 사용할 수 없습니다.")
            return

        def default_on_press(key):
            print(f"[LISTENER] Key pressed: {key}")
            if on_press:
                on_press(key)

        def default_on_release(key):
            print(f"[LISTENER] Key released: {key}")
            if on_release:
                on_release(key)

        self._listener = KeyboardListener(on_press=default_on_press, on_release=default_on_release)
        self._listener.start()
        print("[INFO] 키보드 리스너 시작됨")

    def stop_listener(self):
        """키보드 이벤트 리스너를 중지합니다."""
        if self._listener:
            self._listener.stop()
            self._listener = None
            print("[INFO] 키보드 리스너 중지됨")

    def _get_pynput_key(self, key: str):
        """문자열 키를 pynput 키로 변환합니다."""
        if not PYNPUT_KEYBOARD_AVAILABLE:
            return None

        # 특수 키 확인
        key_lower = key.lower()
        if key_lower in SPECIAL_KEYS:
            return SPECIAL_KEYS[key_lower]

        # 단일 문자인 경우
        if len(key) == 1:
            return key

        # 알 수 없는 키
        print(f"[WARNING] 알 수 없는 키: {key}")
        return key


def test_keyboard_control():
    """키보드 제어 기능 테스트"""
    print("=" * 60)
    print("Keyboard Control Test")
    print("=" * 60)

    if not PYNPUT_KEYBOARD_AVAILABLE:
        print("[SKIP] pynput 라이브러리가 없어 테스트를 건너뜁니다.")
        return

    keyboard = KeyboardController()

    # 1. 키 매핑 테스트
    print("\n[TEST 1] 특수 키 매핑 확인")
    test_keys = ['enter', 'tab', 'escape', 'ctrl', 'alt', 'shift', 'f1']
    for key in test_keys:
        pynput_key = keyboard._get_pynput_key(key)
        print(f"  {key} -> {pynput_key}")

    # 2. 단일 키 탭 테스트 (시뮬레이션)
    print("\n[TEST 2] 단일 키 탭 테스트 (시뮬레이션)")
    print("  [INFO] 실제 키 입력은 안전을 위해 비활성화되었습니다.")
    print("  [INFO] 활성화하려면 아래 코드의 주석을 해제하세요.")
    # keyboard.tap_key('a')

    # 3. 조합키 테스트 (시뮬레이션)
    print("\n[TEST 3] 조합키 테스트 (시뮬레이션)")
    print("  [INFO] 실제 조합키 입력은 안전을 위해 비활성화되었습니다.")
    # keyboard.hotkey('ctrl', 'c')  # Ctrl+C
    # keyboard.hotkey('ctrl', 'v')  # Ctrl+V

    # 4. 텍스트 타이핑 테스트 (시뮬레이션)
    print("\n[TEST 4] 텍스트 타이핑 테스트 (시뮬레이션)")
    print("  [INFO] 실제 텍스트 입력은 안전을 위해 비활성화되었습니다.")
    # keyboard.type_text("Hello, World!")

    # 5. 성능 측정 (실제 입력 없이)
    print("\n[TEST 5] 키보드 제어 객체 상태 확인")
    print(f"  컨트롤러 상태: {'활성' if keyboard.controller else '비활성'}")
    print(f"  리스너 상태: {'실행 중' if keyboard._listener else '중지'}")

    print("\n" + "=" * 60)
    print("Keyboard Control Test Complete")
    print("=" * 60)
    print("\n[NOTE] 실제 키보드 입력 테스트를 수행하려면")
    print("       integration_test.py를 실행하거나")
    print("       이 파일의 주석 처리된 코드를 활성화하세요.")


if __name__ == "__main__":
    test_keyboard_control()
