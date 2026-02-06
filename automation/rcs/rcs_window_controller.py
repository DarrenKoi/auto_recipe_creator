"""
RCS Window Controller

pywinauto를 활용하여 RCS(Remote Control System) 윈도우의 UI 컨트롤에
직접 접근하고 조작합니다. ComboBox, Edit, Button 등의 컨트롤을
이름/ID 기반으로 제어하여 좌표 기반 자동화보다 안정적입니다.

인터랙션 모드:
- MESSAGE: 기존 메시지 기반 (set_edit_text, click, select)
- INPUT: 실제 OS 입력 이벤트 (click_input, type_keys) — 기본값
- VISUAL_DEBUG: INPUT + 컨트롤 하이라이트 + 좌표 로깅 + 딜레이
"""

import time
from enum import Enum

try:
    from pywinauto import Application
    from pywinauto.timings import wait_until_passes
    from pywinauto.findwindows import ElementNotFoundError
    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False
    print("[WARNING] pywinauto 라이브러리가 설치되지 않았습니다. pip install pywinauto")


class InteractionMode(Enum):
    """컨트롤 인터랙션 모드"""
    MESSAGE = "message"        # 메시지 기반 (set_edit_text, click, select)
    INPUT = "input"            # 실제 OS 입력 이벤트 (click_input, type_keys)
    VISUAL_DEBUG = "visual_debug"  # INPUT + 시각 디버그 (하이라이트, 좌표 로깅, 딜레이)


class RCSWindowController:
    """RCS 윈도우 제어 클래스

    pywinauto.Application을 래핑하여 RCS 프로그램의 UI 컨트롤을
    제어하기 위한 메서드를 제공합니다.
    """

    def __init__(
        self,
        backend: str = "uia",
        interaction_mode: str = "input",
        visual_debug_delay_sec: float = 1.0,
        highlight_colour: str = "red",
        highlight_thickness: int = 2,
    ):
        """
        Args:
            backend: pywinauto 백엔드 ("uia" 또는 "win32")
            interaction_mode: 인터랙션 모드 ("message", "input", "visual_debug")
            visual_debug_delay_sec: 시각 디버그 모드에서 액션 간 대기 시간 (초)
            highlight_colour: 시각 디버그 하이라이트 색상
            highlight_thickness: 시각 디버그 하이라이트 테두리 두께
        """
        if not PYWINAUTO_AVAILABLE:
            print("[ERROR] pywinauto 라이브러리를 사용할 수 없습니다.")
            self.app = None
            self.main_window = None
            return

        self.backend = backend
        self.mode = InteractionMode(interaction_mode)
        self.visual_debug_delay_sec = visual_debug_delay_sec
        self.highlight_colour = highlight_colour
        self.highlight_thickness = highlight_thickness
        self.app = None
        self.main_window = None

        if self._is_visual_debug():
            print(f"[INFO] 시각 디버그 모드 활성화 (지연: {visual_debug_delay_sec}초, 색상: {highlight_colour})")
        elif self._use_input_mode():
            print("[INFO] input 모드 활성화 (실제 OS 입력 이벤트 사용)")

    # ------------------------------------------------------------------
    # 인터랙션 모드 헬퍼
    # ------------------------------------------------------------------

    def _use_input_mode(self) -> bool:
        """INPUT 또는 VISUAL_DEBUG 모드인지 확인합니다."""
        return self.mode in (InteractionMode.INPUT, InteractionMode.VISUAL_DEBUG)

    def _is_visual_debug(self) -> bool:
        """VISUAL_DEBUG 모드인지 확인합니다."""
        return self.mode == InteractionMode.VISUAL_DEBUG

    def _highlight_control(self, control, label: str):
        """시각 디버그 모드에서 컨트롤을 하이라이트하고 좌표를 로깅합니다.

        Args:
            control: pywinauto 컨트롤
            label: 컨트롤 설명 (로그용)
        """
        if not self._is_visual_debug():
            return
        try:
            rect = control.rectangle()
            print(f"[DEBUG] 하이라이트: {label} @ ({rect.left}, {rect.top}, {rect.right}, {rect.bottom})")
            control.draw_outline(colour=self.highlight_colour, thickness=self.highlight_thickness)
        except Exception as e:
            print(f"[DEBUG] 하이라이트 실패 ({label}): {e}")

    def _pause(self, reason: str):
        """시각 디버그 모드에서 설정된 시간만큼 대기합니다.

        Args:
            reason: 대기 사유 (로그용)
        """
        if not self._is_visual_debug():
            return
        print(f"[DEBUG] 대기 중: {reason} ({self.visual_debug_delay_sec}초)")
        time.sleep(self.visual_debug_delay_sec)

    def _click_control(self, control, label: str):
        """모드에 따라 컨트롤을 클릭합니다.

        Args:
            control: pywinauto 컨트롤
            label: 컨트롤 설명 (로그용)
        """
        self._highlight_control(control, label)
        if self._use_input_mode():
            rect = control.rectangle()
            center_x = (rect.left + rect.right) // 2
            center_y = (rect.top + rect.bottom) // 2
            print(f"[INFO] click_input: {label} @ ({center_x}, {center_y})")
            control.click_input()
        else:
            control.click()
            print(f"[INFO] click: {label}")
        self._pause(f"{label} 클릭 후")

    def _enter_text(self, control, text: str, field_name: str, is_password: bool = False):
        """모드에 따라 Edit 컨트롤에 텍스트를 입력합니다.

        Args:
            control: pywinauto Edit 컨트롤
            text: 입력할 텍스트
            field_name: 필드 이름 (로그용)
            is_password: 비밀번호 여부 (로그 마스킹용)
        """
        display_text = "****" if is_password else text
        self._highlight_control(control, field_name)

        if self._use_input_mode():
            control.click_input()
            time.sleep(0.1)
            # 기존 텍스트 전체 선택 후 덮어쓰기
            control.type_keys("^a", with_spaces=True)
            time.sleep(0.05)
            control.type_keys(text, with_spaces=True)
            print(f"[INFO] 입력 (input 모드): {field_name} = '{display_text}'")
        else:
            try:
                control.set_edit_text(text)
                print(f"[INFO] 입력 (message 모드): {field_name} = '{display_text}'")
            except Exception:
                print(f"[INFO] {field_name} set_edit_text() 실패, type_keys() 폴백 사용")
                control.click()
                time.sleep(0.1)
                control.type_keys("^a", with_spaces=True)
                time.sleep(0.05)
                control.type_keys(text, with_spaces=True)

        self._pause(f"{field_name} 입력 후")

    # ------------------------------------------------------------------
    # 윈도우 연결 / 실행
    # ------------------------------------------------------------------

    def launch(self, exe_path: str, timeout: int = 30) -> bool:
        """RCS 프로그램을 실행하고 윈도우가 나타날 때까지 대기합니다.

        Args:
            exe_path: RCS 실행 파일 경로
            timeout: 윈도우 대기 시간 (초)

        Returns:
            성공 여부
        """
        if not PYWINAUTO_AVAILABLE:
            print("[ERROR] pywinauto 라이브러리를 사용할 수 없습니다.")
            return False

        try:
            print(f"[INFO] RCS 실행 중: {exe_path}")
            self.app = Application(backend=self.backend).start(exe_path)
            # 윈도우가 나타날 때까지 대기
            self.app.window(title_re="Remote Control System.*").wait(
                "visible", timeout=timeout
            )
            self.main_window = self.app.window(title_re="Remote Control System.*")
            print(f"[INFO] RCS 윈도우 발견: {self.main_window.window_text()}")
            return True
        except Exception as e:
            print(f"[ERROR] RCS 실행 실패: {e}")
            return False

    def connect_to_existing(self, title_re: str = "Remote Control System.*") -> bool:
        """이미 실행 중인 RCS 윈도우에 연결합니다.

        Args:
            title_re: 윈도우 타이틀 정규식

        Returns:
            성공 여부
        """
        if not PYWINAUTO_AVAILABLE:
            print("[ERROR] pywinauto 라이브러리를 사용할 수 없습니다.")
            return False

        try:
            self.app = Application(backend=self.backend).connect(title_re=title_re)
            self.main_window = self.app.window(title_re=title_re)
            print(f"[INFO] 기존 RCS 윈도우에 연결: {self.main_window.window_text()}")
            return True
        except ElementNotFoundError:
            print("[INFO] 실행 중인 RCS 윈도우를 찾을 수 없습니다.")
            return False
        except Exception as e:
            print(f"[ERROR] RCS 윈도우 연결 실패: {e}")
            return False

    # ------------------------------------------------------------------
    # UI 조작
    # ------------------------------------------------------------------

    def select_server(self, server_name: str) -> bool:
        """ComboBox에서 서버를 선택합니다.

        message 모드: select() 우선, 실패 시 click()+type_keys() 폴백
        input/visual_debug 모드: click_input()+type_keys()

        Args:
            server_name: 선택할 서버 이름

        Returns:
            성공 여부
        """
        if not self._check_window():
            return False

        try:
            combobox = self.main_window.child_window(control_type="ComboBox")
            self._highlight_control(combobox, "서버 ComboBox")

            if not self._use_input_mode():
                # MESSAGE 모드: select() 우선 시도
                try:
                    combobox.select(server_name)
                    print(f"[INFO] 서버 선택 완료: {server_name}")
                    return True
                except Exception:
                    pass
                # 폴백: click + type_keys
                print("[INFO] ComboBox select() 실패, type_keys() 폴백 시도 중...")
                combobox.click()
                time.sleep(0.3)
                combobox.type_keys(server_name + "{ENTER}", with_spaces=True)
            else:
                # INPUT / VISUAL_DEBUG 모드
                self._click_control(combobox, "서버 ComboBox")
                self._pause("ComboBox 드롭다운 열림")
                time.sleep(0.3)
                combobox.type_keys(server_name + "{ENTER}", with_spaces=True)

            print(f"[INFO] 서버 선택 완료: {server_name}")
            self._pause("서버 선택 후")
            return True
        except Exception as e:
            print(f"[ERROR] 서버 선택 실패: {e}")
            return False

    def enter_credentials(self, username: str, password: str) -> bool:
        """ID와 비밀번호를 입력합니다.

        Edit 컨트롤을 순서대로 찾아 첫 번째에 ID, 두 번째에 비밀번호를 입력합니다.

        Args:
            username: 사용자 ID
            password: 비밀번호

        Returns:
            성공 여부
        """
        if not self._check_window():
            return False

        try:
            edit_controls = self.main_window.children(control_type="Edit")
            if len(edit_controls) < 2:
                print(f"[ERROR] Edit 컨트롤을 찾을 수 없습니다. (발견: {len(edit_controls)}개)")
                return False

            # 첫 번째 Edit: 사용자 ID
            id_field = edit_controls[0]
            self._enter_text(id_field, username, "사용자 ID")

            # 두 번째 Edit: 비밀번호
            pw_field = edit_controls[1]
            self._enter_text(pw_field, password, "비밀번호", is_password=True)

            print("[INFO] 자격증명 입력 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 자격증명 입력 실패: {e}")
            return False

    def click_login(self) -> bool:
        """로그인/접속 버튼을 클릭합니다.

        Returns:
            성공 여부
        """
        if not self._check_window():
            return False

        try:
            # Button 컨트롤 중 로그인 관련 버튼을 검색
            buttons = self.main_window.children(control_type="Button")
            login_button = None

            # 버튼 텍스트에서 로그인/접속/Login/Connect 등의 키워드 검색
            login_keywords = ["로그인", "접속", "Login", "Connect", "확인", "OK"]
            for btn in buttons:
                btn_text = btn.window_text()
                for keyword in login_keywords:
                    if keyword.lower() in btn_text.lower():
                        login_button = btn
                        break
                if login_button:
                    break

            # 키워드로 찾지 못한 경우 첫 번째 버튼 사용
            if not login_button and buttons:
                login_button = buttons[0]
                print(f"[INFO] 키워드 매칭 실패, 첫 번째 버튼 사용: '{login_button.window_text()}'")

            if not login_button:
                print("[ERROR] 로그인 버튼을 찾을 수 없습니다.")
                return False

            btn_text = login_button.window_text()
            self._click_control(login_button, f"로그인 버튼 '{btn_text}'")
            print(f"[INFO] 로그인 버튼 클릭 완료: '{btn_text}'")
            return True
        except Exception as e:
            print(f"[ERROR] 로그인 버튼 클릭 실패: {e}")
            return False

    def is_login_successful(self, timeout: int = 15) -> bool:
        """로그인 성공 여부를 확인합니다.

        로그인 윈도우가 사라지거나 메인 화면으로 전환되는 것을 감지합니다.
        에러 다이얼로그가 나타나면 실패로 판단합니다.

        Args:
            timeout: 확인 대기 시간 (초)

        Returns:
            성공 여부
        """
        if not PYWINAUTO_AVAILABLE or not self.app:
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 에러 다이얼로그 감지
                error_dialogs = self.app.windows(title_re=".*[Ee]rror.*|.*오류.*|.*실패.*")
                if error_dialogs:
                    error_text = error_dialogs[0].window_text()
                    print(f"[ERROR] 로그인 에러 다이얼로그 감지: {error_text}")
                    return False

                # 로그인 윈도우의 Edit 컨트롤이 사라졌는지 확인
                # (로그인 성공 시 메인 화면으로 전환됨)
                if self.main_window.exists():
                    edit_controls = self.main_window.children(control_type="Edit")
                    if len(edit_controls) == 0:
                        print("[INFO] 로그인 윈도우 전환 감지 (Edit 컨트롤 사라짐)")
                        return True
            except Exception:
                # 윈도우가 변경되는 과정에서 예외 발생 가능
                pass

            time.sleep(1.0)

        print("[WARNING] 로그인 성공 확인 타임아웃 - 수동 확인이 필요합니다.")
        return False

    def print_control_identifiers(self):
        """현재 윈도우의 모든 UI 컨트롤 정보를 출력합니다.

        초기 설정 시 컨트롤 이름, 타입, automation ID를 확인하기 위한
        디버그 도구입니다. 실제 Windows 머신에서 실행하여 컨트롤 트리를
        확인한 후, 정확한 auto_id 값으로 코드를 보강할 수 있습니다.
        """
        if not self._check_window():
            return

        print("=" * 60)
        print("RCS 윈도우 컨트롤 트리")
        print("=" * 60)
        self.main_window.print_control_identifiers()
        print("=" * 60)

    def close(self):
        """RCS 프로그램을 종료합니다."""
        if self.app:
            try:
                self.app.kill()
                print("[INFO] RCS 프로그램 종료 완료")
            except Exception as e:
                print(f"[WARNING] RCS 프로그램 종료 중 오류: {e}")

    def _check_window(self) -> bool:
        """윈도우 연결 상태를 확인합니다."""
        if not PYWINAUTO_AVAILABLE:
            print("[ERROR] pywinauto 라이브러리를 사용할 수 없습니다.")
            return False
        if not self.main_window:
            print("[ERROR] RCS 윈도우에 연결되지 않았습니다. launch() 또는 connect_to_existing()을 먼저 호출하세요.")
            return False
        return True
