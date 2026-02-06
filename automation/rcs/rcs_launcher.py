"""
RCS Launcher

RCS(Remote Control System) 프로그램의 실행부터 로그인까지의
전체 시퀀스를 오케스트레이션합니다.

시퀀스:
1. RCS 실행 중 여부 확인 (실행 중이면 연결, 아니면 새로 실행)
2. 윈도우 대기
3. 서버 선택
4. 자격증명 입력
5. 로그인 버튼 클릭
6. 로그인 성공 확인
"""

import time

from .rcs_config import RCSConfig
from .rcs_window_controller import RCSWindowController


class RCSLauncher:
    """RCS 실행 및 로그인 오케스트레이터

    RCSWindowController를 사용하여 RCS 프로그램의 실행부터
    로그인까지의 전체 과정을 자동화합니다.
    """

    def __init__(self, config: RCSConfig = None):
        """
        Args:
            config: RCS 설정 (None이면 기본값 사용)
        """
        self.config = config or RCSConfig()
        self.controller = self._create_controller()

    def run(self) -> bool:
        """RCS 실행 및 로그인 전체 시퀀스를 실행합니다.

        max_retries 횟수만큼 재시도하며, 각 시도 사이에 정리 작업을 수행합니다.

        Returns:
            로그인 성공 여부
        """
        for attempt in range(1, self.config.max_retries + 1):
            print(f"\n[INFO] === RCS 로그인 시도 {attempt}/{self.config.max_retries} ===")

            success = self._attempt_login()
            if success:
                print("\n[SUCCESS] RCS 로그인 성공!")
                return True

            if attempt < self.config.max_retries:
                print(f"[INFO] {self.config.retry_delay_sec}초 후 재시도...")
                self._cleanup()
                time.sleep(self.config.retry_delay_sec)

        print(f"\n[ERROR] RCS 로그인 실패 ({self.config.max_retries}회 시도)")
        return False

    def debug_controls(self) -> bool:
        """RCS 윈도우의 컨트롤 트리를 출력합니다.

        초기 설정 시 컨트롤 이름과 ID를 확인하기 위한 디버그 모드입니다.
        실제 Windows 머신에서 실행하여 컨트롤 구조를 파악합니다.

        Returns:
            연결 성공 여부
        """
        print("[INFO] === RCS 디버그 모드 ===")

        # 기존 윈도우 연결 시도
        connected = self.controller.connect_to_existing(
            title_re=self.config.window_title_re
        )

        # 연결 실패 시 새로 실행
        if not connected:
            print("[INFO] 실행 중인 RCS가 없어 새로 실행합니다...")
            launched = self.controller.launch(
                exe_path=self.config.exe_path,
                timeout=self.config.launch_timeout_sec,
            )
            if not launched:
                print("[ERROR] RCS 실행 실패 - 디버그를 수행할 수 없습니다.")
                return False

        # 컨트롤 트리 출력
        self.controller.print_control_identifiers()
        return True

    def _create_controller(self) -> RCSWindowController:
        """설정값을 기반으로 RCSWindowController 인스턴스를 생성합니다."""
        return RCSWindowController(
            backend=self.config.backend,
            interaction_mode=self.config.interaction_mode,
            visual_debug_delay_sec=self.config.visual_debug_delay_sec,
            highlight_colour=self.config.highlight_colour,
            highlight_thickness=self.config.highlight_thickness,
        )

    def _attempt_login(self) -> bool:
        """단일 로그인 시도를 수행합니다.

        Returns:
            성공 여부
        """
        # 1. RCS 윈도우 연결 또는 실행
        connected = self.controller.connect_to_existing(
            title_re=self.config.window_title_re
        )
        if not connected:
            launched = self.controller.launch(
                exe_path=self.config.exe_path,
                timeout=self.config.launch_timeout_sec,
            )
            if not launched:
                return False

        # 2. 서버 선택
        if not self.controller.select_server(self.config.server_name):
            return False

        # 3. 자격증명 입력
        if not self.controller.enter_credentials(
            self.config.username, self.config.password
        ):
            return False

        # 4. 로그인 버튼 클릭
        if not self.controller.click_login():
            return False

        # 5. 로그인 성공 확인
        if not self.controller.is_login_successful(
            timeout=self.config.login_timeout_sec
        ):
            return False

        return True

    def _cleanup(self):
        """재시도 전 정리 작업을 수행합니다."""
        try:
            self.controller.close()
        except Exception:
            pass
        # 새 컨트롤러 인스턴스 생성
        self.controller = self._create_controller()
