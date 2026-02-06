"""
RCS Configuration Settings

RCS(Remote Control System) 자동화에 필요한 설정값을 관리합니다.
실행 경로, 자격증명, 타이밍, 백엔드 설정 등을 포함합니다.

사용 전 반드시 exe_path, username, password, server_name을 실제 값으로 변경하세요.
"""

from dataclasses import dataclass


@dataclass
class RCSConfig:
    """RCS 자동화 설정"""

    # RCS 실행 파일 경로
    exe_path: str = r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"

    # 윈도우 타이틀 정규식 (버전 접미사 처리용)
    window_title_re: str = "Remote Control System.*"

    # 로그인 자격증명
    username: str = "YOUR_USERNAME"
    password: str = "YOUR_PASSWORD"

    # 접속할 서버 이름 (ComboBox에서 선택)
    server_name: str = "YOUR_SERVER_NAME"

    # 프로그램 실행 후 윈도우 대기 시간 (초)
    launch_timeout_sec: int = 30

    # 로그인 후 성공 확인 대기 시간 (초)
    login_timeout_sec: int = 15

    # pywinauto 백엔드 ("uia" 또는 "win32")
    # uia: 최신 UI Automation 프레임워크 (기본값, 권장)
    # win32: 레거시 Win32 API (uia에서 컨트롤 감지 실패 시 전환)
    backend: str = "uia"

    # 재시도 횟수
    max_retries: int = 3

    # 재시도 간 대기 시간 (초)
    retry_delay_sec: float = 2.0
