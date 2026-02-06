"""
RCS (Remote Control System) 자동화 모듈

RCS 프로그램의 실행, 로그인, 서버 선택 등을 자동화합니다.
pywinauto 라이브러리를 활용하여 Windows UI 컨트롤에 직접 접근합니다.

주요 컴포넌트:
- rcs_config: RCS 설정 (경로, 자격증명, 타이밍)
- rcs_window_controller: pywinauto 기반 윈도우 제어
- rcs_launcher: 실행 -> 로그인 시퀀스 오케스트레이션
"""

from .rcs_config import RCSConfig
from .rcs_window_controller import InteractionMode, RCSWindowController
from .rcs_launcher import RCSLauncher

__all__ = [
    'RCSConfig',
    'InteractionMode',
    'RCSWindowController',
    'RCSLauncher',
]
