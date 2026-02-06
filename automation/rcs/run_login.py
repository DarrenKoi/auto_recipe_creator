"""
RCS 자동 로그인 CLI 엔트리포인트

사용법:
    python -m automation.rcs.run_login                    # 기본 설정으로 로그인 (input 모드)
    python -m automation.rcs.run_login --debug             # 컨트롤 트리 출력 (초기 설정용)
    python -m automation.rcs.run_login --server MyServer   # 서버 지정
    python -m automation.rcs.run_login --backend win32     # Win32 백엔드 사용
    python -m automation.rcs.run_login --visual-debug      # 시각 디버그 모드 (하이라이트+좌표 로깅)
    python -m automation.rcs.run_login --visual-debug --debug-delay 2.0  # 느린 디버그
    python -m automation.rcs.run_login --interaction-mode message  # 기존 메시지 기반 모드
"""

import argparse
import sys

from .rcs_config import RCSConfig
from .rcs_launcher import RCSLauncher


def parse_args():
    """CLI 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="RCS(Remote Control System) 자동 로그인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m automation.rcs.run_login --debug
  python -m automation.rcs.run_login --server "MyServer" --username "user1" --password "pass1"
  python -m automation.rcs.run_login --exe-path "D:\\RCS\\RcsMainHD.exe" --backend win32
  python -m automation.rcs.run_login --visual-debug --server "MyServer"
  python -m automation.rcs.run_login --visual-debug --debug-delay 2.0
  python -m automation.rcs.run_login --interaction-mode message
        """,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드: 컨트롤 트리만 출력하고 종료 (초기 설정용)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="접속할 서버 이름 (ComboBox 선택값)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="로그인 사용자 ID",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="로그인 비밀번호",
    )
    parser.add_argument(
        "--exe-path",
        type=str,
        default=None,
        help="RCS 실행 파일 경로",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["uia", "win32"],
        default=None,
        help="pywinauto 백엔드 (기본: uia)",
    )
    parser.add_argument(
        "--visual-debug",
        action="store_true",
        help="시각 디버그 모드: 컨트롤 하이라이트 + 좌표 로깅 + 딜레이 (--interaction-mode visual_debug 단축)",
    )
    parser.add_argument(
        "--interaction-mode",
        type=str,
        choices=["message", "input", "visual_debug"],
        default=None,
        help="인터랙션 모드 (기본: input)",
    )
    parser.add_argument(
        "--debug-delay",
        type=float,
        default=None,
        help="시각 디버그 모드에서 액션 간 대기 시간 초 (기본: 1.0)",
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 설정 생성
    config = RCSConfig()

    # CLI 인자로 설정 오버라이드
    if args.server:
        config.server_name = args.server
    if args.username:
        config.username = args.username
    if args.password:
        config.password = args.password
    if args.exe_path:
        config.exe_path = args.exe_path
    if args.backend:
        config.backend = args.backend
    if args.visual_debug:
        config.interaction_mode = "visual_debug"
    if args.interaction_mode:
        config.interaction_mode = args.interaction_mode
    if args.debug_delay is not None:
        config.visual_debug_delay_sec = args.debug_delay

    # 런처 생성 및 실행
    launcher = RCSLauncher(config=config)

    if args.debug:
        # 디버그 모드: 컨트롤 트리 출력
        success = launcher.debug_controls()
        sys.exit(0 if success else 1)

    # 일반 모드: 로그인 실행
    success = launcher.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
