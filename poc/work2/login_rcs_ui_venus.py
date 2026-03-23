"""Deprecated compatibility entrypoint for the removed login benchmark flow."""

import sys


EXIT_DEPRECATED = "deprecated_entrypoint"


def main() -> str:
    """안내 메시지를 출력하고 대체 스크립트를 알려준다."""
    print(
        "[WARNING] 기존 login benchmark 모듈 제거로 "
        "`login_rcs_ui_venus.py` 단일 모델 benchmark 흐름도 함께 종료되었습니다."
    )
    print(
        "[INFO] 로그인 타겟 분석은 "
        "`uv run python poc/work2/login_rcs_ui_venus_mai.py` 를 사용하세요."
    )
    print(
        "[INFO] 클릭 자동화까지 필요하면 "
        "`uv run python poc/work2/action_login.py` 를 사용하세요."
    )
    return EXIT_DEPRECATED


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
