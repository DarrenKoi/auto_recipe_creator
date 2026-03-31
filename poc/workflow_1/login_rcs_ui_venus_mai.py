"""RCS 로그인 창 UI-Venus + MAI-UI 2단계 타겟팅 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
1) `ui-venus` 로 타겟 요소의 coarse bbox 를 찾은 뒤
2) 그 주변을 crop + 확대해서 `mai-ui` 로 refined click point 를 찾는다.

TargetConfig 를 교체하면 userid_input, password_input, login_button 등
임의의 GUI 요소를 동일한 파이프라인으로 찾을 수 있다.

사용법:
  1. uv run python poc/workflow_1/open_rcs.py
  2. uv run python poc/workflow_1/login_rcs_ui_venus_mai.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1.login_rcs_common import WINDOW_TITLE_PREFIX, find_login_window
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.ui_venus_mai_locator import (
    EXIT_SUCCESS,
    EXIT_WINDOW_ACTIVATE_FAILED,
    TargetConfig,
    TargetResult,
    analyze_window_target,
)
from poc.work2.util import format_elapsed_ms

load_dotenv()


PREDEFINED_TARGETS: dict[str, TargetConfig] = {
    "userid_input": TargetConfig(
        key="userid_input",
        description=(
            "the editable text field next to the 'User ID' label "
            "where a user would click to type their user ID"
        ),
    ),
    "password_input": TargetConfig(
        key="password_input",
        description=(
            "the editable password field next to the 'Password' label "
            "where a user would click to type their password"
        ),
    ),
    "server_input": TargetConfig(
        key="server_input",
        description="the server dropdown or combo box control in the first form row",
    ),
    "login_button": TargetConfig(
        key="login_button",
        description="the 'Log In' button near the bottom of the dialog",
        left_pad_ratio=0.6,
        right_pad_ratio=0.6,
        vertical_pad_ratio=1.0,
    ),
    "cancel_button": TargetConfig(
        key="cancel_button",
        description="the 'Cancel' button near the bottom of the dialog",
        left_pad_ratio=0.6,
        right_pad_ratio=0.6,
        vertical_pad_ratio=1.0,
    ),
}

ACTIVE_TARGET_KEY = "userid_input"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = EXIT_WINDOW_ACTIVATE_FAILED


def analyze_login_target(
    login_window,
    window_title: str,
    backend: str,
    target: TargetConfig,
    image=None,
) -> TargetResult:
    """로그인 창에서 지정된 타겟을 2단계로 찾는다.

    image 가 주어지면 창 캡처를 건너뛰고 해당 이미지를 재사용한다.
    """
    return analyze_window_target(
        login_window,
        window_title,
        backend,
        target,
        debug_image_dir=DEBUG_IMAGE_DIR,
        log_name=LOG_NAME,
        component_name=COMPONENT_NAME,
        artifact_prefix="login_rcs",
        result_mode="ui_venus_then_mai_ui_single_target",
        image=image,
    )


def main() -> str:
    """이미 열려 있는 로그인 창에서 ACTIVE_TARGET_KEY 타겟을 찾는다."""
    target = PREDEFINED_TARGETS[ACTIVE_TARGET_KEY]

    script_started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        coarse_service="ui-venus",
        refine_service="mai-ui",
        target_key=target.key,
    )

    login_window, window_title, backend = find_login_window()
    if login_window is None:
        print(
            "[ERROR] 이미 떠 있는 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="login_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=WINDOW_TITLE_PREFIX,
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    result = analyze_login_target(login_window, window_title, backend, target)
    print(f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}")
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=result.exit_code,
        target_key=target.key,
        window_title=window_title,
        backend=backend,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
