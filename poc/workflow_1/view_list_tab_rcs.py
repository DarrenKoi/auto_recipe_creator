"""로그인 후 메인 RCS 창의 List 탭 좌표를 찾고 클릭한다."""

import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import debug_image_path, save_debug_json
from poc.workflow_1.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_1.util import (
    click_at_screen,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
    make_timestamp_tag,
)

load_dotenv()


@dataclass
class MainTabActionResult:
    """메인 탭 클릭 결과."""

    exit_code: str
    target_key: str
    detected_point: dict | None = None
    screen_point: dict | None = None
    clicked: bool = False


LIST_TAB_TARGET = TargetConfig(
    key="list_tab",
    description=(
        "the 'List' tab in the top-left tab strip of the RCS main window. "
        "Use the first letter 'L' as the anchor, then click safely inside the List tab area."
    ),
    left_pad_ratio=0.9,
    right_pad_ratio=0.9,
    vertical_pad_ratio=2.0,
    min_crop_width=260,
    min_crop_height=140,
)

DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "view_list_tab_rcs"
LOG_NAME = "view_list_tab_rcs"
COMPONENT_NAME = LOG_NAME
DEFAULT_ACTION_ENABLED = os.getenv("ACTION_LOGIN_ACTION_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

EXIT_SUCCESS = DETECT_SUCCESS
EXIT_MAIN_WINDOW_NOT_FOUND = "main_window_not_found"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_SCREEN_POINT_FAILED = "screen_point_failed"
EXIT_TAB_NOT_FOUND = "tab_not_found"


def click_list_tab_in_main_window(
    main_window,
    window_title: str,
    backend: str,
    *,
    action_enabled: bool = True,
    image=None,
    pre_click_settle_sec: float = 0.2,
    post_click_settle_sec: float = 1.0,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> MainTabActionResult:
    """메인 창에서 List 탭을 찾아 클릭한다."""
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    detection = analyze_window_target(
        main_window,
        window_title,
        backend,
        LIST_TAB_TARGET,
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix="view_list_tab_rcs_list_tab",
        result_mode="ui_venus_then_mai_ui_main_tabs",
        image=image,
    )
    if detection.exit_code != DETECT_SUCCESS or detection.point is None:
        return MainTabActionResult(
            exit_code=EXIT_TAB_NOT_FOUND,
            target_key=LIST_TAB_TARGET.key,
        )

    screen_point = image_point_to_screen(main_window, detection.point)
    if screen_point is None:
        return MainTabActionResult(
            exit_code=EXIT_SCREEN_POINT_FAILED,
            target_key=LIST_TAB_TARGET.key,
            detected_point=detection.point,
        )

    if not foreground_window(
        main_window,
        debug_label=f"pre_click_{LIST_TAB_TARGET.key}",
    ):
        return MainTabActionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            target_key=LIST_TAB_TARGET.key,
            detected_point=detection.point,
            screen_point=screen_point,
        )

    time.sleep(max(0.0, pre_click_settle_sec))
    clicked = click_at_screen(
        screen_point,
        LIST_TAB_TARGET.key,
        click_count=1,
        action_enabled=action_enabled,
    )
    time.sleep(max(0.0, post_click_settle_sec))

    summary_path = debug_image_path(
        resolved_debug_dir,
        "view_list_tab_rcs_summary.json",
        timestamp_tag=make_timestamp_tag(time.time()),
    )
    save_debug_json(
        summary_path,
        {
            "window_title": window_title,
            "backend": backend,
            "target_key": LIST_TAB_TARGET.key,
            "detected_point": detection.point,
            "screen_point": screen_point,
            "detection_artifacts": detection.artifacts,
            "clicked": clicked,
            "action_enabled": action_enabled,
        },
    )

    return MainTabActionResult(
        exit_code=DETECT_SUCCESS if clicked else EXIT_TAB_NOT_FOUND,
        target_key=LIST_TAB_TARGET.key,
        detected_point=detection.point,
        screen_point=screen_point,
        clicked=clicked,
    )


def main() -> str:
    """메인 RCS 창의 List 탭을 탐지하고 클릭한다."""
    started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        action_enabled=DEFAULT_ACTION_ENABLED,
    )

    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print(
            "[ERROR] 메인 RCS 창을 찾지 못했습니다. "
            "먼저 로그인해서 메인 창을 띄운 뒤 다시 실행하세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="main_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=RCS_MAIN_WINDOW_TITLE_PREFIX,
        )
        return EXIT_MAIN_WINDOW_NOT_FOUND

    result = click_list_tab_in_main_window(
        main_window,
        window_title,
        backend,
        action_enabled=DEFAULT_ACTION_ENABLED,
    )
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(started_at)}, "
        f"result={result.exit_code}"
    )
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != DETECT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
