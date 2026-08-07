"""로그인 후 메인 RCS 창의 View/List 탭 좌표를 찾고 클릭한다.

단독 실행(`main()`)은 **View -> 대기 -> List 순서로 두 탭을 하나씩** 눌러 탭 전환이
실제로 먹는지까지 확인한다(로케이터 조합 A/B 의 점검용). 반면 production 로그인
경로(`workflow_login.py`)는 List 한 번만 필요하므로 `click_list_tab_in_main_window`
를 그대로 쓴다 - 두 용도가 같은 `click_main_tab` 을 공유하되 시퀀스는 분리한다.

사용법:
  1. 로그인까지 완료해서 `RCS - ...` 메인 창이 떠 있는 상태로 둔다.
  2. uv run python poc/workflow_3/rcs/view_list_tab_rcs.py
"""

import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import debug_image_path, save_debug_json
from poc.workflow_3.rcs.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.vlm.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_3.util import (
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


VIEW_TAB_TARGET = TargetConfig(
    key="view_tab",
    description=(
        "the 'View' tab in the top-left tab strip of the RCS main window. "
        "Use the first letter 'V' as the anchor, then click safely inside the View tab area."
    ),
    left_pad_ratio=0.9,
    right_pad_ratio=0.9,
    vertical_pad_ratio=2.0,
    min_crop_width=260,
    min_crop_height=140,
)

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

# 단독 실행 시 눌러볼 순서. View 를 먼저 눌러 탭을 옮겨 놓아야 List 클릭이 "이미
# 그 탭이었다" 가 아니라 실제 전환으로 확인된다.
TAB_ACTION_SEQUENCE = (VIEW_TAB_TARGET, LIST_TAB_TARGET)
# 탭 전환 후 다음 탭을 찾기 전 대기(초). RCS 가 탭 내용을 다시 그리는 동안 캡처하면
# coarse 단계가 전환 중인 화면을 보게 된다.
TAB_SWITCH_WAIT_SEC = 2.0

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
EXIT_PARTIAL_SUCCESS = "partial_success"  # 일부 탭만 성공 - 단독 실행 시퀀스 전용.


def click_main_tab(
    main_window,
    window_title: str,
    backend: str,
    target: TargetConfig,
    *,
    action_enabled: bool = True,
    image=None,
    pre_click_settle_sec: float = 0.2,
    post_click_settle_sec: float = 1.0,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> MainTabActionResult:
    """메인 창 상단 탭 스트립에서 지정 탭을 찾아 클릭한다.

    디버그 산출물 이름에 target.key 를 넣어 View/List 를 연달아 눌러도 파일이
    서로 덮이지 않게 한다.
    """
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    detection = analyze_window_target(
        main_window,
        window_title,
        backend,
        target,
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix=f"view_list_tab_rcs_{target.key}",
        result_mode="ui_venus_then_mai_ui_main_tabs",
        image=image,
    )
    if detection.exit_code != DETECT_SUCCESS or detection.point is None:
        return MainTabActionResult(
            exit_code=EXIT_TAB_NOT_FOUND,
            target_key=target.key,
        )

    screen_point = image_point_to_screen(main_window, detection.point)
    if screen_point is None:
        return MainTabActionResult(
            exit_code=EXIT_SCREEN_POINT_FAILED,
            target_key=target.key,
            detected_point=detection.point,
        )

    if not foreground_window(
        main_window,
        debug_label=f"pre_click_{target.key}",
    ):
        return MainTabActionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            target_key=target.key,
            detected_point=detection.point,
            screen_point=screen_point,
        )

    time.sleep(max(0.0, pre_click_settle_sec))
    clicked = click_at_screen(
        screen_point,
        target.key,
        click_count=1,
        action_enabled=action_enabled,
    )
    time.sleep(max(0.0, post_click_settle_sec))

    summary_path = debug_image_path(
        resolved_debug_dir,
        f"view_list_tab_rcs_{target.key}_summary.json",
        timestamp_tag=make_timestamp_tag(time.time()),
    )
    save_debug_json(
        summary_path,
        {
            "window_title": window_title,
            "backend": backend,
            "target_key": target.key,
            "detected_point": detection.point,
            "screen_point": screen_point,
            "detection_artifacts": detection.artifacts,
            "clicked": clicked,
            "action_enabled": action_enabled,
        },
    )

    return MainTabActionResult(
        exit_code=DETECT_SUCCESS if clicked else EXIT_TAB_NOT_FOUND,
        target_key=target.key,
        detected_point=detection.point,
        screen_point=screen_point,
        clicked=clicked,
    )


def click_list_tab_in_main_window(
    main_window,
    window_title: str,
    backend: str,
    **kwargs,
) -> MainTabActionResult:
    """메인 창에서 List 탭을 찾아 클릭한다(production 로그인 경로 전용 wrapper).

    로그인 직후에는 List 탭만 필요하므로 시퀀스를 돌지 않는다 - 단독 실행의
    View->List 순회는 main() 쪽에 있다.
    """
    return click_main_tab(
        main_window,
        window_title,
        backend,
        LIST_TAB_TARGET,
        **kwargs,
    )


def main() -> str:
    """메인 RCS 창의 View/List 탭을 하나씩 탐지하고 클릭한다.

    한 탭이라도 실패하면 나머지는 계속 시도하되 partial_success 로 끝낸다 -
    "View 는 되는데 List 만 안 된다" 같은 정보가 조합 A/B 진단에 필요하다.
    """
    started_at = time.time()

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

    results: list[MainTabActionResult] = []
    for idx, target in enumerate(TAB_ACTION_SEQUENCE):
        print(f"\n[INFO] === 메인 탭 {idx + 1}/{len(TAB_ACTION_SEQUENCE)}: {target.key} ===")
        result = click_main_tab(
            main_window,
            window_title,
            backend,
            target,
            action_enabled=DEFAULT_ACTION_ENABLED,
        )
        results.append(result)
        print(f"[INFO] {target.key} result={result.exit_code}, clicked={result.clicked}")
        if idx < len(TAB_ACTION_SEQUENCE) - 1:
            # 탭 전환 렌더링이 끝난 뒤 다음 탭을 캡처해야 coarse 가 전환 중 화면을
            # 보지 않는다.
            time.sleep(TAB_SWITCH_WAIT_SEC)

    summary = ", ".join(f"{r.target_key}={r.exit_code}" for r in results)
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(started_at)}, {summary}"
    )

    if all(r.exit_code == DETECT_SUCCESS for r in results):
        return DETECT_SUCCESS
    if any(r.exit_code == DETECT_SUCCESS for r in results):
        return EXIT_PARTIAL_SUCCESS
    return results[-1].exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != DETECT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
