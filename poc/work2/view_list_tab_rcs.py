"""로그인 후 메인 RCS 창의 View/List 탭 좌표를 찾고 순서대로 클릭한다.

사용법:
  1. 로그인까지 완료해서 `RCS - ...` 메인 창이 떠 있는 상태로 둔다.
  2. uv run python poc/work2/view_list_tab_rcs.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.work2.logger import log_work2_event
from poc.work2.ui_venus_mai_locator import (
    EXIT_SUCCESS,
    TargetConfig,
    TargetResult,
    analyze_window_target,
)
from poc.work2.util import (
    debug_image_path,
    foreground_window,
    format_elapsed_ms,
    make_timestamp_tag,
)
from poc.work2.util.debug_image_utils import save_debug_json

try:
    from pynput.mouse import Button, Controller as MouseController

    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False
    print("[WARNING] pynput.mouse 미설치 — 클릭 동작은 로그만 출력됩니다.")

load_dotenv()


TARGET_KEYS = ("view_tab", "list_tab")
ACTION_SEQUENCE = ("view_tab", "list_tab")

PREDEFINED_TARGETS: dict[str, TargetConfig] = {
    "view_tab": TargetConfig(
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
    ),
    "list_tab": TargetConfig(
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
    ),
}

DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

EXIT_MAIN_WINDOW_NOT_FOUND = "main_window_not_found"
EXIT_PARTIAL_SUCCESS = "partial_success"

PRE_CLICK_SETTLE_SEC = 0.2
POST_CLICK_SETTLE_SEC = 0.3
VIEW_TO_LIST_WAIT_SEC = 2.0


def analyze_main_tab_targets(
    main_window,
    window_title: str,
    backend: str,
) -> dict[str, TargetResult]:
    """메인 창에서 View/List 탭을 순차적으로 찾는다."""
    results: dict[str, TargetResult] = {}

    for target_key in TARGET_KEYS:
        target = PREDEFINED_TARGETS[target_key]
        print(f"\n[INFO] === 메인 탭 탐지 시작: {target_key} ===")
        results[target_key] = analyze_window_target(
            main_window,
            window_title,
            backend,
            target,
            debug_image_dir=DEBUG_IMAGE_DIR,
            log_name=LOG_NAME,
            component_name=COMPONENT_NAME,
            artifact_prefix=f"view_list_tab_rcs_{target_key}",
            result_mode="ui_venus_then_mai_ui_main_tabs",
        )

    return results


def _image_point_to_screen(window, image_point: dict) -> dict[str, int] | None:
    """이미지 픽셀 좌표를 스크린 절대 좌표로 변환한다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[ERROR] 창 rectangle 조회 실패: {exc}")
        return None

    return {
        "x": rect.left + image_point["x"],
        "y": rect.top + image_point["y"],
    }


def _click_at_screen(screen_point: dict, target_key: str) -> bool:
    """스크린 좌표에서 마우스 좌클릭을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not PYNPUT_MOUSE_AVAILABLE:
        print(f"[INFO] [DRY-RUN] 클릭 생략 (pynput 없음): target={target_key}, screen=({sx}, {sy})")
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.click(Button.left, 1)
    print(f"[INFO] 클릭 완료: target={target_key}, screen=({sx}, {sy})")
    return True


def perform_tab_actions(
    main_window,
    results: dict[str, TargetResult],
) -> list[dict]:
    """View 클릭 후 2초 대기하고 List 를 클릭한다."""
    action_results: list[dict] = []

    print(f"\n[INFO] === 탭 클릭 액션 시작 ({len(ACTION_SEQUENCE)}단계) ===")
    for index, target_key in enumerate(ACTION_SEQUENCE):
        result = results.get(target_key)
        if result is None or result.exit_code != EXIT_SUCCESS or result.point is None:
            print(f"[WARNING] 클릭 대상 없음: target={target_key}")
            action_results.append({"target": target_key, "clicked": False})
            continue

        foreground_window(
            main_window,
            debug_label=f"pre_click_{target_key}",
        )
        time.sleep(PRE_CLICK_SETTLE_SEC)

        screen_point = _image_point_to_screen(main_window, result.point)
        if screen_point is None:
            print(f"[ERROR] 스크린 좌표 변환 실패: target={target_key}")
            action_results.append({"target": target_key, "clicked": False})
            continue

        print(
            f"[INFO] 클릭 실행: target={target_key}, "
            f"image=({result.point['x']}, {result.point['y']}), "
            f"screen=({screen_point['x']}, {screen_point['y']})"
        )
        clicked = _click_at_screen(screen_point, target_key)
        action_results.append({"target": target_key, "clicked": clicked})

        if target_key == "view_tab" and index < len(ACTION_SEQUENCE) - 1:
            print(f"[INFO] View 클릭 후 대기: {VIEW_TO_LIST_WAIT_SEC:.1f}s")
            time.sleep(VIEW_TO_LIST_WAIT_SEC)
        else:
            time.sleep(POST_CLICK_SETTLE_SEC)

    return action_results


def _save_summary(
    results: dict[str, TargetResult],
    action_results: list[dict],
    window_title: str,
    backend: str,
    started_at: float,
) -> Path:
    """실행 요약 JSON 을 저장한다."""
    debug_stamp = make_timestamp_tag(started_at)
    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "view_list_tab_rcs_summary.json",
        timestamp_tag=debug_stamp,
    )
    payload = {
        "window_title": window_title,
        "backend": backend,
        "results": {
            target_key: {
                "exit_code": result.exit_code,
                "point": result.point,
            }
            for target_key, result in results.items()
        },
        "action_results": action_results,
    }
    save_debug_json(summary_path, payload)
    return summary_path


def main() -> str:
    """메인 RCS 창의 View/List 탭을 탐지하고 클릭 액션을 수행한다."""
    script_started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        target_keys=list(TARGET_KEYS),
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

    results = analyze_main_tab_targets(main_window, window_title, backend)
    success_count = sum(
        1 for result in results.values()
        if result.exit_code == EXIT_SUCCESS and result.point is not None
    )
    for target_key in TARGET_KEYS:
        result = results[target_key]
        if result.point is None:
            print(f"[WARNING] {target_key} 탐지 실패: exit_code={result.exit_code}")
            continue
        print(
            f"[INFO] {target_key} 탐지 성공: "
            f"image_point=({result.point['x']}, {result.point['y']})"
        )

    action_results = perform_tab_actions(main_window, results)
    summary_path = _save_summary(
        results,
        action_results,
        window_title,
        backend,
        script_started_at,
    )

    action_success_count = sum(1 for result in action_results if result["clicked"])
    print(f"[INFO] 요약 JSON 저장: {summary_path}")
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}, "
        f"detection_success={success_count}/{len(TARGET_KEYS)}, "
        f"action_success={action_success_count}/{len(action_results)}"
    )

    exit_code = (
        EXIT_SUCCESS
        if success_count == len(TARGET_KEYS) and action_success_count == len(ACTION_SEQUENCE)
        else EXIT_PARTIAL_SUCCESS
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=exit_code,
        window_title=window_title,
        backend=backend,
        success_count=success_count,
        target_count=len(TARGET_KEYS),
        action_results=action_results,
        action_success_count=action_success_count,
        summary_path=str(summary_path),
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
