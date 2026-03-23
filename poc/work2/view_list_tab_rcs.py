"""로그인 후 메인 RCS 창의 View/List 탭 좌표를 찾는다.

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
from poc.work2.util import debug_image_path, format_elapsed_ms, make_timestamp_tag
from poc.work2.util.debug_image_utils import save_debug_json

load_dotenv()


TARGET_KEYS = ("view_tab", "list_tab")

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


def _save_summary(
    results: dict[str, TargetResult],
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
    }
    save_debug_json(summary_path, payload)
    return summary_path


def main() -> str:
    """메인 RCS 창의 View/List 탭 좌표를 찾는다."""
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
    summary_path = _save_summary(results, window_title, backend, script_started_at)

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

    print(f"[INFO] 요약 JSON 저장: {summary_path}")
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}, "
        f"success={success_count}/{len(TARGET_KEYS)}"
    )

    exit_code = EXIT_SUCCESS if success_count == len(TARGET_KEYS) else EXIT_PARTIAL_SUCCESS
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=exit_code,
        window_title=window_title,
        backend=backend,
        success_count=success_count,
        target_count=len(TARGET_KEYS),
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
