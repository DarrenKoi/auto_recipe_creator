"""RCS List 탭에서 특정 Tool row 의 DVR 아이콘을 클릭한다."""

import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1 import workflow_select_tool as base_select_tool
from poc.workflow_1.debug_artifacts import debug_image_path, save_debug_json
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.login_rcs_common import wait_for_rcs_main_window
from poc.workflow_1.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_1.util import (
    WINDOW_UTILS_AVAILABLE,
    click_at_screen,
    collect_window_rows,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
    make_timestamp_tag,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

load_dotenv()


@dataclass
class ToolCCTVSelectionResult:
    """Tool row 의 DVR 아이콘 클릭 결과."""

    exit_code: str
    target_tool_name: str
    matched_lines: list[str] = field(default_factory=list)
    ocr_target_visible: bool = False
    list_crop_box: dict | None = None
    dvr_icon_point_on_list_crop: dict | None = None
    dvr_icon_point_on_full_image: dict | None = None
    dvr_icon_point_on_screen: dict | None = None
    clicked: bool = False
    dvr_window_verified: bool = False
    detected_player_windows: list[dict] = field(default_factory=list)


DEFAULT_TARGET_TOOL_NAME = base_select_tool.DEFAULT_TARGET_TOOL_NAME
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "workflow_select_tool_cctv"
LOG_NAME = "workflow_select_tool_cctv"
COMPONENT_NAME = LOG_NAME
DEFAULT_ACTION_ENABLED = base_select_tool.DEFAULT_ACTION_ENABLED

PLAYER_PROCESS_NAMES = (
    "whPlayer.exe",
    "RemotePlayer.exe",
    "swPlayer.exe",
)
PLAYER_PROCESS_NAME_SET = {name.lower() for name in PLAYER_PROCESS_NAMES}

EXIT_SUCCESS = DETECT_SUCCESS
EXIT_MAIN_WINDOW_NOT_FOUND = base_select_tool.EXIT_MAIN_WINDOW_NOT_FOUND
EXIT_WINDOW_ACTIVATE_FAILED = base_select_tool.EXIT_WINDOW_ACTIVATE_FAILED
EXIT_CAPTURE_FAILED = base_select_tool.EXIT_CAPTURE_FAILED
EXIT_TOOL_NAME_NOT_VISIBLE = base_select_tool.EXIT_TOOL_NAME_NOT_VISIBLE
EXIT_DVR_ICON_NOT_FOUND = "dvr_icon_not_found"
EXIT_OCR_REQUEST_ERROR = base_select_tool.EXIT_OCR_REQUEST_ERROR
EXIT_INVALID_TOOL_NAME = base_select_tool.EXIT_INVALID_TOOL_NAME
EXIT_INVALID_MAIN_WINDOW = base_select_tool.EXIT_INVALID_MAIN_WINDOW
EXIT_DVR_WINDOW_NOT_FOUND = "dvr_window_not_found"
EXIT_DVR_VERIFY_UNAVAILABLE = "dvr_verify_unavailable"

VERIFY_TIMEOUT_SEC = base_select_tool._env_float("SELECT_TOOL_CCTV_VERIFY_TIMEOUT_SEC", 8.0)
VERIFY_POLL_INTERVAL_SEC = base_select_tool._env_float(
    "SELECT_TOOL_CCTV_VERIFY_POLL_INTERVAL_SEC",
    0.5,
)


def _dvr_icon_target(tool_name: str) -> TargetConfig:
    """지정 Tool row 의 파란 DVR 아이콘을 찾기 위한 타겟 설정."""
    return TargetConfig(
        key="dvr_icon",
        description=(
            f"the blue circular DVR open icon on the same row as the tool name '{tool_name}' "
            f"in the left-side RCS tool list. The icon looks like a blue vinyl record or blue disc. "
            f"Return a safe click point near the center of that blue circle, not the tool text."
        ),
        left_pad_ratio=0.8,
        right_pad_ratio=3.0,
        vertical_pad_ratio=1.2,
        min_crop_width=360,
        min_crop_height=120,
    )


def _normalize_process_name(name: str) -> str:
    """process name 비교용 소문자 문자열을 반환한다."""
    return (name or "").strip().lower()


def _collect_player_windows() -> list[dict]:
    """현재 보이는 DVR player 창 목록을 수집한다."""
    if os.name != "nt":
        return []
    if not WINDOW_UTILS_AVAILABLE:
        return []
    if not PSUTIL_AVAILABLE:
        return []

    process_name_by_pid: dict[int, str] = {}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            name = str(proc.info.get("name") or "").strip()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
            continue

        if pid <= 0:
            continue
        if _normalize_process_name(name) not in PLAYER_PROCESS_NAME_SET:
            continue
        process_name_by_pid[pid] = name

    if not process_name_by_pid:
        return []

    matches: list[dict] = []
    for row in collect_window_rows(visible_only=True):
        process_name = process_name_by_pid.get(row.process_id)
        if not process_name:
            continue
        matches.append(
            {
                "title": row.title,
                "handle": row.handle,
                "process_id": row.process_id,
                "process_name": process_name,
            }
        )
    return matches


def _wait_for_new_player_window(
    existing_windows: list[dict],
    *,
    timeout_sec: float = VERIFY_TIMEOUT_SEC,
    poll_interval_sec: float = VERIFY_POLL_INTERVAL_SEC,
) -> tuple[bool, list[dict]]:
    """새 DVR player 창이 나타날 때까지 폴링한다."""
    if os.name != "nt":
        print("[WARNING] DVR player 창 검증은 Windows 에서만 지원합니다.")
        return False, []
    if not WINDOW_UTILS_AVAILABLE or not PSUTIL_AVAILABLE:
        print(
            "[WARNING] DVR player 창 검증 불가: "
            f"window_utils={WINDOW_UTILS_AVAILABLE}, psutil={PSUTIL_AVAILABLE}"
        )
        return False, []

    baseline_handles = {int(item["handle"]) for item in existing_windows}
    baseline_pids = {int(item["process_id"]) for item in existing_windows}
    deadline = time.time() + max(0.1, timeout_sec)
    attempt = 0
    latest_windows = existing_windows

    print(
        "[INFO] DVR player 창 대기 시작: "
        f"timeout={timeout_sec}s, poll_interval={poll_interval_sec}s, "
        f"player_processes={PLAYER_PROCESS_NAMES}"
    )

    while time.time() < deadline:
        attempt += 1
        latest_windows = _collect_player_windows()
        for item in latest_windows:
            if not baseline_handles and not baseline_pids:
                print(
                    f"[INFO] DVR player 창 발견 (attempt={attempt}): "
                    f"title={item['title']!r}, process={item['process_name']}, pid={item['process_id']}"
                )
                return True, latest_windows

            if int(item["handle"]) not in baseline_handles or int(item["process_id"]) not in baseline_pids:
                print(
                    f"[INFO] 새 DVR player 창 발견 (attempt={attempt}): "
                    f"title={item['title']!r}, process={item['process_name']}, pid={item['process_id']}"
                )
                return True, latest_windows

        remaining_sec = deadline - time.time()
        if remaining_sec <= 0:
            break
        time.sleep(min(max(0.1, poll_interval_sec), remaining_sec))

    print(
        "[WARNING] DVR player 창 타임아웃: "
        f"{timeout_sec}s 내 미발견 (attempts={attempt}, player_processes={PLAYER_PROCESS_NAMES})"
    )
    return False, latest_windows


def select_tool_cctv_from_main_window(
    main_window,
    window_title: str,
    backend: str,
    tool_name: str,
    *,
    action_enabled: bool = True,
    image=None,
    pre_click_settle_sec: float = 0.2,
    post_click_settle_sec: float = 0.5,
    verify_timeout_sec: float = VERIFY_TIMEOUT_SEC,
    verify_poll_interval_sec: float = VERIFY_POLL_INTERVAL_SEC,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> ToolCCTVSelectionResult:
    """현재 List 탭에서 지정 Tool row 의 DVR 아이콘을 클릭한다."""
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    normalized_tool_name = tool_name.strip()
    if not normalized_tool_name:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_INVALID_TOOL_NAME,
            target_tool_name=tool_name,
        )
    if not base_select_tool._is_valid_main_window_title(window_title):
        return ToolCCTVSelectionResult(
            exit_code=EXIT_INVALID_MAIN_WINDOW,
            target_tool_name=normalized_tool_name,
        )

    started_at = time.time()
    timestamp_tag = make_timestamp_tag(started_at)
    main_image = image or base_select_tool._capture_main_window(main_window, window_title, backend)
    if main_image is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
        )

    full_w, full_h = main_image.size
    list_crop_box = base_select_tool._build_relative_crop_box(
        full_w,
        full_h,
        base_select_tool.LIST_REGION_LEFT_RATIO,
        base_select_tool.LIST_REGION_TOP_RATIO,
        base_select_tool.LIST_REGION_RIGHT_RATIO,
        base_select_tool.LIST_REGION_BOTTOM_RATIO,
    )
    list_image = base_select_tool.crop_image(main_image, list_crop_box)

    try:
        ocr_result = base_select_tool._run_list_ocr(
            list_image,
            normalized_tool_name,
            timestamp_tag,
            window_title,
            backend,
            debug_image_dir=resolved_debug_dir,
            log_name=log_name,
        )
    except Exception as exc:
        log_work2_event(
            component=component_name,
            message="ocr_request_failed",
            level="error",
            log_name=log_name,
            target_tool_name=normalized_tool_name,
            error=exc,
        )
        return ToolCCTVSelectionResult(
            exit_code=EXIT_OCR_REQUEST_ERROR,
            target_tool_name=normalized_tool_name,
            list_crop_box=list_crop_box,
        )

    if not ocr_result["target_visible"]:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_TOOL_NAME_NOT_VISIBLE,
            target_tool_name=normalized_tool_name,
            matched_lines=ocr_result["matched_lines"],
            ocr_target_visible=False,
            list_crop_box=list_crop_box,
        )

    icon_result = analyze_window_target(
        main_window,
        window_title,
        backend,
        _dvr_icon_target(normalized_tool_name),
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix=f"workflow_select_tool_cctv_{normalized_tool_name.lower()}",
        result_mode="ui_venus_then_mai_ui_tool_list_dvr_icon",
        image=list_image,
    )
    if icon_result.exit_code != DETECT_SUCCESS or icon_result.point is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_DVR_ICON_NOT_FOUND,
            target_tool_name=normalized_tool_name,
            matched_lines=ocr_result["matched_lines"],
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
        )

    full_image_point = {
        "x": list_crop_box["left"] + icon_result.point["x"],
        "y": list_crop_box["top"] + icon_result.point["y"],
    }
    screen_point = image_point_to_screen(main_window, full_image_point)
    if screen_point is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=ocr_result["matched_lines"],
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            dvr_icon_point_on_list_crop=icon_result.point,
            dvr_icon_point_on_full_image=full_image_point,
        )

    if not foreground_window(
        main_window,
        debug_label=f"pre_click_dvr_icon_{normalized_tool_name}",
    ):
        return ToolCCTVSelectionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=ocr_result["matched_lines"],
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            dvr_icon_point_on_list_crop=icon_result.point,
            dvr_icon_point_on_full_image=full_image_point,
            dvr_icon_point_on_screen=screen_point,
        )

    existing_player_windows = _collect_player_windows()

    time.sleep(max(0.0, pre_click_settle_sec))
    clicked = click_at_screen(
        screen_point,
        f"{normalized_tool_name}_dvr_icon",
        click_count=1,
        action_enabled=action_enabled,
    )
    time.sleep(max(0.0, post_click_settle_sec))

    verify_exit_code = DETECT_SUCCESS
    dvr_window_verified = False
    detected_player_windows = existing_player_windows

    if action_enabled and clicked:
        if os.name != "nt" or not WINDOW_UTILS_AVAILABLE or not PSUTIL_AVAILABLE:
            verify_exit_code = EXIT_DVR_VERIFY_UNAVAILABLE
        else:
            dvr_window_verified, detected_player_windows = _wait_for_new_player_window(
                existing_player_windows,
                timeout_sec=verify_timeout_sec,
                poll_interval_sec=verify_poll_interval_sec,
            )
            if not dvr_window_verified:
                verify_exit_code = EXIT_DVR_WINDOW_NOT_FOUND
    elif clicked:
        verify_exit_code = DETECT_SUCCESS

    summary_path = debug_image_path(
        resolved_debug_dir,
        "workflow_select_tool_cctv_summary.json",
        timestamp_tag=timestamp_tag,
    )
    save_debug_json(
        summary_path,
        {
            "window_title": window_title,
            "backend": backend,
            "target_tool_name": normalized_tool_name,
            "list_crop_box": list_crop_box,
            "ocr_target_visible": ocr_result["target_visible"],
            "ocr_matched_lines": ocr_result["matched_lines"],
            "dvr_icon_point_on_list_crop": icon_result.point,
            "dvr_icon_point_on_full_image": full_image_point,
            "dvr_icon_point_on_screen": screen_point,
            "clicked": clicked,
            "action_enabled": action_enabled,
            "existing_player_windows_before_click": existing_player_windows,
            "detected_player_windows_after_click": detected_player_windows,
            "dvr_window_verified": dvr_window_verified,
            "verify_exit_code": verify_exit_code,
        },
    )

    return ToolCCTVSelectionResult(
        exit_code=verify_exit_code if clicked else EXIT_DVR_ICON_NOT_FOUND,
        target_tool_name=normalized_tool_name,
        matched_lines=ocr_result["matched_lines"],
        ocr_target_visible=True,
        list_crop_box=list_crop_box,
        dvr_icon_point_on_list_crop=icon_result.point,
        dvr_icon_point_on_full_image=full_image_point,
        dvr_icon_point_on_screen=screen_point,
        clicked=clicked,
        dvr_window_verified=dvr_window_verified,
        detected_player_windows=detected_player_windows,
    )


def main() -> str:
    """현재 List 탭에서 지정 Tool row 의 DVR 아이콘을 클릭한다."""
    started_at = time.time()
    target_tool_name = base_select_tool.load_target_tool_name(DEFAULT_TARGET_TOOL_NAME)

    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        target_tool_name=target_tool_name,
        ocr_service=base_select_tool.OCR_SERVICE_SLUG,
        action_enabled=DEFAULT_ACTION_ENABLED,
        player_processes=PLAYER_PROCESS_NAMES,
    )

    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print(
            "[ERROR] 메인 RCS 창을 찾지 못했습니다. "
            "먼저 로그인 후 List 탭까지 연 뒤 다시 실행하세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="main_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=base_select_tool.RCS_MAIN_WINDOW_TITLE_PREFIX,
        )
        return EXIT_MAIN_WINDOW_NOT_FOUND

    result = select_tool_cctv_from_main_window(
        main_window,
        window_title,
        backend,
        target_tool_name,
        action_enabled=DEFAULT_ACTION_ENABLED,
    )
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(started_at)}, "
        f"target_tool_name={target_tool_name!r}, result={result.exit_code}, "
        f"dvr_window_verified={result.dvr_window_verified}"
    )
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != DETECT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
