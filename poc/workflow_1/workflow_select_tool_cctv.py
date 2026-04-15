"""RCS List 탭에서 특정 Tool row 의 DVR 아이콘을 더블클릭한다."""

import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1 import workflow_select_tool as base_select_tool
from poc.workflow_1.debug_artifacts import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
    save_marked_bboxes,
)
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
    """Tool row 의 DVR 아이콘 더블클릭 결과."""

    exit_code: str
    target_tool_name: str
    matched_lines: list[str] = field(default_factory=list)
    ocr_target_visible: bool = False
    list_crop_box: dict | None = None
    tool_point_on_list_crop: dict | None = None
    tool_point_on_full_image: dict | None = None
    selected_attempt: str | None = None
    dvr_icon_point_on_list_crop: dict | None = None
    dvr_icon_point_on_full_image: dict | None = None
    dvr_icon_point_on_screen: dict | None = None
    dvr_search_box_on_list_crop: dict | None = None
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
EXIT_TOOL_ROW_NOT_FOUND = base_select_tool.EXIT_TOOL_ROW_NOT_FOUND
EXIT_DVR_ICON_NOT_FOUND = "dvr_icon_not_found"
EXIT_OCR_REQUEST_ERROR = base_select_tool.EXIT_OCR_REQUEST_ERROR
EXIT_INVALID_TOOL_NAME = base_select_tool.EXIT_INVALID_TOOL_NAME
EXIT_INVALID_MAIN_WINDOW = base_select_tool.EXIT_INVALID_MAIN_WINDOW
EXIT_DVR_ICON_WRONG_ROW = "dvr_icon_wrong_row"
EXIT_DVR_WINDOW_NOT_FOUND = "dvr_window_not_found"
EXIT_DVR_VERIFY_UNAVAILABLE = "dvr_verify_unavailable"

VERIFY_TIMEOUT_SEC = base_select_tool._env_float("SELECT_TOOL_CCTV_VERIFY_TIMEOUT_SEC", 8.0)
VERIFY_POLL_INTERVAL_SEC = base_select_tool._env_float(
    "SELECT_TOOL_CCTV_VERIFY_POLL_INTERVAL_SEC",
    0.5,
)
DVR_SEARCH_ROW_HALF_HEIGHT_RATIO = base_select_tool._env_float(
    "SELECT_TOOL_CCTV_DVR_SEARCH_ROW_HALF_HEIGHT_RATIO",
    0.035,
)


def _env_int(name: str, default: int) -> int:
    """int 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


DVR_SEARCH_MIN_HEIGHT_PX = _env_int("SELECT_TOOL_CCTV_DVR_SEARCH_MIN_HEIGHT_PX", 120)
DVR_SEARCH_MIN_ROW_HALF_HEIGHT_PX = _env_int(
    "SELECT_TOOL_CCTV_DVR_SEARCH_MIN_ROW_HALF_HEIGHT_PX",
    56,
)
DVR_SEARCH_MAX_ROW_HALF_HEIGHT_PX = _env_int(
    "SELECT_TOOL_CCTV_DVR_SEARCH_MAX_ROW_HALF_HEIGHT_PX",
    120,
)
DVR_ICON_Y_TOLERANCE_PX = _env_int(
    "SELECT_TOOL_CCTV_DVR_ICON_Y_TOLERANCE_PX",
    40,
)


def _dvr_icon_target(tool_name: str) -> TargetConfig:
    """지정 Tool row 의 파란 DVR 아이콘을 찾기 위한 타겟 설정."""
    return TargetConfig(
        key="dvr_icon",
        description=(
            f"the blue circular DVR open icon for the tool row '{tool_name}'. "
            f"This image shows the full row of the RCS tool list — the tool name text is on the left "
            f"and the DVR icon is on the right side. First confirm the tool name '{tool_name}' is "
            f"visible on the left, then find the blue LP record / blue disc icon on the same row. "
            f"Return a safe click point near the center of that blue circle, not the tool text."
        ),
        left_pad_ratio=6.0,
        right_pad_ratio=1.0,
        vertical_pad_ratio=0.4,
        min_crop_width=600,
        min_crop_height=60,
    )


def _ensure_min_span(start: int, end: int, total: int, minimum: int) -> tuple[int, int]:
    """최소 길이를 만족하도록 구간을 확장한다."""
    span = end - start
    if span >= minimum:
        return start, end

    extra = minimum - span
    grow_before = extra // 2
    grow_after = extra - grow_before
    start = max(0, start - grow_before)
    end = min(total, end + grow_after)

    if end - start >= minimum:
        return start, end

    if start == 0:
        end = min(total, minimum)
    elif end == total:
        start = max(0, total - minimum)
    return start, end


def _build_dvr_row_strip_on_list_crop(
    tool_point_on_list_crop: dict,
    list_crop_width: int,
    list_crop_height: int,
) -> dict[str, int]:
    """tool row 기준으로 base_image (list crop) 에서 전체 폭 row strip 을 만든다.

    tool name 부터 DVR 아이콘까지 한 줄 전체가 보이도록 left=0 부터 시작한다.
    VLM 이 tool name 을 확인한 뒤 같은 행의 DVR 아이콘을 정확히 찾을 수 있다.
    """
    left = 0
    right = list_crop_width

    band_half_height = int(round(list_crop_height * max(0.0, DVR_SEARCH_ROW_HALF_HEIGHT_RATIO)))
    band_half_height = max(DVR_SEARCH_MIN_ROW_HALF_HEIGHT_PX, band_half_height)
    band_half_height = min(DVR_SEARCH_MAX_ROW_HALF_HEIGHT_PX, band_half_height)
    top = max(0, tool_point_on_list_crop["y"] - band_half_height)
    bottom = min(list_crop_height, tool_point_on_list_crop["y"] + band_half_height + 1)
    top, bottom = _ensure_min_span(top, bottom, list_crop_height, max(1, DVR_SEARCH_MIN_HEIGHT_PX))

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _point_to_tiny_bbox(point: dict, img_w: int, img_h: int, radius: int = 10) -> dict:
    """포인트를 overlay 용 작은 bbox 로 감싼다."""
    return {
        "left": max(0, point["x"] - radius),
        "top": max(0, point["y"] - radius),
        "right": min(img_w, point["x"] + radius + 1),
        "bottom": min(img_h, point["y"] + radius + 1),
    }


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
    """현재 List 탭에서 지정 Tool row 의 DVR 아이콘을 더블클릭한다."""
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

    full_capture_path = debug_image_path(
        resolved_debug_dir,
        "main_window_capture.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(main_image, full_capture_path)

    ocr_attempts, ocr_errors = base_select_tool._run_tool_list_ocr_attempts(
        main_image,
        normalized_tool_name,
        timestamp_tag,
        window_title,
        backend,
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
    )
    if not ocr_attempts and ocr_errors:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_OCR_REQUEST_ERROR,
            target_tool_name=normalized_tool_name,
            list_crop_box=ocr_errors[0]["crop_box"] if ocr_errors else None,
        )

    located_attempt, detection_attempts = base_select_tool._locate_tool_on_attempts(
        main_window,
        window_title,
        backend,
        normalized_tool_name,
        ocr_attempts,
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
    )

    visible_attempts = [attempt for attempt in ocr_attempts if attempt["ocr_result"]["target_visible"]]
    best_visible_attempt = visible_attempts[0] if visible_attempts else None
    selected_attempt = located_attempt["attempt"] if located_attempt is not None else best_visible_attempt
    list_crop_box = selected_attempt["crop_box"] if selected_attempt is not None else None
    matched_lines = (
        selected_attempt["ocr_result"]["matched_lines"]
        if selected_attempt is not None
        else []
    )

    if located_attempt is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_TOOL_ROW_NOT_FOUND if best_visible_attempt is not None else EXIT_TOOL_NAME_NOT_VISIBLE,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=best_visible_attempt is not None,
            list_crop_box=list_crop_box,
            selected_attempt=selected_attempt["name"] if selected_attempt is not None else None,
        )

    tool_point_on_list_crop = located_attempt["mapped_point"]
    tool_point_on_full_image = {
        "x": list_crop_box["left"] + tool_point_on_list_crop["x"],
        "y": list_crop_box["top"] + tool_point_on_list_crop["y"],
    }

    dvr_row_strip_on_list_crop = _build_dvr_row_strip_on_list_crop(
        tool_point_on_list_crop,
        selected_attempt["base_size"]["width"],
        selected_attempt["base_size"]["height"],
    )
    dvr_search_image = base_select_tool.crop_image(
        selected_attempt["base_image"],
        dvr_row_strip_on_list_crop,
    )

    icon_result = analyze_window_target(
        main_window,
        window_title,
        backend,
        _dvr_icon_target(normalized_tool_name),
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix=(
            f"workflow_select_tool_cctv_{normalized_tool_name.lower()}_"
            f"{selected_attempt['name']}_dvr"
        ),
        result_mode="ui_venus_then_mai_ui_tool_row_right_side_dvr_icon",
        image=dvr_search_image,
    )
    if icon_result.exit_code != DETECT_SUCCESS or icon_result.point is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_DVR_ICON_NOT_FOUND,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=tool_point_on_list_crop,
            tool_point_on_full_image=tool_point_on_full_image,
            selected_attempt=selected_attempt["name"],
            dvr_search_box_on_list_crop=dvr_row_strip_on_list_crop,
        )

    list_crop_point = {
        "x": dvr_row_strip_on_list_crop["left"] + icon_result.point["x"],
        "y": dvr_row_strip_on_list_crop["top"] + icon_result.point["y"],
    }

    y_delta = abs(list_crop_point["y"] - tool_point_on_list_crop["y"])
    if y_delta > DVR_ICON_Y_TOLERANCE_PX:
        print(
            f"[WARNING] DVR 아이콘 y 좌표가 tool row 와 불일치: "
            f"tool_y={tool_point_on_list_crop['y']}, dvr_icon_y={list_crop_point['y']}, "
            f"delta={y_delta}px > tolerance={DVR_ICON_Y_TOLERANCE_PX}px"
        )
        return ToolCCTVSelectionResult(
            exit_code=EXIT_DVR_ICON_WRONG_ROW,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=tool_point_on_list_crop,
            tool_point_on_full_image=tool_point_on_full_image,
            selected_attempt=selected_attempt["name"],
            dvr_icon_point_on_list_crop=list_crop_point,
            dvr_search_box_on_list_crop=dvr_row_strip_on_list_crop,
        )

    strip_w, strip_h = dvr_search_image.size
    tool_point_on_strip = {
        "x": tool_point_on_list_crop["x"] - dvr_row_strip_on_list_crop["left"],
        "y": tool_point_on_list_crop["y"] - dvr_row_strip_on_list_crop["top"],
    }
    dvr_icon_on_strip = icon_result.point
    overlay_items = {
        "tool_name": {
            "bbox": _point_to_tiny_bbox(tool_point_on_strip, strip_w, strip_h),
            "center": tool_point_on_strip,
        },
        "dvr_icon": {
            "bbox": _point_to_tiny_bbox(dvr_icon_on_strip, strip_w, strip_h),
            "center": dvr_icon_on_strip,
        },
    }
    overlay_colors = {
        "tool_name": "lime",
        "dvr_icon": "deepskyblue",
    }
    strip_overlay_path = debug_image_path(
        resolved_debug_dir,
        f"workflow_select_tool_cctv_{normalized_tool_name.lower()}_dvr_row_strip_overlay.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_marked_bboxes(dvr_search_image, overlay_items, overlay_colors, strip_overlay_path)

    full_image_point = {
        "x": list_crop_box["left"] + list_crop_point["x"],
        "y": list_crop_box["top"] + list_crop_point["y"],
    }
    screen_point = image_point_to_screen(main_window, full_image_point)
    dvr_search_box_on_list_crop = dvr_row_strip_on_list_crop
    if screen_point is None:
        return ToolCCTVSelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=tool_point_on_list_crop,
            tool_point_on_full_image=tool_point_on_full_image,
            selected_attempt=selected_attempt["name"],
            dvr_icon_point_on_list_crop=list_crop_point,
            dvr_icon_point_on_full_image=full_image_point,
            dvr_search_box_on_list_crop=dvr_search_box_on_list_crop,
        )

    if not foreground_window(
        main_window,
        debug_label=f"pre_click_dvr_icon_{normalized_tool_name}",
    ):
        return ToolCCTVSelectionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=True,
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=tool_point_on_list_crop,
            tool_point_on_full_image=tool_point_on_full_image,
            selected_attempt=selected_attempt["name"],
            dvr_icon_point_on_list_crop=list_crop_point,
            dvr_icon_point_on_full_image=full_image_point,
            dvr_icon_point_on_screen=screen_point,
            dvr_search_box_on_list_crop=dvr_search_box_on_list_crop,
        )

    existing_player_windows = _collect_player_windows()

    time.sleep(max(0.0, pre_click_settle_sec))
    clicked = click_at_screen(
        screen_point,
        f"{normalized_tool_name}_dvr_icon",
        click_count=2,
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
            "selected_attempt": selected_attempt["name"],
            "selected_attempt_resize_meta": selected_attempt["resize_meta"],
            "ocr_target_visible": selected_attempt["ocr_result"]["target_visible"],
            "ocr_matched_lines": matched_lines,
            "ocr_attempts": [
                {
                    "attempt_name": attempt["name"],
                    "crop_box": attempt["crop_box"],
                    "resize_meta": attempt["resize_meta"],
                    "ocr_target_visible": attempt["ocr_result"]["target_visible"],
                    "ocr_matched_lines": attempt["ocr_result"]["matched_lines"],
                }
                for attempt in ocr_attempts
            ],
            "ocr_errors": ocr_errors,
            "detection_attempts": detection_attempts,
            "tool_point_on_list_crop": tool_point_on_list_crop,
            "tool_point_on_full_image": tool_point_on_full_image,
            "dvr_search_box_on_list_crop": dvr_search_box_on_list_crop,
            "dvr_icon_point_on_list_crop": list_crop_point,
            "dvr_icon_point_on_full_image": full_image_point,
            "dvr_icon_point_on_screen": screen_point,
            "dvr_icon_y_delta_px": y_delta,
            "dvr_icon_y_tolerance_px": DVR_ICON_Y_TOLERANCE_PX,
            "double_clicked": clicked,
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
        matched_lines=matched_lines,
        ocr_target_visible=True,
        list_crop_box=list_crop_box,
        tool_point_on_list_crop=tool_point_on_list_crop,
        tool_point_on_full_image=tool_point_on_full_image,
        selected_attempt=selected_attempt["name"],
        dvr_icon_point_on_list_crop=list_crop_point,
        dvr_icon_point_on_full_image=full_image_point,
        dvr_icon_point_on_screen=screen_point,
        dvr_search_box_on_list_crop=dvr_search_box_on_list_crop,
        clicked=clicked,
        dvr_window_verified=dvr_window_verified,
        detected_player_windows=detected_player_windows,
    )


def main() -> str:
    """현재 List 탭에서 지정 Tool row 의 DVR 아이콘을 더블클릭한다."""
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
