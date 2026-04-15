"""DVR player 창에서 Channel 4 를 더블클릭하여 전체화면으로 확대한다."""

import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1 import workflow_select_tool as base_select_tool
from poc.workflow_1 import workflow_select_tool_cctv as cctv_workflow
from poc.workflow_1.debug_artifacts import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
)
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.record_screen_ch4 import record_screen
from poc.workflow_1.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_1.util import (
    WINDOW_UTILS_AVAILABLE,
    click_at_screen,
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

LOG_NAME = "workflow_select_ch4_cctv"
COMPONENT_NAME = LOG_NAME
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "workflow_select_ch4_cctv"
DEFAULT_ACTION_ENABLED = base_select_tool.DEFAULT_ACTION_ENABLED

PLAYER_PROCESS_NAMES = cctv_workflow.PLAYER_PROCESS_NAMES
PLAYER_PROCESS_NAME_SET = cctv_workflow.PLAYER_PROCESS_NAME_SET

EXIT_SUCCESS = DETECT_SUCCESS
EXIT_PLAYER_WINDOW_NOT_FOUND = "player_window_not_found"
EXIT_PLAYER_UTILS_UNAVAILABLE = "player_utils_unavailable"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_CH4_NOT_FOUND = "ch4_not_found"
EXIT_CLICK_FAILED = "click_failed"
EXIT_RECORD_FAILED = "record_failed"

PRE_CLICK_SETTLE_SEC = base_select_tool._env_float(
    "SELECT_CH4_PRE_CLICK_SETTLE_SEC", 0.2,
)
POST_CLICK_SETTLE_SEC = base_select_tool._env_float(
    "SELECT_CH4_POST_CLICK_SETTLE_SEC", 0.5,
)


@dataclass
class Ch4SelectionResult:
    """Channel 4 더블클릭 결과."""

    exit_code: str
    player_window_title: str = ""
    player_process_name: str = ""
    ch4_point_on_full_image: dict | None = None
    ch4_point_on_screen: dict | None = None
    clicked: bool = False


def _normalize_process_name(name: str) -> str:
    """process name 비교용 소문자 문자열을 반환한다."""
    return (name or "").strip().lower()


def _ch4_target() -> TargetConfig:
    """DVR player 내 Channel 4 를 찾기 위한 타겟 설정."""
    return TargetConfig(
        key="ch4_cctv",
        description=(
            "Channel 4 (CH 4, CAM 4, or the fourth camera view) in this multi-channel "
            "DVR / CCTV player window. The window shows a grid of camera channels "
            "(typically 2x2). Channel 4 is normally in the bottom-right quadrant. "
            "It is the channel that shows the tool's software screen with text like "
            "'Image', 'Function', 'Queue', or similar UI elements. "
            "The other channels (1, 2, 3) usually show the physical tool or chamber. "
            "If all channels appear black or off, pick the bottom-right quadrant. "
            "Return a safe double-click point at the center of Channel 4's viewport."
        ),
        left_pad_ratio=1.0,
        right_pad_ratio=1.0,
        vertical_pad_ratio=0.5,
        min_crop_width=200,
        min_crop_height=200,
    )


def _find_player_window() -> tuple[object | None, str, str, str]:
    """현재 열려 있는 DVR player 창을 찾아 pywinauto wrapper 로 반환한다.

    Returns:
        (window, window_title, backend, process_name) 또는 (None, "", "", "")
    """
    if os.name != "nt":
        return None, "", "", ""
    if not WINDOW_UTILS_AVAILABLE or not PSUTIL_AVAILABLE:
        return None, "", "", ""

    from poc.workflow_1.util import collect_window_rows

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
        print(
            f"[WARNING] DVR player 프로세스 미발견: "
            f"expected={PLAYER_PROCESS_NAMES}"
        )
        return None, "", "", ""

    rows = collect_window_rows(visible_only=True)
    for row in rows:
        pname = process_name_by_pid.get(row.process_id)
        if not pname:
            continue

        from pywinauto import Desktop

        for backend in ("uia", "win32"):
            try:
                desktop = Desktop(backend=backend)
                window = desktop.window(handle=row.handle).wrapper_object()
            except Exception:
                continue

            print(
                f"[INFO] DVR player 창 발견: "
                f"title={row.title!r}, process={pname}, pid={row.process_id}, "
                f"backend={backend}"
            )
            return window, row.title, backend, pname

    print(
        f"[WARNING] DVR player 창 미발견 (프로세스는 존재): "
        f"pids={list(process_name_by_pid.keys())}"
    )
    return None, "", "", ""


def select_ch4_from_player_window(
    player_window,
    window_title: str,
    backend: str,
    process_name: str,
    *,
    action_enabled: bool = True,
    pre_click_settle_sec: float = PRE_CLICK_SETTLE_SEC,
    post_click_settle_sec: float = POST_CLICK_SETTLE_SEC,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> Ch4SelectionResult:
    """DVR player 창에서 Channel 4 를 찾아 더블클릭한다."""
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    started_at = time.time()
    timestamp_tag = make_timestamp_tag(started_at)

    from poc.workflow_1.util import capture_window

    if not foreground_window(
        player_window,
        debug_label=f"dvr_player_{process_name}",
    ):
        return Ch4SelectionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            player_window_title=window_title,
            player_process_name=process_name,
        )

    try:
        player_image = capture_window(player_window)
    except Exception as exc:
        print(f"[ERROR] DVR player 창 캡처 실패: {exc}")
        return Ch4SelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            player_window_title=window_title,
            player_process_name=process_name,
        )

    capture_path = debug_image_path(
        resolved_debug_dir,
        "player_window_capture.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(player_image, capture_path)

    ch4_result = analyze_window_target(
        player_window,
        window_title,
        backend,
        _ch4_target(),
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix="workflow_select_ch4_cctv",
        result_mode="ui_venus_then_mai_ui_ch4_cctv",
        image=player_image,
    )

    if ch4_result.exit_code != DETECT_SUCCESS or ch4_result.point is None:
        print(
            f"[WARNING] Channel 4 위치 감지 실패: "
            f"exit_code={ch4_result.exit_code}, title={window_title!r}"
        )
        return Ch4SelectionResult(
            exit_code=EXIT_CH4_NOT_FOUND,
            player_window_title=window_title,
            player_process_name=process_name,
        )

    ch4_full_point = ch4_result.point
    ch4_screen_point = image_point_to_screen(player_window, ch4_full_point)
    if ch4_screen_point is None:
        return Ch4SelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            player_window_title=window_title,
            player_process_name=process_name,
            ch4_point_on_full_image=ch4_full_point,
        )

    if not foreground_window(
        player_window,
        debug_label=f"pre_dblclick_ch4_{process_name}",
    ):
        return Ch4SelectionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            player_window_title=window_title,
            player_process_name=process_name,
            ch4_point_on_full_image=ch4_full_point,
            ch4_point_on_screen=ch4_screen_point,
        )

    time.sleep(max(0.0, pre_click_settle_sec))
    clicked = click_at_screen(
        ch4_screen_point,
        "ch4_cctv_enlarge",
        click_count=2,
        action_enabled=action_enabled,
    )
    time.sleep(max(0.0, post_click_settle_sec))

    summary_path = debug_image_path(
        resolved_debug_dir,
        "workflow_select_ch4_cctv_summary.json",
        timestamp_tag=timestamp_tag,
    )
    save_debug_json(
        summary_path,
        {
            "window_title": window_title,
            "backend": backend,
            "process_name": process_name,
            "ch4_point_on_full_image": ch4_full_point,
            "ch4_point_on_screen": ch4_screen_point,
            "double_clicked": clicked,
            "action_enabled": action_enabled,
            "elapsed_ms": f"{(time.time() - started_at) * 1000:.1f}",
        },
    )

    exit_code = EXIT_SUCCESS if clicked else EXIT_CLICK_FAILED
    print(
        f"[INFO] Channel 4 더블클릭 {'완료' if clicked else '실패'}: "
        f"screen=({ch4_screen_point['x']}, {ch4_screen_point['y']}), "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return Ch4SelectionResult(
        exit_code=exit_code,
        player_window_title=window_title,
        player_process_name=process_name,
        ch4_point_on_full_image=ch4_full_point,
        ch4_point_on_screen=ch4_screen_point,
        clicked=clicked,
    )


def main() -> str:
    """DVR player 창에서 Channel 4 를 더블클릭하여 전체화면으로 확대한다."""
    started_at = time.time()

    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        action_enabled=DEFAULT_ACTION_ENABLED,
        player_processes=PLAYER_PROCESS_NAMES,
    )

    if os.name != "nt" or not WINDOW_UTILS_AVAILABLE or not PSUTIL_AVAILABLE:
        print(
            "[ERROR] DVR player 창 탐색은 Windows + psutil + window_utils 가 필요합니다. "
            f"os={os.name}, window_utils={WINDOW_UTILS_AVAILABLE}, "
            f"psutil={PSUTIL_AVAILABLE}"
        )
        return EXIT_PLAYER_UTILS_UNAVAILABLE

    player_window, window_title, backend, process_name = _find_player_window()
    if player_window is None:
        print(
            "[ERROR] DVR player 창을 찾지 못했습니다. "
            "먼저 workflow_select_tool_cctv 를 실행하여 DVR 을 열어 주세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="player_window_not_found",
            level="error",
            log_name=LOG_NAME,
            player_processes=PLAYER_PROCESS_NAMES,
        )
        return EXIT_PLAYER_WINDOW_NOT_FOUND

    result = select_ch4_from_player_window(
        player_window,
        window_title,
        backend,
        process_name,
        action_enabled=DEFAULT_ACTION_ENABLED,
    )

    if result.exit_code == EXIT_SUCCESS and result.clicked:
        print("[INFO] Channel 4 클릭 성공 — 화면 녹화를 시작합니다.")
        log_work2_event(
            component=COMPONENT_NAME,
            message="ch4_click_success_record_start",
            log_name=LOG_NAME,
            window_title=window_title,
            process_name=process_name,
        )
        recording_path = record_screen(
            output_stem=f"ch4_cctv_{process_name or 'player'}",
            log_name=LOG_NAME,
            component_name=COMPONENT_NAME,
        )
        if recording_path is None:
            print("[ERROR] Channel 4 클릭 후 화면 녹화 실패")
            return EXIT_RECORD_FAILED
        print(f"[INFO] Channel 4 화면 녹화 저장 완료: {recording_path}")

    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(started_at)}, "
        f"title={window_title!r}, result={result.exit_code}, "
        f"clicked={result.clicked}"
    )
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != DETECT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
