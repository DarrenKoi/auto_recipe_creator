"""RCS List 탭에서 특정 Tool 이름을 찾아 더블클릭한다."""

import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
    save_debug_text,
    save_debug_webp,
    save_marked_bboxes,
)
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.workflow_1.prompts import build_ocr_assist_prompt
from poc.workflow_1.ui_venus_mai_locator import (
    EXIT_SUCCESS as DETECT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
from poc.workflow_1.util import (
    activate_window,
    capture_window,
    click_at_screen,
    crop_image,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
    make_timestamp_tag,
    point_to_tiny_bbox,
)
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()


@dataclass
class ToolSelectionResult:
    """Tool row 더블클릭 결과."""

    exit_code: str
    target_tool_name: str
    matched_lines: list[str] = field(default_factory=list)
    ocr_target_visible: bool = False
    list_crop_box: dict | None = None
    tool_point_on_list_crop: dict | None = None
    tool_point_on_full_image: dict | None = None
    tool_point_on_screen: dict | None = None
    double_clicked: bool = False
    selected_attempt: str | None = None
    click_overlay_path: str | None = None


@dataclass
class ToolListVisibilityResult:
    """List 탭 가시성 검증 결과."""

    exit_code: str
    target_tool_name: str
    matched_lines: list[str] = field(default_factory=list)
    target_visible: bool = False
    list_crop_box: dict | None = None
    selected_attempt: str | None = None
    visibility_source: str | None = None


OCR_SERVICE_SLUG = "paddleocr-vl-1.5"
DEFAULT_TARGET_TOOL_NAME = "MCD630"
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "workflow_select_tool"
LOG_NAME = "workflow_select_tool"
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
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_TOOL_NAME_NOT_VISIBLE = "tool_name_not_visible"
EXIT_TOOL_ROW_NOT_FOUND = "tool_row_not_found"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"
EXIT_INVALID_TOOL_NAME = "invalid_tool_name"
EXIT_INVALID_MAIN_WINDOW = "invalid_main_window"

OCR_MAX_TOKENS = 4096
LIST_OCR_MIN_WIDTH = int(os.getenv("SELECT_TOOL_LIST_OCR_MIN_WIDTH", "960"))
LIST_OCR_MIN_HEIGHT = int(os.getenv("SELECT_TOOL_LIST_OCR_MIN_HEIGHT", "900"))
LIST_OCR_MAX_WIDTH = int(os.getenv("SELECT_TOOL_LIST_OCR_MAX_WIDTH", "1400"))
LIST_OCR_MAX_HEIGHT = int(os.getenv("SELECT_TOOL_LIST_OCR_MAX_HEIGHT", "1800"))


def _env_float(name: str, default: float) -> float:
    """float 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


LIST_REGION_LEFT_RATIO = _env_float("SELECT_TOOL_LIST_LEFT_RATIO", 0.00)
LIST_REGION_TOP_RATIO = _env_float("SELECT_TOOL_LIST_TOP_RATIO", 0.10)
LIST_REGION_RIGHT_RATIO = _env_float("SELECT_TOOL_LIST_RIGHT_RATIO", 0.42)
LIST_REGION_BOTTOM_RATIO = _env_float("SELECT_TOOL_LIST_BOTTOM_RATIO", 0.98)
LIST_OCR_MAX_UPSCALE = _env_float("SELECT_TOOL_LIST_OCR_MAX_UPSCALE", 3.0)


def load_target_tool_name(default: str = "") -> str:
    """환경변수에서 목표 Tool 이름을 읽는다."""
    for env_name in (
        "ACTION_TARGET_TOOL_NAME",
        "ACTION_SELECT_TOOL_NAME",
        "SELECT_TOOL_TARGET_ID",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return default.strip()


def _is_valid_main_window_title(window_title: str) -> bool:
    """List 탭 체크가 수행될 메인 RCS 창 제목인지 확인한다."""
    normalized_title = (window_title or "").strip()
    if not normalized_title.startswith(RCS_MAIN_WINDOW_TITLE_PREFIX):
        return False

    lowered = normalized_title.lower()
    return "server" in lowered and "user" in lowered


def _tool_row_target(tool_name: str) -> TargetConfig:
    """지정 Tool row 를 찾기 위한 타겟 설정을 반환한다."""
    return TargetConfig(
        key="tool_row",
        description=(
            f"the tool row in the left-side RCS tool list whose visible tool name text is exactly "
            f"'{tool_name}'. A small colored status square is immediately to the left of the text. "
            f"Return a safe point on that same row where a user would double-click to open the tool."
        ),
        left_pad_ratio=0.7,
        right_pad_ratio=1.8,
        vertical_pad_ratio=1.0,
        min_crop_width=360,
        min_crop_height=120,
    )


def _normalize_tool_text(text: str) -> str:
    """OCR 텍스트 비교를 위해 영숫자만 남기고 대문자로 정규화한다."""
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _normalize_lines(raw_text: str, max_items: int = 300) -> list[str]:
    """OCR raw text 를 한 줄 리스트로 정규화한다."""
    lines: list[str] = []
    seen: set[str] = set()
    for line in (raw_text or "").replace("\r", "\n").split("\n"):
        normalized = " ".join(line.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(normalized)
        if len(lines) >= max_items:
            break
    return lines


def _build_relative_crop_box(
    width: int,
    height: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> dict[str, int]:
    """이미지 크기와 비율로 crop box 를 만든다."""
    left = int(round(width * min(max(left_ratio, 0.0), 1.0)))
    top = int(round(height * min(max(top_ratio, 0.0), 1.0)))
    right = int(round(width * min(max(right_ratio, 0.0), 1.0)))
    bottom = int(round(height * min(max(bottom_ratio, 0.0), 1.0)))

    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _resize_tool_list_image(image):
    """Tool list crop 을 OCR/VLM 입력용으로 확대한다."""
    width, height = image.size
    min_scale = max(
        1.0,
        LIST_OCR_MIN_WIDTH / max(1, width),
        LIST_OCR_MIN_HEIGHT / max(1, height),
    )
    max_scale = min(
        LIST_OCR_MAX_UPSCALE,
        LIST_OCR_MAX_WIDTH / max(1, width),
        LIST_OCR_MAX_HEIGHT / max(1, height),
    )
    scale = max(1.0, min(min_scale, max_scale))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if resized_width == width and resized_height == height:
        return image, {
            "resized": False,
            "scale": 1.0,
            "width": width,
            "height": height,
        }

    resized = image.resize((resized_width, resized_height))
    return resized, {
        "resized": True,
        "scale": scale,
        "width": resized_width,
        "height": resized_height,
    }


def _map_point_from_working_image(point: dict, base_width: int, base_height: int, working_width: int, working_height: int) -> dict[str, int]:
    """리사이즈된 작업 이미지 좌표를 원본 list crop 좌표로 복원한다."""
    mapped_x = int(round(point["x"] * max(base_width - 1, 0) / max(working_width - 1, 1)))
    mapped_y = int(round(point["y"] * max(base_height - 1, 0) / max(working_height - 1, 1)))
    return {
        "x": max(0, min(mapped_x, base_width - 1)),
        "y": max(0, min(mapped_y, base_height - 1)),
    }


def _save_tool_click_overlay(
    image,
    list_crop_box: dict,
    click_point_on_full_image: dict,
    *,
    debug_image_dir,
    timestamp_tag: str,
    filename: str,
) -> str:
    """full main-window screenshot 위에 list crop 과 최종 click point 를 저장한다."""
    img_w, img_h = image.size
    overlay_path = debug_image_path(
        debug_image_dir,
        filename,
        timestamp_tag=timestamp_tag,
    )
    save_marked_bboxes(
        image,
        {
            "tool_list_region": {"bbox": list_crop_box},
            "tool_click_point": {
                "bbox": point_to_tiny_bbox(click_point_on_full_image, img_w, img_h),
                "center": click_point_on_full_image,
            },
        },
        {
            "tool_list_region": "white",
            "tool_click_point": "deepskyblue",
        },
        overlay_path,
    )
    return str(overlay_path)


def _build_list_crop_attempts(main_image) -> list[dict]:
    """Tool list 인식용 crop 시도 목록을 구성한다."""
    full_w, full_h = main_image.size
    focused_right_ratio = min(
        LIST_REGION_RIGHT_RATIO,
        max(LIST_REGION_LEFT_RATIO + 0.20, LIST_REGION_RIGHT_RATIO - 0.08),
    )
    attempt_specs = [
        {
            "name": "focused_left",
            "left_ratio": LIST_REGION_LEFT_RATIO,
            "top_ratio": max(0.0, LIST_REGION_TOP_RATIO - 0.02),
            "right_ratio": focused_right_ratio,
            "bottom_ratio": min(1.0, LIST_REGION_BOTTOM_RATIO + 0.01),
        },
        {
            "name": "default",
            "left_ratio": LIST_REGION_LEFT_RATIO,
            "top_ratio": LIST_REGION_TOP_RATIO,
            "right_ratio": LIST_REGION_RIGHT_RATIO,
            "bottom_ratio": LIST_REGION_BOTTOM_RATIO,
        },
    ]

    attempts: list[dict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for spec in attempt_specs:
        crop_box = _build_relative_crop_box(
            full_w,
            full_h,
            spec["left_ratio"],
            spec["top_ratio"],
            spec["right_ratio"],
            spec["bottom_ratio"],
        )
        crop_key = (
            crop_box["left"],
            crop_box["top"],
            crop_box["right"],
            crop_box["bottom"],
        )
        if crop_key in seen_boxes:
            continue
        seen_boxes.add(crop_key)

        base_image = crop_image(main_image, crop_box)
        working_image, resize_meta = _resize_tool_list_image(base_image)
        attempts.append(
            {
                "name": spec["name"],
                "crop_box": crop_box,
                "base_image": base_image,
                "working_image": working_image,
                "base_size": {
                    "width": base_image.size[0],
                    "height": base_image.size[1],
                },
                "working_size": {
                    "width": working_image.size[0],
                    "height": working_image.size[1],
                },
                "resize_meta": resize_meta,
            }
        )
    return attempts


def _capture_main_window(main_window, window_title: str, backend: str):
    """메인 창을 활성화하고 한 번 캡처한다."""
    if not _is_valid_main_window_title(window_title):
        print(f"[ERROR] 메인 RCS 창 제목이 예상 형식이 아닙니다: title={window_title!r}")
        return None

    if not activate_window(
        main_window,
        debug_label=f"workflow_select_tool activate backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 활성화 실패: title={window_title!r}")
        return None

    if not foreground_window(
        main_window,
        debug_label=f"workflow_select_tool screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 foreground 실패: title={window_title!r}")
        return None

    try:
        return capture_window(main_window)
    except Exception as exc:
        print(f"[ERROR] 메인 창 캡처 실패: {exc}")
        return None


def _run_list_ocr(
    list_image,
    tool_name: str,
    timestamp_tag: str,
    window_title: str,
    backend: str,
    *,
    debug_image_dir,
    log_name: str,
    artifact_label: str = "tool_list",
) -> dict:
    """좌측 Tool List crop 을 OCR 로 읽고 대상 Tool 이름 존재 여부를 확인한다."""
    client = Workflow1VLMClient(
        service_slug=OCR_SERVICE_SLUG,
        timeout_sec=120.0,
        log_name=log_name,
    )
    system_message, user_text = build_ocr_assist_prompt(
        list_image.size[0],
        list_image.size[1],
        context_label="tool_list",
        focus_words=[tool_name],
    )

    list_capture_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_crop.jpg",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    list_webp_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_input.webp",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    raw_response_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_ocr_response.txt",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_ocr_result.json",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )

    save_debug_jpeg(list_image, list_capture_path)
    save_debug_webp(list_image, list_webp_path, quality=90)

    response = client.chat_with_image_path(
        image_path=list_webp_path,
        system_message=system_message,
        user_text=user_text,
        image_mime="image/webp",
        temperature=0.0,
        max_tokens=OCR_MAX_TOKENS,
    )

    raw_text = response.text.strip()
    normalized_lines = _normalize_lines(raw_text, max_items=300)
    normalized_target = _normalize_tool_text(tool_name)
    matched_lines = [
        line for line in normalized_lines
        if normalized_target and normalized_target in _normalize_tool_text(line)
    ]
    target_visible = bool(matched_lines)

    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": response.service_slug,
            "model_name": response.model_name,
            "api_url": response.api_url,
            "endpoint": client.endpoint,
            "window_title": window_title,
            "backend": backend,
            "target_tool_name": tool_name,
            "prompt_text": user_text,
            "raw_text": raw_text,
            "normalized_lines": normalized_lines,
            "matched_lines": matched_lines,
            "target_visible": target_visible,
            "token_usage": response.token_usage,
        },
    )
    return {
        "raw_text": raw_text,
        "normalized_lines": normalized_lines,
        "matched_lines": matched_lines,
        "target_visible": target_visible,
    }


def _run_tool_list_ocr_attempts(
    main_image,
    tool_name: str,
    timestamp_tag: str,
    window_title: str,
    backend: str,
    *,
    debug_image_dir,
    log_name: str,
    component_name: str,
) -> tuple[list[dict], list[dict]]:
    """여러 list crop 시도에서 OCR 을 수행한다."""
    attempts = _build_list_crop_attempts(main_image)
    ocr_attempts: list[dict] = []
    errors: list[dict] = []

    for attempt in attempts:
        try:
            ocr_result = _run_list_ocr(
                attempt["working_image"],
                tool_name,
                timestamp_tag,
                window_title,
                backend,
                debug_image_dir=debug_image_dir,
                log_name=log_name,
                artifact_label=f"tool_list_{attempt['name']}",
            )
        except Exception as exc:
            log_work2_event(
                component=component_name,
                message="ocr_request_failed",
                level="error",
                log_name=log_name,
                target_tool_name=tool_name,
                attempt_name=attempt["name"],
                error=exc,
            )
            errors.append(
                {
                    "attempt_name": attempt["name"],
                    "crop_box": attempt["crop_box"],
                    "resize_meta": attempt["resize_meta"],
                    "error": str(exc),
                }
            )
            continue

        ocr_attempts.append(
            {
                **attempt,
                "ocr_result": ocr_result,
            }
        )
    return ocr_attempts, errors


def _locate_tool_on_attempts(
    main_window,
    window_title: str,
    backend: str,
    normalized_tool_name: str,
    attempts: list[dict],
    *,
    debug_image_dir,
    log_name: str,
    component_name: str,
) -> tuple[dict | None, list[dict]]:
    """OCR 결과가 있는 crop 시도들에서 tool row grounding 을 순차 시도한다."""
    detection_attempts: list[dict] = []
    visible_first = [attempt for attempt in attempts if attempt["ocr_result"]["target_visible"]]
    fallback_attempts = [attempt for attempt in attempts if not attempt["ocr_result"]["target_visible"]]

    for attempt in [*visible_first, *fallback_attempts]:
        tool_result = analyze_window_target(
            main_window,
            window_title,
            backend,
            _tool_row_target(normalized_tool_name),
            debug_image_dir=debug_image_dir,
            log_name=log_name,
            component_name=component_name,
            artifact_prefix=f"workflow_select_tool_{normalized_tool_name.lower()}_{attempt['name']}",
            result_mode="ui_venus_then_mai_ui_tool_list",
            image=attempt["working_image"],
        )

        mapped_point = None
        if tool_result.point is not None:
            mapped_point = _map_point_from_working_image(
                tool_result.point,
                attempt["base_size"]["width"],
                attempt["base_size"]["height"],
                attempt["working_size"]["width"],
                attempt["working_size"]["height"],
            )

        detection_attempt = {
            "attempt_name": attempt["name"],
            "crop_box": attempt["crop_box"],
            "resize_meta": attempt["resize_meta"],
            "ocr_target_visible": attempt["ocr_result"]["target_visible"],
            "ocr_matched_lines": attempt["ocr_result"]["matched_lines"],
            "tool_result_exit_code": tool_result.exit_code,
            "tool_point_on_working_image": tool_result.point,
            "tool_point_on_list_crop": mapped_point,
            "tool_detection_artifacts": tool_result.artifacts,
        }
        detection_attempts.append(detection_attempt)

        if tool_result.exit_code == DETECT_SUCCESS and mapped_point is not None:
            return {
                "attempt": attempt,
                "tool_result": tool_result,
                "mapped_point": mapped_point,
            }, detection_attempts

    return None, detection_attempts


def select_tool_from_main_window(
    main_window,
    window_title: str,
    backend: str,
    tool_name: str,
    *,
    action_enabled: bool = True,
    image=None,
    pre_click_settle_sec: float = 0.2,
    post_double_click_settle_sec: float = 0.5,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> ToolSelectionResult:
    """현재 List 탭에서 지정 Tool 이름을 찾아 더블클릭한다."""
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    normalized_tool_name = tool_name.strip()
    if not normalized_tool_name:
        return ToolSelectionResult(
            exit_code=EXIT_INVALID_TOOL_NAME,
            target_tool_name=tool_name,
        )
    if not _is_valid_main_window_title(window_title):
        return ToolSelectionResult(
            exit_code=EXIT_INVALID_MAIN_WINDOW,
            target_tool_name=normalized_tool_name,
        )

    started_at = time.time()
    timestamp_tag = make_timestamp_tag(started_at)
    main_image = image or _capture_main_window(main_window, window_title, backend)
    if main_image is None:
        return ToolSelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
        )

    full_capture_path = debug_image_path(
        resolved_debug_dir,
        "main_window_capture.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(main_image, full_capture_path)

    ocr_attempts, ocr_errors = _run_tool_list_ocr_attempts(
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
        return ToolSelectionResult(
            exit_code=EXIT_OCR_REQUEST_ERROR,
            target_tool_name=normalized_tool_name,
            list_crop_box=ocr_errors[0]["crop_box"] if ocr_errors else None,
        )

    located_attempt, detection_attempts = _locate_tool_on_attempts(
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
        return ToolSelectionResult(
            exit_code=EXIT_TOOL_ROW_NOT_FOUND if best_visible_attempt is not None else EXIT_TOOL_NAME_NOT_VISIBLE,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=best_visible_attempt is not None,
            list_crop_box=list_crop_box,
            selected_attempt=selected_attempt["name"] if selected_attempt is not None else None,
        )

    tool_result = located_attempt["tool_result"]
    list_crop_point = located_attempt["mapped_point"]
    full_image_point = {
        "x": list_crop_box["left"] + list_crop_point["x"],
        "y": list_crop_box["top"] + list_crop_point["y"],
    }
    click_overlay_path = _save_tool_click_overlay(
        main_image,
        list_crop_box,
        full_image_point,
        debug_image_dir=resolved_debug_dir,
        timestamp_tag=timestamp_tag,
        filename=f"workflow_select_tool_{normalized_tool_name.lower()}_click_overlay.jpg",
    )
    screen_point = image_point_to_screen(main_window, full_image_point)
    if screen_point is None:
        return ToolSelectionResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=selected_attempt["ocr_result"]["target_visible"],
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=list_crop_point,
            tool_point_on_full_image=full_image_point,
            selected_attempt=selected_attempt["name"],
            click_overlay_path=click_overlay_path,
        )

    if not foreground_window(
        main_window,
        debug_label=f"pre_double_click_{normalized_tool_name}",
    ):
        return ToolSelectionResult(
            exit_code=EXIT_WINDOW_ACTIVATE_FAILED,
            target_tool_name=normalized_tool_name,
            matched_lines=matched_lines,
            ocr_target_visible=selected_attempt["ocr_result"]["target_visible"],
            list_crop_box=list_crop_box,
            tool_point_on_list_crop=list_crop_point,
            tool_point_on_full_image=full_image_point,
            tool_point_on_screen=screen_point,
            selected_attempt=selected_attempt["name"],
            click_overlay_path=click_overlay_path,
        )

    time.sleep(max(0.0, pre_click_settle_sec))
    double_clicked = click_at_screen(
        screen_point,
        normalized_tool_name,
        click_count=2,
        action_enabled=action_enabled,
    )
    time.sleep(max(0.0, post_double_click_settle_sec))

    summary_path = debug_image_path(
        resolved_debug_dir,
        "workflow_select_tool_summary.json",
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
            "tool_point_on_list_crop": list_crop_point,
            "tool_point_on_full_image": full_image_point,
            "tool_point_on_screen": screen_point,
            "click_overlay_path": click_overlay_path,
            "double_clicked": double_clicked,
            "action_enabled": action_enabled,
        },
    )

    return ToolSelectionResult(
        exit_code=DETECT_SUCCESS if double_clicked else EXIT_TOOL_ROW_NOT_FOUND,
        target_tool_name=normalized_tool_name,
        matched_lines=matched_lines,
        ocr_target_visible=selected_attempt["ocr_result"]["target_visible"],
        list_crop_box=list_crop_box,
        tool_point_on_list_crop=list_crop_point,
        tool_point_on_full_image=full_image_point,
        tool_point_on_screen=screen_point,
        double_clicked=double_clicked,
        selected_attempt=selected_attempt["name"],
        click_overlay_path=click_overlay_path,
    )


def verify_tool_visible_in_list(
    main_window,
    window_title: str,
    backend: str,
    tool_name: str,
    *,
    image=None,
    debug_image_dir=None,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> ToolListVisibilityResult:
    """현재 메인 창 List 영역에서 대상 Tool 이름이 보이는지 검증한다."""
    resolved_debug_dir = debug_image_dir or DEBUG_ARTIFACT_DIR
    normalized_tool_name = tool_name.strip()
    if not normalized_tool_name:
        return ToolListVisibilityResult(
            exit_code=EXIT_INVALID_TOOL_NAME,
            target_tool_name=tool_name,
        )
    if not _is_valid_main_window_title(window_title):
        return ToolListVisibilityResult(
            exit_code=EXIT_INVALID_MAIN_WINDOW,
            target_tool_name=normalized_tool_name,
        )

    started_at = time.time()
    timestamp_tag = make_timestamp_tag(started_at)
    main_image = image or _capture_main_window(main_window, window_title, backend)
    if main_image is None:
        return ToolListVisibilityResult(
            exit_code=EXIT_CAPTURE_FAILED,
            target_tool_name=normalized_tool_name,
        )

    ocr_attempts, ocr_errors = _run_tool_list_ocr_attempts(
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
        return ToolListVisibilityResult(
            exit_code=EXIT_OCR_REQUEST_ERROR,
            target_tool_name=normalized_tool_name,
            list_crop_box=ocr_errors[0]["crop_box"] if ocr_errors else None,
        )

    visible_attempts = [attempt for attempt in ocr_attempts if attempt["ocr_result"]["target_visible"]]
    if visible_attempts:
        selected_attempt = visible_attempts[0]
        return ToolListVisibilityResult(
            exit_code=DETECT_SUCCESS,
            target_tool_name=normalized_tool_name,
            matched_lines=selected_attempt["ocr_result"]["matched_lines"],
            target_visible=True,
            list_crop_box=selected_attempt["crop_box"],
            selected_attempt=selected_attempt["name"],
            visibility_source="ocr",
        )

    located_attempt, _ = _locate_tool_on_attempts(
        main_window,
        window_title,
        backend,
        normalized_tool_name,
        ocr_attempts,
        debug_image_dir=resolved_debug_dir,
        log_name=log_name,
        component_name=component_name,
    )
    if located_attempt is not None:
        selected_attempt = located_attempt["attempt"]
        return ToolListVisibilityResult(
            exit_code=DETECT_SUCCESS,
            target_tool_name=normalized_tool_name,
            matched_lines=[],
            target_visible=True,
            list_crop_box=selected_attempt["crop_box"],
            selected_attempt=selected_attempt["name"],
            visibility_source="vlm_grounding",
        )

    fallback_attempt = ocr_attempts[0] if ocr_attempts else None
    return ToolListVisibilityResult(
        exit_code=EXIT_TOOL_NAME_NOT_VISIBLE,
        target_tool_name=normalized_tool_name,
        matched_lines=[],
        target_visible=False,
        list_crop_box=fallback_attempt["crop_box"] if fallback_attempt is not None else None,
        selected_attempt=fallback_attempt["name"] if fallback_attempt is not None else None,
        visibility_source="ocr",
    )


def main() -> str:
    """현재 List 탭에서 지정 Tool 이름을 찾아 더블클릭한다."""
    started_at = time.time()
    target_tool_name = load_target_tool_name(DEFAULT_TARGET_TOOL_NAME)

    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        target_tool_name=target_tool_name,
        ocr_service=OCR_SERVICE_SLUG,
        action_enabled=DEFAULT_ACTION_ENABLED,
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
            title_prefix=RCS_MAIN_WINDOW_TITLE_PREFIX,
        )
        return EXIT_MAIN_WINDOW_NOT_FOUND

    result = select_tool_from_main_window(
        main_window,
        window_title,
        backend,
        target_tool_name,
        action_enabled=DEFAULT_ACTION_ENABLED,
    )
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(started_at)}, "
        f"target_tool_name={target_tool_name!r}, result={result.exit_code}"
    )
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != DETECT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
