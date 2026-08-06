"""재사용 가능한 coarse(bbox) + fine(point) 2단계 타겟 로케이터.

기본 조합은 ui-venus(coarse) -> mai-ui(fine) 이지만, `VLM_LOCATOR_COMBO` env 로
런타임에 바꿀 수 있다(형식은 bench_tool_locator 의 BENCH_COMBOS 와 동일한 "coarse>fine").
env 를 지우면 즉시 production 기본값으로 되돌아온다 - 롤백 지점이 여기 한 곳뿐이다.

    VLM_LOCATOR_COMBO="mai-ui>mai-ui"   # 양 단계 모두 mai-ui
    VLM_LOCATOR_COMBO=""(미설정)        # ui-venus>mai-ui (production 기본)
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from poc.workflow_3.debug_artifacts import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
    save_debug_text,
    save_marked_bboxes,
)
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.vlm.prompts.prompt_login_rcs_mai_ui import build_mai_ui_zoom_prompt
from poc.workflow_3.vlm.prompts.prompt_login_rcs_ui_venus import (
    build_ui_venus_single_element_bbox_prompt,
)
from poc.workflow_3.util import (
    activate_window,
    bbox_1000_to_pixels,
    bbox_center,
    capture_window,
    crop_image,
    encode_image_webp,
    ensure_min_span,
    foreground_window,
    format_elapsed_ms,
    make_timestamp_tag,
    normalize_bbox_1000,
    parse_coords,
    point_to_tiny_bbox,
)
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient


DEFAULT_COARSE_SERVICE = "ui-venus"   # coarse 단계 기본 서비스(route_slug, 모델명 아님).
DEFAULT_REFINE_SERVICE = "mai-ui"     # fine 단계 기본 서비스(route_slug).
LOCATOR_COMBO_ENV = "VLM_LOCATOR_COMBO"

_announced_combo = None   # 콘솔에 조합을 1회만 알리기 위한 가드.


def parse_locator_combo(raw, *, warn=True):
    """"coarse>fine" 문자열을 (coarse_slug, refine_slug) 로 판다.

    형식이 깨졌으면 경고만 하고 production 기본값을 돌려준다(로케이터가 못 뜨면
    로그인부터 막히므로 죽이지 않는다). warn=False 는 조회 전용 호출(시작 로그 등)
    에서 같은 경고를 두 번 찍지 않게 하려는 것이다.
    """
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_COARSE_SERVICE, DEFAULT_REFINE_SERVICE

    if ">" not in raw:
        if warn:
            print(
                f"[WARNING] {LOCATOR_COMBO_ENV}={raw!r} 에 '>' 가 없습니다 "
                f"(형식: coarse>fine). 기본 조합으로 진행합니다."
            )
        return DEFAULT_COARSE_SERVICE, DEFAULT_REFINE_SERVICE

    left, _, right = raw.partition(">")
    left, right = left.strip(), right.strip()
    if not (left and right):
        if warn:
            print(
                f"[WARNING] {LOCATOR_COMBO_ENV}={raw!r} 파싱 실패 "
                f"(양쪽 모두 필요). 기본 조합으로 진행합니다."
            )
        return DEFAULT_COARSE_SERVICE, DEFAULT_REFINE_SERVICE

    return left, right


def describe_locator_combo(raw=""):
    """시작 로그 한 줄 - 실제로 적용될 조합을 보여준다.

    raw 를 그대로 되돌리지 않고 파싱을 거치는 이유: 형식이 깨진 값을 그대로 찍으면
    "설정한 대로 돌고 있다" 고 오독하게 된다. 실제 적용값을 보여줘야 진단이 된다.
    """
    coarse, refine = parse_locator_combo(raw, warn=False)
    suffix = "" if (raw or "").strip() else " (기본)"
    return f"{coarse}>{refine}{suffix}"


def resolve_locator_services():
    """(coarse_slug, refine_slug) 를 env 에서 해석한다 - 호출 시점 read.

    import 시점이 아니라 호출 시점에 읽는 이유: rcs 단독 스크립트는 seed_env() 를
    부르지 않고 shell env 로만 제어하는데, import 시점에 고정하면 나중에 주입한
    값이 무시되어 "env 를 줬는데 안 먹는" 함정이 생긴다.
    """
    global _announced_combo

    raw = os.getenv(LOCATOR_COMBO_ENV, "").strip()
    coarse, refine = parse_locator_combo(raw)

    combo = f"{coarse}>{refine}"
    if combo != _announced_combo:
        default_mark = "" if raw else " (기본값)"
        print(f"[INFO] VLM locator 조합: coarse={coarse}, fine={refine}{default_mark}")
        _announced_combo = combo
    return coarse, refine


@dataclass
class TargetResult:
    """2단계 파이프라인 실행 결과."""

    exit_code: str
    target_key: str
    point: dict | None = None
    bbox: dict | None = None   # coarse(ui-venus) 단계 영역 bbox(full px) — 영역 crop 재사용용.
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class TargetConfig:
    """2단계 파이프라인의 타겟 요소 설정."""

    key: str
    description: str
    left_pad_ratio: float = 1.25
    right_pad_ratio: float = 0.45
    vertical_pad_ratio: float = 1.6
    min_crop_width: int = 320
    min_crop_height: int = 120
    # 세로 여백의 하한(px). 촘촘한 리스트 행처럼 bbox 가 얇을 때는 이 하한이 비율보다
    # 커서 crop 이 위아래 행까지 삼킨다. 그런 타겟은 하한을 낮춰 잡아야 한다.
    vertical_pad_min_px: int = 28


EXIT_SUCCESS = "success"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_CAPTURE_FAILED = "capture_failed"

TARGET_MIN_RESIZED_WIDTH = 960
TARGET_MIN_RESIZED_HEIGHT = 320
MAX_RESIZED_WIDTH = 1400
MAX_RESIZED_HEIGHT = 900
MAX_UPSCALE = 4.0

OVERLAY_COLORS = {
    "coarse": "gold",
    "crop": "white",
    "refined": "deepskyblue",
}


def _print_vlm_understanding(service_slug: str, response_text: str, token_usage: dict | None) -> None:
    """VLM 응답 텍스트를 콘솔에서 바로 읽기 좋게 출력한다."""
    print(f"[INFO] [{service_slug}] understanding:")
    stripped = (response_text or "").strip()
    if stripped:
        print(stripped)
    else:
        print("<empty>")
    print(f"[INFO] [{service_slug}] tokens={token_usage or {}}")


def _build_crop_box(
    coarse_bbox: dict,
    img_w: int,
    img_h: int,
    target: TargetConfig,
) -> dict:
    """coarse bbox 주변에 여유 영역을 둔 crop box 를 만든다."""
    bbox_w = max(1, coarse_bbox["right"] - coarse_bbox["left"])
    bbox_h = max(1, coarse_bbox["bottom"] - coarse_bbox["top"])
    left_pad = max(96, int(round(bbox_w * target.left_pad_ratio)))
    right_pad = max(48, int(round(bbox_w * target.right_pad_ratio)))
    vertical_pad = max(target.vertical_pad_min_px, int(round(bbox_h * target.vertical_pad_ratio)))

    crop_left = max(0, coarse_bbox["left"] - left_pad)
    crop_top = max(0, coarse_bbox["top"] - vertical_pad)
    crop_right = min(img_w, coarse_bbox["right"] + right_pad)
    crop_bottom = min(img_h, coarse_bbox["bottom"] + vertical_pad)

    crop_left, crop_right = ensure_min_span(
        crop_left, crop_right, img_w, target.min_crop_width,
    )
    crop_top, crop_bottom = ensure_min_span(
        crop_top, crop_bottom, img_h, target.min_crop_height,
    )
    return {
        "left": crop_left,
        "top": crop_top,
        "right": crop_right,
        "bottom": crop_bottom,
    }


def _resize_crop_for_mai(image: Image.Image) -> tuple[Image.Image, dict]:
    """작은 crop 을 MAI-UI 입력용으로 확대한다."""
    width, height = image.size
    min_scale = max(
        1.0,
        TARGET_MIN_RESIZED_WIDTH / max(1, width),
        TARGET_MIN_RESIZED_HEIGHT / max(1, height),
    )
    max_scale = min(
        MAX_UPSCALE,
        MAX_RESIZED_WIDTH / max(1, width),
        MAX_RESIZED_HEIGHT / max(1, height),
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

    resized = image.resize((resized_width, resized_height), Image.LANCZOS)
    return resized, {
        "resized": True,
        "scale": scale,
        "width": resized_width,
        "height": resized_height,
    }


def _map_resized_point_to_full_image(
    resized_point: dict,
    crop_box: dict,
    crop_w: int,
    crop_h: int,
    resized_w: int,
    resized_h: int,
) -> dict[str, int]:
    """리사이즈된 crop 좌표를 원본 full image 좌표로 복원한다."""
    crop_x = int(
        round(resized_point["x"] * max(crop_w - 1, 0) / max(resized_w - 1, 1))
    )
    crop_y = int(
        round(resized_point["y"] * max(crop_h - 1, 0) / max(resized_h - 1, 1))
    )
    crop_x = max(0, min(crop_x, crop_w - 1))
    crop_y = max(0, min(crop_y, crop_h - 1))
    return {
        "x": crop_box["left"] + crop_x,
        "y": crop_box["top"] + crop_y,
    }


def _scale_bbox_to_resized_crop(bbox: dict, crop_box: dict, resized_w: int, resized_h: int) -> dict:
    """full-image bbox 를 resized crop 좌표계로 변환한다."""
    crop_w = max(1, crop_box["right"] - crop_box["left"])
    crop_h = max(1, crop_box["bottom"] - crop_box["top"])
    relative = {
        "left": max(0, bbox["left"] - crop_box["left"]),
        "top": max(0, bbox["top"] - crop_box["top"]),
        "right": max(1, bbox["right"] - crop_box["left"]),
        "bottom": max(1, bbox["bottom"] - crop_box["top"]),
    }
    return {
        "left": int(round(relative["left"] * resized_w / crop_w)),
        "top": int(round(relative["top"] * resized_h / crop_h)),
        "right": int(round(relative["right"] * resized_w / crop_w)),
        "bottom": int(round(relative["bottom"] * resized_h / crop_h)),
    }


def _save_pipeline_inputs(
    image: Image.Image,
    zoom_image: Image.Image,
    pipeline_model_name: str,
    debug_stamp: str,
    *,
    debug_image_dir: Path,
    log_name: str,
    artifact_prefix: str,
) -> dict[str, Path]:
    """원본/zoom JPEG artifact 를 저장한다."""
    capture_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_capture.jpg",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    zoom_capture_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_zoom_crop.jpg",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    save_debug_jpeg(image, capture_path)
    save_debug_jpeg(zoom_image, zoom_capture_path)
    return {
        "capture": capture_path,
        "zoom_capture": zoom_capture_path,
    }


def _save_pipeline_failure_capture(
    image: Image.Image,
    pipeline_model_name: str,
    debug_stamp: str,
    *,
    debug_image_dir: Path,
    artifact_prefix: str,
) -> Path:
    """coarse 단계 실패 시에도 원본 캡처를 저장한다."""
    capture_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_capture.jpg",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    save_debug_jpeg(image, capture_path)
    return capture_path


def _save_full_pipeline_overlay(
    image: Image.Image,
    target: TargetConfig,
    coarse_result: dict,
    crop_box: dict | None,
    refined_full_point: dict | None,
    pipeline_model_name: str,
    debug_stamp: str,
    *,
    debug_image_dir: Path,
    artifact_prefix: str,
    filename_suffix: str = "ui_venus_mai_overlay",
) -> Path:
    """full image 위에 UI-Venus bbox 와 MAI-UI point 를 마킹한다."""
    full_w, full_h = image.size
    coarse_key = f"{target.key}_ui_venus"
    crop_key = f"{target.key}_crop_region"
    refined_key = f"{target.key}_mai_ui"
    overlay_colors = {
        coarse_key: OVERLAY_COLORS["coarse"],
        crop_key: OVERLAY_COLORS["crop"],
        refined_key: OVERLAY_COLORS["refined"],
    }

    overlay_items = {
        coarse_key: {
            "bbox": coarse_result["bbox_pixels"],
            "center": coarse_result["center"],
        },
    }
    if crop_box is not None:
        overlay_items[crop_key] = {"bbox": crop_box}
    if refined_full_point is not None:
        overlay_items[refined_key] = {
            "bbox": point_to_tiny_bbox(refined_full_point, full_w, full_h),
            "center": refined_full_point,
        }

    overlay_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_{filename_suffix}.jpg",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    save_marked_bboxes(image, overlay_items, overlay_colors, overlay_path)
    return overlay_path


def _save_zoom_pipeline_overlay(
    zoom_image: Image.Image,
    target: TargetConfig,
    coarse_bbox_pixels: dict,
    crop_box: dict,
    refine_point: dict | None,
    pipeline_model_name: str,
    debug_stamp: str,
    *,
    debug_image_dir: Path,
    artifact_prefix: str,
    filename_suffix: str = "ui_venus_mai_zoom_overlay",
) -> Path:
    """MAI-UI zoom crop 위에 coarse bbox 와 refined point 를 마킹한다."""
    zoom_w, zoom_h = zoom_image.size
    coarse_key = f"{target.key}_ui_venus"
    refined_key = f"{target.key}_mai_ui"
    overlay_colors = {
        coarse_key: OVERLAY_COLORS["coarse"],
        refined_key: OVERLAY_COLORS["refined"],
    }

    overlay_items = {
        coarse_key: {
            "bbox": _scale_bbox_to_resized_crop(
                coarse_bbox_pixels, crop_box, zoom_w, zoom_h,
            ),
        },
    }
    if refine_point is not None:
        overlay_items[refined_key] = {
            "bbox": point_to_tiny_bbox(refine_point, zoom_w, zoom_h),
            "center": refine_point,
        }

    overlay_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_{filename_suffix}.jpg",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    save_marked_bboxes(zoom_image, overlay_items, overlay_colors, overlay_path)
    return overlay_path


def _run_ui_venus_coarse_bbox(
    client: Workflow1VLMClient,
    image_b64: str,
    img_w: int,
    img_h: int,
    target: TargetConfig,
) -> dict | None:
    """coarse 서비스로 full image bbox 를 찾는다(기본 ui-venus, env 로 교체 가능).

    로그 라벨은 하드코딩하지 않고 client.service_slug 를 쓴다 - 조합을 바꿨을 때
    콘솔이 실제로 어떤 모델이 답했는지 말해줘야 오피스에서 진단이 된다.
    """
    slug = client.service_slug
    system_message, user_text = build_ui_venus_single_element_bbox_prompt(
        target.description,
    )
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=system_message,
        user_text=user_text,
        temperature=0.0,
    )
    _print_vlm_understanding(slug, response.text, response.token_usage)

    base_result = {
        "response_text": response.text,
        "token_usage": response.token_usage or {},
    }

    try:
        parsed = extract_json(response.text)
    except Exception as exc:
        print(f"[WARNING] [{slug}] coarse JSON 파싱 실패: {exc}")
        return {**base_result, "bbox_1000": None, "bbox_pixels": None, "center": None}

    bbox_1000 = normalize_bbox_1000(parsed.get("bbox"))
    if bbox_1000 is None:
        print(f"[INFO] [{slug}] coarse bbox 미검출")
        return {**base_result, "bbox_1000": None, "bbox_pixels": None, "center": None}

    bbox_pixels = bbox_1000_to_pixels(bbox_1000, img_w, img_h)
    center = bbox_center(bbox_pixels)
    print(
        f"[INFO] [{slug}] bbox1000={bbox_1000} -> px={bbox_pixels}, "
        f"center=({center['x']}, {center['y']})"
    )
    return {
        **base_result,
        "bbox_1000": bbox_1000,
        "bbox_pixels": bbox_pixels,
        "center": center,
    }


def _run_mai_ui_refinement(
    client: Workflow1VLMClient,
    zoom_b64: str,
    zoom_w: int,
    zoom_h: int,
    target: TargetConfig,
) -> dict | None:
    """fine 서비스로 zoom crop 안의 refined click point 를 찾는다(기본 mai-ui)."""
    slug = client.service_slug
    system_message, user_text = build_mai_ui_zoom_prompt(
        target.key, target.description,
    )
    response = client.chat_with_image_b64(
        image_b64=zoom_b64,
        system_message=system_message,
        user_text=user_text,
        temperature=0.0,
    )
    _print_vlm_understanding(slug, response.text, response.token_usage)

    try:
        parsed = extract_json(response.text)
    except Exception as exc:
        print(f"[WARNING] [{slug}] refine JSON 파싱 실패: {exc}")
        return {
            "response_text": response.text,
            "token_usage": response.token_usage or {},
            "point": None,
        }

    parsed = parse_coords(parsed, [target.key], zoom_w, zoom_h)
    point = parsed.get(target.key)
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        print(f"[INFO] [{slug}] refined point 미검출")
        return {
            "response_text": response.text,
            "token_usage": response.token_usage or {},
            "point": None,
        }

    print(f"[INFO] [{slug}] refined point on zoom=({point['x']}, {point['y']})")
    return {
        "response_text": response.text,
        "token_usage": response.token_usage or {},
        "point": {"x": int(point["x"]), "y": int(point["y"])},
    }


def analyze_window_target(
    window,
    window_title: str,
    backend: str,
    target: TargetConfig,
    *,
    debug_image_dir: Path,
    log_name: str,
    component_name: str,
    artifact_prefix: str,
    coarse_service_slug: str | None = None,
    refine_service_slug: str | None = None,
    result_mode: str = "ui_venus_then_mai_ui_single_target",
    image: Image.Image | None = None,
    timeout_sec: float | None = None,
) -> TargetResult:
    """임의의 윈도우에서 지정된 타겟을 2단계로 찾는다.

    image 가 주어지면 창 활성화/캡처를 건너뛰고 해당 이미지를 사용한다.
    coarse/refine slug 를 명시하지 않으면 VLM_LOCATOR_COMBO env(기본 ui-venus>mai-ui)
    를 따른다 - 벤치만 조합을 직접 지정한다.
    """
    if coarse_service_slug is None or refine_service_slug is None:
        env_coarse, env_refine = resolve_locator_services()
        coarse_service_slug = coarse_service_slug or env_coarse
        refine_service_slug = refine_service_slug or env_refine

    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)

    if image is None:
        if not activate_window(
            window,
            debug_label=f"window backend={backend} title={window_title!r}",
        ):
            print(f"[ERROR] 창 활성화 실패: title={window_title!r}")
            return TargetResult(EXIT_WINDOW_ACTIVATE_FAILED, target.key)

        if not foreground_window(
            window,
            debug_label=f"window screenshot backend={backend} title={window_title!r}",
        ):
            print(f"[ERROR] 창 foreground 실패: title={window_title!r}")
            return TargetResult(EXIT_WINDOW_ACTIVATE_FAILED, target.key)

        try:
            image = capture_window(window)
        except Exception as exc:
            print(f"[ERROR] 창 캡처 실패: {exc}")
            log_work2_event(
                component=component_name,
                message="capture_failed",
                level="error",
                log_name=log_name,
                backend=backend,
                window_title=window_title,
                error=exc,
            )
            return TargetResult(EXIT_CAPTURE_FAILED, target.key)

    _client_kw = {"log_name": log_name}
    if timeout_sec is not None:
        _client_kw["timeout_sec"] = timeout_sec
    coarse_client = Workflow1VLMClient(service_slug=coarse_service_slug, **_client_kw)
    refine_client = Workflow1VLMClient(service_slug=refine_service_slug, **_client_kw)
    pipeline_model_name = f"{coarse_client.model_name}__{refine_client.model_name}"

    full_b64, full_w, full_h = encode_image_webp(image)
    coarse_result = _run_ui_venus_coarse_bbox(
        coarse_client, full_b64, full_w, full_h, target,
    )
    if coarse_result is None or coarse_result["bbox_pixels"] is None:
        capture_path = _save_pipeline_failure_capture(
            image=image,
            pipeline_model_name=pipeline_model_name,
            debug_stamp=debug_stamp,
            debug_image_dir=debug_image_dir,
            artifact_prefix=artifact_prefix,
        )
        coarse_response_path = debug_image_path(
            debug_image_dir,
            f"{artifact_prefix}_ui_venus_response.txt",
            model_name=pipeline_model_name,
            timestamp_tag=debug_stamp,
        )
        save_debug_text(
            coarse_response_path,
            "" if coarse_result is None else coarse_result["response_text"],
        )
        result_json_path = debug_image_path(
            debug_image_dir,
            f"{artifact_prefix}_ui_venus_mai_result.json",
            model_name=pipeline_model_name,
            timestamp_tag=debug_stamp,
        )
        artifacts = {
            "capture": str(capture_path),
            "ui_venus_response": str(coarse_response_path),
            "result_json": str(result_json_path),
        }
        save_debug_json(
            result_json_path,
            {
                "mode": result_mode,
                "status": EXIT_VLM_NO_DETECTION,
                "failure_stage": "ui_venus",
                "target_key": target.key,
                "target_description": target.description,
                "image_width": full_w,
                "image_height": full_h,
                "coarse_service": coarse_service_slug,
                "coarse_model": coarse_client.model_name,
                "refine_service": refine_service_slug,
                "refine_model": refine_client.model_name,
                "coarse_bbox_1000": None,
                "coarse_bbox_pixels": None,
                "artifacts": artifacts,
            },
        )
        log_work2_event(
            component=component_name,
            message="coarse_bbox_missing",
            level="warning",
            log_name=log_name,
            backend=backend,
            window_title=window_title,
            target_key=target.key,
            coarse_service=coarse_service_slug,
        )
        return TargetResult(EXIT_VLM_NO_DETECTION, target.key, artifacts=artifacts)

    crop_box = _build_crop_box(coarse_result["bbox_pixels"], full_w, full_h, target)
    cropped = crop_image(image, crop_box)
    crop_w, crop_h = cropped.size
    zoom_image, zoom_meta = _resize_crop_for_mai(cropped)
    zoom_b64, zoom_w, zoom_h = encode_image_webp(zoom_image)

    print(
        f"[INFO] coarse->zoom pipeline 시작: target={target.key}, "
        f"full={full_w}x{full_h}, crop={crop_w}x{crop_h}, zoom={zoom_w}x{zoom_h}, "
        f"coarse_service={coarse_service_slug}, refine_service={refine_service_slug}"
    )

    refine_result = _run_mai_ui_refinement(
        refine_client, zoom_b64, zoom_w, zoom_h, target,
    )

    input_paths = _save_pipeline_inputs(
        image=image,
        zoom_image=zoom_image,
        pipeline_model_name=pipeline_model_name,
        debug_stamp=debug_stamp,
        debug_image_dir=debug_image_dir,
        log_name=log_name,
        artifact_prefix=artifact_prefix,
    )
    coarse_response_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_ui_venus_response.txt",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    refine_response_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_mai_ui_response.txt",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    save_debug_text(coarse_response_path, coarse_result["response_text"])
    if refine_result is not None:
        save_debug_text(refine_response_path, refine_result["response_text"])
    if refine_result is None or refine_result["point"] is None:
        overlay_path = _save_full_pipeline_overlay(
            image=image,
            target=target,
            coarse_result=coarse_result,
            crop_box=crop_box,
            refined_full_point=None,
            pipeline_model_name=pipeline_model_name,
            debug_stamp=debug_stamp,
            debug_image_dir=debug_image_dir,
            artifact_prefix=artifact_prefix,
            filename_suffix="ui_venus_partial_overlay",
        )
        zoom_overlay_path = _save_zoom_pipeline_overlay(
            zoom_image=zoom_image,
            target=target,
            coarse_bbox_pixels=coarse_result["bbox_pixels"],
            crop_box=crop_box,
            refine_point=None,
            pipeline_model_name=pipeline_model_name,
            debug_stamp=debug_stamp,
            debug_image_dir=debug_image_dir,
            artifact_prefix=artifact_prefix,
            filename_suffix="ui_venus_partial_zoom_overlay",
        )
        result_json_path = debug_image_path(
            debug_image_dir,
            f"{artifact_prefix}_ui_venus_mai_result.json",
            model_name=pipeline_model_name,
            timestamp_tag=debug_stamp,
        )
        artifacts = {
            "capture": str(input_paths["capture"]),
            "zoom_capture": str(input_paths["zoom_capture"]),
            "ui_venus_response": str(coarse_response_path),
            "mai_ui_response": str(refine_response_path),
            "partial_overlay": str(overlay_path),
            "partial_zoom_overlay": str(zoom_overlay_path),
            "result_json": str(result_json_path),
        }
        save_debug_json(
            result_json_path,
            {
                "mode": result_mode,
                "status": EXIT_VLM_NO_DETECTION,
                "failure_stage": "mai_ui",
                "target_key": target.key,
                "target_description": target.description,
                "image_width": full_w,
                "image_height": full_h,
                "crop_box_pixels": crop_box,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "zoom_width": zoom_w,
                "zoom_height": zoom_h,
                "zoom_meta": zoom_meta,
                "coarse_service": coarse_service_slug,
                "coarse_model": coarse_client.model_name,
                "coarse_bbox_1000": coarse_result["bbox_1000"],
                "coarse_bbox_pixels": coarse_result["bbox_pixels"],
                "coarse_center_pixels": coarse_result["center"],
                "refine_service": refine_service_slug,
                "refine_model": refine_client.model_name,
                "refined_point_zoom_pixels": None,
                "refined_point_full_pixels": None,
                "artifacts": artifacts,
            },
        )
        log_work2_event(
            component=component_name,
            message="refined_point_missing",
            level="warning",
            log_name=log_name,
            backend=backend,
            window_title=window_title,
            target_key=target.key,
            refine_service=refine_service_slug,
        )
        return TargetResult(EXIT_VLM_NO_DETECTION, target.key, artifacts=artifacts)

    refined_full_point = _map_resized_point_to_full_image(
        refine_result["point"], crop_box, crop_w, crop_h, zoom_w, zoom_h,
    )
    print(
        f"[INFO] refined full-image point=({refined_full_point['x']}, "
        f"{refined_full_point['y']})"
    )

    overlay_path = _save_full_pipeline_overlay(
        image=image,
        target=target,
        coarse_result=coarse_result,
        crop_box=crop_box,
        refined_full_point=refined_full_point,
        pipeline_model_name=pipeline_model_name,
        debug_stamp=debug_stamp,
        debug_image_dir=debug_image_dir,
        artifact_prefix=artifact_prefix,
    )
    zoom_overlay_path = _save_zoom_pipeline_overlay(
        zoom_image=zoom_image,
        target=target,
        coarse_bbox_pixels=coarse_result["bbox_pixels"],
        crop_box=crop_box,
        refine_point=refine_result["point"],
        pipeline_model_name=pipeline_model_name,
        debug_stamp=debug_stamp,
        debug_image_dir=debug_image_dir,
        artifact_prefix=artifact_prefix,
    )

    result_payload = {
        "mode": result_mode,
        "target_key": target.key,
        "target_description": target.description,
        "image_width": full_w,
        "image_height": full_h,
        "crop_box_pixels": crop_box,
        "crop_width": crop_w,
        "crop_height": crop_h,
        "zoom_width": zoom_w,
        "zoom_height": zoom_h,
        "zoom_meta": zoom_meta,
        "coarse_service": coarse_service_slug,
        "coarse_model": coarse_client.model_name,
        "coarse_bbox_1000": coarse_result["bbox_1000"],
        "coarse_bbox_pixels": coarse_result["bbox_pixels"],
        "coarse_center_pixels": coarse_result["center"],
        "refine_service": refine_service_slug,
        "refine_model": refine_client.model_name,
        "refined_point_zoom_pixels": refine_result["point"],
        "refined_point_full_pixels": refined_full_point,
        "artifacts": {
            "capture": str(input_paths["capture"]),
            "zoom_capture": str(input_paths["zoom_capture"]),
            "ui_venus_response": str(coarse_response_path),
            "mai_ui_response": str(refine_response_path),
            "overlay": str(overlay_path),
            "zoom_overlay": str(zoom_overlay_path),
        },
    }
    result_json_path = debug_image_path(
        debug_image_dir,
        f"{artifact_prefix}_ui_venus_mai_result.json",
        model_name=pipeline_model_name,
        timestamp_tag=debug_stamp,
    )
    result_payload["artifacts"]["result_json"] = str(result_json_path)
    save_debug_json(result_json_path, result_payload)
    print(
        f"[INFO] pipeline 완료: target={target.key}, "
        f"point=({refined_full_point['x']}, {refined_full_point['y']}), "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return TargetResult(
        EXIT_SUCCESS,
        target.key,
        point=refined_full_point,
        bbox=coarse_result["bbox_pixels"],
        artifacts=result_payload["artifacts"],
    )


__all__ = [
    "EXIT_CAPTURE_FAILED",
    "EXIT_SUCCESS",
    "EXIT_VLM_NO_DETECTION",
    "EXIT_WINDOW_ACTIVATE_FAILED",
    "TargetConfig",
    "TargetResult",
    "analyze_window_target",
]
