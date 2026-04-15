"""캡처된 CH4 프레임들에서 2단계 VLM으로 마우스 커서를 정밀 탐지한다.

1. ui-venus 로 전체 프레임에서 coarse bbox 탐지
2. coarse bbox 주변을 crop + zoom 한 뒤 mai-ui 로 precise cursor tip/bbox 탐지

기본 실행:
  uv run python poc/workflow_1/locate_cursor_in_captured_frames.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
from poc.workflow_1.flask_vlm import MAI_UI_MODEL_NAME, UI_VENUS_MODEL_NAME
from poc.workflow_1.util import env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
    parse_coords,
)
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "locate_cursor_in_captured_frames"
DEFAULT_CAPTURE_ROOT = DEBUG_IMAGE_DIR / "capture_window_frames_ch4"
DEFAULT_OUTPUT_ROOT = DEBUG_IMAGE_DIR / LOG_NAME
DEFAULT_MAX_FRAMES = env_int("CH4_CURSOR_MAX_FRAMES", 0)
DEFAULT_CROP_PADDING_PX = env_int("CH4_CURSOR_CROP_PADDING_PX", 48)
DEFAULT_ZOOM_SCALE = max(1, env_int("CH4_CURSOR_ZOOM_SCALE", 3))
DEFAULT_COARSE_SERVICE = os.getenv("CH4_CURSOR_COARSE_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_REFINE_SERVICE = os.getenv("CH4_CURSOR_REFINE_SERVICE", "mai-ui").strip() or "mai-ui"
DEFAULT_COARSE_MODEL = os.getenv("CH4_CURSOR_COARSE_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME
DEFAULT_REFINE_MODEL = os.getenv("CH4_CURSOR_REFINE_MODEL_NAME", MAI_UI_MODEL_NAME).strip() or MAI_UI_MODEL_NAME


def _resolve_capture_dir() -> Path | None:
    """분석할 캡처 프레임 디렉터리를 결정한다."""
    raw_path = os.getenv("CH4_CURSOR_FRAMES_DIR", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] CH4_CURSOR_FRAMES_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    if not DEFAULT_CAPTURE_ROOT.exists():
        print(f"[ERROR] 캡처 프레임 디렉터리가 없습니다: {DEFAULT_CAPTURE_ROOT}")
        return None

    candidates = sorted(
        (path for path in DEFAULT_CAPTURE_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[ERROR] 캡처 프레임 세트가 없습니다: {DEFAULT_CAPTURE_ROOT}")
        return None

    latest = candidates[0].resolve()
    print(f"[INFO] 최신 캡처 세트 선택: {latest}")
    return latest


def _load_capture_summary(capture_dir: Path) -> dict | None:
    """캡처 단계 summary.json 을 읽는다."""
    summary_path = capture_dir / "summary.json"
    if not summary_path.is_file():
        print(f"[ERROR] summary.json 이 없습니다: {summary_path}")
        return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] summary.json 파싱 실패: {exc}")
        return None


def _build_output_dir(capture_dir: Path) -> Path:
    """이번 커서 탐지 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    out_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{capture_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _coarse_system_prompt() -> str:
    """coarse cursor detection 시스템 프롬프트."""
    return (
        "You locate a Windows mouse cursor in a full software screenshot. "
        "Return strict JSON only. "
        "Find only the mouse cursor, not UI elements. "
        "If the cursor is not visible, say so."
    )


def _coarse_user_prompt() -> str:
    """coarse cursor detection 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "cursor_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "cursor_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string"\n'
        "}\n"
        "The bbox must tightly enclose the entire visible mouse cursor. "
        "If no cursor is visible, set cursor_visible=false and cursor_bbox=null."
    )


def _refine_system_prompt() -> str:
    """refine cursor detection 시스템 프롬프트."""
    return (
        "You refine a mouse cursor location inside a zoomed crop. "
        "Return strict JSON only. "
        "Find the exact cursor tip and a tight bbox around the cursor. "
        "If uncertain, do not invent."
    )


def _refine_user_prompt() -> str:
    """refine cursor detection 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "cursor_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "cursor_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "cursor_tip": {"x": 0, "y": 0},\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string"\n'
        "}\n"
        "cursor_tip must be the actual click hotspot or arrow tip, not the center. "
        "If no cursor is visible, set cursor_visible=false, cursor_bbox=null, cursor_tip=null."
    )


def _expand_bbox(bbox: dict, img_w: int, img_h: int, padding_px: int) -> dict:
    """bbox 주변을 padding 만큼 확장한다."""
    return {
        "left": max(0, int(bbox["left"]) - padding_px),
        "top": max(0, int(bbox["top"]) - padding_px),
        "right": min(img_w, int(bbox["right"]) + padding_px),
        "bottom": min(img_h, int(bbox["bottom"]) + padding_px),
    }


def _crop_image(image: Image.Image, bbox: dict) -> Image.Image:
    """PIL image 에서 bbox 영역을 crop 한다."""
    return image.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))


def _translate_local_point_to_full(local_point: dict, crop_box: dict) -> dict[str, int]:
    """crop 내부 point 를 full image point 로 변환한다."""
    return {
        "x": crop_box["left"] + int(local_point["x"]),
        "y": crop_box["top"] + int(local_point["y"]),
    }


def _translate_local_bbox_to_full(local_bbox: dict, crop_box: dict) -> dict:
    """crop 내부 bbox 를 full image bbox 로 변환한다."""
    return {
        "left": crop_box["left"] + int(local_bbox["left"]),
        "top": crop_box["top"] + int(local_bbox["top"]),
        "right": crop_box["left"] + int(local_bbox["right"]),
        "bottom": crop_box["top"] + int(local_bbox["bottom"]),
    }


def _save_zoom_crop(
    *,
    image_path: Path,
    crop_box: dict,
    output_path: Path,
    zoom_scale: int,
) -> tuple[Path, tuple[int, int]]:
    """crop 후 확대된 이미지를 저장한다."""
    with Image.open(image_path) as image:
        crop_image = _crop_image(image.convert("RGB"), crop_box)
        crop_size = crop_image.size
        if zoom_scale > 1:
            zoomed = crop_image.resize(
                (crop_size[0] * zoom_scale, crop_size[1] * zoom_scale),
                resample=Image.Resampling.LANCZOS,
            )
        else:
            zoomed = crop_image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        zoomed.save(output_path, format="JPEG", quality=95)
        return output_path, crop_size


def _save_crop_overlay(
    *,
    crop_image_path: Path,
    local_bbox: dict | None,
    local_tip: dict | None,
    output_path: Path,
) -> str:
    """zoom crop 위에 refined bbox/tip overlay 를 저장한다."""
    with Image.open(crop_image_path) as image:
        elements = {}
        if local_bbox is not None:
            elements["refined_cursor"] = {
                "bbox": local_bbox,
                "center": local_tip or bbox_center(local_bbox),
            }
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={"refined_cursor": "red"},
            out_path=output_path,
        )
        return str(output_path)


def _save_full_overlay(
    *,
    frame_path: Path,
    coarse_bbox: dict | None,
    refined_bbox: dict | None,
    tip_point: dict | None,
    output_path: Path,
) -> str:
    """전체 프레임 위에 coarse/refined 결과를 함께 마킹한다."""
    with Image.open(frame_path) as image:
        elements = {}
        if coarse_bbox is not None:
            elements["coarse_cursor"] = {
                "bbox": coarse_bbox,
                "center": bbox_center(coarse_bbox),
            }
        if refined_bbox is not None:
            elements["refined_cursor"] = {
                "bbox": refined_bbox,
                "center": tip_point or bbox_center(refined_bbox),
            }
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={
                "coarse_cursor": "cyan",
                "refined_cursor": "red",
            },
            out_path=output_path,
        )
        return str(output_path)


def _run_coarse_detection(
    *,
    frame_path: Path,
    coarse_client: Workflow1VLMClient,
) -> tuple[dict | None, dict | None]:
    """전체 프레임에서 coarse bbox 를 탐지한다."""
    response = coarse_client.chat_with_image_path(
        image_path=frame_path,
        system_message=_coarse_system_prompt(),
        user_text=_coarse_user_prompt(),
        image_mime="image/jpeg",
        temperature=0.0,
    )

    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return parsed, None

    with Image.open(frame_path) as image:
        coarse_bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
        if coarse_bbox_1000 is None:
            return parsed, None
        coarse_bbox = bbox_1000_to_pixels(coarse_bbox_1000, image.size[0], image.size[1])
        return parsed, coarse_bbox


def _run_refine_detection(
    *,
    zoom_crop_path: Path,
    crop_size: tuple[int, int],
    refine_client: Workflow1VLMClient,
) -> tuple[dict | None, dict | None, dict | None]:
    """zoom crop 에서 refined bbox 와 tip 을 탐지한다."""
    response = refine_client.chat_with_image_path(
        image_path=zoom_crop_path,
        system_message=_refine_system_prompt(),
        user_text=_refine_user_prompt(),
        image_mime="image/jpeg",
        temperature=0.0,
    )

    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return parsed, None, None

    refine_bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
    local_bbox = None
    if refine_bbox_1000 is not None:
        local_bbox = bbox_1000_to_pixels(refine_bbox_1000, crop_size[0], crop_size[1])

    parsed = parse_coords(parsed, ["cursor_tip"], crop_size[0], crop_size[1])
    local_tip = parsed.get("cursor_tip")
    if not isinstance(local_tip, dict) or "x" not in local_tip or "y" not in local_tip:
        local_tip = None

    return parsed, local_bbox, local_tip


def _build_timeline_text(results: list[dict]) -> str:
    """사람이 읽기 쉬운 탐지 타임라인을 만든다."""
    lines = []
    for item in results:
        final = item.get("final_cursor_tip") or {}
        lines.append(
            f"[{float(item.get('timestamp_sec') or 0.0):>7.3f}s] "
            f"frame={int(item.get('frame_index') or 0):04d} "
            f"visible={item.get('cursor_visible')} "
            f"tip=({final.get('x', '-')}, {final.get('y', '-')}) "
            f"coarse_conf={item.get('coarse_confidence', '')} "
            f"refine_conf={item.get('refine_confidence', '')}"
        )
    return "\n".join(lines) + "\n"


def locate_cursors() -> str:
    """캡처 프레임들에서 2단계 VLM으로 커서를 탐지한다."""
    started_at = time.time()
    capture_dir = _resolve_capture_dir()
    if capture_dir is None:
        return "capture_dir_not_found"

    summary = _load_capture_summary(capture_dir)
    if summary is None:
        return "summary_not_found"

    frame_items = summary.get("frame_items") or []
    if not frame_items:
        print(f"[ERROR] 분석할 frame_items 가 없습니다: {capture_dir}")
        return "frame_items_empty"

    output_dir = _build_output_dir(capture_dir)
    coarse_dir = output_dir / "coarse"
    crops_dir = output_dir / "zoom_crops"
    crop_overlays_dir = output_dir / "crop_overlays"
    full_overlays_dir = output_dir / "full_overlays"
    results_dir = output_dir / "results"
    for directory in (coarse_dir, crops_dir, crop_overlays_dir, full_overlays_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    coarse_client = Workflow1VLMClient(
        service_slug=DEFAULT_COARSE_SERVICE,
        model_name=DEFAULT_COARSE_MODEL,
        log_name=LOG_NAME,
    )
    refine_client = Workflow1VLMClient(
        service_slug=DEFAULT_REFINE_SERVICE,
        model_name=DEFAULT_REFINE_MODEL,
        log_name=LOG_NAME,
    )

    max_frames = DEFAULT_MAX_FRAMES if DEFAULT_MAX_FRAMES > 0 else None
    print(
        f"[INFO] 커서 탐지 시작: capture_dir={capture_dir}, "
        f"coarse={DEFAULT_COARSE_SERVICE}/{DEFAULT_COARSE_MODEL}, "
        f"refine={DEFAULT_REFINE_SERVICE}/{DEFAULT_REFINE_MODEL}, "
        f"zoom_scale={DEFAULT_ZOOM_SCALE}, "
        f"max_frames={max_frames or 'all'}"
    )

    results: list[dict] = []
    processed = 0
    for frame_item in frame_items:
        if max_frames and processed >= max_frames:
            break

        frame_path = Path(str(frame_item.get("frame_path") or "")).expanduser()
        if not frame_path.is_file():
            print(f"[WARNING] 프레임 파일 누락: {frame_path}")
            continue

        frame_index = int(frame_item.get("index") or 0)
        timestamp_sec = float(frame_item.get("timestamp_sec") or 0.0)
        print(f"[INFO] 커서 탐지: frame={frame_index}, ts={timestamp_sec:.3f}s")

        coarse_payload = None
        coarse_bbox = None
        refine_payload = None
        local_refined_bbox = None
        local_tip = None
        full_refined_bbox = None
        full_tip = None
        zoom_crop_path = ""
        crop_overlay_path = ""
        full_overlay_path = ""

        try:
            coarse_payload, coarse_bbox = _run_coarse_detection(
                frame_path=frame_path,
                coarse_client=coarse_client,
            )
        except Exception as exc:
            print(f"[ERROR] coarse cursor detection 실패: frame={frame_index}, error={exc}")

        if coarse_bbox is not None:
            with Image.open(frame_path) as frame_image:
                img_w, img_h = frame_image.size
            crop_box = _expand_bbox(coarse_bbox, img_w, img_h, DEFAULT_CROP_PADDING_PX)
            zoom_crop_file = crops_dir / f"frame_{frame_index:04d}_zoom.jpg"
            try:
                saved_crop_path, crop_size = _save_zoom_crop(
                    image_path=frame_path,
                    crop_box=crop_box,
                    output_path=zoom_crop_file,
                    zoom_scale=DEFAULT_ZOOM_SCALE,
                )
                zoom_crop_path = str(saved_crop_path)
                refine_payload, local_refined_bbox, local_tip = _run_refine_detection(
                    zoom_crop_path=saved_crop_path,
                    crop_size=crop_size,
                    refine_client=refine_client,
                )

                if local_refined_bbox is not None:
                    full_refined_bbox = _translate_local_bbox_to_full(local_refined_bbox, crop_box)
                if local_tip is not None:
                    full_tip = _translate_local_point_to_full(local_tip, crop_box)

                crop_overlay_path = _save_crop_overlay(
                    crop_image_path=saved_crop_path,
                    local_bbox=local_refined_bbox,
                    local_tip=local_tip,
                    output_path=crop_overlays_dir / f"frame_{frame_index:04d}_crop_overlay.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] refine cursor detection 실패: frame={frame_index}, error={exc}")

        if coarse_bbox is not None or full_refined_bbox is not None:
            full_overlay_path = _save_full_overlay(
                frame_path=frame_path,
                coarse_bbox=coarse_bbox,
                refined_bbox=full_refined_bbox,
                tip_point=full_tip,
                output_path=full_overlays_dir / f"frame_{frame_index:04d}_full_overlay.jpg",
            )

        result = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "frame_path": str(frame_path),
            "cursor_visible": bool(
                (refine_payload or coarse_payload or {}).get("cursor_visible", False)
            ),
            "coarse_payload": coarse_payload or {},
            "coarse_bbox": coarse_bbox or {},
            "coarse_confidence": (coarse_payload or {}).get("confidence"),
            "zoom_crop_path": zoom_crop_path,
            "crop_overlay_path": crop_overlay_path,
            "refine_payload": refine_payload or {},
            "refine_bbox_local": local_refined_bbox or {},
            "refine_confidence": (refine_payload or {}).get("confidence"),
            "final_cursor_bbox": full_refined_bbox or {},
            "final_cursor_tip": full_tip or {},
            "full_overlay_path": full_overlay_path,
        }
        result_path = results_dir / f"frame_{frame_index:04d}.json"
        save_debug_json(result_path, result)
        results.append(result)
        processed += 1

    summary_payload = {
        "capture_dir": str(capture_dir),
        "coarse_service": DEFAULT_COARSE_SERVICE,
        "coarse_model_name": DEFAULT_COARSE_MODEL,
        "refine_service": DEFAULT_REFINE_SERVICE,
        "refine_model_name": DEFAULT_REFINE_MODEL,
        "crop_padding_px": DEFAULT_CROP_PADDING_PX,
        "zoom_scale": DEFAULT_ZOOM_SCALE,
        "processed_frames": processed,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "results": results,
    }
    save_debug_json(output_dir / "summary.json", summary_payload)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(results))

    print(
        f"[INFO] 커서 탐지 완료: processed={processed}, "
        f"elapsed={format_elapsed_ms(started_at)}, output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if locate_cursors() == "success" else 1)
