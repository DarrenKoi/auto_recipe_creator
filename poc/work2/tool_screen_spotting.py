"""저장된 tool screen 캡처 이미지로 PaddleOCR Spotting 디버그를 수행한다.

목적:
  1. `poc/work2/capture_images/` 의 PNG/JPG/WebP 스크린샷을 읽는다.
  2. `paddleocr-vl-1.5` 의 `Spotting:` 태스크를 별도 실행한다.
  3. 원본 이미지와 spotting 처리 이미지에 bbox overlay 를 저장한다.
  4. target text 예: `Recipe Monitor` 가 어디서 검출됐는지 확인한다.

사용법:
  1. `poc/work2/capture_images/` 에 스크린샷을 넣는다.
  2. 필요 시 `.env` 에 아래 값을 넣는다.
     - `TOOL_SCREEN_SPOTTING_TARGET_TEXT=Recipe Monitor`
     - `TOOL_SCREEN_SPOTTING_IMAGE_FILTER=RecipeMonitor`
     - `TOOL_SCREEN_SPOTTING_MAX_TOKENS=1024`
  3. `uv run python poc/work2/tool_screen_spotting.py`
"""

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.work2.flask_vlm import get_service_by_slug
from poc.work2.logger import log_work2_event
from poc.work2.util import (
    debug_image_path,
    format_elapsed_ms,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_bboxes,
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text
from poc.work2.util.json_utils import bbox_center
from poc.work2.vlm_client import Work2VLMClient


load_dotenv()

WORK2_DIR = Path(__file__).resolve().parent
CAPTURE_IMAGE_DIR = WORK2_DIR / "capture_images"
DEBUG_IMAGE_DIR = WORK2_DIR / "debug_images" / "tool_screen_spotting"
LOG_NAME = Path(__file__).stem
SERVICE_SLUG = "paddleocr-vl-1.5"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_MAX_TOKENS = 1024
SPOTTING_PROMPT = "Spotting:"
SPOTTING_UPSCALE_THRESHOLD = 1500

TARGET_TEXT = os.getenv("TOOL_SCREEN_SPOTTING_TARGET_TEXT", "Recipe Monitor").strip() or "Recipe Monitor"
IMAGE_FILTER = os.getenv("TOOL_SCREEN_SPOTTING_IMAGE_FILTER", "").strip().lower()


def _sanitize_stem(text: str) -> str:
    """파일명/아티팩트 prefix 에 사용할 안전한 문자열로 정규화한다."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()).strip("._-")
    return normalized or "capture"


def _collect_capture_images() -> list[Path]:
    """분석 대상 이미지 목록을 반환한다."""
    if not CAPTURE_IMAGE_DIR.is_dir():
        return []

    results: list[Path] = []
    for path in sorted(CAPTURE_IMAGE_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if IMAGE_FILTER and IMAGE_FILTER not in path.name.lower():
            continue
        results.append(path)
    return results


def _resolve_max_tokens() -> int:
    """spotting 응답용 max_tokens 를 안전 범위로 맞춘다."""
    raw_value = os.getenv("TOOL_SCREEN_SPOTTING_MAX_TOKENS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_TOKENS

    try:
        resolved = int(raw_value)
    except ValueError:
        print(
            "[WARNING] TOOL_SCREEN_SPOTTING_MAX_TOKENS 값이 잘못되었습니다. "
            f"default={DEFAULT_MAX_TOKENS} 를 사용합니다: {raw_value!r}"
        )
        return DEFAULT_MAX_TOKENS

    if resolved <= 0:
        return DEFAULT_MAX_TOKENS
    return min(resolved, 2048)


def _load_image(image_path: Path) -> Image.Image:
    """이미지 파일을 RGB PIL Image 로 읽는다."""
    with Image.open(image_path) as opened:
        return opened.convert("RGB")


def _prepare_spotting_image(image: Image.Image) -> tuple[Image.Image, dict]:
    """spotting 작업에 유리하도록 작은 이미지를 확대한다."""
    orig_w, orig_h = image.size
    if orig_w < SPOTTING_UPSCALE_THRESHOLD and orig_h < SPOTTING_UPSCALE_THRESHOLD:
        resized_w = orig_w * 2
        resized_h = orig_h * 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        resized = image.resize((resized_w, resized_h), resample_filter)
        return resized, {
            "resized": True,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "process_width": resized_w,
            "process_height": resized_h,
            "scale_x": resized_w / max(1, orig_w),
            "scale_y": resized_h / max(1, orig_h),
        }

    return image.copy(), {
        "resized": False,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "process_width": orig_w,
        "process_height": orig_h,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }


def _save_source_artifacts(
    original_image: Image.Image,
    spotting_image: Image.Image,
    debug_dir: Path,
    artifact_prefix: str,
    timestamp_tag: str,
) -> dict[str, Path]:
    """원본/spotting 입력 이미지를 JPEG/WebP 로 저장한다."""
    original_capture_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_original_capture.jpg",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    original_webp_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_original_input.webp",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    spotting_capture_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_capture.jpg",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    spotting_webp_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_input.webp",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(original_image, original_capture_path, log_name=LOG_NAME)
    save_debug_webp(original_image, original_webp_path, quality=90, log_name=LOG_NAME)
    save_debug_jpeg(spotting_image, spotting_capture_path, log_name=LOG_NAME)
    save_debug_webp(spotting_image, spotting_webp_path, quality=90, log_name=LOG_NAME)
    return {
        "original_capture": original_capture_path,
        "original_webp": original_webp_path,
        "spotting_capture": spotting_capture_path,
        "spotting_webp": spotting_webp_path,
    }


def _parse_json_like(text: str):
    """raw 텍스트를 JSON/파이썬 리터럴로 best-effort 파싱한다."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    candidates = [stripped]
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        candidates.append(stripped[start_obj:end_obj + 1])
    start_arr = stripped.find("[")
    end_arr = stripped.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        candidates.append(stripped[start_arr:end_arr + 1])

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    return None


def _extract_text_label(item: object) -> str:
    """spotting item 에서 텍스트 라벨을 추출한다."""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""

    for key in (
        "text",
        "content",
        "block_content",
        "transcription",
        "label",
        "word",
        "rec_text",
        "caption",
        "value",
    ):
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _is_number_list(values: object, length: int) -> bool:
    """지정 길이의 숫자 리스트인지 확인한다."""
    if not isinstance(values, list) or len(values) != length:
        return False
    return all(isinstance(value, (int, float)) for value in values)


def _polygon_to_bbox(points: list[list[float]] | list[tuple[float, float]]) -> dict | None:
    """polygon 좌표를 bbox 로 변환한다."""
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        if not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float)):
            return None
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None
    left = int(round(min(xs)))
    top = int(round(min(ys)))
    right = int(round(max(xs)))
    bottom = int(round(max(ys)))
    if right <= left or bottom <= top:
        return None
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _coerce_bbox(raw_box: object) -> dict | None:
    """여러 bbox 표현을 공통 {left, top, right, bottom} 으로 변환한다."""
    if raw_box is None:
        return None

    if _is_number_list(raw_box, 4):
        values = [int(round(float(value))) for value in raw_box]
        left, top, right, bottom = values
        if right <= left or bottom <= top:
            return None
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }

    if isinstance(raw_box, list) and raw_box and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in raw_box
    ):
        return _polygon_to_bbox(raw_box)

    if isinstance(raw_box, dict):
        if {"left", "top", "right", "bottom"} <= raw_box.keys():
            left = int(round(float(raw_box["left"])))
            top = int(round(float(raw_box["top"])))
            right = int(round(float(raw_box["right"])))
            bottom = int(round(float(raw_box["bottom"])))
            if right <= left or bottom <= top:
                return None
            return {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }
        if {"x1", "y1", "x2", "y2"} <= raw_box.keys():
            return _coerce_bbox([
                raw_box["x1"],
                raw_box["y1"],
                raw_box["x2"],
                raw_box["y2"],
            ])
        for key in ("bbox", "block_bbox", "box", "polygon", "points"):
            if key in raw_box:
                nested = _coerce_bbox(raw_box.get(key))
                if nested is not None:
                    return nested
    return None


def _collect_spotting_candidates(node: object, results: list[dict], hinted_text: str = "") -> None:
    """중첩된 응답 구조에서 spotting 후보들을 재귀적으로 수집한다."""
    if isinstance(node, list):
        for item in node:
            _collect_spotting_candidates(item, results, hinted_text=hinted_text)
        return

    if not isinstance(node, dict):
        return

    text = _extract_text_label(node) or hinted_text
    bbox = None
    for key in ("bbox", "block_bbox", "box", "polygon", "points"):
        if key in node:
            bbox = _coerce_bbox(node.get(key))
            if bbox is not None:
                break

    if bbox is None and text and len(node) == 1:
        only_value = next(iter(node.values()))
        bbox = _coerce_bbox(only_value)

    if text and bbox is not None:
        results.append(
            {
                "text": text,
                "bbox": bbox,
            }
        )

    wrapper_keys = {
        "layoutParsingResults",
        "prunedResult",
        "spotting_res",
        "parsing_res_list",
        "result",
        "data",
        "items",
        "blocks",
        "detections",
        "texts",
    }
    for key, value in node.items():
        if key in wrapper_keys:
            _collect_spotting_candidates(value, results, hinted_text=hinted_text)
            continue

        if isinstance(value, (dict, list)):
            next_hint = ""
            if isinstance(value, list) and key not in wrapper_keys:
                next_hint = str(key).strip()
            _collect_spotting_candidates(value, results, hinted_text=next_hint)


def _dedupe_spotting_items(items: list[dict]) -> list[dict]:
    """동일 텍스트/좌표 후보를 중복 제거한다."""
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for item in items:
        bbox = item["bbox"]
        signature = (
            item["text"],
            bbox["left"],
            bbox["top"],
            bbox["right"],
            bbox["bottom"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _parse_spotting_items(raw_text: str) -> list[dict]:
    """spotting raw 텍스트에서 text+bbox 후보를 추출한다."""
    parsed = _parse_json_like(raw_text)
    if parsed is None:
        return []

    results: list[dict] = []
    _collect_spotting_candidates(parsed, results)
    return _dedupe_spotting_items(results)


def _bbox_to_original_coords(bbox: dict, scale_x: float, scale_y: float, orig_w: int, orig_h: int) -> dict:
    """spotting 이미지 bbox 를 원본 이미지 좌표계로 되돌린다."""
    left = max(0, min(orig_w - 1, int(round(bbox["left"] / max(scale_x, 1e-6)))))
    top = max(0, min(orig_h - 1, int(round(bbox["top"] / max(scale_y, 1e-6)))))
    right = max(left + 1, min(orig_w, int(round(bbox["right"] / max(scale_x, 1e-6)))))
    bottom = max(top + 1, min(orig_h, int(round(bbox["bottom"] / max(scale_y, 1e-6)))))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _text_matches_target(text: str) -> bool:
    """검출 텍스트가 target text 와 일치/포함되는지 확인한다."""
    normalized_text = " ".join((text or "").lower().split())
    normalized_target = " ".join((TARGET_TEXT or "").lower().split())
    if not normalized_text or not normalized_target:
        return False
    return normalized_target in normalized_text or normalized_text in normalized_target


def _save_spotting_overlay(
    image: Image.Image,
    items: list[dict],
    out_path: Path,
    *,
    matched_only: bool = False,
) -> None:
    """spotting 검출 결과를 bbox overlay 로 저장한다."""
    elements: dict[str, dict] = {}
    colors: dict[str, str] = {}

    display_index = 0
    for item in items:
        matched = bool(item.get("matched"))
        if matched_only and not matched:
            continue
        display_index += 1
        key_prefix = "target" if matched else "spot"
        key = f"{key_prefix}_{display_index:02d}_{_sanitize_stem(item.get('text', 'text'))[:28]}"
        elements[key] = {
            "bbox": item["bbox"],
            "center": bbox_center(item["bbox"]),
        }
        colors[key] = "gold" if matched else "lime"

    if not elements:
        return
    save_marked_bboxes(image, elements, colors, out_path)


def _run_spotting(
    image_webp_path: Path,
    *,
    max_tokens: int,
) -> tuple[str, dict]:
    """PaddleOCR-VL `Spotting:` 호출을 수행한다."""
    client = Work2VLMClient(
        service_slug=SERVICE_SLUG,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        log_name=LOG_NAME,
    )
    response = client.chat_with_image_path(
        image_path=image_webp_path,
        system_message="",
        user_text=SPOTTING_PROMPT,
        image_mime="image/webp",
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.text.strip(), {
        "service_slug": response.service_slug,
        "model_name": response.model_name,
        "api_url": response.api_url,
        "endpoint": client.endpoint,
        "token_usage": response.token_usage,
    }


def _collect_artifact_paths(debug_dir: Path, artifact_prefix: str) -> list[str]:
    """artifact prefix 로 생성된 디버그 파일 경로를 반환한다."""
    if not debug_dir.is_dir():
        return []

    results: list[str] = []
    for path in sorted(debug_dir.rglob(f"*{artifact_prefix}*")):
        if path.is_file():
            results.append(str(path))
    return results


def _analyze_single_image(image_path: Path) -> dict:
    """단일 이미지에 대해 spotting 디버그를 수행한다."""
    started_at = time.time()
    timestamp_tag = time.strftime("%y%m%d_%H%M%S", time.localtime(started_at))
    artifact_prefix = _sanitize_stem(image_path.stem)
    debug_dir = DEBUG_IMAGE_DIR / artifact_prefix

    print(f"[INFO] spotting 분석 시작: image={image_path}")
    original_image = _load_image(image_path)
    spotting_image, spotting_meta = _prepare_spotting_image(original_image)
    source_artifacts = _save_source_artifacts(
        original_image=original_image,
        spotting_image=spotting_image,
        debug_dir=debug_dir,
        artifact_prefix=artifact_prefix,
        timestamp_tag=timestamp_tag,
    )

    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        raise ValueError(f"서비스 설정을 찾지 못했습니다: {SERVICE_SLUG}")

    raw_response_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    parsed_result_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    process_overlay_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_overlay_process.jpg",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    original_overlay_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_overlay_original.jpg",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    target_overlay_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_spotting_target_overlay.jpg",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )

    raw_text, response_meta = _run_spotting(
        image_webp_path=source_artifacts["spotting_webp"],
        max_tokens=_resolve_max_tokens(),
    )
    save_debug_text(raw_response_path, raw_text)

    parsed_items = _parse_spotting_items(raw_text)
    process_items: list[dict] = []
    original_items: list[dict] = []
    for item in parsed_items:
        process_bbox = item["bbox"]
        matched = _text_matches_target(item["text"])
        process_items.append(
            {
                "text": item["text"],
                "bbox": process_bbox,
                "matched": matched,
            }
        )
        original_bbox = _bbox_to_original_coords(
            process_bbox,
            spotting_meta["scale_x"],
            spotting_meta["scale_y"],
            spotting_meta["orig_width"],
            spotting_meta["orig_height"],
        )
        original_items.append(
            {
                "text": item["text"],
                "bbox": original_bbox,
                "matched": matched,
            }
        )

    matched_items = [item for item in original_items if item["matched"]]
    if process_items:
        _save_spotting_overlay(spotting_image, process_items, process_overlay_path)
        _save_spotting_overlay(original_image, original_items, original_overlay_path)
    if matched_items:
        _save_spotting_overlay(original_image, matched_items, target_overlay_path, matched_only=True)

    for index, item in enumerate(matched_items[:10], start=1):
        bbox = item["bbox"]
        print(
            f"[INFO] target spotting {index:02d}: text={item['text']!r}, "
            f"bbox=({bbox['left']},{bbox['top']},{bbox['right']},{bbox['bottom']})"
        )

    result_payload = {
        "image_path": str(image_path),
        "status": "success",
        "target_text": TARGET_TEXT,
        "prompt": SPOTTING_PROMPT,
        "spotting_meta": spotting_meta,
        "response_meta": response_meta,
        "raw_text": raw_text,
        "parsed_item_count": len(original_items),
        "matched_item_count": len(matched_items),
        "parsed_items_process_image": process_items,
        "parsed_items_original_image": original_items,
        "matched_items_original_image": matched_items,
        "artifacts": {
            "original_capture": str(source_artifacts["original_capture"]),
            "original_webp": str(source_artifacts["original_webp"]),
            "spotting_capture": str(source_artifacts["spotting_capture"]),
            "spotting_webp": str(source_artifacts["spotting_webp"]),
            "response_text": str(raw_response_path),
            "process_overlay": str(process_overlay_path) if process_items else "",
            "original_overlay": str(original_overlay_path) if process_items else "",
            "target_overlay": str(target_overlay_path) if matched_items else "",
        },
        "elapsed_ms": round((time.time() - started_at) * 1000, 1),
    }
    save_debug_json(parsed_result_path, result_payload)
    result_payload["artifacts"]["result_json"] = str(parsed_result_path)
    result_payload["all_debug_artifacts"] = _collect_artifact_paths(debug_dir, artifact_prefix)
    print(
        f"[INFO] spotting 분석 완료: image={image_path.name}, "
        f"parsed={len(original_items)}, matched={len(matched_items)}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return result_payload


def main() -> int:
    """capture_images 폴더의 모든 이미지에 대해 spotting 디버그를 수행한다."""
    started_at = time.time()
    if not CAPTURE_IMAGE_DIR.is_dir():
        print(f"[ERROR] capture_images 폴더를 찾지 못했습니다: {CAPTURE_IMAGE_DIR}")
        log_work2_event(
            component="tool_screen_spotting",
            message="capture_dir_missing",
            level="error",
            log_name=LOG_NAME,
            capture_dir=CAPTURE_IMAGE_DIR,
        )
        return 1

    image_paths = _collect_capture_images()
    if not image_paths:
        print(
            "[WARNING] 분석할 이미지가 없습니다. "
            f"지원 확장자={sorted(SUPPORTED_IMAGE_EXTENSIONS)}, dir={CAPTURE_IMAGE_DIR}"
        )
        log_work2_event(
            component="tool_screen_spotting",
            message="no_images_found",
            level="warning",
            log_name=LOG_NAME,
            capture_dir=CAPTURE_IMAGE_DIR,
            image_filter=IMAGE_FILTER,
        )
        return 1

    print(
        f"[INFO] tool screen spotting 시작: image_count={len(image_paths)}, "
        f"target_text={TARGET_TEXT!r}, service={SERVICE_SLUG}"
    )
    results: list[dict] = []
    error_count = 0
    for image_path in image_paths:
        try:
            results.append(_analyze_single_image(image_path))
        except Exception as exc:
            error_count += 1
            error_text = str(exc)
            print(f"[ERROR] spotting 분석 실패: image={image_path}, error={error_text}")
            results.append(
                {
                    "image_path": str(image_path),
                    "status": "error",
                    "error": error_text,
                }
            )

    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_screen_spotting_summary.json",
        model_name="summary",
        timestamp_tag=time.strftime("%y%m%d_%H%M%S", time.localtime(started_at)),
    )
    save_debug_json(
        summary_path,
        {
            "capture_dir": str(CAPTURE_IMAGE_DIR),
            "target_text": TARGET_TEXT,
            "service_slug": SERVICE_SLUG,
            "image_filter": IMAGE_FILTER,
            "image_count": len(image_paths),
            "error_count": error_count,
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            "results": results,
        },
    )
    print(
        f"[INFO] spotting 전체 완료: images={len(image_paths)}, errors={error_count}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    print(f"[INFO] summary 저장: {summary_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
