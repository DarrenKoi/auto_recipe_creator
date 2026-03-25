"""저장된 tool screen 캡처 이미지로 OCR/VLM 사전 판독을 수행한다.

목적:
  1. `poc/work2/capture_images/` 에 저장된 스크린샷을 직접 읽는다.
  2. OCR 모델로 화면 내 텍스트 판독 결과를 저장한다.
  3. `ui-venus` + `mai-ui` 2단계 파이프라인으로 여러 UI target 을
     one-by-one 으로 순차 탐지한다.

사용법:
  1. `poc/work2/capture_images/` 에 JPG/PNG/WebP 스크린샷을 넣는다.
  2. 필요 시 `.env` 에 아래 값을 넣는다.
     - `TOOL_SCREEN_READ_TARGET_TITLE=Recipe Monitor`
     - `TOOL_SCREEN_READ_TARGETS=Recipe Monitor,Recipe Button under Port-C,Queue button,File Manager Button`
     - `TOOL_SCREEN_READ_OCR_SERVICES=paddleocr-vl-1.5`
     - `TOOL_SCREEN_READ_IMAGE_FILTER=RecipeMonitor`
  3. `uv run python poc/work2/tool_screen_read.py`
"""

import base64
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from poc.work2.flask_vlm import (
    get_service_by_slug,
    resolve_service_api_key,
    resolve_service_proxy_url,
)
from poc.work2.logger import log_work2_event
from poc.work2.prompts import build_ocr_assist_prompt
from poc.work2.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.work2.util import (
    debug_image_path,
    format_elapsed_ms,
    make_timestamp_tag,
    normalize_lines,
    save_debug_jpeg,
    save_debug_webp,
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text
from poc.work2.vlm_client import Work2VLMClient


load_dotenv()

WORK2_DIR = Path(__file__).resolve().parent
CAPTURE_IMAGE_DIR = WORK2_DIR / "capture_images"
DEBUG_IMAGE_DIR = WORK2_DIR / "debug_images" / "tool_screen_read"
LOG_NAME = Path(__file__).stem

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_OCR_SERVICES = ("paddleocr-vl-1.5",)
DEFAULT_OCR_MAX_TOKENS = 512
DEFAULT_TIMEOUT_SEC = 120.0

TARGET_TITLE = os.getenv("TOOL_SCREEN_READ_TARGET_TITLE", "Recipe Monitor").strip() or "Recipe Monitor"
IMAGE_FILTER = os.getenv("TOOL_SCREEN_READ_IMAGE_FILTER", "").strip().lower()
DEFAULT_ITERATION_TARGET_LABELS = (
    "Recipe Button under Port-C",
    "Queue button",
    "File Manager Button",
)
GOT_FORMAT_OUTPUT = os.getenv("TOOL_SCREEN_READ_GOT_FORMAT_OUTPUT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "y",
}
GOT_CROP_TO_PATCHES = os.getenv("TOOL_SCREEN_READ_GOT_CROP_TO_PATCHES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "y",
}

EXIT_SUCCESS = "success"
EXIT_CAPTURE_DIR_MISSING = "capture_dir_missing"
EXIT_NO_IMAGES = "no_images"
EXIT_PARTIAL = "partial"


@dataclass
class ScreenReadTarget:
    """tool screen 에서 순차 탐지할 단일 target 설정."""

    key: str
    label: str
    description: str
    focus_words: tuple[str, ...]
    left_pad_ratio: float = 1.25
    right_pad_ratio: float = 0.45
    vertical_pad_ratio: float = 1.6
    min_crop_width: int = 320
    min_crop_height: int = 120

    def to_target_config(self) -> TargetConfig:
        """ui_venus_mai_locator.TargetConfig 로 변환한다."""
        return TargetConfig(
            key=self.key,
            description=self.description,
            left_pad_ratio=self.left_pad_ratio,
            right_pad_ratio=self.right_pad_ratio,
            vertical_pad_ratio=self.vertical_pad_ratio,
            min_crop_width=self.min_crop_width,
            min_crop_height=self.min_crop_height,
        )


def _parse_ocr_services() -> list[str]:
    """OCR 서비스 목록을 환경변수에서 읽는다."""
    raw_value = os.getenv("TOOL_SCREEN_READ_OCR_SERVICES", "").strip()
    if not raw_value:
        return list(DEFAULT_OCR_SERVICES)

    resolved: list[str] = []
    for part in raw_value.split(","):
        service_slug = part.strip()
        if service_slug and service_slug not in resolved:
            resolved.append(service_slug)
    return resolved or list(DEFAULT_OCR_SERVICES)


def _parse_list_env(raw_value: str) -> list[str]:
    """comma/semicolon/newline 구분 문자열을 중복 없이 파싱한다."""
    results: list[str] = []
    for part in re.split(r"[\n,;]+", raw_value or ""):
        item = part.strip()
        if item and item not in results:
            results.append(item)
    return results


def _normalize_target_name(text: str) -> str:
    """target 비교용 문자열을 느슨하게 정규화한다."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").strip().lower()).strip()


def _sanitize_target_key(text: str) -> str:
    """target key 용 안전한 snake_case 문자열을 만든다."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return normalized or "target"


def _parse_target_labels() -> list[str]:
    """순차 탐지할 target label 목록을 반환한다."""
    raw_value = os.getenv("TOOL_SCREEN_READ_TARGETS", "").strip()
    if raw_value:
        parsed = _parse_list_env(raw_value)
        if parsed:
            return parsed

    defaults = [TARGET_TITLE, *DEFAULT_ITERATION_TARGET_LABELS]
    resolved: list[str] = []
    for item in defaults:
        if item and item not in resolved:
            resolved.append(item)
    return resolved


def _build_title_target(title_text: str) -> ScreenReadTarget:
    """작은 내부 창 제목 탐지용 target 설정을 만든다."""
    return ScreenReadTarget(
        key="recipe_monitor_title",
        label=title_text,
        description=(
            f"the visible title text '{title_text}' of the small child window "
            "inside the application screenshot"
        ),
        focus_words=(title_text,),
        left_pad_ratio=2.0,
        right_pad_ratio=2.0,
        vertical_pad_ratio=2.2,
        min_crop_width=360,
        min_crop_height=140,
    )


def _build_generic_control_target(target_text: str) -> ScreenReadTarget:
    """일반 button/control 탐지용 target 설정을 만든다."""
    normalized_key = _sanitize_target_key(target_text)
    return ScreenReadTarget(
        key=normalized_key,
        label=target_text,
        description=(
            f"the clickable UI control, button, or tab that best matches '{target_text}' "
            "in the application screenshot. Return the center of the actual clickable target. "
            "If multiple similar controls exist, prefer the one whose nearby text most strongly "
            "matches the full target phrase."
        ),
        focus_words=(target_text,),
        left_pad_ratio=1.2,
        right_pad_ratio=1.2,
        vertical_pad_ratio=1.8,
        min_crop_width=420,
        min_crop_height=180,
    )


def _make_unique_target_key(base_key: str, used_keys: set[str]) -> str:
    """중복 target key 가 생기지 않도록 suffix 를 붙인다."""
    if base_key not in used_keys:
        return base_key

    index = 2
    while True:
        candidate = f"{base_key}_{index:02d}"
        if candidate not in used_keys:
            return candidate
        index += 1


def _resolve_targets() -> list[ScreenReadTarget]:
    """환경변수/기본값으로부터 실행 target 목록을 구성한다."""
    resolved: list[ScreenReadTarget] = []
    used_keys: set[str] = set()
    title_name = _normalize_target_name(TARGET_TITLE)

    for label in _parse_target_labels():
        if _normalize_target_name(label) == title_name:
            target = _build_title_target(label)
        else:
            target = _build_generic_control_target(label)

        target.key = _make_unique_target_key(target.key, used_keys)
        used_keys.add(target.key)
        resolved.append(target)
    return resolved


def _collect_target_focus_words(targets: list[ScreenReadTarget]) -> list[str]:
    """OCR prompt 에 넣을 focus words 를 target 목록에서 모은다."""
    results: list[str] = []
    seen: set[str] = set()

    for target in targets:
        for focus_word in target.focus_words:
            normalized = focus_word.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(normalized)
    return results


def _resolve_ocr_max_tokens() -> int:
    """OCR 응답용 최대 토큰 수를 안전 범위로 맞춘다."""
    raw_value = os.getenv("TOOL_SCREEN_READ_OCR_MAX_TOKENS", "").strip()
    if not raw_value:
        return DEFAULT_OCR_MAX_TOKENS

    try:
        resolved = int(raw_value)
    except ValueError:
        print(
            "[WARNING] TOOL_SCREEN_READ_OCR_MAX_TOKENS 값이 잘못되었습니다. "
            f"default={DEFAULT_OCR_MAX_TOKENS} 를 사용합니다: {raw_value!r}"
        )
        return DEFAULT_OCR_MAX_TOKENS

    if resolved <= 0:
        print(
            "[WARNING] TOOL_SCREEN_READ_OCR_MAX_TOKENS 는 1 이상이어야 합니다. "
            f"default={DEFAULT_OCR_MAX_TOKENS} 를 사용합니다: {resolved}"
        )
        return DEFAULT_OCR_MAX_TOKENS

    return min(resolved, 1024)


def _sanitize_stem(text: str) -> str:
    """파일명/아티팩트 prefix 에 사용할 안전한 문자열로 정규화한다."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()).strip("._-")
    return normalized or "capture"


def _collect_capture_images() -> list[Path]:
    """분석 대상 캡처 이미지 목록을 반환한다."""
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


def _collect_artifact_paths(debug_dir: Path, artifact_prefix: str) -> list[str]:
    """artifact prefix 로 생성된 디버그 파일 경로를 모아 반환한다."""
    if not debug_dir.is_dir():
        return []

    results: list[str] = []
    pattern = f"*{artifact_prefix}*"
    for path in sorted(debug_dir.rglob(pattern)):
        if path.is_file():
            results.append(str(path))
    return results


def _load_image(image_path: Path) -> Image.Image:
    """이미지 파일을 RGB PIL Image 로 읽는다."""
    with Image.open(image_path) as opened:
        return opened.convert("RGB")


def _save_source_artifacts(
    image: Image.Image,
    debug_dir: Path,
    artifact_prefix: str,
    timestamp_tag: str,
) -> dict[str, Path]:
    """원본 캡처와 전송용 WebP 를 저장한다."""
    capture_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_capture.jpg",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    webp_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_input.webp",
        model_name="source",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, capture_path, log_name=LOG_NAME)
    save_debug_webp(image, webp_path, quality=90, log_name=LOG_NAME)
    return {
        "capture": capture_path,
        "webp": webp_path,
    }


def _print_ocr_preview(service_slug: str, lines: list[str]) -> None:
    """OCR 텍스트 앞부분을 콘솔에 출력한다."""
    if not lines:
        print(f"[INFO] OCR preview 없음: service={service_slug}")
        return

    print(f"[INFO] OCR preview: service={service_slug}")
    for index, line in enumerate(lines[:12], start=1):
        print(f"  {index:02d}. {line}")
    if len(lines) > 12:
        print(f"  ... ({len(lines) - 12} more line(s))")


def _build_got_ocr_endpoint(base_url: str) -> str:
    """GOT-OCR `/v1/ocr` 엔드포인트를 구성한다."""
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/ocr"
    return f"{normalized}/v1/ocr"


def _extract_got_ocr_text(body) -> str:
    """GOT-OCR 응답 본문에서 텍스트 필드를 추출한다."""
    if isinstance(body, dict):
        for key in ("text", "output_text", "content", "result"):
            value = body.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return str(body).strip()


def _run_chat_ocr(
    service_slug: str,
    image_webp_path: Path,
    image_width: int,
    image_height: int,
    focus_words: list[str],
    debug_dir: Path,
    artifact_prefix: str,
    timestamp_tag: str,
) -> dict:
    """chat-completions 기반 OCR 서비스를 호출한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return {
            "service_slug": service_slug,
            "status": "error",
            "error": "service_not_found",
        }

    system_message, user_text = build_ocr_assist_prompt(
        image_width,
        image_height,
        context_label="tool_screen",
        focus_words=focus_words or [TARGET_TITLE],
    )
    response_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_{service_slug}_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_{service_slug}_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )

    client = Work2VLMClient(
        service_slug=service_slug,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        log_name=LOG_NAME,
    )
    started_at = time.time()
    try:
        response = client.chat_with_image_path(
            image_path=image_webp_path,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=_resolve_ocr_max_tokens(),
        )
        raw_text = response.text.strip()
        normalized_lines = normalize_lines(raw_text)
        save_debug_text(response_path, raw_text)
        save_debug_json(
            result_path,
            {
                "service_slug": response.service_slug,
                "model_name": response.model_name,
                "api_url": response.api_url,
                "endpoint": client.endpoint,
                "prompt_text": user_text,
                "raw_text": raw_text,
                "normalized_lines": normalized_lines,
                "line_count": len(normalized_lines),
                "token_usage": response.token_usage,
                "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            },
        )
        _print_ocr_preview(service_slug, normalized_lines)
        return {
            "service_slug": response.service_slug,
            "model_name": response.model_name,
            "status": "success",
            "line_count": len(normalized_lines),
            "normalized_lines": normalized_lines,
            "token_usage": response.token_usage,
            "artifacts": {
                "response_text": str(response_path),
                "result_json": str(result_path),
            },
        }
    except Exception as exc:
        error_text = str(exc)
        save_debug_json(
            result_path,
            {
                "service_slug": service_slug,
                "model_name": service_entry.model_name,
                "status": "error",
                "error": error_text,
                "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            },
        )
        print(f"[ERROR] OCR 요청 실패: service={service_slug}, error={error_text}")
        return {
            "service_slug": service_slug,
            "model_name": service_entry.model_name,
            "status": "error",
            "error": error_text,
            "artifacts": {
                "result_json": str(result_path),
            },
        }


def _run_got_ocr(
    image_webp_path: Path,
    debug_dir: Path,
    artifact_prefix: str,
    timestamp_tag: str,
) -> dict:
    """GOT-OCR `/v1/ocr` 엔드포인트를 직접 호출한다."""
    service_slug = "got-ocr"
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return {
            "service_slug": service_slug,
            "status": "error",
            "error": "service_not_found",
        }

    result_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_{service_slug}_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )
    response_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_{service_slug}_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=timestamp_tag,
    )

    endpoint = _build_got_ocr_endpoint(resolve_service_proxy_url(service_slug))
    headers = {"Content-Type": "application/json"}
    api_key = resolve_service_api_key(service_slug)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "image": base64.b64encode(image_webp_path.read_bytes()).decode("utf-8"),
        "format_output": GOT_FORMAT_OUTPUT,
        "crop_to_patches": GOT_CROP_TO_PATCHES,
    }
    started_at = time.time()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        response.raise_for_status()

        try:
            body = response.json()
        except ValueError:
            body = response.text

        raw_text = _extract_got_ocr_text(body)
        normalized_lines = normalize_lines(raw_text)
        save_debug_text(response_path, raw_text)
        save_debug_json(
            result_path,
            {
                "service_slug": service_slug,
                "model_name": service_entry.model_name,
                "api_url": resolve_service_proxy_url(service_slug),
                "endpoint": endpoint,
                "status_code": response.status_code,
                "format_output": GOT_FORMAT_OUTPUT,
                "crop_to_patches": GOT_CROP_TO_PATCHES,
                "raw_text": raw_text,
                "normalized_lines": normalized_lines,
                "line_count": len(normalized_lines),
                "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            },
        )
        _print_ocr_preview(service_slug, normalized_lines)
        return {
            "service_slug": service_slug,
            "model_name": service_entry.model_name,
            "status": "success",
            "line_count": len(normalized_lines),
            "normalized_lines": normalized_lines,
            "artifacts": {
                "response_text": str(response_path),
                "result_json": str(result_path),
            },
        }
    except Exception as exc:
        error_text = str(exc)
        save_debug_json(
            result_path,
            {
                "service_slug": service_slug,
                "model_name": service_entry.model_name,
                "status": "error",
                "error": error_text,
                "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            },
        )
        print(f"[ERROR] OCR 요청 실패: service={service_slug}, error={error_text}")
        return {
            "service_slug": service_slug,
            "model_name": service_entry.model_name,
            "status": "error",
            "error": error_text,
            "artifacts": {
                "result_json": str(result_path),
            },
        }


def _run_ocr_services(
    image_webp_path: Path,
    image_width: int,
    image_height: int,
    focus_words: list[str],
    debug_dir: Path,
    artifact_prefix: str,
    timestamp_tag: str,
) -> list[dict]:
    """설정된 OCR 서비스들을 순차 호출한다."""
    results: list[dict] = []
    for service_slug in _parse_ocr_services():
        print(f"[INFO] OCR 시작: service={service_slug}")
        if service_slug == "got-ocr":
            result = _run_got_ocr(
                image_webp_path=image_webp_path,
                debug_dir=debug_dir,
                artifact_prefix=artifact_prefix,
                timestamp_tag=timestamp_tag,
            )
        else:
            result = _run_chat_ocr(
                service_slug=service_slug,
                image_webp_path=image_webp_path,
                image_width=image_width,
                image_height=image_height,
                focus_words=focus_words,
                debug_dir=debug_dir,
                artifact_prefix=artifact_prefix,
                timestamp_tag=timestamp_tag,
            )
        results.append(result)
    return results


def _run_vlm_target_locates(
    image: Image.Image,
    image_path: Path,
    targets: list[ScreenReadTarget],
    debug_dir: Path,
    artifact_prefix: str,
) -> list[dict]:
    """ui-venus + mai-ui 로 여러 target 을 순차 탐지한다."""
    results: list[dict] = []
    for target in targets:
        print(f"[INFO] VLM target 탐지 시작: key={target.key}, label={target.label!r}")
        current_prefix = f"{artifact_prefix}_{target.key}"
        locate_result = analyze_window_target(
            window=None,
            window_title=image_path.name,
            backend="offline_file",
            target=target.to_target_config(),
            debug_image_dir=debug_dir,
            log_name=LOG_NAME,
            component_name="tool_screen_read",
            artifact_prefix=current_prefix,
            coarse_service_slug="ui-venus",
            refine_service_slug="mai-ui",
            result_mode=f"tool_screen_{target.key}_offline",
            image=image,
        )
        generated_artifacts = _collect_artifact_paths(debug_dir, current_prefix)
        results.append(
            {
                "status": locate_result.exit_code,
                "target_key": locate_result.target_key,
                "target_label": target.label,
                "target_description": target.description,
                "point": locate_result.point,
                "artifacts": generated_artifacts,
            }
        )
    return results


def _find_title_target_result(
    target_results: list[dict],
    targets: list[ScreenReadTarget],
) -> dict | None:
    """기존 summary 호환용으로 title target 결과를 하나 반환한다."""
    title_name = _normalize_target_name(TARGET_TITLE)
    target_map = {target.key: target for target in targets}
    for result in target_results:
        target = target_map.get(result.get("target_key", ""))
        if target is None:
            continue
        if _normalize_target_name(target.label) == title_name:
            return result
    return None


def _analyze_single_image(image_path: Path, targets: list[ScreenReadTarget]) -> dict:
    """단일 스크린샷에 대해 OCR/VLM 분석을 수행한다."""
    started_at = time.time()
    timestamp_tag = make_timestamp_tag(started_at)
    artifact_prefix = _sanitize_stem(image_path.stem)
    debug_dir = DEBUG_IMAGE_DIR / artifact_prefix
    focus_words = _collect_target_focus_words(targets)

    print(f"[INFO] 분석 시작: image={image_path}")
    image = _load_image(image_path)
    source_artifacts = _save_source_artifacts(
        image,
        debug_dir=debug_dir,
        artifact_prefix=artifact_prefix,
        timestamp_tag=timestamp_tag,
    )
    ocr_results = _run_ocr_services(
        image_webp_path=source_artifacts["webp"],
        image_width=image.size[0],
        image_height=image.size[1],
        focus_words=focus_words,
        debug_dir=debug_dir,
        artifact_prefix=artifact_prefix,
        timestamp_tag=timestamp_tag,
    )
    target_results = _run_vlm_target_locates(
        image=image,
        image_path=image_path,
        targets=targets,
        debug_dir=debug_dir,
        artifact_prefix=artifact_prefix,
    )
    title_result = _find_title_target_result(target_results, targets)

    status = EXIT_SUCCESS
    if any(result.get("status") != EXIT_SUCCESS for result in target_results) or any(
        result.get("status") != EXIT_SUCCESS for result in ocr_results
    ):
        status = EXIT_PARTIAL

    result_payload = {
        "status": status,
        "image_path": str(image_path),
        "image_width": image.size[0],
        "image_height": image.size[1],
        "target_title": TARGET_TITLE,
        "target_labels": [target.label for target in targets],
        "ocr_results": ocr_results,
        "vlm_recipe_monitor": title_result,
        "target_results": target_results,
        "source_artifacts": {
            "capture": str(source_artifacts["capture"]),
            "webp": str(source_artifacts["webp"]),
        },
        "all_debug_artifacts": _collect_artifact_paths(debug_dir, artifact_prefix),
        "elapsed_ms": round((time.time() - started_at) * 1000, 1),
    }
    result_path = debug_image_path(
        debug_dir,
        f"{artifact_prefix}_analysis_summary.json",
        model_name="summary",
        timestamp_tag=timestamp_tag,
    )
    save_debug_json(result_path, result_payload)
    print(
        f"[INFO] 분석 완료: image={image_path.name}, status={status}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return result_payload


def main() -> int:
    """capture_images 폴더의 모든 이미지를 분석한다."""
    started_at = time.time()
    if not CAPTURE_IMAGE_DIR.is_dir():
        print(f"[ERROR] capture_images 폴더를 찾지 못했습니다: {CAPTURE_IMAGE_DIR}")
        log_work2_event(
            component="tool_screen_read",
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
            component="tool_screen_read",
            message="no_images_found",
            level="warning",
            log_name=LOG_NAME,
            capture_dir=CAPTURE_IMAGE_DIR,
            image_filter=IMAGE_FILTER,
        )
        return 1

    targets = _resolve_targets()
    print(
        f"[INFO] tool screen 판독 시작: image_count={len(image_paths)}, "
        f"targets={[target.label for target in targets]}, "
        f"ocr_services={_parse_ocr_services()}"
    )
    results: list[dict] = []
    for image_path in image_paths:
        try:
            results.append(_analyze_single_image(image_path, targets))
        except Exception as exc:
            error_text = str(exc)
            print(f"[ERROR] 이미지 분석 실패: image={image_path}, error={error_text}")
            results.append(
                {
                    "status": "error",
                    "image_path": str(image_path),
                    "error": error_text,
                }
            )

    success_count = sum(1 for item in results if item.get("status") == EXIT_SUCCESS)
    partial_count = sum(1 for item in results if item.get("status") == EXIT_PARTIAL)
    error_count = sum(1 for item in results if item.get("status") == "error")

    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_screen_read_summary.json",
        model_name="summary",
        timestamp_tag=make_timestamp_tag(started_at),
    )
    save_debug_json(
        summary_path,
        {
            "capture_dir": str(CAPTURE_IMAGE_DIR),
            "target_title": TARGET_TITLE,
            "target_labels": [target.label for target in targets],
            "ocr_services": _parse_ocr_services(),
            "image_filter": IMAGE_FILTER,
            "image_count": len(image_paths),
            "success_count": success_count,
            "partial_count": partial_count,
            "error_count": error_count,
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            "results": results,
        },
    )

    print(
        f"[INFO] 전체 완료: success={success_count}, partial={partial_count}, "
        f"error={error_count}, elapsed={format_elapsed_ms(started_at)}"
    )
    print(f"[INFO] summary 저장: {summary_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
