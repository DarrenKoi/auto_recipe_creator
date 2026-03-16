"""단일 모니터 스크린샷을 여러 UI VLM 에 보내 UI component + click 좌표 응답을 비교한다.

기본적으로 Flask health payload 에서 현재 `serving` 상태인 UI 계열 모델을 자동 선택한다.
필요하면 아래 환경변수로 대상과 프롬프트를 고정할 수 있다.

사용법:
  uv run python poc/work2/reading_check.py

환경변수:
  READING_CHECK_MONITOR_INDEX=1
  READING_CHECK_SERVICES=ui-venus,mai-ui
  READING_CHECK_INCLUDE_OCR=true
  READING_CHECK_MAX_COMPONENTS=12
  READING_CHECK_FOCUS_TEXTS=View,List,Login
"""

import base64
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import mss
import mss.tools
import requests
from PIL import Image, ImageDraw, ImageFont

from poc.work2 import debug_image_dir
from poc.work2.vlm_client import ChatImageRequest, OpenAICompatibleVLMClient
from poc.work2.flask_vlm import (
    apply_pipeline_env_defaults,
    fetch_vlm_health,
    get_service_by_slug,
    normalize_vlm_health_entries,
)
from poc.work2.logger import log_vlm_call


DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
SOURCE_JPEG_QUALITY = 85
WEBP_QUALITY = 90
MAX_VLM_BYTES = 1_000_000
REQUEST_TIMEOUT_SEC = 120.0
DEFAULT_MONITOR_INDEX = 1
DEFAULT_TARGET_SERVICE_SLUGS = ("ui-venus", "mai-ui")
DEFAULT_MAX_COMPONENTS = 12
OVERLAY_COLORS = (
    "orange",
    "cyan",
    "lime",
    "magenta",
    "yellow",
    "red",
    "white",
    "deepskyblue",
)


def build_ui_component_locator_prompt(
    width: int,
    height: int,
    *,
    max_components: int,
    focus_texts: tuple[str, ...],
) -> tuple[str, str]:
    """UI component + click 좌표 비교용 프롬프트를 구성한다."""
    system_message = (
        "You are a precise desktop GUI element locator. "
        f"The image is {width}x{height} pixels. "
        "The origin (0, 0) is the top-left corner. "
        "Return integer pixel coordinates for where a human should click. "
        "Use only visible evidence from the screenshot. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Read this computer screenshot and identify visible, clickable UI components.",
        f"Return up to {max_components} components that are most useful for GUI automation.",
        "Prefer tabs, buttons, menu items, text inputs, tree/list rows, toolbar icons, and dialog actions.",
        "For each component, provide one click point at the visual center that a human should click.",
        "If text is unclear, keep the component but mention the uncertainty in uncertain_areas.",
        "",
        "Return ONLY this JSON:",
        "{",
        '  "screen_summary": "...",',
        '  "components": [',
        "    {",
        '      "name": "...",',
        '      "role": "button|tab|input|menu|list_item|icon|checkbox|radio|dropdown|other",',
        '      "text": "...",',
        '      "click_point": {"x": 0, "y": 0},',
        '      "confidence": 0.0',
        "    }",
        "  ],",
        '  "uncertain_areas": ["..."]',
        "}",
        "",
        f"Image size: {width} x {height}.",
        "All x and y values must be integers inside the image bounds when possible.",
    ]

    if focus_texts:
        lines.extend(
            [
                "",
                "Prioritize components whose visible text matches or is close to these anchors:",
                ", ".join(focus_texts),
            ]
        )

    return system_message, "\n".join(lines)


def _slugify(value: str) -> str:
    """파일명용 안전 문자열로 변환한다."""
    allowed = []
    for char in value.strip().lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def _ensure_debug_dir(debug_dir: Path) -> None:
    """지정한 디버그 디렉터리를 보장한다."""
    debug_dir.mkdir(parents=True, exist_ok=True)


def _parse_csv_env(name: str, default_values: tuple[str, ...]) -> tuple[str, ...]:
    """콤마/세미콜론 구분 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default_values

    values = []
    seen: set[str] = set()
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(values) or default_values


def _env_flag(name: str, default: bool = False) -> bool:
    """bool 형 환경변수를 해석한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _resolve_monitor_index() -> int:
    """reading_check 에서 사용할 물리 모니터 인덱스를 반환한다."""
    raw = os.environ.get("READING_CHECK_MONITOR_INDEX", str(DEFAULT_MONITOR_INDEX)).strip()
    try:
        monitor_index = int(raw)
    except ValueError:
        print(
            f"[WARNING] READING_CHECK_MONITOR_INDEX 값이 잘못되었습니다: {raw!r} "
            f"-> 기본값 {DEFAULT_MONITOR_INDEX} 사용"
        )
        return DEFAULT_MONITOR_INDEX

    if monitor_index < 1:
        print(
            f"[WARNING] READING_CHECK_MONITOR_INDEX 는 1 이상이어야 합니다: {monitor_index} "
            f"-> 기본값 {DEFAULT_MONITOR_INDEX} 사용"
        )
        return DEFAULT_MONITOR_INDEX

    return monitor_index


def _resolve_max_components() -> int:
    """요청할 최대 UI component 수를 반환한다."""
    raw = os.environ.get("READING_CHECK_MAX_COMPONENTS", str(DEFAULT_MAX_COMPONENTS)).strip()
    try:
        value = int(raw)
    except ValueError:
        print(
            f"[WARNING] READING_CHECK_MAX_COMPONENTS 값이 잘못되었습니다: {raw!r} "
            f"-> 기본값 {DEFAULT_MAX_COMPONENTS} 사용"
        )
        return DEFAULT_MAX_COMPONENTS

    return max(1, value)


def _resolve_focus_texts() -> tuple[str, ...]:
    """우선 탐색할 텍스트 anchor 목록을 반환한다."""
    return _parse_csv_env("READING_CHECK_FOCUS_TEXTS", ())


def _looks_like_ocr_service(service_row: dict[str, Any]) -> bool:
    """service/model 이름 기반으로 OCR 계열 여부를 추정한다."""
    haystack = " ".join(
        [
            str(service_row.get("service") or ""),
            str(service_row.get("display_name") or ""),
            str(service_row.get("expected_model") or ""),
        ]
    ).lower()
    return "ocr" in haystack


def _capture_monitor_image(monitor_index: int) -> Image.Image | None:
    """단일 물리 모니터를 캡처하여 PIL 이미지로 반환한다."""
    try:
        sct = mss.mss()
    except Exception as exc:
        print(f"[ERROR] mss 초기화 실패: {exc}")
        return None

    try:
        monitors = sct.monitors
        physical_count = max(0, len(monitors) - 1)
        if physical_count <= 0:
            print("[ERROR] 현재 환경에서는 물리 모니터가 감지되지 않았습니다.")
            return None

        if monitor_index > physical_count:
            available = ", ".join(str(idx) for idx in range(1, physical_count + 1))
            print(
                f"[ERROR] 요청한 monitor index {monitor_index} 는 범위를 벗어났습니다. "
                f"사용 가능: {available}"
            )
            return None

        screenshot = sct.grab(monitors[monitor_index])
        png_data = mss.tools.to_png(screenshot.rgb, screenshot.size)
        print(
            f"[INFO] 모니터 {monitor_index} 캡처 완료: "
            f"{screenshot.width}x{screenshot.height}"
        )
    except Exception as exc:
        print(f"[ERROR] 모니터 {monitor_index} 캡처 실패: {exc}")
        return None
    finally:
        try:
            sct.close()
        except Exception:
            pass

    image = Image.open(BytesIO(png_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _save_source_jpeg(image: Image.Image, stem: str, debug_dir: Path) -> Path:
    """원본 확인용 JPEG를 저장한다."""
    out_path = debug_dir / f"{stem}_source.jpg"
    image.save(out_path, format="JPEG", quality=SOURCE_JPEG_QUALITY)
    print(f"[INFO] 원본 확인용 JPEG 저장: {out_path}")
    return out_path


def _encode_image_for_vlm(
    image: Image.Image,
    stem: str,
    debug_dir: Path,
) -> tuple[str, int, int, Path]:
    """전송용 WebP를 저장하고 base64 문자열을 반환한다."""
    working = image
    quality = WEBP_QUALITY
    webp_bytes = b""

    while quality >= 10:
        buffer = BytesIO()
        working.save(buffer, format="WEBP", quality=quality)
        webp_bytes = buffer.getvalue()
        if len(webp_bytes) <= MAX_VLM_BYTES:
            break
        print(f"[INFO] WebP {len(webp_bytes):,}B > 1MB, quality {quality} -> {quality - 10}")
        quality -= 10

    if len(webp_bytes) > MAX_VLM_BYTES:
        scale = (MAX_VLM_BYTES / len(webp_bytes)) ** 0.5
        new_w = max(1, int(working.width * scale))
        new_h = max(1, int(working.height * scale))
        print(f"[INFO] WebP가 계속 큼 -> 리사이즈 적용: {working.width}x{working.height} -> {new_w}x{new_h}")
        working = working.resize((new_w, new_h), Image.LANCZOS)
        buffer = BytesIO()
        working.save(buffer, format="WEBP", quality=70)
        webp_bytes = buffer.getvalue()

    sent_path = debug_dir / f"{stem}_sent.webp"
    sent_path.write_bytes(webp_bytes)
    print(
        f"[INFO] 전송용 WebP 저장: {sent_path} "
        f"({len(webp_bytes) / 1024:.1f}KB, {working.width}x{working.height})"
    )

    image_b64 = base64.b64encode(webp_bytes).decode("utf-8")
    return image_b64, working.width, working.height, sent_path


def _save_text(path: Path, text: str) -> None:
    """UTF-8 텍스트 파일로 저장한다."""
    path.write_text(text, encoding="utf-8")
    print(f"[INFO] 응답 저장: {path}")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    """JSON 파일로 저장한다."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] JSON 저장: {path}")


def _copy_debug_artifact(src_path: Path, dst_dir: Path) -> Path:
    """공용 캡처 산출물을 모델별 디렉터리로 복사한다."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    if src_path.resolve() == dst_path.resolve():
        return dst_path
    dst_path.write_bytes(src_path.read_bytes())
    print(f"[INFO] 디버그 산출물 복사: {dst_path}")
    return dst_path


def _extract_json(text: str) -> dict[str, Any]:
    """모델 응답에서 첫 JSON 객체를 추출한다."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return json.loads(text[start:end].strip())

    if "{" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return json.loads(text[start : end + 1])

    return json.loads(text)


def _resolve_request_model_name(service_name: str, configured_model: str) -> str:
    """서비스 slug 기준의 canonical served model name 을 반환한다."""
    service_entry = get_service_by_slug(service_name)
    if service_entry is None:
        model_name = configured_model.strip()
        if not model_name:
            raise ValueError(f"등록되지 않은 서비스이며 model name 도 비어 있습니다: {service_name}")
        return model_name

    canonical_model = service_entry.model_name.strip()
    configured = configured_model.strip()
    if configured and configured != canonical_model:
        print(
            f"[WARNING] {service_name} configured model mismatch: "
            f"{configured} -> {canonical_model} 로 교정합니다."
        )
    return canonical_model


def _models_endpoint(api_url: str) -> str:
    """OpenAI-compatible /v1/models endpoint 를 계산한다."""
    base_url = api_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/models"
    return f"{base_url}/v1/models"


def _fetch_advertised_models(
    *,
    api_url: str,
    api_key: str,
    timeout_sec: float = 10.0,
) -> list[str]:
    """서비스가 실제로 광고하는 model id 목록을 가져온다."""
    headers = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    url = _models_endpoint(api_url)
    print(f"[INFO] model probe 호출: {url}")
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"/v1/models 응답 형식이 올바르지 않습니다: {payload}")

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id:
            models.append(model_id)

    if not models:
        raise ValueError("/v1/models 응답에 model id 가 없습니다.")
    return models


def _verify_service_model(
    *,
    service_name: str,
    configured_model: str,
    api_url: str,
    api_key: str,
) -> tuple[str, list[str]]:
    """서비스 endpoint 와 요청 model name 이 일치하는지 검증한다."""
    request_model = _resolve_request_model_name(service_name, configured_model)
    advertised_models = _fetch_advertised_models(
        api_url=api_url,
        api_key=api_key,
    )
    if request_model not in advertised_models:
        raise ValueError(
            f"service={service_name} expected model={request_model}, "
            f"but endpoint advertises {advertised_models}"
        )

    print(
        f"[INFO] model probe 확인 완료: service={service_name}, "
        f"request_model={request_model}, advertised={advertised_models}"
    )
    return request_model, advertised_models


def _build_proxy_url(flask_base_url: str, service_slug: str) -> str:
    """Flask base URL 에서 service proxy base URL 을 구성한다."""
    base_url = flask_base_url.rstrip("/")
    if not base_url:
        return ""

    service_path = f"/api/vlm_serve/{service_slug}"
    if base_url.endswith(service_path):
        return base_url
    if base_url.endswith("/api/vlm_serve"):
        return f"{base_url}/{service_slug}"
    if base_url.endswith("/api"):
        return f"{base_url}/vlm_serve/{service_slug}"
    return f"{base_url}{service_path}"


def _discover_health_rows(pipeline: dict[str, object]) -> list[dict[str, Any]]:
    """Flask health 에서 현재 서비스 상태 목록을 가져온다."""
    flask_base_url = str(pipeline.get("flask_api_base_url") or "").rstrip("/")
    if not flask_base_url:
        return []

    try:
        health_body = fetch_vlm_health(flask_base_url=flask_base_url, timeout_sec=10.0)
    except (requests.RequestException, ValueError) as exc:
        print(f"[WARNING] health 기반 서비스 탐색 실패: {exc}")
        return []

    rows = normalize_vlm_health_entries(health_body, flask_base_url=flask_base_url)
    if rows:
        print("[INFO] health 기반 현재 서비스 상태:")
        for row in rows:
            print(
                "  "
                f"{row['service']} -> health={row.get('health_status') or '-'}, "
                f"model={row.get('expected_model') or '-'}, "
                f"proxy_registered={row.get('proxy_registered')}"
            )
    return rows


def _build_service_target(
    service_slug: str,
    pipeline: dict[str, object],
    *,
    health_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """service slug + optional health row 로 실제 호출 대상 정보를 구성한다."""
    flask_base_url = str(pipeline.get("flask_api_base_url") or "").rstrip("/")
    screen_analysis_service = str(pipeline.get("screen_analysis_service") or "").strip()
    screen_analysis_api_url = str(pipeline.get("screen_analysis_api_url") or "").strip()
    screen_analysis_api_key = str(pipeline.get("screen_analysis_api_key") or "").strip()
    screen_analysis_model = str(pipeline.get("screen_analysis_model_name") or "").strip()
    ocr_service = str(pipeline.get("ocr_service") or "").strip()
    ocr_api_url = str(pipeline.get("ocr_api_url") or "").strip()
    ocr_api_key = str(pipeline.get("ocr_api_key") or "").strip()
    ocr_model = str(pipeline.get("ocr_model_name") or "").strip()
    shared_api_key = str(pipeline.get("shared_api_key") or "").strip()

    service_entry = get_service_by_slug(service_slug)
    if service_slug == screen_analysis_service:
        api_url = screen_analysis_api_url
        api_key = screen_analysis_api_key
        configured_model = screen_analysis_model
    elif service_slug == ocr_service:
        api_url = ocr_api_url
        api_key = ocr_api_key
        configured_model = ocr_model
    else:
        api_url = str((health_row or {}).get("api_url") or "").strip() or _build_proxy_url(flask_base_url, service_slug)
        api_key = shared_api_key
        configured_model = str((health_row or {}).get("expected_model") or "").strip()
        if not configured_model and service_entry is not None:
            configured_model = service_entry.model_name

    return {
        "service": service_slug,
        "api_url": api_url,
        "api_key": api_key,
        "configured_model": configured_model,
        "service_known": service_entry is not None,
        "service_enabled": None if service_entry is None else service_entry.enabled,
        "display_name": str((health_row or {}).get("display_name") or service_slug),
        "health_status": str((health_row or {}).get("health_status") or "").strip(),
        "proxy_registered": bool((health_row or {}).get("proxy_registered")),
        "reason": str((health_row or {}).get("reason") or "").strip(),
    }


def _resolve_service_targets(
    pipeline: dict[str, object],
    health_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """비교 대상 service/api/model 정보를 계산한다."""
    explicit_services = _parse_csv_env("READING_CHECK_SERVICES", ())
    include_ocr = _env_flag("READING_CHECK_INCLUDE_OCR", False)

    if explicit_services:
        health_map = {str(row["service"]): row for row in health_rows}
        return [
            _build_service_target(service_slug, pipeline, health_row=health_map.get(service_slug))
            for service_slug in explicit_services
        ]

    if health_rows:
        serving_rows = [
            row for row in health_rows
            if str(row.get("health_status") or "").strip().lower() == "serving"
            and bool(row.get("proxy_registered"))
        ]
        if not include_ocr:
            ui_rows = [row for row in serving_rows if not _looks_like_ocr_service(row)]
            if ui_rows:
                serving_rows = ui_rows

        if serving_rows:
            return [
                _build_service_target(str(row["service"]), pipeline, health_row=row)
                for row in serving_rows
            ]

    return [
        _build_service_target(service_slug, pipeline)
        for service_slug in DEFAULT_TARGET_SERVICE_SLUGS
    ]


def _normalize_response_payload(parsed_json: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """응답 JSON 을 component 목록 형태로 정규화한다."""
    components_raw = parsed_json.get("components")
    if not isinstance(components_raw, list):
        components_raw = parsed_json.get("ui_components")
    if not isinstance(components_raw, list):
        components_raw = []

    uncertain_raw = parsed_json.get("uncertain_areas")
    uncertain_areas = []
    if isinstance(uncertain_raw, list):
        uncertain_areas = [str(item).strip() for item in uncertain_raw if str(item).strip()]
    elif isinstance(uncertain_raw, str) and uncertain_raw.strip():
        uncertain_areas = [uncertain_raw.strip()]

    normalized = {
        "screen_summary": str(parsed_json.get("screen_summary", "") or "").strip(),
        "components": [],
        "uncertain_areas": uncertain_areas,
    }

    for idx, component in enumerate(components_raw, start=1):
        if not isinstance(component, dict):
            continue

        click_point = component.get("click_point")
        if not isinstance(click_point, dict):
            click_point = component.get("point")
        if not isinstance(click_point, dict):
            click_point = {"x": component.get("x"), "y": component.get("y")}

        try:
            x = int(float(click_point.get("x")))  # type: ignore[arg-type]
            y = int(float(click_point.get("y")))  # type: ignore[arg-type]
        except (AttributeError, TypeError, ValueError):
            continue

        raw_confidence = component.get("confidence")
        confidence = None
        try:
            if raw_confidence is not None:
                confidence = round(float(raw_confidence), 3)
        except (TypeError, ValueError):
            confidence = None

        normalized["components"].append(
            {
                "index": idx,
                "name": str(component.get("name") or component.get("label") or component.get("text") or "").strip(),
                "role": str(component.get("role") or component.get("type") or "other").strip() or "other",
                "text": str(component.get("text") or component.get("name") or "").strip(),
                "click_point": {"x": x, "y": y},
                "in_bounds": 0 <= x <= width and 0 <= y <= height,
                "confidence": confidence,
            }
        )

    return normalized


def _save_component_overlay(image: Image.Image, normalized: dict[str, Any], out_path: Path) -> None:
    """정규화된 click 좌표를 원본 이미지 위에 마킹하여 저장한다."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    radius = 12
    for idx, component in enumerate(normalized.get("components", []), start=1):
        if not isinstance(component, dict):
            continue
        point = component.get("click_point")
        if not isinstance(point, dict):
            continue

        x = int(point.get("x", 0))
        y = int(point.get("y", 0))
        color = OVERLAY_COLORS[(idx - 1) % len(OVERLAY_COLORS)]
        label_text = str(component.get("text") or component.get("name") or component.get("role") or "component")
        label = f"{idx}. {label_text[:24]} ({x},{y})"

        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)
        draw.text((x + radius + 4, y - 14), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    print(f"[INFO] overlay 저장: {out_path}")


def _call_model(
    *,
    service_name: str,
    model_name: str,
    api_url: str,
    api_key: str,
    system_message: str,
    user_text: str,
    image_b64: str,
    image_width: int,
    image_height: int,
    image_mime: str = "image/webp",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """한 모델에 reading check 요청을 보낸다."""
    client = OpenAICompatibleVLMClient(
        base_url=api_url,
        api_key=api_key,
        timeout_sec=REQUEST_TIMEOUT_SEC,
    )
    request_model = ""
    try:
        request_model = _resolve_request_model_name(service_name, model_name)
    except Exception:
        request_model = ""

    probe_started = time.time()
    try:
        request_model, advertised_models = _verify_service_model(
            service_name=service_name,
            configured_model=model_name,
            api_url=api_url,
            api_key=api_key,
        )
    except Exception as exc:
        return {
            "service": service_name,
            "configured_model": model_name,
            "model": request_model,
            "advertised_models": [],
            "ok": False,
            "latency_ms": round((time.time() - probe_started) * 1000, 1),
            "error": f"model validation failed: {exc}",
            "raw_response": "",
            "parsed_json": None,
            "normalized_json": None,
        }

    request = ChatImageRequest(
        model=request_model,
        system_message=system_message,
        user_text=user_text,
        image_b64=image_b64,
        image_mime=image_mime,
        temperature=temperature,
    )

    print(
        f"[INFO] reading check 호출: service={service_name}, "
        f"model={request_model}, endpoint={client.endpoint}"
    )
    start = time.time()
    try:
        raw_response = client.chat_with_image(request)
    except Exception as exc:
        latency_ms = round((time.time() - start) * 1000, 1)
        log_vlm_call(
            service=service_name,
            model=request_model,
            status="error",
            latency_ms=latency_ms,
            error=str(exc),
            endpoint=client.endpoint,
        )
        return {
            "service": service_name,
            "configured_model": model_name,
            "model": request_model,
            "advertised_models": advertised_models,
            "ok": False,
            "latency_ms": latency_ms,
            "error": str(exc),
            "raw_response": "",
            "parsed_json": None,
            "normalized_json": None,
        }

    latency_ms = round((time.time() - start) * 1000, 1)
    log_vlm_call(
        service=service_name,
        model=request_model,
        status="ok",
        latency_ms=latency_ms,
        token_usage=client.last_token_usage,
        endpoint=client.endpoint,
        response_text=raw_response,
    )

    parsed_json = None
    normalized_json = None
    try:
        parsed_json = _extract_json(raw_response)
        normalized_json = _normalize_response_payload(parsed_json, image_width, image_height)
    except Exception as exc:
        print(f"[WARNING] {service_name} JSON 파싱/정규화 실패: {exc}")

    return {
        "service": service_name,
        "configured_model": model_name,
        "model": request_model,
        "advertised_models": advertised_models,
        "ok": True,
        "latency_ms": latency_ms,
        "error": None,
        "raw_response": raw_response,
        "parsed_json": parsed_json,
        "normalized_json": normalized_json,
    }


def _print_result_block(result: dict[str, Any]) -> None:
    """모델별 결과 블록을 출력한다."""
    print("\n" + "=" * 80)
    print(f"[INFO] 결과 - {result['service']} ({result['model']})")
    print("=" * 80)
    if result.get("health_status"):
        print(
            f"[INFO] health status: {result['health_status']}"
            + (f" ({result['health_reason']})" if result.get("health_reason") else "")
        )
    if result.get("configured_model") and result.get("configured_model") != result.get("model"):
        print(
            f"[INFO] 요청 model 교정: configured={result['configured_model']} "
            f"-> request={result['model']}"
        )
    if result.get("advertised_models"):
        print(f"[INFO] endpoint advertised models: {result['advertised_models']}")
    if not result["ok"]:
        print(f"[ERROR] 호출 실패: {result['error']}")
        return

    print(f"[INFO] 응답 시간: {result['latency_ms']}ms")
    print(f"[INFO] 원문 응답:\n{result['raw_response']}\n")

    normalized = result.get("normalized_json")
    if not isinstance(normalized, dict):
        print("[WARNING] JSON 파싱은 되었지만 component 정규화 결과가 없습니다.")
        return

    components = normalized.get("components", [])
    print(f"[INFO] 정규화된 component 수: {len(components)}")
    if normalized.get("screen_summary"):
        print(f"[INFO] screen_summary: {normalized['screen_summary']}")
    if normalized.get("uncertain_areas"):
        print(f"[INFO] uncertain_areas: {normalized['uncertain_areas']}")

    for component in components[:8]:
        if not isinstance(component, dict):
            continue
        point = component.get("click_point", {})
        print(
            "  "
            f"[{component.get('index', '-')}] "
            f"{component.get('role', 'other'):10s} "
            f"name={str(component.get('name') or '-')[:18]!s:18s} "
            f"text={str(component.get('text') or '-')[:20]!s:20s} "
            f"click=({point.get('x', '-')}, {point.get('y', '-')}) "
            f"in_bounds={component.get('in_bounds')}"
        )


def _print_summary(results: list[dict[str, Any]]) -> None:
    """요약 테이블을 출력한다."""
    print("\n" + "-" * 100)
    print("  reading_check 결과")
    print("-" * 100)
    print(f"  {'서비스':<18} {'health':<12} {'상태':<8} {'JSON':<8} {'컴포넌트':<10} {'응답시간':<12} {'비고'}")
    print(f"  {'─' * 18} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 12} {'─' * 22}")

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        health_status = str(result.get("health_status") or "-")
        json_status = "ok" if isinstance(result.get("parsed_json"), dict) else "fail"
        normalized = result.get("normalized_json")
        component_count = "-"
        if isinstance(normalized, dict):
            component_count = str(len(normalized.get("components", [])))
        latency = f"{result['latency_ms']}ms"
        note = ""
        if result["error"]:
            note = result["error"][:36]
        elif isinstance(normalized, dict) and normalized.get("screen_summary"):
            note = str(normalized["screen_summary"])[:36]
        elif isinstance(normalized, dict):
            note = "normalized json ok"
        else:
            note = "non-json response"

        print(
            f"  {result['service']:<18} {health_status[:12]:<12} {status:<8} {json_status:<8} "
            f"{component_count:<10} {latency:<12} {note}"
        )

    print("-" * 100)


def main() -> int:
    """전체 reading check 흐름을 실행한다."""
    pipeline = apply_pipeline_env_defaults()
    monitor_index = _resolve_monitor_index()
    max_components = _resolve_max_components()
    focus_texts = _resolve_focus_texts()
    health_rows = _discover_health_rows(pipeline)
    service_targets = _resolve_service_targets(pipeline, health_rows)

    if not service_targets:
        print("[ERROR] 비교 대상 서비스가 없습니다.")
        return 2

    shared_debug_dir = debug_image_dir(
        DEBUG_IMAGE_DIR,
        "__vs__".join(
            str(target.get("configured_model") or target["service"])
            for target in service_targets
        ),
    )
    _ensure_debug_dir(shared_debug_dir)

    print("[INFO] reading_check 시작")
    print(f"[INFO] capture monitor:      {monitor_index}")
    print(f"[INFO] compare services:     {', '.join(str(t['service']) for t in service_targets)}")
    print(f"[INFO] max components:       {max_components}")
    print(f"[INFO] focus texts:          {', '.join(focus_texts) if focus_texts else '(없음)'}")
    print(f"[INFO] health rows found:    {len(health_rows)}")
    print(f"[INFO] shared debug dir:     {shared_debug_dir}")

    missing_api_targets = [
        str(target["service"])
        for target in service_targets
        if not str(target.get("api_url") or "").strip()
    ]
    if missing_api_targets:
        print(f"[ERROR] API URL을 계산하지 못한 서비스가 있습니다: {', '.join(missing_api_targets)}")
        return 3

    image = _capture_monitor_image(monitor_index)
    if image is None:
        return 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"reading_check_{timestamp}"
    if health_rows:
        health_snapshot_path = shared_debug_dir / f"{stem}_health_rows.json"
        _save_json(
            health_snapshot_path,
            {
                "services": health_rows,
            },
        )
    source_path = _save_source_jpeg(image, stem, shared_debug_dir)
    image_b64, width, height, sent_path = _encode_image_for_vlm(image, stem, shared_debug_dir)
    print(f"[INFO] 실제 전송 이미지: {sent_path}")

    system_message, user_text = build_ui_component_locator_prompt(
        width,
        height,
        max_components=max_components,
        focus_texts=focus_texts,
    )

    results = []
    for target in service_targets:
        result = _call_model(
            service_name=str(target["service"]),
            model_name=str(target.get("configured_model") or ""),
            api_url=str(target["api_url"]),
            api_key=str(target.get("api_key") or ""),
            system_message=system_message,
            user_text=user_text,
            image_b64=image_b64,
            image_width=width,
            image_height=height,
        )
        result["health_status"] = str(target.get("health_status") or "")
        result["display_name"] = str(target.get("display_name") or target["service"])
        result["proxy_registered"] = bool(target.get("proxy_registered"))
        result["health_reason"] = str(target.get("reason") or "")
        results.append(result)

    for result in results:
        result_debug_dir = debug_image_dir(
            DEBUG_IMAGE_DIR,
            str(result.get("model") or result.get("configured_model") or result["service"]),
        )
        service_slug = _slugify(str(result["service"]))

        _copy_debug_artifact(source_path, result_debug_dir)
        _copy_debug_artifact(sent_path, result_debug_dir)
        if health_rows:
            _copy_debug_artifact(health_snapshot_path, result_debug_dir)

        probe_path = result_debug_dir / f"{stem}_{service_slug}_model_probe.json"
        _save_json(
            probe_path,
            {
                "service": result["service"],
                "configured_model": result.get("configured_model", ""),
                "request_model": result.get("model", ""),
                "advertised_models": result.get("advertised_models", []),
                "display_name": result.get("display_name", ""),
                "health_status": result.get("health_status", ""),
                "proxy_registered": result.get("proxy_registered", False),
                "health_reason": result.get("health_reason", ""),
                "ok": result["ok"],
                "error": result["error"],
            },
        )

        raw_path = result_debug_dir / f"{stem}_{service_slug}_response.txt"
        _save_text(raw_path, str(result["raw_response"] or result["error"] or ""))

        if isinstance(result.get("parsed_json"), dict):
            json_path = result_debug_dir / f"{stem}_{service_slug}_response.json"
            _save_json(json_path, result["parsed_json"])

        if isinstance(result.get("normalized_json"), dict):
            normalized_path = result_debug_dir / f"{stem}_{service_slug}_normalized.json"
            _save_json(normalized_path, result["normalized_json"])
            overlay_path = result_debug_dir / f"{stem}_{service_slug}_overlay.png"
            _save_component_overlay(image, result["normalized_json"], overlay_path)

        _print_result_block(result)

    _print_summary(results)

    if all(result["ok"] for result in results):
        print("[INFO] reading_check 완료: 모든 비교 모델이 응답을 반환했습니다.")
        return 0

    print("[WARNING] reading_check 완료: 일부 모델 호출이 실패했습니다.")
    return 5


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
