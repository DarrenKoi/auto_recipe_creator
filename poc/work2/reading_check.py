"""현재 컴퓨터의 단일 모니터 스크린샷을 UI-Venus / PaddleOCR-VL 에 보내 읽기 응답을 확인한다.

동작:
  1) 지정한 모니터 1개만 캡처한다.
  2) 원본 확인용 JPEG와 실제 전송용 WebP를 모델 기반 하위 폴더에 저장한다.
  3) 동일한 이미지를 UI-Venus 와 PaddleOCR-VL 에 각각 전송한다.
  4) 원문 응답과 파싱 결과를 각 모델 폴더에 저장하고 요약을 출력한다.

사용법:
  uv run python poc/work2/reading_check.py

환경변수:
  READING_CHECK_MONITOR_INDEX=1  # 기본값: 1 (첫 번째 물리 모니터)
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

import requests
from PIL import Image

from poc.work.screen_capture import ScreenCapture
from poc.work.vlm_openai_client import ChatImageRequest, OpenAICompatibleVLMClient
from flask_api.vlm_serve.config import get_service_by_slug
from poc.work2 import debug_image_dir
from poc.work2.flask_vlm import apply_work2_pipeline_env_defaults, load_work2_env
from poc.work2.logger import log_vlm_call
from poc.work2.rcs_utils import extract_json


DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
SOURCE_JPEG_QUALITY = 85
WEBP_QUALITY = 90
MAX_VLM_BYTES = 1_000_000
REQUEST_TIMEOUT_SEC = 120.0
DEFAULT_MONITOR_INDEX = 1


def build_ui_venus_reading_prompt(width: int, height: int) -> tuple[str, str]:
    """UI-Venus 용 일반 화면 읽기 프롬프트를 구성한다."""
    system_message = (
        "You are a desktop GUI screenshot reader. "
        f"The image is {width}x{height} pixels. "
        "Read only visible content. "
        "Use exact visible text when possible. "
        "If something is too small or unclear, say so instead of inventing it. "
        "Respond ONLY with valid JSON."
    )

    user_text = "\n".join(
        [
            "Read this computer screenshot and summarize what is visible.",
            "Focus on the main app/window or desktop context, major UI regions, and important visible texts.",
            "Include menus, tabs, buttons, labels, dialog titles, and status text when they are readable.",
            "Do not infer hidden or cropped content.",
            "",
            "Return ONLY this JSON:",
            "{",
            '  "screen_summary": "...",',
            '  "visible_context": "...",',
            '  "key_texts": ["..."],',
            '  "readable": true,',
            '  "uncertain_areas": ["..."]',
            "}",
        ]
    )
    return system_message, user_text


def build_paddleocr_reading_prompt(width: int, height: int) -> tuple[str, str]:
    """PaddleOCR-VL 용 태스크 키워드 프롬프트를 구성한다.

    PaddleOCR-VL-1.5 는 0.9B 파라미터 모델로 6가지 태스크 키워드에 대해 학습되었다:
    OCR:, Table Recognition:, Formula Recognition:,
    Chart Recognition:, Spotting:, Seal Recognition:
    시스템 메시지나 복잡한 지시 없이 태스크 키워드만 전송해야 한다.
    """
    return "", "OCR:"


def _slugify(value: str) -> str:
    """파일명용 안전 문자열로 변환한다."""
    allowed = []
    for char in value.strip().lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def _ensure_debug_dir(debug_dir: Path) -> None:
    """지정한 디버그 디렉터리를 보장한다."""
    debug_dir.mkdir(parents=True, exist_ok=True)


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


def _capture_monitor_image(monitor_index: int) -> Image.Image | None:
    """단일 물리 모니터를 캡처하여 PIL 이미지로 반환한다."""
    capture = ScreenCapture(output_dir=str(DEBUG_IMAGE_DIR))
    monitors = []
    try:
        monitors = list(getattr(capture.sct, "monitors", []) or []) if capture.sct else []
        physical_count = max(0, len(monitors) - 1)
        if physical_count <= 0:
            print("[ERROR] 현재 환경에서는 물리 모니터가 감지되지 않았습니다. monitor 0(전체 데스크톱)만 노출됩니다.")
            return None

        if monitor_index > physical_count:
            available = ", ".join(str(idx) for idx in range(1, physical_count + 1))
            print(
                f"[ERROR] 요청한 monitor index {monitor_index} 는 범위를 벗어났습니다. "
                f"사용 가능: {available}"
            )
            return None

        png_data = capture.capture_monitor(monitor_index=monitor_index, save=False)
    except Exception as exc:
        print(f"[ERROR] 모니터 {monitor_index} 캡처 실패: {exc}")
        return None
    finally:
        try:
            capture.close()
        except Exception:
            pass

    if not png_data:
        print(f"[ERROR] 모니터 {monitor_index} 캡처 데이터가 비어 있습니다.")
        return None

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


def _call_model(
    *,
    service_name: str,
    model_name: str,
    api_url: str,
    api_key: str,
    system_message: str,
    user_text: str,
    image_b64: str,
    image_mime: str = "image/webp",
    temperature: float = 0.0,
    expect_json: bool = True,
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
            "expect_json": expect_json,
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
            "expect_json": expect_json,
        }

    latency_ms = round((time.time() - start) * 1000, 1)
    log_vlm_call(
        service=service_name,
        model=request_model,
        status="ok",
        latency_ms=latency_ms,
        token_usage=client.last_token_usage,
        endpoint=client.endpoint,
    )
    parsed_json = None
    if expect_json:
        try:
            parsed_json = extract_json(raw_response)
        except Exception as exc:
            print(f"[WARNING] {service_name} JSON 파싱 실패: {exc}")

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
        "expect_json": expect_json,
    }


def _print_result_block(result: dict[str, Any]) -> None:
    """모델별 결과 블록을 출력한다."""
    print("\n" + "=" * 80)
    print(f"[INFO] 결과 - {result['service']} ({result['model']})")
    print("=" * 80)
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
    if result.get("expect_json") and result["parsed_json"] is not None:
        print("[INFO] 파싱된 JSON:")
        print(json.dumps(result["parsed_json"], ensure_ascii=False, indent=2))


def _print_summary(results: list[dict[str, Any]]) -> None:
    """요약 테이블을 출력한다."""
    print("\n" + "-" * 90)
    print("  reading_check 결과")
    print("-" * 90)
    print(f"  {'서비스':<22} {'상태':<10} {'형식':<8} {'응답시간':<12} {'비고'}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 8} {'─' * 12} {'─' * 32}")

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        response_format = "-"
        if result["ok"] and result["parsed_json"] is not None:
            response_format = "json"
        elif result["ok"] and not result.get("expect_json"):
            response_format = "free"
        elif result["ok"]:
            response_format = "raw"
        latency = f"{result['latency_ms']}ms"
        note = ""
        if result["error"]:
            note = result["error"][:32]
        elif isinstance(result["parsed_json"], dict):
            if result["parsed_json"].get("screen_summary"):
                note = str(result["parsed_json"]["screen_summary"])[:32]
            else:
                note = "json ok"
        elif not result.get("expect_json") and result.get("raw_response"):
            note = " ".join(str(result["raw_response"]).split())[:32]
        else:
            note = "non-json response"[:32]
        print(f"  {result['service']:<22} {status:<10} {response_format:<8} {latency:<12} {note}")

    print("-" * 90)


def main() -> int:
    """전체 reading check 흐름을 실행한다."""
    load_work2_env()
    pipeline = apply_work2_pipeline_env_defaults()

    primary_service = str(pipeline.get("primary_service") or "ui-venus")
    primary_model = str(pipeline.get("primary_model_name") or "ui-venus-1.5-8b")
    primary_api_url = str(pipeline.get("primary_api_url") or "").strip()
    primary_api_key = str(pipeline.get("primary_api_key") or "").strip()

    ocr_service = str(pipeline.get("ocr_service") or "paddleocr-vl-1.5")
    ocr_model = str(pipeline.get("ocr_model_name") or "paddleocr-vl-1.5")
    ocr_api_url = str(pipeline.get("ocr_api_url") or "").strip()
    ocr_api_key = str(pipeline.get("ocr_api_key") or "").strip()
    monitor_index = _resolve_monitor_index()
    shared_debug_dir = debug_image_dir(
        DEBUG_IMAGE_DIR,
        f"{primary_model}__{ocr_model}",
    )
    _ensure_debug_dir(shared_debug_dir)

    print("[INFO] reading_check 시작")
    print(f"[INFO] primary configured: {primary_service} ({primary_model})")
    print(f"[INFO] ocr configured:     {ocr_service} ({ocr_model})")
    print(f"[INFO] capture monitor:    {monitor_index}")
    print(f"[INFO] shared debug dir:   {shared_debug_dir}")

    if not primary_api_url:
        print("[ERROR] primary API URL이 비어 있습니다. WORK2_FLASK_API_BASE_URL 또는 WORK2_VLM_API_URL을 확인하세요.")
        return 2
    if not ocr_api_url:
        print("[ERROR] OCR API URL이 비어 있습니다. WORK2_FLASK_API_BASE_URL 또는 WORK2_OCR_API_URL을 확인하세요.")
        return 3

    image = _capture_monitor_image(monitor_index)
    if image is None:
        return 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"reading_check_{timestamp}"
    _save_source_jpeg(image, stem, shared_debug_dir)
    image_b64, width, height, sent_path = _encode_image_for_vlm(image, stem, shared_debug_dir)
    print(f"[INFO] 실제 전송 이미지: {sent_path}")

    ui_system_message, ui_user_text = build_ui_venus_reading_prompt(width, height)
    ocr_system_message, ocr_user_text = build_paddleocr_reading_prompt(width, height)

    results = [
        _call_model(
            service_name=primary_service,
            model_name=primary_model,
            api_url=primary_api_url,
            api_key=primary_api_key,
            system_message=ui_system_message,
            user_text=ui_user_text,
            image_b64=image_b64,
        ),
        _call_model(
            service_name=ocr_service,
            model_name=ocr_model,
            api_url=ocr_api_url,
            api_key=ocr_api_key,
            system_message=ocr_system_message,
            user_text=ocr_user_text,
            image_b64=image_b64,
            expect_json=False,
        ),
    ]

    for result in results:
        result_debug_dir = debug_image_dir(
            DEBUG_IMAGE_DIR,
            str(result.get("model") or result.get("configured_model") or result["service"]),
        )
        service_slug = _slugify(str(result["service"]))
        probe_path = result_debug_dir / f"{stem}_{service_slug}_model_probe.json"
        _save_json(
            probe_path,
            {
                "service": result["service"],
                "configured_model": result.get("configured_model", ""),
                "request_model": result.get("model", ""),
                "advertised_models": result.get("advertised_models", []),
                "ok": result["ok"],
                "error": result["error"],
                "expect_json": result.get("expect_json", True),
            },
        )

        raw_path = result_debug_dir / f"{stem}_{service_slug}_response.txt"
        _save_text(raw_path, str(result["raw_response"] or result["error"] or ""))

        if isinstance(result["parsed_json"], dict):
            json_path = result_debug_dir / f"{stem}_{service_slug}_response.json"
            _save_json(json_path, result["parsed_json"])

        _print_result_block(result)

    _print_summary(results)

    if all(result["ok"] for result in results):
        print("[INFO] reading_check 완료: 두 모델 모두 응답을 반환했습니다.")
        return 0

    print("[WARNING] reading_check 완료: 일부 모델 호출이 실패했습니다.")
    return 5


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
