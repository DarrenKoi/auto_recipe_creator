"""현재 컴퓨터 스크린샷을 UI-Venus / PaddleOCR-VL 에 보내 읽기 응답을 확인한다.

동작:
  1) 전체 데스크톱을 캡처한다.
  2) 원본 확인용 JPEG와 실제 전송용 WebP를 `poc/work2/debug_images/`에 저장한다.
  3) 동일한 이미지를 UI-Venus 와 PaddleOCR-VL 에 각각 전송한다.
  4) 원문 응답과 파싱 결과를 debug_images 폴더에 저장하고 요약을 출력한다.

사용법:
  uv run python poc/work2/reading_check.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from poc.work.screen_capture import ScreenCapture
from poc.work.vlm_openai_client import ChatImageRequest, OpenAICompatibleVLMClient
from poc.work2.flask_vlm import apply_work2_pipeline_env_defaults, load_work2_env
from poc.work2.prompts import build_ocr_assist_prompt
from poc.work2.rcs_utils import extract_json


DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
SOURCE_JPEG_QUALITY = 85
WEBP_QUALITY = 90
MAX_VLM_BYTES = 1_000_000
REQUEST_TIMEOUT_SEC = 120.0


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


def _ensure_debug_dir() -> None:
    """디버그 디렉터리를 보장한다."""
    DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _capture_desktop_image() -> Image.Image | None:
    """전체 데스크톱을 캡처하여 PIL 이미지로 반환한다."""
    capture = ScreenCapture(output_dir=str(DEBUG_IMAGE_DIR))
    try:
        png_data = capture.capture_full_screen(save=False)
    except Exception as exc:
        print(f"[ERROR] 전체 화면 캡처 실패: {exc}")
        return None
    finally:
        try:
            capture.close()
        except Exception:
            pass

    if not png_data:
        print("[ERROR] 캡처 데이터가 비어 있습니다.")
        return None

    if not PIL_AVAILABLE:
        print("[ERROR] Pillow 패키지가 필요합니다. `uv sync --extra dev` 환경을 확인하세요.")
        return None

    image = Image.open(BytesIO(png_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _save_source_jpeg(image: Image.Image, stem: str) -> Path:
    """원본 확인용 JPEG를 저장한다."""
    out_path = DEBUG_IMAGE_DIR / f"{stem}_source.jpg"
    image.save(out_path, format="JPEG", quality=SOURCE_JPEG_QUALITY)
    print(f"[INFO] 원본 확인용 JPEG 저장: {out_path}")
    return out_path


def _encode_image_for_vlm(image: Image.Image, stem: str) -> tuple[str, int, int, Path]:
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

    sent_path = DEBUG_IMAGE_DIR / f"{stem}_sent.webp"
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
) -> dict[str, Any]:
    """한 모델에 reading check 요청을 보낸다."""
    client = OpenAICompatibleVLMClient(
        base_url=api_url,
        api_key=api_key,
        timeout_sec=REQUEST_TIMEOUT_SEC,
    )
    request = ChatImageRequest(
        model=model_name,
        system_message=system_message,
        user_text=user_text,
        image_b64=image_b64,
        image_mime=image_mime,
        temperature=temperature,
    )

    print(
        f"[INFO] reading check 호출: service={service_name}, "
        f"model={model_name}, endpoint={client.endpoint}"
    )
    start = time.time()
    try:
        raw_response = client.chat_with_image(request)
    except Exception as exc:
        return {
            "service": service_name,
            "model": model_name,
            "ok": False,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "error": str(exc),
            "raw_response": "",
            "parsed_json": None,
        }

    latency_ms = round((time.time() - start) * 1000, 1)
    parsed_json = None
    try:
        parsed_json = extract_json(raw_response)
    except Exception as exc:
        print(f"[WARNING] {service_name} JSON 파싱 실패: {exc}")

    return {
        "service": service_name,
        "model": model_name,
        "ok": True,
        "latency_ms": latency_ms,
        "error": None,
        "raw_response": raw_response,
        "parsed_json": parsed_json,
    }


def _print_result_block(result: dict[str, Any]) -> None:
    """모델별 결과 블록을 출력한다."""
    print("\n" + "=" * 80)
    print(f"[INFO] 결과 - {result['service']} ({result['model']})")
    print("=" * 80)
    if not result["ok"]:
        print(f"[ERROR] 호출 실패: {result['error']}")
        return
    print(f"[INFO] 응답 시간: {result['latency_ms']}ms")
    print(f"[INFO] 원문 응답:\n{result['raw_response']}\n")
    if result["parsed_json"] is not None:
        print("[INFO] 파싱된 JSON:")
        print(json.dumps(result["parsed_json"], ensure_ascii=False, indent=2))


def _print_summary(results: list[dict[str, Any]]) -> None:
    """요약 테이블을 출력한다."""
    print("\n" + "-" * 90)
    print("  reading_check 결과")
    print("-" * 90)
    print(f"  {'서비스':<22} {'상태':<10} {'JSON':<8} {'응답시간':<12} {'비고'}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 8} {'─' * 12} {'─' * 32}")

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        json_status = "O" if result["parsed_json"] is not None else "X"
        latency = f"{result['latency_ms']}ms"
        note = ""
        if result["error"]:
            note = result["error"][:32]
        elif result["service"] == "ui-venus" and isinstance(result["parsed_json"], dict):
            note = str(result["parsed_json"].get("screen_summary", ""))[:32]
        elif isinstance(result["parsed_json"], dict):
            texts = result["parsed_json"].get("texts") or []
            note = f"texts={len(texts)}"
        print(f"  {result['service']:<22} {status:<10} {json_status:<8} {latency:<12} {note}")

    print("-" * 90)


def main() -> int:
    """전체 reading check 흐름을 실행한다."""
    _ensure_debug_dir()
    if not PIL_AVAILABLE:
        print("[ERROR] Pillow 패키지가 필요합니다.")
        return 1

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

    print("[INFO] reading_check 시작")
    print(f"[INFO] primary: {primary_service} ({primary_model})")
    print(f"[INFO] ocr:     {ocr_service} ({ocr_model})")
    print(f"[INFO] debug dir: {DEBUG_IMAGE_DIR}")

    if not primary_api_url:
        print("[ERROR] primary API URL이 비어 있습니다. WORK2_FLASK_API_BASE_URL 또는 WORK2_VLM_API_URL을 확인하세요.")
        return 2
    if not ocr_api_url:
        print("[ERROR] OCR API URL이 비어 있습니다. WORK2_FLASK_API_BASE_URL 또는 WORK2_OCR_API_URL을 확인하세요.")
        return 3

    image = _capture_desktop_image()
    if image is None:
        return 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"reading_check_{timestamp}"
    _save_source_jpeg(image, stem)
    image_b64, width, height, sent_path = _encode_image_for_vlm(image, stem)
    print(f"[INFO] 실제 전송 이미지: {sent_path}")

    ui_system_message, ui_user_text = build_ui_venus_reading_prompt(width, height)
    ocr_system_message, ocr_user_text = build_ocr_assist_prompt(
        width=width,
        height=height,
        context_label="computer screenshot",
        focus_words=None,
        max_items=20,
    )

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
        ),
    ]

    for result in results:
        service_slug = _slugify(str(result["service"]))
        raw_path = DEBUG_IMAGE_DIR / f"{stem}_{service_slug}_response.txt"
        _save_text(raw_path, str(result["raw_response"] or result["error"] or ""))

        if isinstance(result["parsed_json"], dict):
            json_path = DEBUG_IMAGE_DIR / f"{stem}_{service_slug}_response.json"
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
