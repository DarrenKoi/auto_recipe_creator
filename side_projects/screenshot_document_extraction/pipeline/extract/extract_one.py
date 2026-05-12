"""페이지 한 장에 대해 paddleocr-vl-1.5 + ui-venus 호출을 수행한다."""

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from poc.work2.prompts.prompt_ocr_assist import build_ocr_assist_prompt
from poc.work2.util.image_utils import encode_image_webp
from poc.work2.util.json_utils import extract_json
from poc.work2.vlm_client import Work2VLMClient

from pipeline.extract.prompts import build_doc_region_prompt
from pipeline.settings import (
    PADDLEOCR_SERVICE_SLUG,
    PIPELINE_LOG_NAME,
    UI_VENUS_SERVICE_SLUG,
    VLM_TIMEOUT_SEC,
    WEBP_QUALITY,
)


@dataclass(frozen=True)
class ExtractResult:
    """페이지 추출 결과."""

    page_index: int
    raw_path: Path
    paddleocr_path: Path
    uivenus_path: Path


def _now_iso() -> str:
    """타임존 없는 ISO-8601."""
    return datetime.now().replace(microsecond=0).isoformat()


def _read_jpeg(jpeg_path: Path) -> tuple[bytes, int, int]:
    """JPEG 를 읽어 WebP 로 재인코딩한 (bytes, width, height) 를 반환한다."""
    image = Image.open(jpeg_path)
    b64, width, height = encode_image_webp(image, quality=WEBP_QUALITY)
    webp_bytes = base64.b64decode(b64)
    return webp_bytes, width, height


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_uivenus(text: str) -> dict:
    """ui-venus 응답을 JSON 으로 파싱한다. 실패 시 raw 텍스트만 보존."""
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        print("[WARNING] ui-venus 응답에서 JSON 추출 실패, raw 텍스트만 보존한다.")
        return {"raw_text": text, "parse_error": True}


def process(jpeg_path: Path, *, doc_id: str, page_index: int, out_dir: Path) -> ExtractResult:
    """페이지 한 장을 추출한다.

    Returns: ExtractResult — 저장된 결과 파일 경로들.
    """
    if not jpeg_path.exists():
        raise FileNotFoundError(f"page JPEG 가 없다: {jpeg_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    paddleocr_path = out_dir / f"page_{page_index:03d}.paddleocr.json"
    uivenus_path = out_dir / f"page_{page_index:03d}.uivenus.json"
    raw_path = out_dir / f"page_{page_index:03d}.raw.json"

    image_bytes, width, height = _read_jpeg(jpeg_path)

    # 1) PaddleOCR-VL: OCR 키워드만 보낸다.
    paddle_client = Work2VLMClient(
        service_slug=PADDLEOCR_SERVICE_SLUG,
        timeout_sec=VLM_TIMEOUT_SEC,
        log_name=PIPELINE_LOG_NAME,
    )
    paddle_system, paddle_user = build_ocr_assist_prompt(width, height)
    paddle_response = paddle_client.chat_with_image_bytes(
        image_bytes=image_bytes,
        system_message=paddle_system,
        user_text=paddle_user,
        image_mime="image/webp",
    )
    paddle_payload = {
        "service": paddle_response.service_slug,
        "model": paddle_response.model_name,
        "text": paddle_response.text,
        "token_usage": paddle_response.token_usage,
        "called_at": _now_iso(),
    }
    _write_json(paddleocr_path, paddle_payload)

    # 2) UI-Venus: 문서 영역 검출.
    uivenus_client = Work2VLMClient(
        service_slug=UI_VENUS_SERVICE_SLUG,
        timeout_sec=VLM_TIMEOUT_SEC,
        log_name=PIPELINE_LOG_NAME,
    )
    uivenus_system, uivenus_user = build_doc_region_prompt(width, height)
    uivenus_response = uivenus_client.chat_with_image_bytes(
        image_bytes=image_bytes,
        system_message=uivenus_system,
        user_text=uivenus_user,
        image_mime="image/webp",
    )
    uivenus_parsed = _parse_uivenus(uivenus_response.text)
    uivenus_payload = {
        "service": uivenus_response.service_slug,
        "model": uivenus_response.model_name,
        "raw_text": uivenus_response.text,
        "parsed": uivenus_parsed,
        "token_usage": uivenus_response.token_usage,
        "called_at": _now_iso(),
    }
    _write_json(uivenus_path, uivenus_payload)

    # 3) 머지본.
    raw_payload = {
        "doc_id": doc_id,
        "page_index": page_index,
        "width": width,
        "height": height,
        "source_image": str(jpeg_path),
        "paddleocr": paddle_payload,
        "uivenus": uivenus_payload,
        "merged_at": _now_iso(),
    }
    _write_json(raw_path, raw_payload)

    return ExtractResult(
        page_index=page_index,
        raw_path=raw_path,
        paddleocr_path=paddleocr_path,
        uivenus_path=uivenus_path,
    )
