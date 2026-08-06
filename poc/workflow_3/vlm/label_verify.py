"""VLM 이 고른 좌표 주변의 텍스트를 읽어 '거기가 맞는지' 확인하는 공용 도구.

원칙은 프로젝트 규칙 그대로다: **좌표는 VLM 이 정하고, OCR 은 확인만 한다.** 이 모듈은
좌표를 만들지 않는다 - 이미 정해진 점 주위를 좁게 잘라 읽어줄 뿐이다.

PaddleOCR-VL 은 문서 파서라 layout -> crop -> recognize 가 정상 경로이고, 전체 UI
스크린샷을 통째로 주면 환각이 잦다. 그래서 확인은 항상 **좁은 crop** 으로 한다.

쓰는 곳:
  - `rcs/tool_row_verify.py`  : list 행의 장비 ID 확인 (가로로 긴 strip)
  - `rcs/bench_tool_window_reader.py` : 버튼 라벨 확인 (버튼 크기의 작은 box)
"""

import time
from dataclasses import dataclass, field

from poc.workflow_3.debug_artifacts import (
    debug_image_path,
    save_debug_text,
    save_debug_webp,
)
from poc.workflow_3.util import crop_image
from poc.workflow_3.vlm.prompts import build_ocr_assist_prompt
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

OCR_SERVICE_SLUG = "paddleocr-vl-1.5"
OCR_MAX_TOKENS = 512


@dataclass
class PointTextRead:
    """한 점 주변에서 읽어낸 텍스트."""

    ok: bool
    raw_text: str = ""
    tokens: list[str] = field(default_factory=list)
    box: dict | None = None
    upscale: float = 1.0
    crop_image_path: str = ""
    response_path: str = ""
    error: str = ""
    elapsed_sec: float = 0.0


def crop_box_around_point(
    point: dict,
    image_width: int,
    image_height: int,
    *,
    left_ratio: float,
    right_ratio: float,
    half_height_ratio: float,
    min_left_px: int = 10,
    min_right_px: int = 10,
    min_half_height_px: int = 6,
) -> dict:
    """점을 감싸는 crop box 를 이미지 크기 대비 비율로 만든다 (경계 clamp 포함)."""
    half_h = max(min_half_height_px, int(image_height * half_height_ratio))
    left_pad = max(min_left_px, int(image_width * left_ratio))
    right_pad = max(min_right_px, int(image_width * right_ratio))
    return {
        "left": max(0, int(point["x"]) - left_pad),
        "top": max(0, int(point["y"]) - half_h),
        "right": min(image_width, int(point["x"]) + right_pad),
        "bottom": min(image_height, int(point["y"]) + half_h),
    }


def upscale_for_ocr(image, min_height_px: int, max_upscale: float):
    """작은 crop 을 OCR 가독 최소 높이까지 확대한다. (image, scale) 반환."""
    width, height = image.size
    if height <= 0 or height >= min_height_px:
        return image, 1.0
    scale = min(max_upscale, min_height_px / float(height))
    if scale <= 1.0:
        return image, 1.0
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return resized, scale


def tokens_from_text(raw_text: str) -> list[str]:
    """OCR 평문에서 공백/개행 기준 토큰을 뽑는다."""
    tokens: list[str] = []
    for line in (raw_text or "").splitlines():
        for token in line.split():
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _normalize_label(text: str) -> str:
    """라벨 비교용 정규화 - 영숫자만 남기고 소문자."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def label_matches(tokens: list[str], expected_label: str) -> bool:
    """읽은 토큰 중 기대 라벨과 맞는 것이 있는지.

    버튼 라벨은 짧아서(Stop/PM/Queue) OCR 이 주변 글자를 붙여 오는 경우가 있다.
    그래서 정확 일치뿐 아니라 '토큰이 라벨을 포함' 도 인정한다. 다만 라벨이 2자 이하
    (PM 등)면 포함 매칭이 너무 헐거워지므로(예: 'PMX', 'RPM') 정확 일치만 본다.
    """
    expected = _normalize_label(expected_label)
    if not expected:
        return False

    normalized = [_normalize_label(token) for token in tokens]
    if expected in normalized:
        return True
    if len(expected) <= 2:
        return False
    return any(expected in token for token in normalized if token)


def read_text_near_point(
    image,
    box: dict,
    *,
    debug_image_dir,
    timestamp_tag: str,
    artifact_label: str,
    log_name: str = "label_verify",
    min_height_px: int = 72,
    max_upscale: float = 6.0,
    client: Workflow1VLMClient | None = None,
) -> PointTextRead:
    """지정 box 를 잘라 OCR 로 읽는다 (좌표는 만들지 않는다)."""
    started_at = time.time()
    crop = crop_image(image, box)
    crop, upscale = upscale_for_ocr(crop, min_height_px, max_upscale)

    crop_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_crop.webp",
        model_name=OCR_SERVICE_SLUG,
        timestamp_tag=timestamp_tag,
    )
    save_debug_webp(crop, crop_path, quality=90)

    ocr_client = client
    if ocr_client is None:
        try:
            ocr_client = Workflow1VLMClient(
                service_slug=OCR_SERVICE_SLUG,
                timeout_sec=30.0,
                log_name=log_name,
            )
        except Exception as exc:
            print(f"[WARNING] 확인 OCR client 생성 실패: {exc}")
            return PointTextRead(
                ok=False,
                box=box,
                upscale=upscale,
                crop_image_path=str(crop_path),
                error=f"{type(exc).__name__}: {exc}",
                elapsed_sec=time.time() - started_at,
            )

    system_message, user_text = build_ocr_assist_prompt(*crop.size)
    try:
        response = ocr_client.chat_with_image_path(
            image_path=crop_path,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[WARNING] 확인 OCR 호출 실패: {exc}")
        return PointTextRead(
            ok=False,
            box=box,
            upscale=upscale,
            crop_image_path=str(crop_path),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_sec=time.time() - started_at,
        )

    raw_text = (response.text or "").strip()
    response_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_ocr.txt",
        model_name=OCR_SERVICE_SLUG,
        timestamp_tag=timestamp_tag,
    )
    save_debug_text(response_path, raw_text)

    return PointTextRead(
        ok=True,
        raw_text=raw_text,
        tokens=tokens_from_text(raw_text),
        box=box,
        upscale=upscale,
        crop_image_path=str(crop_path),
        response_path=str(response_path),
        elapsed_sec=time.time() - started_at,
    )


__all__ = [
    "PointTextRead",
    "crop_box_around_point",
    "label_matches",
    "read_text_near_point",
    "tokens_from_text",
    "upscale_for_ocr",
]
