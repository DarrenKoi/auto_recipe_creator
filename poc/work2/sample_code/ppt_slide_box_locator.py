"""PPT 슬라이드 이미지 → 레이아웃 박스 추출.

슬라이드 내 텍스트 박스, 차트, 표, 이미지 등의 위치(bbox)를 추출한다.
원본 이미지 위에 overlay 박스를 그린 이미지와 좌표 JSON 을 저장한다.

사용법:
    pip install requests Pillow
    python ppt_slide_box_locator.py

    # 이 폴더에 .jpg 또는 .png 이미지를 넣어두면 자동으로 읽는다.

파이프라인:
    1단계: paddleocr-vl-1.5 로 OCR 텍스트 추출 (라벨 참고용)
    2단계: ui-venus-1.5-8b 로 레이아웃 박스 좌표 추출

필요 패키지: requests, Pillow (overlay 이미지용, 없으면 JSON 만 저장)
"""

import json
import sys
import time
from pathlib import Path

from vlm_common import (
    OCR_SERVICE,
    OCR_URL,
    VLM_MODEL,
    VLM_SERVICE,
    VLM_URL,
    build_combined_ocr,
    collect_images,
    parse_json_response,
    print_token_usage,
    vlm_chat,
)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
LAYOUT_COORD_SCALE = 1000
OVERLAY_IMAGE_SUFFIX = ".overlay.jpg"
LAYOUT_JSON_SUFFIX = ".layout.json"

BOX_COLORS = {
    "title": (21, 101, 192),
    "subtitle": (25, 118, 210),
    "text_box": (30, 136, 229),
    "chart": (239, 108, 0),
    "table": (46, 125, 50),
    "image": (123, 31, 162),
    "diagram": (123, 31, 162),
    "footer": (97, 97, 97),
    "other": (0, 121, 107),
}

# ──────────────────────────────────────────────
# VLM 프롬프트
# ──────────────────────────────────────────────
LAYOUT_SYSTEM_MESSAGE = """\
You are an expert presentation layout analyst.
Your task is to identify the visible bounding boxes of the major PPT objects in a slide screenshot.
Respond ONLY with a JSON object (no markdown fences, no explanation).

The OCR text is provided as a reference for short labels only.
Your coordinates must describe the visual object positions in the screenshot."""

LAYOUT_PROMPT_TEMPLATE = """\
Below is the OCR-extracted text from this slide (use it only as a reference for labels):
--- OCR TEXT ---
{ocr_text}
--- END OCR TEXT ---

Identify the main layout objects needed to reconstruct this slide later.
Focus especially on title/subtitle text boxes, body text boxes, charts, tables, and major images/diagrams.

Return JSON in exactly this format:
{{
  "elements": [
    {{
      "type": "title|subtitle|text_box|chart|table|image|diagram|footer|other",
      "label": "short label or first few words",
      "bbox_1000": {{
        "left": 0,
        "top": 0,
        "right": 1000,
        "bottom": 1000
      }}
    }}
  ]
}}

Rules:
- Coordinates are normalized to a 1000 x 1000 slide space.
- Return one box per logical PPT object, not one box per text line.
- For charts, include the full chart area, including axis labels and legend.
- Exclude the slide background and tiny decorative shapes that are not meaningful content.
- Order elements top-to-bottom, then left-to-right."""


# ──────────────────────────────────────────────
# bbox 정규화 헬퍼
# ──────────────────────────────────────────────
def _coerce_float(value) -> float | None:
    """문자열/숫자를 float 로 변환한다."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_bbox_1000(raw_bbox) -> dict | None:
    """레이아웃 박스를 정규화한다."""
    if isinstance(raw_bbox, dict):
        if {"left", "top", "right", "bottom"} <= raw_bbox.keys():
            left = _coerce_float(raw_bbox.get("left"))
            top = _coerce_float(raw_bbox.get("top"))
            right = _coerce_float(raw_bbox.get("right"))
            bottom = _coerce_float(raw_bbox.get("bottom"))
        elif {"x", "y", "width", "height"} <= raw_bbox.keys():
            left = _coerce_float(raw_bbox.get("x"))
            top = _coerce_float(raw_bbox.get("y"))
            w = _coerce_float(raw_bbox.get("width"))
            h = _coerce_float(raw_bbox.get("height"))
            if None in {left, top, w, h}:
                return None
            right = left + w
            bottom = top + h
        elif {"x", "y", "w", "h"} <= raw_bbox.keys():
            left = _coerce_float(raw_bbox.get("x"))
            top = _coerce_float(raw_bbox.get("y"))
            w = _coerce_float(raw_bbox.get("w"))
            h = _coerce_float(raw_bbox.get("h"))
            if None in {left, top, w, h}:
                return None
            right = left + w
            bottom = top + h
        else:
            return None
    elif isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        left = _coerce_float(raw_bbox[0])
        top = _coerce_float(raw_bbox[1])
        right = _coerce_float(raw_bbox[2])
        bottom = _coerce_float(raw_bbox[3])
    else:
        return None

    if None in {left, top, right, bottom}:
        return None

    left = int(round(max(0.0, min(LAYOUT_COORD_SCALE, left))))
    top = int(round(max(0.0, min(LAYOUT_COORD_SCALE, top))))
    right = int(round(max(0.0, min(LAYOUT_COORD_SCALE, right))))
    bottom = int(round(max(0.0, min(LAYOUT_COORD_SCALE, bottom))))

    if right <= left or bottom <= top:
        return None

    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _normalize_element_type(raw_type: str) -> str:
    """모델이 반환한 요소 타입을 통일한다."""
    normalized = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "body": "text_box", "body_text": "text_box", "bullet": "text_box",
        "paragraph": "text_box", "text": "text_box", "textbox": "text_box",
        "text_area": "text_box", "textblock": "text_box",
        "picture": "image", "photo": "image", "logo": "image", "screenshot": "image",
    }
    normalized = alias_map.get(normalized, normalized)
    return normalized if normalized in BOX_COLORS else "other"


def _normalize_layout_elements(raw_elements) -> list[dict]:
    """레이아웃 박스 목록을 정리한다."""
    if not isinstance(raw_elements, list):
        return []

    elements = []
    for item in raw_elements:
        if not isinstance(item, dict):
            continue
        bbox = _extract_bbox_1000(
            item.get("bbox_1000") or item.get("bbox") or item.get("box")
        )
        if bbox is None:
            continue

        element_type = _normalize_element_type(str(item.get("type") or "other"))
        label = str(
            item.get("label") or item.get("text") or item.get("title")
            or item.get("description") or ""
        ).strip()

        elements.append({"type": element_type, "label": label, "bbox_1000": bbox})

    elements.sort(key=lambda e: (e["bbox_1000"]["top"], e["bbox_1000"]["left"]))
    for idx, elem in enumerate(elements, 1):
        elem["index"] = idx
    return elements


def _bbox_1000_to_pixels(bbox_1000: dict, image_size: tuple[int, int]) -> dict:
    """1000 기준 좌표를 실제 이미지 픽셀 좌표로 변환한다."""
    w, h = image_size
    left = int(round((bbox_1000["left"] / LAYOUT_COORD_SCALE) * w))
    top = int(round((bbox_1000["top"] / LAYOUT_COORD_SCALE) * h))
    right = int(round((bbox_1000["right"] / LAYOUT_COORD_SCALE) * w))
    bottom = int(round((bbox_1000["bottom"] / LAYOUT_COORD_SCALE) * h))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _enrich_with_pixels(elements: list[dict], image_size: tuple[int, int] | None) -> list[dict]:
    """레이아웃 요소에 픽셀 좌표를 추가한다."""
    if image_size is None:
        return elements
    enriched = []
    for elem in elements:
        e = dict(elem)
        e["bbox_pixels"] = _bbox_1000_to_pixels(elem["bbox_1000"], image_size)
        enriched.append(e)
    return enriched


# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────
def analyze_layout(image_path: Path) -> dict | None:
    """단일 슬라이드 이미지에서 레이아웃 박스를 추출한다."""
    print(f"\n{'='*60}")
    print(f"[INFO] 레이아웃 분석 시작: {image_path.name}")
    print(f"{'='*60}")

    if not image_path.exists():
        print(f"[WARNING] 파일이 존재하지 않습니다: {image_path}")
        return None

    file_size_kb = image_path.stat().st_size / 1024
    print(f"[INFO] 파일 크기: {file_size_kb:.1f} KB")

    # ── OCR (라벨 참고용) ──
    combined_ocr = build_combined_ocr(image_path)

    # ── VLM 레이아웃 분석 ──
    print(f"[INFO] [VLM] 레이아웃 박스 분석 중... ({VLM_SERVICE})")
    user_prompt = LAYOUT_PROMPT_TEMPLATE.format(ocr_text=combined_ocr)

    start = time.time()
    try:
        raw_text, usage = vlm_chat(
            VLM_URL, VLM_MODEL, image_path, LAYOUT_SYSTEM_MESSAGE, user_prompt, temperature=0.0,
        )
    except Exception as exc:
        print(f"[ERROR] VLM 호출 실패: {exc}")
        return None

    elapsed = time.time() - start
    print(f"[INFO] 레이아웃 응답 시간: {elapsed:.1f}초")
    print_token_usage(usage, prefix="레이아웃 ")

    parsed = parse_json_response(raw_text, "레이아웃")
    if not isinstance(parsed, dict):
        return None

    elements = _normalize_layout_elements(parsed.get("elements"))
    print(f"[INFO] 레이아웃 박스 추출: {len(elements)}개")

    # 픽셀 좌표 추가
    image_size = None
    if PIL_AVAILABLE:
        with Image.open(image_path) as img:
            image_size = img.size

    result = {"_ocr_text": combined_ocr, "_layout_elements": _enrich_with_pixels(elements, image_size)}
    if image_size:
        result["_image_size"] = {"width": image_size[0], "height": image_size[1]}
    return result


# ──────────────────────────────────────────────
# 출력 / 저장
# ──────────────────────────────────────────────
def print_layout(image_name: str, result: dict) -> None:
    """레이아웃 결과를 출력한다."""
    elements = result.get("_layout_elements") or []

    print(f"\n{'─'*60}")
    print(f"  레이아웃 결과: {image_name}")
    print(f"{'─'*60}")

    if not elements:
        print("[WARNING] 추출된 레이아웃 박스가 없습니다.")
        return

    print(f"[INFO] 레이아웃 박스: {len(elements)}개")
    for item in elements:
        bbox = item.get("bbox_pixels") or item.get("bbox_1000") or {}
        label = item.get("label") or "-"
        print(
            f"  [{item.get('index', '?')}] {item.get('type', 'other'):<10} "
            f"({bbox.get('left', '?')}, {bbox.get('top', '?')}) - "
            f"({bbox.get('right', '?')}, {bbox.get('bottom', '?')})  "
            f"label={label}"
        )
    print()


def save_layout_json(image_path: Path, result: dict) -> Path | None:
    """레이아웃 박스를 JSON 파일로 저장한다."""
    elements = result.get("_layout_elements") or []
    if not elements:
        return None

    layout_path = image_path.with_suffix(LAYOUT_JSON_SUFFIX)
    payload = {
        "source_image": image_path.name,
        "image_size": result.get("_image_size"),
        "elements": elements,
    }
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 레이아웃 저장: {layout_path.name}")
    return layout_path


def save_overlay_image(image_path: Path, result: dict) -> Path | None:
    """원본 이미지 위에 레이아웃 박스를 그린 overlay 이미지를 저장한다."""
    elements = result.get("_layout_elements") or []
    if not elements:
        return None
    if not PIL_AVAILABLE:
        print("[WARNING] Pillow 미설치로 overlay 이미지 저장을 건너뜁니다.")
        return None

    overlay_path = image_path.with_suffix(OVERLAY_IMAGE_SUFFIX)

    with Image.open(image_path) as img:
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        font = ImageFont.load_default()

        line_width = max(2, min(base.size) // 250)
        pad = max(4, line_width + 1)

        for item in elements:
            bbox = item.get("bbox_pixels")
            if not bbox:
                bbox = _bbox_1000_to_pixels(item["bbox_1000"], base.size)

            color = BOX_COLORS.get(item.get("type", "other"), BOX_COLORS["other"])

            draw.rectangle(
                (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
                outline=(*color, 255),
                width=line_width,
                fill=(*color, 36),
            )

            label_text = f"{item.get('index', '?')}. {item.get('type', 'other')}"
            tb = draw.textbbox((0, 0), label_text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]

            lw, lh = tw + pad * 2, th + pad * 2
            ll = min(bbox["left"], max(0, base.width - lw))
            lt = bbox["top"] - lh - pad
            if lt < 0:
                lt = min(max(0, bbox["top"] + pad), max(0, base.height - lh))

            draw.rectangle((ll, lt, ll + lw, lt + lh), fill=(*color, 220))
            draw.text((ll + pad, lt + pad), label_text, fill=(255, 255, 255, 255), font=font)

        merged = Image.alpha_composite(base, overlay).convert("RGB")
        merged.save(overlay_path, quality=95)

    print(f"[INFO] Overlay 저장: {overlay_path.name}")
    return overlay_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main() -> None:
    """메인 실행."""
    image_paths = collect_images()

    print("[INFO] PPT 슬라이드 박스 로케이터 시작")
    print(f"[INFO] OCR: {OCR_SERVICE} ({OCR_URL})")
    print(f"[INFO] VLM: {VLM_SERVICE} ({VLM_URL})")
    print(f"[INFO] 대상 이미지: {len(image_paths)}개")

    if not image_paths:
        print("[ERROR] 분석할 이미지가 없습니다.")
        print("[INFO] 이 폴더에 .jpg 또는 .png 이미지를 넣어주세요.")
        sys.exit(1)

    success = 0
    for image_path in image_paths:
        result = analyze_layout(image_path)
        if result is None:
            continue
        print_layout(image_path.name, result)
        save_layout_json(image_path, result)
        save_overlay_image(image_path, result)
        success += 1

    print(f"\n{'='*60}")
    print(f"[INFO] 레이아웃 분석 완료: {success}/{len(image_paths)}개 성공")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
