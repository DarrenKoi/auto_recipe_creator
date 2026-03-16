"""PPT 슬라이드 이미지 → 구조화된 내용/레이아웃 추출 샘플 코드.

이 폴더는 독립 실행 가능 — 프로젝트 내 다른 모듈에 의존하지 않는다.
동료에게 이 폴더만 복사해서 공유하면 바로 사용 가능.

사용법:
    # 이 폴더에서 바로 실행
    pip install requests
    python ppt_slide_reader.py

    # 또는 프로젝트 루트에서
    uv run python poc/work2/sample_code/ppt_slide_reader.py

    # 이 폴더에 .jpg 또는 .png 이미지를 넣어두면 자동으로 읽는다.

파이프라인 (2단계):
    1단계: paddleocr-vl-1.5 로 이미지에서 텍스트를 정확히 추출 (OCR)
    2단계: ui-venus-1.5-8b 가 OCR 텍스트를 참조하여 슬라이드 구조를 분석 (VLM)

    OCR 결과를 VLM 에게 참고 자료로 제공하므로 텍스트 오탈자가 크게 줄어든다.

필요 패키지: requests, Pillow(overlay 이미지 저장용)
"""

import base64
import json
import sys
import time
from pathlib import Path

import requests

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

# ──────────────────────────────────────────────
# 설정 — 필요 시 여기만 수정
# ──────────────────────────────────────────────
FLASK_API_BASE = "http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api"

VLM_SERVICE = "ui-venus"
VLM_MODEL = "ui-venus-1.5-8b"
VLM_URL = f"{FLASK_API_BASE}/vlm_serve/{VLM_SERVICE}"

OCR_SERVICE = "paddleocr-vl-1.5"
OCR_MODEL = "paddleocr-vl-1.5"
OCR_URL = f"{FLASK_API_BASE}/vlm_serve/{OCR_SERVICE}"

TIMEOUT_SEC = 120.0
SAMPLE_DIR = Path(__file__).parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
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
# 최소 VLM 클라이언트 (독립 실행용)
# ──────────────────────────────────────────────
def _detect_mime(image_bytes: bytes) -> str:
    """이미지 header 로 MIME type 을 추정한다."""
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def vlm_chat(
    base_url: str,
    model: str,
    image_path: Path,
    system_message: str,
    user_text: str,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """OpenAI-compatible VLM endpoint 에 이미지 + 텍스트를 보내고 응답을 반환한다.

    Returns:
        (response_text, token_usage_dict)
    """
    image_bytes = image_path.read_bytes()
    mime = _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ],
    })

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        if endpoint.endswith("/v1"):
            endpoint += "/chat/completions"
        else:
            endpoint += "/v1/chat/completions"

    resp = requests.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"VLM 응답에 choices 없음: {json.dumps(data, ensure_ascii=False)[:300]}")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        ).strip()

    return str(content), data.get("usage") or {}


# ──────────────────────────────────────────────
# VLM 프롬프트
# ──────────────────────────────────────────────
SYSTEM_MESSAGE = """\
You are an expert document analyst specializing in technical presentation slides.
Your task is to extract and structure ALL visible content from a PPT slide screenshot.
Be thorough — capture every piece of text, number, label, and visual element you can see.
Respond ONLY with a JSON object (no markdown fences, no extra text).

IMPORTANT: An OCR engine has already extracted the text from this slide.
The OCR text is provided below as a reference. Use it to ensure accurate spelling
of all words, names, numbers, and technical terms. When the OCR text and your
visual reading conflict, prefer the OCR text for exact spelling."""

USER_PROMPT_TEMPLATE = """\
Below is the OCR-extracted text from this slide (use it as a spelling reference):
--- OCR TEXT ---
{ocr_text}
--- END OCR TEXT ---

Now analyze this PPT slide screenshot and extract its full content into the following JSON structure.
Fill in every field you can identify. If a field is not present in the slide, use null.

{{
  "slide_title": "Main title text of the slide",
  "subtitle": "Subtitle or secondary heading if present",
  "body_text": [
    "Each bullet point or paragraph as a separate string",
    "Preserve the original text as closely as possible"
  ],
  "charts": [
    {{
      "chart_type": "bar|line|pie|scatter|table|diagram|flowchart|other",
      "chart_title": "Title or caption of the chart",
      "description": "What the chart shows — axes, categories, trends",
      "data_points": ["Key numbers, labels, or values visible in the chart"],
      "key_takeaway": "The main insight or conclusion from this chart"
    }}
  ],
  "tables": [
    {{
      "table_title": "Title or caption if any",
      "headers": ["column1", "column2"],
      "rows": [["val1", "val2"]]
    }}
  ],
  "images_and_diagrams": [
    {{
      "type": "photo|icon|diagram|logo|screenshot|other",
      "description": "What the image or diagram depicts"
    }}
  ],
  "footer_or_notes": "Page number, copyright, footnotes, or source citations",
  "overall_topic": "One-sentence summary of what this slide is about",
  "key_points": [
    "Top 3-5 key takeaways or important facts from the slide"
  ]
}}

Important:
- Use the OCR text above as the authoritative source for exact spelling of all words and numbers.
- For charts, try to read actual data values (numbers, percentages) where visible.
- Preserve technical terms, abbreviations, and units exactly as shown.
- Preserve any Korean text from the slide exactly as shown.
- Write ALL your descriptions, summaries, takeaways, and key_points in Korean.
  Exception: if the slide is entirely in English with no Korean at all, respond in English."""

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


def _clean_model_text(raw_text: str) -> str:
    """모델 응답에서 코드 펜스를 제거한다."""
    text = raw_text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _parse_json_response(raw_text: str, label: str) -> dict | None:
    """모델 응답을 JSON 으로 파싱한다."""
    cleaned_text = _clean_model_text(raw_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        print(f"[WARNING] {label} JSON 파싱 실패: {exc}")
        print(f"[INFO] {label} Raw 응답:\n{cleaned_text}")
        return None


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
            width = _coerce_float(raw_bbox.get("width"))
            height = _coerce_float(raw_bbox.get("height"))
            if None in {left, top, width, height}:
                return None
            right = left + width
            bottom = top + height
        elif {"x", "y", "w", "h"} <= raw_bbox.keys():
            left = _coerce_float(raw_bbox.get("x"))
            top = _coerce_float(raw_bbox.get("y"))
            width = _coerce_float(raw_bbox.get("w"))
            height = _coerce_float(raw_bbox.get("h"))
            if None in {left, top, width, height}:
                return None
            right = left + width
            bottom = top + height
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

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _normalize_element_type(raw_type: str) -> str:
    """모델이 반환한 요소 타입을 통일한다."""
    normalized = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "body": "text_box",
        "body_text": "text_box",
        "bullet": "text_box",
        "paragraph": "text_box",
        "text": "text_box",
        "textbox": "text_box",
        "text_area": "text_box",
        "textblock": "text_box",
        "picture": "image",
        "photo": "image",
        "logo": "image",
        "screenshot": "image",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in BOX_COLORS:
        return "other"
    return normalized


def _normalize_layout_elements(raw_elements) -> list[dict]:
    """레이아웃 박스 목록을 정리한다."""
    if not isinstance(raw_elements, list):
        return []

    normalized_elements = []
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
            item.get("label")
            or item.get("text")
            or item.get("title")
            or item.get("description")
            or ""
        ).strip()

        normalized_elements.append({
            "type": element_type,
            "label": label,
            "bbox_1000": bbox,
        })

    normalized_elements.sort(
        key=lambda item: (item["bbox_1000"]["top"], item["bbox_1000"]["left"])
    )

    for index, item in enumerate(normalized_elements, 1):
        item["index"] = index

    return normalized_elements


def _bbox_1000_to_pixels(bbox_1000: dict, image_size: tuple[int, int]) -> dict:
    """1000 기준 좌표를 실제 이미지 픽셀 좌표로 변환한다."""
    width, height = image_size
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)

    left = int(round((bbox_1000["left"] / LAYOUT_COORD_SCALE) * width))
    top = int(round((bbox_1000["top"] / LAYOUT_COORD_SCALE) * height))
    right = int(round((bbox_1000["right"] / LAYOUT_COORD_SCALE) * width))
    bottom = int(round((bbox_1000["bottom"] / LAYOUT_COORD_SCALE) * height))

    left = max(0, min(max_x, left))
    top = max(0, min(max_y, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _get_image_size(image_path: Path) -> tuple[int, int] | None:
    """Pillow 로 이미지 크기를 읽는다."""
    if not PIL_AVAILABLE:
        return None

    with Image.open(image_path) as image:
        return image.size


def _enrich_layout_elements(
    raw_elements: list[dict],
    image_size: tuple[int, int] | None,
) -> list[dict]:
    """레이아웃 요소에 픽셀 좌표를 추가한다."""
    enriched_elements = []
    for item in raw_elements:
        enriched_item = dict(item)
        if image_size is not None:
            enriched_item["bbox_pixels"] = _bbox_1000_to_pixels(
                item["bbox_1000"], image_size,
            )
        enriched_elements.append(enriched_item)
    return enriched_elements


# ──────────────────────────────────────────────
# 파이프라인 함수들
# ──────────────────────────────────────────────
def run_ocr(image_path: Path) -> str:
    """PaddleOCR-VL 로 이미지에서 텍스트를 추출한다."""
    print(f"[INFO] [1단계] OCR 텍스트 추출 중... ({OCR_SERVICE})")
    start = time.time()
    try:
        text, _ = vlm_chat(OCR_URL, OCR_MODEL, image_path, "", "OCR:")
    except Exception as exc:
        print(f"[WARNING] OCR 호출 실패 (VLM 단독으로 진행): {exc}")
        return ""
    elapsed = time.time() - start
    text = text.strip()
    print(f"[INFO] OCR 완료: {elapsed:.1f}초, {len(text)}자 추출")
    return text


def run_chart_ocr(image_path: Path) -> str:
    """PaddleOCR-VL 의 Chart Recognition 으로 차트 데이터를 추출한다."""
    print(f"[INFO] [1단계] 차트 인식 중... ({OCR_SERVICE})")
    start = time.time()
    try:
        text, _ = vlm_chat(OCR_URL, OCR_MODEL, image_path, "", "Chart Recognition:")
    except Exception as exc:
        print(f"[WARNING] Chart OCR 호출 실패 (무시): {exc}")
        return ""
    elapsed = time.time() - start
    text = text.strip()
    print(f"[INFO] Chart OCR 완료: {elapsed:.1f}초, {len(text)}자 추출")
    return text


def run_layout_analysis(image_path: Path, ocr_text: str) -> list[dict]:
    """VLM 으로 텍스트 박스/차트 등의 레이아웃 박스를 추출한다."""
    print(f"[INFO] [3단계] 레이아웃 박스 분석 중... ({VLM_SERVICE})")
    user_prompt = LAYOUT_PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    start = time.time()
    try:
        raw_text, usage = vlm_chat(
            VLM_URL, VLM_MODEL, image_path, LAYOUT_SYSTEM_MESSAGE, user_prompt, temperature=0.0,
        )
    except Exception as exc:
        print(f"[WARNING] 레이아웃 분석 실패: {exc}")
        return []

    elapsed = time.time() - start
    print(f"[INFO] 레이아웃 응답 시간: {elapsed:.1f}초")

    if usage:
        print(
            f"[INFO] 레이아웃 토큰: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}, "
            f"total={usage.get('total_tokens', '?')}"
        )

    parsed = _parse_json_response(raw_text, "레이아웃")
    if not isinstance(parsed, dict):
        return []

    elements = _normalize_layout_elements(parsed.get("elements"))
    print(f"[INFO] 레이아웃 박스 추출: {len(elements)}개")
    return elements


def build_combined_ocr(image_path: Path) -> str:
    """OCR 텍스트와 차트 인식 결과를 하나로 합친다."""
    ocr_text = run_ocr(image_path)
    chart_text = run_chart_ocr(image_path)

    combined_ocr = ocr_text
    if chart_text:
        combined_ocr += f"\n\n[Chart Data]\n{chart_text}"
    if not combined_ocr.strip():
        combined_ocr = "(OCR 결과 없음 — 이미지만으로 분석)"
    return combined_ocr


def analyze_slide_layout(image_path: Path) -> dict | None:
    """단일 슬라이드 이미지에서 레이아웃 박스만 추출한다."""
    print(f"\n{'='*60}")
    print(f"[INFO] 레이아웃 분석 시작: {image_path.name}")
    print(f"{'='*60}")

    if not image_path.exists():
        print(f"[WARNING] 파일이 존재하지 않습니다: {image_path}")
        return None

    file_size_kb = image_path.stat().st_size / 1024
    print(f"[INFO] 파일 크기: {file_size_kb:.1f} KB")

    combined_ocr = build_combined_ocr(image_path)
    layout_elements = run_layout_analysis(image_path, combined_ocr)

    result = {
        "_ocr_text": combined_ocr,
        "_layout_elements": [],
    }

    image_size = _get_image_size(image_path)
    if image_size is not None:
        result["_image_size"] = {"width": image_size[0], "height": image_size[1]}
    if layout_elements:
        result["_layout_elements"] = _enrich_layout_elements(layout_elements, image_size)

    return result


def analyze_slide(image_path: Path) -> dict | None:
    """단일 슬라이드 이미지를 분석하여 구조화된 결과를 반환한다."""
    print(f"\n{'='*60}")
    print(f"[INFO] 분석 시작: {image_path.name}")
    print(f"{'='*60}")

    if not image_path.exists():
        print(f"[WARNING] 파일이 존재하지 않습니다: {image_path}")
        return None

    file_size_kb = image_path.stat().st_size / 1024
    print(f"[INFO] 파일 크기: {file_size_kb:.1f} KB")

    # ── 1단계: OCR 텍스트 추출 ──
    combined_ocr = build_combined_ocr(image_path)

    # ── 2단계: VLM 구조 분석 (OCR 텍스트 참조) ──
    print(f"[INFO] [2단계] VLM 구조 분석 중... ({VLM_SERVICE})")
    user_prompt = USER_PROMPT_TEMPLATE.format(ocr_text=combined_ocr)

    start = time.time()
    try:
        raw_text, usage = vlm_chat(
            VLM_URL, VLM_MODEL, image_path, SYSTEM_MESSAGE, user_prompt, temperature=0.1,
        )
    except Exception as exc:
        print(f"[ERROR] VLM 호출 실패: {exc}")
        return None

    elapsed = time.time() - start
    print(f"[INFO] VLM 응답 시간: {elapsed:.1f}초")

    if usage:
        print(
            f"[INFO] 토큰: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}, "
            f"total={usage.get('total_tokens', '?')}"
        )

    result = _parse_json_response(raw_text, "구조 분석")
    if result is None:
        return {"raw_response": _clean_model_text(raw_text), "_ocr_text": combined_ocr}

    print(f"[INFO] JSON 파싱 성공")
    result["_ocr_text"] = combined_ocr

    layout_elements = run_layout_analysis(image_path, combined_ocr)
    image_size = _get_image_size(image_path)
    if image_size is not None:
        result["_image_size"] = {"width": image_size[0], "height": image_size[1]}
    if layout_elements:
        result["_layout_elements"] = _enrich_layout_elements(layout_elements, image_size)

    return result


# ──────────────────────────────────────────────
# 출력 / 저장
# ──────────────────────────────────────────────
def print_result(image_name: str, result: dict) -> None:
    """분석 결과를 보기 좋게 출력한다."""
    print(f"\n{'─'*60}")
    print(f"  결과: {image_name}")
    print(f"{'─'*60}")

    if "raw_response" in result:
        print(result["raw_response"])
        return

    title = result.get("slide_title")
    if title:
        print(f"\n■ 제목: {title}")
    subtitle = result.get("subtitle")
    if subtitle:
        print(f"  부제: {subtitle}")

    topic = result.get("overall_topic")
    if topic:
        print(f"\n■ 주제: {topic}")

    body = result.get("body_text") or []
    if body:
        print(f"\n■ 본문 ({len(body)}항목):")
        for i, text in enumerate(body, 1):
            print(f"  {i}. {text}")

    key_points = result.get("key_points") or []
    if key_points:
        print(f"\n■ 핵심 포인트:")
        for point in key_points:
            print(f"  • {point}")

    charts = result.get("charts") or []
    if charts:
        print(f"\n■ 차트 ({len(charts)}개):")
        for i, chart in enumerate(charts, 1):
            print(f"  [{i}] {chart.get('chart_type', '?')} — {chart.get('chart_title', '제목 없음')}")
            desc = chart.get("description")
            if desc:
                print(f"      설명: {desc}")
            data = chart.get("data_points") or []
            if data:
                print(f"      데이터: {', '.join(str(d) for d in data)}")
            takeaway = chart.get("key_takeaway")
            if takeaway:
                print(f"      핵심: {takeaway}")

    tables = result.get("tables") or []
    if tables:
        print(f"\n■ 표 ({len(tables)}개):")
        for i, table in enumerate(tables, 1):
            print(f"  [{i}] {table.get('table_title', '제목 없음')}")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                print(f"      헤더: {' | '.join(headers)}")
            for row in rows[:5]:
                print(f"      → {' | '.join(str(v) for v in row)}")
            if len(rows) > 5:
                print(f"      ... 외 {len(rows) - 5}행")

    images = result.get("images_and_diagrams") or []
    if images:
        print(f"\n■ 이미지/다이어그램 ({len(images)}개):")
        for img in images:
            print(f"  • [{img.get('type', '?')}] {img.get('description', '')}")

    footer = result.get("footer_or_notes")
    if footer:
        print(f"\n■ 푸터/노트: {footer}")

    layout_elements = result.get("_layout_elements") or []
    if layout_elements:
        print(f"\n■ 레이아웃 박스 ({len(layout_elements)}개):")
        for item in layout_elements:
            bbox = item.get("bbox_pixels") or item.get("bbox_1000") or {}
            print(
                f"  [{item.get('index', '?')}] {item.get('type', 'other')} "
                f"({bbox.get('left', '?')}, {bbox.get('top', '?')}) - "
                f"({bbox.get('right', '?')}, {bbox.get('bottom', '?')})"
            )

    print()


def _format_result_text(result: dict) -> str:
    """분석 결과를 사람이 읽기 좋은 텍스트로 변환한다."""
    if "raw_response" in result:
        return result["raw_response"]

    lines: list[str] = []

    title = result.get("slide_title")
    if title:
        lines.append(f"제목: {title}")
    subtitle = result.get("subtitle")
    if subtitle:
        lines.append(f"부제: {subtitle}")

    topic = result.get("overall_topic")
    if topic:
        lines.append(f"\n주제: {topic}")

    body = result.get("body_text") or []
    if body:
        lines.append(f"\n본문 ({len(body)}항목):")
        for i, text in enumerate(body, 1):
            lines.append(f"  {i}. {text}")

    key_points = result.get("key_points") or []
    if key_points:
        lines.append("\n핵심 포인트:")
        for point in key_points:
            lines.append(f"  - {point}")

    charts = result.get("charts") or []
    if charts:
        lines.append(f"\n차트 ({len(charts)}개):")
        for i, chart in enumerate(charts, 1):
            lines.append(f"  [{i}] {chart.get('chart_type', '?')} - {chart.get('chart_title', '제목 없음')}")
            desc = chart.get("description")
            if desc:
                lines.append(f"      설명: {desc}")
            data = chart.get("data_points") or []
            if data:
                for d in data:
                    lines.append(f"      - {d}")
            takeaway = chart.get("key_takeaway")
            if takeaway:
                lines.append(f"      핵심: {takeaway}")

    tables = result.get("tables") or []
    if tables:
        lines.append(f"\n표 ({len(tables)}개):")
        for i, table in enumerate(tables, 1):
            lines.append(f"  [{i}] {table.get('table_title', '제목 없음')}")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                lines.append(f"      {' | '.join(headers)}")
                lines.append(f"      {'-+-'.join('-' * len(h) for h in headers)}")
            for row in rows:
                lines.append(f"      {' | '.join(str(v) for v in row)}")

    images = result.get("images_and_diagrams") or []
    if images:
        lines.append(f"\n이미지/다이어그램 ({len(images)}개):")
        for img in images:
            lines.append(f"  - [{img.get('type', '?')}] {img.get('description', '')}")

    footer = result.get("footer_or_notes")
    if footer:
        lines.append(f"\n푸터/노트: {footer}")

    layout_elements = result.get("_layout_elements") or []
    if layout_elements:
        lines.append(f"\n레이아웃 박스 ({len(layout_elements)}개):")
        for item in layout_elements:
            bbox_1000 = item.get("bbox_1000") or {}
            pixels = item.get("bbox_pixels") or {}
            label = item.get("label")
            line = (
                f"  [{item.get('index', '?')}] {item.get('type', 'other')} "
                f"norm=({bbox_1000.get('left', '?')}, {bbox_1000.get('top', '?')}, "
                f"{bbox_1000.get('right', '?')}, {bbox_1000.get('bottom', '?')})"
            )
            if pixels:
                line += (
                    f" px=({pixels.get('left', '?')}, {pixels.get('top', '?')}, "
                    f"{pixels.get('right', '?')}, {pixels.get('bottom', '?')})"
                )
            if label:
                line += f" label={label}"
            lines.append(line)

    return "\n".join(lines)


def save_layout_result(image_path: Path, result: dict) -> Path | None:
    """레이아웃 박스를 JSON 파일로 저장한다."""
    layout_elements = result.get("_layout_elements") or []
    if not layout_elements:
        return None

    layout_path = image_path.with_suffix(LAYOUT_JSON_SUFFIX)
    payload = {
        "source_image": image_path.name,
        "image_size": result.get("_image_size"),
        "elements": layout_elements,
    }

    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 레이아웃 저장: {layout_path.name}")
    return layout_path


def save_overlay_image(image_path: Path, result: dict) -> Path | None:
    """원본 이미지 위에 레이아웃 박스를 그린 overlay 이미지를 저장한다."""
    layout_elements = result.get("_layout_elements") or []
    if not layout_elements:
        return None

    if not PIL_AVAILABLE:
        print("[WARNING] Pillow 미설치로 overlay 이미지 저장을 건너뜁니다.")
        return None

    overlay_path = image_path.with_suffix(OVERLAY_IMAGE_SUFFIX)

    with Image.open(image_path) as image:
        base_image = image.convert("RGBA")
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        font = ImageFont.load_default()

        line_width = max(2, min(base_image.size) // 250)
        label_padding = max(4, line_width + 1)

        for item in layout_elements:
            bbox = item.get("bbox_pixels")
            if not bbox:
                bbox = _bbox_1000_to_pixels(item["bbox_1000"], base_image.size)

            color = BOX_COLORS.get(item.get("type", "other"), BOX_COLORS["other"])
            outline_color = (*color, 255)
            fill_color = (*color, 36)

            left = bbox["left"]
            top = bbox["top"]
            right = bbox["right"]
            bottom = bbox["bottom"]

            draw.rectangle(
                (left, top, right, bottom),
                outline=outline_color,
                width=line_width,
                fill=fill_color,
            )

            label_text = f"{item.get('index', '?')}. {item.get('type', 'other')}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            label_width = text_width + label_padding * 2
            label_height = text_height + label_padding * 2
            label_left = min(left, max(0, base_image.width - label_width))
            label_top = top - label_height - label_padding
            if label_top < 0:
                label_top = min(max(0, top + label_padding), max(0, base_image.height - label_height))

            label_right = label_left + label_width
            label_bottom = label_top + label_height

            draw.rectangle(
                (label_left, label_top, label_right, label_bottom),
                fill=(*color, 220),
            )
            draw.text(
                (label_left + label_padding, label_top + label_padding),
                label_text,
                fill=(255, 255, 255, 255),
                font=font,
            )

        merged = Image.alpha_composite(base_image, overlay).convert("RGB")
        merged.save(overlay_path, quality=95)

    print(f"[INFO] Overlay 저장: {overlay_path.name}")
    return overlay_path


def save_result(image_path: Path, result: dict) -> Path:
    """분석 결과를 읽기 좋은 텍스트 파일로 저장한다."""
    txt_path = image_path.with_suffix(".result.txt")
    text = _format_result_text(result)

    ocr_text = result.get("_ocr_text", "")
    if ocr_text:
        text += f"\n\n{'─'*40}\nOCR 원문 (참고용):\n{'─'*40}\n{ocr_text}\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[INFO] 결과 저장: {txt_path.name}")
    return txt_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    """메인 실행."""
    image_paths = sorted(
        p for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("[INFO] PPT 슬라이드 분석기 시작")
    print(f"[INFO] OCR: {OCR_SERVICE} ({OCR_URL})")
    print(f"[INFO] VLM: {VLM_SERVICE} ({VLM_URL})")
    print(f"[INFO] 대상 이미지: {len(image_paths)}개")

    if not image_paths:
        print(f"[ERROR] 분석할 이미지가 없습니다.")
        print(f"[INFO] 이 폴더에 .jpg 또는 .png 이미지를 넣어주세요.")
        print(f"[INFO] 경로: {SAMPLE_DIR}")
        sys.exit(1)

    results = {}
    for image_path in image_paths:
        result = analyze_slide(image_path)
        if result is not None:
            results[image_path.name] = result
            print_result(image_path.name, result)
            save_result(image_path, result)
            save_layout_result(image_path, result)
            save_overlay_image(image_path, result)

    print(f"\n{'='*60}")
    print(f"[INFO] 분석 완료: {len(results)}/{len(image_paths)}개 성공")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
