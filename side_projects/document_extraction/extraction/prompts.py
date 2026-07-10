"""스테이지별 VLM 프롬프트 빌더.

각 빌더는 repo 컨벤션에 맞춰 `(system_message, user_message)` 튜플을 반환한다.
이미지 width/height 를 받아 좌표를 픽셀 기준으로 요구한다(0-1 정규화 혼선 방지).

모델은 항상 "보이는 것만" 추출하고, 읽을 수 없으면 unknown 으로 표기하도록
지시한다(pipeline_plan.md Stage 6 important rule). 값 창작 금지가 핵심.
"""


_JSON_ONLY = (
    "Return ONLY a single JSON object. No prose, no markdown fences. "
    "If a field is unreadable or not visible, use an empty string or null. "
    "Never invent values that are not visible in the image."
)


def prompt_first_pass_ocr(width: int, height: int) -> tuple[str, str]:
    """Stage 2: paddleocr-vl-1.5 OCR/document parsing 프롬프트."""
    system = (
        "You are a document OCR and parsing engine. "
        "Extract visible text, reading order, tables, charts, and formulas "
        "from the screenshot. " + _JSON_ONLY
    )
    user = (
        f"Image size is {width}x{height} pixels (origin top-left).\n"
        "Extract everything visible and return JSON with keys:\n"
        '{"raw_text": "...", "reading_order": ["..."], '
        '"tables": [{"title": "", "header": [], "rows": [[]]}], '
        '"charts": [{"title": "", "axis_labels": [], "legend_labels": [], '
        '"visible_values": []}], '
        '"formulas": [{"latex": "", "nearby_label": ""}]}'
    )
    return system, user


def prompt_layout_regions(width: int, height: int) -> tuple[str, str]:
    """Stage 3: ui-venus layout/region detection 프롬프트."""
    system = (
        "You are a document layout analyzer. Classify the screenshot type and "
        "detect visible regions with approximate pixel bounding boxes. " + _JSON_ONLY
    )
    user = (
        f"Image size is {width}x{height} pixels (origin top-left).\n"
        "Identify whether this is a powerpoint slide, pdf page, excel sheet, or "
        "unknown, and list visible regions. Return JSON:\n"
        '{"source_type": "powerpoint|pdf|excel|unknown", '
        '"regions": [{"type": "title|body|table|chart|formula|footer|legend|other", '
        '"bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0}}]}'
    )
    return system, user


def prompt_crop_refine(width: int, height: int, region_type: str) -> tuple[str, str]:
    """Stage 4: mai-ui / paddleocr crop refinement 프롬프트.

    crop 이미지(원본의 일부)를 입력으로 받아 작은 영역의 텍스트/라벨을 재인식한다.
    width/height 는 crop 이미지의 크기.
    """
    system = (
        "You are refining text recognition on a small cropped region of a "
        "document screenshot. Read every visible character precisely. " + _JSON_ONLY
    )
    user = (
        f"This crop is {width}x{height} pixels and is a '{region_type}' region.\n"
        "Return JSON: {\"text\": \"\", \"header\": [], \"rows\": [[]], "
        '"labels": []}. Use only what is visible.'
    )
    return system, user


def prompt_synthesis(source_type: str) -> tuple[str, str]:
    """Stage 6: kimi-k2.6 합성 프롬프트 (텍스트 evidence + 저해상 원본).

    evidence JSON 을 user_text 에 붙여 호출한다. 표/수식은 evidence 를 그대로
    쓰고 재생성하지 않도록 강하게 지시한다.
    """
    system = (
        "You synthesize a final document extraction from OCR and layout evidence. "
        "Use the provided evidence as the source of truth for exact text, tables, "
        "and formulas; do NOT regenerate or alter those values. Mark anything "
        "uncertain as unknown. Never fabricate numbers or labels. " + _JSON_ONLY
    )
    user = (
        f"Source type: {source_type}. Below is the extracted evidence as JSON, "
        "plus the original (low-res) screenshot for overall context.\n"
        "Produce JSON: {\"summary_markdown\": \"\", "
        '"overall_confidence": 0.0, "unresolved": ["..."]}.\n'
        "EVIDENCE:\n"
    )
    return system, user


def prompt_synthesis_text_only(source_type: str) -> tuple[str, str]:
    """Stage 6 텍스트 전용 합성 프롬프트 (glm-5.2 등 text LLM, 이미지 없음).

    prompt_synthesis 와 동일 계약이되 스크린샷 언급을 제거한다 — evidence JSON 만
    보고 합성하며, evidence 에 없는 값은 절대 만들지 않는다.
    """
    system = (
        "You synthesize a final document extraction from OCR and layout evidence. "
        "You are given ONLY structured evidence JSON (no image). "
        "Use the evidence as the sole source of truth for exact text, tables, "
        "and formulas; do NOT regenerate or alter those values. Mark anything "
        "uncertain as unknown. Never fabricate numbers or labels. " + _JSON_ONLY
    )
    user = (
        f"Source type: {source_type}. Below is the extracted evidence as JSON.\n"
        "Produce JSON: {\"summary_markdown\": \"\", "
        '"overall_confidence": 0.0, "unresolved": ["..."]}.\n'
        "EVIDENCE:\n"
    )
    return system, user


__all__ = [
    "prompt_crop_refine",
    "prompt_first_pass_ocr",
    "prompt_layout_regions",
    "prompt_synthesis",
    "prompt_synthesis_text_only",
]
