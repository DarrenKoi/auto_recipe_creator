"""문서 영역(region) 검출용 ui-venus prompt builder.

`poc/work2/prompts/prompt_screen_analysis.py` 의 스타일을 따르되,
GUI 가 아닌 문서 페이지(슬라이드, PDF 페이지, 스프레드시트 캡처)를 대상으로 한다.
"""

from typing import Iterable


_REGION_TYPES: tuple[str, ...] = (
    "title",
    "body",
    "list",
    "table",
    "chart",
    "formula",
    "figure",
    "footer",
    "header",
    "legend",
    "caption",
    "note",
    "page_number",
    "other",
)


def build_doc_region_prompt(
    image_width: int | None = None,
    image_height: int | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """문서 페이지의 영역과 bbox 를 JSON 으로 요청하는 prompt 를 반환한다.

    Returns:
        (system_message, user_text) — Work2VLMClient 에 그대로 넘긴다.
    """
    image_size_hint = ""
    if image_width and image_height and image_width > 0 and image_height > 0:
        image_size_hint = (
            f"\n이미지 해상도는 {image_width}x{image_height} 입니다. "
            "좌표는 반드시 이 픽셀 좌표계를 사용하세요."
        )

    extras_text = ""
    if extra_instructions:
        bullets = "\n".join(
            f"- {item.strip()}" for item in extra_instructions if item and item.strip()
        )
        if bullets:
            extras_text = f"\n\n추가 지침:\n{bullets}"

    type_list = ", ".join(_REGION_TYPES)

    system_message = (
        "당신은 문서 페이지 스크린샷의 시각적 구조를 분석하는 전문가입니다. "
        "출력은 항상 유효한 JSON 만 포함해야 합니다. 추가 설명 텍스트를 붙이지 마세요."
    )

    user_text = f"""주어진 이미지는 문서(슬라이드, PDF 페이지, 스프레드시트 등)의 스크린샷입니다.
화면에서 보이는 시각적 영역을 식별해 다음 JSON 으로 반환해주세요.
{image_size_hint}

```json
{{
  "page_type": "powerpoint|pdf|excel|word|unknown",
  "regions": [
    {{
      "type": "title|body|list|table|chart|formula|figure|footer|header|legend|caption|note|page_number|other",
      "bbox": {{"left": 0, "top": 0, "right": 0, "bottom": 0}},
      "text": "영역에서 보이는 핵심 텍스트 또는 짧은 설명",
      "confidence": 0.0
    }}
  ],
  "notes": "전체 페이지에 대한 간단한 코멘트 (선택)"
}}
```

규칙:
- bbox 는 픽셀 좌표 정수입니다. 좌표는 0 이상 (left,top)<(right,bottom) 입니다.
- 가능한 region type 목록: {type_list}.
- 보이지 않거나 잘려서 추정할 수 없는 영역은 포함하지 마세요.
- text 가 너무 길면 핵심만 200자 이내로 요약하세요.
- 결과는 JSON 으로만 반환해주세요. 다른 설명 텍스트는 금지입니다.{extras_text}
"""
    return system_message, user_text
