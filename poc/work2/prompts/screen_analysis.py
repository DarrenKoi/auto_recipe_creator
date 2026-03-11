"""Screen analysis prompt builders for `poc.work2`."""

from typing import Iterable


def _render_state_context(state_definitions: dict | None) -> str:
    if not state_definitions:
        return ""
    state_list = "\n".join(
        [
            f"- {sid}: {sdef.get('state_name', '')} - {sdef.get('description', '')}"
            for sid, sdef in state_definitions.items()
        ]
    )
    return f"\n\n알려진 상태 목록:\n{state_list}"


def _render_extra_instructions(extra_instructions: Iterable[str] | None) -> str:
    items = [item.strip() for item in (extra_instructions or ()) if item and item.strip()]
    if not items:
        return ""
    rendered = "\n".join([f"- {item}" for item in items])
    return f"\n\n추가 지침:\n{rendered}"


def build_state_recognition_prompt(
    image_width: int | None = None,
    image_height: int | None = None,
    state_definitions: dict | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> str:
    """Build state-recognition prompt with optional OCR hints."""
    image_size_hint = ""
    if image_width and image_height and image_width > 0 and image_height > 0:
        image_size_hint = (
            f"\n이미지 해상도는 {image_width}x{image_height} 입니다. "
            "좌표는 반드시 이 픽셀 좌표계를 사용하세요."
        )

    return f"""당신은 GUI 화면 분석 전문가입니다. 주어진 스크린샷을 분석하여 현재 화면의 상태를 파악해주세요.
{image_size_hint}

다음 정보를 JSON 형식으로 반환해주세요:
{{
    "state_id": "화면 상태 식별자 (예: main_menu, recipe_editor, error_popup)",
    "state_name": "화면 상태 이름 (한글)",
    "confidence": 0.0-1.0 사이의 확신도,
    "description": "현재 화면에 대한 상세 설명",
    "ui_elements": [
        {{
            "name": "요소 이름",
            "type": "button/input/label/etc",
            "location": "위치 설명",
            "x": 0,
            "y": 0,
            "coord_anchor": "요소 중심점 설명"
        }}
    ],
    "suggested_actions": ["가능한 액션 1", "가능한 액션 2"]
}}{_render_state_context(state_definitions)}{_render_extra_instructions(extra_instructions)}

주의사항:
- ui_elements의 x, y는 클릭 가능한 지점을 의미합니다.
- x, y는 정수 픽셀 좌표여야 합니다.
- 좌표 범위는 0 <= x < 이미지 너비, 0 <= y < 이미지 높이 입니다.
- 좌표를 확신할 수 없는 요소는 ui_elements에서 제외하세요.

분석 결과를 JSON으로만 반환해주세요."""


def build_measurement_judgment_prompt(
    extra_instructions: Iterable[str] | None = None,
) -> str:
    """Build measurement-judgment prompt."""
    return f"""당신은 반도체 측정 장비의 결과 분석 전문가입니다. 주어진 측정 결과 화면을 분석하여 측정 성공 여부를 판단해주세요.

다음 정보를 JSON 형식으로 반환해주세요:
{{
    "success": true/false,
    "confidence": 0.0-1.0 사이의 확신도,
    "failure_reason": "실패 시 원인 (position_offset, focus_error, pattern_mismatch 등)",
    "suggested_adjustment": {{
        "direction": "left/right/up/down",
        "amount": "small/medium/large"
    }}
}}{_render_extra_instructions(extra_instructions)}

측정 성공 기준:
- 측정값이 명확하게 표시되어 있음
- 에러 메시지가 없음
- 측정 패턴이 올바르게 인식됨

분석 결과를 JSON으로만 반환해주세요."""


def build_general_query_prompt(
    question: str,
    extra_instructions: Iterable[str] | None = None,
) -> str:
    """Build generic QA prompt for screenshots."""
    extra_text = _render_extra_instructions(extra_instructions)
    return f"""당신은 GUI 화면 분석 전문가입니다. 주어진 화면에 대해 다음 질문에 답변해주세요.

질문: {question}{extra_text}

답변은 명확하고 간결하게 해주세요."""
