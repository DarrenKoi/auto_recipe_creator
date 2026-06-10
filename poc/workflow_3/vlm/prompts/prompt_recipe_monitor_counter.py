"""Recipe Monitor 측정 카운터(분자) grounding 프롬프트 빌더.

tool 창의 Recipe Monitor 패널에서 측정 진행 카운터(예: 2/350)의 분자 위치를
ui-venus 공식 단일 요소 grounding 형식으로 요청한다. 창이 드래그로 움직일 수
있어 고정 ROI 가 불가하므로, 사이클당 1회 이 grounding 으로 위치를 캐시한다.
"""

from poc.workflow_3.vlm.prompts.prompt_login_rcs_ui_venus import (
    UI_VENUS_OFFICIAL_PROMPT_TEMPLATE,
)

# 오피스 캘리브레이션에서 문구를 조정할 수 있도록 instruction 을 상수로 분리한다.
# 첫-글자 anchoring 원칙: 'Recipe Monitor' 텍스트를 먼저 찾게 한 뒤 행 -> 카운터 순.
RECIPE_MONITOR_NUMERATOR_INSTRUCTION = (
    "Find the visible text 'Recipe Monitor' first, then inside that Recipe Monitor "
    "area find the row showing Port, Slot and Recipe. In that row, locate the "
    "measurement progress counter that looks like 'N/M' (for example '2/350'). "
    "Output the center point of the numerator N only (the integer BEFORE the slash '/')"
)


def build_recipe_monitor_counter_prompt() -> tuple[str, str]:
    """ui-venus 공식 단일 요소 형식으로 분자 중심점을 요청한다."""
    user_text = UI_VENUS_OFFICIAL_PROMPT_TEMPLATE.format(
        instruction=RECIPE_MONITOR_NUMERATOR_INSTRUCTION
    )
    return "", user_text


__all__ = [
    "RECIPE_MONITOR_NUMERATOR_INSTRUCTION",
    "build_recipe_monitor_counter_prompt",
]
