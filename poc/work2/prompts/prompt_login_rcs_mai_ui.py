"""RCS 로그인 화면 crop 확대용 MAI-UI 프롬프트 빌더.

임의의 타겟 요소에 대해 zoom crop 이미지에서
정밀 클릭 포인트를 찾는 프롬프트를 생성한다.
"""

_SYSTEM_MESSAGE = (
    "You are a precise Windows GUI click-point locator. "
    "You receive a zoomed-in crop from a dialog window, not the full desktop. "
    "Return ONLY valid JSON."
)

_USER_TEXT_TEMPLATE = """\
This image is a zoomed-in crop from a Windows dialog.

Target:
- "{target_key}" = {target_description}

Grounding rules:
- Coordinates are relative to THIS cropped image only, not the original full screenshot.
- Return a safe click point inside the interactive area of the target element.
- For editable text fields, prefer the left-inner text entry area where typed text would begin.
- For buttons, prefer the center of the button surface.
- For combo boxes or dropdowns, prefer the center of the dropdown control.
- Do not click on label text, borders, highlights, shadows, or empty background around the target.
- If nearby label text is visible, use it only as context to identify the correct element.
- If the target is not visible enough, return null for the target.

Return ONLY this JSON:
{{
  "coord_system": "relative_1000",
  "{target_key}": {{
    "x": 0,
    "y": 0
  }}
}}

If the target is not visible enough, return ONLY:
{{
  "coord_system": "relative_1000",
  "{target_key}": null
}}
"""


def build_mai_ui_zoom_prompt(
    target_key: str,
    target_description: str,
) -> tuple[str, str]:
    """MAI-UI 확대 crop 용 단일 타겟 포인트 프롬프트를 반환한다.

    Args:
        target_key: 타겟 식별 키 (e.g. "userid_input", "login_button").
        target_description: VLM 에 전달할 자연어 설명.

    Returns:
        (system_message, user_message) 튜플.
    """
    user_text = _USER_TEXT_TEMPLATE.format(
        target_key=target_key,
        target_description=target_description,
    )
    return _SYSTEM_MESSAGE, user_text


# 하위 호환용 래퍼 — 기존 호출처에서 TARGET_KEY 참조 시 사용
TARGET_KEY = "userid_input"
_USERID_INPUT_DESCRIPTION = (
    "the editable text field where the user types their user ID"
)


def build_login_rcs_mai_ui_zoom_prompt() -> tuple[str, str]:
    """기존 userid_input 전용 프롬프트 (하위 호환)."""
    return build_mai_ui_zoom_prompt(TARGET_KEY, _USERID_INPUT_DESCRIPTION)


__all__ = [
    "TARGET_KEY",
    "build_login_rcs_mai_ui_zoom_prompt",
    "build_mai_ui_zoom_prompt",
]
