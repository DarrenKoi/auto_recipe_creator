"""workflow_1 전용 RCS 로그인 화면 crop 확대용 MAI-UI 프롬프트 빌더."""

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
    """MAI-UI 확대 crop 용 단일 타겟 포인트 프롬프트를 반환한다."""
    user_text = _USER_TEXT_TEMPLATE.format(
        target_key=target_key,
        target_description=target_description,
    )
    return _SYSTEM_MESSAGE, user_text


__all__ = ["build_mai_ui_zoom_prompt"]

