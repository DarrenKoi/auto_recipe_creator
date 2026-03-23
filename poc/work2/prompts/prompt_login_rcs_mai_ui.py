"""RCS 로그인 화면 crop 확대용 MAI-UI 프롬프트 빌더."""

TARGET_KEY = "userid_input"

_SYSTEM_MESSAGE = (
    "You are a precise Windows GUI click-point locator. "
    "You receive a zoomed-in crop from a login dialog, not the full desktop. "
    "Return ONLY valid JSON."
)

_USER_TEXT = """\
This image is a zoomed-in crop around the 'User ID' row of a Windows 'Remote Control System' login dialog.

Target:
- "userid_input" = the editable text field where the user types their user ID

Grounding rules:
- Coordinates are relative to THIS cropped image only, not the original full screenshot.
- Return a safe click point inside the editable typing area of the input field.
- Prefer the left-inner text entry area where typed text would begin.
- Aim around 12-22% of the field width from the left edge and around 55-68% of the field height from the top edge.
- Do not click the label text, border, highlight, shadow, or empty background around the field.
- If the label text is visible, use it only as context to identify the correct field.
- If the target is not visible enough, return null for the target.

Return ONLY this JSON:
{
  "coord_system": "relative_1000",
  "userid_input": {
    "x": 0,
    "y": 0
  }
}

If the target is not visible enough, return ONLY:
{
  "coord_system": "relative_1000",
  "userid_input": null
}
"""


def build_login_rcs_mai_ui_zoom_prompt() -> tuple[str, str]:
    """MAI-UI 확대 crop 용 단일 타겟 포인트 프롬프트를 반환한다."""
    return _SYSTEM_MESSAGE, _USER_TEXT


__all__ = ["TARGET_KEY", "build_login_rcs_mai_ui_zoom_prompt"]
