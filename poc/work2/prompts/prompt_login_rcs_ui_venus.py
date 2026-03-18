"""RCS 로그인 화면 UI-Venus 전용 grounding 프롬프트 빌더.

UI-Venus 에게 각 요소의 좌표를 직접 규정하지 않고,
스크린샷 안에서 실제로 보이는 text/control 을 찾은 뒤
사용자가 실제 클릭할 grounding point 를 고르게 유도한다.
"""

from typing import Iterable


UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS: dict[str, str] = {
    "window_title_text": "the visible window title text 'Remote Control System' in the title bar",
    "close_button": "the standard Windows close button with an 'X' in the title bar",
    "server_label": "the visible text label 'Server' in the first form row",
    "server_input": "the server dropdown or combo box control in the first form row",
    "userid_label": "the visible text label 'User ID' in the second form row",
    "userid_input": "the editable text field in the second form row",
    "password_label": "the visible text label 'Password' in the third form row",
    "password_input": "the editable password field in the third form row",
    "login_button": "the 'Log In' button near the bottom of the dialog",
    "cancel_button": "the 'Cancel' button near the bottom of the dialog",
    "shortcut_button": "the Korean text button in the bottom area (for example '바로가기 설정')",
}

DEFAULT_UI_VENUS_TARGET_KEYS = (
    "window_title_text",
    "close_button",
    "server_label",
    "server_input",
    "userid_label",
    "userid_input",
    "password_label",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
)


def _json_stub(target_keys: tuple[str, ...]) -> str:
    """응답 JSON 예시를 생성한다."""
    lines = ['{', '    "coord_system": "relative_1000",']
    for idx, key in enumerate(target_keys):
        comma = "," if idx < len(target_keys) - 1 else ""
        lines.append(f'    "{key}": {{"x": ..., "y": ...}}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_login_rcs_ui_venus_prompt(
    width: int,
    height: int,
    target_keys: Iterable[str] | None = None,
) -> tuple[str, str]:
    """UI-Venus 용 grounding 중심 system/user 프롬프트를 구성한다."""
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_UI_VENUS_TARGET_KEYS
    missing = [key for key in keys if key not in UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS]
    if missing:
        raise ValueError(f"Unknown target keys for UI-Venus prompt: {missing}")

    system_message = (
        "GROUNDING task for a desktop GUI screenshot. "
        "Identify each requested visible UI element and return the point you would actually click "
        "to ground that element. "
        f"The screenshot is {width}x{height} pixels. "
        "Use coord_system='relative_1000' where x and y are integers from 0 to 1000. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "This screenshot shows a Windows 'Remote Control System' login dialog.",
        "For each requested target, visually identify the actual visible text or control first,",
        "then return the grounding point based on what a user would click in this screenshot.",
        "",
        "GROUNDING RULES:",
        "- Use only what is visibly present in the screenshot.",
        "- For title text and labels, ground the visible text itself.",
        "- For editable fields and the server combo box, ground the interactive area a user would click to focus or open the control.",
        "- For buttons, ground the clickable button surface.",
        "- Do not invent off-screen, hidden, or merged elements.",
        "- If an element is not visible enough to ground, omit that key.",
        "- Do not add explanation, markdown, comments, or extra fields.",
        "",
        f"Return click-grounding coordinates for these {len(keys)} elements:",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        lines.append(f'{idx}. "{key}" — {UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS[key]}')

    lines.extend(
        [
            "",
            f"Image dimensions: {width} x {height} pixels.",
            "Return the point that best grounds the visible element a user would click.",
            "Use coord_system='relative_1000'.",
            "x and y must be integers from 0 to 1000.",
            "",
            "Return ONLY this JSON:",
            _json_stub(keys),
        ]
    )
    return system_message, "\n".join(lines)
