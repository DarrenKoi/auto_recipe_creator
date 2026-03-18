"""RCS 로그인 화면 UI-Venus 전용 grounding 프롬프트 빌더.

UI-Venus 1.5 공식 grounding 프롬프트 형식을 따른다.
한 번에 하나의 요소만 요청하고, 모델이 [x, y] (0-1000) 좌표를 반환한다.
요소가 보이지 않으면 [-1, -1] 을 반환한다.
"""

from typing import Iterable

# 로그인 대화상자 UI 요소 설명 — 단일 요소 프롬프트와 레거시 batch 프롬프트 공용
UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS: dict[str, str] = {
    "window_title_text": "the visible window title text 'Remote Control System' in the title bar",
    "close_button": "the standard Windows close button with an 'X' in the title bar",
    "server_label": "the visible text label 'Server' in the first form row",
    "server_input": "the server dropdown or combo box control in the first form row",
    "userid_label": "the visible text label 'User ID' in the second form row",
    "userid_input": "the editable text field next to the 'User ID' label where a user would click to type their user ID",
    "password_label": "the visible text label 'Password' in the third form row",
    "password_input": "the editable password field in the third form row",
    "login_button": "the 'Log In' button near the bottom of the dialog",
    "cancel_button": "the 'Cancel' button near the bottom of the dialog",
    "shortcut_button": "the Korean text button in the bottom area (for example '바로가기 설정')",
}

# ---------------------------------------------------------------------------
# 공식 UI-Venus 1.5 단일 요소 grounding 프롬프트
# ---------------------------------------------------------------------------

UI_VENUS_OFFICIAL_PROMPT_TEMPLATE = (
    "Output the center point of the position corresponding to the following instruction: "
    "{instruction}. "
    "The output should just be the coordinates of a point, in the format [x,y]. "
    "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
    "the output should be [-1,-1]."
)


def build_ui_venus_single_element_prompt(
    instruction: str,
) -> tuple[str, str]:
    """UI-Venus 1.5 공식 grounding 형식으로 단일 요소 좌표를 요청한다.

    Args:
        instruction: 찾을 요소에 대한 자연어 설명.

    Returns:
        (system_message, user_message) 튜플.
        system_message 는 빈 문자열 — 공식 형식은 user message 만 사용한다.
    """
    user_text = UI_VENUS_OFFICIAL_PROMPT_TEMPLATE.format(instruction=instruction)
    return "", user_text


def build_ui_venus_single_element_prompt_by_key(
    element_key: str,
) -> tuple[str, str]:
    """element_key 를 사용해 공식 단일 요소 프롬프트를 구성한다."""
    if element_key not in UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS:
        raise ValueError(f"Unknown element key: {element_key}")

    instruction = UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS[element_key]
    return build_ui_venus_single_element_prompt(instruction)


# ---------------------------------------------------------------------------
# 레거시: 여러 요소를 한 번에 요청하는 batch 프롬프트 (참고용 보존)
# ---------------------------------------------------------------------------


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
    """UI-Venus 용 batch grounding system/user 프롬프트를 구성한다.

    NOTE: 공식 UI-Venus 1.5 형식은 단일 요소 프롬프트이다.
    이 함수는 레거시 호환용으로 보존한다.
    새 코드는 build_ui_venus_single_element_prompt 를 사용할 것.
    """
    keys = tuple(target_keys) if target_keys is not None else tuple(UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS)
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
