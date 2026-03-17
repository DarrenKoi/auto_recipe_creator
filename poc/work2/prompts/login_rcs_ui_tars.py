"""RCS 로그인 화면 UI 요소 좌표 추출용 UI-TARS 전용 프롬프트 빌더.

UI-TARS-1.5 는 Qwen2.5-VL 기반 GUI agent 모델로,
일반 VLM 과 달리 `Thought: / Action: click(start_box='(x,y)')` 형식으로 출력한다.
좌표는 smart-resize 된 이미지 공간의 절대 픽셀값이다.
"""

from typing import Iterable


UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS: dict[str, str] = {
    "window_title_text": "the window title text 'Remote Control System' in the title bar at the top",
    "close_button": "the close button (X) at the top-right corner of the title bar",
    "server_label": "the text label 'Server' on the left side of the first form row",
    "server_input": "the dropdown/combobox input field next to the 'Server' label",
    "userid_label": "the text label 'User ID' on the left side of the second form row",
    "userid_input": "the text input field next to the 'User ID' label",
    "password_label": "the text label 'Password' on the left side of the third form row",
    "password_input": "the text input field next to the 'Password' label",
    "login_button": "the 'Log In' button at the bottom of the dialog",
    "cancel_button": "the 'Cancel' button at the bottom of the dialog",
    "shortcut_button": "the Korean text button (e.g. '바로가기 설정') at the bottom area",
}

DEFAULT_UI_TARS_TARGET_KEYS = (
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

GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of a desktop application.\n"
    "You need to identify the requested GUI elements and report their positions.\n"
    "For each element, output its name and position using the format:\n"
    "element_name: click(start_box='(x,y)')\n"
    "\n"
    "Rules:\n"
    "- Output one line per element.\n"
    "- Coordinates (x, y) are absolute pixel positions in the screenshot.\n"
    "- x=0 is the left edge, y=0 is the top edge.\n"
    "- If an element is not visible, skip it.\n"
    "- Do not output any other text, explanation, or markdown."
)


def build_login_rcs_ui_tars_prompt(
    target_keys: Iterable[str] | None = None,
) -> tuple[str, str]:
    """UI-TARS 용 RCS 로그인 화면 요소 탐색 system/user 프롬프트를 구성한다.

    UI-TARS-1.5-7B 는 별도의 system role 메시지를 지원하지 않으므로,
    시스템 지시사항을 user 텍스트 앞에 병합하여 반환한다.

    Returns:
        (system_message, user_text) 튜플. system_message 는 빈 문자열.
    """
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_UI_TARS_TARGET_KEYS
    missing = [key for key in keys if key not in UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS]
    if missing:
        raise ValueError(f"Unknown target keys for UI-TARS prompt: {missing}")

    lines = [
        GROUNDING_SYSTEM_PROMPT,
        "",
        "This screenshot shows a Windows 'Remote Control System' login dialog.",
        "Find the following GUI elements and report their center positions.",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        desc = UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS[key]
        lines.append(f"{idx}. {key}: {desc}")

    lines.extend([
        "",
        "Output format (one line per element):",
        "element_name: click(start_box='(x,y)')",
    ])

    return "", "\n".join(lines)


def build_single_element_prompt(element_key: str) -> tuple[str, str]:
    """UI-TARS GROUNDING 모드로 단일 요소 좌표를 요청하는 프롬프트를 구성한다.

    한 번에 하나의 요소만 찾도록 요청하여 정확도를 높인다.
    UI-TARS-1.5-7B 는 별도의 system role 을 지원하지 않으므로
    지시사항을 user 텍스트에 병합한다.
    """
    if element_key not in UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS:
        raise ValueError(f"Unknown element key: {element_key}")

    desc = UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS[element_key]
    user_text = (
        "You are a GUI agent. You are given a task and a screenshot.\n"
        "You need to find the requested GUI element and click on it.\n"
        "Output only: click(start_box='(x,y)') with absolute pixel coordinates.\n"
        f"\nClick on {desc}."
    )

    return "", user_text
