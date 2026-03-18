"""RCS 로그인 화면 UI 요소 좌표 추출용 UI-TARS 전용 프롬프트 빌더.

UI-TARS 공식 문서는 grounding 용도로 단일 `Action:` 출력을 권장한다.
이 모듈은 그 규약을 따르되, batch 모드에서는 각 줄 앞에 실제 target key 를
붙여 여러 요소를 한 번에 요청할 수 있도록 보조 프롬프트를 제공한다.
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

UI_TARS_SINGLE_ACTION_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of a desktop application.\n"
    "You need to perform the next grounding action to complete the task.\n"
    "\n"
    "## Output Format\n"
    "Action: ...\n"
    "\n"
    "## Action Space\n"
    "click(point='x y')\n"
    "\n"
    "Rules:\n"
    "- Output exactly one action line.\n"
    "- Do not describe the screenshot.\n"
    "- Do not explain the coordinate space.\n"
    "- Do not output markdown or extra text."
)

UI_TARS_MULTI_ACTION_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of a desktop application.\n"
    "You need to identify multiple requested GUI elements and output one grounding action per detected element.\n"
    "\n"
    "## Output Format\n"
    "<target_key>: Action: click(point='x y')\n"
    "\n"
    "Rules:\n"
    "- Use the exact target_key provided by the user.\n"
    "- Never literally output the placeholder word element_name.\n"
    "- Do not describe the screenshot.\n"
    "- Do not explain the coordinate space.\n"
    "- If an element is not visible, omit it.\n"
    "- Do not output markdown or extra text."
)


def build_login_rcs_ui_tars_prompt(
    target_keys: Iterable[str] | None = None,
) -> tuple[str, str]:
    """UI-TARS 용 batch 요소 탐색 system/user 프롬프트를 구성한다."""
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_UI_TARS_TARGET_KEYS
    missing = [key for key in keys if key not in UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS]
    if missing:
        raise ValueError(f"Unknown target keys for UI-TARS prompt: {missing}")

    lines = [
        "This screenshot shows a Windows 'Remote Control System' login dialog.",
        "Locate the following GUI elements and report one click action per detected item.",
        "Use the exact target key before each action line.",
        "Do not literally output the word element_name.",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        desc = UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS[key]
        lines.append(f"{idx}. {key}: {desc}")

    lines.extend([
        "",
        "Output format (one line per detected element):",
        "<target_key>: Action: click(point='x y')",
        "Example:",
        "server_input: Action: click(point='344 182')",
    ])

    return UI_TARS_MULTI_ACTION_SYSTEM_PROMPT, "\n".join(lines)


def build_single_element_prompt(element_key: str) -> tuple[str, str]:
    """UI-TARS 공식 grounding 형식으로 단일 요소 좌표를 요청한다."""
    if element_key not in UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS:
        raise ValueError(f"Unknown element key: {element_key}")

    desc = UI_TARS_LOGIN_ELEMENT_DESCRIPTIONS[element_key]
    user_text = (
        "This screenshot shows a Windows 'Remote Control System' login dialog.\n"
        "Find the requested GUI element and click its center.\n"
        f"Target key: {element_key}\n"
        f"Target description: {desc}\n"
        "\n"
        "Return only:\n"
        "Action: click(point='x y')"
    )

    return UI_TARS_SINGLE_ACTION_SYSTEM_PROMPT, user_text
