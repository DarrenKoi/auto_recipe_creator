"""RCS 로그인 화면 UI 요소 좌표 추출용 VLM 프롬프트 빌더.

UI-specialized VLM (ui-venus, mai-ui 등)에 최적화된 프롬프트를 구성한다.
레이블 텍스트는 중심점, 입력 컨트롤은 좌측 내부 click-safe point 위주로 반환하도록 유도한다.
"""

from typing import Iterable


RCS_LOGIN_TARGET_SPECS = {
    "window_title_text": (
        "TITLE TEXT — The visible window title text in the title bar near the top-left area. "
        "It starts with 'Remote Control System'. "
        "Return the center of the full visible bounding box of the title text only. "
        "Do not return the icon area or the form controls below."
    ),
    "close_button": (
        "TITLE BAR BUTTON — The standard Windows close button with an 'X' mark at the top-right corner "
        "of the title bar. Return the center of the full clickable close-button rectangle. "
        "Do not return the nearby border or title-bar background."
    ),
    "server_label": (
        "TEXT LABEL — The static text 'Server' displayed on the left side of the first form row. "
        "Return the center of the full bounding box of the visible label text only. "
        "Do not return the combobox area, and do not bias toward the top of the letters."
    ),
    "server_input": (
        "COMBOBOX / DROPDOWN — The server selection control next to the 'Server' label. "
        "This is a white rectangular area with a small dropdown arrow (▼) on its right edge. "
        "Return a safe click point in the left inner text/value area of the combobox, where a user would click "
        "to focus the current value area. Use a point around 20-30% of the control width from the left edge "
        "and around 58-65% of the control height from the top edge. "
        "Do not return the dropdown arrow, the top highlight, the upper border, or the adjacent 'Server' label text."
    ),
    "userid_label": (
        "TEXT LABEL — The static text 'User ID' displayed on the left side of the second form row. "
        "Return the center of the full bounding box of the visible label text only. "
        "Do not return the editable field, and do not bias toward the top of the letters."
    ),
    "userid_input": (
        "TEXT INPUT — The editable text field next to the 'User ID' label. "
        "This is a white rectangular input area with a thin border. "
        "Return a safe click point in the left inner typing area, near where typed text would begin. "
        "Use a point around 12-20% of the control width from the left edge and around 58-65% of the control "
        "height from the top edge. Do not return the top border, the upper highlight, or the adjacent "
        "'User ID' label text."
    ),
    "password_label": (
        "TEXT LABEL — The static text 'Password' displayed on the left side of the third form row. "
        "Return the center of the full bounding box of the visible label text only. "
        "Do not return the password field, and do not bias toward the top of the letters."
    ),
    "password_input": (
        "TEXT INPUT — The editable text field next to the 'Password' label. "
        "This is a white rectangular input area with a thin border. "
        "Return a safe click point in the left inner typing area, near where typed text would begin. "
        "Use a point around 12-20% of the control width from the left edge and around 58-65% of the control "
        "height from the top edge. Do not return the top border, the upper highlight, or the adjacent "
        "'Password' label text."
    ),
    "login_button": (
        "BUTTON — The button labeled 'Log In' at the bottom of the dialog. "
        "It has raised 3D borders in Windows classic style. "
        "Return a safe click point at the geometric center of the full clickable button rectangle. "
        "Do not return the text baseline, top highlight, or button border."
    ),
    "cancel_button": (
        "BUTTON — The button labeled 'Cancel' at the bottom of the dialog. "
        "It has raised 3D borders in Windows classic style. "
        "Return a safe click point at the geometric center of the full clickable button rectangle. "
        "Do not return the text baseline, top highlight, or button border."
    ),
    "shortcut_button": (
        "BUTTON — A button with Korean text (e.g. '바로가기 설정') in the bottom area. "
        "It may be positioned separately from the Log In and Cancel buttons. "
        "Return a safe click point at the geometric center of the full clickable button rectangle. "
        "Do not return the text baseline, top highlight, or button border."
    ),
}

DEFAULT_RCS_LOGIN_TARGET_KEYS = (
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
    """응답 JSON 형태의 예시 스텁을 생성한다."""
    lines = ['{', '    "coord_system": "relative_1000",']
    for idx, key in enumerate(target_keys):
        comma = "," if idx < len(target_keys) - 1 else ""
        lines.append(f'    "{key}": {{"x": ..., "y": ...}}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_login_rcs_locator_prompt(
    width: int,
    height: int,
    target_keys: Iterable[str] | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """RCS 로그인 화면 UI 요소 좌표 추출용 system/user 프롬프트를 구성한다.

    UI-specialized VLM 에 최적화: 레이블 텍스트는 중심, 입력 컨트롤은 좌측 내부 click-safe point 로 유도한다.
    """
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_RCS_LOGIN_TARGET_KEYS
    missing = [key for key in keys if key not in RCS_LOGIN_TARGET_SPECS]
    if missing:
        raise ValueError(f"Unknown target keys for prompt: {missing}")

    system_message = (
        "You are a precise desktop GUI element locator "
        "specialized in Windows application interfaces. "
        f"The screenshot is {width}x{height} pixels. "
        "Coordinate origin (0, 0) is the top-left corner; "
        "x increases rightward, y increases downward. "
        "Return precise anchor coordinates using coord_system='relative_1000'. "
        "In this coordinate system, x and y are integers from 0 to 1000 relative to the image. "
        "Respond ONLY with valid JSON — no explanation, no markdown fences."
    )

    lines = [
        "This screenshot shows a Windows 'Remote Control System' login dialog.",
        "The screenshot contains the currently captured login dialog window, including its visible title bar and borders.",
        "Do not reason about the full desktop or the larger main window shown after login succeeds.",
        "All coordinates are relative to this full login dialog image only.",
        "",
        "DIALOG STRUCTURE:",
        "- Top title bar with visible title text and a standard close button at the top-right corner",
        "- Three labeled form rows arranged vertically:",
        "  Row 1: 'Server' label (left) + combobox/dropdown (right, white area with ▼ arrow)",
        "  Row 2: 'User ID' label (left) + text input field (right, white editable area)",
        "  Row 3: 'Password' label (left) + text input field (right, white editable area)",
        "- Buttons below the form rows (e.g. 'Log In', 'Cancel', Korean text button)",
        "",
        "VISUAL CUES:",
        "- The title bar is at the very top of the window and contains the window title text.",
        "- The close button is the small standard Windows button with an 'X' mark at the far top-right.",
        "- Labels are static text on a gray dialog background.",
        "- Input fields and the combobox are white rectangular areas with thin borders.",
        "- The Server combobox has a dropdown arrow (▼) on its right edge.",
        "- Buttons have raised 3D borders typical of Windows classic style.",
        "- For each row, the left-side label text and the right-side editable control must be treated as separate targets.",
        "- Measure coordinates on the full screenshot exactly as shown, including title bar and outer border.",
        "- For text labels and the title text, use the center of the full visible text bounding box, not the top of the letters.",
        "- For input fields and the combobox, use a left-inner click point where text would begin, not the first visible character and not the geometric center.",
        "- For buttons, use the center of the full clickable button rectangle, not the text baseline and not the border.",
        "- For interactive controls, avoid the top highlight, the caption baseline, the upper border, and the far-right arrow area.",
        "- Within each form row, the label anchor and the input anchor should lie on the same horizontal band; if uncertain, choose slightly lower rather than higher.",
        "",
        f"Find the required coordinates of these {len(keys)} UI elements:",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        lines.append(f'{idx}. "{key}" — {RCS_LOGIN_TARGET_SPECS[key]}')

    if extra_instructions:
        lines.append("")
        lines.append("ADDITIONAL CONTEXT:")
        for instruction in extra_instructions:
            lines.append(f"- {instruction}")

    lines.extend(
        [
            "",
            f"Image dimensions: {width} x {height} pixels.",
            "Use coord_system='relative_1000'.",
            "x and y must be integers from 0 to 1000.",
            "0 means the left/top edge. 1000 means the right/bottom edge.",
            "For text labels and the title text, return the center of the full visible text bounding box.",
            "For buttons, return the center of the full clickable button rectangle.",
            "For input fields and the combobox, return a safe left-inner click point, not a border point.",
            "For form rows, prefer a slightly lower y within the row rather than a higher y if uncertain.",
            "",
            "Return ONLY this JSON (all coordinate values must be integers):",
            _json_stub(keys),
        ]
    )

    return system_message, "\n".join(lines)
