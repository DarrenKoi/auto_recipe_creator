"""RCS 로그인 화면 UI 요소 좌표 추출용 VLM 프롬프트 빌더.

UI-specialized VLM (ui-venus, mai-ui 등)에 최적화된 프롬프트를 구성한다.
각 UI 요소의 시각적 중심(center point)을 정수 좌표로 반환하도록 유도한다.
"""

from typing import Iterable


RCS_LOGIN_TARGET_SPECS = {
    "server_label": (
        "TEXT LABEL — The static text 'Server' displayed on the left side of the first form row. "
        "Return the center of this label text."
    ),
    "server_input": (
        "COMBOBOX / DROPDOWN — The server selection control next to the 'Server' label. "
        "This is a white rectangular area with a small dropdown arrow (▼) on its right edge. "
        "Return a safe click point inside the usable middle area of this dropdown control."
    ),
    "userid_label": (
        "TEXT LABEL — The static text 'User ID' displayed on the left side of the second form row. "
        "Return the center of this label text."
    ),
    "userid_input": (
        "TEXT INPUT — The editable text field next to the 'User ID' label. "
        "This is a white rectangular input area with a thin border. "
        "Return a safe click point inside the editable area, away from the border."
    ),
    "password_label": (
        "TEXT LABEL — The static text 'Password' displayed on the left side of the third form row. "
        "Return the center of this label text."
    ),
    "password_input": (
        "TEXT INPUT — The editable text field next to the 'Password' label. "
        "This is a white rectangular input area with a thin border. "
        "Return a safe click point inside the editable area, away from the border."
    ),
    "login_button": (
        "BUTTON — The button labeled 'Log In' at the bottom of the dialog. "
        "It has raised 3D borders in Windows classic style. "
        "Return a safe click point inside this button, away from the border."
    ),
    "cancel_button": (
        "BUTTON — The button labeled 'Cancel' at the bottom of the dialog. "
        "It has raised 3D borders in Windows classic style. "
        "Return a safe click point inside this button, away from the border."
    ),
    "shortcut_button": (
        "BUTTON — A button with Korean text (e.g. '바로가기 설정') in the bottom area. "
        "It may be positioned separately from the Log In and Cancel buttons. "
        "Return a safe click point inside this button, away from the border."
    ),
}

DEFAULT_RCS_LOGIN_TARGET_KEYS = (
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


def build_rcs_login_locator_prompt(
    width: int,
    height: int,
    target_keys: Iterable[str] | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """RCS 로그인 화면 UI 요소 좌표 추출용 system/user 프롬프트를 구성한다.

    UI-specialized VLM 에 최적화: 각 요소의 시각적 중심 좌표를 정수로 반환하도록 유도한다.
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
        "Return click-safe coordinates using coord_system='relative_1000'. "
        "In this coordinate system, x and y are integers from 0 to 1000 relative to the image. "
        "Respond ONLY with valid JSON — no explanation, no markdown fences."
    )

    lines = [
        "This screenshot shows a Windows 'Remote Control System' login dialog.",
        "",
        "DIALOG STRUCTURE:",
        "- Three labeled form rows arranged vertically:",
        "  Row 1: 'Server' label (left) + combobox/dropdown (right, white area with ▼ arrow)",
        "  Row 2: 'User ID' label (left) + text input field (right, white editable area)",
        "  Row 3: 'Password' label (left) + text input field (right, white editable area)",
        "- Buttons below the form rows (e.g. 'Log In', 'Cancel', Korean text button)",
        "",
        "VISUAL CUES:",
        "- Labels are static text on a gray dialog background.",
        "- Input fields and the combobox are white rectangular areas with thin borders.",
        "- The Server combobox has a dropdown arrow (▼) on its right edge.",
        "- Buttons have raised 3D borders typical of Windows classic style.",
        "",
        f"Find the CLICK-SAFE coordinates of these {len(keys)} UI elements:",
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
            "Return a safe click point inside each control, not on the border.",
            "",
            "Return ONLY this JSON (all coordinate values must be integers):",
            _json_stub(keys),
        ]
    )

    return system_message, "\n".join(lines)
