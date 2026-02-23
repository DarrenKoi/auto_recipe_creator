"""Prompt builders for VLM-based GUI element localization."""

from __future__ import annotations

from typing import Iterable


RCS_LOGIN_TARGET_SPECS = {
    "server_label": "TEXT LABEL. Find the first letter 'S' in 'Server' and return its center.",
    "server_input": (
        "INPUT FIELD. Find the left-most vertical edge of the white Server combobox area. "
        "(A +50px x-shift is applied later to hit the arrow reliably.)"
    ),
    "userid_label": "TEXT LABEL. Find the first letter 'U' in 'User ID' and return its center.",
    "userid_input": (
        "INPUT FIELD. Find the left-most vertical edge of the white field next to 'User ID'."
    ),
    "password_label": "TEXT LABEL. Find the first letter 'P' in 'Password' and return its center.",
    "password_input": (
        "INPUT FIELD. Find the left-most vertical edge of the white field next to 'Password'."
    ),
    "login_button": "BUTTON. Find the left-most edge of the 'Log In' button.",
    "cancel_button": "BUTTON. Find the left-most edge of the 'Cancel' button.",
    "shortcut_button": "BUTTON. Find the left-most edge of the Korean text button.",
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
    lines = ["{"]
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
    """Build system/user prompts for RCS login UI coordinate extraction."""
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_RCS_LOGIN_TARGET_KEYS
    missing = [key for key in keys if key not in RCS_LOGIN_TARGET_SPECS]
    if missing:
        raise ValueError(f"Unknown target keys for prompt: {missing}")

    system_message = (
        "You are a precise GUI element locator. "
        f"The image is {width}x{height} pixels. "
        "The origin (0, 0) is the top-left corner of the image. "
        "Return coordinates as integer pixel values. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Locate GUI elements in this Remote Control System login dialog.",
        "",
        "The dialog has three labeled rows and three buttons.",
        "",
        f"Find the pixel coordinates of these {len(keys)} elements:",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        lines.append(f'{idx}. "{key}" — {RCS_LOGIN_TARGET_SPECS[key]}')

    if extra_instructions:
        lines.append("")
        lines.append("Additional instructions:")
        for instruction in extra_instructions:
            lines.append(f"- {instruction}")

    lines.extend(
        [
            "",
            f"Image size: {width} x {height} pixels.",
            f"x range: 0 (left edge) to {width} (right edge).",
            f"y range: 0 (top edge) to {height} (bottom edge).",
            "",
            "Return ONLY this JSON (all values are integers):",
            _json_stub(keys),
        ]
    )

    return system_message, "\n".join(lines)

