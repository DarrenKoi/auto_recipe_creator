"""RCS 메인 화면 View/List 탭 좌표 추출용 VLM 프롬프트 빌더."""

from typing import Iterable


RCS_MAIN_TAB_TARGET_SPECS = {
    "view_tab": (
        "TAB. In the top-left tab strip, locate the first letter 'V' of the 'View' tab label. "
        "Use 'V' as the anchor to identify the tab, then return a safe click point inside the View tab area."
    ),
    "list_tab": (
        "TAB. In the top-left tab strip, locate the first letter 'L' of the 'List' tab label. "
        "Use 'L' as the anchor to identify the tab, then return a safe click point inside the List tab area."
    ),
}

DEFAULT_RCS_MAIN_TAB_TARGET_KEYS = (
    "view_tab",
    "list_tab",
)


def _json_stub(target_keys: tuple[str, ...]) -> str:
    lines = ['{', '    "coord_system": "relative_1000",']
    for idx, key in enumerate(target_keys):
        comma = "," if idx < len(target_keys) - 1 else ""
        lines.append(f'    "{key}": {{"x": ..., "y": ...}}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_rcs_main_tab_locator_prompt(
    width: int,
    height: int,
    target_keys: Iterable[str] | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Build system/user prompts for RCS main window tab coordinate extraction."""
    keys = tuple(target_keys) if target_keys is not None else DEFAULT_RCS_MAIN_TAB_TARGET_KEYS
    missing = [key for key in keys if key not in RCS_MAIN_TAB_TARGET_SPECS]
    if missing:
        raise ValueError(f"Unknown target keys for prompt: {missing}")

    system_message = (
        "You are a precise GUI element locator. "
        f"The image is {width}x{height} pixels. "
        "The origin (0, 0) is the top-left corner of the image. "
        "Return click-safe coordinates using coord_system='relative_1000'. "
        "In this coordinate system, x and y are integers from 0 to 1000 relative to the image. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Locate tab elements in this Remote Control System main window.",
        "",
        "The window has 'View' and 'List' tabs near the top-left corner.",
        "They are adjacent in the same tab strip.",
        "",
        "IMPORTANT — First-letter anchoring strategy:",
        "To locate each tab, first find its distinctive first letter in the tab strip:",
        "  - 'V' -> identifies the 'View' tab",
        "  - 'L' -> identifies the 'List' tab",
        "Once you find the first letter, use it to determine the full tab boundary,",
        "then return the center point of that tab label.",
        "Ignore similar words elsewhere in the window.",
        "",
        f"Find the click-safe coordinates of these {len(keys)} elements:",
        "",
    ]
    for idx, key in enumerate(keys, start=1):
        lines.append(f'{idx}. "{key}" — {RCS_MAIN_TAB_TARGET_SPECS[key]}')

    if extra_instructions:
        lines.append("")
        lines.append("Additional instructions:")
        for instruction in extra_instructions:
            lines.append(f"- {instruction}")

    lines.extend(
        [
            "",
            f"Image size: {width} x {height} pixels.",
            "Use coord_system='relative_1000'.",
            "x and y must be integers from 0 to 1000.",
            "0 means the left/top edge. 1000 means the right/bottom edge.",
            "Return a safe click point inside each tab area, not on the border.",
            "",
            "Return ONLY this JSON (all values are integers):",
            _json_stub(keys),
        ]
    )

    return system_message, "\n".join(lines)
