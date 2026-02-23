"""RCS 메인 화면 View/List 탭 좌표 추출용 VLM 프롬프트 빌더."""

from typing import Iterable


RCS_MAIN_TAB_TARGET_SPECS = {
    "view_tab": (
        "TAB. In the top-left tab strip, find the center of the 'View' tab label. "
        "Prioritize the first letter 'V' as the anchor and place the point at the tab label center."
    ),
    "list_tab": (
        "TAB. In the top-left tab strip, find the center of the 'List' tab label. "
        "Prioritize the first letter 'L' as the anchor and place the point at the tab label center."
    ),
}

DEFAULT_RCS_MAIN_TAB_TARGET_KEYS = (
    "view_tab",
    "list_tab",
)


def _json_stub(target_keys: tuple[str, ...]) -> str:
    lines = ["{"]
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
        "Return coordinates as integer pixel values. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Locate tab elements in this Remote Control System main window.",
        "",
        "The window has 'View' and 'List' tabs near the top-left corner.",
        "They are adjacent in the same tab strip.",
        "Use the first letter anchors: 'V' for View, 'L' for List.",
        "Ignore similar words elsewhere in the window.",
        "",
        f"Find the pixel coordinates of these {len(keys)} elements:",
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
            f"x range: 0 (left edge) to {width} (right edge).",
            f"y range: 0 (top edge) to {height} (bottom edge).",
            "",
            "Return ONLY this JSON (all values are integers):",
            _json_stub(keys),
        ]
    )

    return system_message, "\n".join(lines)
