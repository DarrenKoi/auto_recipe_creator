"""RCS List 탭의 툴 이름/상태 추출용 VLM 프롬프트 빌더."""

from typing import Iterable


def build_rcs_tool_list_reader_prompt(
    width: int,
    height: int,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Build system/user prompts for reading tool names and on/off light states."""
    system_message = (
        "You are a precise GUI list reader. "
        f"The image is {width}x{height} pixels. "
        "Read only what is visible in the screenshot. "
        "Do not hallucinate missing rows. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Read the tool list from this Remote Control System main window.",
        "",
        "The active tab is List.",
        "Each visible row has a tool name on the left.",
        "On the right side of each row, there is one status light.",
        "Green light means running (status=on).",
        "Black light means off (status=off).",
        "Return rows in top-to-bottom order.",
        "Ignore tabs, buttons, and unrelated text outside the tool list rows.",
        "",
    ]

    if extra_instructions:
        lines.append("Additional instructions:")
        for instruction in extra_instructions:
            lines.append(f"- {instruction}")
        lines.append("")

    lines.extend(
        [
            f"Image size: {width} x {height} pixels.",
            "",
            "Return ONLY this JSON schema:",
            "{",
            '  "tools": [',
            '    {"name": "<tool name>", "status": "on|off", "indicator_color": "green|black"}',
            "  ]",
            "}",
        ]
    )

    return system_message, "\n".join(lines)
