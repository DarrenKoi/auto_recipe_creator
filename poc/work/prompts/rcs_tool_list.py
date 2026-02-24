"""RCS List 탭의 툴 이름/상태/좌표 추출용 VLM 프롬프트 빌더."""

from typing import Iterable


def build_rcs_tool_list_reader_prompt(
    width: int,
    height: int,
    extra_instructions: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Build prompts for reading tool names, statuses, and row coordinates."""
    system_message = (
        "You are a precise GUI list and coordinate reader. "
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
        "Tool-name transcription rules (critical):",
        "- Copy each tool name exactly as visible.",
        "- Preserve leading digits and symbols in the name.",
        "- Never drop numeric prefixes even if they look like row numbers.",
        "- Keep punctuation/spaces as-is (for example: '.', '-', '_', '/', '()').",
        "- If a name starts with a number, that number must remain in `name`.",
        "",
        "Examples:",
        '- visible: "1ETCH_MAIN" -> name: "1ETCH_MAIN"',
        '- visible: "02-CD_MEASURE" -> name: "02-CD_MEASURE"',
        '- visible: "3.INSPECT TOOL" -> name: "3.INSPECT TOOL"',
        '- do not return: "ETCH_MAIN", "CD_MEASURE", "INSPECT TOOL"',
        "",
        "Coordinate rules (critical):",
        "- Return one click coordinate per tool row in image pixel space.",
        "- Use a click point on the FIRST LETTER of the tool name (coord_anchor='first_letter').",
        "- Never use status-light position as the click coordinate.",
        "- Never use a point from the rest of the tool name text.",
        "- The first letter means the leftmost visible character of the tool name text.",
        f"- x range: 0 to {width}. y range: 0 to {height}.",
        "- x and y must be integers.",
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
            '    {"name": "<tool name>", "status": "on|off", "indicator_color": "green|black", "x": 0, "y": 0, "coord_anchor": "first_letter"}',
            "  ]",
            "}",
        ]
    )

    return system_message, "\n".join(lines)
