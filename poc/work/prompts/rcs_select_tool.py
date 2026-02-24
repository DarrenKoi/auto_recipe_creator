"""RCS List 탭에서 특정 툴 좌표 추출용 VLM 프롬프트 빌더."""


def build_rcs_select_tool_prompt(
    width: int,
    height: int,
    target_tool_name: str,
) -> tuple[str, str]:
    """Build prompts for locating one target tool row and click coordinate."""
    target_name = target_tool_name.strip()

    system_message = (
        "You locate one tool row in a GUI list screenshot. "
        f"Image size is {width}x{height}. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Find exactly one target tool row in the list:",
        f"- target_tool_name: {target_name!r}",
        "",
        "The row target is the COMBINED visual unit of:",
        "- the colored box/marker for that row",
        "- and the exact tool name text in the same row",
        "Treat these two as one object and return one point for that combined object.",
        "Preferred point: center of the combined color-box + tool-name region.",
        "Use exact text match only.",
        "x and y must be integer pixel coordinates within the image.",
        "",
        "Return JSON only:",
        "{",
        '  "found": true,',
        '  "matched_name": "<tool name>",',
        '  "match_type": "exact|none",',
        '  "x": 0,',
        '  "y": 0,',
        '  "coord_anchor": "name_color_box_center"',
        "}",
        "",
        "If not found, return:",
        "{",
        '  "found": false,',
        '  "matched_name": "",',
        '  "match_type": "none",',
        '  "coord_anchor": "name_color_box_center"',
        "}",
    ]

    return system_message, "\n".join(lines)
