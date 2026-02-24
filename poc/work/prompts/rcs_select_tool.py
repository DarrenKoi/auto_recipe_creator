"""RCS List 탭에서 특정 툴 좌표 추출용 VLM 프롬프트 빌더."""


def build_rcs_select_tool_prompt(
    width: int,
    height: int,
    target_tool_name: str,
) -> tuple[str, str]:
    """Build prompts for locating one target tool row and click coordinate."""
    target_name = target_tool_name.strip()

    system_message = (
        "You are a precise GUI list row locator. "
        f"The image is {width}x{height} pixels. "
        "Respond ONLY with valid JSON."
    )

    lines = [
        "Locate the requested tool in this Remote Control System List-tab screenshot.",
        "",
        "The screenshot is already on the List tab.",
        "Each row has a tool name on the left and a status light on the right.",
        "",
        "Matching rules:",
        "- Use EXACT match only with the requested tool name.",
        "- If exact match does not exist, return found=false.",
        "- Keep tool-name text exactly as visible.",
        "",
        "Coordinate rules (critical):",
        "- Return one coordinate for DOUBLE-CLICK on the FIRST LETTER of the matched tool name.",
        "- Do not use status-light position.",
        "- x and y must be integer pixel values.",
        f"- x range: 0 to {width}. y range: 0 to {height}.",
        "",
        f"Requested tool name: {target_name!r}",
        "",
        "Return ONLY this JSON schema:",
        "{",
        '  "found": true,',
        '  "matched_name": "<tool name>",',
        '  "match_type": "exact|none",',
        '  "x": 0,',
        '  "y": 0,',
        '  "coord_anchor": "first_letter"',
        "}",
        "",
        "If not found, return:",
        "{",
        '  "found": false,',
        '  "matched_name": "",',
        '  "match_type": "none",',
        '  "coord_anchor": "first_letter"',
        "}",
    ]

    return system_message, "\n".join(lines)
