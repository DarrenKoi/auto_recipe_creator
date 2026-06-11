"""CD-SEM 환경 마우스 커서 coarse 탐지 프롬프트 (vlm_cursor_click_filter 에서 이식).

DVR / RCS / RCS-on-SEM-Monitor 세 가지 커서 변형을 모두 인지하도록 설명한다.
"""


def cursor_system_prompt() -> str:
    """커서 coarse 탐지 시스템 프롬프트."""
    return (
        "You locate the mouse cursor inside a screenshot of CD-SEM tooling. "
        "The cursor can appear in one of three forms depending on which window the pointer is over:\n"
        "  1) DVR camera feed: a small black 'X' (crosshair) glyph, ~10-20 px on each side.\n"
        "  2) RCS application (default Windows pointer): a small black arrow with a thin white outline.\n"
        "  3) RCS SEM Monitor box (the dark live-SEM image area): the same arrow inverted to "
        "white with a thin black outline so it stays visible against the dark background.\n"
        "Return strict JSON only. Locate ONLY the mouse cursor; do not confuse it with similar-looking "
        "but static UI artifacts such as SEM crosshair reticles, alignment-key marks, toolbar icon glyphs, "
        "or measurement annotations. A real cursor sits on top of the underlying content, is small "
        "(typically 12-32 px on a side), and never has anti-aliased text or numbers attached to it. "
        "If no cursor is visible, say so."
    )


def cursor_user_prompt() -> str:
    """커서 coarse 탐지 사용자 프롬프트(JSON 스키마 지정)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "cursor_visible": true,\n'
        '  "cursor_kind": "dvr_x | rcs_black_arrow | rcs_white_arrow",\n'
        '  "coord_system": "relative_1000",\n'
        '  "cursor_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string describing the glyph and where it sits"\n'
        "}\n"
        "The bbox must tightly enclose the entire visible cursor glyph (X for dvr_x, "
        "the full arrow shape for rcs_black_arrow / rcs_white_arrow). "
        "Set cursor_kind to whichever of the three variants you actually see; if you cannot tell, "
        'set it to "unknown". '
        "If no cursor is visible, set cursor_visible=false, cursor_bbox=null, and cursor_kind=null."
    )
