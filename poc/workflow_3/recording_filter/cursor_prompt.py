"""CD-SEM 환경 마우스 커서 coarse 탐지 프롬프트 (vlm_cursor_click_filter 에서 이식).

DVR / RCS / RCS-on-SEM-Monitor 세 가지 커서 변형을 모두 인지하도록 설명한다.
"""


def cursor_system_prompt() -> str:
    """커서 coarse 탐지 시스템 프롬프트.

    (2026-08-12) 손 모양(pointing hand) 커서를 추가했다. 예전 프롬프트는 커서가
    "세 가지 형태 중 하나" 라고 못박고 크기를 12-32 px 로 한정해, 버튼/링크 위에서
    손 모양으로 바뀌는 프레임을 모델이 커서 아님으로 처리했다 - 오피스 실측에서
    탐지율이 약 50% 에 머문 원인이다. 형태를 열거하되 "표준 Windows 커서면 무엇이든"
    으로 열어 두는 편이, 못 보던 글리프를 침묵으로 버리는 것보다 낫다.
    """
    return (
        "You locate the mouse cursor inside a screenshot of CD-SEM tooling. "
        "The pointer takes whatever shape Windows is currently showing. Common forms here:\n"
        "  1) DVR camera feed: a small black 'X' (crosshair) glyph, ~10-20 px on each side.\n"
        "  2) RCS application (default Windows pointer): a small black arrow with a thin white outline.\n"
        "  3) RCS SEM Monitor box (the dark live-SEM image area): the same arrow inverted to "
        "white with a thin black outline so it stays visible against the dark background.\n"
        "  4) Pointing hand: over buttons and other clickable controls the arrow becomes a hand with "
        "the INDEX FINGER EXTENDED upward, the other fingers curled into a fist. This shape is often "
        "what is visible at the exact moment of a click.\n"
        "Any other standard Windows cursor may also appear (I-beam text caret, busy/hourglass or "
        "spinning ring, four-way move, or resize arrows). Report whichever one you see.\n"
        "\n"
        "CRITICAL - this window contains three FIXED graphics that are frequently mistaken for the "
        "cursor. All three are painted into the window itself: they appear in EVERY screenshot at the "
        "SAME place and they never move. None of them is ever the mouse cursor:\n"
        "  A) An OPEN-PALM hand icon (fingers spread, no single extended finger) used as a toolbar "
        "control, sitting between the 'Full Size' button and the live SEM image area.\n"
        "  B) The window's CLOSE BUTTON - an 'X' in the title bar at the top-right corner of the "
        "window, alongside the minimize/maximize buttons.\n"
        "  C) A '>' chevron mark in the TOP-LEFT corner INSIDE the live SEM image area.\n"
        "Never report A, B or C as the cursor.\n"
        "\n"
        "The hardest case: when the pointer moves to the top-right to reach the close button, the "
        "cursor is an ARROW sitting ON or NEXT TO that X. Report the arrow glyph itself - its tip and "
        "outline are distinct from the thin, centered X of the button - and give the arrow's bbox, "
        "NOT the button's. Do not answer with the palm icon (A) or the chevron (C) just because the "
        "pointer is hard to separate from the title bar; those are on the opposite side of the window "
        "from the pointer and are never the answer. When you genuinely cannot isolate the pointer, "
        "answer cursor_visible=false - that is always better than naming a fixed graphic.\n"
        "\n"
        "Return strict JSON only. Locate ONLY the mouse cursor; do not confuse it with similar-looking "
        "but static UI artifacts such as SEM crosshair reticles, alignment-key marks, toolbar icon glyphs, "
        "or measurement annotations. A real cursor sits on top of the underlying content, is small "
        "(typically 12-48 px on a side, hand and busy shapes at the larger end), and never has "
        "anti-aliased text or numbers attached to it. If no cursor is visible, say so."
    )


def cursor_user_prompt() -> str:
    """커서 coarse 탐지 사용자 프롬프트(JSON 스키마 지정)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "cursor_visible": true,\n'
        '  "cursor_kind": "dvr_x | rcs_black_arrow | rcs_white_arrow | hand | other",\n'
        '  "coord_system": "relative_1000",\n'
        '  "cursor_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string describing the glyph and where it sits"\n'
        "}\n"
        "The bbox must tightly enclose the entire visible cursor glyph (X for dvr_x, "
        "the full arrow shape for rcs_black_arrow / rcs_white_arrow, the whole hand including the "
        "extended finger for hand). "
        "Set cursor_kind to whichever variant you actually see; use \"other\" for any standard "
        "Windows cursor not listed (I-beam, busy, move, resize). Report the location even when the "
        "shape is unfamiliar - an unlisted shape is still a cursor. If you truly cannot tell which "
        'kind it is, set it to "unknown" but still return the bbox. '
        "If no cursor is visible, set cursor_visible=false, cursor_bbox=null, and cursor_kind=null."
    )
