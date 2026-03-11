"""OCR assist prompt builders for `poc.work2` pipeline."""

from typing import Iterable


def build_ocr_assist_prompt(
    width: int,
    height: int,
    context_label: str = "",
    focus_words: Iterable[str] | None = None,
    max_items: int = 12,
) -> tuple[str, str]:
    """Build OCR extraction prompt for the PaddleOCR stage."""
    focus_items = [word.strip() for word in (focus_words or ()) if word and word.strip()]
    context_text = context_label.strip() or "GUI screenshot"

    system_message = (
        "You are a precise OCR extraction assistant for software screenshots. "
        f"The image is {width}x{height} pixels. "
        "Read visible text faithfully. Respond ONLY with valid JSON."
    )

    lines = [
        f"Extract the most relevant visible text from this {context_text}.",
        "Focus on UI labels, tab names, button captions, field labels, window titles, and tool names.",
        "Do not invent text that is not visible.",
        "Prefer short, exact strings as they appear on screen.",
    ]

    if focus_items:
        lines.extend(
            [
                "",
                "Prioritize these target words if they are visible:",
                ", ".join(focus_items),
            ]
        )

    lines.extend(
        [
            "",
            f"Return at most {max_items} text items.",
            "Return ONLY this JSON:",
            "{",
            '  "texts": ["..."],',
            '  "focus_hits": ["..."]',
            "}",
        ]
    )

    return system_message, "\n".join(lines)
