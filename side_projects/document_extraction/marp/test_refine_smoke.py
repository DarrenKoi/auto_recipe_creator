"""refine 스모크 테스트 (검증 로직 + 오프라인/주입 llm_call, 서버 불필요).

실행:
    uv run python -m side_projects.document_extraction.marp.test_refine_smoke
"""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.marp.generate import frontmatter_for_theme
from side_projects.document_extraction.marp.refine import (
    join_deck,
    refine_deck,
    split_deck,
    validate_refined_slide,
)


_SLIDE = (
    "# Q2 Setup\n\n- automation improved\n- setup faster\n\n"
    "| mode | min |\n| --- | --- |\n| manual | 30 |\n| AI | 18 |\n\n"
    "$$ x = a + b $$\n\n![w:600](crops/c001.jpg)"
)


def test_validate_accepts_structural_change() -> None:
    refined = (
        "# Q2 Setup\n\n## 개선 효과\n\n- **automation improved**\n"
        "  - setup faster\n\n"
        "| mode | min |\n| --- | --- |\n| manual | 30 |\n| AI | 18 |\n\n"
        "$$ x = a + b $$\n\n![w:600](crops/c001.jpg)"
    )
    ok, reasons = validate_refined_slide(_SLIDE, refined)
    assert ok, reasons
    print("[PASS] test_validate_accepts_structural_change")


def test_validate_rejects_new_number() -> None:
    refined = _SLIDE + "\n- improved by 42%"
    ok, reasons = validate_refined_slide(_SLIDE, refined)
    assert not ok and any("42" in r for r in reasons), reasons
    print("[PASS] test_validate_rejects_new_number")


def test_validate_rejects_lost_table_row() -> None:
    refined = _SLIDE.replace("| manual | 30 |\n", "")
    ok, reasons = validate_refined_slide(_SLIDE, refined)
    assert not ok and any("표 행" in r for r in reasons), reasons
    print("[PASS] test_validate_rejects_lost_table_row")


def test_validate_rejects_lost_formula_and_image() -> None:
    no_formula = _SLIDE.replace("$$ x = a + b $$\n\n", "")
    ok, reasons = validate_refined_slide(_SLIDE, no_formula)
    assert not ok and any("수식" in r for r in reasons), reasons

    no_image = _SLIDE.replace("![w:600](crops/c001.jpg)", "")
    ok, reasons = validate_refined_slide(_SLIDE, no_image)
    assert not ok and any("이미지" in r for r in reasons), reasons
    print("[PASS] test_validate_rejects_lost_formula_and_image")


def test_validate_rejects_empty() -> None:
    ok, reasons = validate_refined_slide(_SLIDE, "   ")
    assert not ok
    print("[PASS] test_validate_rejects_empty")


def test_split_join_roundtrip() -> None:
    front = frontmatter_for_theme("doc-restore")
    deck = front + "\n" + _SLIDE + "\n\n---\n\n# Slide 2\n\n- b\n"
    f, slides = split_deck(deck)
    assert f == front, (f, front)
    assert len(slides) == 2 and slides[1].startswith("# Slide 2")
    rejoined = join_deck(f, slides)
    f2, slides2 = split_deck(rejoined)
    assert slides2 == slides
    print("[PASS] test_split_join_roundtrip")


def test_refine_deck_offline_passthrough() -> None:
    deck = frontmatter_for_theme() + "\n" + _SLIDE + "\n"
    out, adopted = refine_deck(deck, offline=True)
    assert out == deck and adopted == 0
    print("[PASS] test_refine_deck_offline_passthrough")


def test_refine_deck_adopts_valid_and_keeps_invalid() -> None:
    deck = (frontmatter_for_theme() + "\n" + _SLIDE
            + "\n\n---\n\n# Slide 2\n\n- keep me\n")

    def fake_llm(system: str, user: str) -> str:
        slide = user.split("SLIDE MARKDOWN:\n", 1)[1]
        if slide.startswith("# Q2 Setup"):
            # 유효한 다듬기: 강조만 추가
            return slide.replace("- automation improved", "- **automation improved**")
        # 무효한 다듬기: 새 숫자 창작 -> 기각돼야 함
        return slide + "\n- fabricated 99"

    out, adopted = refine_deck(deck, llm_call=fake_llm, offline=False)
    assert adopted == 1, adopted
    assert "**automation improved**" in out
    assert "fabricated 99" not in out
    assert "- keep me" in out  # 기각된 슬라이드는 원본 유지
    print("[PASS] test_refine_deck_adopts_valid_and_keeps_invalid")


def test_refine_deck_skips_raster_slide() -> None:
    deck = (frontmatter_for_theme() + "\n![bg fit](cap/page_001.webp)"
            + "\n\n---\n\n" + _SLIDE + "\n")
    calls = []

    def fake_llm(system: str, user: str) -> str:
        calls.append(user)
        return user.split("SLIDE MARKDOWN:\n", 1)[1]

    out, adopted = refine_deck(deck, llm_call=fake_llm, offline=False)
    assert len(calls) == 1, "래스터 강등 슬라이드는 LLM 호출 없이 보존돼야 함"
    assert "![bg fit](cap/page_001.webp)" in out
    print("[PASS] test_refine_deck_skips_raster_slide")


def main() -> int:
    test_validate_accepts_structural_change()
    test_validate_rejects_new_number()
    test_validate_rejects_lost_table_row()
    test_validate_rejects_lost_formula_and_image()
    test_validate_rejects_empty()
    test_split_join_roundtrip()
    test_refine_deck_offline_passthrough()
    test_refine_deck_adopts_valid_and_keeps_invalid()
    test_refine_deck_skips_raster_slide()
    print("\n[INFO] 모든 refine 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
