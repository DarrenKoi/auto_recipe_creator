"""retrieval(검색 + quality gate) 스모크 테스트 (순수 함수, 서버 불필요).

실행:
    uv run python -m side_projects.document_extraction.extraction.test_retrieval_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction import retrieval
from side_projects.document_extraction.extraction.search import run_search


def _chunk(**kw) -> dict:
    base = {
        "chunk_id": "c1", "source_image": "captures/s1.webp", "screenshot_index": 1,
        "source_type": "powerpoint", "document_id": "doc1", "region_type": "region_text",
        "content": "recipe setup automation improved yield", "parent_heading": "Section A",
        "confidence": 0.85, "review_status": "approved",
    }
    base.update(kw)
    # embedding_text 는 실제 chunk 처럼 content/heading 에서 일관되게 유도
    base.setdefault(
        "embedding_text",
        f"Heading: {base['parent_heading']}\nContent: {base['content']}",
    )
    return base


def test_quality_gate_trusted_vs_lower() -> None:
    trusted, reasons = retrieval.quality_gate(_chunk())
    assert trusted == "trusted", reasons

    # 저신뢰 + 미승인 -> lower_trust
    tier, reasons = retrieval.quality_gate(_chunk(confidence=0.3, review_status="needs_review"))
    assert tier == "lower_trust" and "low confidence and not approved" in reasons

    # 빈 content + source_image 없음 + unknown type
    tier, reasons = retrieval.quality_gate(
        {"region_type": "bogus", "content": "", "source_image": "", "confidence": 0.9}
    )
    assert tier == "lower_trust"
    assert "empty content" in reasons
    assert "missing source_image" in reasons
    assert any("unknown region_type" in r for r in reasons)
    print("[PASS] test_quality_gate_trusted_vs_lower")


def test_keyword_search_and_filters() -> None:
    chunks = [
        _chunk(chunk_id="c1", content="recipe setup automation"),
        _chunk(chunk_id="c2", content="unrelated weather report", parent_heading="Weather"),
        _chunk(chunk_id="c3", region_type="table_summary",
               content="Table recipe timing. Columns: step, sec.",
               parent_heading="recipe timing"),
    ]
    # AND 매칭: "recipe" 는 c1, c3
    hits = retrieval.keyword_search(chunks, "recipe")
    ids = {h["chunk"]["chunk_id"] for h in hits}
    assert ids == {"c1", "c3"}, ids
    # heading 매칭 가중: c3(heading 에 recipe) 가 c1 보다 상위
    assert hits[0]["chunk"]["chunk_id"] == "c3", [h["chunk"]["chunk_id"] for h in hits]

    # 필터: region_type=table_summary -> c3 만
    hits = retrieval.keyword_search(chunks, "recipe", filters={"region_type": "table_summary"})
    assert {h["chunk"]["chunk_id"] for h in hits} == {"c3"}

    # AND 미스: 두 토큰 모두 있어야
    hits = retrieval.keyword_search(chunks, "recipe weather")
    assert hits == []
    print("[PASS] test_keyword_search_and_filters")


def test_tier_filter() -> None:
    chunks = [
        _chunk(chunk_id="good", confidence=0.9, review_status="approved"),
        _chunk(chunk_id="weak", confidence=0.2, review_status="needs_review"),
    ]
    trusted = retrieval.keyword_search(chunks, "recipe", tier="trusted")
    assert {h["chunk"]["chunk_id"] for h in trusted} == {"good"}
    lower = retrieval.keyword_search(chunks, "recipe", tier="lower_trust")
    assert {h["chunk"]["chunk_id"] for h in lower} == {"weak"}
    print("[PASS] test_tier_filter")


def test_malformed_confidence_does_not_crash() -> None:
    """비정상 confidence(문자열)가 섞여도 검색이 죽지 않는다."""
    chunks = [
        _chunk(chunk_id="bad", confidence="high"),
        _chunk(chunk_id="ok", confidence=0.9),
    ]
    hits = retrieval.keyword_search(chunks, "recipe")  # sort 키에서 crash 안 나야
    assert {h["chunk"]["chunk_id"] for h in hits} == {"bad", "ok"}
    # bad 는 confidence 강제 0.0 -> trusted 게이트 실패(approved 라 통과할 수도)
    print("[PASS] test_malformed_confidence_does_not_crash")


def test_run_search_over_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rag_chunks.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for c in [_chunk(chunk_id="c1"),
                      _chunk(chunk_id="c2", content="recipe setup details")]:
                fp.write(json.dumps(c, ensure_ascii=False) + "\n")
        n = run_search(path, "recipe setup", {}, "all", 20)
        assert n == 2, n
    print("[PASS] test_run_search_over_jsonl")


def main() -> int:
    test_quality_gate_trusted_vs_lower()
    test_keyword_search_and_filters()
    test_tier_filter()
    test_malformed_confidence_does_not_crash()
    test_run_search_over_jsonl()
    print("\n[INFO] 모든 retrieval 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
