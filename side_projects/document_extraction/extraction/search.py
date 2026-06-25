"""rag_chunks.jsonl keyword 검색 엔트리 (CLI 인자 없음).

extract_screenshot 이 만든 retrieval store(rag_chunks.jsonl)에 대해 keyword +
metadata 필터 검색을 실행하고, quality tier 와 provenance 를 함께 출력한다.
embedding 없이 동작하는 첫 단계 검색(rag_db_plan.md "첫 구현 계획" 5번).

실행 전 아래 상수를 수정:
    CHUNKS_JSONL  검색할 JSONL 경로
    QUERY         keyword (공백 = AND). 빈 문자열이면 필터/tier 브라우징
    FILTERS       metadata 필터 (빈 값은 무시)
    TIER          "all" | "trusted" | "lower_trust"

실행:
    uv run python -m side_projects.document_extraction.extraction.search
"""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction import retrieval


# === 실행 전 매번 채워 넣을 것 =================================================
CHUNKS_JSONL: Path = Path("")   # 예: Path(r"C:\...\_rag\rag_chunks.jsonl")
QUERY: str = "recipe setup"
FILTERS: dict = {
    # "source_type": "powerpoint",
    # "document_id": "doc001",
    # "region_type": "table_summary",
    # "review_status": "approved",
    # "min_confidence": 0.7,
}
TIER: str = "all"               # "all" | "trusted" | "lower_trust"
LIMIT: int = 20
# ==============================================================================


def run_search(chunks_jsonl: Path, query: str, filters: dict, tier: str, limit: int) -> int:
    """검색을 실행하고 결과를 출력한다. 반환: 매칭 개수."""
    chunks = retrieval.load_chunks(chunks_jsonl)
    part = retrieval.partition_by_tier(chunks)
    print(
        f"[INFO] 로드: {part['counts']['total']} chunk "
        f"(trusted={part['counts']['trusted']}, lower_trust={part['counts']['lower_trust']})"
    )

    hits = retrieval.keyword_search(
        chunks, query, filters=filters, tier=tier, limit=limit
    )
    print(f"[INFO] query={query!r} filters={filters} tier={tier} -> {len(hits)} hit")

    for rank, hit in enumerate(hits, start=1):
        chunk = hit["chunk"]
        content = str(chunk.get("content") or "")
        snippet = content[:120].replace("\n", " ")
        print(
            f"[INFO] #{rank} score={hit['score']} tier={hit['tier']} "
            f"type={chunk.get('region_type')} conf={chunk.get('confidence')}"
        )
        print(
            f"       src={chunk.get('source_image')} "
            f"idx={chunk.get('screenshot_index')} region={chunk.get('region_id')}"
        )
        print(f"       heading={chunk.get('parent_heading') or '(none)'}")
        print(f"       {snippet}")
        if hit["gate_reasons"]:
            print(f"       [gate] {', '.join(hit['gate_reasons'])}")
    return len(hits)


def main() -> int:
    if str(CHUNKS_JSONL) in {"", "."}:
        print("[ERROR] CHUNKS_JSONL 가 비어 있습니다. search.py 상단 상수를 수정하세요.")
        return 1
    path = CHUNKS_JSONL.expanduser().resolve()
    if not path.exists():
        print(f"[ERROR] JSONL 파일이 없습니다: {path}")
        return 1
    print(f"[INFO] CHUNKS_JSONL = {path}")
    try:
        run_search(path, QUERY, FILTERS, TIER, LIMIT)
    except Exception as exc:
        print(f"[ERROR] 검색 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
