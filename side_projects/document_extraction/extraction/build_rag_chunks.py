"""B1 빌드 엔트리: harvest 번들 -> rag_chunks.jsonl (loader -> structure -> chunkers).

embedding 없이도 keyword 검색 가능한 첫 RAG store 를 만든다(rag_db_plan 의 "첫 구현").
CLI 인자 없음 - 실행 전 아래 상수를 채운다.

    uv run python -m side_projects.document_extraction.extraction.build_rag_chunks
"""

from pathlib import Path

from side_projects.document_extraction.extraction.harvest_loader import load_bundle
from side_projects.document_extraction.extraction.structure import assign_structure
from side_projects.document_extraction.extraction.chunkers import chunk_bundle
from side_projects.document_extraction.extraction.rag_chunks import write_chunks_jsonl

# === 실행 전 채울 것 ==========================================================
INPUT_BUNDLE: Path = Path(r"")   # harvest 번들 루트 (예: D:\harvest\tool_manual)
OUTPUT_JSONL: Path = Path(r"")   # 예: D:\harvest\tool_manual\rag_chunks.jsonl
# =============================================================================


def build_chunks(bundle_root: Path) -> list:
    """번들 -> RagChunk 목록 (loader -> structure -> chunkers)."""
    bundle = load_bundle(bundle_root)
    assign_structure(bundle)
    chunks = chunk_bundle(bundle)
    warnings = sum(len(p.load_warnings) for p in bundle.pages)
    if warnings:
        print(f"[WARNING] 로드 경고 {warnings}건 (페이지 load_warnings 참조)")
    return chunks


def build_to_jsonl(bundle_root: Path, out_path: Path) -> int:
    """번들 -> rag_chunks.jsonl. 기록한 chunk 수 반환."""
    chunks = build_chunks(bundle_root)
    n = write_chunks_jsonl(chunks, Path(out_path))
    print(f"[INFO] chunk {n}건 -> {out_path}")
    return n


def main() -> None:
    if not str(INPUT_BUNDLE).strip() or not str(OUTPUT_JSONL).strip():
        raise SystemExit("[ERROR] INPUT_BUNDLE / OUTPUT_JSONL 를 채워 주세요.")
    build_to_jsonl(INPUT_BUNDLE, OUTPUT_JSONL)


if __name__ == "__main__":
    main()
