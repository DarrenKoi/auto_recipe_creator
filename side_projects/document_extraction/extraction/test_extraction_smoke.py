"""추출 파이프라인 skeleton 스모크 테스트 (VLM 서버 불필요).

OFFLINE 모드로 파이프라인 골격 + merge + chunk 생성 + JSONL keyword 검색을
서버 없이 검증한다(rag_db_plan.md "첫 구현 계획" 5번: keyword retrieval smoke).

실행:
    uv run python -m side_projects.document_extraction.extraction.test_extraction_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction import merge, rag_chunks
from side_projects.document_extraction.extraction.extract_screenshot import extract_folder
from side_projects.document_extraction.extraction.schemas import ExtractionResult


def _make_synth_image(path: Path, size: tuple[int, int] = (1280, 720)) -> None:
    from PIL import Image

    Image.new("RGB", size, (245, 245, 245)).save(path, format="WEBP", quality=80)


def test_merge_pure_function() -> None:
    """merge_evidence 는 VLM 없이 순수 동작하고 충돌을 표시한다."""
    ocr = {
        "raw_text": "Q2 CD-SEM recipe setup improvement",
        "tables": [{"title": "Setup time", "header": ["mode", "min"], "rows": [["manual", "30"]]}],
        "charts": [],
        "formulas": [],
    }
    layout = {
        "source_type": "powerpoint",
        "regions": [{"type": "title", "bbox": {"left": 10, "top": 10, "right": 800, "bottom": 80}}],
    }
    result = merge.merge_evidence(
        source_image="x/slide_001.webp",
        ocr=ocr,
        layout=layout,
        screenshot_index=1,
        document_id="doc001",
    )
    assert result.source_type == "powerpoint"
    assert len(result.regions) == 1
    assert result.regions[0].text == "Q2 CD-SEM recipe setup improvement"
    assert len(result.tables) == 1
    print("[PASS] test_merge_pure_function")


def test_merge_conflict_marking() -> None:
    """layout 이 table 인데 OCR 표가 없으면 conflict 로 표시한다."""
    ocr = {"raw_text": "some paragraph text", "tables": [], "charts": [], "formulas": []}
    layout = {
        "source_type": "pdf",
        "regions": [{"type": "table", "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100}}],
    }
    result = merge.merge_evidence(source_image="x/p1.webp", ocr=ocr, layout=layout)
    assert result.regions[0].conflicts, "table/paragraph 충돌이 표시돼야 함"
    assert result.unresolved, "unresolved 에 충돌이 누적돼야 함"
    print("[PASS] test_merge_conflict_marking")


def test_chunk_generation_and_embedding_text() -> None:
    """chunk 생성 + embedding_text + quality gate review_status 검증."""
    result = ExtractionResult(
        source_image="x/slide_003.webp",
        source_type="powerpoint",
        document_id="doc001",
        screenshot_id="doc001_s003",
        screenshot_index=3,
    )
    from side_projects.document_extraction.extraction.schemas import Region

    result.regions.append(
        Region(region_id="r001", type="title", text="Manual vs AI-assisted setup time",
               confidence=0.9, model_sources=["paddleocr-vl-1.5"])
    )
    chunks = rag_chunks.generate_chunks(result)
    assert chunks, "chunk 이 생성돼야 함"
    chunk = chunks[0]
    assert "Heading:" in chunk.embedding_text
    assert chunk.review_status == "approved"  # confidence 0.9 >= 0.7
    print("[PASS] test_chunk_generation_and_embedding_text")


def test_offline_pipeline_and_keyword_search() -> None:
    """OFFLINE 파이프라인 e2e + JSONL keyword 검색 스모크."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        doc_dir = tmp_path / "presentation_A"
        doc_dir.mkdir()
        _make_synth_image(doc_dir / "page_001.webp")
        _make_synth_image(doc_dir / "page_002.webp")
        out_dir = tmp_path / "_rag"

        total = extract_folder(
            doc_dir,
            out_dir,
            collection_id="test_collection",
            enable_crop_refine=True,  # crop 훅 경로도 태움
            enable_synthesis=True,
            offline=True,
        )
        assert total > 0, "offline 모드에서도 chunk 가 생성돼야 함"

        chunks_path = out_dir / "rag_chunks.jsonl"
        assert chunks_path.exists()
        lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == total

        # 각 chunk 가 provenance 를 보존하는지 + JSON 파싱 가능한지
        records = [json.loads(line) for line in lines]
        for rec in records:
            assert rec["source_image"], "source_image provenance 필수"
            assert rec["screenshot_index"] >= 1
            assert "embedding_text" in rec

        # keyword 검색 스모크: 'offline' 토큰이 stub content 에 있어야 함
        hits = [r for r in records if "offline" in (r["content"] + r["embedding_text"]).lower()]
        assert hits, "keyword 검색이 최소 1건은 맞아야 함"

        # raw evidence 저장 검증
        raw_dir = out_dir / "raw_evidence"
        assert raw_dir.exists() and any(raw_dir.iterdir())
    print("[PASS] test_offline_pipeline_and_keyword_search")


def main() -> int:
    test_merge_pure_function()
    test_merge_conflict_marking()
    test_chunk_generation_and_embedding_text()
    test_offline_pipeline_and_keyword_search()
    print("\n[INFO] 모든 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
