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


def test_crop_geometry() -> None:
    """compute_crop_box: margin 추가 + frame clamp + 무효 bbox 거부."""
    from side_projects.document_extraction.extraction.crop import (
        compute_crop_box, crop_region)

    # 정상: margin 10%
    box = compute_crop_box((100, 100, 200, 200), (1000, 1000), margin_ratio=0.1)
    assert box == (90, 90, 210, 210), box
    # frame 경계 clamp
    box = compute_crop_box((0, 0, 50, 50), (40, 40), margin_ratio=0.5)
    assert box == (0, 0, 40, 40), box
    # 무효 bbox(너비 0)
    assert compute_crop_box((100, 100, 100, 200), (1000, 1000)) is None
    assert compute_crop_box((100, 100, 200, 200), (0, 0)) is None

    # crop_region: 실제 이미지에서 잘라 저장
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / "src.webp"
        _make_synth_image(img_path, (800, 600))
        meta = crop_region(img_path, "r001", "table", (100, 100, 300, 250),
                           tmp_path / "crops", margin_ratio=0.05)
        assert meta is not None
        assert Path(meta.crop_path).exists()
        assert meta.crop_wh[0] > 0 and meta.crop_wh[1] > 0
        assert meta.parent_bbox == (100, 100, 300, 250)
    print("[PASS] test_crop_geometry")


def test_crop_merge_lossless() -> None:
    """_merge_crop_refine: 별개 표는 보존, 정확 중복만 제거."""
    from side_projects.document_extraction.extraction.extract_screenshot import (
        _merge_crop_refine)

    ocr = {"tables": [{"header": ["a", "b"], "rows": [["1", "2"]]}]}
    # 같은 header -> 중복으로 skip
    _merge_crop_refine("table", {"header": ["a", "b"], "rows": [["9", "9"]]}, ocr)
    assert len(ocr["tables"]) == 1, "정확 중복은 추가되면 안 됨"
    # 다른 header -> 별개 표로 추가(손실 금지)
    _merge_crop_refine("table", {"header": ["x", "y"], "rows": [["3", "4"]]}, ocr)
    assert len(ocr["tables"]) == 2, "별개 표는 보존돼야 함"

    # 두 개의 distinct chart 영역 -> charts[0] 로 뭉치지 않음
    ocr2 = {}
    _merge_crop_refine("chart", {"labels": ["red", "blue"]}, ocr2)
    _merge_crop_refine("chart", {"labels": ["cat", "dog"]}, ocr2)
    assert len(ocr2["charts"]) == 2, "별개 차트는 보존돼야 함"
    print("[PASS] test_crop_merge_lossless")


def test_table_row_chunks_and_heading() -> None:
    """table_row chunk 생성 + bbox 기반 nearest-heading."""
    from side_projects.document_extraction.extraction.schemas import (
        BBox, Region, Table)

    result = ExtractionResult(source_image="x/s1.webp", document_id="doc1",
                              screenshot_id="doc1_s001", screenshot_index=1)
    # 위쪽 title + 아래 body
    result.regions.append(Region(region_id="r001", type="title", text="Section A",
                                 bbox=BBox(0, 0, 400, 50)))
    result.regions.append(Region(region_id="r002", type="body", text="some body",
                                 bbox=BBox(0, 60, 400, 120)))
    result.tables.append(Table(region_id="t001", title="Setup time",
                               header=["mode", "min"],
                               cells=[["manual", "30"], ["AI", "18"]], confidence=0.8))
    chunks = rag_chunks.generate_chunks(result)
    rows = [c for c in chunks if c.region_type == "table_row"]
    assert len(rows) == 2, len(rows)
    assert "mode: manual" in rows[0].content and "min: 30" in rows[0].content
    assert rows[0].parent_heading == "Setup time"
    # body chunk 의 heading 은 위쪽 title
    body = next(c for c in chunks if c.region_id == "r002")
    assert body.parent_heading == "Section A", body.parent_heading
    print("[PASS] test_table_row_chunks_and_heading")


def test_deterministic_synthesis_no_model() -> None:
    """무-LLM 합성: evidence 만으로 summary/confidence/unresolved 조립."""
    from side_projects.document_extraction.extraction.schemas import (
        Chart, Region, Table)
    from side_projects.document_extraction.extraction.synthesis import (
        synthesize_deterministic)

    result = ExtractionResult(source_image="x/s1.webp", source_type="powerpoint")
    result.regions.append(Region(region_id="r001", type="title", text="Q2 setup", confidence=0.8))
    result.regions.append(Region(region_id="r002", type="body", text="yield improved", confidence=0.6))
    result.tables.append(Table(region_id="t001", title="Setup time", header=["mode", "min"],
                               cells=[["manual", "30"]], confidence=0.7))
    result.charts.append(Chart(region_id="c001", title="Trend", trend_summary="AI lower"))

    synth = synthesize_deterministic(result)
    assert synth["summary_markdown"].startswith("# Q2 setup")
    assert "## Tables" in synth["summary_markdown"]
    assert "## Charts" in synth["summary_markdown"]
    # confidence = mean(0.8, 0.6, 0.7) = 0.7
    assert abs(synth["overall_confidence"] - 0.7) < 1e-6
    assert isinstance(synth["unresolved"], list)
    print("[PASS] test_deterministic_synthesis_no_model")


def test_synthesis_chain_offline_stub_and_glm_mode() -> None:
    """Stage 6 폴백 체인: offline 에서 kimi/glm 모두 stub + synthesis_service 기록."""
    from side_projects.document_extraction.extraction.models import StageRunner
    from side_projects.document_extraction.extraction.extract_screenshot import extract_one

    runner = StageRunner(offline=True)
    synth = runner.run_synthesis("x.webp", "powerpoint", "{}")
    assert synth["synthesis_service"] == "offline"
    synth_text = runner.run_synthesis_text("powerpoint", "{}")
    assert synth_text["synthesis_service"] == "offline"
    assert synth_text["summary_markdown"].startswith("[offline-stub]")

    # glm 모드 e2e(offline): summary_model_sources 가 실제 사용 서비스(offline)를 기록
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page_001.webp"
        _make_synth_image(img)
        result = extract_one(
            img,
            screenshot_index=1,
            document_id="doc_glm",
            collection_id="c",
            source_type_hint="powerpoint",
            runner=StageRunner(offline=True),
            synthesis_mode="glm",
        )
        assert result.summary_model_sources == ["offline"], result.summary_model_sources
    print("[PASS] test_synthesis_chain_offline_stub_and_glm_mode")


def test_crop_metas_persisted_for_marp_mapping() -> None:
    """crop refine 이 crops.json(CropMeta 목록)을 남겨 marp crop 대응이 가능해야 함."""
    from side_projects.document_extraction.extraction.extract_screenshot import (
        _apply_crop_refine)
    from side_projects.document_extraction.extraction.models import StageRunner

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page_001.webp"
        _make_synth_image(img)
        layout = {"regions": [
            {"type": "title", "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 40}},
            {"type": "chart", "bbox": {"left": 10, "top": 100, "right": 600, "bottom": 400}},
        ]}
        ocr: dict = {}
        crop_dir = Path(tmp) / "_crops" / "doc_s001"
        _apply_crop_refine(StageRunner(offline=True), img, layout, ocr,
                           (1280, 720), crop_dir)
        meta_path = crop_dir / "crops.json"
        assert meta_path.exists(), "crops.json 이 저장돼야 함"
        metas = json.loads(meta_path.read_text(encoding="utf-8"))
        assert len(metas) == 1 and metas[0]["region_type"] == "chart"
        assert metas[0]["region_id"] == "r002"
        assert Path(metas[0]["crop_path"]).exists()
    print("[PASS] test_crop_metas_persisted_for_marp_mapping")


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
            synthesis_mode="deterministic",  # 모델 0콜 합성 경로
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

        # document_summary chunk 의 provenance 가 deterministic 으로 정직해야 함
        doc_summaries = [r for r in records if r["region_type"] == "document_summary"]
        for ds in doc_summaries:
            assert ds["model_sources"] == ["deterministic"], ds["model_sources"]

        # raw evidence 저장 검증
        raw_dir = out_dir / "raw_evidence"
        assert raw_dir.exists() and any(raw_dir.iterdir())
    print("[PASS] test_offline_pipeline_and_keyword_search")


def main() -> int:
    test_merge_pure_function()
    test_merge_conflict_marking()
    test_chunk_generation_and_embedding_text()
    test_crop_geometry()
    test_crop_merge_lossless()
    test_table_row_chunks_and_heading()
    test_deterministic_synthesis_no_model()
    test_synthesis_chain_offline_stub_and_glm_mode()
    test_crop_metas_persisted_for_marp_mapping()
    test_offline_pipeline_and_keyword_search()
    print("\n[INFO] 모든 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
