"""Marp 생성 스모크 테스트 (순수 함수, 서버/marp-cli 불필요).

실행:
    uv run python -m side_projects.document_extraction.marp.test_marp_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.schemas import (
    Chart, ExtractionResult, Formula, Region, Table)
from side_projects.document_extraction.marp.build_marp import build_deck
from side_projects.document_extraction.marp.generate import (
    evidence_to_marp, results_to_deck)


def _sample_result() -> ExtractionResult:
    result = ExtractionResult(source_image="cap/s1.webp", source_type="powerpoint",
                              document_id="doc1", screenshot_id="doc1_s001",
                              screenshot_index=1)
    result.regions.append(Region(region_id="r001", type="title", text="Q2 Recipe Setup"))
    result.regions.append(Region(region_id="r002", type="body", text="automation improved"))
    result.tables.append(Table(region_id="t001", title="Setup time",
                               header=["mode", "min"], cells=[["manual", "30"], ["AI", "18"]]))
    result.formulas.append(Formula(region_id="f001", latex="x = a + b"))
    result.charts.append(Chart(region_id="c001", title="Trend",
                               legend_labels=["manual", "AI"], visible_values=["30", "18"],
                               trend_summary="AI lower"))
    return result


def test_text_tracks_render_native() -> None:
    md = evidence_to_marp(_sample_result())
    assert "# Q2 Recipe Setup" in md
    assert "- automation improved" in md
    # 표가 GFM 마크다운 표로
    assert "| mode | min |" in md
    assert "| --- | --- |" in md
    assert "| manual | 30 |" in md
    # 수식 KaTeX
    assert "$$ x = a + b $$" in md
    print("[PASS] test_text_tracks_render_native")


def test_chart_raster_vs_datatable() -> None:
    result = _sample_result()
    # crop 없음 -> 데이터 표 대체 + 노트
    md_no_crop = evidence_to_marp(result)
    assert "| series | value |" in md_no_crop
    assert "<!--" in md_no_crop and "chart data c001" in md_no_crop
    # crop 있음 -> 이미지 재삽입
    md_crop = evidence_to_marp(result, crop_lookup={"c001": "crops/c001.jpg"})
    assert "![w:600](crops/c001.jpg)" in md_crop
    print("[PASS] test_chart_raster_vs_datatable")


def test_deck_join_and_frontmatter() -> None:
    r1, r2 = _sample_result(), _sample_result()
    r2.screenshot_id = "doc1_s002"
    r2.screenshot_index = 2
    deck = results_to_deck([r1, r2])
    assert deck.startswith("---\nmarp: true")
    assert deck.count("\n---\n") >= 2  # 프론트매터 닫기 + 슬라이드 구분
    print("[PASS] test_deck_join_and_frontmatter")


def test_no_value_fabrication() -> None:
    """행이 header 보다 짧으면 빈 칸 패딩(값 창작 금지)."""
    result = ExtractionResult(source_image="x", screenshot_id="s1")
    result.tables.append(Table(region_id="t001", header=["a", "b", "c"],
                               cells=[["1", "2"]]))  # 2칸뿐
    md = evidence_to_marp(result)
    assert "| 1 | 2 |  |" in md, md
    print("[PASS] test_no_value_fabrication")


def test_table_pipe_escape_and_widening() -> None:
    """셀의 파이프는 escape, row 가 header 보다 길면 데이터 보존(잘라내기 금지)."""
    from side_projects.document_extraction.marp.generate import _md_table

    # 파이프 escape
    lines = _md_table(["a|b"], [["x|y"]])
    assert "a\\|b" in lines[0]
    assert "x\\|y" in lines[2]
    # row 가 header 보다 김 -> colN 으로 보강, 데이터 보존
    lines = _md_table(["h1"], [["v1", "v2", "v3"]])
    assert "| h1 | col2 | col3 |" in lines[0]
    assert "| v1 | v2 | v3 |" in lines[2]
    print("[PASS] test_table_pipe_escape_and_widening")


def test_empty_slide_filtered() -> None:
    """빈 evidence 슬라이드는 deck 에서 제외된다(빈 페이지 방지)."""
    full = _sample_result()
    empty = ExtractionResult(source_image="x", screenshot_id="s2", screenshot_index=2)
    deck = results_to_deck([full, empty])
    # 슬라이드 구분자는 1개만(빈 슬라이드 제외로)
    body = deck.split("paginate: true\n---\n", 1)[1]
    assert body.count("\n---\n") == 0, body
    print("[PASS] test_empty_slide_filtered")


def test_from_dict_tolerant_numbers() -> None:
    """screenshot_index='3.0' 같은 문자열도 from_dict 가 죽지 않는다."""
    r = ExtractionResult.from_dict(
        {"source_image": "x", "screenshot_index": "3.0", "overall_confidence": "bad"}
    )
    assert r.screenshot_index == 3
    assert r.overall_confidence == 0.0
    # index 0 도 0 으로 보존(`or 1` 오류 없음)
    r0 = ExtractionResult.from_dict({"source_image": "x", "screenshot_index": 0})
    assert r0.screenshot_index == 0
    print("[PASS] test_from_dict_tolerant_numbers")


def test_build_deck_from_raw_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw_evidence"
        raw_dir.mkdir()
        # to_dict -> from_dict 라운드트립 검증 포함
        (raw_dir / "doc1_s001.json").write_text(
            json.dumps(_sample_result().to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        out_md = tmp_path / "deck.md"
        n = build_deck(raw_dir, out_md, {})
        assert n == 1
        text = out_md.read_text(encoding="utf-8")
        assert "# Q2 Recipe Setup" in text
        assert "| mode | min |" in text
    print("[PASS] test_build_deck_from_raw_evidence")


def main() -> int:
    test_text_tracks_render_native()
    test_chart_raster_vs_datatable()
    test_deck_join_and_frontmatter()
    test_no_value_fabrication()
    test_table_pipe_escape_and_widening()
    test_empty_slide_filtered()
    test_from_dict_tolerant_numbers()
    test_build_deck_from_raw_evidence()
    print("\n[INFO] 모든 Marp 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
