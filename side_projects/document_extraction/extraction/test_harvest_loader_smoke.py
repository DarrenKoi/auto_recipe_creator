"""C1 harvest_loader 스모크 테스트 (사외 OK, VLM/PyMuPDF 불필요).

    uv run python -m side_projects.document_extraction.extraction.test_harvest_loader_smoke
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.extraction._b1_testing import write_synthetic_bundle
from side_projects.document_extraction.extraction.harvest_loader import load_bundle


def test_load_bundle_basic():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        bundle = load_bundle(root)
        assert len(bundle.pages) == 3, bundle.pages
        assert bundle.doc_id == "syn_manual"
        p1 = bundle.pages[0]
        assert p1.page_no == 1
        assert "alignment system" in p1.plain_text
        assert p1.has_text is True


def test_blocks_parsed_with_spans():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        p1 = load_bundle(root).pages[0]
        assert len(p1.blocks) == 2, p1.blocks
        head, body = p1.blocks[0], p1.blocks[1]
        assert head.text == "1.1 Setup"
        assert head.max_size == 18.0
        assert "alignment system" in body.text
        assert body.max_size == 9.96   # real PDFs scatter sizes; structure buckets them
        # bbox preserved as [x0,y0,x1,y1]
        assert head.bbox == [72, 60, 300, 82]


def test_tables_figures_render_loaded():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        b = load_bundle(root)
        p1, p2 = b.pages[0], b.pages[1]
        # tables: loader keeps BOTH raw sources (pymupdf+pdfplumber); dedup is the chunker's job
        assert len(p1.tables) == 2
        assert p1.tables[0]["rows"][0] == ["Parameter", "Range"]
        assert p2.tables == []
        # figures: file path resolved against bundle root, points at a real file
        assert len(p1.figures) == 1
        assert Path(p1.figures[0]["path"]).exists()
        assert p1.figures[0]["bboxes_on_page"] == [[350, 60, 420, 130]]
        # render path resolved + exists
        assert p1.render_path and Path(p1.render_path).exists()
        assert p2.render_path and Path(p2.render_path).exists()


def test_missing_files_tolerated():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        # manifest 는 page3 tables=1 이라 했지만 파일을 지움 -> 경고, 크래시 금지
        (root / "tables" / "page_0003.json").unlink()
        b = load_bundle(root)  # must not raise
        p3 = b.pages[2]
        assert p3.tables == []
        assert any("tables 누락" in w for w in p3.load_warnings), p3.load_warnings


def test_block_preserves_line_breaks():
    # real get_text('dict') puts each visual line in its own 'lines' entry;
    # the loader must keep newlines so step detection works downstream.
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        proc_block = load_bundle(root).pages[1].blocks[1]   # the 3-line procedure
        assert proc_block.text.count("\n") == 2             # 3 lines -> 2 breaks
        assert proc_block.text.splitlines()[0].startswith("Step 1")


def test_non_int_page_record_skipped():
    import json
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        man["per_page"].append({"page": None, "text": True})   # corrupt rec
        (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        bundle = load_bundle(root)                              # must not raise
        assert len(bundle.pages) == 3                           # bad rec skipped


def main():
    test_load_bundle_basic()
    test_blocks_parsed_with_spans()
    test_tables_figures_render_loaded()
    test_missing_files_tolerated()
    test_block_preserves_line_breaks()
    test_non_int_page_record_skipped()
    print("[PASS] test_harvest_loader_smoke")


if __name__ == "__main__":
    main()
