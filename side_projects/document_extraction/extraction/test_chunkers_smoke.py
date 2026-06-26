"""C3 chunkers 스모크 테스트 (사외 OK).

    uv run python -m side_projects.document_extraction.extraction.test_chunkers_smoke
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.extraction._b1_testing import write_synthetic_bundle
from side_projects.document_extraction.extraction.harvest_loader import load_bundle
from side_projects.document_extraction.extraction.structure import assign_structure
from side_projects.document_extraction.extraction.chunkers import chunk_bundle, chunk_page


def _prepared_bundle(td):
    root = write_synthetic_bundle(Path(td) / "syn_manual")
    bundle = load_bundle(root)
    assign_structure(bundle)
    return bundle


def _by_type(chunks, region_type):
    return [c for c in chunks if c.region_type == region_type]


def test_region_text_chunk_carries_context():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_page(bundle.pages[0], bundle.doc_id)
        rt = _by_type(chunks, "region_text")
        assert len(rt) == 1, [c.region_type for c in chunks]
        c = rt[0]
        assert "alignment system" in c.content
        assert c.parent_heading == "1.1 Setup"
        assert c.document_id == "syn_manual"
        assert c.screenshot_index == 1
        # section path preserved for retrieval (embedding_text built from it)
        assert "Chapter 1 Overview" in c.embedding_text
        assert c.review_status == "approved"   # digital text = high confidence


def test_procedure_kept_whole():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_page(bundle.pages[1], bundle.doc_id)  # page 2 = procedure
        procs = _by_type(chunks, "procedure")
        assert len(procs) == 1, [c.region_type for c in chunks]
        c = procs[0]
        for step in ("Step 1", "Step 2", "Step 3"):
            assert step in c.content              # all steps in ONE chunk
        assert c.parent_heading == "2.1 Power-on Procedure"
        # the consumed step block must NOT also appear as region_text
        assert _by_type(chunks, "region_text") == []


def test_param_table_rows_exact():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_page(bundle.pages[0], bundle.doc_id)  # param table
        assert len(_by_type(chunks, "table_summary")) == 1
        rows = _by_type(chunks, "table_row")
        assert len(rows) == 2, [c.content for c in rows]   # Focus, Gain (header excluded)
        focus = [c for c in rows if "Focus" in c.content][0]
        assert "Range: 0-100" in focus.content              # column context, no flatten


def test_error_code_table():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_page(bundle.pages[2], bundle.doc_id)  # error code table
        err = _by_type(chunks, "error_code")
        assert len(err) == 2, [c.content for c in chunks]
        e9006 = [c for c in err if "E9006" in c.content][0]
        assert "Align fail" in e9006.content and "Re-center" in e9006.content
        assert "E9006" in e9006.keywords
        assert _by_type(chunks, "table_row") == []          # error table != generic rows


def test_figure_chunk_points_at_original_bytes():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_page(bundle.pages[0], bundle.doc_id)
        figs = _by_type(chunks, "figure")
        assert len(figs) == 1
        c = figs[0]
        assert Path(c.source_image).exists()      # the deduped figure file
        assert c.parent_heading == "1.1 Setup"    # section context for retrieval


def test_chunk_bundle_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        bundle = _prepared_bundle(td)
        chunks = chunk_bundle(bundle)
        types = {c.region_type for c in chunks}
        # all four query-type chunk kinds present across the manual
        assert {"procedure", "table_row", "error_code", "figure", "region_text"} <= types, types
        # every chunk has provenance + an embedding text
        for c in chunks:
            assert c.document_id == "syn_manual"
            assert c.embedding_text
            assert c.chunk_id


def main():
    test_region_text_chunk_carries_context()
    test_procedure_kept_whole()
    test_param_table_rows_exact()
    test_error_code_table()
    test_figure_chunk_points_at_original_bytes()
    test_chunk_bundle_end_to_end()
    print("[PASS] test_chunkers_smoke")


if __name__ == "__main__":
    main()
