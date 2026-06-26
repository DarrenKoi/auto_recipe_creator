"""C2 structure 스모크 테스트 (사외 OK).

    uv run python -m side_projects.document_extraction.extraction.test_structure_smoke
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.extraction._b1_testing import write_synthetic_bundle
from side_projects.document_extraction.extraction.harvest_loader import load_bundle
from side_projects.document_extraction.extraction.structure import (
    assign_structure,
    build_section_index,
)


def test_section_index_from_toc():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        toc = load_bundle(root).toc
        idx = build_section_index(toc, page_count=3)
        assert idx[1] == ["Chapter 1 Overview", "1.1 Setup"]
        assert idx[2] == ["Chapter 1 Overview", "1.1 Setup"]  # 다음 섹션 전까지 상속
        assert idx[3] == ["Chapter 2 Errors"]


def test_assign_structure_headings_and_paths():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        bundle = load_bundle(root)
        assign_structure(bundle)
        p1 = bundle.pages[0]
        assert p1.section_path == ["Chapter 1 Overview", "1.1 Setup"]
        head, body = p1.blocks[0], p1.blocks[1]
        assert head.is_heading is True       # size 18 >> body 10
        assert body.is_heading is False
        assert body.parent_heading == "1.1 Setup"   # nearest heading above
        # page 3 has only a heading-sized block but section_path still gives context
        assert bundle.pages[2].section_path == ["Chapter 2 Errors"]


def main():
    test_section_index_from_toc()
    test_assign_structure_headings_and_paths()
    print("[PASS] test_structure_smoke")


if __name__ == "__main__":
    main()
