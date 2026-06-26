"""B1 end-to-end 스모크: harvest 번들 -> jsonl -> keyword 검색 (embedding 불필요).

    uv run python -m side_projects.document_extraction.extraction.test_b1_pipeline_smoke
"""

import tempfile
from pathlib import Path

from side_projects.document_extraction.extraction._b1_testing import write_synthetic_bundle
from side_projects.document_extraction.extraction.build_rag_chunks import build_to_jsonl
from side_projects.document_extraction.extraction.retrieval import keyword_search, load_chunks


def test_pipeline_jsonl_searchable():
    with tempfile.TemporaryDirectory() as td:
        root = write_synthetic_bundle(Path(td) / "syn_manual")
        out = Path(td) / "rag_chunks.jsonl"
        n = build_to_jsonl(root, out)
        assert n > 0
        chunks = load_chunks(out)

        # error-code lookup lands the error_code chunk in the trusted tier
        res = keyword_search(chunks, "E9006")
        assert res, "E9006 not found"
        top = res[0]
        assert top["chunk"]["region_type"] == "error_code"
        assert top["tier"] == "trusted", top["gate_reasons"]

        # param lookup returns the exact structured row (no flatten)
        gain = keyword_search(chunks, "Gain")
        assert gain, "Gain not found"
        assert gain[0]["chunk"]["region_type"] == "table_row"
        assert "Range: 1-5" in gain[0]["chunk"]["content"]


def main():
    test_pipeline_jsonl_searchable()
    print("[PASS] test_b1_pipeline_smoke")


if __name__ == "__main__":
    main()
