"""OpenSearch 색인기 + hybrid 검색 스모크 (fake transport, 서버/임베딩 불필요).

실행:
    uv run python -m side_projects.document_extraction.extraction.test_opensearch_smoke
"""

import json
import math
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.crop import map_charts_to_crop_paths
from side_projects.document_extraction.extraction.embeddings import (
    EMBED_DIM,
    EmbeddingClient,
    offline_embedding,
)
from side_projects.document_extraction.extraction.hybrid_search import (
    MACHINE_EXTRACT_NOTE,
    build_bm25_query,
    build_knn_query,
    build_reader_payload,
    rrf_fuse,
    search_hybrid,
)
from side_projects.document_extraction.extraction.opensearch_index import (
    OpenSearchClient,
    build_bulk_body,
    build_index_mapping,
    build_bm25_text,
    chunk_to_doc,
    index_chunks_jsonl,
)
from side_projects.document_extraction.extraction.rag_chunks import generate_chunks
from side_projects.document_extraction.extraction.schemas import (
    Chart,
    ExtractionResult,
    Region,
)


# --- fake transport: 최소 OpenSearch 흉내(인덱스/문서 in-memory) -----------------


class FakeOpenSearch:
    """(method, url, body, is_ndjson) -> (status, dict). 색인/검색 배관 검증용."""

    def __init__(self, search_responses: list[dict] | None = None):
        self.indices: dict[str, dict] = {}
        self.docs: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self._search_responses = list(search_responses or [])

    def transport(self, method: str, url: str, body, is_ndjson: bool):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1] if "/" in url.split("://", 1)[-1] else "/"
        self.calls.append((method, path))

        if method == "GET" and path == "/":
            return 200, {"cluster_name": "fake"}
        if method == "PUT":
            self.indices[path.lstrip("/")] = json.loads(body or "{}")
            return 200, {"acknowledged": True}
        if method == "GET":
            name = path.lstrip("/")
            return (200, {name: self.indices[name]}) if name in self.indices else (
                404, {"error": "index_not_found"})
        if method == "DELETE":
            self.indices.pop(path.lstrip("/"), None)
            return 200, {"acknowledged": True}
        if method == "POST" and path == "/_bulk":
            lines = [l for l in (body or "").splitlines() if l.strip()]
            for action_line, source_line in zip(lines[0::2], lines[1::2]):
                action = json.loads(action_line)["index"]
                self.docs[action.get("_id") or f"auto{len(self.docs)}"] = json.loads(
                    source_line)
            return 200, {"errors": False, "items": []}
        if method == "POST" and path.endswith("/_refresh"):
            return 200, {}
        if method == "POST" and path.endswith("/_search"):
            if self._search_responses:
                return 200, self._search_responses.pop(0)
            hits = [
                {"_id": doc_id, "_score": 1.0, "_source": doc}
                for doc_id, doc in self.docs.items()
            ]
            return 200, {"hits": {"hits": hits}}
        return 400, {"error": f"unhandled {method} {path}"}


def _chart_result() -> ExtractionResult:
    result = ExtractionResult(
        source_image="cap/page_001.webp", source_type="powerpoint",
        document_id="doc1", screenshot_id="doc1_s001", screenshot_index=1,
    )
    result.regions.append(Region(region_id="r001", type="title", text="Q2 수율"))
    result.charts.append(Chart(region_id="c001", title="Yield Trend",
                               legend_labels=["manual", "AI"],
                               visible_values=["30", "18"],
                               trend_summary="AI lower", confidence=0.8))
    return result


def test_offline_embedding_deterministic_unit() -> None:
    a1 = offline_embedding("hello chart")
    a2 = offline_embedding("hello chart")
    b = offline_embedding("other text")
    assert a1 == a2, "같은 텍스트는 같은 벡터여야 함"
    assert a1 != b
    assert len(a1) == EMBED_DIM
    norm = math.sqrt(sum(v * v for v in a1))
    assert abs(norm - 1.0) < 1e-9, norm
    print("[PASS] test_offline_embedding_deterministic_unit")


def test_mapping_fields_and_dim() -> None:
    mapping = build_index_mapping()
    props = mapping["mappings"]["properties"]
    assert props["embedding"]["dimension"] == EMBED_DIM
    assert mapping["settings"]["index"]["knn"] is True
    for field in ("crop_path", "source_image", "chunk_id", "region_type"):
        assert props[field]["type"] == "keyword", field
    for field in ("content", "bm25_text", "parent_heading"):
        assert props[field]["type"] == "text", field
    assert props["sparse_features"]["type"] == "rank_features"  # Phase 2 예약
    print("[PASS] test_mapping_fields_and_dim")


def test_chunk_to_doc_preserves_provenance_and_crop_path() -> None:
    result = _chart_result()
    lookup = map_charts_to_crop_paths(
        result.charts,
        [{"region_id": "r002", "region_type": "chart", "crop_path": "x/r002_chart.jpg"}],
    )
    chunks = generate_chunks(result, chart_crop_lookup=lookup)
    chart_chunks = [c for c in chunks if c.region_type == "chart_summary"]
    assert chart_chunks and chart_chunks[0].crop_path == "x/r002_chart.jpg"

    chunk_dict = chart_chunks[0].to_dict()
    doc = chunk_to_doc(chunk_dict, embedding=[0.0] * 4)
    assert doc["crop_path"] == "x/r002_chart.jpg"
    assert doc["source_image"] == "cap/page_001.webp"
    assert doc["screenshot_index"] == 1
    assert doc["embedding"] == [0.0] * 4
    assert "Yield Trend" in doc["bm25_text"]
    # 구버전 JSONL(crop_path 없음)도 안전
    old = dict(chunk_dict)
    old.pop("crop_path")
    assert chunk_to_doc(old)["crop_path"] == ""
    print("[PASS] test_chunk_to_doc_preserves_provenance_and_crop_path")


def test_bulk_body_format_idempotent_ids() -> None:
    docs = [{"chunk_id": "a1", "content": "x"}, {"chunk_id": "a2", "content": "y"}]
    body = build_bulk_body(docs, "idx")
    lines = [l for l in body.splitlines() if l]
    assert len(lines) == 4
    action = json.loads(lines[0])
    assert action["index"]["_index"] == "idx" and action["index"]["_id"] == "a1"
    assert json.loads(lines[1])["content"] == "x"
    assert body.endswith("\n")
    print("[PASS] test_bulk_body_format_idempotent_ids")


def test_rrf_fuse_math_and_determinism() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "c"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "b", fused  # 양 arm 상위 -> 최상
    score_b = 1 / 62 + 1 / 61
    assert abs(fused[0][1] - score_b) < 1e-12
    # 동점(같은 rank 구성)은 id 로 결정론적
    tie = rrf_fuse([["x"], ["y"]], k=60)
    assert [d for d, _ in tie] == ["x", "y"]
    try:
        rrf_fuse([["a"]], weights=[1.0, 2.0])
        raise AssertionError("weights 길이 불일치는 ValueError 여야 함")
    except ValueError:
        pass
    print("[PASS] test_rrf_fuse_math_and_determinism")


def test_query_builders() -> None:
    q = build_bm25_query("yield trend", size=7, filters=[{"term": {"source_type": "powerpoint"}}])
    assert q["size"] == 7
    assert q["query"]["bool"]["must"][0]["multi_match"]["query"] == "yield trend"
    assert q["query"]["bool"]["filter"][0]["term"]["source_type"] == "powerpoint"
    kq = build_knn_query([0.1, 0.2], size=3)
    assert kq["query"]["knn"]["embedding"]["k"] == 3
    assert kq["query"]["knn"]["embedding"]["vector"] == [0.1, 0.2]
    print("[PASS] test_query_builders")


def test_index_chunks_jsonl_e2e_fake_transport() -> None:
    """JSONL -> (offline 임베딩) -> fake OpenSearch 색인 배관 e2e."""
    result = _chart_result()
    lookup = {"c001": "x/r002_chart.jpg"}
    chunks = generate_chunks(result, chart_crop_lookup=lookup)

    fake = FakeOpenSearch()
    client = OpenSearchClient("http://fake:9200", transport=fake.transport)
    embedder = EmbeddingClient(offline=True, dim=8)

    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "rag_chunks.jsonl"
        with jsonl.open("w", encoding="utf-8") as fp:
            for chunk in chunks:
                fp.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        total = index_chunks_jsonl(
            jsonl, client=client, embedder=embedder, index_name="test_idx")

    assert total == len(chunks) and len(fake.docs) == len(chunks)
    assert "test_idx" in fake.indices
    assert fake.indices["test_idx"]["mappings"]["properties"]["embedding"]["dimension"] == 8
    chart_doc = fake.docs["doc1_s001_c001"]
    assert chart_doc["crop_path"] == "x/r002_chart.jpg"
    assert len(chart_doc["embedding"]) == 8
    print("[PASS] test_index_chunks_jsonl_e2e_fake_transport")


def test_search_hybrid_fused_order_and_provenance() -> None:
    """canned 2-arm 응답 -> RRF 융합 순서 + crop_path 전달 검증."""
    def hit(doc_id, crop=""):
        return {"_id": doc_id, "_score": 1.0, "_source": {
            "chunk_id": doc_id, "content": f"content {doc_id}",
            "region_type": "chart_summary", "crop_path": crop,
            "source_image": "cap/page_001.webp", "screenshot_index": 1,
        }}

    bm25_resp = {"hits": {"hits": [hit("a"), hit("b", "x/b.jpg"), hit("c")]}}
    knn_resp = {"hits": {"hits": [hit("b", "x/b.jpg"), hit("d")]}}
    fake = FakeOpenSearch(search_responses=[bm25_resp, knn_resp])
    client = OpenSearchClient("http://fake:9200", transport=fake.transport)
    embedder = EmbeddingClient(offline=True, dim=8)

    hits = search_hybrid("trend", client=client, embedder=embedder,
                         index_name="test_idx", top_k=3)
    assert [h["_id"] for h in hits][0] == "b", hits
    assert len(hits) == 3
    assert hits[0]["crop_path"] == "x/b.jpg"
    assert all("_rrf" in h for h in hits)
    print("[PASS] test_search_hybrid_fused_order_and_provenance")


def test_reader_payload_dvi_contract() -> None:
    hits = [
        {"_id": "h1", "chunk_id": "h1", "region_type": "chart_summary",
         "content": "chart labels", "crop_path": "x/c1.jpg",
         "source_image": "cap/p1.webp", "screenshot_index": 1},
        {"_id": "h2", "chunk_id": "h2", "region_type": "region_text",
         "content": "plain text", "crop_path": "",
         "source_image": "cap/p2.webp", "screenshot_index": 2},
        {"_id": "h3", "chunk_id": "h3", "region_type": "table_row",
         "content": "row", "crop_path": "x/c1.jpg",  # 중복 이미지
         "source_image": "cap/p1.webp", "screenshot_index": 1},
    ]
    payload = build_reader_payload(hits, max_images=4)
    # 관련 최고(h1)가 꼬리(마지막 블록)에 배치
    assert payload["context_text"].strip().split("\n\n")[-1].startswith("[h1]")
    # 기계 추출 라벨: chart/table 에는 있고 region_text 에는 없음
    assert MACHINE_EXTRACT_NOTE in payload["context_text"]
    text_block = [b for b in payload["context_text"].split("\n\n") if "[h2]" in b][0]
    assert MACHINE_EXTRACT_NOTE not in text_block
    # 이미지: crop 우선 + 중복 제거(h1/h3 같은 crop) + source_image 폴백(h2)
    assert payload["image_paths"] == ["x/c1.jpg", "cap/p2.webp"]
    assert len(payload["citations"]) == 3
    print("[PASS] test_reader_payload_dvi_contract")


def main() -> int:
    test_offline_embedding_deterministic_unit()
    test_mapping_fields_and_dim()
    test_chunk_to_doc_preserves_provenance_and_crop_path()
    test_bulk_body_format_idempotent_ids()
    test_rrf_fuse_math_and_determinism()
    test_query_builders()
    test_index_chunks_jsonl_e2e_fake_transport()
    test_search_hybrid_fused_order_and_provenance()
    test_reader_payload_dvi_contract()
    print("\n[INFO] 모든 OpenSearch/hybrid 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
