"""Hybrid 검색: BM25 arm + dense kNN arm -> RRF 융합 -> rerank 훅 -> reader payload.

rag_chart_heavy_architecture.md §3 의 Phase 1 검색 경로. 융합은 클라이언트측
RRF(k=60)로 구현한다(OpenSearch 버전 비의존 + 순수 함수라 사외 검증 가능 +
workflow_3 ensemble 의 RRF 철학과 동일 - 점수 결정은 결정론 로직).

arm 별 가중은 질의유형별 튜닝이 필요하므로(고정 가중 금지 - Chart-MRAG 교훈)
rrf_fuse 의 weights 인자로 열어 둔다. rerank 는 사내 bge-reranker 게이트 뒤의
훅으로만 두고 기본은 passthrough.

reader payload(DVI): top-K 를 "관련 높은 것이 꼬리(질의에 가까운 위치)" 순서로
조립하고, R1 이미지(crop_path 우선, 없으면 source_image)를 첨부 목록으로 만들며,
기계 추출 표/차트 텍스트에는 오류 가능 라벨을 붙인다(reader 의 text-over-visual
bias 로 틀린 표가 옳은 차트를 덮지 않게).

env:
    DOC_EXTRACT_RERANK_API_URL   설정 시 rerank 시도(미설정 = passthrough)

실행(사내 데모):
    uv run python -m side_projects.document_extraction.extraction.hybrid_search
"""

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.embeddings import EmbeddingClient
from side_projects.document_extraction.extraction.opensearch_index import (
    OpenSearchClient,
    resolve_index_name,
)


# === 실행 전 매번 채워 넣을 것 (main 데모용) ======================================
QUERY: str = ""          # 예: "2분기 수율 추세 차트"
TOP_K: int = 5
# ==============================================================================

RRF_K = 60               # RRF 상수(rag_context_architecture 레버 #2)
ARM_SIZE = 20            # arm 당 후보 수(융합 전)
BM25_FIELDS = ["content^2", "parent_heading^2", "bm25_text"]

# 기계 추출 텍스트 라벨(reader 계약): 틀린 표가 옳은 차트 이미지를 덮지 않게.
MACHINE_EXTRACT_NOTE = "[기계 추출 텍스트 - 오류 가능, 첨부 이미지로 검증할 것]"
_MACHINE_EXTRACT_TYPES = {"table_summary", "table_row", "chart_summary", "formula"}


# --- 순수: 쿼리 빌더 + RRF ------------------------------------------------------


def build_bm25_query(query_text: str, *, size: int = ARM_SIZE,
                     filters: list[dict] | None = None) -> dict:
    """BM25 arm 쿼리(순수). filters: OpenSearch filter 절 목록(선택)."""
    bool_query: dict = {
        "must": [{"multi_match": {"query": query_text, "fields": BM25_FIELDS}}]
    }
    if filters:
        bool_query["filter"] = list(filters)
    return {"size": size, "query": {"bool": bool_query}}


def build_knn_query(vector: list[float], *, size: int = ARM_SIZE,
                    filters: list[dict] | None = None) -> dict:
    """dense kNN arm 쿼리(순수)."""
    knn_body: dict = {"vector": list(vector), "k": size}
    if filters:
        knn_body["filter"] = {"bool": {"filter": list(filters)}}
    return {"size": size, "query": {"knn": {"embedding": knn_body}}}


def rrf_fuse(ranked_lists: list[list[str]], *, k: int = RRF_K,
             weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion(순수): id 랭킹 리스트들 -> (id, 점수) 내림차순.

    score(id) = sum_arm( weight_arm / (k + rank_arm(id)) ), rank 는 1부터.
    weights 미지정이면 전 arm 동일 가중(질의유형별 튜닝은 호출부 몫).
    동점은 (최고 단일 rank, id) 로 결정론적으로 깬다.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights 길이 불일치: arms={len(ranked_lists)}, weights={len(weights)}"
        )
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for arm_idx, ranked in enumerate(ranked_lists):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weights[arm_idx] / (k + rank)
            if rank < best_rank.get(doc_id, 1 << 30):
                best_rank[doc_id] = rank
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank.get(item[0], 1 << 30), item[0]),
    )


def hits_from_response(response: dict) -> list[dict]:
    """OpenSearch 응답 -> [{"_id", ...source}] 목록(순수)."""
    hits = []
    for hit in ((response.get("hits") or {}).get("hits")) or []:
        doc = dict(hit.get("_source") or {})
        doc["_id"] = hit.get("_id") or doc.get("chunk_id") or ""
        doc["_score"] = hit.get("_score")
        hits.append(doc)
    return hits


# --- rerank 훅 (게이트: 사내 bge-reranker) ---------------------------------------


def rerank_hits(query_text: str, hits: list[dict], *, top_k: int) -> list[dict]:
    """cross-encoder rerank 훅. DOC_EXTRACT_RERANK_API_URL 미설정이면 passthrough.

    사내 배선 시: (query, content) 쌍을 bge-reranker-v2-m3 에 보내 점수로
    재정렬한다(최대 단일 레버 - rag_context_architecture 레버 #3). Phase 1
    골격에서는 인터페이스만 고정하고 passthrough 로 둔다.
    """
    api_url = os.getenv("DOC_EXTRACT_RERANK_API_URL", "").strip()
    if not api_url:
        return hits[:top_k]
    print("[WARNING] rerank 엔드포인트 배선은 사내 TODO - passthrough 로 진행")
    return hits[:top_k]


# --- 검색 + reader payload -------------------------------------------------------


def search_hybrid(
    query_text: str,
    *,
    client: OpenSearchClient | None = None,
    embedder: EmbeddingClient | None = None,
    index_name: str | None = None,
    top_k: int = TOP_K,
    arm_size: int = ARM_SIZE,
    rrf_k: int = RRF_K,
    weights: list[float] | None = None,
    filters: list[dict] | None = None,
) -> list[dict]:
    """BM25 + dense 2-arm 검색 -> RRF 융합 -> rerank 훅 -> top-K hit 목록.

    각 hit 은 _source 전체(+ "_rrf" 점수)를 갖는다 - crop_path/source_image
    provenance 가 그대로 실려 reader 단계(DVI)로 이어진다.
    """
    client = client or OpenSearchClient()
    embedder = embedder or EmbeddingClient()
    index_name = index_name or resolve_index_name()

    bm25_resp = client.search(
        index_name, build_bm25_query(query_text, size=arm_size, filters=filters)
    )
    bm25_hits = hits_from_response(bm25_resp)

    query_vec = embedder.embed_one(query_text)
    knn_resp = client.search(
        index_name, build_knn_query(query_vec, size=arm_size, filters=filters)
    )
    knn_hits = hits_from_response(knn_resp)

    by_id: dict[str, dict] = {}
    for hit in bm25_hits + knn_hits:
        by_id.setdefault(hit["_id"], hit)

    fused = rrf_fuse(
        [[h["_id"] for h in bm25_hits], [h["_id"] for h in knn_hits]],
        k=rrf_k, weights=weights,
    )
    ranked = []
    for doc_id, score in fused:
        doc = dict(by_id[doc_id])
        doc["_rrf"] = score
        ranked.append(doc)

    print(
        f"[INFO] hybrid 검색: bm25={len(bm25_hits)}, knn={len(knn_hits)}, "
        f"fused={len(ranked)} -> top{top_k}"
    )
    return rerank_hits(query_text, ranked, top_k=top_k)


def build_reader_payload(hits: list[dict], *, max_images: int = 4) -> dict:
    """top-K hit -> reader(Kimi-K2.6) 입력 payload(순수).

    - context_text: 관련 낮은 것부터 높은 것 순서로 배치(관련 최고가 꼬리 =
      질의에 가장 가까운 위치, lost-in-the-middle 대응 레버 #5).
    - image_paths: R1 첨부(crop_path 우선, 없으면 source_image; 중복 제거,
      관련 높은 순으로 max_images 개).
    - 기계 추출 표/차트/수식 텍스트에는 오류 가능 라벨을 붙인다.
    """
    blocks: list[str] = []
    image_paths: list[str] = []
    citations: list[dict] = []

    for hit in hits:  # hits 는 관련 높은 순
        region_type = str(hit.get("region_type") or "other")
        content = str(hit.get("content") or "").strip()
        heading = str(hit.get("parent_heading") or "").strip()
        note = f" {MACHINE_EXTRACT_NOTE}" if region_type in _MACHINE_EXTRACT_TYPES else ""
        header = f"[{hit.get('_id') or hit.get('chunk_id') or '?'}] ({region_type}){note}"
        body = (f"{heading}: {content}" if heading else content) or "(내용 없음)"
        blocks.append(f"{header}\n{body}")

        image = str(hit.get("crop_path") or "").strip() or str(
            hit.get("source_image") or "").strip()
        if image and image not in image_paths and len(image_paths) < max_images:
            image_paths.append(image)

        citations.append({
            "chunk_id": str(hit.get("chunk_id") or hit.get("_id") or ""),
            "document_id": str(hit.get("document_id") or ""),
            "screenshot_index": int(hit.get("screenshot_index") or 0),
            "region_id": str(hit.get("region_id") or ""),
            "source_image": str(hit.get("source_image") or ""),
            "crop_path": str(hit.get("crop_path") or ""),
        })

    # 관련 최고 chunk 가 꼬리에 오도록 역순 배치
    context_text = "\n\n".join(reversed(blocks))
    return {
        "context_text": context_text,
        "image_paths": image_paths,
        "citations": citations,
    }


def main() -> int:
    if not QUERY.strip():
        print("[ERROR] QUERY 가 비어 있습니다. 상단 상수를 수정하세요.")
        return 1
    client = OpenSearchClient()
    if not client.ping():
        print("[ERROR] OpenSearch 에 연결할 수 없습니다 (사내 서버/env 확인).")
        return 1
    hits = search_hybrid(QUERY, client=client, top_k=TOP_K)
    payload = build_reader_payload(hits)
    print(f"[INFO] top-{len(hits)} / 이미지 첨부 {len(payload['image_paths'])}건")
    for citation in payload["citations"]:
        print(f"[INFO]   - {citation['chunk_id']} <- {citation['crop_path'] or citation['source_image']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ARM_SIZE",
    "MACHINE_EXTRACT_NOTE",
    "RRF_K",
    "build_bm25_query",
    "build_knn_query",
    "build_reader_payload",
    "hits_from_response",
    "rerank_hits",
    "rrf_fuse",
    "search_hybrid",
]
