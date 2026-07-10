"""OpenSearch 색인기: rag_chunks.jsonl -> BM25 + dense kNN 단일 인덱스 (Phase 1).

rag_chart_heavy_architecture.md Phase 1 / rag_context_architecture.md §3 의
"OpenSearch 단일 인덱스(BM25 + dense kNN [+ neural sparse])" 를 구현한다.
순수 빌더(매핑/문서 변환/bulk 본문)와 얇은 REST 클라이언트(requests, transport
주입 가능)를 분리해, 사외에서는 순수부 + fake transport 로 스모크 검증하고
사내에서는 실제 OpenSearch 에 색인한다.

3-표상과의 대응 (chunk 하나가 세 표상을 함께 나른다):
    R1 래스터  -> source_image / crop_path (인덱스에 keyword 로 저장만, 검색 키 아님)
    R2 구조 표 -> content (table_row/table_summary/chart_summary 텍스트) = BM25 arm
    R3 컨텍스트 -> embedding_text = dense arm 입력 (+ bm25_text 보조)
    R4 vision  -> Phase 2 게이트(page_embedding 필드 자리만 예약, 기본 미사용)

env:
    DOC_EXTRACT_OPENSEARCH_URL       기본 http://localhost:9200
    DOC_EXTRACT_OPENSEARCH_INDEX     기본 doc_extract_chunks
    DOC_EXTRACT_OPENSEARCH_USER/PASSWORD  선택(basic auth)

실행(사내):
    uv run python -m side_projects.document_extraction.extraction.opensearch_index
"""

import json
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.embeddings import (
    EMBED_DIM,
    EmbeddingClient,
)
from side_projects.document_extraction.extraction.retrieval import load_chunks


# === 실행 전 매번 채워 넣을 것 =================================================
RAG_CHUNKS_JSONL: Path = Path("")   # 예: Path(r"C:\...\_rag\rag_chunks.jsonl")
RECREATE_INDEX: bool = False        # True 면 기존 인덱스 삭제 후 재생성
# ==============================================================================

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "doc_extract_chunks"
BULK_BATCH_SIZE = 200


def resolve_opensearch_url() -> str:
    return (
        os.getenv("DOC_EXTRACT_OPENSEARCH_URL", "").strip()
        or DEFAULT_OPENSEARCH_URL
    ).rstrip("/")


def resolve_index_name() -> str:
    return os.getenv("DOC_EXTRACT_OPENSEARCH_INDEX", "").strip() or DEFAULT_INDEX_NAME


# --- 순수 빌더 ---------------------------------------------------------------


def build_index_mapping(*, dim: int = EMBED_DIM) -> dict:
    """단일 인덱스 매핑(순수). BM25(text) + dense kNN + provenance keyword 들.

    neural sparse(rank_features)와 R4 vision(page_embedding)은 자리만 예약 -
    문서에 해당 필드가 없으면 비용이 들지 않고, 나중에 재색인 없이 채울 수 있다.
    """
    return {
        "settings": {
            "index": {"knn": True},
        },
        "mappings": {
            "properties": {
                # 식별/provenance
                "chunk_id": {"type": "keyword"},
                "collection_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "screenshot_id": {"type": "keyword"},
                "screenshot_index": {"type": "integer"},
                "region_id": {"type": "keyword"},
                "region_type": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "review_status": {"type": "keyword"},
                "confidence": {"type": "float"},
                "bbox": {"type": "object", "enabled": False},
                # R1 래스터 payload (검색 키 아님, 답변 시 reader 첨부용)
                "source_image": {"type": "keyword"},
                "crop_path": {"type": "keyword"},
                # R2/R3 텍스트 arm
                "content": {"type": "text"},
                "parent_heading": {"type": "text"},
                "bm25_text": {"type": "text"},
                # dense arm (bge-m3)
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                # 예약: neural sparse(bge-m3 sparse) / R4 vision arm (Phase 2)
                "sparse_features": {"type": "rank_features"},
            }
        },
    }


def build_bm25_text(chunk: dict) -> str:
    """BM25 색인 텍스트(순수): content + heading + keywords + raw OCR.

    dense 용 embedding_text(메타 포함 서술형)와 달리 정확 토큰 매칭에 유리한
    원문 위주로 구성한다(레버: 에러코드/파라미터명은 BM25 가 이긴다).
    """
    parts = [
        str(chunk.get("content") or ""),
        str(chunk.get("parent_heading") or ""),
        " ".join(str(k) for k in (chunk.get("keywords") or [])),
        str(chunk.get("raw_ocr_text") or ""),
    ]
    return "\n".join(p for p in parts if p.strip())


def chunk_to_doc(chunk: dict, *, embedding: list[float] | None = None) -> dict:
    """rag_chunks.jsonl 의 chunk dict -> OpenSearch 문서(순수).

    구버전 JSONL(crop_path 없음)도 그대로 처리한다(빈 문자열).
    """
    doc = {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "collection_id": str(chunk.get("collection_id") or ""),
        "document_id": str(chunk.get("document_id") or ""),
        "screenshot_id": str(chunk.get("screenshot_id") or ""),
        "screenshot_index": int(chunk.get("screenshot_index") or 0),
        "region_id": str(chunk.get("region_id") or ""),
        "region_type": str(chunk.get("region_type") or "other"),
        "source_type": str(chunk.get("source_type") or "unknown"),
        "review_status": str(chunk.get("review_status") or "needs_review"),
        "confidence": float(chunk.get("confidence") or 0.0),
        "bbox": chunk.get("bbox") or {},
        "source_image": str(chunk.get("source_image") or ""),
        "crop_path": str(chunk.get("crop_path") or ""),
        "content": str(chunk.get("content") or ""),
        "parent_heading": str(chunk.get("parent_heading") or ""),
        "bm25_text": build_bm25_text(chunk),
    }
    if embedding is not None:
        doc["embedding"] = embedding
    return doc


def build_bulk_body(docs: list[dict], index_name: str) -> str:
    """_bulk NDJSON 본문(순수). 문서당 (action, source) 두 줄. chunk_id 를 _id 로
    써서 재색인이 idempotent(중복 대신 갱신)가 되게 한다."""
    lines: list[str] = []
    for doc in docs:
        action = {"index": {"_index": index_name}}
        chunk_id = doc.get("chunk_id")
        if chunk_id:
            action["index"]["_id"] = chunk_id
        lines.append(json.dumps(action, ensure_ascii=False))
        lines.append(json.dumps(doc, ensure_ascii=False))
    return "\n".join(lines) + "\n"


# --- 얇은 REST 클라이언트 ------------------------------------------------------


class OpenSearchClient:
    """requests 기반 최소 OpenSearch REST 클라이언트.

    transport: 테스트용 주입 지점 - (method, url, body_str|None, is_ndjson) ->
    (status_code, parsed_json). None 이면 requests 를 lazy import 해 실제 호출.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        username: str = "",
        password: str = "",
        timeout_sec: float = 60.0,
        transport=None,
    ):
        self.base_url = (base_url or resolve_opensearch_url()).rstrip("/")
        self.username = username or os.getenv("DOC_EXTRACT_OPENSEARCH_USER", "").strip()
        self.password = password or os.getenv(
            "DOC_EXTRACT_OPENSEARCH_PASSWORD", "").strip()
        self.timeout_sec = timeout_sec
        self._transport = transport

    def _request(self, method: str, path: str, *, body: str | None = None,
                 is_ndjson: bool = False) -> dict:
        url = f"{self.base_url}{path}"
        if self._transport is not None:
            status, parsed = self._transport(method, url, body, is_ndjson)
        else:
            import requests

            headers = {
                "Content-Type": (
                    "application/x-ndjson" if is_ndjson else "application/json"
                )
            }
            auth = (self.username, self.password) if self.username else None
            resp = requests.request(
                method, url, headers=headers, data=body, auth=auth,
                timeout=self.timeout_sec,
            )
            status = resp.status_code
            try:
                parsed = resp.json()
            except ValueError:
                parsed = {"raw": resp.text}
        if status >= 400:
            raise RuntimeError(
                f"OpenSearch {method} {path} 실패(status={status}): "
                f"{json.dumps(parsed, ensure_ascii=False)[:300]}"
            )
        return parsed

    def ping(self) -> bool:
        try:
            self._request("GET", "/")
            return True
        except Exception as exc:
            print(f"[WARNING] OpenSearch 연결 실패: {exc}")
            return False

    def index_exists(self, index_name: str) -> bool:
        try:
            self._request("GET", f"/{index_name}")
            return True
        except Exception:
            return False

    def create_index(self, index_name: str, mapping: dict) -> None:
        self._request("PUT", f"/{index_name}",
                      body=json.dumps(mapping, ensure_ascii=False))
        print(f"[INFO] 인덱스 생성: {index_name}")

    def delete_index(self, index_name: str) -> None:
        self._request("DELETE", f"/{index_name}")
        print(f"[INFO] 인덱스 삭제: {index_name}")

    def bulk(self, ndjson_body: str) -> dict:
        result = self._request("POST", "/_bulk", body=ndjson_body, is_ndjson=True)
        if result.get("errors"):
            failed = [
                item for item in result.get("items") or []
                if (item.get("index") or {}).get("status", 200) >= 400
            ]
            print(f"[WARNING] bulk 색인 일부 실패: {len(failed)}건")
        return result

    def search(self, index_name: str, query: dict) -> dict:
        return self._request(
            "POST", f"/{index_name}/_search",
            body=json.dumps(query, ensure_ascii=False),
        )

    def refresh(self, index_name: str) -> None:
        self._request("POST", f"/{index_name}/_refresh")


# --- 색인 오케스트레이션 --------------------------------------------------------


def index_chunks_jsonl(
    jsonl_path: Path,
    *,
    client: OpenSearchClient | None = None,
    embedder: EmbeddingClient | None = None,
    index_name: str | None = None,
    recreate: bool = False,
    batch_size: int = BULK_BATCH_SIZE,
) -> int:
    """rag_chunks.jsonl -> OpenSearch 색인. 색인한 문서 수를 반환.

    embedding_text 를 dense 벡터로 임베딩(offline 이면 stub)해 문서에 싣는다.
    """
    client = client or OpenSearchClient()
    embedder = embedder or EmbeddingClient()
    index_name = index_name or resolve_index_name()

    chunks = load_chunks(jsonl_path)
    if not chunks:
        print(f"[WARNING] 색인할 chunk 가 없습니다: {jsonl_path}")
        return 0

    if recreate and client.index_exists(index_name):
        client.delete_index(index_name)
    if not client.index_exists(index_name):
        client.create_index(index_name, build_index_mapping(dim=embedder.dim))

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [
            str(c.get("embedding_text") or c.get("content") or "") for c in batch
        ]
        vectors = embedder.embed_texts(texts)
        docs = [
            chunk_to_doc(chunk, embedding=vec)
            for chunk, vec in zip(batch, vectors)
        ]
        client.bulk(build_bulk_body(docs, index_name))
        total += len(docs)
        print(f"[INFO] 색인 진행: {total}/{len(chunks)}")

    client.refresh(index_name)
    print(f"[INFO] 색인 완료: {total} chunk -> {client.base_url}/{index_name}")
    return total


def main() -> int:
    if str(RAG_CHUNKS_JSONL) in {"", "."}:
        print("[ERROR] RAG_CHUNKS_JSONL 가 비어 있습니다. 상단 상수를 수정하세요.")
        return 1
    jsonl = RAG_CHUNKS_JSONL.expanduser().resolve()
    print(f"[INFO] RAG_CHUNKS_JSONL = {jsonl}")
    print(f"[INFO] OPENSEARCH      = {resolve_opensearch_url()}/{resolve_index_name()}")

    client = OpenSearchClient()
    if not client.ping():
        print("[ERROR] OpenSearch 에 연결할 수 없습니다 (사내 서버/env 확인).")
        return 1
    try:
        index_chunks_jsonl(jsonl, client=client, recreate=RECREATE_INDEX)
    except Exception as exc:
        print(f"[ERROR] 색인 중단: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BULK_BATCH_SIZE",
    "DEFAULT_INDEX_NAME",
    "DEFAULT_OPENSEARCH_URL",
    "OpenSearchClient",
    "build_bm25_text",
    "build_bulk_body",
    "build_index_mapping",
    "chunk_to_doc",
    "index_chunks_jsonl",
    "resolve_index_name",
    "resolve_opensearch_url",
]
