"""
OpenSearch 저장소 핸들러

레시피 자동화 지식(프레임 임베딩, 워크플로우)을 OpenSearch에 저장하고 검색.

Usage:
    from poc.config import PocConfig
    from poc.opensearch_handler import OpenSearchHandler

    config = PocConfig.load()
    handler = OpenSearchHandler(config.opensearch)
    handler.initialize()
"""

from typing import List, Dict, Any, Optional
from dataclasses import asdict

# opensearch-py 임포트 가드
try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False


# 임베딩 차원 (CLIP ViT-L/14 기본값)
EMBEDDING_DIM = 768


class OpenSearchHandler:
    """OpenSearch 벡터/텍스트 검색 핸들러"""

    def __init__(self, config):
        """
        Args:
            config: OpenSearchConfig 인스턴스
        """
        if not OPENSEARCH_AVAILABLE:
            raise ImportError(
                "opensearch-py 패키지가 필요합니다: pip install opensearch-py"
            )

        self.config = config
        self.client: Optional[OpenSearch] = None
        self.index_name = config.index

    def initialize(self):
        """OpenSearch 연결 및 인덱스 생성"""
        auth = None
        if self.config.username and self.config.password:
            auth = (self.config.username, self.config.password)

        self.client = OpenSearch(
            hosts=[self.config.url],
            http_auth=auth,
            use_ssl=self.config.url.startswith("https"),
            verify_certs=False,
            connection_class=RequestsHttpConnection,
        )

        # 연결 확인
        info = self.client.info()
        print(f"[INFO] OpenSearch 연결 성공: {info['version']['distribution']} "
              f"{info['version']['number']}")

        # kNN 인덱스 생성 (없을 경우)
        if not self.client.indices.exists(index=self.index_name):
            self._create_knn_index()
            print(f"[INFO] 인덱스 '{self.index_name}' 생성 완료")
        else:
            print(f"[INFO] 인덱스 '{self.index_name}' 이미 존재")

    def _create_knn_index(self):
        """kNN 벡터 검색이 가능한 인덱스 생성"""
        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "frame_id": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24,
                            },
                        },
                    },
                    "timestamp": {"type": "date"},
                    "source_video": {"type": "keyword"},
                    "frame_type": {"type": "keyword"},
                    "description": {"type": "text", "analyzer": "standard"},
                    "actions": {"type": "text", "analyzer": "standard"},
                    "metadata": {"type": "object", "enabled": True},
                    # 워크플로우 전용 필드
                    "workflow_id": {"type": "keyword"},
                    "workflow_name": {"type": "text"},
                    "steps": {"type": "nested"},
                }
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def store_frame(
        self,
        frame_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> str:
        """
        프레임 문서 저장

        Args:
            frame_id: 프레임 고유 ID
            embedding: 벡터 임베딩 (EMBEDDING_DIM 차원)
            metadata: 추가 메타데이터

        Returns:
            문서 ID
        """
        doc = {
            "frame_id": frame_id,
            "embedding": embedding,
            **metadata,
        }

        result = self.client.index(
            index=self.index_name,
            id=frame_id,
            body=doc,
            refresh="wait_for",
        )
        return result["_id"]

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        kNN 벡터 유사도 검색

        Args:
            query_vector: 쿼리 벡터
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트 [{_id, _score, _source}, ...]
        """
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k,
                    }
                }
            },
        }

        result = self.client.search(index=self.index_name, body=body)
        return [
            {
                "_id": hit["_id"],
                "_score": hit["_score"],
                "_source": hit["_source"],
            }
            for hit in result["hits"]["hits"]
        ]

    def search_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        전문(full-text) 검색

        Args:
            query_text: 검색할 텍스트
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트
        """
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "description^2",
                        "actions",
                        "workflow_name",
                    ],
                }
            },
        }

        result = self.client.search(index=self.index_name, body=body)
        return [
            {
                "_id": hit["_id"],
                "_score": hit["_score"],
                "_source": hit["_source"],
            }
            for hit in result["hits"]["hits"]
        ]

    def store_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """
        워크플로우 데이터 저장

        Args:
            workflow_data: 워크플로우 딕셔너리 (workflow_id, workflow_name, steps, ...)

        Returns:
            문서 ID
        """
        doc_id = workflow_data.get("workflow_id", None)

        result = self.client.index(
            index=self.index_name,
            id=doc_id,
            body=workflow_data,
            refresh="wait_for",
        )
        return result["_id"]

    def retrieve_context(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        통합 RAG 컨텍스트 검색 (벡터 + 텍스트)

        Args:
            query_vector: 쿼리 벡터 (None이면 텍스트만)
            query_text: 쿼리 텍스트 (None이면 벡터만)
            top_k: 반환할 최대 결과 수

        Returns:
            중복 제거된 검색 결과 리스트
        """
        results_map: Dict[str, Dict[str, Any]] = {}

        # 벡터 검색
        if query_vector:
            for hit in self.search_similar(query_vector, top_k):
                doc_id = hit["_id"]
                results_map[doc_id] = hit

        # 텍스트 검색
        if query_text:
            for hit in self.search_text(query_text, top_k):
                doc_id = hit["_id"]
                if doc_id not in results_map:
                    results_map[doc_id] = hit

        # 점수 기준 정렬
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x["_score"],
            reverse=True,
        )

        return sorted_results[:top_k]

    def close(self):
        """연결 종료"""
        if self.client:
            self.client.close()
            print("[INFO] OpenSearch 연결 종료")
