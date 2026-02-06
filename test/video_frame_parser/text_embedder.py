"""
Text Embedder using bge-m3

텍스트 임베딩 생성을 위한 bge-m3 모델 래퍼
"""

from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available. Install with: pip install sentence-transformers")


class TextEmbedder:
    """
    bge-m3 기반 텍스트 임베딩 생성기

    1024차원 임베딩 생성 (semantic search용)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True
    ):
        """
        Args:
            model_name: 사용할 모델 이름
            device: 디바이스 ("cpu" 또는 "cuda")
            normalize: 임베딩 정규화 여부
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.embedding_dim = 1024  # bge-m3 embedding dimension

        print(f"[INFO] Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[INFO] Model loaded successfully on {device}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        단일 텍스트의 임베딩 생성

        Args:
            text: 입력 텍스트

        Returns:
            1024차원 임베딩 벡터
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        배치 텍스트의 임베딩 생성

        Args:
            texts: 입력 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            (N, 1024) 임베딩 행렬
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        검색 쿼리의 임베딩 생성 (embed_text의 alias)

        Args:
            query: 검색 쿼리

        Returns:
            1024차원 임베딩 벡터
        """
        return self.embed_text(query)

    def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        두 텍스트(또는 임베딩) 간의 유사도 계산

        Args:
            text1: 텍스트 또는 임베딩
            text2: 텍스트 또는 임베딩

        Returns:
            코사인 유사도 (-1 ~ 1)
        """
        # 텍스트인 경우 임베딩 생성
        emb1 = self.embed_text(text1) if isinstance(text1, str) else text1
        emb2 = self.embed_text(text2) if isinstance(text2, str) else text2

        # 정규화된 벡터의 내적 = 코사인 유사도
        if self.normalize:
            similarity = np.dot(emb1, emb2)
        else:
            # 정규화되지 않은 경우 명시적 계산
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

        return float(similarity)

    def compute_similarity_matrix(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        쿼리와 문서 간의 유사도 행렬 계산

        Args:
            queries: 쿼리 리스트
            documents: 문서 리스트
            batch_size: 배치 크기

        Returns:
            (len(queries), len(documents)) 유사도 행렬
        """
        query_embeddings = self.embed_batch(queries, batch_size=batch_size)
        doc_embeddings = self.embed_batch(documents, batch_size=batch_size)

        # 내적 계산 (정규화된 경우 코사인 유사도)
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)

        return similarity_matrix

    @property
    def dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim


def create_text_embedder(
    device: Optional[str] = None,
    use_gpu: bool = True
) -> TextEmbedder:
    """
    TextEmbedder 생성 헬퍼 함수

    Args:
        device: 디바이스 지정 (None이면 자동 선택)
        use_gpu: GPU 사용 여부

    Returns:
        TextEmbedder 인스턴스
    """
    if device is None:
        import torch
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return TextEmbedder(device=device)


if __name__ == "__main__":
    # 사용 예시
    embedder = create_text_embedder(use_gpu=False)

    # 단일 텍스트 임베딩
    text = "RCS 서버에 연결합니다"
    embedding = embedder.embed_text(text)
    print(f"[INFO] Text: {text}")
    print(f"[INFO] Embedding shape: {embedding.shape}")
    print(f"[INFO] Embedding norm: {np.linalg.norm(embedding):.4f}")

    # 배치 임베딩
    texts = [
        "RCS 로그인 화면",
        "서버 연결 실패",
        "사용자 인증 진행 중"
    ]
    embeddings = embedder.embed_batch(texts, show_progress=True)
    print(f"\n[INFO] Batch embeddings shape: {embeddings.shape}")

    # 유사도 계산
    query = "RCS 접속"
    for i, doc in enumerate(texts):
        similarity = embedder.compute_similarity(query, doc)
        print(f"[INFO] Similarity('{query}', '{doc}'): {similarity:.4f}")
