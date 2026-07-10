"""bge-m3 dense embedding 클라이언트 (offline 결정론 stub 포함).

Phase 1 (rag_chart_heavy_architecture.md): dense arm 은 사내 bge-m3 를
OpenAI-호환 embeddings API 로 호출한다. 사외/서버 부재 시에는 SHA-256 시드
기반의 결정론적 유사난수 벡터를 돌려주는 OFFLINE stub 으로 강등해, 색인/검색
배관(차원 일치, bulk 포맷, kNN 쿼리, RRF)을 모델 없이 end-to-end 검증한다
(StageRunner 의 offline stub 과 동일 철학 - stub 은 의미 검색이 아니라 배관 검증용).

OFFLINE 토글:
    - 환경변수 DOC_EXTRACT_OFFLINE=1  -> 강제 offline
    - DOC_EXTRACT_EMBED_API_URL 미설정 -> offline
    - 실제 호출 실패 시 자동 offline 폴백

env:
    DOC_EXTRACT_EMBED_API_URL   예: http://<사내 임베딩 서버>/v1  (미설정 = offline)
    DOC_EXTRACT_EMBED_MODEL     기본 "bge-m3"
    DOC_EXTRACT_EMBED_API_KEY   선택(Bearer)
"""

import hashlib
import math
import os
import struct


# bge-m3 dense 벡터 차원. opensearch_index 의 knn_vector dimension 과 단일 출처 공유.
EMBED_DIM = 1024

EMBED_MODEL_DEFAULT = "bge-m3"
EMBED_BATCH_SIZE = 16
EMBED_TIMEOUT_SEC = 60.0


def _offline_env() -> bool:
    return os.getenv("DOC_EXTRACT_OFFLINE", "").strip() in {"1", "true", "True"}


def _resolve_api_url() -> str:
    return os.getenv("DOC_EXTRACT_EMBED_API_URL", "").strip().rstrip("/")


def offline_embedding(text: str, *, dim: int = EMBED_DIM) -> list[float]:
    """텍스트 -> 결정론적 단위 벡터(순수). 같은 텍스트는 항상 같은 벡터.

    SHA-256(text) 를 시드로 counter 블록을 재해시해 dim 개의 float 을 만들고
    L2 정규화한다. 의미 유사도는 없지만(배관 검증용) 코사인 공간에서 안전하다.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(seed + struct.pack("<I", counter)).digest()
        # 32바이트 블록 -> uint32 8개 -> [-1, 1) float 8개
        for i in range(0, 32, 4):
            (u,) = struct.unpack("<I", block[i : i + 4])
            values.append((u / 2147483648.0) - 1.0)
            if len(values) >= dim:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class EmbeddingClient:
    """bge-m3 dense embedding 호출 래퍼 (offline 자동 폴백).

    OpenAI-호환 계약: POST {base}/embeddings {"model","input":[...]}
    -> {"data":[{"embedding":[...]}, ...]} (input 순서 보존).
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        dim: int = EMBED_DIM,
        offline: bool | None = None,
        timeout_sec: float = EMBED_TIMEOUT_SEC,
    ):
        self.api_url = (_resolve_api_url() if api_url is None else api_url).rstrip("/")
        self.model = (
            model or os.getenv("DOC_EXTRACT_EMBED_MODEL", "").strip()
            or EMBED_MODEL_DEFAULT
        )
        self.api_key = (
            api_key if api_key is not None
            else os.getenv("DOC_EXTRACT_EMBED_API_KEY", "").strip()
        )
        self.dim = dim
        self.timeout_sec = timeout_sec
        if offline is None:
            self.offline = _offline_env() or not self.api_url
        else:
            self.offline = bool(offline)
        if self.offline:
            print(f"[INFO] embedding offline stub 사용 (dim={self.dim})")

    def _embed_online(self, texts: list[str]) -> list[list[float]]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.api_url
        if not endpoint.endswith("/embeddings"):
            endpoint = f"{endpoint}/embeddings"
        resp = requests.post(
            endpoint,
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if len(data) != len(texts):
            raise ValueError(
                f"embedding 응답 개수 불일치: input={len(texts)}, data={len(data)}"
            )
        return [[float(v) for v in item.get("embedding") or []] for item in data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록 -> 벡터 목록(입력 순서 보존). 실패 시 offline 폴백."""
        if not texts:
            return []
        if self.offline:
            return [offline_embedding(t, dim=self.dim) for t in texts]

        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start : start + EMBED_BATCH_SIZE]
                vectors.extend(self._embed_online(batch))
        except Exception as exc:
            print(f"[WARNING] embedding 호출 실패 -> offline stub 폴백: {exc}")
            self.offline = True
            return [offline_embedding(t, dim=self.dim) for t in texts]
        return vectors

    def embed_one(self, text: str) -> list[float]:
        """단일 텍스트 -> 벡터."""
        return self.embed_texts([text])[0]


__all__ = ["EMBED_DIM", "EmbeddingClient", "offline_embedding"]
