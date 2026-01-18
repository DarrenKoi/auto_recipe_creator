"""
Database Handler

MongoDB 및 FAISS를 사용한 데이터 저장 및 검색 모듈.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
    from pymongo.collection import Collection
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .config import DatabaseConfig
from .models import VideoMetadata, FrameData, AnalysisResult, AnalysisStatus

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    데이터베이스 핸들러.

    MongoDB: 메타데이터 및 분석 결과 저장
    FAISS: 임베딩 벡터 인덱스 및 검색
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._client: Optional[MongoClient] = None
        self._db = None
        self._faiss_index = None
        self._embedding_dim = 512  # 기본 CLIP 임베딩 차원
        self._id_mapping: Dict[int, str] = {}  # FAISS ID -> frame_id 매핑
        self._initialized = False

    def initialize(self, embedding_dim: int = 512) -> None:
        """데이터베이스 초기화"""
        if self._initialized:
            return

        self._embedding_dim = embedding_dim

        # MongoDB 연결
        if MONGO_AVAILABLE:
            self._init_mongodb()
        else:
            logger.warning("pymongo not installed. MongoDB features disabled.")

        # FAISS 인덱스 초기화
        if FAISS_AVAILABLE:
            self._init_faiss()
        else:
            logger.warning("faiss not installed. Vector search disabled.")

        self._initialized = True
        logger.info("DatabaseHandler initialized")

    def _init_mongodb(self) -> None:
        """MongoDB 연결 및 컬렉션 설정"""
        try:
            self._client = MongoClient(self.config.mongo_uri)
            self._db = self._client[self.config.database_name]

            # 컬렉션 인덱스 생성
            self._create_indexes()

            logger.info(f"Connected to MongoDB: {self.config.database_name}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def _create_indexes(self) -> None:
        """컬렉션 인덱스 생성"""
        # 비디오 메타데이터 인덱스
        videos = self._db[self.config.videos_collection]
        videos.create_indexes([
            IndexModel([("video_id", ASCENDING)], unique=True),
            IndexModel([("created_at", DESCENDING)]),
        ])

        # 프레임 데이터 인덱스
        frames = self._db[self.config.frames_collection]
        frames.create_indexes([
            IndexModel([("frame_id", ASCENDING)], unique=True),
            IndexModel([("video_id", ASCENDING), ("frame_number", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)]),
        ])

        # 분석 결과 인덱스
        results = self._db[self.config.results_collection]
        results.create_indexes([
            IndexModel([("result_id", ASCENDING)], unique=True),
            IndexModel([("frame_id", ASCENDING)]),
            IndexModel([("video_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
        ])

    def _init_faiss(self) -> None:
        """FAISS 인덱스 초기화"""
        # IVF 인덱스 사용 (대규모 검색에 효율적)
        quantizer = faiss.IndexFlatIP(self._embedding_dim)
        self._faiss_index = faiss.IndexIVFFlat(
            quantizer,
            self._embedding_dim,
            self.config.faiss_nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        # 기존 인덱스 로드 시도
        if self.config.faiss_index_path:
            self._load_faiss_index()

    def _load_faiss_index(self) -> bool:
        """저장된 FAISS 인덱스 로드"""
        index_path = Path(self.config.faiss_index_path)

        if not index_path.exists():
            return False

        try:
            self._faiss_index = faiss.read_index(str(index_path / "index.faiss"))

            # ID 매핑 로드
            import pickle
            mapping_path = index_path / "id_mapping.pkl"
            if mapping_path.exists():
                with open(mapping_path, "rb") as f:
                    self._id_mapping = pickle.load(f)

            logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            return False

    def save_faiss_index(self) -> None:
        """FAISS 인덱스 저장"""
        if not FAISS_AVAILABLE or self._faiss_index is None:
            return

        if not self.config.faiss_index_path:
            return

        index_path = Path(self.config.faiss_index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(index_path / "index.faiss"))

        # ID 매핑 저장
        import pickle
        with open(index_path / "id_mapping.pkl", "wb") as f:
            pickle.dump(self._id_mapping, f)

        logger.info(f"Saved FAISS index with {self._faiss_index.ntotal} vectors")

    # ========== Video Metadata Operations ==========

    def save_video_metadata(self, metadata: VideoMetadata) -> str:
        """비디오 메타데이터 저장"""
        if not MONGO_AVAILABLE or self._db is None:
            logger.warning("MongoDB not available")
            return metadata.video_id

        collection = self._db[self.config.videos_collection]
        doc = metadata.to_dict()

        collection.update_one(
            {"video_id": metadata.video_id},
            {"$set": doc},
            upsert=True
        )

        logger.debug(f"Saved video metadata: {metadata.video_id}")
        return metadata.video_id

    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """비디오 메타데이터 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return None

        collection = self._db[self.config.videos_collection]
        doc = collection.find_one({"video_id": video_id})

        if doc:
            doc.pop("_id", None)
            return VideoMetadata.from_dict(doc)
        return None

    def list_videos(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> List[VideoMetadata]:
        """비디오 목록 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db[self.config.videos_collection]
        cursor = collection.find().sort("created_at", DESCENDING).skip(skip).limit(limit)

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(VideoMetadata.from_dict(doc))

        return results

    # ========== Frame Data Operations ==========

    def save_frame(self, frame: FrameData) -> str:
        """프레임 데이터 저장"""
        if not MONGO_AVAILABLE or self._db is None:
            logger.warning("MongoDB not available")
            return frame.frame_id

        collection = self._db[self.config.frames_collection]
        doc = frame.to_dict()

        collection.update_one(
            {"frame_id": frame.frame_id},
            {"$set": doc},
            upsert=True
        )

        return frame.frame_id

    def save_frames_batch(self, frames: List[FrameData]) -> int:
        """프레임 배치 저장"""
        if not MONGO_AVAILABLE or self._db is None or not frames:
            return 0

        collection = self._db[self.config.frames_collection]
        operations = []

        from pymongo import UpdateOne
        for frame in frames:
            operations.append(
                UpdateOne(
                    {"frame_id": frame.frame_id},
                    {"$set": frame.to_dict()},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count

    def get_frame(self, frame_id: str) -> Optional[FrameData]:
        """프레임 데이터 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return None

        collection = self._db[self.config.frames_collection]
        doc = collection.find_one({"frame_id": frame_id})

        if doc:
            doc.pop("_id", None)
            return FrameData.from_dict(doc)
        return None

    def get_frames_by_video(
        self,
        video_id: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> List[FrameData]:
        """비디오의 프레임 목록 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db[self.config.frames_collection]
        query = {"video_id": video_id, "frame_number": {"$gte": start_frame}}

        if end_frame is not None:
            query["frame_number"]["$lte"] = end_frame

        cursor = collection.find(query).sort("frame_number", ASCENDING)

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(FrameData.from_dict(doc))

        return results

    # ========== Analysis Result Operations ==========

    def save_analysis_result(self, result: AnalysisResult) -> str:
        """분석 결과 저장"""
        if not MONGO_AVAILABLE or self._db is None:
            logger.warning("MongoDB not available")
            return result.result_id

        collection = self._db[self.config.results_collection]
        doc = result.to_dict()

        collection.update_one(
            {"result_id": result.result_id},
            {"$set": doc},
            upsert=True
        )

        # 임베딩을 FAISS에 추가
        if result.embedding is not None and FAISS_AVAILABLE:
            self._add_embedding_to_index(result.frame_id, result.embedding)

        return result.result_id

    def save_analysis_results_batch(
        self,
        results: List[AnalysisResult],
    ) -> int:
        """분석 결과 배치 저장"""
        if not results:
            return 0

        saved_count = 0

        # MongoDB 저장
        if MONGO_AVAILABLE and self._db is not None:
            collection = self._db[self.config.results_collection]
            operations = []

            from pymongo import UpdateOne
            for result in results:
                operations.append(
                    UpdateOne(
                        {"result_id": result.result_id},
                        {"$set": result.to_dict()},
                        upsert=True
                    )
                )

            bulk_result = collection.bulk_write(operations)
            saved_count = bulk_result.upserted_count + bulk_result.modified_count

        # FAISS 인덱스 업데이트
        if FAISS_AVAILABLE and self._faiss_index is not None:
            embeddings = []
            frame_ids = []

            for result in results:
                if result.embedding is not None:
                    embeddings.append(result.embedding)
                    frame_ids.append(result.frame_id)

            if embeddings:
                self._add_embeddings_batch(frame_ids, embeddings)

        return saved_count

    def get_analysis_result(self, result_id: str) -> Optional[AnalysisResult]:
        """분석 결과 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return None

        collection = self._db[self.config.results_collection]
        doc = collection.find_one({"result_id": result_id})

        if doc:
            doc.pop("_id", None)
            return AnalysisResult.from_dict(doc)
        return None

    def get_results_by_video(
        self,
        video_id: str,
        status: Optional[AnalysisStatus] = None,
    ) -> List[AnalysisResult]:
        """비디오의 분석 결과 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db[self.config.results_collection]
        query = {"video_id": video_id}

        if status:
            query["status"] = status.value

        cursor = collection.find(query)

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(AnalysisResult.from_dict(doc))

        return results

    # ========== Vector Search Operations ==========

    def _add_embedding_to_index(self, frame_id: str, embedding: np.ndarray) -> None:
        """FAISS 인덱스에 임베딩 추가"""
        if not FAISS_AVAILABLE or self._faiss_index is None:
            return

        # 인덱스 학습 여부 확인
        if not self._faiss_index.is_trained:
            logger.warning("FAISS index not trained. Skipping embedding.")
            return

        idx = self._faiss_index.ntotal
        embedding = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(embedding)
        self._faiss_index.add(embedding)
        self._id_mapping[idx] = frame_id

    def _add_embeddings_batch(
        self,
        frame_ids: List[str],
        embeddings: List[np.ndarray],
    ) -> None:
        """FAISS 인덱스에 임베딩 배치 추가"""
        if not FAISS_AVAILABLE or self._faiss_index is None:
            return

        if not embeddings:
            return

        # 학습 안 됐으면 학습 수행
        if not self._faiss_index.is_trained:
            train_data = np.array(embeddings).astype(np.float32)
            faiss.normalize_L2(train_data)
            self._faiss_index.train(train_data)
            logger.info("FAISS index trained")

        start_idx = self._faiss_index.ntotal
        vectors = np.array(embeddings).astype(np.float32)
        faiss.normalize_L2(vectors)
        self._faiss_index.add(vectors)

        for i, frame_id in enumerate(frame_ids):
            self._id_mapping[start_idx + i] = frame_id

    def search_similar_frames(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        유사 프레임 검색.

        Args:
            query_embedding: 쿼리 임베딩
            top_k: 반환할 최대 개수
            threshold: 최소 유사도

        Returns:
            (frame_id, similarity) 튜플 목록
        """
        if not FAISS_AVAILABLE or self._faiss_index is None:
            return []

        if self._faiss_index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # nprobe 설정 (검색 정확도/속도 조절)
        if hasattr(self._faiss_index, 'nprobe'):
            self._faiss_index.nprobe = min(10, self.config.faiss_nlist)

        distances, indices = self._faiss_index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if dist < threshold:
                continue

            frame_id = self._id_mapping.get(idx)
            if frame_id:
                results.append((frame_id, float(dist)))

        return results

    def get_embedding_by_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """프레임 ID로 임베딩 조회"""
        # ID 매핑에서 인덱스 찾기
        for idx, fid in self._id_mapping.items():
            if fid == frame_id:
                if FAISS_AVAILABLE and self._faiss_index is not None:
                    return self._faiss_index.reconstruct(idx)
        return None

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {
            "initialized": self._initialized,
            "mongodb_available": MONGO_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
        }

        if MONGO_AVAILABLE and self._db is not None:
            stats["videos_count"] = self._db[self.config.videos_collection].count_documents({})
            stats["frames_count"] = self._db[self.config.frames_collection].count_documents({})
            stats["results_count"] = self._db[self.config.results_collection].count_documents({})

        if FAISS_AVAILABLE and self._faiss_index is not None:
            stats["embeddings_count"] = self._faiss_index.ntotal
            stats["faiss_trained"] = self._faiss_index.is_trained

        return stats

    def close(self) -> None:
        """연결 종료"""
        # FAISS 인덱스 저장
        self.save_faiss_index()

        # MongoDB 연결 종료
        if self._client:
            self._client.close()
            self._client = None

        self._initialized = False
        logger.info("DatabaseHandler closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
