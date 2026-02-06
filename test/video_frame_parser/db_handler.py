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
from .models import (
    VideoMetadata, FrameData, AnalysisResult, AnalysisStatus,
    OCRResult, ActionSequence, ErrorPattern, UIElement
)

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

        # RAG 관련 추가 인덱스
        self._text_faiss_index = None  # 텍스트 임베딩용 FAISS 인덱스
        self._text_embedding_dim = 1024  # bge-m3 임베딩 차원
        self._text_id_mapping: Dict[int, str] = {}  # FAISS ID -> text_id 매핑

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

        # RAG 관련 컬렉션 인덱스
        # OCR 결과 인덱스
        ocr = self._db["ocr_results"]
        ocr.create_indexes([
            IndexModel([("ocr_id", ASCENDING)], unique=True),
            IndexModel([("frame_id", ASCENDING)]),
            IndexModel([("confidence", DESCENDING)]),
        ])

        # 작업 시퀀스 인덱스
        actions = self._db["action_sequences"]
        actions.create_indexes([
            IndexModel([("action_id", ASCENDING)], unique=True),
            IndexModel([("video_id", ASCENDING)]),
            IndexModel([("start_time", ASCENDING)]),
            IndexModel([("success", ASCENDING)]),
        ])

        # 에러 패턴 인덱스
        errors = self._db["error_patterns"]
        errors.create_indexes([
            IndexModel([("error_id", ASCENDING)], unique=True),
            IndexModel([("frame_id", ASCENDING)]),
            IndexModel([("error_type", ASCENDING)]),
            IndexModel([("severity", ASCENDING)]),
        ])

        # UI 요소 인덱스
        ui_elements = self._db["ui_elements"]
        ui_elements.create_indexes([
            IndexModel([("element_id", ASCENDING)], unique=True),
            IndexModel([("frame_id", ASCENDING)]),
            IndexModel([("element_type", ASCENDING)]),
        ])

        # 텍스트 임베딩 인덱스
        text_embeddings = self._db["text_embeddings"]
        text_embeddings.create_indexes([
            IndexModel([("text_id", ASCENDING)], unique=True),
            IndexModel([("frame_id", ASCENDING)]),
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

    # ========== RAG-specific Operations ==========

    def save_ocr_results_batch(self, ocr_results: List[OCRResult]) -> int:
        """OCR 결과 배치 저장"""
        if not MONGO_AVAILABLE or self._db is None or not ocr_results:
            return 0

        collection = self._db["ocr_results"]
        operations = []

        from pymongo import UpdateOne
        for ocr in ocr_results:
            operations.append(
                UpdateOne(
                    {"ocr_id": ocr.ocr_id},
                    {"$set": ocr.to_dict()},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count

    def get_ocr_results_by_frame(self, frame_id: str) -> List[OCRResult]:
        """프레임의 OCR 결과 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db["ocr_results"]
        cursor = collection.find({"frame_id": frame_id})

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(OCRResult.from_dict(doc))

        return results

    def save_action_sequence(self, action: ActionSequence) -> str:
        """작업 시퀀스 저장"""
        if not MONGO_AVAILABLE or self._db is None:
            return action.action_id

        collection = self._db["action_sequences"]
        doc = action.to_dict()

        collection.update_one(
            {"action_id": action.action_id},
            {"$set": doc},
            upsert=True
        )

        return action.action_id

    def get_action_sequences_by_video(self, video_id: str) -> List[ActionSequence]:
        """비디오의 작업 시퀀스 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db["action_sequences"]
        cursor = collection.find({"video_id": video_id}).sort("start_time", ASCENDING)

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(ActionSequence.from_dict(doc))

        return results

    def get_action_sequences_by_frame(self, frame_id: str) -> List[ActionSequence]:
        """특정 프레임과 연관된 작업 시퀀스 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db["action_sequences"]
        cursor = collection.find({
            "$or": [
                {"start_frame_id": frame_id},
                {"end_frame_id": frame_id}
            ]
        })

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(ActionSequence.from_dict(doc))

        return results

    def save_error_patterns_batch(self, errors: List[ErrorPattern]) -> int:
        """에러 패턴 배치 저장"""
        if not MONGO_AVAILABLE or self._db is None or not errors:
            return 0

        collection = self._db["error_patterns"]
        operations = []

        from pymongo import UpdateOne
        for error in errors:
            operations.append(
                UpdateOne(
                    {"error_id": error.error_id},
                    {"$set": error.to_dict()},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count

    def get_error_patterns_by_frame(self, frame_id: str) -> List[ErrorPattern]:
        """프레임의 에러 패턴 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db["error_patterns"]
        cursor = collection.find({"frame_id": frame_id})

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(ErrorPattern.from_dict(doc))

        return results

    def save_ui_elements_batch(self, elements: List[UIElement]) -> int:
        """UI 요소 배치 저장"""
        if not MONGO_AVAILABLE or self._db is None or not elements:
            return 0

        collection = self._db["ui_elements"]
        operations = []

        from pymongo import UpdateOne
        for element in elements:
            operations.append(
                UpdateOne(
                    {"element_id": element.element_id},
                    {"$set": element.to_dict()},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count

    def get_ui_elements_by_frame(self, frame_id: str) -> List[UIElement]:
        """프레임의 UI 요소 조회"""
        if not MONGO_AVAILABLE or self._db is None:
            return []

        collection = self._db["ui_elements"]
        cursor = collection.find({"frame_id": frame_id})

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(UIElement.from_dict(doc))

        return results

    # ========== Text Embedding Operations ==========

    def init_text_faiss_index(self, embedding_dim: int = 1024) -> None:
        """텍스트 임베딩용 FAISS 인덱스 초기화"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available")
            return

        self._text_embedding_dim = embedding_dim

        # IVF 인덱스 사용
        quantizer = faiss.IndexFlatIP(self._text_embedding_dim)
        self._text_faiss_index = faiss.IndexIVFFlat(
            quantizer,
            self._text_embedding_dim,
            self.config.faiss_nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        logger.info(f"Text FAISS index initialized with dimension {embedding_dim}")

        # 기존 인덱스 로드 시도
        if self.config.faiss_index_path:
            self._load_text_faiss_index()

    def _load_text_faiss_index(self) -> bool:
        """저장된 텍스트 FAISS 인덱스 로드"""
        if not self.config.faiss_index_path:
            return False

        index_path = Path(self.config.faiss_index_path)
        text_index_file = index_path / "text_index.faiss"

        if not text_index_file.exists():
            return False

        try:
            self._text_faiss_index = faiss.read_index(str(text_index_file))

            # ID 매핑 로드
            import pickle
            mapping_path = index_path / "text_id_mapping.pkl"
            if mapping_path.exists():
                with open(mapping_path, "rb") as f:
                    self._text_id_mapping = pickle.load(f)

            logger.info(f"Loaded text FAISS index with {self._text_faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            logger.warning(f"Failed to load text FAISS index: {e}")
            return False

    def save_text_faiss_index(self) -> None:
        """텍스트 FAISS 인덱스 저장"""
        if not FAISS_AVAILABLE or self._text_faiss_index is None:
            return

        if not self.config.faiss_index_path:
            return

        index_path = Path(self.config.faiss_index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._text_faiss_index, str(index_path / "text_index.faiss"))

        # ID 매핑 저장
        import pickle
        with open(index_path / "text_id_mapping.pkl", "wb") as f:
            pickle.dump(self._text_id_mapping, f)

        logger.info(f"Saved text FAISS index with {self._text_faiss_index.ntotal} vectors")

    def add_text_embeddings_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        frame_ids: List[str]
    ) -> int:
        """텍스트 임베딩 배치 추가

        Args:
            texts: 텍스트 리스트
            embeddings: 임베딩 행렬 (N, 1024)
            frame_ids: 프레임 ID 리스트

        Returns:
            추가된 임베딩 수
        """
        if not FAISS_AVAILABLE or self._text_faiss_index is None:
            logger.warning("Text FAISS index not initialized")
            return 0

        if len(texts) != len(frame_ids) or len(texts) != len(embeddings):
            logger.error("texts, embeddings, and frame_ids must have the same length")
            return 0

        # 정규화
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # 인덱스 학습 (처음만)
        if not self._text_faiss_index.is_trained:
            if len(embeddings) < self.config.faiss_nlist:
                logger.warning(f"Need at least {self.config.faiss_nlist} embeddings to train index")
                return 0
            self._text_faiss_index.train(embeddings)

        # 임베딩 추가
        start_id = self._text_faiss_index.ntotal
        self._text_faiss_index.add(embeddings)

        # MongoDB에 메타데이터 저장
        if MONGO_AVAILABLE and self._db is not None:
            collection = self._db["text_embeddings"]
            from pymongo import UpdateOne
            operations = []

            for i, (text, frame_id) in enumerate(zip(texts, frame_ids)):
                text_id = f"text_{start_id + i}"
                self._text_id_mapping[start_id + i] = text_id

                operations.append(
                    UpdateOne(
                        {"text_id": text_id},
                        {"$set": {
                            "text_id": text_id,
                            "frame_id": frame_id,
                            "text": text,
                            "faiss_id": start_id + i
                        }},
                        upsert=True
                    )
                )

            if operations:
                collection.bulk_write(operations)

        logger.debug(f"Added {len(embeddings)} text embeddings")
        return len(embeddings)

    def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """텍스트 임베딩으로 검색

        Args:
            query_embedding: 쿼리 임베딩 (1024-dim)
            top_k: 반환할 결과 수

        Returns:
            (frame_id, similarity_score) 튜플 리스트
        """
        if not FAISS_AVAILABLE or self._text_faiss_index is None:
            return []

        if not self._text_faiss_index.is_trained:
            logger.warning("Text FAISS index not trained yet")
            return []

        # 정규화
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # 검색
        self._text_faiss_index.nprobe = self.config.faiss_nprobe
        distances, indices = self._text_faiss_index.search(query_embedding, top_k)

        # 결과 변환
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            text_id = self._text_id_mapping.get(idx)
            if text_id and MONGO_AVAILABLE and self._db is not None:
                # MongoDB에서 frame_id 조회
                collection = self._db["text_embeddings"]
                doc = collection.find_one({"text_id": text_id})
                if doc:
                    results.append((doc["frame_id"], float(dist)))

        return results

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

            # RAG 관련 통계
            stats["ocr_results_count"] = self._db["ocr_results"].count_documents({})
            stats["action_sequences_count"] = self._db["action_sequences"].count_documents({})
            stats["error_patterns_count"] = self._db["error_patterns"].count_documents({})
            stats["ui_elements_count"] = self._db["ui_elements"].count_documents({})
            stats["text_embeddings_count"] = self._db["text_embeddings"].count_documents({})

        if FAISS_AVAILABLE and self._faiss_index is not None:
            stats["embeddings_count"] = self._faiss_index.ntotal
            stats["faiss_trained"] = self._faiss_index.is_trained

        if FAISS_AVAILABLE and self._text_faiss_index is not None:
            stats["text_faiss_count"] = self._text_faiss_index.ntotal
            stats["text_faiss_trained"] = self._text_faiss_index.is_trained

        return stats

    def close(self) -> None:
        """연결 종료"""
        # FAISS 인덱스 저장
        self.save_faiss_index()
        self.save_text_faiss_index()

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
