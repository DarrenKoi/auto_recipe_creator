"""
Frame Analyzer

프레임 분석 및 CLIP 임베딩 생성 모듈.
H200 GPU 클러스터 활용 최적화.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from PIL import Image

from .config import AnalyzerConfig
from .models import FrameData, AnalysisResult, AnalysisStatus

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """
    프레임 분석기.

    CLIP 모델을 사용하여 프레임의 임베딩을 생성하고,
    상태 매칭 및 UI 요소 감지를 수행합니다.
    """

    SUPPORTED_MODELS = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self.model = None
        self.preprocess = None
        self.device = None
        self._initialized = False

    def initialize(self) -> None:
        """모델 초기화"""
        if self._initialized:
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")

        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git")

        # 디바이스 설정
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # GPU 정보 로깅
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")

        # CLIP 모델 로드
        model_name = self.config.clip_model
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self._initialized = True
        logger.info("FrameAnalyzer initialized successfully")

    def analyze_frame(self, frame_data: FrameData) -> AnalysisResult:
        """
        단일 프레임 분석.

        Args:
            frame_data: 분석할 프레임 데이터

        Returns:
            AnalysisResult: 분석 결과
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.now()
        result_id = f"{frame_data.frame_id}_result"

        try:
            # 이미지 전처리
            image = self._prepare_image(frame_data)

            # 임베딩 생성
            embedding = self._generate_embedding(image)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return AnalysisResult(
                result_id=result_id,
                frame_id=frame_data.frame_id,
                video_id=frame_data.video_id,
                embedding=embedding,
                embedding_model=self.config.clip_model,
                status=AnalysisStatus.COMPLETED,
                analyzed_at=datetime.now(),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return AnalysisResult(
                result_id=result_id,
                frame_id=frame_data.frame_id,
                video_id=frame_data.video_id,
                status=AnalysisStatus.FAILED,
                error_message=str(e),
                analyzed_at=datetime.now(),
            )

    def analyze_frames_batch(
        self, frames: List[FrameData]
    ) -> List[AnalysisResult]:
        """
        프레임 배치 분석.

        Args:
            frames: 분석할 프레임 목록

        Returns:
            분석 결과 목록
        """
        if not self._initialized:
            self.initialize()

        if not frames:
            return []

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(frames) + batch_size - 1) // batch_size}")

        return results

    def _process_batch(self, frames: List[FrameData]) -> List[AnalysisResult]:
        """배치 처리"""
        start_time = datetime.now()
        results = []

        try:
            # 이미지 배치 준비
            images = []
            valid_indices = []

            for idx, frame in enumerate(frames):
                try:
                    image = self._prepare_image(frame)
                    images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to prepare image for frame {frame.frame_id}: {e}")
                    results.append(AnalysisResult(
                        result_id=f"{frame.frame_id}_result",
                        frame_id=frame.frame_id,
                        video_id=frame.video_id,
                        status=AnalysisStatus.FAILED,
                        error_message=str(e),
                    ))

            if not images:
                return results

            # 배치 임베딩 생성
            image_batch = torch.stack(images).to(self.device)

            with torch.no_grad():
                embeddings = self.model.encode_image(image_batch)
                embeddings = F.normalize(embeddings, dim=-1)
                embeddings = embeddings.cpu().numpy()

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            per_frame_time = processing_time / len(valid_indices)

            # 결과 생성
            for i, idx in enumerate(valid_indices):
                frame = frames[idx]
                results.insert(idx, AnalysisResult(
                    result_id=f"{frame.frame_id}_result",
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    embedding=embeddings[i],
                    embedding_model=self.config.clip_model,
                    status=AnalysisStatus.COMPLETED,
                    analyzed_at=datetime.now(),
                    processing_time_ms=per_frame_time,
                ))

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for frame in frames:
                if not any(r.frame_id == frame.frame_id for r in results):
                    results.append(AnalysisResult(
                        result_id=f"{frame.frame_id}_result",
                        frame_id=frame.frame_id,
                        video_id=frame.video_id,
                        status=AnalysisStatus.FAILED,
                        error_message=str(e),
                    ))

        return results

    def _prepare_image(self, frame_data: FrameData) -> torch.Tensor:
        """이미지 전처리"""
        if frame_data.image_data is not None:
            # numpy array에서 PIL Image로 변환
            if frame_data.image_data.ndim == 2:
                # 그레이스케일
                image = Image.fromarray(frame_data.image_data, mode='L').convert('RGB')
            elif frame_data.image_data.shape[2] == 3:
                # BGR to RGB
                import cv2
                rgb = cv2.cvtColor(frame_data.image_data, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
            else:
                image = Image.fromarray(frame_data.image_data)
        elif frame_data.image_path:
            image = Image.open(frame_data.image_path).convert('RGB')
        else:
            raise ValueError(f"Frame {frame_data.frame_id} has no image data or path")

        return self.preprocess(image)

    def _generate_embedding(self, image: torch.Tensor) -> np.ndarray:
        """단일 이미지 임베딩 생성"""
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image)
            embedding = F.normalize(embedding, dim=-1)

        return embedding.cpu().numpy().squeeze()

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """두 임베딩 간의 코사인 유사도 계산"""
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def find_similar_frames(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        유사한 프레임 검색.

        Args:
            query_embedding: 쿼리 임베딩
            candidate_embeddings: 후보 임베딩 목록
            top_k: 반환할 최대 개수
            threshold: 최소 유사도 임계값

        Returns:
            (인덱스, 유사도) 튜플 목록
        """
        if not candidate_embeddings:
            return []

        # 배치로 유사도 계산
        candidates = np.array(candidate_embeddings)
        query = query_embedding.reshape(1, -1)

        similarities = np.dot(candidates, query.T).squeeze()

        # 임계값 필터링 및 정렬
        indices = np.argsort(similarities)[::-1]
        results = []

        for idx in indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((int(idx), sim))

        return results

    def match_state(
        self,
        frame_embedding: np.ndarray,
        state_embeddings: Dict[str, np.ndarray],
        threshold: float = 0.85,
    ) -> Optional[Tuple[str, float]]:
        """
        프레임과 가장 유사한 상태 매칭.

        Args:
            frame_embedding: 프레임 임베딩
            state_embeddings: 상태별 임베딩 딕셔너리
            threshold: 매칭 임계값

        Returns:
            (상태 이름, 유사도) 또는 None
        """
        best_state = None
        best_similarity = 0.0

        for state_name, state_emb in state_embeddings.items():
            similarity = self.compute_similarity(frame_embedding, state_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_state = state_name

        if best_similarity >= threshold:
            return (best_state, best_similarity)

        return None

    @property
    def embedding_dim(self) -> int:
        """임베딩 차원"""
        return self.SUPPORTED_MODELS.get(self.config.clip_model, 512)

    def cleanup(self) -> None:
        """리소스 정리"""
        if self.model is not None:
            del self.model
            self.model = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("FrameAnalyzer cleaned up")
