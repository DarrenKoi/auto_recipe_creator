"""
Video Frame Parser - Main Interface

동영상에서 프레임을 추출하고 분석하는 통합 인터페이스.
H200 클러스터를 활용한 대규모 병렬 처리 지원.
"""

import logging
from typing import List, Optional, Dict, Any, Callable, Iterator
from pathlib import Path
from datetime import datetime

from .config import (
    VideoParserConfig,
    ExtractorConfig,
    AnalyzerConfig,
    BatchProcessorConfig,
    DatabaseConfig,
)
from .models import (
    VideoMetadata,
    FrameData,
    AnalysisResult,
    BatchResult,
    AnalysisStatus,
)
from .extractor import VideoFrameExtractor
from .analyzer import FrameAnalyzer
from .batch_processor import BatchProcessor
from .db_handler import DatabaseHandler

logger = logging.getLogger(__name__)


class VideoFrameParser:
    """
    동영상 프레임 파서 메인 클래스.

    동영상에서 프레임을 추출하고, CLIP 임베딩을 생성하며,
    결과를 DB에 저장하는 전체 파이프라인을 제공합니다.

    Example:
        >>> config = VideoParserConfig()
        >>> parser = VideoFrameParser(config)
        >>>
        >>> # 단일 동영상 처리
        >>> result = parser.process_video("video.avi")
        >>>
        >>> # 프레임 유사도 검색
        >>> similar = parser.search_similar_frames(query_embedding, top_k=5)
    """

    def __init__(self, config: Optional[VideoParserConfig] = None):
        """
        VideoFrameParser 초기화.

        Args:
            config: 파서 설정. None이면 기본 설정 사용.
        """
        self.config = config or VideoParserConfig()

        # 컴포넌트 초기화
        self._extractor: Optional[VideoFrameExtractor] = None
        self._analyzer: Optional[FrameAnalyzer] = None
        self._batch_processor: Optional[BatchProcessor] = None
        self._db_handler: Optional[DatabaseHandler] = None

        self._initialized = False

        # 로깅 설정
        self._setup_logging()

    def _setup_logging(self) -> None:
        """로깅 설정"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def initialize(self, use_gpu: bool = True) -> None:
        """
        파서 컴포넌트 초기화.

        Args:
            use_gpu: GPU 사용 여부
        """
        if self._initialized:
            return

        logger.info("Initializing VideoFrameParser...")

        # Extractor 초기화
        self._extractor = VideoFrameExtractor(self.config.extractor)

        # Analyzer 초기화
        if use_gpu:
            self.config.analyzer.device = "cuda"
        else:
            self.config.analyzer.device = "cpu"

        self._analyzer = FrameAnalyzer(self.config.analyzer)

        # BatchProcessor 초기화
        self._batch_processor = BatchProcessor(self.config.batch_processor)

        # DatabaseHandler 초기화
        self._db_handler = DatabaseHandler(self.config.database)
        self._db_handler.initialize(embedding_dim=self._analyzer.embedding_dim)

        self._initialized = True
        logger.info("VideoFrameParser initialized successfully")

    def process_video(
        self,
        video_path: str | Path,
        frame_interval: Optional[float] = None,
        max_frames: Optional[int] = None,
        save_to_db: bool = True,
        save_frames_to_disk: bool = False,
        output_dir: Optional[str | Path] = None,
        callback: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> BatchResult:
        """
        동영상 전체 처리.

        Args:
            video_path: 동영상 파일 경로
            frame_interval: 프레임 추출 간격 (초). None이면 설정값 사용.
            max_frames: 최대 처리 프레임 수
            save_to_db: DB에 저장 여부
            save_frames_to_disk: 프레임 이미지를 디스크에 저장할지 여부
            output_dir: 프레임 저장 디렉토리
            callback: 프레임 처리 완료 시 콜백 함수

        Returns:
            BatchResult: 배치 처리 결과
        """
        if not self._initialized:
            self.initialize()

        video_path = Path(video_path)
        frame_interval = frame_interval or self.config.extractor.frame_interval

        start_time = datetime.now()
        logger.info(f"Processing video: {video_path.name}")

        # 메타데이터 추출
        metadata = self._extractor.open(video_path)

        if save_to_db:
            self._db_handler.save_video_metadata(metadata)

        # 프레임 추출
        frames = []
        for frame_data in self._extractor.extract_frames(
            max_frames=max_frames
        ):
            frames.append(frame_data)

            # 프레임 저장
            if save_frames_to_disk:
                output = output_dir or self.config.temp_dir / "frames" / metadata.video_id
                self._extractor.save_frame(frame_data, output)

        logger.info(f"Extracted {len(frames)} frames")

        # 프레임 분석
        results = self._batch_processor.process_video(
            video_path,
            frame_interval=frame_interval,
            max_frames=max_frames,
            callback=callback,
        )

        # DB 저장
        if save_to_db and results.results:
            self._db_handler.save_frames_batch(frames)
            self._db_handler.save_analysis_results_batch(results.results)
            logger.info(f"Saved {len(results.results)} analysis results to DB")

        self._extractor.close()

        return results

    def process_videos_batch(
        self,
        video_paths: List[str | Path],
        frame_interval: Optional[float] = None,
        max_frames_per_video: Optional[int] = None,
        save_to_db: bool = True,
    ) -> List[BatchResult]:
        """
        여러 동영상 배치 처리.

        Args:
            video_paths: 동영상 파일 경로 목록
            frame_interval: 프레임 추출 간격 (초)
            max_frames_per_video: 동영상당 최대 프레임 수
            save_to_db: DB에 저장 여부

        Returns:
            BatchResult 목록
        """
        if not self._initialized:
            self.initialize()

        results = []
        total = len(video_paths)

        for idx, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing video {idx}/{total}: {Path(video_path).name}")

            try:
                result = self.process_video(
                    video_path,
                    frame_interval=frame_interval,
                    max_frames=max_frames_per_video,
                    save_to_db=save_to_db,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append(BatchResult(
                    batch_id=f"error_{Path(video_path).stem}",
                    video_id="",
                    errors=[{"video": str(video_path), "error": str(e)}]
                ))

        return results

    def extract_frames_only(
        self,
        video_path: str | Path,
        frame_interval: Optional[float] = None,
        max_frames: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> List[FrameData]:
        """
        프레임만 추출 (분석 없이).

        Args:
            video_path: 동영상 파일 경로
            frame_interval: 프레임 추출 간격 (초)
            max_frames: 최대 추출 프레임 수
            output_dir: 프레임 저장 디렉토리

        Returns:
            추출된 FrameData 목록
        """
        video_path = Path(video_path)

        extractor_config = ExtractorConfig(
            frame_interval=frame_interval or self.config.extractor.frame_interval
        )
        extractor = VideoFrameExtractor(extractor_config)

        try:
            extractor.open(video_path)
            frames = list(extractor.extract_frames(max_frames=max_frames))

            if output_dir:
                output_dir = Path(output_dir)
                for frame in frames:
                    extractor.save_frame(frame, output_dir)

            return frames
        finally:
            extractor.close()

    def analyze_frames(
        self,
        frames: List[FrameData],
        batch_size: Optional[int] = None,
    ) -> List[AnalysisResult]:
        """
        프레임 배치 분석.

        Args:
            frames: 분석할 프레임 목록
            batch_size: 배치 크기

        Returns:
            분석 결과 목록
        """
        if not self._initialized:
            self.initialize()

        if batch_size:
            self._analyzer.config.batch_size = batch_size

        return self._analyzer.analyze_frames_batch(frames)

    def search_similar_frames(
        self,
        query_embedding: Any,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        유사 프레임 검색.

        Args:
            query_embedding: 쿼리 임베딩 (numpy array)
            top_k: 반환할 최대 개수
            threshold: 최소 유사도 임계값

        Returns:
            유사 프레임 정보 목록
        """
        if not self._initialized:
            self.initialize()

        results = self._db_handler.search_similar_frames(
            query_embedding, top_k=top_k, threshold=threshold
        )

        # 프레임 정보 추가
        detailed_results = []
        for frame_id, similarity in results:
            frame = self._db_handler.get_frame(frame_id)
            result = self._db_handler.get_analysis_result(f"{frame_id}_result")

            detailed_results.append({
                "frame_id": frame_id,
                "similarity": similarity,
                "frame_data": frame.to_dict() if frame else None,
                "analysis_result": result.to_dict() if result else None,
            })

        return detailed_results

    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """비디오 메타데이터 조회"""
        if not self._initialized:
            self.initialize()
        return self._db_handler.get_video_metadata(video_id)

    def get_frame(self, frame_id: str) -> Optional[FrameData]:
        """프레임 데이터 조회"""
        if not self._initialized:
            self.initialize()
        return self._db_handler.get_frame(frame_id)

    def get_analysis_result(self, frame_id: str) -> Optional[AnalysisResult]:
        """분석 결과 조회"""
        if not self._initialized:
            self.initialize()
        return self._db_handler.get_analysis_result(f"{frame_id}_result")

    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        stats = {
            "initialized": self._initialized,
            "config": {
                "frame_interval": self.config.extractor.frame_interval,
                "clip_model": self.config.analyzer.clip_model,
                "batch_size": self.config.batch_processor.batch_size_per_gpu,
                "num_gpus": self.config.batch_processor.num_gpus,
            }
        }

        if self._db_handler:
            stats["database"] = self._db_handler.get_stats()

        if self._batch_processor:
            stats["processor"] = self._batch_processor.get_stats()

        return stats

    def cleanup(self) -> None:
        """리소스 정리"""
        if self._extractor:
            self._extractor.close()

        if self._analyzer:
            self._analyzer.cleanup()

        if self._batch_processor:
            self._batch_processor.cleanup()

        if self._db_handler:
            self._db_handler.close()

        self._initialized = False
        logger.info("VideoFrameParser cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def create_h200_optimized_parser(
    num_gpus: int = 8,
    mongo_uri: str = "mongodb://localhost:27017",
) -> VideoFrameParser:
    """
    H200 클러스터에 최적화된 파서 생성.

    Args:
        num_gpus: 사용할 GPU 수
        mongo_uri: MongoDB 연결 URI

    Returns:
        최적화된 VideoFrameParser 인스턴스
    """
    config = VideoParserConfig()

    # H200 최적화 설정
    config.batch_processor.num_gpus = num_gpus
    config.batch_processor.batch_size_per_gpu = 64
    config.batch_processor.max_workers = num_gpus * 2

    # 분석기 설정
    config.analyzer.device = "cuda"
    config.analyzer.batch_size = 64

    # DB 설정
    config.database.mongo_uri = mongo_uri

    parser = VideoFrameParser(config)
    parser.initialize(use_gpu=True)

    return parser
