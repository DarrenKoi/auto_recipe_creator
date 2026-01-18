"""
Batch Processor for H200 Cluster

H200 GPU 클러스터를 활용한 대규모 병렬 처리 모듈.
멀티 GPU 분산 처리 및 효율적인 배치 처리 지원.
"""

import logging
from typing import List, Optional, Iterator, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import BatchProcessorConfig
from .models import FrameData, AnalysisResult, BatchResult, AnalysisStatus
from .extractor import VideoFrameExtractor
from .analyzer import FrameAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """처리 작업 단위"""
    task_id: str
    frames: List[FrameData]
    priority: int = 0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BatchProcessor:
    """
    H200 클러스터용 배치 프로세서.

    Features:
    - 멀티 GPU 병렬 처리
    - 동적 배치 크기 조절
    - 메모리 효율적 처리
    - 체크포인트 및 재시작 지원
    """

    def __init__(self, config: Optional[BatchProcessorConfig] = None):
        self.config = config or BatchProcessorConfig()
        self._analyzers: List[FrameAnalyzer] = []
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._result_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._running = False
        self._workers: List[threading.Thread] = []

    def initialize(self) -> None:
        """프로세서 초기화"""
        if self._initialized:
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for BatchProcessor")

        # GPU 가용성 확인
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.config.num_gpus = min(self.config.num_gpus, num_gpus)

            for i in range(self.config.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f}GB")
        else:
            logger.warning("No GPU available, falling back to CPU")
            self.config.num_gpus = 0

        # GPU별 분석기 초기화
        self._init_analyzers()

        # 분산 처리 초기화
        if self.config.enable_distributed and self.config.num_gpus > 1:
            self._init_distributed()

        self._initialized = True
        logger.info(f"BatchProcessor initialized with {self.config.num_gpus} GPUs")

    def _init_analyzers(self) -> None:
        """GPU별 분석기 초기화"""
        from .config import AnalyzerConfig

        if self.config.num_gpus == 0:
            # CPU 모드
            config = AnalyzerConfig(device="cpu", batch_size=self.config.micro_batch_size)
            analyzer = FrameAnalyzer(config)
            analyzer.initialize()
            self._analyzers.append(analyzer)
        else:
            # GPU 모드
            gpu_ids = self.config.gpu_ids or list(range(self.config.num_gpus))
            for gpu_id in gpu_ids:
                config = AnalyzerConfig(
                    device=f"cuda:{gpu_id}",
                    batch_size=self.config.micro_batch_size
                )
                analyzer = FrameAnalyzer(config)
                analyzer.initialize()
                self._analyzers.append(analyzer)
                logger.info(f"Analyzer initialized on GPU {gpu_id}")

    def _init_distributed(self) -> None:
        """분산 처리 환경 초기화"""
        if not dist.is_initialized():
            # 환경 변수 기반 초기화 (SLURM/Kubernetes 호환)
            import os
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            try:
                dist.init_process_group(
                    backend="nccl",
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
                logger.info(f"Distributed processing initialized: rank {self.config.rank}/{self.config.world_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed processing: {e}")
                self.config.enable_distributed = False

    def process_video(
        self,
        video_path: str | Path,
        frame_interval: float = 1.0,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> BatchResult:
        """
        동영상 전체 처리.

        Args:
            video_path: 동영상 파일 경로
            frame_interval: 프레임 추출 간격 (초)
            max_frames: 최대 처리 프레임 수
            callback: 결과 콜백 함수

        Returns:
            BatchResult: 배치 처리 결과
        """
        if not self._initialized:
            self.initialize()

        from .config import ExtractorConfig

        start_time = datetime.now()
        video_path = Path(video_path)

        # 프레임 추출
        extractor_config = ExtractorConfig(frame_interval=frame_interval)
        extractor = VideoFrameExtractor(extractor_config)

        try:
            metadata = extractor.open(video_path)
            video_id = metadata.video_id
            batch_id = f"{video_id}_batch_{start_time.strftime('%Y%m%d%H%M%S')}"

            logger.info(f"Starting video processing: {video_path.name}")

            # 프레임 수집
            frames = list(extractor.extract_frames(max_frames=max_frames))
            total_frames = len(frames)

            logger.info(f"Extracted {total_frames} frames for processing")

            # 배치 처리
            results = self._process_frames_parallel(frames, callback)

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # 결과 집계
            processed = sum(1 for r in results if r.status == AnalysisStatus.COMPLETED)
            failed = sum(1 for r in results if r.status == AnalysisStatus.FAILED)

            batch_result = BatchResult(
                batch_id=batch_id,
                video_id=video_id,
                total_frames=total_frames,
                processed_frames=processed,
                failed_frames=failed,
                results=results,
                start_time=start_time,
                end_time=end_time,
                total_time_seconds=total_time,
            )

            logger.info(
                f"Video processing completed: {processed}/{total_frames} frames, "
                f"Time: {total_time:.2f}s, Rate: {batch_result.frames_per_second:.1f} fps"
            )

            return batch_result

        finally:
            extractor.close()

    def _process_frames_parallel(
        self,
        frames: List[FrameData],
        callback: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> List[AnalysisResult]:
        """병렬 프레임 처리"""
        if not frames:
            return []

        results = []
        batch_size = self.config.batch_size
        num_analyzers = len(self._analyzers)

        if num_analyzers == 0:
            raise RuntimeError("No analyzers available")

        # 배치 분할 및 처리
        with ThreadPoolExecutor(max_workers=num_analyzers) as executor:
            futures = []

            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]

                # 라운드 로빈 방식으로 분석기 할당
                analyzer_idx = (i // batch_size) % num_analyzers
                analyzer = self._analyzers[analyzer_idx]

                future = executor.submit(
                    self._process_batch_with_analyzer,
                    analyzer,
                    batch,
                    callback
                )
                futures.append(future)

            # 결과 수집
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")

        return results

    def _process_batch_with_analyzer(
        self,
        analyzer: FrameAnalyzer,
        frames: List[FrameData],
        callback: Optional[Callable[[AnalysisResult], None]] = None,
    ) -> List[AnalysisResult]:
        """특정 분석기로 배치 처리"""
        results = analyzer.analyze_frames_batch(frames)

        if callback:
            for result in results:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

        return results

    def process_frames_stream(
        self,
        frame_iterator: Iterator[FrameData],
        batch_size: Optional[int] = None,
    ) -> Iterator[AnalysisResult]:
        """
        스트리밍 방식 프레임 처리.

        Args:
            frame_iterator: 프레임 이터레이터
            batch_size: 배치 크기

        Yields:
            AnalysisResult: 분석 결과
        """
        if not self._initialized:
            self.initialize()

        batch_size = batch_size or self.config.batch_size
        batch = []

        for frame in frame_iterator:
            batch.append(frame)

            if len(batch) >= batch_size:
                results = self._process_frames_parallel(batch)
                for result in results:
                    yield result
                batch = []

        # 남은 프레임 처리
        if batch:
            results = self._process_frames_parallel(batch)
            for result in results:
                yield result

    def process_videos_batch(
        self,
        video_paths: List[str | Path],
        frame_interval: float = 1.0,
        max_frames_per_video: Optional[int] = None,
    ) -> List[BatchResult]:
        """
        여러 동영상 배치 처리.

        Args:
            video_paths: 동영상 파일 경로 목록
            frame_interval: 프레임 추출 간격
            max_frames_per_video: 동영상당 최대 프레임 수

        Returns:
            BatchResult 목록
        """
        results = []

        for video_path in video_paths:
            try:
                result = self.process_video(
                    video_path,
                    frame_interval=frame_interval,
                    max_frames=max_frames_per_video,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {e}")
                # 에러 결과 추가
                results.append(BatchResult(
                    batch_id=f"error_{Path(video_path).stem}",
                    video_id="",
                    total_frames=0,
                    errors=[{"video": str(video_path), "error": str(e)}]
                ))

        return results

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """체크포인트 저장"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "num_analyzers": len(self._analyzers),
            "timestamp": datetime.now().isoformat(),
        }

        import pickle
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """체크포인트 로드"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        import pickle
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.config = checkpoint["config"]
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = {
            "initialized": self._initialized,
            "num_gpus": self.config.num_gpus,
            "num_analyzers": len(self._analyzers),
            "batch_size": self.config.batch_size,
            "distributed": self.config.enable_distributed,
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(min(self.config.num_gpus, torch.cuda.device_count())):
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                stats[f"gpu_{i}_memory_allocated_gb"] = memory_allocated
                stats[f"gpu_{i}_memory_reserved_gb"] = memory_reserved

        return stats

    def cleanup(self) -> None:
        """리소스 정리"""
        self._running = False

        # 분석기 정리
        for analyzer in self._analyzers:
            analyzer.cleanup()
        self._analyzers.clear()

        # 분산 처리 정리
        if self.config.enable_distributed and dist.is_initialized():
            dist.destroy_process_group()

        # GPU 메모리 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("BatchProcessor cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
