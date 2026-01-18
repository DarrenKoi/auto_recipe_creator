"""
Configuration settings for Video Frame Parser
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExtractorConfig:
    """프레임 추출기 설정"""

    # 프레임 추출 간격 (초 단위, None이면 모든 프레임)
    frame_interval: Optional[float] = 1.0

    # 키프레임만 추출할지 여부
    keyframes_only: bool = False

    # 출력 이미지 포맷
    output_format: str = "png"

    # 이미지 품질 (JPEG의 경우 0-100)
    quality: int = 95

    # 리사이즈 크기 (None이면 원본 크기 유지)
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None

    # 그레이스케일 변환 여부
    grayscale: bool = False


@dataclass
class AnalyzerConfig:
    """프레임 분석기 설정"""

    # CLIP 모델 이름
    clip_model: str = "ViT-B/32"

    # GPU 디바이스 ID (H200 클러스터용)
    device: str = "cuda"

    # 배치 크기
    batch_size: int = 32

    # 임베딩 차원
    embedding_dim: int = 512

    # 유사도 임계값
    similarity_threshold: float = 0.85


@dataclass
class BatchProcessorConfig:
    """배치 처리기 설정 (H200 클러스터용)"""

    # 최대 워커 수
    max_workers: int = 8

    # GPU 당 배치 크기
    batch_size_per_gpu: int = 64

    # 사용할 GPU 개수
    num_gpus: int = 1

    # 큐 크기
    queue_size: int = 1000

    # 타임아웃 (초)
    timeout: int = 300

    # 재시도 횟수
    max_retries: int = 3


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""

    # MongoDB 연결 URI
    mongo_uri: str = "mongodb://localhost:27017"

    # 데이터베이스 이름
    database_name: str = "recipe_automation"

    # 컬렉션 이름
    frames_collection: str = "video_frames"
    embeddings_collection: str = "frame_embeddings"
    metadata_collection: str = "video_metadata"

    # FAISS 인덱스 경로
    faiss_index_path: str = "./faiss_index"


@dataclass
class VideoParserConfig:
    """통합 파서 설정"""

    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    batch_processor: BatchProcessorConfig = field(default_factory=BatchProcessorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # 임시 파일 저장 경로
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/video_parser"))

    # 로그 레벨
    log_level: str = "INFO"

    # 변화 감지 임계값 (프레임 간 변화가 이 값 이상이면 저장)
    change_threshold: float = 0.1

    def __post_init__(self):
        self.temp_dir = Path(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
