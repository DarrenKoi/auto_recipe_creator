"""
Video Frame Parser Module

H200 클러스터를 활용한 동영상 프레임 추출 및 분석 모듈.
AVI 등 동영상 파일에서 프레임 단위로 주요 정보를 파싱하여 DB에 저장.
"""

from .extractor import VideoFrameExtractor
from .analyzer import FrameAnalyzer
from .batch_processor import BatchProcessor
from .models import FrameData, VideoMetadata, AnalysisResult
from .parser import VideoFrameParser, create_h200_optimized_parser
from .db_handler import DatabaseHandler
from .config import VideoParserConfig, ExtractorConfig, AnalyzerConfig, BatchProcessorConfig, DatabaseConfig

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "VideoFrameParser",
    "VideoFrameExtractor",
    "FrameAnalyzer",
    "BatchProcessor",
    "DatabaseHandler",
    # Data models
    "FrameData",
    "VideoMetadata",
    "AnalysisResult",
    # Config classes
    "VideoParserConfig",
    "ExtractorConfig",
    "AnalyzerConfig",
    "BatchProcessorConfig",
    "DatabaseConfig",
    # Factory functions
    "create_h200_optimized_parser",
]
