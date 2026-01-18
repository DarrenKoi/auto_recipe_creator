"""
Data models for Video Frame Parser

프레임 데이터, 메타데이터, 분석 결과를 위한 데이터 모델 정의
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
from pathlib import Path


class FrameType(Enum):
    """프레임 타입"""
    KEYFRAME = "keyframe"
    REGULAR = "regular"
    TRANSITION = "transition"
    STATIC = "static"


class AnalysisStatus(Enum):
    """분석 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoMetadata:
    """동영상 메타데이터"""

    video_id: str
    file_path: str
    file_name: str

    # 기본 정보
    duration: float  # 초 단위
    fps: float
    total_frames: int

    # 해상도
    width: int
    height: int

    # 코덱 정보
    codec: str
    fourcc: str

    # 파일 정보
    file_size: int  # 바이트
    created_at: datetime = field(default_factory=datetime.now)

    # 추가 메타데이터
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "video_id": self.video_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "fourcc": self.fourcc,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FrameData:
    """프레임 데이터"""

    frame_id: str
    video_id: str
    frame_number: int
    timestamp: float  # 초 단위

    # 이미지 데이터 (numpy array 또는 파일 경로)
    image_path: Optional[str] = None
    image_data: Optional[np.ndarray] = None

    # 프레임 특성
    frame_type: FrameType = FrameType.REGULAR
    is_keyframe: bool = False

    # 변화 감지
    change_score: float = 0.0  # 이전 프레임 대비 변화량

    # 메타데이터
    width: int = 0
    height: int = 0
    channels: int = 3

    # 타임스탬프
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환 (이미지 데이터 제외)"""
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "frame_type": self.frame_type.value,
            "is_keyframe": self.is_keyframe,
            "change_score": self.change_score,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "extracted_at": self.extracted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameData":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("frame_type"), str):
            data["frame_type"] = FrameType(data["frame_type"])
        if isinstance(data.get("extracted_at"), str):
            data["extracted_at"] = datetime.fromisoformat(data["extracted_at"])
        data.pop("image_data", None)  # numpy array는 제외
        return cls(**data)


@dataclass
class AnalysisResult:
    """프레임 분석 결과"""

    result_id: str
    frame_id: str
    video_id: str

    # CLIP 임베딩
    embedding: Optional[np.ndarray] = None
    embedding_model: str = "ViT-B/32"

    # 상태 매칭 결과
    matched_state: Optional[str] = None
    match_confidence: float = 0.0

    # UI 요소 감지
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)

    # 화면 변화 분석
    is_transition: bool = False
    transition_type: Optional[str] = None

    # 텍스트 추출 (OCR)
    extracted_text: List[str] = field(default_factory=list)

    # 분석 상태
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: Optional[str] = None

    # 타임스탬프
    analyzed_at: Optional[datetime] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        result = {
            "result_id": self.result_id,
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "embedding_model": self.embedding_model,
            "matched_state": self.matched_state,
            "match_confidence": self.match_confidence,
            "detected_elements": self.detected_elements,
            "is_transition": self.is_transition,
            "transition_type": self.transition_type,
            "extracted_text": self.extracted_text,
            "status": self.status.value,
            "error_message": self.error_message,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "processing_time_ms": self.processing_time_ms,
        }
        # 임베딩은 별도 저장 (FAISS 또는 별도 컬렉션)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = AnalysisStatus(data["status"])
        if isinstance(data.get("analyzed_at"), str):
            data["analyzed_at"] = datetime.fromisoformat(data["analyzed_at"])
        data.pop("embedding", None)  # numpy array는 제외
        return cls(**data)


@dataclass
class BatchResult:
    """배치 처리 결과"""

    batch_id: str
    video_id: str

    # 처리 통계
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0

    # 결과 목록
    results: List[AnalysisResult] = field(default_factory=list)

    # 처리 시간
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_time_seconds: float = 0.0

    # 에러 정보
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames - self.failed_frames) / self.total_frames

    @property
    def frames_per_second(self) -> float:
        """초당 처리 프레임 수"""
        if self.total_time_seconds == 0:
            return 0.0
        return self.processed_frames / self.total_time_seconds
