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


@dataclass
class OCRResult:
    """OCR 추출 결과"""

    ocr_id: str  # 고유 ID
    frame_id: str  # 연결된 프레임 ID
    text: str  # 추출된 텍스트
    language: str  # 언어 코드 ("en", "ko", "mixed")
    confidence: float  # 신뢰도 점수 (0.0 ~ 1.0)
    bbox: List[int]  # 바운딩 박스 [x1, y1, x2, y2]

    # 메타데이터
    extracted_at: datetime = field(default_factory=datetime.now)
    ocr_model: str = "easyocr"

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "ocr_id": self.ocr_id,
            "frame_id": self.frame_id,
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "extracted_at": self.extracted_at.isoformat(),
            "ocr_model": self.ocr_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("extracted_at"), str):
            data["extracted_at"] = datetime.fromisoformat(data["extracted_at"])
        return cls(**data)


@dataclass
class ActionSequence:
    """작업 시퀀스 (CCTV에서 추출된 전문가 작업)"""

    action_id: str  # 고유 ID
    video_id: str  # 원본 비디오 ID
    start_frame_id: str  # 시작 프레임 ID
    end_frame_id: str  # 종료 프레임 ID
    start_time: float  # 시작 시간 (초)
    end_time: float  # 종료 시간 (초)
    actions: List[Dict[str, Any]]  # 작업 목록 [{type, x, y, text, timestamp}, ...]
    description: str  # 작업 설명
    success: bool  # 성공 여부

    # 메타데이터
    extracted_at: datetime = field(default_factory=datetime.now)
    extraction_method: str = "manual"  # "manual", "automated", "hybrid"
    confidence: float = 1.0  # 추출 신뢰도

    # 추가 정보
    error_type: Optional[str] = None  # 실패 시 에러 타입
    recovery_action: Optional[str] = None  # 복구 작업

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "action_id": self.action_id,
            "video_id": self.video_id,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "actions": self.actions,
            "description": self.description,
            "success": self.success,
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "error_type": self.error_type,
            "recovery_action": self.recovery_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionSequence":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("extracted_at"), str):
            data["extracted_at"] = datetime.fromisoformat(data["extracted_at"])
        return cls(**data)


@dataclass
class ErrorPattern:
    """에러 패턴"""

    error_id: str  # 고유 ID
    frame_id: str  # 연결된 프레임 ID
    error_type: str  # 에러 타입 ("connection", "authentication", "timeout", "ui_error", etc.)
    error_message: str  # 에러 메시지
    severity: str  # 심각도 ("low", "medium", "high", "critical")
    recovery_action: str  # 복구 작업 설명
    bbox: List[int]  # 에러 메시지 위치 [x1, y1, x2, y2]
    detected_method: str  # 감지 방법 ("color", "vlm", "color+vlm")

    # 메타데이터
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # 감지 신뢰도

    # 추가 정보
    associated_action_id: Optional[str] = None  # 연결된 작업 시퀀스 ID
    occurred_count: int = 1  # 발생 횟수

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "error_id": self.error_id,
            "frame_id": self.frame_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity,
            "recovery_action": self.recovery_action,
            "bbox": self.bbox,
            "detected_method": self.detected_method,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
            "associated_action_id": self.associated_action_id,
            "occurred_count": self.occurred_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorPattern":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("detected_at"), str):
            data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        return cls(**data)


@dataclass
class UIElement:
    """UI 요소"""

    element_id: str  # 고유 ID
    frame_id: str  # 연결된 프레임 ID
    element_type: str  # 요소 타입 ("button", "input", "label", "dropdown", "checkbox", etc.)
    label: str  # 요소 라벨/텍스트
    bbox: List[int]  # 바운딩 박스 [x1, y1, x2, y2]
    confidence: float  # 감지 신뢰도
    detection_method: str  # 감지 방법 ("vlm", "yolo", "traditional_cv")

    # 메타데이터
    detected_at: datetime = field(default_factory=datetime.now)

    # 추가 속성
    is_clickable: bool = True  # 클릭 가능 여부
    is_visible: bool = True  # 표시 여부
    state: Optional[str] = None  # 상태 ("enabled", "disabled", "focused", etc.)

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "element_id": self.element_id,
            "frame_id": self.frame_id,
            "element_type": self.element_type,
            "label": self.label,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "detected_at": self.detected_at.isoformat(),
            "is_clickable": self.is_clickable,
            "is_visible": self.is_visible,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UIElement":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if isinstance(data.get("detected_at"), str):
            data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        return cls(**data)


@dataclass
class RAGContext:
    """RAG 검색 컨텍스트"""

    similar_frames: List[FrameData]  # 시각적으로 유사한 프레임 (Top-K)
    action_sequences: List[ActionSequence]  # 전문가 작업 시퀀스
    error_patterns: List[ErrorPattern]  # 에러 패턴
    ui_elements: List[UIElement]  # 감지된 UI 요소
    temporal_frames: List[FrameData]  # 시간적 전후 컨텍스트
    retrieval_scores: List[float]  # 검색 신뢰도 점수
    retrieval_time_ms: float  # 검색 소요 시간 (밀리초)

    # 검색 메타데이터
    query_frame_id: Optional[str] = None  # 쿼리 프레임 ID
    query_text: Optional[str] = None  # 쿼리 텍스트
    retrieval_method: str = "hybrid"  # "visual", "text", "hybrid"
    top_k: int = 3  # 검색된 프레임 수

    # OCR 결과 (옵션)
    ocr_results: List[OCRResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "similar_frames": [f.to_dict() for f in self.similar_frames],
            "action_sequences": [a.to_dict() for a in self.action_sequences],
            "error_patterns": [e.to_dict() for e in self.error_patterns],
            "ui_elements": [u.to_dict() for u in self.ui_elements],
            "temporal_frames": [f.to_dict() for f in self.temporal_frames],
            "retrieval_scores": self.retrieval_scores,
            "retrieval_time_ms": self.retrieval_time_ms,
            "query_frame_id": self.query_frame_id,
            "query_text": self.query_text,
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
            "ocr_results": [o.to_dict() for o in self.ocr_results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGContext":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        data["similar_frames"] = [FrameData.from_dict(f) for f in data.get("similar_frames", [])]
        data["action_sequences"] = [ActionSequence.from_dict(a) for a in data.get("action_sequences", [])]
        data["error_patterns"] = [ErrorPattern.from_dict(e) for e in data.get("error_patterns", [])]
        data["ui_elements"] = [UIElement.from_dict(u) for u in data.get("ui_elements", [])]
        data["temporal_frames"] = [FrameData.from_dict(f) for f in data.get("temporal_frames", [])]
        data["ocr_results"] = [OCRResult.from_dict(o) for o in data.get("ocr_results", [])]
        return cls(**data)
