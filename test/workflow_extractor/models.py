"""
Workflow Extractor 데이터 모델

화면 녹화에서 추출된 워크플로우 단계 및 어노테이션을 위한 데이터 모델 정의.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class ActionType(Enum):
    """작업 타입"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    VERIFY = "verify"
    HOTKEY = "hotkey"


class RecipeType(Enum):
    """레시피 워크플로우 타입"""
    RCS_LOGIN = "RCS_LOGIN"
    RECIPE_REPLACE = "RECIPE_REPLACE"
    MEASUREMENT_SETUP = "MEASUREMENT_SETUP"
    RECIPE_CREATE = "RECIPE_CREATE"
    RECIPE_EDIT = "RECIPE_EDIT"
    CALIBRATION = "CALIBRATION"
    OTHER = "OTHER"


class ExtractionMethod(Enum):
    """추출 방법"""
    MANUAL = "manual"
    AUTOMATED = "automated"
    HYBRID = "hybrid"


@dataclass
class WorkflowStep:
    """워크플로우 단일 단계"""

    step_number: int  # 순서
    action_type: str  # ActionType 값 ("click", "type", etc.)
    target_description: str  # 대상 설명 ("Login 버튼", "Server 입력 필드")
    timestamp: float  # 영상에서의 시간 (초)
    screenshot_frame: int  # 해당 프레임 번호

    # 선택 필드
    coordinates: Optional[Tuple[int, int]] = None  # 클릭 좌표 (x, y)
    input_text: Optional[str] = None  # 입력 텍스트
    notes: Optional[str] = None  # 추가 메모
    confidence: float = 1.0  # 신뢰도 (자동 추출 시)
    duration: Optional[float] = None  # 작업 소요 시간 (초)

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "step_number": self.step_number,
            "action_type": self.action_type,
            "target_description": self.target_description,
            "timestamp": self.timestamp,
            "screenshot_frame": self.screenshot_frame,
            "coordinates": list(self.coordinates) if self.coordinates else None,
            "input_text": self.input_text,
            "notes": self.notes,
            "confidence": self.confidence,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if data.get("coordinates") is not None:
            data["coordinates"] = tuple(data["coordinates"])
        return cls(**data)


@dataclass
class WorkflowAnnotation:
    """워크플로우 어노테이션 (전체 작업 흐름)"""

    workflow_id: str  # 고유 ID
    video_path: str  # 원본 영상 경로
    recipe_type: str  # RecipeType 값 ("RCS_LOGIN", "RECIPE_REPLACE", etc.)
    description: str  # 워크플로우 설명 ("RCS 서버 로그인 후 레시피 교체")
    steps: List[WorkflowStep]  # 단계 목록
    total_duration: float  # 전체 소요 시간 (초)
    success: bool  # 성공 여부
    annotated_by: str  # 작성자

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    extraction_method: str = "manual"  # ExtractionMethod 값
    video_resolution: Optional[Tuple[int, int]] = None  # 영상 해상도 (width, height)
    tags: List[str] = field(default_factory=list)  # 검색용 태그

    def to_dict(self) -> Dict[str, Any]:
        """MongoDB 저장용 딕셔너리 변환"""
        return {
            "workflow_id": self.workflow_id,
            "video_path": self.video_path,
            "recipe_type": self.recipe_type,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration": self.total_duration,
            "success": self.success,
            "annotated_by": self.annotated_by,
            "created_at": self.created_at.isoformat(),
            "extraction_method": self.extraction_method,
            "video_resolution": list(self.video_resolution) if self.video_resolution else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowAnnotation":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        data["steps"] = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("video_resolution") is not None:
            data["video_resolution"] = tuple(data["video_resolution"])
        return cls(**data)


@dataclass
class InferredAction:
    """VLM이 추론한 프레임 간 작업"""

    action_type: str  # ActionType 값
    target: str  # 대상 버튼/필드 이름
    coordinates: Optional[Tuple[int, int]] = None  # 추정 좌표
    input_text: Optional[str] = None  # 입력된 텍스트
    confidence: float = 0.0  # VLM 신뢰도
    description: str = ""  # 작업 설명

    # 프레임 참조
    before_frame: Optional[int] = None  # 이전 프레임 번호
    after_frame: Optional[int] = None  # 이후 프레임 번호
    timestamp: float = 0.0  # 추정 시간

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "coordinates": list(self.coordinates) if self.coordinates else None,
            "input_text": self.input_text,
            "confidence": self.confidence,
            "description": self.description,
            "before_frame": self.before_frame,
            "after_frame": self.after_frame,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferredAction":
        """딕셔너리에서 객체 생성"""
        data = data.copy()
        if data.get("coordinates") is not None:
            data["coordinates"] = tuple(data["coordinates"])
        return cls(**data)


@dataclass
class KeyFrame:
    """이벤트 감지 기반 핵심 프레임"""

    frame_number: int  # 프레임 번호
    timestamp: float  # 시간 (초)
    change_score: float  # 이전 프레임 대비 변화량
    cluster_id: Optional[int] = None  # 클러스터 ID
    is_representative: bool = False  # 클러스터 대표 프레임 여부
    image_path: Optional[str] = None  # 저장된 이미지 경로

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "change_score": self.change_score,
            "cluster_id": self.cluster_id,
            "is_representative": self.is_representative,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyFrame":
        """딕셔너리에서 객체 생성"""
        return cls(**data)
