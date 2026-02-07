"""
Workflow Extractor - CCTV 화면 녹화에서 워크플로우 추출 모듈

화면 녹화를 분석하여 전문가 워크플로우를 구조화된 데이터로 변환하고,
RAG 시스템을 통해 VLM 기반 자동화에 활용합니다.
"""

from .models import (
    ActionType,
    RecipeType,
    ExtractionMethod,
    WorkflowStep,
    WorkflowAnnotation,
    InferredAction,
    KeyFrame,
)
from .workflow_annotator import WorkflowAnnotator
from .auto_extractor import AutoExtractor, AutoExtractorConfig
from .workflow_ingester import WorkflowIngester
from .workflow_orchestrator import (
    WorkflowOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
    StepResult,
)

__all__ = [
    # 데이터 모델
    "ActionType",
    "RecipeType",
    "ExtractionMethod",
    "WorkflowStep",
    "WorkflowAnnotation",
    "InferredAction",
    "KeyFrame",
    # 수동 어노테이션
    "WorkflowAnnotator",
    # 자동 추출
    "AutoExtractor",
    "AutoExtractorConfig",
    # DB 인제스트
    "WorkflowIngester",
    # 오케스트레이터
    "WorkflowOrchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
    "StepResult",
]

__version__ = "0.1.0"
