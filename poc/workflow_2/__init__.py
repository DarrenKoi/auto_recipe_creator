"""
Workflow 2 package — Align Key 탐색 연구/평가 하니스 (legacy).

production CV 엔진(매칭/ensemble/자산 해석/보정/라이브 탐색)은
`poc.workflow_3.vision` 으로 전면 이전되었다. 본 패키지에는 평가·AB·튜닝·
probe 스크립트(golden_*, localization_*, reranker_*, tune_*, vlm_* 등)와
fixture 의존 테스트만 남으며, 엔진은 workflow_3 에서 import 한다.
"""

from pathlib import Path

WORKFLOW_2_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW_2_DIR / "debug_images"
LOG_DIR = WORKFLOW_2_DIR / "logs"

# 도메인 상수는 workflow_3.vision 이 단일 정의 — 남은 평가 하니스 호환을 위해
# 같은 이름으로 재export 한다.
from poc.workflow_3.vision import (
    ALIGN_IMAGES_ROOT,
    FROM_MSR_DIRNAME,
    FROM_RCP_DIRNAME,
    RCP_OM_STEM,
    RCP_SEM_STEM,
)

__all__ = [
    "ALIGN_IMAGES_ROOT",
    "DEBUG_IMAGE_DIR",
    "FROM_MSR_DIRNAME",
    "FROM_RCP_DIRNAME",
    "LOG_DIR",
    "RCP_OM_STEM",
    "RCP_SEM_STEM",
    "WORKFLOW_2_DIR",
]
