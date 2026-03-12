"""VLM 서비스 중앙 설정.

모든 VLM 서비스의 포트, 모델명, 활성 여부를 한 곳에서 관리한다.
현재 활성 포트: 8001 (ui-venus), 8004 (paddleocr-vl-1.5), 8005 (got-ocr)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMServiceEntry:
    """VLM 서비스 정의."""

    route_slug: str  # Flask proxy route slug
    display_name: str  # 표시용 이름
    model_name: str  # vLLM --served-model-name
    upstream_port: int  # vLLM 서빙 포트
    enabled: bool = True  # 활성 여부


# ── 전체 VLM 서비스 목록 ──────────────────────────────────────────────
# 포트 및 활성 여부를 변경하려면 이 목록만 수정하면 된다.
ALL_VLM_SERVICES: list[VLMServiceEntry] = [
    VLMServiceEntry("ui-venus", "UI-Venus-1.5-8B", "ui-venus-1.5-8b", 8001, enabled=True),
    VLMServiceEntry("mai-ui", "MAI-UI-8B", "mai-ui-8b", 8002, enabled=False),
    VLMServiceEntry("ui-tars", "UI-TARS-1.5-7B", "ui-tars-1.5-7b", 8003, enabled=False),
    VLMServiceEntry("paddleocr-vl-1.5", "PaddleOCR-VL-1.5", "paddleocr-vl-1.5", 8004, enabled=True),
    VLMServiceEntry("got-ocr", "GOT-OCR-2.0-hf", "got-ocr-2.0-hf", 8005, enabled=True),
]

# 활성 서비스만 필터링
ENABLED_VLM_SERVICES: list[VLMServiceEntry] = [s for s in ALL_VLM_SERVICES if s.enabled]

# route_slug → VLMServiceEntry 매핑
_SERVICE_MAP: dict[str, VLMServiceEntry] = {s.route_slug: s for s in ALL_VLM_SERVICES}


def get_service_by_slug(slug: str) -> VLMServiceEntry | None:
    """route_slug 으로 서비스를 찾는다."""
    return _SERVICE_MAP.get(slug)


def get_enabled_slugs() -> set[str]:
    """활성 서비스 route_slug 집합을 반환한다."""
    return {s.route_slug for s in ENABLED_VLM_SERVICES}


__all__ = [
    "ALL_VLM_SERVICES",
    "ENABLED_VLM_SERVICES",
    "VLMServiceEntry",
    "get_enabled_slugs",
    "get_service_by_slug",
]
