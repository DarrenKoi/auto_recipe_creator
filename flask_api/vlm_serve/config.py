"""VLM 서비스 중앙 설정.

모든 VLM 서비스의 포트, 모델명, 활성 여부를 한 곳에서 관리한다.
현재 활성 포트: 8002 (mai-ui), 8004 (paddleocr-vl-1.5)

2026-08-11 부터 grounding 은 mai-ui 단일 모델, OCR 보조는 paddleocr 만 사용한다.
ui-venus / ui-tars / got-ocr 은 GPU 서버에서 기동하지 않으므로 enabled=False 로 둔다
(호스트 RAM 16GB 제약 - 프로세스 수가 GPU 메모리보다 먼저 한계에 닿는다).
다시 쓰려면 여기서 enabled=True 로 되돌리고 deploy_vlms 쪽에서도 기동해야 한다.
"""

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
    VLMServiceEntry("ui-venus", "UI-Venus-1.5-8B", "ui-venus-1.5-8b", 8001, enabled=False),
    VLMServiceEntry("mai-ui", "MAI-UI-8B", "mai-ui-8b", 8002, enabled=True),
    VLMServiceEntry("ui-tars", "UI-TARS-1.5-7B", "ui-tars-1.5-7b", 8003, enabled=False),
    VLMServiceEntry("paddleocr-vl-1.5", "PaddleOCR-VL-1.5", "paddleocr-vl-1.5", 8004, enabled=True),
    VLMServiceEntry("got-ocr", "GOT-OCR-2.0-hf", "got-ocr-2.0-hf", 8005, enabled=False),
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
