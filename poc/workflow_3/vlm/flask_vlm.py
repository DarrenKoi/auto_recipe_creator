"""workflow_3 에서 공용으로 사용하는 VLM 설정 유틸리티."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


WORKFLOW_3_VLM_DIR = Path(__file__).resolve().parent
WORKFLOW_3_VLM_DOTENV_PATH = WORKFLOW_3_VLM_DIR / ".env"
COMMON_LLM_API_KEY_ENV = "COMMON_LLM_API_KEY"

if DOTENV_AVAILABLE and WORKFLOW_3_VLM_DOTENV_PATH.is_file():
    load_dotenv(WORKFLOW_3_VLM_DOTENV_PATH)


@dataclass(frozen=True)
class VLMServiceEntry:
    """workflow_3 전용 고정 VLM 서비스 정의."""

    route_slug: str
    display_name: str
    model_name: str
    api_url: str
    enabled: bool = True
    connection_mode: str = "proxy"
    prefer_stream: bool = False


DEFAULT_FLASK_API_BASE_URL = "http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api"
DEFAULT_COMPANY_LLM_BASE_URL = "http://common.llm.skhynix.com/v1"

KIMI_K2_6_MODEL_NAME = "Kimi-K2.6"
GLM_5_2_MODEL_NAME = "GLM-5.2"
MAI_UI_MODEL_NAME = "mai-ui-8b"
MAI_UI_2B_MODEL_NAME = "mai-ui-2b"
QWEN3_8_27B_MODEL_NAME = "qwen3.8-27b"
PADDLEOCR_VL_1_5_MODEL_NAME = "paddleocr-vl-1.5"

KIMI_K2_6_API_URL = DEFAULT_COMPANY_LLM_BASE_URL
GLM_5_2_API_URL = DEFAULT_COMPANY_LLM_BASE_URL
MAI_UI_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/mai-ui"
MAI_UI_2B_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/mai-ui-2b"
QWEN3_8_27B_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/qwen3.8-27b"
PADDLEOCR_VL_1_5_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/paddleocr-vl-1.5"

DEFAULT_SCREEN_ANALYSIS_SERVICE = "mai-ui"
DEFAULT_SCREEN_ANALYSIS_MODEL_NAME = MAI_UI_MODEL_NAME
DEFAULT_MAIN_TABS_SERVICE = "mai-ui"
DEFAULT_MAIN_TABS_MODEL_NAME = MAI_UI_MODEL_NAME
DEFAULT_OCR_SERVICE = "paddleocr-vl-1.5"
DEFAULT_OCR_MODEL_NAME = PADDLEOCR_VL_1_5_MODEL_NAME

ALL_VLM_SERVICES: list[VLMServiceEntry] = [
    VLMServiceEntry(
        "kimi-k2.6",
        "Kimi-K2.6",
        KIMI_K2_6_MODEL_NAME,
        KIMI_K2_6_API_URL,
        enabled=True,
        connection_mode="direct",
    ),
    VLMServiceEntry(
        "glm-5.2",
        "GLM-5.2",
        GLM_5_2_MODEL_NAME,
        GLM_5_2_API_URL,
        enabled=True,
        connection_mode="direct",
    ),
    VLMServiceEntry(
        "mai-ui",
        "MAI-UI-8B",
        MAI_UI_MODEL_NAME,
        MAI_UI_API_URL,
    ),
    # A/B 벤치 후보. slug 가 resolve 돼야 BENCH_COMBOS="mai-ui-2b>mai-ui-2b" 가 동작한다.
    # 서버에서 안 띄워져 있으면 호출이 404 로 떨어질 뿐, 등록만으로는 아무것도 안 바뀐다.
    VLMServiceEntry(
        "mai-ui-2b",
        "MAI-UI-2B",
        MAI_UI_2B_MODEL_NAME,
        MAI_UI_2B_API_URL,
    ),
    VLMServiceEntry(
        "paddleocr-vl-1.5",
        "PaddleOCR-VL-1.5",
        PADDLEOCR_VL_1_5_MODEL_NAME,
        PADDLEOCR_VL_1_5_API_URL,
    ),
    # 범용 추론/멀티모달 (GPU 1 단독, 262k 컨텍스트). grounding/OCR 기본값이 아니다 -
    # 명시적으로 이 slug 을 고른 호출만 여기로 간다.
    # [주의] 이 모델은 thinking 이 기본 on 이고, 지금 클라이언트는 요청에
    # temperature/max_tokens/frequency_penalty/stream 만 싣는다 - chat_template_kwargs 로
    # enable_thinking=false 를 보낼 방법이 없다. 짧은 max_tokens 로 부르면 답 대신
    # 사고 과정으로 예산을 다 쓸 수 있다. 짧은 응답 용도로 쓰려면 클라이언트에
    # chat_template_kwargs 지원을 먼저 넣을 것.
    VLMServiceEntry(
        "qwen3.8-27b",
        "Qwen3.8-27B",
        QWEN3_8_27B_MODEL_NAME,
        QWEN3_8_27B_API_URL,
    ),
]

_SERVICE_MAP: dict[str, VLMServiceEntry] = {service.route_slug: service for service in ALL_VLM_SERVICES}


def get_service_by_slug(slug: str) -> VLMServiceEntry | None:
    """route_slug 으로 서비스를 찾는다."""
    return _SERVICE_MAP.get(slug)


def _build_proxy_url(flask_base_url: str, service_slug: str) -> str:
    """Flask base URL 에서 service proxy base URL 을 구성한다."""
    base_url = flask_base_url.rstrip("/")
    if not base_url:
        return ""

    service_path = f"/api/vlm_serve/{service_slug}"
    if base_url.endswith(service_path):
        return base_url
    if base_url.endswith("/api/vlm_serve"):
        return f"{base_url}/{service_slug}"
    if base_url.endswith("/api"):
        return f"{base_url}/vlm_serve/{service_slug}"
    return f"{base_url}{service_path}"


def resolve_service_proxy_url(
    service_slug: str,
    *,
    flask_base_url: str | None = None,
) -> str:
    """service slug 기준 proxy base URL 을 반환한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return ""
    if service_entry.connection_mode == "direct":
        return service_entry.api_url.rstrip("/")

    base_url = (flask_base_url or DEFAULT_FLASK_API_BASE_URL).rstrip("/")
    if not base_url or base_url == DEFAULT_FLASK_API_BASE_URL.rstrip("/"):
        return service_entry.api_url.rstrip("/")
    return _build_proxy_url(base_url, service_slug)


def resolve_company_llm_api_key() -> str:
    """회사 공용 direct LLM API key 를 반환한다."""
    return os.getenv(COMMON_LLM_API_KEY_ENV, "").strip()


def resolve_service_api_key(service_slug: str, default: str = "") -> str:
    """service slug 별 API key 를 반환한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return default
    if service_entry.connection_mode == "direct":
        return resolve_company_llm_api_key() or default
    return default


__all__ = [
    "ALL_VLM_SERVICES",
    "DEFAULT_COMPANY_LLM_BASE_URL",
    "DEFAULT_FLASK_API_BASE_URL",
    "DEFAULT_MAIN_TABS_MODEL_NAME",
    "DEFAULT_MAIN_TABS_SERVICE",
    "DEFAULT_OCR_MODEL_NAME",
    "DEFAULT_OCR_SERVICE",
    "DEFAULT_SCREEN_ANALYSIS_MODEL_NAME",
    "DEFAULT_SCREEN_ANALYSIS_SERVICE",
    "VLMServiceEntry",
    "get_service_by_slug",
    "resolve_service_api_key",
    "resolve_service_proxy_url",
]
