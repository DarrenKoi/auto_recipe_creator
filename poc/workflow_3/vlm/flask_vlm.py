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

KIMI_K2_5_MODEL_NAME = "Kimi-K2.5"
QWEN3_VL_30B_INSTRUCT_MODEL_NAME = "Qwen3-VL-30B-Instruct"
UI_VENUS_MODEL_NAME = "ui-venus-1.5-8b"
MAI_UI_MODEL_NAME = "mai-ui-8b"
UI_TARS_MODEL_NAME = "ui-tars-1.5-7b"
PADDLEOCR_VL_1_5_MODEL_NAME = "paddleocr-vl-1.5"
GOT_OCR_MODEL_NAME = "got-ocr-2.0-hf"

KIMI_K2_5_API_URL = DEFAULT_COMPANY_LLM_BASE_URL
QWEN3_VL_30B_INSTRUCT_API_URL = DEFAULT_COMPANY_LLM_BASE_URL
UI_VENUS_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/ui-venus"
MAI_UI_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/mai-ui"
UI_TARS_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/ui-tars"
PADDLEOCR_VL_1_5_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/paddleocr-vl-1.5"
GOT_OCR_API_URL = f"{DEFAULT_FLASK_API_BASE_URL}/vlm_serve/got-ocr"

DEFAULT_SCREEN_ANALYSIS_SERVICE = "ui-venus"
DEFAULT_SCREEN_ANALYSIS_MODEL_NAME = UI_VENUS_MODEL_NAME
DEFAULT_MAIN_TABS_SERVICE = "ui-venus"
DEFAULT_MAIN_TABS_MODEL_NAME = UI_VENUS_MODEL_NAME
DEFAULT_OCR_SERVICE = "paddleocr-vl-1.5"
DEFAULT_OCR_MODEL_NAME = PADDLEOCR_VL_1_5_MODEL_NAME

ALL_VLM_SERVICES: list[VLMServiceEntry] = [
    VLMServiceEntry(
        "kimi-k2.5",
        "Kimi-K2.5",
        KIMI_K2_5_MODEL_NAME,
        KIMI_K2_5_API_URL,
        enabled=True,
        connection_mode="direct",
    ),
    VLMServiceEntry(
        "qwen3-vl-30b-instruct",
        "Qwen3-VL-30B-Instruct",
        QWEN3_VL_30B_INSTRUCT_MODEL_NAME,
        QWEN3_VL_30B_INSTRUCT_API_URL,
        enabled=True,
        connection_mode="direct",
    ),
    VLMServiceEntry(
        "ui-venus",
        "UI-Venus-1.5-8B",
        UI_VENUS_MODEL_NAME,
        UI_VENUS_API_URL,
    ),
    VLMServiceEntry(
        "mai-ui",
        "MAI-UI-8B",
        MAI_UI_MODEL_NAME,
        MAI_UI_API_URL,
    ),
    VLMServiceEntry(
        "ui-tars",
        "UI-TARS-1.5-7B",
        UI_TARS_MODEL_NAME,
        UI_TARS_API_URL,
    ),
    VLMServiceEntry(
        "paddleocr-vl-1.5",
        "PaddleOCR-VL-1.5",
        PADDLEOCR_VL_1_5_MODEL_NAME,
        PADDLEOCR_VL_1_5_API_URL,
    ),
    VLMServiceEntry(
        "got-ocr",
        "GOT-OCR-2.0-hf",
        GOT_OCR_MODEL_NAME,
        GOT_OCR_API_URL,
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
