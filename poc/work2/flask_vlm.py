"""`poc.work2`에서 공용으로 사용하는 purpose-based VLM 설정 유틸리티.

이 파일은 팀 공용 기본값의 단일 진입점이다.
Flask proxy 주소와 목적별(screen_analysis, main_tabs, ocr) 모델 선택을 여기서 함께 관리한다.
민감한 값이 필요한 경우에는 코드 수정 대신 `poc/work2/.env` 에서 읽는다.
즉, `poc/work2` 는 서버 저장소 내부의 `flask_api` 코드나 환경변수에 의존하지 않는다.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


WORK2_DIR = Path(__file__).resolve().parent
WORK2_DOTENV_PATH = WORK2_DIR / ".env"
COMMON_LLM_API_KEY_ENV = "COMMON_LLM_API_KEY"

if DOTENV_AVAILABLE and WORK2_DOTENV_PATH.is_file():
    load_dotenv(WORK2_DOTENV_PATH)


@dataclass(frozen=True)
class VLMServiceEntry:
    """`poc.work2` 전용 고정 VLM 서비스 정의."""

    route_slug: str
    display_name: str
    model_name: str
    api_url: str
    enabled: bool = True
    connection_mode: str = "proxy"
    prompt_family: str = "gui"
    benchmark_role: str = ""
    prefer_stream: bool = False

DEFAULT_FLASK_API_BASE_URL = "http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api"
DEFAULT_COMPANY_LLM_BASE_URL = "http://common.llm.skhynix.com/v1"
DEFAULT_VLM_HEALTH_TIMEOUT_SEC = 5.0

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
DEFAULT_SCREEN_ANALYSIS_API_URL = UI_VENUS_API_URL
DEFAULT_MAIN_TABS_SERVICE = "ui-venus"
DEFAULT_MAIN_TABS_MODEL_NAME = UI_VENUS_MODEL_NAME
DEFAULT_MAIN_TABS_API_URL = UI_VENUS_API_URL
DEFAULT_OCR_SERVICE = "paddleocr-vl-1.5"
DEFAULT_OCR_MODEL_NAME = PADDLEOCR_VL_1_5_MODEL_NAME
DEFAULT_OCR_API_URL = PADDLEOCR_VL_1_5_API_URL
DEFAULT_OCR_PIPELINE_ENABLED = True

# 팀 공용 pipeline 기본값.
# 이 블록만 수정하면 `poc/work2` 하위 스크립트 전체에 동일한 설정이 반영된다.
# 의도:
# 1) 동료 PC마다 별도 `.env` 파일을 만들지 않는다.
# 2) 목적별 모델 선택을 코드에서 명시한다.
# 3) 민감한 API key 는 코드가 아니라 `poc/work2/.env` 에 둔다.
SHARED_PIPELINE_SETTINGS: dict[str, str | bool] = {
    "flask_api_base_url": DEFAULT_FLASK_API_BASE_URL,

    # 회사 공용 direct LLM (`common.llm.skhynix.com`) 연결 정보.
    # `Kimi-K2.5`, `Qwen3-VL-30B-Instruct` 가 이 base URL 을 사용한다.
    # API key 는 `poc/work2/.env` 의 `COMMON_LLM_API_KEY` 에 넣는다.
    "company_llm_base_url": DEFAULT_COMPANY_LLM_BASE_URL,

    "screen_analysis_service": DEFAULT_SCREEN_ANALYSIS_SERVICE,
    "screen_analysis_model_name": DEFAULT_SCREEN_ANALYSIS_MODEL_NAME,
    "screen_analysis_api_url": DEFAULT_SCREEN_ANALYSIS_API_URL,
    "main_tabs_service": DEFAULT_MAIN_TABS_SERVICE,
    "main_tabs_model_name": DEFAULT_MAIN_TABS_MODEL_NAME,
    "main_tabs_api_url": DEFAULT_MAIN_TABS_API_URL,
    "ocr_pipeline_enabled": DEFAULT_OCR_PIPELINE_ENABLED,
    "ocr_service": DEFAULT_OCR_SERVICE,
    "ocr_model_name": DEFAULT_OCR_MODEL_NAME,
    "ocr_api_url": DEFAULT_OCR_API_URL,
}

ALL_VLM_SERVICES: list[VLMServiceEntry] = [
    VLMServiceEntry(
        "kimi-k2.5",
        "Kimi-K2.5",
        KIMI_K2_5_MODEL_NAME,
        KIMI_K2_5_API_URL,
        enabled=True,
        connection_mode="direct",
        prompt_family="gui",
        benchmark_role="primary_gui",
    ),
    VLMServiceEntry(
        "qwen3-vl-30b-instruct",
        "Qwen3-VL-30B-Instruct",
        QWEN3_VL_30B_INSTRUCT_MODEL_NAME,
        QWEN3_VL_30B_INSTRUCT_API_URL,
        enabled=True,
        connection_mode="direct",
        prompt_family="gui",
        benchmark_role="primary_gui",
    ),
    VLMServiceEntry(
        "ui-venus",
        "UI-Venus-1.5-8B",
        UI_VENUS_MODEL_NAME,
        UI_VENUS_API_URL,
        enabled=True,
        connection_mode="proxy",
        prompt_family="gui",
        benchmark_role="primary_gui",
    ),
    VLMServiceEntry(
        "mai-ui",
        "MAI-UI-8B",
        MAI_UI_MODEL_NAME,
        MAI_UI_API_URL,
        enabled=True,
        connection_mode="proxy",
        prompt_family="gui",
        benchmark_role="zoom_in_sidecar",
    ),
    VLMServiceEntry(
        "ui-tars",
        "UI-TARS-1.5-7B",
        UI_TARS_MODEL_NAME,
        UI_TARS_API_URL,
        enabled=True,
        connection_mode="proxy",
        prompt_family="gui",
        benchmark_role="primary_gui",
    ),
    VLMServiceEntry(
        "paddleocr-vl-1.5",
        "PaddleOCR-VL-1.5",
        PADDLEOCR_VL_1_5_MODEL_NAME,
        PADDLEOCR_VL_1_5_API_URL,
        enabled=True,
        connection_mode="proxy",
        prompt_family="ocr",
        benchmark_role="ocr_default",
    ),
    VLMServiceEntry(
        "got-ocr",
        "GOT-OCR-2.0-hf",
        GOT_OCR_MODEL_NAME,
        GOT_OCR_API_URL,
        enabled=True,
        connection_mode="proxy",
        prompt_family="ocr",
        benchmark_role="ocr_fallback",
    ),
]
ENABLED_VLM_SERVICES: list[VLMServiceEntry] = [service for service in ALL_VLM_SERVICES if service.enabled]
_SERVICE_MAP: dict[str, VLMServiceEntry] = {service.route_slug: service for service in ALL_VLM_SERVICES}
SERVICE_API_URLS: dict[str, str] = {service.route_slug: service.api_url for service in ALL_VLM_SERVICES}
SERVICE_MODEL_NAMES: dict[str, str] = {service.route_slug: service.model_name for service in ALL_VLM_SERVICES}


def get_service_by_slug(slug: str) -> VLMServiceEntry | None:
    """route_slug 으로 서비스를 찾는다."""
    return _SERVICE_MAP.get(slug)


def get_enabled_slugs() -> set[str]:
    """활성 서비스 route_slug 집합을 반환한다."""
    return {service.route_slug for service in ENABLED_VLM_SERVICES}


def get_enabled_services_by_role(benchmark_role: str) -> list[VLMServiceEntry]:
    """지정 benchmark_role 의 활성 서비스 목록을 반환한다."""
    role = (benchmark_role or "").strip()
    if not role:
        return []
    return [
        service
        for service in ENABLED_VLM_SERVICES
        if service.benchmark_role == role
    ]


def _shared_text(name: str) -> str:
    """공유 설정 문자열 값을 trim 하여 반환한다."""
    return str(SHARED_PIPELINE_SETTINGS.get(name, "") or "").strip()


def _shared_flag(name: str, default: bool) -> bool:
    """공유 설정의 bool 값을 반환한다."""
    value = SHARED_PIPELINE_SETTINGS.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _build_proxy_url(flask_base_url: str, service_slug: str) -> str:
    """Flask base URL 에서 service proxy base URL 을 구성한다."""
    # 동료들이 base URL 을 `/api`, `/api/vlm_serve`, 또는 서비스 route 자체로 적어도
    # 모두 정상 동작하도록 허용한다.
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


def resolve_flask_api_base_url() -> str:
    """공유 Flask API root 를 반환한다."""
    return (_shared_text("flask_api_base_url") or DEFAULT_FLASK_API_BASE_URL).rstrip("/")


def resolve_vlm_health_url(flask_base_url: str | None = None) -> str:
    """Flask VLM health endpoint URL 을 반환한다."""
    base_url = (flask_base_url or resolve_flask_api_base_url()).rstrip("/")
    if base_url.endswith("/vlm_serve"):
        return f"{base_url}/health"
    return f"{base_url}/vlm_serve/health"


def resolve_service_proxy_url(
    service_slug: str,
    *,
    flask_base_url: str | None = None,
) -> str:
    """service slug 기준 Flask proxy base URL 을 반환한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is not None:
        if service_entry.connection_mode == "direct":
            return service_entry.api_url.rstrip("/")
        if flask_base_url is None:
            return service_entry.api_url.rstrip("/")

        base_url = flask_base_url.rstrip("/")
        if not base_url or base_url == DEFAULT_FLASK_API_BASE_URL.rstrip("/"):
            return service_entry.api_url.rstrip("/")
        return _build_proxy_url(base_url, service_slug)

    base_url = (flask_base_url or resolve_flask_api_base_url()).rstrip("/")
    return _build_proxy_url(base_url, service_slug)


def fetch_vlm_health(
    *,
    flask_base_url: str | None = None,
    timeout_sec: float = DEFAULT_VLM_HEALTH_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Flask VLM health payload 를 가져온다."""
    health_url = resolve_vlm_health_url(flask_base_url=flask_base_url)
    response = requests.get(health_url, timeout=timeout_sec)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"VLM health 응답 형식이 올바르지 않습니다: {payload!r}")
    return payload


def normalize_vlm_health_entries(
    health_body: dict[str, Any],
    *,
    flask_base_url: str | None = None,
) -> list[dict[str, Any]]:
    """health payload 의 `vlm_statuses` 를 공통 row 형식으로 정규화한다."""
    statuses = health_body.get("vlm_statuses")
    if not isinstance(statuses, list):
        return []

    registered_services = {
        str(item).strip()
        for item in health_body.get("registered_vlms", [])
        if str(item).strip()
    }
    base_url = (flask_base_url or resolve_flask_api_base_url()).rstrip("/")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in statuses:
        if not isinstance(item, dict):
            continue

        service_slug = str(item.get("service", "") or "").strip()
        if not service_slug or service_slug in seen:
            continue
        seen.add(service_slug)

        service_entry = get_service_by_slug(service_slug)
        health_status = str(item.get("health_status", "") or "").strip()
        display_name = str(
            item.get("display_name")
            or (service_entry.display_name if service_entry else service_slug)
            or service_slug
        ).strip()
        expected_model = str(
            item.get("served_model_name")
            or (service_entry.model_name if service_entry else "")
            or ""
        ).strip()
        proxy_registered = bool(item.get("proxy_registered")) or service_slug in registered_services

        rows.append(
            {
                "service": service_slug,
                "display_name": display_name,
                "expected_model": expected_model,
                "health_status": health_status,
                "proxy_registered": proxy_registered,
                "config_known": service_entry is not None,
                "config_enabled": None if service_entry is None else service_entry.enabled,
                "api_url": (
                    service_entry.api_url
                    if proxy_registered and service_entry is not None and base_url == DEFAULT_FLASK_API_BASE_URL.rstrip("/")
                    else _build_proxy_url(base_url, service_slug) if proxy_registered else ""
                ),
                "upstream_base_url": str(item.get("upstream_base_url", "") or "").strip(),
                "reason": str(item.get("reason", "") or "").strip(),
                "source_env": str(item.get("source_env", "") or "").strip(),
                "runtime": str(item.get("runtime", "") or "").strip(),
            }
        )

    return rows


def resolve_company_llm_api_base_url() -> str:
    """회사 공용 direct LLM base URL 을 반환한다."""
    return (_shared_text("company_llm_base_url") or DEFAULT_COMPANY_LLM_BASE_URL).rstrip("/")


def resolve_company_llm_api_key() -> str:
    """회사 공용 direct LLM API key 를 반환한다.

    `poc/work2/.env` 의 `COMMON_LLM_API_KEY` 에서만 읽는다.
    Flask proxy 서비스는 API key 를 사용하지 않는다.
    """
    return os.getenv(COMMON_LLM_API_KEY_ENV, "").strip()


def resolve_service_api_key(service_slug: str, default: str = "") -> str:
    """service slug 별 API key 를 반환한다.

    direct 서비스만 `COMMON_LLM_API_KEY` 를 사용하고,
    proxy 서비스는 API key 없이 동작한다.
    """
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return default

    if service_entry.connection_mode == "direct":
        return resolve_company_llm_api_key() or default
    return default


def resolve_screen_analysis_service(default: str = DEFAULT_SCREEN_ANALYSIS_SERVICE) -> str:
    """화면 분석 목적에 사용할 Flask proxy service slug 를 반환한다."""
    return _shared_text("screen_analysis_service") or default


def resolve_screen_analysis_api_url() -> str:
    """화면 분석 목적에 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _shared_text("screen_analysis_api_url")
    if direct_url:
        return direct_url.rstrip("/")

    return resolve_service_proxy_url(
        resolve_screen_analysis_service(),
        flask_base_url=resolve_flask_api_base_url(),
    )


def resolve_screen_analysis_model_name(default: str = DEFAULT_SCREEN_ANALYSIS_MODEL_NAME) -> str:
    """화면 분석 목적에 사용할 model name 을 반환한다."""
    return _shared_text("screen_analysis_model_name") or default


def resolve_main_tabs_service(default: str = DEFAULT_MAIN_TABS_SERVICE) -> str:
    """RCS main tabs 목적에 사용할 Flask proxy service slug 를 반환한다."""
    return _shared_text("main_tabs_service") or default


def resolve_main_tabs_api_url() -> str:
    """RCS main tabs 목적에 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _shared_text("main_tabs_api_url")
    if direct_url:
        return direct_url.rstrip("/")

    return resolve_service_proxy_url(
        resolve_main_tabs_service(),
        flask_base_url=resolve_flask_api_base_url(),
    )


def resolve_main_tabs_model_name(default: str = DEFAULT_MAIN_TABS_MODEL_NAME) -> str:
    """RCS main tabs 목적에 사용할 model name 을 반환한다."""
    return _shared_text("main_tabs_model_name") or default


def resolve_ocr_vlm_service(default: str = DEFAULT_OCR_SERVICE) -> str:
    """OCR stage 에 사용할 Flask proxy service slug 를 반환한다."""
    return _shared_text("ocr_service") or default


def resolve_ocr_vlm_api_url() -> str:
    """OCR stage 에 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _shared_text("ocr_api_url")
    if direct_url:
        return direct_url.rstrip("/")

    return resolve_service_proxy_url(
        resolve_ocr_vlm_service(),
        flask_base_url=resolve_flask_api_base_url(),
    )


def resolve_ocr_vlm_model_name(default: str = DEFAULT_OCR_MODEL_NAME) -> str:
    """OCR stage 에 사용할 model name 을 반환한다."""
    return _shared_text("ocr_model_name") or default


def resolve_pipeline_config(
    default_screen_analysis_model: str = DEFAULT_SCREEN_ANALYSIS_MODEL_NAME,
    default_main_tabs_model: str = DEFAULT_MAIN_TABS_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_MODEL_NAME,
) -> dict[str, object]:
    """공유 purpose-based pipeline 설정을 반환한다."""
    return {
        "flask_api_base_url": resolve_flask_api_base_url(),
        "company_llm_base_url": resolve_company_llm_api_base_url(),
        "company_llm_api_key": resolve_company_llm_api_key(),
        "screen_analysis_service": resolve_screen_analysis_service(),
        "screen_analysis_api_url": resolve_screen_analysis_api_url(),
        "screen_analysis_model_name": resolve_screen_analysis_model_name(
            default=default_screen_analysis_model
        ),
        "main_tabs_service": resolve_main_tabs_service(),
        "main_tabs_api_url": resolve_main_tabs_api_url(),
        "main_tabs_model_name": resolve_main_tabs_model_name(default=default_main_tabs_model),
        "ocr_pipeline_enabled": _shared_flag("ocr_pipeline_enabled", DEFAULT_OCR_PIPELINE_ENABLED),
        "ocr_service": resolve_ocr_vlm_service(),
        "ocr_api_url": resolve_ocr_vlm_api_url(),
        "ocr_model_name": resolve_ocr_vlm_model_name(default=default_ocr_model),
    }


def apply_pipeline_env_defaults(
    default_screen_analysis_model: str = DEFAULT_SCREEN_ANALYSIS_MODEL_NAME,
    default_main_tabs_model: str = DEFAULT_MAIN_TABS_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_MODEL_NAME,
) -> dict[str, object]:
    """공유 pipeline 설정을 계산하고 screen_analysis 값을 공통 env 에 반영한다."""
    # 기존 `poc.work` 쪽 클라이언트 일부가 아직 `VLM_API_URL`, `VLM_MODEL_NAME`
    # 같은 공통 환경변수 이름을 읽기 때문에, 공용 설정을 계산한 뒤 여기서 다시 주입한다.
    # 즉, 실제 source of truth 는 purpose-based `SHARED_PIPELINE_SETTINGS` 이고
    # 아래 `os.environ[...]` 할당은 하위 호환용 브리지라고 보면 된다.
    config = resolve_pipeline_config(
        default_screen_analysis_model=default_screen_analysis_model,
        default_main_tabs_model=default_main_tabs_model,
        default_ocr_model=default_ocr_model,
    )

    api_url = str(config.get("screen_analysis_api_url", "") or "")
    model_name = str(config.get("screen_analysis_model_name", "") or "")

    os.environ["VLM_API_URL"] = api_url
    os.environ["VLM_API_BASE_URL"] = api_url
    os.environ["VLM_MODEL_NAME"] = model_name

    return config


__all__ = [
    "ALL_VLM_SERVICES",
    "DEFAULT_COMPANY_LLM_BASE_URL",
    "DEFAULT_FLASK_API_BASE_URL",
    "DEFAULT_MAIN_TABS_API_URL",
    "DEFAULT_MAIN_TABS_MODEL_NAME",
    "DEFAULT_MAIN_TABS_SERVICE",
    "DEFAULT_OCR_PIPELINE_ENABLED",
    "DEFAULT_OCR_API_URL",
    "DEFAULT_OCR_MODEL_NAME",
    "DEFAULT_OCR_SERVICE",
    "DEFAULT_SCREEN_ANALYSIS_API_URL",
    "DEFAULT_SCREEN_ANALYSIS_MODEL_NAME",
    "DEFAULT_SCREEN_ANALYSIS_SERVICE",
    "DEFAULT_VLM_HEALTH_TIMEOUT_SEC",
    "ENABLED_VLM_SERVICES",
    "GOT_OCR_API_URL",
    "GOT_OCR_MODEL_NAME",
    "KIMI_K2_5_API_URL",
    "KIMI_K2_5_MODEL_NAME",
    "MAI_UI_API_URL",
    "MAI_UI_MODEL_NAME",
    "PADDLEOCR_VL_1_5_API_URL",
    "PADDLEOCR_VL_1_5_MODEL_NAME",
    "QWEN3_VL_30B_INSTRUCT_API_URL",
    "QWEN3_VL_30B_INSTRUCT_MODEL_NAME",
    "SERVICE_API_URLS",
    "SERVICE_MODEL_NAMES",
    "SHARED_PIPELINE_SETTINGS",
    "UI_TARS_API_URL",
    "UI_TARS_MODEL_NAME",
    "UI_VENUS_API_URL",
    "UI_VENUS_MODEL_NAME",
    "VLMServiceEntry",
    "apply_pipeline_env_defaults",
    "fetch_vlm_health",
    "resolve_company_llm_api_base_url",
    "resolve_company_llm_api_key",
    "get_enabled_slugs",
    "get_enabled_services_by_role",
    "get_service_by_slug",
    "normalize_vlm_health_entries",
    "resolve_flask_api_base_url",
    "resolve_main_tabs_api_url",
    "resolve_main_tabs_model_name",
    "resolve_main_tabs_service",
    "resolve_ocr_vlm_api_url",
    "resolve_ocr_vlm_model_name",
    "resolve_ocr_vlm_service",
    "resolve_pipeline_config",
    "resolve_screen_analysis_api_url",
    "resolve_screen_analysis_model_name",
    "resolve_screen_analysis_service",
    "resolve_service_api_key",
    "resolve_service_proxy_url",
    "resolve_vlm_health_url",
]
