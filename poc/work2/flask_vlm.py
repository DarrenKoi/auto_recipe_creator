"""`poc.work2`에서 공용으로 사용하는 Flask proxy VLM 설정 유틸리티.

이 파일은 팀 공용 기본값의 단일 진입점이다.
동료들은 별도 `.env` 없이도 `poc/work2` 스크립트를 바로 실행할 수 있어야 하므로,
Flask proxy 주소, primary VLM, OCR 보조 모델 설정을 여기서 함께 관리한다.
"""

import os

# 팀 공용 pipeline 기본값.
# 이 블록만 수정하면 `poc/work2` 하위 스크립트 전체에 동일한 설정이 반영된다.
# 의도:
# 1) 동료 PC마다 별도 `.env` 파일을 만들지 않는다.
# 2) Flask proxy route / model slug / OCR 사용 여부를 코드에서 명시한다.
# 3) 필요 시 direct URL/API key 도 여기서만 관리한다.
SHARED_PIPELINE_SETTINGS: dict[str, str | bool] = {
    # Flask 앱의 공용 진입점. `/api` 까지 포함된 주소를 권장한다.
    "flask_api_base_url": "http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api",
    # Primary VLM: 기본 화면 해석 담당 모델
    "primary_service": "ui-venus",
    "primary_model_name": "ui-venus-1.5-8b",
    # 비워두면 `flask_api_base_url + /vlm_serve/{service}` 형태로 자동 구성된다.
    "primary_api_url": "",
    # Flask proxy 인증이 없으면 빈 문자열로 둔다.
    "primary_api_key": "",
    # OCR 보조 단계 사용 여부. 화면의 작은 텍스트가 중요할 때 True 유지.
    "ocr_pipeline_enabled": True,
    # OCR 보조 모델: primary 결과를 보강하는 텍스트 읽기 전용 stage
    "ocr_service": "paddleocr-vl-1.5",
    "ocr_model_name": "paddleocr-vl-1.5",
    # 비워두면 `flask_api_base_url + /vlm_serve/{service}` 형태로 자동 구성된다.
    "ocr_api_url": "",
    # OCR route 에 별도 key 가 필요할 때만 채운다.
    "ocr_api_key": "",
}

DEFAULT_FLASK_API_BASE_URL = str(SHARED_PIPELINE_SETTINGS["flask_api_base_url"])
DEFAULT_PRIMARY_VLM_SERVICE = str(SHARED_PIPELINE_SETTINGS["primary_service"])
DEFAULT_PRIMARY_VLM_MODEL_NAME = str(SHARED_PIPELINE_SETTINGS["primary_model_name"])
DEFAULT_OCR_PIPELINE_ENABLED = bool(SHARED_PIPELINE_SETTINGS["ocr_pipeline_enabled"])
DEFAULT_OCR_VLM_SERVICE = str(SHARED_PIPELINE_SETTINGS["ocr_service"])
DEFAULT_OCR_VLM_MODEL_NAME = str(SHARED_PIPELINE_SETTINGS["ocr_model_name"])


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


def resolve_primary_vlm_service(default: str = DEFAULT_PRIMARY_VLM_SERVICE) -> str:
    """Primary VLM 에 사용할 Flask proxy service slug 를 반환한다."""
    return _shared_text("primary_service") or default


def resolve_primary_vlm_api_url() -> str:
    """Primary VLM 에 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _shared_text("primary_api_url")
    if direct_url:
        return direct_url.rstrip("/")

    flask_base_url = resolve_flask_api_base_url()
    return _build_proxy_url(
        flask_base_url=flask_base_url,
        service_slug=resolve_primary_vlm_service(),
    )


def resolve_primary_vlm_api_key() -> str:
    """Primary VLM 에 사용할 API key 를 반환한다."""
    return _shared_text("primary_api_key")


def resolve_primary_vlm_model_name(default: str = DEFAULT_PRIMARY_VLM_MODEL_NAME) -> str:
    """Primary VLM 에 사용할 model name 을 반환한다."""
    return _shared_text("primary_model_name") or default


def resolve_ocr_vlm_service(default: str = DEFAULT_OCR_VLM_SERVICE) -> str:
    """OCR stage 에 사용할 Flask proxy service slug 를 반환한다."""
    return _shared_text("ocr_service") or default


def resolve_ocr_vlm_api_url() -> str:
    """OCR stage 에 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _shared_text("ocr_api_url")
    if direct_url:
        return direct_url.rstrip("/")

    return _build_proxy_url(
        flask_base_url=resolve_flask_api_base_url(),
        service_slug=resolve_ocr_vlm_service(),
    )


def resolve_ocr_vlm_api_key() -> str:
    """OCR stage 에 사용할 API key 를 반환한다."""
    return _shared_text("ocr_api_key") or resolve_primary_vlm_api_key()


def resolve_ocr_vlm_model_name(default: str = DEFAULT_OCR_VLM_MODEL_NAME) -> str:
    """OCR stage 에 사용할 model name 을 반환한다."""
    return _shared_text("ocr_model_name") or default


def resolve_pipeline_config(
    default_primary_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_VLM_MODEL_NAME,
) -> dict[str, object]:
    """공유 primary + OCR pipeline 설정을 반환한다."""
    return {
        "flask_api_base_url": resolve_flask_api_base_url(),
        "primary_service": resolve_primary_vlm_service(),
        "primary_api_url": resolve_primary_vlm_api_url(),
        "primary_api_key": resolve_primary_vlm_api_key(),
        "primary_model_name": resolve_primary_vlm_model_name(default=default_primary_model),
        "ocr_pipeline_enabled": _shared_flag("ocr_pipeline_enabled", DEFAULT_OCR_PIPELINE_ENABLED),
        "ocr_service": resolve_ocr_vlm_service(),
        "ocr_api_url": resolve_ocr_vlm_api_url(),
        "ocr_api_key": resolve_ocr_vlm_api_key(),
        "ocr_model_name": resolve_ocr_vlm_model_name(default=default_ocr_model),
    }


def apply_pipeline_env_defaults(
    default_primary_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_VLM_MODEL_NAME,
) -> dict[str, object]:
    """공유 pipeline 설정을 계산하고 primary VLM 값을 공통 env 에 반영한다."""
    # 기존 `poc.work` 쪽 클라이언트 일부가 아직 `VLM_API_URL`, `VLM_MODEL_NAME`
    # 같은 공통 환경변수 이름을 읽기 때문에, 공용 설정을 계산한 뒤 여기서 다시 주입한다.
    # 즉, 실제 source of truth 는 `SHARED_PIPELINE_SETTINGS` 이고
    # 아래 `os.environ[...]` 할당은 하위 호환용 브리지라고 보면 된다.
    config = resolve_pipeline_config(
        default_primary_model=default_primary_model,
        default_ocr_model=default_ocr_model,
    )

    api_url = str(config.get("primary_api_url", "") or "")
    api_key = str(config.get("primary_api_key", "") or "")
    model_name = str(config.get("primary_model_name", "") or "")

    os.environ["VLM_API_URL"] = api_url
    os.environ["VLM_API_BASE_URL"] = api_url
    os.environ["VLM_API_KEY"] = api_key
    os.environ["VLM_MODEL_NAME"] = model_name

    return config


def apply_primary_vlm_env_defaults(
    default_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
) -> dict[str, object]:
    """Primary VLM 공통 env 값을 반영하고 OCR 정보도 함께 반환한다."""
    # 신규 코드는 `apply_pipeline_env_defaults()` 사용을 권장한다.
    # 이 함수는 primary VLM 중심의 옛 호출부를 깨지 않기 위한 얇은 래퍼다.
    config = apply_pipeline_env_defaults(default_primary_model=default_model)
    return {
        "api_url": config["primary_api_url"],
        "api_key": config["primary_api_key"],
        "model_name": config["primary_model_name"],
        "service_slug": config["primary_service"],
        "ocr_service_slug": config["ocr_service"],
        "ocr_api_url": config["ocr_api_url"],
        "ocr_model_name": config["ocr_model_name"],
        "ocr_pipeline_enabled": config["ocr_pipeline_enabled"],
    }


__all__ = [
    "DEFAULT_FLASK_API_BASE_URL",
    "DEFAULT_OCR_PIPELINE_ENABLED",
    "DEFAULT_OCR_VLM_MODEL_NAME",
    "DEFAULT_OCR_VLM_SERVICE",
    "DEFAULT_PRIMARY_VLM_MODEL_NAME",
    "DEFAULT_PRIMARY_VLM_SERVICE",
    "SHARED_PIPELINE_SETTINGS",
    "apply_pipeline_env_defaults",
    "apply_primary_vlm_env_defaults",
    "resolve_flask_api_base_url",
    "resolve_ocr_vlm_api_key",
    "resolve_ocr_vlm_api_url",
    "resolve_ocr_vlm_model_name",
    "resolve_ocr_vlm_service",
    "resolve_pipeline_config",
    "resolve_primary_vlm_api_key",
    "resolve_primary_vlm_api_url",
    "resolve_primary_vlm_model_name",
    "resolve_primary_vlm_service",
]
