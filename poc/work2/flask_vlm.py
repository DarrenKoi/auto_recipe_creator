"""Helpers for resolving Flask-proxied VLM settings in `poc.work2`."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

DEFAULT_CLOUD_FLASK_API_BASE_URL = (
    "http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api"
)
DEFAULT_PRIMARY_VLM_SERVICE = "ui-venus"
DEFAULT_PRIMARY_VLM_MODEL_NAME = "ui-venus-1.5-8b"
DEFAULT_OCR_VLM_SERVICE = "paddleocr-vl-1.5"
DEFAULT_OCR_VLM_MODEL_NAME = "paddleocr-vl-1.5"


def _env(name: str) -> str:
    """환경변수를 trim 된 문자열로 반환한다."""
    return os.environ.get(name, "").strip()


def _flag(name: str, default: bool) -> bool:
    value = _env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "y"}


def load_work2_env() -> None:
    """`poc/work2/.env`를 우선 로드하고, 기존 `poc/work/.env`도 이어서 로드한다."""
    if not DOTENV_AVAILABLE:
        return

    work2_dir = Path(__file__).resolve().parent
    env_candidates = [
        work2_dir / ".env",
        work2_dir.parent / "work" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)

    load_dotenv(override=False)


def resolve_work2_flask_api_base_url() -> str:
    """work2 에서 사용할 Flask API root 를 반환한다."""
    return (
        _env("WORK2_FLASK_API_BASE_URL")
        or _env("FLASK_API_BASE_URL")
        or DEFAULT_CLOUD_FLASK_API_BASE_URL
    ).rstrip("/")


def resolve_work2_vlm_service(default: str = DEFAULT_PRIMARY_VLM_SERVICE) -> str:
    """work2 에서 사용할 Flask proxy service slug 를 반환한다."""
    return _env("WORK2_VLM_SERVICE") or _env("VLM_SERVE_ROUTE_SLUG") or default


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


def resolve_work2_vlm_api_url() -> str:
    """work2 에서 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _env("WORK2_VLM_API_URL") or _env("WORK2_VLM_API_BASE_URL")
    if direct_url:
        return direct_url.rstrip("/")

    flask_base_url = resolve_work2_flask_api_base_url()
    return _build_proxy_url(
        flask_base_url=flask_base_url,
        service_slug=resolve_work2_vlm_service(),
    )


def resolve_work2_vlm_api_key() -> str:
    """work2 에서 사용할 API key 를 반환한다."""
    return _env("WORK2_VLM_API_KEY") or _env("VLM_API_KEY")


def resolve_work2_vlm_model_name(default: str = DEFAULT_PRIMARY_VLM_MODEL_NAME) -> str:
    """work2 에서 사용할 model name 을 반환한다."""
    return _env("WORK2_VLM_MODEL_NAME") or _env("VLM_MODEL_NAME") or default


def resolve_work2_ocr_service(default: str = DEFAULT_OCR_VLM_SERVICE) -> str:
    """work2 OCR stage 에서 사용할 Flask proxy service slug 를 반환한다."""
    return _env("WORK2_OCR_SERVICE") or default


def resolve_work2_ocr_api_url() -> str:
    """work2 OCR stage 에서 사용할 OpenAI-compatible base URL 을 반환한다."""
    direct_url = _env("WORK2_OCR_API_URL") or _env("WORK2_OCR_API_BASE_URL")
    if direct_url:
        return direct_url.rstrip("/")

    return _build_proxy_url(
        flask_base_url=resolve_work2_flask_api_base_url(),
        service_slug=resolve_work2_ocr_service(),
    )


def resolve_work2_ocr_api_key() -> str:
    """work2 OCR stage 에서 사용할 API key 를 반환한다."""
    return _env("WORK2_OCR_API_KEY") or resolve_work2_vlm_api_key()


def resolve_work2_ocr_model_name(default: str = DEFAULT_OCR_VLM_MODEL_NAME) -> str:
    """work2 OCR stage 에서 사용할 model name 을 반환한다."""
    return _env("WORK2_OCR_MODEL_NAME") or default


def resolve_work2_pipeline_config(
    default_primary_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_VLM_MODEL_NAME,
) -> dict[str, object]:
    """work2 의 primary + OCR pipeline 설정을 반환한다."""
    return {
        "flask_api_base_url": resolve_work2_flask_api_base_url(),
        "primary_service": resolve_work2_vlm_service(),
        "primary_api_url": resolve_work2_vlm_api_url(),
        "primary_api_key": resolve_work2_vlm_api_key(),
        "primary_model_name": resolve_work2_vlm_model_name(default=default_primary_model),
        "ocr_pipeline_enabled": _flag("WORK2_OCR_PIPELINE_ENABLED", True),
        "ocr_service": resolve_work2_ocr_service(),
        "ocr_api_url": resolve_work2_ocr_api_url(),
        "ocr_api_key": resolve_work2_ocr_api_key(),
        "ocr_model_name": resolve_work2_ocr_model_name(default=default_ocr_model),
    }


def apply_work2_pipeline_env_defaults(
    default_primary_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
    default_ocr_model: str = DEFAULT_OCR_VLM_MODEL_NAME,
) -> dict[str, object]:
    """work2 pipeline 설정을 계산하고 primary VLM 값을 공통 env 에 반영한다."""
    config = resolve_work2_pipeline_config(
        default_primary_model=default_primary_model,
        default_ocr_model=default_ocr_model,
    )

    api_url = str(config.get("primary_api_url", "") or "")
    api_key = str(config.get("primary_api_key", "") or "")
    model_name = str(config.get("primary_model_name", "") or "")

    if api_url:
        os.environ["VLM_API_URL"] = api_url
        os.environ["VLM_API_BASE_URL"] = api_url
    if api_key or "WORK2_VLM_API_KEY" in os.environ:
        os.environ["VLM_API_KEY"] = api_key
    if model_name:
        os.environ["VLM_MODEL_NAME"] = model_name

    return config


def apply_work2_vlm_env_defaults(
    default_model: str = DEFAULT_PRIMARY_VLM_MODEL_NAME,
) -> dict[str, object]:
    """Backward-compatible alias returning primary VLM fields plus OCR info."""
    config = apply_work2_pipeline_env_defaults(default_primary_model=default_model)
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
