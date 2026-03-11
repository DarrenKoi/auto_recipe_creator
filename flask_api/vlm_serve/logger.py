"""VLM proxy logger helpers."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = "flask_api.vlm_serve"
DEFAULT_BODY_LIMIT = 20000
DEFAULT_LOG_DIRNAME = "logs"
DEFAULT_SERVICE_LOG_DIRNAME = "vlm_service"
DEFAULT_LOG_FILENAME = "vlm_serve.log"
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 10
FILE_HANDLER_MARKER = "_vlm_serve_file_handler"
SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
}
LARGE_BODY_KEYS = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "b64_json",
    "file",
    "bytes",
}
LARGE_BODY_KEY_LIMIT = 128


def _log_level() -> int:
    """환경변수 기반 logger level 을 반환한다."""
    level_name = os.environ.get("VLM_SERVE_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _repo_root() -> Path:
    """현재 cloud repo root 경로를 반환한다."""
    override = os.environ.get("VLM_SERVE_REPO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    """VLM service 로그 디렉터리를 반환한다."""
    override = os.environ.get("VLM_SERVE_LOG_DIR", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = _repo_root() / candidate
        return candidate.resolve()
    return _repo_root() / DEFAULT_LOG_DIRNAME / DEFAULT_SERVICE_LOG_DIRNAME


def _log_file_path() -> Path:
    """VLM service 로그 파일 경로를 반환한다."""
    return _log_dir() / DEFAULT_LOG_FILENAME


def _log_max_bytes() -> int:
    """로테이팅 로그 최대 파일 크기를 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_LOG_MAX_BYTES", str(DEFAULT_LOG_MAX_BYTES)).strip()
    try:
        return max(int(raw_value), 1024)
    except ValueError:
        return DEFAULT_LOG_MAX_BYTES


def _log_backup_count() -> int:
    """로테이팅 로그 백업 개수를 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_LOG_BACKUP_COUNT", str(DEFAULT_LOG_BACKUP_COUNT)).strip()
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return DEFAULT_LOG_BACKUP_COUNT


def _log_formatter() -> logging.Formatter:
    """파일 로그 formatter 를 반환한다."""
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )


def _configure_root_logger() -> logging.Logger:
    """루트 VLM logger 에 파일 핸들러를 보장한다."""
    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(_log_level())

    desired_path = _log_file_path()
    existing_handler: RotatingFileHandler | None = None
    for handler in list(root_logger.handlers):
        if not getattr(handler, FILE_HANDLER_MARKER, False):
            continue
        if Path(getattr(handler, "baseFilename", "")).resolve() == desired_path.resolve():
            existing_handler = handler
            continue
        root_logger.removeHandler(handler)
        handler.close()

    if existing_handler is None:
        try:
            desired_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                desired_path,
                maxBytes=_log_max_bytes(),
                backupCount=_log_backup_count(),
                encoding="utf-8",
            )
            setattr(file_handler, FILE_HANDLER_MARKER, True)
            root_logger.addHandler(file_handler)
            existing_handler = file_handler
        except OSError as exc:
            print(f"[WARNING] Failed to initialize VLM service logger at {desired_path}: {exc}")
            return root_logger

    existing_handler.setLevel(_log_level())
    existing_handler.setFormatter(_log_formatter())
    return root_logger


def _body_limit() -> int:
    """로그 body 최대 길이를 반환한다."""
    raw_value = os.environ.get("VLM_SERVE_LOG_MAX_BODY_CHARS", str(DEFAULT_BODY_LIMIT)).strip()
    try:
        return max(int(raw_value), 256)
    except ValueError:
        return DEFAULT_BODY_LIMIT


def get_vlm_logger(name: str | None = None) -> logging.Logger:
    """`flask_api.vlm_serve` 하위 logger 를 반환한다."""
    root_logger = _configure_root_logger()
    if not name:
        return root_logger

    logger = logging.getLogger(f"{LOGGER_NAME}.{name}")
    logger.setLevel(root_logger.level)
    logger.propagate = True
    return logger


def sanitize_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """민감 헤더를 마스킹한 사본을 반환한다."""
    if not headers:
        return {}

    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            sanitized[key] = "<redacted>"
            continue
        sanitized[key] = value
    return sanitized


def _truncate_text(value: str) -> str:
    """지정 길이까지만 로그에 남긴다."""
    limit = _body_limit()
    if len(value) <= limit:
        return value

    omitted = len(value) - limit
    return f"{value[:limit]}... <truncated {omitted} chars>"


def _summarize_data_url(value: str) -> str:
    """data URL 은 요약 문자열로 치환한다."""
    header, _, payload = value.partition(",")
    mime_type = "unknown"
    if ":" in header and ";" in header:
        mime_type = header.split(":", 1)[1].split(";", 1)[0]
    return f"<data-url mime={mime_type} chars={len(value)} payload_chars={len(payload)}>"


def _sanitize_json_value(value: Any, parent_key: str | None = None) -> Any:
    """로그용 JSON payload 를 정리한다."""
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json_value(item, parent_key=str(key))
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_json_value(item, parent_key=parent_key)
            for item in value
        ]

    if isinstance(value, (bytes, bytearray)):
        return f"<binary {len(value)} bytes>"

    if isinstance(value, str):
        if value.startswith("data:"):
            return _summarize_data_url(value)
        if parent_key and parent_key.lower() in LARGE_BODY_KEYS:
            if len(value) <= LARGE_BODY_KEY_LIMIT:
                return value
            omitted = len(value) - LARGE_BODY_KEY_LIMIT
            return f"{value[:LARGE_BODY_KEY_LIMIT]}... <truncated {omitted} chars>"
        return _truncate_text(value)

    return value


def format_body_for_log(
    body: bytes | str | None,
    *,
    content_type: str = "",
    sanitize_json: bool = False,
) -> str:
    """요청/응답 body 를 로그 문자열로 변환한다."""
    if body in (None, b"", ""):
        return "<empty>"

    if isinstance(body, bytes):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary {len(body)} bytes>"
    else:
        text = body

    stripped = text.strip()
    content_type_lower = content_type.lower()
    looks_like_json = stripped.startswith("{") or stripped.startswith("[")
    if "json" in content_type_lower or looks_like_json:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _truncate_text(text)

        payload = _sanitize_json_value(data) if sanitize_json else data
        return _truncate_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    return _truncate_text(text)


__all__ = [
    "LOGGER_NAME",
    "format_body_for_log",
    "get_vlm_logger",
    "sanitize_headers",
]
