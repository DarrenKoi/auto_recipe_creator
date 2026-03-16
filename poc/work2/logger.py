"""work2 파일 로거.

VLM 호출 및 일반 이벤트를 파일에 기록한다.
기본 로그 파일:
- VLM 호출: poc/work2/logs/vlm_calls.log
- 일반 이벤트: poc/work2/logs/work2.log
"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 5
_HANDLER_MARKER = "_work2_vlm_logger"

_loggers: dict[str, logging.Logger] = {}


def _sanitize_log_name(log_name: str) -> str:
    """로그 파일명에 안전한 식별자로 정규화한다."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", (log_name or "").strip()).strip("-._")
    return name or "work2"


def _get_logger(log_name: str) -> logging.Logger:
    """log_name 별 싱글턴 로거를 반환한다."""
    safe_name = _sanitize_log_name(log_name)
    existing = _loggers.get(safe_name)
    if existing is not None:
        return existing

    logger = logging.getLogger(f"poc.work2.{safe_name}")
    level_name = os.environ.get("WORK2_LOG_LEVEL", "INFO").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            _loggers[safe_name] = logger
            return logger

    log_path = _LOG_DIR / f"{safe_name}.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        setattr(file_handler, _HANDLER_MARKER, True)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"),
        )
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"[WARNING] work2 로거 초기화 실패: {exc}")

    _loggers[safe_name] = logger
    return logger


def _format_tokens(token_usage: dict[str, int] | None) -> str:
    """토큰 사용량을 로그 문자열로 변환한다."""
    if not token_usage:
        return "tokens=N/A"
    prompt = token_usage.get("prompt_tokens", "?")
    completion = token_usage.get("completion_tokens", "?")
    total = token_usage.get("total_tokens", "?")
    return f"prompt_tokens={prompt} completion_tokens={completion} total_tokens={total}"


def _format_fields(fields: dict[str, object]) -> str:
    """구조화 필드를 로그 문자열로 변환한다."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n")
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_vlm_call(
    *,
    service: str,
    model: str,
    status: str,
    latency_ms: float,
    token_usage: dict[str, int] | None = None,
    error: str = "",
    endpoint: str = "",
    log_name: str = "vlm_calls",
) -> None:
    """VLM 호출 결과를 로그에 기록한다."""
    logger = _get_logger(log_name)
    tokens = _format_tokens(token_usage)

    if status == "ok":
        logger.info(
            "service=%s model=%s status=ok latency_ms=%.1f %s endpoint=%s",
            service, model, latency_ms, tokens, endpoint,
        )
    else:
        logger.error(
            "service=%s model=%s status=error latency_ms=%.1f %s error=%s endpoint=%s",
            service, model, latency_ms, tokens, error, endpoint,
        )


def log_work2_event(
    *,
    component: str,
    message: str,
    level: str = "info",
    log_name: str = "work2",
    **fields: object,
) -> None:
    """Phase 2 일반 이벤트를 파일 로그에 기록한다."""
    logger = _get_logger(log_name)
    extras = _format_fields(fields)
    line = f"component={component} message={message}"
    if extras:
        line = f"{line} {extras}"

    normalized_level = level.strip().lower()
    if normalized_level == "error":
        logger.error(line)
    elif normalized_level in {"warn", "warning"}:
        logger.warning(line)
    else:
        logger.info(line)


__all__ = ["log_vlm_call", "log_work2_event"]
