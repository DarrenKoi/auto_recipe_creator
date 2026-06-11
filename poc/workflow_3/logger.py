"""workflow_3 파일 로거.

기본값은 운영 중 바로 봐야 하는 warning/error 만 파일에 남긴다.
상세 추적이 필요할 때만 WORKFLOW3_FILE_LOG_DETAIL=1 로 info 로그를 켠다.
"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler

from poc.workflow_3 import LOG_DIR

_LOG_DIR = LOG_DIR
_MAX_BYTES = 1024 * 1024
_BACKUP_COUNT = 1
_HANDLER_MARKER = "_workflow_3_vlm_logger"
_DETAIL_ENV = "WORKFLOW3_FILE_LOG_DETAIL"

_loggers: dict[str, logging.Logger] = {}


def _sanitize_log_name(log_name: str) -> str:
    """로그 파일명에 안전한 식별자로 정규화한다."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", (log_name or "").strip()).strip("-._")
    return name or "workflow_3"


def _get_logger(log_name: str) -> logging.Logger:
    """log_name 별 싱글턴 로거를 반환한다."""
    safe_name = _sanitize_log_name(log_name)
    existing = _loggers.get(safe_name)
    if existing is not None:
        return existing

    logger = logging.getLogger(f"poc.workflow_3.{safe_name}")
    level_name = (
        os.environ.get("WORKFLOW3_LOG_LEVEL")
        or os.environ.get("WORK2_LOG_LEVEL")
        or "INFO"
    ).strip().upper()
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
        print(f"[WARNING] workflow_3 로거 초기화 실패: {exc}")

    _loggers[safe_name] = logger
    return logger


def _detail_enabled() -> bool:
    """파일 info 로그 활성 여부를 반환한다."""
    value = os.environ.get(_DETAIL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on", "debug", "verbose"}


def _should_write(level: str) -> bool:
    """warning/error 는 항상, info 는 상세 모드에서만 기록한다."""
    normalized = (level or "info").strip().lower()
    if normalized in {"error", "warn", "warning"}:
        return True
    return _detail_enabled()


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
    response_text: str = "",
    log_name: str = "vlm_calls",
) -> None:
    """VLM 호출 결과를 로그에 기록한다."""
    if not _should_write("error" if status != "ok" else "info"):
        return

    logger = _get_logger(log_name)
    tokens = _format_tokens(token_usage)

    if status == "ok":
        logger.info(
            "service=%s model=%s status=ok latency_ms=%.1f %s",
            service, model, latency_ms, tokens,
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
    """workflow_3 일반 이벤트를 파일 로그에 기록한다."""
    if not _should_write(level):
        return

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
