"""work2 VLM 호출 로거.

VLM 호출의 성공/실패, 응답 시간, 토큰 사용량을 파일에 기록한다.
로그 파일: poc/work2/logs/vlm_calls.log
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = "vlm_calls.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 5
_HANDLER_MARKER = "_work2_vlm_logger"

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """싱글턴 로거를 반환한다."""
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("poc.work2.vlm")
    level_name = os.environ.get("WORK2_LOG_LEVEL", "INFO").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            _logger = logger
            return logger

    log_path = _LOG_DIR / _LOG_FILE
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

    _logger = logger
    return logger


def _format_tokens(token_usage: dict[str, int] | None) -> str:
    """토큰 사용량을 로그 문자열로 변환한다."""
    if not token_usage:
        return "tokens=N/A"
    prompt = token_usage.get("prompt_tokens", "?")
    completion = token_usage.get("completion_tokens", "?")
    total = token_usage.get("total_tokens", "?")
    return f"prompt_tokens={prompt} completion_tokens={completion} total_tokens={total}"


def log_vlm_call(
    *,
    service: str,
    model: str,
    status: str,
    latency_ms: float,
    token_usage: dict[str, int] | None = None,
    error: str = "",
    endpoint: str = "",
) -> None:
    """VLM 호출 결과를 로그에 기록한다."""
    logger = _get_logger()
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


__all__ = ["log_vlm_call"]
