"""시간 측정 유틸리티."""

import time


def make_timestamp_tag(now: float | None = None) -> str:
    """파일명에 넣기 좋은 초 단위 타임스탬프를 반환한다."""
    resolved_now = time.time() if now is None else now
    return time.strftime("%y%m%d_%H%M%S", time.localtime(resolved_now))


def format_elapsed_ms(start_time: float) -> str:
    """start_time 이후 경과 시간을 사람이 읽기 쉬운 문자열로 반환한다."""
    elapsed_ms = (time.time() - start_time) * 1000
    if elapsed_ms < 1000:
        return f"{elapsed_ms:.0f}ms"
    return f"{elapsed_ms / 1000:.2f}s"
