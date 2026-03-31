"""워크플로 실행 기본 설정."""

from dataclasses import asdict, dataclass

from poc.work2.util import env_flag, env_float, env_int


@dataclass(frozen=True)
class WorkflowSettings:
    """워크플로 공통 설정."""

    verify_service: str = "paddleocr-vl-1.5"
    service_fallback_order: tuple[str, ...] = ("ui-venus", "mai-ui")
    total_retry_budget: int = 10
    settle_max_wait_sec: float = 3.0
    settle_similarity_threshold: float = 0.98
    allow_optional_imagehash: bool = False
    safe_mode: bool = True
    action_enabled: bool = False
    typing_enabled: bool = True
    pre_click_settle_sec: float = 0.2
    post_click_settle_sec: float = 0.3
    pre_type_click_settle_sec: float = 1.0
    pre_type_double_click_settle_sec: float = 0.1
    post_type_backspace_settle_sec: float = 0.05
    char_type_delay_sec: float = 0.03
    post_type_settle_sec: float = 0.3
    post_login_wait_sec: float = 3.0
    login_verify_timeout_sec: float = 15.0
    login_verify_poll_interval_sec: float = 2.0

    def to_snapshot(self) -> dict:
        """상태 저장용 설정 스냅샷."""
        return asdict(self)


def load_workflow_settings() -> WorkflowSettings:
    """환경변수와 기본값으로 WorkflowSettings 를 생성한다."""
    safe_mode = env_flag("SAFE_MODE", default=True)
    action_enabled = env_flag("ACTION_LOGIN_ACTION_ENABLED", default=not safe_mode)

    return WorkflowSettings(
        verify_service="paddleocr-vl-1.5",
        service_fallback_order=("ui-venus", "mai-ui"),
        total_retry_budget=env_int("WORKFLOW_TOTAL_RETRY_BUDGET", 10),
        settle_max_wait_sec=env_float("WORKFLOW_SETTLE_MAX_WAIT_SEC", 3.0),
        settle_similarity_threshold=env_float("WORKFLOW_SETTLE_SIMILARITY_THRESHOLD", 0.98),
        allow_optional_imagehash=env_flag("WORKFLOW_ALLOW_OPTIONAL_IMAGEHASH", default=False),
        safe_mode=safe_mode,
        action_enabled=action_enabled,
        typing_enabled=env_flag("ACTION_LOGIN_TYPING_ENABLED", default=True),
        pre_click_settle_sec=env_float("ACTION_LOGIN_PRE_CLICK_SETTLE_SEC", 0.2),
        post_click_settle_sec=env_float("ACTION_LOGIN_POST_CLICK_SETTLE_SEC", 0.3),
        pre_type_click_settle_sec=env_float("ACTION_LOGIN_PRE_TYPE_CLICK_SETTLE_SEC", 1.0),
        pre_type_double_click_settle_sec=env_float(
            "ACTION_LOGIN_PRE_TYPE_DOUBLE_CLICK_SETTLE_SEC",
            0.1,
        ),
        post_type_backspace_settle_sec=env_float(
            "ACTION_LOGIN_POST_TYPE_BACKSPACE_SETTLE_SEC",
            0.05,
        ),
        char_type_delay_sec=env_float("ACTION_LOGIN_CHAR_TYPE_DELAY_SEC", 0.03),
        post_type_settle_sec=env_float("ACTION_LOGIN_POST_TYPE_SETTLE_SEC", 0.3),
        post_login_wait_sec=env_float("ACTION_LOGIN_POST_LOGIN_WAIT_SEC", 3.0),
        login_verify_timeout_sec=env_float("ACTION_LOGIN_VERIFY_TIMEOUT_SEC", 15.0),
        login_verify_poll_interval_sec=env_float("ACTION_LOGIN_VERIFY_POLL_INTERVAL_SEC", 2.0),
    )


__all__ = ["WorkflowSettings", "load_workflow_settings"]
