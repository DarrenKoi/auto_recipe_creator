"""워크플로 타입 정의."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path


def _serialize_value(value):
    """dataclass / enum 을 JSON 직렬화 가능한 값으로 변환한다."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            key: _serialize_value(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


class ConditionType(Enum):
    """step 조건의 종류."""

    ALWAYS = "always"
    WINDOW_VISIBLE = "window_visible"
    WINDOW_FOUND = "window_found"
    WINDOW_APPEARED = "window_appeared"
    DIALOG_DISAPPEARED = "dialog_disappeared"
    PROCESS_ALIVE = "process_alive"
    FIELD_READY_FOR_INPUT = "field_ready_for_input"
    TEXT_APPEARED = "text_appeared"
    TEXT_ALREADY_PRESENT = "text_already_present"
    MASKED_TEXT_PRESENT = "masked_text_present"
    # context[target_key] 가 None 이 아니면 충족 — executor 가 context 에 심은
    # 산출물(창 핸들, controller 등)의 존재를 success_criteria 로 검증할 때 쓴다.
    CONTEXT_KEY_SET = "context_key_set"


class ConditionGroupType(Enum):
    """조건 그룹 결합 방식."""

    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class StepCondition:
    """단일 원자 조건."""

    condition_type: ConditionType
    title_fragment: str | None = None
    title_prefix: str | None = None
    exe_name: str | None = None
    expected_text: str | None = None
    verify_method: str | None = None
    target_key: str | None = None

    def to_dict(self) -> dict:
        """JSON 저장용 dict 로 변환한다."""
        return _serialize_value(self)


@dataclass(frozen=True)
class ConditionGroup:
    """여러 원자 조건을 AND / OR 로 묶는다."""

    group_type: ConditionGroupType = ConditionGroupType.ALL
    conditions: list[StepCondition] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON 저장용 dict 로 변환한다."""
        return _serialize_value(self)


@dataclass(frozen=True)
class WorkflowStep:
    """워크플로의 단일 실행 단위."""

    step_id: str
    step_type: str
    target_description: str
    target_key: str | None = None
    preconditions: ConditionGroup = field(default_factory=ConditionGroup)
    success_criteria: ConditionGroup = field(default_factory=ConditionGroup)
    skip_if: ConditionGroup | None = None
    safety_tier: int = 2
    max_retries: int = 0
    retry_profile: str = "phase1_no_retry"
    depends_on: list[str] | None = None
    idempotent: bool = True
    timeout_sec: float = 30.0
    detect_timeout_sec: float = 15.0
    act_timeout_sec: float = 5.0
    verify_timeout_sec: float = 10.0
    reserved_retry_budget: int = 0
    input_text: str | None = None
    action_value: str | None = None
    redact_input_text: bool = False

    def to_dict(self) -> dict:
        """JSON 저장용 dict 로 변환한다."""
        return _serialize_value(self)


@dataclass
class StepResult:
    """단일 step 의 실행 결과."""

    step_id: str
    status: str
    failure_class: str | None
    attempt_count: int
    strategy_used: str
    vlm_service_used: str
    detected_point: dict | None
    detected_bbox: dict | None
    screen_point: dict | None
    verification_result: dict | None
    before_screenshot: str | None
    after_screenshot: str | None
    error_message: str | None
    elapsed_ms: float
    timestamp: str
    window_title_before: str | None = None
    window_title_after: str | None = None
    safe_mode: bool = False
    artifact_redacted: bool = False
    needs_manual_check: bool = False
    manual_check_reason: str | None = None

    def to_dict(self) -> dict:
        """JSON 저장용 dict 로 변환한다."""
        return _serialize_value(self)


@dataclass
class WorkflowRun:
    """워크플로 전체 실행 결과."""

    run_id: str
    workflow_name: str
    status: str
    started_at: str
    finished_at: str | None = None
    current_step_index: int = -1
    total_retries_used: int = 0
    retry_budget_remaining: int = 0
    settings_snapshot: dict = field(default_factory=dict)
    interrupts_encountered: list[dict] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    run_dir: str | None = None

    def to_dict(self) -> dict:
        """JSON 저장용 dict 로 변환한다."""
        return _serialize_value(self)


__all__ = [
    "ConditionGroup",
    "ConditionGroupType",
    "ConditionType",
    "StepCondition",
    "StepResult",
    "WorkflowRun",
    "WorkflowStep",
]
