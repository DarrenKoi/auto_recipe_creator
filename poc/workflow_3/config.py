"""workflow_3 모니터링 루프 설정.

`runner.workflow_config.WorkflowSettings`(클릭/타이핑/검증 타이밍 공통)를 상속해
align fail 루프 전용 필드를 추가한다. env 이름은 align_fail_alarm_record 시절의
값을 그대로 재사용해 오피스 .env / 운영 습관과 호환을 유지한다.

actuation 게이트 정리:
  * SAFE_MODE=1            → action_enabled=False (모든 마우스/키보드 차단, 상속 동작)
  * ALIGN_FAIL_CORRECTION_DRY_RUN (기본 1) → CV 보정의 move/click 만 추가로 차단.
    실제 보정 클릭은 SAFE_MODE off **이고** 이 값이 0일 때만 나간다(이중 게이트).
"""

import os
from dataclasses import dataclass

from poc.workflow_3.runner.workflow_config import WorkflowSettings, load_workflow_settings
from poc.workflow_3.util import env_flag, env_float, env_int


def _env_str(name: str, default: str) -> str:
    """공백 제거한 문자열 env (비어 있으면 default)."""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_alarm_source() -> str:
    """알람 소스 env ("office" | "replay"). 그 외 값은 경고 후 office."""
    value = _env_str("ALIGN_FAIL_ALARM_SOURCE", "office").lower()
    if value not in {"office", "replay"}:
        print(f"[WARNING] ALIGN_FAIL_ALARM_SOURCE={value!r} 미지원 - office 로 진행")
        return "office"
    return value


@dataclass(frozen=True)
class Workflow3Settings(WorkflowSettings):
    """align fail 모니터링 루프 설정."""

    # --- 알람 폴링 ---
    poll_interval_sec: int = 10
    detection_window_sec: int = 60
    alarm_source: str = "office"  # "office" | "replay"

    # --- 알림 ---
    popup_enabled: bool = True
    popup_timeout_sec: int = 60
    alert_close_timeout_sec: int = 3
    rich_notify_enabled: bool = True

    # --- 알람별 사이클 ---
    cycle_enabled: bool = True
    connect_action_enabled: bool = True  # tool 더블클릭 수행 여부(off=인식만 dry-run).
    connect_window_timeout_sec: int = 3
    rcs_window_max_trials: int = 10
    rcs_recovery_enabled: bool = False  # RCS 재실행+재로그인 복구(검증 전 기본 off).
    keep_awake: bool = True

    # --- 상시 녹화 (변화 감지 기반 적응 캡처) ---
    recording_poll_sec: float = 0.3  # 샘플링 간격 — 조작 중 커서 궤적 추적 밀도.
    recording_heartbeat_sec: float = 5.0  # 변화 없어도 이 간격마다 1장 저장.
    recording_change_min_px: int = 4  # 변화 판정: delta>15 인 다운샘플 픽셀 최소 개수.
    recording_max_sec: float = 900.0
    engineer_watch_sec: float = 600.0

    # --- CV 보정 ---
    correction_enabled: bool = True
    correction_dry_run: bool = True  # False 는 SAFE_MODE off + env 명시(0)일 때만.
    ok_button_vlm_service: str = "ui-venus-1.5-8b"
    sem_mode_default: str = "SEM"
    sem_controller_settle_sec: float = 0.5
    zoom_scroll_dy: int = 1


def load_workflow3_settings() -> Workflow3Settings:
    """env 오버라이드를 적용해 Workflow3Settings 를 생성한다."""
    base = load_workflow_settings()

    # 보정 actuation 이중 게이트: SAFE_MODE 가 켜져 있으면 env 와 무관하게 dry-run.
    dry_run_requested = env_flag("ALIGN_FAIL_CORRECTION_DRY_RUN", default=True)
    correction_dry_run = dry_run_requested or base.safe_mode

    return Workflow3Settings(
        **base.to_snapshot(),
        poll_interval_sec=env_int("ALIGN_FAIL_POLL_SEC", 10),
        detection_window_sec=env_int("ALIGN_FAIL_WINDOW_SEC", 60),
        alarm_source=_env_alarm_source(),
        popup_enabled=env_flag("ALIGN_FAIL_POPUP", default=True),
        popup_timeout_sec=env_int("ALIGN_FAIL_POPUP_TIMEOUT_SEC", 60),
        alert_close_timeout_sec=env_int("ALIGN_FAIL_ALERT_CLOSE_TIMEOUT_SEC", 3),
        rich_notify_enabled=env_flag("ALIGN_FAIL_RICH_NOTIFY", default=True),
        cycle_enabled=env_flag("ALIGN_FAIL_RECORD_CYCLE", default=True),
        connect_action_enabled=env_flag("ALIGN_FAIL_CONNECT_ACTION", default=True),
        connect_window_timeout_sec=env_int("ALIGN_FAIL_CONNECT_WINDOW_TIMEOUT_SEC", 3),
        rcs_window_max_trials=env_int("ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS", 10),
        rcs_recovery_enabled=env_flag("ALIGN_FAIL_RCS_RECOVERY", default=False),
        keep_awake=env_flag("ALIGN_FAIL_KEEP_AWAKE", default=True),
        recording_poll_sec=env_float("ALIGN_FAIL_RECORDING_POLL_SEC", 0.3),
        recording_heartbeat_sec=env_float("ALIGN_FAIL_RECORDING_HEARTBEAT_SEC", 5.0),
        recording_change_min_px=env_int("ALIGN_FAIL_RECORDING_CHANGE_MIN_PX", 4),
        recording_max_sec=env_float("ALIGN_FAIL_RECORDING_MAX_SEC", 900.0),
        engineer_watch_sec=env_float("ALIGN_FAIL_ENGINEER_WATCH_SEC", 600.0),
        correction_enabled=env_flag("ALIGN_FAIL_CORRECTION", default=True),
        correction_dry_run=correction_dry_run,
        ok_button_vlm_service=_env_str("ALIGN_OK_BUTTON_VLM_SERVICE", "ui-venus-1.5-8b"),
        sem_mode_default=_env_str("ALIGN_SEM_MODE_DEFAULT", "SEM"),
        sem_controller_settle_sec=env_float("ALIGN_SEM_SETTLE_SEC", 0.5),
        zoom_scroll_dy=env_int("ALIGN_SEM_ZOOM_SCROLL_DY", 1),
    )


__all__ = ["Workflow3Settings", "load_workflow3_settings"]
