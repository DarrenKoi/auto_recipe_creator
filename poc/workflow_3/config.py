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
    # 자동 GUI 구간 동안 사용자 물리 마우스/키보드 입력 차단(Windows BlockInput).
    # 사용자가 다른 앱을 쓰면 foreground lock 으로 RCS 가 안 떠서 방해되는 문제 대응.
    # 기본 off(opt-in) + SAFE_MODE off 일 때만 실제 적용. engineer watch 구간은 제외
    # (엔지니어가 직접 조작해야 하므로). Ctrl+Alt+Del 로 항상 해제 가능.
    block_input_enabled: bool = False

    # --- 상시 녹화 (변화 감지 기반 적응 캡처) ---
    recording_poll_sec: float = 0.3  # 샘플링 간격 — 조작 중 커서 궤적 추적 밀도.
    recording_heartbeat_sec: float = 5.0  # 변화 없어도 이 간격마다 1장 저장.
    recording_change_min_px: int = 4  # 변화 판정: delta>15 인 다운샘플 픽셀 최소 개수.
    recording_max_sec: float = 900.0
    engineer_watch_sec: float = 300.0  # 미보정 watch 상한(cap, 5분) - done 감지 시 조기 종료.

    # --- engineer watch 측정-시작 감지 (Recipe Monitor 카운터) ---
    # 미보정 watch 중 측정 카운터 분자(N/M 의 N)가 증가하면 align 완료로 보고
    # 녹화를 조기 종료한다. VLM grounding 1회 + CV gate + OCR confirm(연속 2회).
    engineer_done_detect_enabled: bool = False  # 오피스 캘리브레이션 검증 전 기본 off.
    engineer_done_poll_sec: float = 8.0  # watch 안 detector 호출 간격.
    engineer_done_min_count: int = 6  # done(=watch 종료+tool 닫기 트리거) 최소 분자값 — N>5 까지 측정 확인 후 닫기.
    engineer_done_change_min_px: int = 4  # CV gate 변화 픽셀 임계(다운샘플).
    engineer_done_relocalize_after_miss: int = 3  # 변화 후 OCR 연속 미검출 시 재grounding.
    # 재정렬 진행 중에는 카운터(N/M)가 빈칸이라 grounding 이 거부될 수 있다(정상).
    # 거부/실패 후 이 간격으로 재시도한다 (VLM 호출 폭주 방지 throttle).
    engineer_done_reground_sec: float = 30.0
    engineer_done_roi_pad_x: float = 0.03  # grounding 점 -> crop 확장 비율(가로, 창 대비).
    engineer_done_roi_pad_y: float = 0.02  # grounding 점 -> crop 확장 비율(세로, 창 대비).
    # 주의: Workflow1VLMClient 는 모델명이 아니라 flask_vlm 의 route_slug 를 받는다
    # ("ui-venus" O, "ui-venus-1.5-8b" X - workflow_2 스크립트들과 동일 규약).
    engineer_done_vlm_service: str = "ui-venus"  # grounding 서비스 slug.
    engineer_done_ocr_service: str = "paddleocr-vl-1.5"  # 분자 OCR 서비스 slug.

    # --- consensus S-image gather ---
    gather_enabled: bool = True
    gather_max_events: int = 5  # vision/consensus_gather.py 의 GATHER_MAX_EVENTS 와 동일 값 유지.

    # --- CV 보정 ---
    correction_enabled: bool = True
    correction_dry_run: bool = True  # False 는 SAFE_MODE off + env 명시(0)일 때만.
    ok_button_vlm_service: str = "ui-venus"  # route_slug (모델명 "ui-venus-1.5-8b" 아님).
    sem_mode_default: str = "SEM"
    sem_controller_settle_sec: float = 0.5
    zoom_scroll_dy: int = 1

    # --- 모호 키 재등록 알림 ---
    # second_ratio(2nd/best chamfer)가 이보다 크면 만성적으로 모호한 align key 로 보고
    # 엔지니어에게 재등록을 권고한다. tau*(S-LOO golden 보정, AUC 0.91) 유래의 시작점이며
    # fail-frame 재보정 대상 · matcher 의 0.94 visibility 게이트(max_second_ratio)와는 별개.
    reregister_second_ratio_threshold: float = 0.98

    # --- cond box-crop template (Tier 1.1) ---
    # True(기본): cond.box_ltrb 로 box-crop template + decoupled offset(office 검증 rank1 +0.16~0.18).
    # False: whole-template(구 동작) 롤백 — env ALIGN_FAIL_COND_BOX_CROP=0.
    cond_box_crop: bool = True


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
        block_input_enabled=env_flag("ALIGN_FAIL_BLOCK_INPUT", default=False),
        recording_poll_sec=env_float("ALIGN_FAIL_RECORDING_POLL_SEC", 0.3),
        recording_heartbeat_sec=env_float("ALIGN_FAIL_RECORDING_HEARTBEAT_SEC", 5.0),
        recording_change_min_px=env_int("ALIGN_FAIL_RECORDING_CHANGE_MIN_PX", 4),
        recording_max_sec=env_float("ALIGN_FAIL_RECORDING_MAX_SEC", 900.0),
        engineer_watch_sec=env_float("ALIGN_FAIL_ENGINEER_WATCH_SEC", 300.0),
        gather_enabled=env_flag("ALIGN_FAIL_GATHER_SUCCESS", default=True),
        gather_max_events=env_int("ALIGN_FAIL_GATHER_MAX_EVENTS", 5),
        correction_enabled=env_flag("ALIGN_FAIL_CORRECTION", default=True),
        correction_dry_run=correction_dry_run,
        ok_button_vlm_service=_env_str("ALIGN_OK_BUTTON_VLM_SERVICE", "ui-venus"),
        sem_mode_default=_env_str("ALIGN_SEM_MODE_DEFAULT", "SEM"),
        sem_controller_settle_sec=env_float("ALIGN_SEM_SETTLE_SEC", 0.5),
        zoom_scroll_dy=env_int("ALIGN_SEM_ZOOM_SCROLL_DY", 1),
        engineer_done_detect_enabled=env_flag("ALIGN_FAIL_ENGINEER_DONE_DETECT", default=False),
        engineer_done_poll_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_POLL_SEC", 8.0),
        engineer_done_min_count=env_int("ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT", 6),
        engineer_done_change_min_px=env_int("ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX", 4),
        engineer_done_relocalize_after_miss=env_int("ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS", 3),
        engineer_done_reground_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_REGROUND_SEC", 30.0),
        engineer_done_roi_pad_x=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_X", 0.03),
        engineer_done_roi_pad_y=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_Y", 0.02),
        engineer_done_vlm_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE", "ui-venus"),
        engineer_done_ocr_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE", "paddleocr-vl-1.5"),
        reregister_second_ratio_threshold=env_float("ALIGN_FAIL_REREGISTER_RATIO", 0.98),
        cond_box_crop=env_flag("ALIGN_FAIL_COND_BOX_CROP", default=True),
    )


__all__ = ["Workflow3Settings", "load_workflow3_settings"]
