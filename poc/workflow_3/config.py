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
    rcs_window_max_trials: int = 3  # 점유 'select' 팝업 조기 감지가 있어 상한을 낮춤(과거 10).
    rcs_recovery_enabled: bool = False  # RCS 재실행+재로그인 복구(검증 전 기본 off).
    # 점유 'select' 팝업(타 사용자 사용 중) 검출 — 떠 있으면 접속 포기 + cooldown 후 재시도.
    occupied_popup_detect_enabled: bool = True
    occupied_popup_vlm_service: str = "ui-venus"  # 제목 검출 후 옵션 확인용(route_slug).
    occupied_retry_cooldown_sec: float = 300.0    # 점유로 포기한 tool 재시도 유예(초).
    # 점유 외 사유로 사이클이 실패한 tool 의 재시도 유예(초). 없으면 매 poll 재시도해
    # 직렬화된 단일 RCS 커서를 독점하고 다른 알람을 굶긴다(F2).
    failure_retry_cooldown_sec: float = 300.0
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

    # --- rcp 입력 이미지 office 다운로드 ---
    # align_img_from_rcp(등록 align key)는 보정/점검의 런타임 입력이다. 보정/feasibility 는
    # 라이브 캡처 프레임에 consensus(우선)/rcp(폴백) 템플릿을 매칭하며, align_img_from_msr
    # (측정 궤적)은 런타임에서 소비하지 않으므로 프로덕션 gather 는 rcp 만 받는다(rcp-only).
    # msr 은 오프라인 벤치에서만 fetch_msr_offline.py 로 받는다. 기본 계약은 office MES 가
    # align_images 트리에 직접 적재하는 것이지만, 못 받는 환경에선 office_rcp_msr_downloader
    # 가 알람 시점에 동기로 내려받는다(cycle 이 assets 읽기 전 디스크 적재 보장).
    rcp_msr_gather_enabled: bool = True
    # 동기 rcp 다운로드 대기 상한(초). office 호출이 걸려도 모니터가 무한 정지하지
    # 않게 한다(F3). 초과 시 받은 만큼으로 진행 - assets 부분/부재 가능성 있음.
    rcp_gather_timeout_sec: float = 60.0

    # --- consensus S-image gather ---
    gather_enabled: bool = True
    gather_max_events: int = 4  # align/consensus_gather.py 의 GATHER_MAX_EVENTS 와 동일 값 유지.
    consensus_enabled: bool = True            # consensus 라우팅 마스터 토글(off -> 순수 rcp).
    consensus_min_s: int = 4                  # modality별 build·신뢰 최소 S(floor 3).
    consensus_sync_timeout_sec: float = 8.0   # cold-cache bounded 대기(초).
    consensus_refresh_ttl_sec: int = 21600    # gather 재fetch TTL(초, 6h).

    # --- 점검 모니터 보정 가능성 마킹 ---
    # 점검 전용 사이클(align_fail_monitor_only_check)에서 캡처 후 rcp 엔진으로 보정
    # 가능/불가를 판정해 캡처 옆 _marked.jpg + _feasibility.json 으로 남길지. consensus
    # cache 의 S event 수도 read-only 로 표기. production 보정 사이클에는 영향 없음.
    feasibility_mark_enabled: bool = True
    # 점검 사이클에서 보정 가능성 판정으로 구한 align point 로 마우스 커서를 옮겨(클릭 없이)
    # live SEM 박스 위 좌표 매핑을 눈으로 검증할지. tool 을 닫기 전에 이동하고, 커서가
    # 안착한 화면을 다시 캡처(_rcs_cursor.jpg)한다. 기본 off(opt-in) + SAFE_MODE off 일
    # 때만 실제 이동(켜도 action_enabled=False 면 DRY-RUN 로그만). production 보정과 무관.
    reposition_preview_enabled: bool = False
    # 점검 사이클의 보정 가능성 마킹에서 live SEM box 를 VLM 으로 검출해 (1) PM 박스로
    # OM/SEM modality 를 정하고 (2) box 안쪽만 매칭한 뒤 align point 를 풀프레임으로
    # 되돌리고 (3) box 를 overlay 에 그릴지. off 면 기존 전체 창 매칭으로 폴백한다.
    sem_box_detect_enabled: bool = True
    sem_box_vlm_service: str = "ui-venus"  # route_slug (모델명 "ui-venus-1.5-8b" 아님).
    # PM 모드 읽기 2단계: off(기본)=단일 호출의 inline pm_box_text. on=같은 호출이 준 PM
    # 위치를 crop 해 PaddleOCR 로 재독(작은 영역 정확도↑). PM crop 은 항상 디버그 저장된다.
    pm_two_stage_ocr_enabled: bool = False
    pm_ocr_service: str = "paddleocr-vl-1.5"  # slug==모델명(비대칭). crop OCR 용.

    # --- 점검 모니터 zoom in/out 보정탐색(ladder) ---
    # feasibility verdict 가 모호(ambiguous)/부재(not_visible)라 "어느 점이 align point 인지"
    # 가릴 수 없을 때, tool 이 열린 동안 live SEM box 안으로 커서를 옮긴 뒤 mouse wheel 로
    # fail 시점 배율 기준 OUT(배율↓) · IN(배율↑) 양방향으로 한 칸씩 훑어 각 배율(rung)의
    # 화면을 저장한다. zoom-out 만으로는 키를 못 찾으므로(좁은 FOV→넓게 보고, 다시 좁혀
    # 정확한 점 확인) 양방향이 필요하다. 클릭/recenter 없음 = 순수 wheel+캡처. rematch 가
    # 켜져 있으면 각 rung 에서 rcp 키를 재매칭(mark_align_feasibility)해 키가 또렷해지는
    # 배율을 표시한다. 기본 on + SAFE_MODE off 일 때만 실제 wheel(켜도
    # action_enabled=False 면 DRY-RUN 로그만). wheel 대상은 반드시 검출된 live SEM box
    # 중심(없으면 탐색 생략 — 창 중심에 잘못 스크롤 방지). 배율 복원은 arm 전환 시 baseline
    # 복귀에만 쓰고 종료 시엔 복원하지 않는다(장비 fail 정지 → 엔지니어 재셋업). PM 버튼
    # 드롭다운(절대 배율 선택) 방식은 추후 옵션(미구현). production 보정과 무관.
    zoom_probe_enabled: bool = True
    zoom_probe_steps: int = 2                   # OUT(배율↓) 방향 단계 수.
    zoom_probe_in_steps: int = 2               # IN(배율↑) 방향 단계 수.
    zoom_probe_scroll_dy: int = -1              # 음수 = wheel down = OUT = 배율↓ (pynput scroll dy). IN 은 부호 반전.
    zoom_probe_scrolls_per_step: int = 5        # 단계당 scroll notch 수. 1-2 notch 로는 배율이 거의
                                                # 무의미하게 바뀌어 의미 있는 step 이 되도록 5 로 상향(오피스 rung 별 mag 보고 조정).
    zoom_probe_settle_sec: float = 0.6          # wheel 후 FOV 재렌더+커서 안착 대기(초).
    zoom_probe_rematch_enabled: bool = True     # 각 rung 에서 rcp 키 재매칭(off 면 캡처만).
    # wheel 이 배율을 안 바꾸는 tool 대비 fallback: out1 wheel 후 PM 배율이 그대로면
    # 'PM' 버튼 드롭다운(절대 배율 선택)으로 전환해 ladder 를 마저 돈다. 기본 on.
    pm_dropdown_enabled: bool = True
    # zoom ladder 방식: "auto"=wheel 먼저 시도 후 무효(out1 후 PM 배율 불변)면 PM 드롭다운
    # fallback(기본 — wheel 이 안 듣던 장비가 fab-out 되어 wheel 우선이 안전), "pm_dropdown"=
    # wheel 생략하고 곧장 PM 버튼 드롭다운(wheel 이 mag 을 안 바꾸고 RCS 가 recenter 로 오해하는
    # tool), "wheel"=wheel 만(드롭다운 fallback 없음, 다른 tool 용).
    zoom_method: str = "auto"

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

    # --- VLM 2단계 로케이터 조합 (로그인 / List 탭 / tool 선택 / PM 버튼 공통) ---
    # "coarse>fine" route_slug 조합. 빈 문자열 = production 기본(ui-venus>mai-ui).
    # A/B 시험 예: "mai-ui>mai-ui". env 이름은 ALIGN_FAIL_* 가 아니라 VLM_LOCATOR_COMBO 다
    # (모니터 루프뿐 아니라 rcs/ 단독 스크립트도 같은 스위치를 쓰기 때문).
    #
    # 주의: 실제 actuator 는 vlm/ui_venus_mai_locator.resolve_locator_services() 이고,
    # 그쪽이 호출 시점에 env 를 직접 읽는다. 이 필드는 같은 env 를 읽어 미러링하는
    # 선언/가시성용이라, 이 필드만 코드에서 바꿔 써도 로케이터는 안 바뀐다.
    locator_combo: str = ""

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
        rcs_window_max_trials=env_int("ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS", 3),
        occupied_popup_detect_enabled=env_flag("ALIGN_FAIL_OCCUPIED_POPUP_DETECT", default=True),
        occupied_popup_vlm_service=_env_str("ALIGN_FAIL_OCCUPIED_POPUP_SERVICE", "ui-venus"),
        occupied_retry_cooldown_sec=env_float("ALIGN_FAIL_OCCUPIED_COOLDOWN_SEC", 300.0),
        failure_retry_cooldown_sec=env_float("ALIGN_FAIL_FAILURE_COOLDOWN_SEC", 300.0),
        rcs_recovery_enabled=env_flag("ALIGN_FAIL_RCS_RECOVERY", default=False),
        keep_awake=env_flag("ALIGN_FAIL_KEEP_AWAKE", default=True),
        block_input_enabled=env_flag("ALIGN_FAIL_BLOCK_INPUT", default=False),
        recording_poll_sec=env_float("ALIGN_FAIL_RECORDING_POLL_SEC", 0.3),
        recording_heartbeat_sec=env_float("ALIGN_FAIL_RECORDING_HEARTBEAT_SEC", 5.0),
        recording_change_min_px=env_int("ALIGN_FAIL_RECORDING_CHANGE_MIN_PX", 4),
        recording_max_sec=env_float("ALIGN_FAIL_RECORDING_MAX_SEC", 900.0),
        engineer_watch_sec=env_float("ALIGN_FAIL_ENGINEER_WATCH_SEC", 300.0),
        rcp_msr_gather_enabled=env_flag("ALIGN_FAIL_GATHER_RCP_MSR", default=True),
        rcp_gather_timeout_sec=env_float("ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC", 60.0),
        gather_enabled=env_flag("ALIGN_FAIL_GATHER_SUCCESS", default=True),
        gather_max_events=env_int("ALIGN_FAIL_GATHER_MAX_EVENTS", 4),
        consensus_enabled=env_flag("ALIGN_FAIL_CONSENSUS", default=True),
        consensus_min_s=max(3, env_int("ALIGN_FAIL_CONSENSUS_MIN_S", 4)),
        consensus_sync_timeout_sec=env_float("ALIGN_FAIL_CONSENSUS_SYNC_TIMEOUT", 8.0),
        consensus_refresh_ttl_sec=env_int("ALIGN_FAIL_CONSENSUS_REFRESH_TTL", 21600),
        feasibility_mark_enabled=env_flag("ALIGN_FAIL_FEASIBILITY_MARK", default=True),
        reposition_preview_enabled=env_flag("ALIGN_FAIL_REPOSITION_PREVIEW", default=False),
        sem_box_detect_enabled=env_flag("ALIGN_FAIL_SEM_BOX_DETECT", default=True),
        sem_box_vlm_service=_env_str("ALIGN_FAIL_SEM_BOX_SERVICE", "ui-venus"),
        pm_two_stage_ocr_enabled=env_flag("ALIGN_FAIL_PM_TWO_STAGE_OCR", default=False),
        pm_ocr_service=_env_str("ALIGN_FAIL_PM_OCR_SERVICE", "paddleocr-vl-1.5"),
        zoom_probe_enabled=env_flag("ALIGN_FAIL_ZOOM_PROBE", default=True),
        zoom_probe_steps=env_int("ALIGN_FAIL_ZOOM_PROBE_STEPS", 2),
        zoom_probe_in_steps=env_int("ALIGN_FAIL_ZOOM_PROBE_IN_STEPS", 2),
        zoom_probe_scroll_dy=env_int("ALIGN_FAIL_ZOOM_PROBE_SCROLL_DY", -1),
        zoom_probe_scrolls_per_step=env_int("ALIGN_FAIL_ZOOM_PROBE_SCROLLS_PER_STEP", 5),
        zoom_probe_settle_sec=env_float("ALIGN_FAIL_ZOOM_PROBE_SETTLE_SEC", 0.6),
        zoom_probe_rematch_enabled=env_flag("ALIGN_FAIL_ZOOM_PROBE_REMATCH", default=True),
        pm_dropdown_enabled=env_flag("ALIGN_FAIL_PM_DROPDOWN", default=True),
        zoom_method=os.environ.get("ALIGN_FAIL_ZOOM_METHOD", "auto").strip().lower() or "auto",
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
        locator_combo=_env_str("VLM_LOCATOR_COMBO", ""),
    )


__all__ = ["Workflow3Settings", "load_workflow3_settings"]
