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


# --- 녹화 샘플링 기본값의 단일 출처 ---
# 알람 녹화(Workflow3Settings + ALIGN_FAIL_*)와 수동 녹화(ManualRecordSettings +
# MANUAL_RECORD_*)는 env 네임스페이스가 다르지만 **같은 캡처 메커니즘**을 쓰므로
# 기본값이 갈리면 안 된다. 예전에는 같은 숫자가 dataclass 기본값 2곳 + env 로더
# 2곳 + RecordingSession 생성자에 흩어져 있어, 한 번 튜닝하려면 다섯 곳을 맞춰
# 고쳐야 했고 하나를 놓치면 두 녹화 경로의 샘플링 주기가 조용히 달라졌다.
DEFAULT_RECORDING_POLL_SEC = 0.05
DEFAULT_RECORDING_HEARTBEAT_SEC = 5.0
DEFAULT_RECORDING_CHANGE_MIN_PX = 2


def _env_str(name: str, default: str) -> str:
    """공백 제거한 문자열 env (비어 있으면 default)."""
    value = os.environ.get(name, "").strip()
    return value or default


def _load_share_confirm_policy() -> str:
    """공유 요청 확인 정책 env ("strict" | "lenient" | "off"). 그 외 값은 경고 후 strict.

    오타를 조용히 흘려보내면 안 된다. 이 값은 남의 세션을 건드리는 클릭을 막는 게이트라,
    `ALIGN_FAIL_SHARE_CONFIRM=strcit` 같은 오타가 의도치 않은 정책으로 해석되면 안전
    장치가 사실상 꺼진 것과 같다. `SELECT_TOOL_ROW_CONFIRM` 과 같은 관용구를 쓴다.
    """
    valid = {"strict", "lenient", "off"}
    value = _env_str("ALIGN_FAIL_SHARE_CONFIRM", "strict").lower()
    if value in valid:
        return value
    print(
        f"[WARNING] ALIGN_FAIL_SHARE_CONFIRM={value!r} 는 알 수 없는 값 "
        f"({sorted(valid)}) -> 기본값 'strict' 사용"
    )
    return "strict"


def _env_alarm_source() -> str:
    """알람 소스 env ("office" | "replay"). 그 외 값은 경고 후 office."""
    value = _env_str("ALIGN_FAIL_ALARM_SOURCE", "office").lower()
    if value not in {"office", "replay"}:
        print(f"[WARNING] ALIGN_FAIL_ALARM_SOURCE={value!r} 미지원 - office 로 진행")
        return "office"
    return value


_ENGINEER_DONE_POSITIVE_SETTINGS = (
    ("engineer_done_min_ok_rows", "ALIGN_FAIL_ENGINEER_DONE_MIN_OK_ROWS"),
    (
        "engineer_done_assist_unusable_after",
        "ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER",
    ),
    (
        "engineer_done_numerator_increase_reads",
        "ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS",
    ),
)


def validate_engineer_done_priority_settings(settings) -> None:
    """engineer-done 완료 임계값이 모두 양수인지 확인한다."""
    invalid = [
        f"{env_name}={getattr(settings, field_name, None)!r}"
        for field_name, env_name in _ENGINEER_DONE_POSITIVE_SETTINGS
        if not isinstance(getattr(settings, field_name, None), int)
        or getattr(settings, field_name) <= 0
    ]
    if invalid:
        raise ValueError(
            "engineer-done priority thresholds must be positive: "
            + ", ".join(invalid)
        )


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
    # 감지 시점 cube 사전 고지("이 장비 들어갑니다"). 기본 off — 켜면 알람 1건당 cube 가
    # 2회 나간다(감지 + 결과). 반자동 모드는 결과 알림이 항상 발송되므로(awaiting_engineer_ok)
    # 사전 고지까지 켜면 매번 2건이다. 행동을 요구하는 쪽은 결과 알림이라 그쪽을 기본으로 둔다.
    detection_notify_enabled: bool = False
    # 결과-후-알림 정책의 안전장치. 사이클이 이 시간을 넘도록 결과를 못 내면 "자동 보정
    # 진행 중" 을 1회 고지한다 — 알람 중 장비는 멈춰 있으므로 무한 침묵은 그대로
    # downtime 이다. 0 이면 watchdog 을 끈다(결과가 나올 때까지 완전 침묵).
    notify_delay_sec: float = 90.0

    # --- 알람별 사이클 ---
    cycle_enabled: bool = True
    connect_action_enabled: bool = True  # tool 더블클릭 수행 여부(off=인식만 dry-run).
    connect_window_timeout_sec: int = 3
    rcs_window_max_trials: int = 3  # 점유 'select' 팝업 조기 감지가 있어 상한을 낮춤(과거 10).
    # RCS 재실행+재로그인 복구. 기본 on - RCS 가 떠 있지 않다는 이유로 알람을 통째로
    # 놓치는 것이 복구 시도보다 나쁘다. 복구는 tool 에 접속하지 않고 메인 창까지만
    # 간다(monitor/rcs_recovery.py). 롤백은 ALIGN_FAIL_RCS_RECOVERY=0.
    rcs_recovery_enabled: bool = True
    # 복구 로그인 후 메인 창 출현 대기 상한(초). connect_window_timeout_sec 와 같은
    # 계열이라 여기 둔다 - 모듈 상수로 두면 seed_env() 보다 먼저 읽혀
    # workflow_3_config.py 로 조정할 수 없다(셸 env 만 먹힘).
    rcs_recovery_window_timeout_sec: float = 30.0
    # 창이 하나도 없는 RCS 프로세스(작업 표시줄에 안 보이는 좀비)를 종료하고 재실행할지.
    # 기본 off - 프로세스 종료는 되돌릴 수 없어 opt-in 이다. 켜도 **창을 가진** 프로세스는
    # 절대 건드리지 않는다(그건 누군가 쓰는 세션이다). 2026-08-19 오피스에서 이 좀비가
    # 중복 실행 가드에 걸려 복구가 통째로 막히는 것을 실측했다.
    rcs_kill_stale_enabled: bool = False
    # 모니터 기동 시 RCS 준비(실행 -> 로그인 -> List 탭)를 루프 진입 전 1회 수행.
    # 기본 on - 안 하면 첫 알람이 RCS 부팅+로그인 비용을 통째로 낸다(장비는 그동안
    # 멈춰 있다). 실행/로그인 여부는 rcs_recovery_enabled 게이트도 함께 본다.
    # 롤백은 ALIGN_FAIL_RCS_PREFLIGHT=0 (알람 시 복구는 그대로 유지된다).
    rcs_preflight_enabled: bool = True
    # 점유 'select' 팝업(타 사용자 사용 중) 검출 — 떠 있으면 접속 포기 + cooldown 후 재시도.
    occupied_popup_detect_enabled: bool = True
    occupied_popup_vlm_service: str = "mai-ui"  # 제목 검출 후 옵션 확인용(route_slug).
    occupied_retry_cooldown_sec: float = 300.0    # 점유로 포기한 tool 재시도 유예(초).
    # 점유 외 사유로 사이클이 실패한 tool 의 재시도 유예(초). 없으면 매 poll 재시도해
    # 직렬화된 단일 RCS 커서를 독점하고 다른 알람을 굶긴다(F2).
    failure_retry_cooldown_sec: float = 300.0
    # --- 점유 tool 화면 공유 요청 (2026-08-18) ---
    # 점유 'select' 팝업에서 "화면 공유"를 골라 Request 를 눌러 관전 세션을 얻는다.
    # 안전은 env 게이트가 아니라 클릭 전 라벨 OCR 확인 게이트가 담당한다.
    share_request_enabled: bool = True     # Select 팝업에서 화면 공유 요청 발송.
    share_confirm_policy: str = "strict"   # strict | lenient | off - 클릭 전 라벨 확인.
    # 승낙 대기는 블로킹이고 단일 RCS 커서를 모든 알람이 직렬 공유하므로 짧게 둔다.
    # 10초 안에 못 받아도 손해가 크지 않다 - 알람이 유지되는 한 cooldown 후 다시
    # 요청하므로, 엔지니어는 다음 사이클에서 또 기회를 얻는다.
    share_wait_sec: float = 10.0
    share_max_attempts: int = 2            # EQP 별 연속 view-only 재시도 상한.
    # 점유 중에도 보정을 시도한다(opt-in). 화면 공유는 원래 view-only 라 클릭이 장비에
    # 안 먹는 게 기본 가정이지만, 엔지니어와 구두로 조율해 제어를 넘겨받은 상황에서는
    # 관전만 하고 끝나면 알람이 영영 안 풀린다. 켜도 결과는 corrected_unverified 로
    # 강등되어 cube 가 반드시 나간다 - 클릭이 먹었는지는 여전히 되읽지 못하기 때문이다.
    correct_when_occupied: bool = False
    # --- 점유 중 들어온 접근 요청 허용 (2026-08-20) ---
    # 우리가 tool 을 점유한 동안 다른 엔지니어가 접속하면 우리 화면에 허용/거부 팝업이
    # 뜨고, 응답하지 않으면 상대가 강제 종료로 우리 세션을 끊을 수 있다. 첫 실행에서는
    # 팝업 문구를 모르므로 기본은 관찰 전용(감지+토큰 로깅, 클릭 없음)이다.
    access_request_watch_enabled: bool = True   # 세션 중 접근 요청 팝업 감시.
    access_grant_enabled: bool = False          # 실제 허용 클릭(문구 확인 후 1 로).
    access_confirm_policy: str = "strict"       # strict | lenient | off.
    access_watch_poll_sec: float = 2.0          # 감시 주기 - 상대가 기다려 주는 시간이 짧다.
    keep_awake: bool = True
    # 자동 GUI 구간 동안 사용자 물리 마우스/키보드 입력 차단(Windows BlockInput).
    # 사용자가 다른 앱을 쓰면 foreground lock 으로 RCS 가 안 떠서 방해되는 문제 대응.
    # 기본 off(opt-in) + SAFE_MODE off 일 때만 실제 적용. engineer watch 구간은 제외
    # (엔지니어가 직접 조작해야 하므로). Ctrl+Alt+Del 로 항상 해제 가능.
    block_input_enabled: bool = False

    # --- 상시 녹화 (변화 감지 기반 적응 캡처) ---
    recording_poll_sec: float = DEFAULT_RECORDING_POLL_SEC  # 샘플링 간격.
    recording_heartbeat_sec: float = DEFAULT_RECORDING_HEARTBEAT_SEC  # 변화 없어도 이 간격마다 1장.
    recording_change_min_px: int = DEFAULT_RECORDING_CHANGE_MIN_PX  # 변화 판정 픽셀 최소 개수.
    recording_max_sec: float = 900.0

    # --- 접속 구간 prelude 녹화 (시연용, 기본 off) ---
    # 본 녹화는 tool 창 rect 를 찍으므로 '창이 뜨기 전' 인 RCS 실행/로그인/tool 진입
    # 구간은 원리상 프레임이 없다. prelude 는 그 구간만 **화면 전체**를 찍어
    # recording/prelude/ 에 따로 쌓는다(하위 폴더라 recording_filter 의 비재귀
    # glob 에는 안 걸린다 - 그 파이프라인은 tool 창 rect 를 전제한다).
    # 기본 off: 화면 전체 그랩은 다른 앱까지 담기고 프레임도 크므로, 상시 운전이
    # 아니라 시연 촬영 때만 켠다.
    record_prelude_enabled: bool = False
    prelude_poll_sec: float = 0.2  # 전체 화면은 프레임이 커서 본 녹화(0.05s)보다 성기게.
    prelude_max_sec: float = 300.0  # 접속만 5분 넘게 끌면 시연이 아니라 사고다.
    prelude_max_disk_mb: float = 800.0  # 백스톱 - 접속이 안 끝나도 디스크를 안 먹게.
    prelude_jpeg_quality: int = 85
    prelude_monitor_index: int = 1  # mss 규약(0=전 모니터 합침, 1=주 모니터).

    engineer_watch_sec: float = 300.0  # 미보정 watch 상한(cap, 5분) - done 감지 시 조기 종료.

    # --- engineer watch 측정-시작 감지 (Assist 우선, Recipe Monitor 카운터 fallback) ---
    engineer_done_detect_enabled: bool = True  # 기본 on (2026-08-19 사용자 결정).
    engineer_done_poll_sec: float = 8.0  # watch 안 detector 호출 간격.
    # 커서 정지 완료 판정(초). 감지 시작 이후 물리 마우스 커서가 이만큼 안 움직이면
    # 엔지니어가 손을 뗀 것으로 보고 완료 처리한다. 0 이하면 이 신호를 끈다.
    # 근거: 분자(N) 증가는 '측정이 시작됐다'는 강한 증거지만, 엔지니어가 align 만
    # 고치고 측정을 시작하지 않은 채 자리를 뜨는 경우엔 영영 안 뜬다 - 그때 tool 이
    # watch cap(5분)까지 잡혀 있다. 커서 정지는 그 공백을 메우는 약한 증거다.
    # 주의: '정지 = 완료' 는 추론이다. 화면만 보며 생각 중이거나 키보드만 쓰는
    # 중일 수도 있다. 사용자가 그 오판 비용(창을 닫으면 다시 열면 된다)을 받아들이고
    # 채택했다(2026-08-19). 되돌리려면 ALIGN_FAIL_ENGINEER_DONE_IDLE_SEC=0.
    engineer_done_idle_sec: float = 120.0
    # Assist 표에서 정상(검정)으로 끝난 측정 행이 이만큼 쌓이면 완료로 본다.
    # 이름/의미가 구 engineer_done_ok_streak(연속 정상 횟수)과 다르므로 env 이름도
    # 새로 뒀다 - 같은 이름을 재사용하면 오피스에 남은 기존 값이 새 의미로 조용히
    # 해석된다.
    engineer_done_min_ok_rows: int = 5
    # Assist 패널 판독 사용 여부. 기본 off (2026-08-19) - 판독이 아직 신뢰 수준에
    # 못 미쳐, 예전처럼 Recipe Monitor 분자(N) 단독으로 판정한다. off 면 Assist 를
    # 아예 읽지 않고(패널 VLM grounding 도 안 함) 분자 판정이 곧바로 primary 다 -
    # unusable streak 를 기다리지 않는다(그 대기는 'Assist 가 primary 인데 못 읽는
    # 중' 을 뜻하므로, Assist 를 안 쓰기로 한 상태에서는 의미가 없다).
    engineer_done_assist_enabled: bool = False
    engineer_done_assist_unusable_after: int = 3  # 분자 fallback 개방 전 unusable 횟수.
    engineer_done_numerator_increase_reads: int = 3  # 엄격 증가 분자 표본 요구 횟수.
    engineer_done_change_min_px: int = 4  # CV gate 변화 픽셀 임계(다운샘플).
    # 변화로 인정할 픽셀 delta 하한. 개수(위)와 하한(여기)은 한 판정의 두 반쪽인데
    # 예전에는 개수만 env 로 노출되고 하한은 recording.py 의 모듈 상수라, 오피스에서
    # 민감도를 조정할 방법이 반쪽뿐이었다(그리고 녹화 쪽을 튜닝하면 이쪽이 같이 움직였다).
    engineer_done_pixel_delta_min: float = 10.0
    engineer_done_relocalize_after_miss: int = 3  # 변화 후 OCR 연속 미검출 시 재grounding.
    # 재정렬 진행 중에는 카운터(N/M)가 빈칸이라 grounding 이 거부될 수 있다(정상).
    # 거부/실패 후 이 간격으로 재시도한다 (VLM 호출 폭주 방지 throttle).
    engineer_done_reground_sec: float = 30.0
    engineer_done_roi_pad_x: float = 0.03  # grounding 점 -> crop 확장 비율(가로, 창 대비).
    engineer_done_roi_pad_y: float = 0.02  # grounding 점 -> crop 확장 비율(세로, 창 대비).
    # 주의: Workflow1VLMClient 는 모델명이 아니라 flask_vlm 의 route_slug 를 받는다
    # ("mai-ui" O, "mai-ui-8b" X - workflow_2 스크립트들과 동일 규약).
    engineer_done_vlm_service: str = "mai-ui"  # grounding 서비스 slug.
    engineer_done_ocr_service: str = "paddleocr-vl-1.5"  # 분자 OCR 서비스 slug.

    def __post_init__(self) -> None:
        validate_engineer_done_priority_settings(self)

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
    # 긴급 해제 전역 단축키. 자동화가 쥔 마우스를 즉시 돌려받는 유일한 경로다.
    abort_hotkey: str = "<ctrl>+<alt>+q"
    sem_box_detect_enabled: bool = True
    sem_box_vlm_service: str = "mai-ui"  # route_slug (모델명 "mai-ui-8b" 아님).
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
    ok_button_vlm_service: str = "mai-ui"  # route_slug (모델명 "mai-ui-8b" 아님).
    # OK 버튼 자동 클릭. 기본 off = 반자동(reposition 까지만 자동, OK 는 엔지니어).
    # 좌표가 틀린 채 OK 가 눌리면 잘못된 위치로 측정이 확정되므로, 실전 신뢰가 쌓이기
    # 전까지는 사람이 마지막 확정을 쥔다. 켜려면 ALIGN_FAIL_OK_CLICK=1.
    ok_click_enabled: bool = False
    # paused 화면에서 key 를 못 찾았을 때 live_align_search(zoom-out + 사각 spiral pan)로
    # 넘길지. 기본 on(설계된 동작). off 면 pan 하지 않고 escalated_key_not_visible 로
    # 엔지니어에게 넘긴다 - 실장비에서 spiral 이 stage 를 최대 pan_budget(10) 회 끌고
    # 다니는데, 그 사이 커서를 되찾기 어려워 "보정을 못 한 것" 보다 비용이 큰 상황이
    # 있다. 롤백/재활성은 ALIGN_FAIL_FALLBACK_SEARCH=1.
    fallback_search_enabled: bool = True
    # fallback 이 켜져 있을 때 spiral pan 시도 상한. 낮출수록 stage 를 덜 끌고
    # 다니는 대신 못 찾고 escalate 할 확률이 는다.
    search_pan_budget: int = 5
    # PM 판독으로 OM/SEM 을 확정하지 못했을 때 보정을 보류할지(기본 on).
    # modality 를 틀리면 다른 template(IMAP0001 OM vs IMAP0002 SEM)로 매칭해 좌표가
    # 근본적으로 틀리므로, 추측해서 누르느니 엔지니어에게 넘긴다. off 면 sem_mode_default 사용.
    require_pm_mode: bool = True
    sem_mode_default: str = "SEM"
    sem_controller_settle_sec: float = 0.5
    zoom_scroll_dy: int = 1

    # --- 모호 키 재등록 알림 ---
    # second_ratio(2nd/best chamfer)가 이보다 크면 만성적으로 모호한 align key 로 보고
    # 엔지니어에게 재등록을 권고한다. tau*(S-LOO golden 보정, AUC 0.91) 유래의 시작점이며
    # fail-frame 재보정 대상 · matcher 의 0.94 visibility 게이트(max_second_ratio)와는 별개.
    reregister_second_ratio_threshold: float = 0.98

    # --- VLM 2단계 로케이터 조합 (로그인 / List 탭 / tool 선택 / PM 버튼 공통) ---
    # "coarse>fine" route_slug 조합. 빈 문자열 = 코드 기본값(현재 mai-ui>mai-ui,
    # vlm/ui_venus_mai_locator.py 의 DEFAULT_* 상수). 옛 조합 임시 복귀 예: "ui-venus>mai-ui".
    # env 이름은 ALIGN_FAIL_* 가 아니라 VLM_LOCATOR_COMBO 다
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
        detection_notify_enabled=env_flag("ALIGN_FAIL_DETECTION_NOTIFY", default=False),
        notify_delay_sec=env_float("ALIGN_FAIL_NOTIFY_DELAY_SEC", 90.0),
        cycle_enabled=env_flag("ALIGN_FAIL_RECORD_CYCLE", default=True),
        connect_action_enabled=env_flag("ALIGN_FAIL_CONNECT_ACTION", default=True),
        connect_window_timeout_sec=env_int("ALIGN_FAIL_CONNECT_WINDOW_TIMEOUT_SEC", 3),
        rcs_window_max_trials=env_int("ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS", 3),
        occupied_popup_detect_enabled=env_flag("ALIGN_FAIL_OCCUPIED_POPUP_DETECT", default=True),
        occupied_popup_vlm_service=_env_str("ALIGN_FAIL_OCCUPIED_POPUP_SERVICE", "mai-ui"),
        occupied_retry_cooldown_sec=env_float("ALIGN_FAIL_OCCUPIED_COOLDOWN_SEC", 300.0),
        failure_retry_cooldown_sec=env_float("ALIGN_FAIL_FAILURE_COOLDOWN_SEC", 300.0),
        share_request_enabled=env_flag("ALIGN_FAIL_SHARE_REQUEST", default=True),
        share_confirm_policy=_load_share_confirm_policy(),
        share_wait_sec=env_float("ALIGN_FAIL_SHARE_WAIT_SEC", 10.0),
        share_max_attempts=env_int("ALIGN_FAIL_SHARE_MAX_ATTEMPTS", 2),
        correct_when_occupied=env_flag("ALIGN_FAIL_CORRECT_WHEN_OCCUPIED", default=False),
        access_request_watch_enabled=env_flag("ALIGN_FAIL_ACCESS_WATCH", default=True),
        access_grant_enabled=env_flag("ALIGN_FAIL_ACCESS_GRANT", default=False),
        access_confirm_policy=os.getenv("ALIGN_FAIL_ACCESS_CONFIRM", "strict").strip().lower(),
        access_watch_poll_sec=env_float("ALIGN_FAIL_ACCESS_WATCH_POLL_SEC", 2.0),
        rcs_recovery_enabled=env_flag("ALIGN_FAIL_RCS_RECOVERY", default=True),
        rcs_kill_stale_enabled=env_flag("ALIGN_FAIL_RCS_KILL_STALE", default=False),
        rcs_recovery_window_timeout_sec=env_float(
            "ALIGN_FAIL_RCS_RECOVERY_WINDOW_SEC", 30.0
        ),
        rcs_preflight_enabled=env_flag("ALIGN_FAIL_RCS_PREFLIGHT", default=True),
        keep_awake=env_flag("ALIGN_FAIL_KEEP_AWAKE", default=True),
        block_input_enabled=env_flag("ALIGN_FAIL_BLOCK_INPUT", default=False),
        recording_poll_sec=env_float(
            "ALIGN_FAIL_RECORDING_POLL_SEC", DEFAULT_RECORDING_POLL_SEC
        ),
        recording_heartbeat_sec=env_float(
            "ALIGN_FAIL_RECORDING_HEARTBEAT_SEC", DEFAULT_RECORDING_HEARTBEAT_SEC
        ),
        recording_change_min_px=env_int(
            "ALIGN_FAIL_RECORDING_CHANGE_MIN_PX", DEFAULT_RECORDING_CHANGE_MIN_PX
        ),
        recording_max_sec=env_float("ALIGN_FAIL_RECORDING_MAX_SEC", 900.0),
        record_prelude_enabled=env_flag("ALIGN_FAIL_RECORD_PRELUDE", False),
        prelude_poll_sec=env_float("ALIGN_FAIL_PRELUDE_POLL_SEC", 0.2),
        prelude_max_sec=env_float("ALIGN_FAIL_PRELUDE_MAX_SEC", 300.0),
        prelude_max_disk_mb=env_float("ALIGN_FAIL_PRELUDE_MAX_DISK_MB", 800.0),
        prelude_jpeg_quality=env_int("ALIGN_FAIL_PRELUDE_JPEG_QUALITY", 85),
        prelude_monitor_index=env_int("ALIGN_FAIL_PRELUDE_MONITOR_INDEX", 1),
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
        abort_hotkey=_env_str("ALIGN_FAIL_ABORT_HOTKEY", "<ctrl>+<alt>+q"),
        sem_box_detect_enabled=env_flag("ALIGN_FAIL_SEM_BOX_DETECT", default=True),
        sem_box_vlm_service=_env_str("ALIGN_FAIL_SEM_BOX_SERVICE", "mai-ui"),
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
        ok_button_vlm_service=_env_str("ALIGN_OK_BUTTON_VLM_SERVICE", "mai-ui"),
        ok_click_enabled=env_flag("ALIGN_FAIL_OK_CLICK", default=False),
        fallback_search_enabled=env_flag("ALIGN_FAIL_FALLBACK_SEARCH", default=True),
        search_pan_budget=env_int("ALIGN_FAIL_SEARCH_PAN_BUDGET", 5),
        require_pm_mode=env_flag("ALIGN_FAIL_REQUIRE_PM_MODE", default=True),
        sem_mode_default=_env_str("ALIGN_SEM_MODE_DEFAULT", "SEM"),
        sem_controller_settle_sec=env_float("ALIGN_SEM_SETTLE_SEC", 0.5),
        zoom_scroll_dy=env_int("ALIGN_SEM_ZOOM_SCROLL_DY", 1),
        engineer_done_detect_enabled=env_flag("ALIGN_FAIL_ENGINEER_DONE_DETECT", default=True),
        engineer_done_poll_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_POLL_SEC", 8.0),
        engineer_done_idle_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_IDLE_SEC", 120.0),
        engineer_done_min_ok_rows=env_int("ALIGN_FAIL_ENGINEER_DONE_MIN_OK_ROWS", 5),
        engineer_done_assist_enabled=env_flag(
            "ALIGN_FAIL_ENGINEER_DONE_ASSIST", default=False
        ),
        engineer_done_assist_unusable_after=env_int(
            "ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER", 3
        ),
        engineer_done_numerator_increase_reads=env_int(
            "ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS", 3
        ),
        engineer_done_change_min_px=env_int("ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX", 4),
        engineer_done_pixel_delta_min=env_float(
            "ALIGN_FAIL_ENGINEER_DONE_PIXEL_DELTA_MIN", 10.0
        ),
        engineer_done_relocalize_after_miss=env_int("ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS", 3),
        engineer_done_reground_sec=env_float("ALIGN_FAIL_ENGINEER_DONE_REGROUND_SEC", 30.0),
        engineer_done_roi_pad_x=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_X", 0.03),
        engineer_done_roi_pad_y=env_float("ALIGN_FAIL_ENGINEER_DONE_ROI_PAD_Y", 0.02),
        engineer_done_vlm_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE", "mai-ui"),
        engineer_done_ocr_service=_env_str("ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE", "paddleocr-vl-1.5"),
        reregister_second_ratio_threshold=env_float("ALIGN_FAIL_REREGISTER_RATIO", 0.98),
        cond_box_crop=env_flag("ALIGN_FAIL_COND_BOX_CROP", default=True),
        locator_combo=_env_str("VLM_LOCATOR_COMBO", ""),
    )


__all__ = [
    "Workflow3Settings",
    "load_workflow3_settings",
    "validate_engineer_done_priority_settings",
]
