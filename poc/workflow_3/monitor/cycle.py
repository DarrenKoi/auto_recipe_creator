"""알람 1건 처리 사이클 — WorkflowRunner step 시퀀스 + 보장된 teardown.

align fail 알람 한 건에 대해 아래 step 을 순차 실행한다. 각 step 은
precondition/success_criteria 를 갖고, 실패 시 runner 가 중단하지만 녹화 중지·
tool 닫기·알림 발송은 step 이 아니라 `run_alarm_cycle` 의 후처리/finally 가
보장한다(러너가 중간에 죽어도 teardown 이 실행되게).

  1. ensure_rcs_ready   — RCS 메인 창 확보. 떠 있으면 전면화(activate)만, 없으면
                          (rcs_recovery_enabled 시) 재실행+재로그인 복구. ← 자동화 공백 (a)
  2. close_alert_popup  — 감지 팝업 닫기(screenshot 오염 방지)
  3. connect_tool       — List 탭에서 tool 더블클릭(VLM locator)
  4. wait_tool_window   — Remote Monitoring 창 대기 (RCS 점유 시 rcs_occupied)
  5. start_recording    — 상시 녹화 시작 (모든 align fail, 성공/실패 무관)
  6. locate_sem_panel   — landmark 로 SEM panel ROI → RCSSEMMonitor 생성 ← 공백 (b)
  7. run_correction     — correct_align_fail_auto (CV 가 좌표 결정, dry-run 게이트)

후처리(런 종료 후 항상):
  * notify_correction_outcome — status != corrected 면 cube 알림(outcome 요약)
  * engineer watch — 미보정 시 창이 닫히거나 watch 시간이 끝날 때까지 대기
    (녹화 스레드가 엔지니어 수동 조작을 계속 캡처)
  * finally: 녹화 중지(manifest) → tool 닫기 → 알림 팝업 backstop
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR, DEBUG_IMAGE_DIR
from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.notify import close_alert_window, notify_correction_outcome
from poc.workflow_3.monitor.recording import RecordingSession
from poc.workflow_3.sem_monitor.controller import build_rcs_sem_monitor
from poc.workflow_3.runner.workflow_runner import WorkflowRunner
from poc.workflow_3.runner.workflow_types import (
    ConditionGroup,
    ConditionType,
    StepCondition,
    StepResult,
    WorkflowStep,
)
from poc.workflow_3.util import (
    activate_window,
    block_input,
    capture_window,
    click_at_screen,
    env_float,
    image_point_to_screen,
    make_timestamp_tag,
    move_cursor_to_screen,
    scroll_at_screen,
)

LOG_COMPONENT = "align_fail_cycle"

# zoom-out(wheel-down) 보정탐색을 발동할 feasibility verdict — "어느 점이 align point 인지"
# 가릴 수 없는 두 경우만(ambiguous=모호, not_visible=프레임에 키 부재). possible(확신)·
# no_assets(rcp 자산 없음)는 제외. PM_OM_VALUES 처럼 settings 밖 모듈 상수로 둔다.
_ZOOM_PROBE_VERDICTS = frozenset({"ambiguous", "not_visible"})


def _should_block_input(settings: Workflow3Settings) -> bool:
    """자동 GUI 구간 입력 차단을 적용할지 — opt-in + SAFE_MODE off + Windows 한정."""
    return (
        getattr(settings, "block_input_enabled", False)
        and not settings.safe_mode
        and block_input is not None
    )

# 점검 전용(check-only) 캡처 직전 SEM 영상 렌더 대기(초) — rcs_screenshot 의 settle 과 동일 취지.
_CHECK_CAPTURE_SETTLE_SEC = env_float("ALIGN_FAIL_CHECK_SETTLE_SEC", 2.0)

# align point 커서 이동 후 안착 화면 재캡처 전 대기(초) — 커서가 멈춘 뒤 한 장.
_PREVIEW_SETTLE_SEC = env_float("ALIGN_FAIL_REPOSITION_PREVIEW_SETTLE_SEC", 0.4)

# zoom ladder: 커서를 SEM box 로 옮긴 뒤 wheel 전 짧은 hover 안착(초) — 일부 Windows
# 컨트롤은 커서가 막 텔레포트한 직후 wheel 을 무시하므로 hover 를 잠깐 인식시킨다.
_ZOOM_AIM_SETTLE_SEC = env_float("ALIGN_FAIL_ZOOM_AIM_SETTLE_SEC", 0.15)
# PM 드롭다운 fallback: 'PM' 버튼 클릭 후 드롭다운이 그려질 때까지 대기(초).
_PM_DROPDOWN_OPEN_SETTLE_SEC = env_float("ALIGN_FAIL_PM_DROPDOWN_OPEN_SETTLE_SEC", 0.5)

# RCS GUI 의존 모듈(pywinauto/VLM)은 선택 의존성 — 없는 환경(개발 PC)에서도
# 본체 import 는 살아 있어야 한다(기존 align_fail_alarm_record 의 패턴).
try:
    from poc.workflow_3.rcs.login_rcs_common import (
        wait_for_rcs_main_window,
        wait_for_remote_monitoring_window,
    )
    from poc.workflow_3.rcs.rcs_screenshot import captured_dir_for
    from poc.workflow_3.rcs.workflow_select_tool import connect_to_tool

    RCS_MODULES_AVAILABLE = True
except Exception as _rcs_import_exc:
    wait_for_rcs_main_window = None
    wait_for_remote_monitoring_window = None
    captured_dir_for = None
    connect_to_tool = None
    RCS_MODULES_AVAILABLE = False
    print(f"[WARNING] RCS 모듈 로드 실패 - 사이클 비활성(감지/로그만 동작): {_rcs_import_exc}")

try:
    from poc.workflow_3.rcs.workflow_close_tool import close_tool

    CLOSE_TOOL_AVAILABLE = True
except Exception as _close_tool_import_exc:
    close_tool = None
    CLOSE_TOOL_AVAILABLE = False
    print(f"[WARNING] workflow_close_tool 로드 실패 - tool 창 닫기 비활성: {_close_tool_import_exc}")


@dataclass
class CycleResult:
    """알람 1건 사이클의 결과 요약 (manifest 기록용)."""

    eqp_id: str
    recipe_id: str
    tag: str
    run_status: str = "not_run"
    run_dir: str = ""
    outcome_status: str = ""
    outcome_path: str = ""
    key_decision: str = ""
    best_xy: str = ""
    recording_dir: str = ""
    frame_count: int = 0
    failed_step: str = ""
    failure_class: str = ""
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# step 정의.
# ------------------------------------------------------------------


def _ctx_set(target_key: str) -> ConditionGroup:
    """context[target_key] 존재를 요구하는 success_criteria 그룹."""
    return ConditionGroup(
        conditions=[
            StepCondition(ConditionType.CONTEXT_KEY_SET, target_key=target_key)
        ]
    )


def build_cycle_steps(eqp_id: str) -> list[WorkflowStep]:
    """알람 1건 사이클의 step 목록을 만든다."""
    return [
        WorkflowStep(
            step_id="ensure_rcs_ready",
            step_type="recover",
            target_description="RCS 메인 창 확보(전면화/재실행+재로그인)",
            success_criteria=_ctx_set("rcs_main_window"),
        ),
        WorkflowStep(
            step_id="close_alert_popup",
            step_type="cleanup",
            target_description="감지 알림 팝업 닫기(screenshot 오염 방지)",
        ),
        WorkflowStep(
            step_id="connect_tool",
            step_type="action",
            target_description=f"List 탭에서 tool 더블클릭: {eqp_id}",
            depends_on=["ensure_rcs_ready"],
        ),
        WorkflowStep(
            step_id="wait_tool_window",
            step_type="detect",
            target_description="Remote Monitoring 창 대기",
            depends_on=["connect_tool"],
            success_criteria=_ctx_set("tool_window"),
        ),
        WorkflowStep(
            step_id="start_recording",
            step_type="action",
            target_description="상시 녹화 시작(모든 align fail)",
            depends_on=["wait_tool_window"],
        ),
        WorkflowStep(
            step_id="locate_sem_panel",
            step_type="detect",
            target_description="SEM panel ROI 확보 → 실장비 controller 생성",
            depends_on=["wait_tool_window"],
            success_criteria=_ctx_set("controller"),
        ),
        WorkflowStep(
            step_id="run_correction",
            step_type="action",
            target_description="CV align fail 보정(correct_align_fail_auto)",
            depends_on=["locate_sem_panel"],
        ),
    ]


# ------------------------------------------------------------------
# executor.
# ------------------------------------------------------------------


def _make_result(
    step: WorkflowStep,
    status: str,
    started_at: float,
    settings: Workflow3Settings,
    *,
    failure_class: str | None = None,
    error_message: str | None = None,
) -> StepResult:
    """executor 용 StepResult 빌더."""
    return StepResult(
        step_id=step.step_id,
        status=status,
        failure_class=failure_class,
        attempt_count=1,
        strategy_used="align_fail_cycle",
        vlm_service_used="",
        detected_point=None,
        detected_bbox=None,
        screen_point=None,
        verification_result=None,
        before_screenshot=None,
        after_screenshot=None,
        error_message=error_message,
        elapsed_ms=(time.time() - started_at) * 1000,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        safe_mode=settings.safe_mode,
    )


def _exec_ensure_rcs_ready(step, context, settings: Workflow3Settings) -> StepResult:
    """① RCS 메인 창 확보 — 떠 있으면 전면화, 없으면(옵션) 재실행+재로그인."""
    started_at = time.time()
    window, title, backend = wait_for_rcs_main_window(timeout_sec=settings.connect_window_timeout_sec)
    if window is None and settings.rcs_recovery_enabled:
        print("[WARNING] RCS 메인 창 없음 - 재실행+재로그인 복구 시도(ALIGN_FAIL_RCS_RECOVERY=on)")
        try:
            from poc.workflow_3.rcs.open_rcs import RCS_EXE, launch_rcs
            from poc.workflow_3.rcs.workflow_login import run_login_workflow

            launch_rcs(RCS_EXE)
            run_login_workflow(settings)
            window, title, backend = wait_for_rcs_main_window(timeout_sec=30.0)
        except Exception as exc:
            return _make_result(
                step, "failed", started_at, settings,
                failure_class="rcs_recovery_error", error_message=f"{type(exc).__name__}: {exc}",
            )
    if window is None:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="rcs_unavailable",
            error_message="RCS 메인 창 없음 (복구 비활성 또는 복구 실패) - 엔지니어 직접 처리",
        )

    # 프로세스는 살아 있지만 창이 뒤/최소화에 있을 수 있음 — 전면화. (자동화 공백 (a))
    if activate_window is not None:
        activate_window(window, debug_label=f"rcs_main title={title!r}")
    context["rcs_main_window"] = window
    context["rcs_main_title"] = title
    context["rcs_main_backend"] = backend
    return _make_result(step, "success", started_at, settings)


def _exec_close_alert_popup(step, context, settings: Workflow3Settings) -> StepResult:
    """② 감지 팝업 닫기 — SYSTEMMODAL 팝업이 screenshot 에 겹치지 않게."""
    started_at = time.time()
    close_alert_window(timeout_sec=settings.alert_close_timeout_sec)
    return _make_result(step, "success", started_at, settings)


def _exec_connect_tool(step, context, settings: Workflow3Settings) -> StepResult:
    """③ tool 더블클릭 접속 — 알람당 1회만 느슨하게 시도(실패 시 엔지니어 직접)."""
    started_at = time.time()
    eqp_id = context["eqp_id"]
    action_enabled = settings.action_enabled and settings.connect_action_enabled
    try:
        result = connect_to_tool(
            eqp_id,
            action_enabled=action_enabled,
            main_window_timeout_sec=settings.connect_window_timeout_sec,
        )
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="connect_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    double_clicked = bool(getattr(result, "double_clicked", False))
    context["connect_result"] = result
    if not double_clicked:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="connect_not_clicked",
            error_message=f"tool 더블클릭 미수행(dry-run/인식 실패): action_enabled={action_enabled}",
        )
    return _make_result(step, "success", started_at, settings)


def _exec_wait_tool_window(step, context, settings: Workflow3Settings) -> StepResult:
    """④ tool 창 대기 — RCS 점유(select 팝업) 시 건드리지 않고 포기.

    점유 검출(opt-in): 매 시도 전 detect_select_popup 으로 'select' 팝업을 조기 감지해,
    떠 있으면 창 탐색을 더 돌지 않고 즉시 포기한다(세 옵션은 클릭하지 않음). 검출은 제목
    열거 + (가능하면) VLM 확인의 hybrid 이며, 실패해도 접속을 막지 않는다(전체 흐름은
    기존 '미발견 → rcs_occupied' 로 폴백). 점유 조기 감지 시 failure_class 를 구분해
    상위 루프가 cooldown 후 재시도하도록 한다.
    """
    started_at = time.time()
    eqp_id = context["eqp_id"]

    occupied = {"select": False}
    abort_check = None
    if settings.occupied_popup_detect_enabled:
        popup_client = None
        try:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            # 짧은 timeout: 점유 확인 VLM 이 접속 흐름을 길게 막지 않게 한다.
            popup_client = Workflow1VLMClient(
                settings.occupied_popup_vlm_service, timeout_sec=15.0
            )
        except Exception as exc:
            print(f"[WARNING] 점유 팝업 VLM client 생성 실패(제목만으로 검출): {exc}")

        def abort_check() -> bool:
            from poc.workflow_3.monitor.occupied_popup import detect_select_popup

            if detect_select_popup(popup_client):
                occupied["select"] = True
                return True
            return False

    window, title, backend = wait_for_remote_monitoring_window(
        eqp_id, max_attempts=settings.rcs_window_max_trials, abort_check=abort_check
    )
    if occupied["select"]:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="rcs_occupied_select",
            error_message=(
                "점유 'select'(공유/종료) 팝업 감지 - 접속 포기, 옵션 미선택(사람 판단 "
                "영역). 다른 사용자 해제 후 cooldown 지나면 재시도."
            ),
        )
    if window is None:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="rcs_occupied",
            error_message=(
                f"tool 창 미발견({settings.rcs_window_max_trials}회) - RCS 점유(select "
                f"공유/종료 팝업) 가능성. select 팝업은 사람 판단 영역이라 건드리지 않음."
            ),
        )
    context["tool_window"] = window
    context["tool_window_title"] = title
    context["tool_window_backend"] = backend
    return _make_result(step, "success", started_at, settings)


def _recording_dir_for(eqp_id: str, recipe_id: str, tag: str) -> Path:
    """녹화 저장 폴더 — captured_img_from_rcs/<tag>/recording (recipe 없으면 _unregistered)."""
    if recipe_id and captured_dir_for is not None:
        return captured_dir_for(eqp_id, recipe_id) / tag / "recording"
    return ALIGN_IMAGES_DIR / eqp_id / "_unregistered" / tag / "recording"


def _exec_start_recording(step, context, settings: Workflow3Settings) -> StepResult:
    """⑤ 상시 녹화 시작 — 실패해도 사이클은 계속(녹화는 best-effort)."""
    started_at = time.time()
    out_dir = _recording_dir_for(context["eqp_id"], context["recipe_id"], context["tag"])
    try:
        session = RecordingSession(
            context["tool_window"],
            out_dir,
            tag=context["tag"],
            poll_sec=settings.recording_poll_sec,
            heartbeat_sec=settings.recording_heartbeat_sec,
            change_min_px=settings.recording_change_min_px,
            max_sec=settings.recording_max_sec,
        ).start()
        context["recording"] = session
    except Exception as exc:
        print(f"[WARNING] 녹화 시작 실패(사이클은 계속): {exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="recording_start_failed", level="warning",
            error=str(exc),
        )
    return _make_result(step, "success", started_at, settings)


def _exec_locate_sem_panel(step, context, settings: Workflow3Settings) -> StepResult:
    """⑥ SEM panel ROI → RCSSEMMonitor — landmark 미캘리브레이션이면 panel_not_found."""
    started_at = time.time()
    controller_action = settings.action_enabled and not settings.correction_dry_run
    try:
        controller = build_rcs_sem_monitor(
            context["tool_window"],
            action_enabled=controller_action,
            settle_sec=settings.sem_controller_settle_sec,
            zoom_scroll_dy=settings.zoom_scroll_dy,
            mode_default=settings.sem_mode_default,
        )
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="panel_locate_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    if controller is None:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="panel_not_found",
            error_message="SEM panel landmark 미캘리브레이션/신뢰도 부족 - 보정 생략",
        )
    context["controller"] = controller
    return _make_result(step, "success", started_at, settings)


def _exec_run_correction(step, context, settings: Workflow3Settings) -> StepResult:
    """⑦ CV 보정 — RECIPE_ID 가 있고 correction_enabled 일 때만."""
    started_at = time.time()
    eqp_id, recipe_id = context["eqp_id"], context["recipe_id"]
    if not settings.correction_enabled:
        return _make_result(step, "skipped", started_at, settings)
    if not recipe_id:
        print(f"[INFO] RECIPE_ID 없음 - 보정 생략, 엔지니어 직접 처리 (EQP_ID={eqp_id})")
        return _make_result(step, "skipped", started_at, settings)

    from poc.workflow_3.align.correction import CorrectionConfig, correct_align_fail_auto

    vlm_client = None
    try:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        vlm_client = Workflow1VLMClient(settings.ok_button_vlm_service)
    except Exception as exc:
        print(f"[WARNING] OK 버튼 VLM 클라이언트 생성 실패 - OK 탐지 없이 진행: {exc}")
    context["vlm_client"] = vlm_client

    def _escalation_log(state, history):
        log_work2_event(
            component=LOG_COMPONENT, message="live_search_escalation", level="warning",
            eqp_id=eqp_id, low_streak=state.low_streak, pan_count=state.pan_count,
        )

    debug_dir = DEBUG_IMAGE_DIR / "align_fail_cycle" / context["tag"]
    try:
        outcome = correct_align_fail_auto(
            context["controller"],
            vlm_client=vlm_client,
            notify_fn=_escalation_log,
            config=CorrectionConfig(
                # 만성 모호 키 게이트(Tier 0.1) 활성화 — present 하나 second_ratio>tau 면
                # 자동 reposition+OK 대신 engineer_review 로 보류한다. notify 임계와 동일 값.
                reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
                # cond box-crop template(Tier 1.1; env ALIGN_FAIL_COND_BOX_CROP 로 롤백 가능).
                cond_box_crop=settings.cond_box_crop,
                # consensus 라우팅 설정(Workflow3Settings 에서 주입).
                consensus_enabled=settings.consensus_enabled,
                consensus_min_s=settings.consensus_min_s,
                consensus_max_events=settings.gather_max_events,
                consensus_sync_timeout_sec=settings.consensus_sync_timeout_sec,
            ),
            dry_run=settings.correction_dry_run,
            debug_dir=debug_dir,
            eqp_id=eqp_id,
            recipe_name=recipe_id,
        )
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="correction_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    context["outcome"] = outcome
    print(f"[INFO] 보정 결과: status={outcome.status} path={outcome.path} decision={outcome.key_decision}")
    return _make_result(step, "success", started_at, settings)


_STEP_EXECUTORS = {
    "ensure_rcs_ready": _exec_ensure_rcs_ready,
    "close_alert_popup": _exec_close_alert_popup,
    "connect_tool": _exec_connect_tool,
    "wait_tool_window": _exec_wait_tool_window,
    "start_recording": _exec_start_recording,
    "locate_sem_panel": _exec_locate_sem_panel,
    "run_correction": _exec_run_correction,
}


# ------------------------------------------------------------------
# 사이클 본체.
# ------------------------------------------------------------------


def _engineer_watch(
    recording: RecordingSession,
    watch_sec: float,
    *,
    done_detector=None,
    poll_sec: float = 8.0,
) -> None:
    """미보정 시 엔지니어 수동 조작 구간 대기 — 녹화 스레드가 계속 캡처한다.

    종료 조건(첫 충족 시): ① 녹화 스레드 자체 종료(창 닫힘=window_gone/max_sec)
    ② done_detector() True (측정 시작 = align 완료, engineer_done_align_adjustment 모듈)
    ③ watch_sec 경과 (이제 backstop cap). detector 예외는 ②만 무력화한다.
    """
    if watch_sec <= 0:
        return
    print(
        f"[INFO] engineer watch 시작: 최대 {watch_sec:.0f}s "
        f"(창 닫힘/측정시작 감지/녹화 종료 시 조기 종료, "
        f"감지={'on' if done_detector is not None else 'off'})"
    )
    deadline = time.time() + watch_sec
    next_check = 0.0
    while time.time() < deadline and recording.is_alive():
        if done_detector is not None and time.time() >= next_check:
            try:
                if done_detector():
                    print("[INFO] 측정 시작 감지(align 완료 추정) - engineer watch 조기 종료")
                    break
            except Exception as exc:
                print(f"[WARNING] done detector 예외(무시, cap 으로 진행): {exc}")
            next_check = time.time() + max(poll_sec, 0.0)
        time.sleep(2.0)
    print("[INFO] engineer watch 종료")


def run_alarm_cycle(
    eqp_id: str,
    recipe_id: str,
    settings: Workflow3Settings,
    *,
    tag: str | None = None,
) -> CycleResult:
    """알람 1건에 대한 전체 사이클을 실행하고 결과 요약을 반환한다.

    step 실패로 runner 가 중단돼도 cube 알림·녹화 중지·tool 닫기·팝업 backstop 은
    항상 실행된다. 예외는 삼켜 상위 폴링 루프가 죽지 않게 한다.
    """
    tag = tag or make_timestamp_tag()
    result = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag)

    if not RCS_MODULES_AVAILABLE:
        result.run_status = "rcs_unavailable"
        result.notes.append("RCS 모듈 비활성 — 감지/로그만")
        notify_correction_outcome(
            eqp_id, recipe_id, None, enabled=settings.rich_notify_enabled
        )
        return result

    context: dict = {"eqp_id": eqp_id, "recipe_id": recipe_id, "tag": tag}
    runner = WorkflowRunner(
        settings,
        workflow_name=f"align_fail_cycle_{eqp_id}",
        log_name="work2",
        component_name=LOG_COMPONENT,
    )

    def executor(step, step_context):
        return _STEP_EXECUTORS[step.step_id](step, step_context, settings)

    recording: RecordingSession | None = None
    input_blocked = False
    try:
        # 자동 GUI 구간 동안 사용자 물리 입력 차단(opt-in) — foreground lock/클릭 방해 방지.
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"align_fail_cycle {eqp_id}")
        run = runner.run(build_cycle_steps(eqp_id), context, executor)
        result.run_status = run.status
        result.run_dir = str(run.run_dir or "")
        for step_result in run.step_results:
            if step_result.status == "failed":
                result.failed_step = step_result.step_id
                result.failure_class = step_result.failure_class or ""
                break

        recording = context.get("recording")
        outcome = context.get("outcome")
        if outcome is not None:
            result.outcome_status = outcome.status
            result.outcome_path = outcome.path
            result.key_decision = outcome.key_decision
            if outcome.best_xy is not None:
                result.best_xy = f"({outcome.best_xy[0]},{outcome.best_xy[1]})"

        # 처리 결과 알림 — corrected 외 전부(보정 미수행 포함) cube 발송.
        recording_dir = str(recording.out_dir) if recording is not None else ""
        notify_correction_outcome(
            eqp_id, recipe_id, outcome,
            recording_dir=recording_dir, enabled=settings.rich_notify_enabled,
            reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
        )

        # 자동 GUI 구간 종료 — 엔지니어 수동 조작 전에 입력 차단 해제(엔지니어가 직접 조작해야 함).
        if input_blocked:
            block_input(False, debug_label=f"align_fail_cycle {eqp_id}")
            input_blocked = False

        # 미보정이면 엔지니어 수동 조작을 녹화하며 대기 (측정 시작 감지 시 조기 종료).
        if recording is not None and (outcome is None or outcome.status != "corrected"):
            done_detector = None
            if settings.engineer_done_detect_enabled and context.get("tool_window") is not None:
                try:
                    from poc.workflow_3.monitor.engineer_done_align_adjustment import (
                        build_engineer_done_detector,
                    )

                    done_detector = build_engineer_done_detector(
                        context["tool_window"], settings,
                        vlm_client=context.get("vlm_client"),
                        # tool 별/알람 별로 debug crop 이 안 섞이게 폴더를 분리한다.
                        debug_dir=DEBUG_IMAGE_DIR / "engineer_done" / f"{eqp_id}_{tag}",
                    )
                except Exception as exc:
                    print(f"[WARNING] done detector 생성 실패(고정 timeout 으로 진행): {exc}")
            _engineer_watch(
                recording, settings.engineer_watch_sec,
                done_detector=done_detector,
                poll_sec=settings.engineer_done_poll_sec,
            )
    except Exception as exc:
        result.run_status = "error"
        result.notes.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] 사이클 예외: EQP_ID={eqp_id}, error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="cycle_error", level="error",
            eqp_id=eqp_id, error=str(exc),
        )
    finally:
        # 입력 차단 backstop — 위에서 예외로 못 풀었어도 반드시 해제(사용자 잠김 방지).
        if input_blocked:
            block_input(False, debug_label=f"align_fail_cycle {eqp_id}")
            input_blocked = False
        # teardown 보장 — 녹화 중지(manifest) → tool 닫기 → 알림 팝업 backstop.
        recording = recording or context.get("recording")
        if recording is not None:
            frames = recording.stop("cycle_teardown")
            result.recording_dir = str(recording.out_dir)
            result.frame_count = len(frames)
        # tool 창 닫기 — 보정 성공/실패, 그리고 engineer_done_align_adjustment 가
        # 측정 시작(N>5 연속 2회)을 감지해 watch 가 조기 종료된 경우 모두 여기서 닫힌다.
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            try:
                close_tool(eqp_id)
            except Exception as exc:
                print(f"[WARNING] tool 창 닫기 실패: {exc}")
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return result


# ------------------------------------------------------------------
# 점검 전용(check-only) 사이클 — 접속 → 첫 화면 1장 캡처 → 닫기.
# ------------------------------------------------------------------


def _capture_dir_for(eqp_id: str, recipe_id: str, tag: str) -> Path:
    """첫 화면 캡처 저장 폴더 — captured_img_from_rcs/<tag> (recipe 없으면 _unregistered)."""
    if recipe_id and captured_dir_for is not None:
        return captured_dir_for(eqp_id, recipe_id) / tag
    return ALIGN_IMAGES_DIR / eqp_id / "_unregistered" / tag


def _exec_capture_screen(step, context, settings: Workflow3Settings) -> StepResult:
    """첫 화면 1장 캡처 — 장비는 fail 시 정지라 단일 스크린샷으로 충분."""
    started_at = time.time()
    out_dir = _capture_dir_for(context["eqp_id"], context["recipe_id"], context["tag"])
    try:
        if _CHECK_CAPTURE_SETTLE_SEC > 0:
            time.sleep(_CHECK_CAPTURE_SETTLE_SEC)
        out_dir.mkdir(parents=True, exist_ok=True)
        image = capture_window(context["tool_window"])
        out_path = out_dir / f"{context['tag']}_rcs.jpg"
        save_debug_jpeg(image, out_path)
        context["capture_path"] = out_path
        print(f"[INFO] 첫 화면 캡처 저장: {out_path}")
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="capture_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    return _make_result(step, "success", started_at, settings)


def build_check_steps(eqp_id: str) -> list[WorkflowStep]:
    """점검 전용 사이클 step — 보정/녹화 없이 접속 → 캡처까지만."""
    return [
        WorkflowStep(
            step_id="ensure_rcs_ready",
            step_type="recover",
            target_description="RCS 메인 창 확보(전면화/재실행+재로그인)",
            success_criteria=_ctx_set("rcs_main_window"),
        ),
        WorkflowStep(
            step_id="close_alert_popup",
            step_type="cleanup",
            target_description="감지 알림 팝업 닫기(screenshot 오염 방지)",
        ),
        WorkflowStep(
            step_id="connect_tool",
            step_type="action",
            target_description=f"List 탭에서 tool 더블클릭: {eqp_id}",
            depends_on=["ensure_rcs_ready"],
        ),
        WorkflowStep(
            step_id="wait_tool_window",
            step_type="detect",
            target_description="Remote Monitoring 창 대기",
            depends_on=["connect_tool"],
            success_criteria=_ctx_set("tool_window"),
        ),
        WorkflowStep(
            step_id="capture_screen",
            step_type="action",
            target_description="첫 화면 1장 캡처",
            depends_on=["wait_tool_window"],
            success_criteria=_ctx_set("capture_path"),
        ),
    ]


_CHECK_STEP_EXECUTORS = {
    "ensure_rcs_ready": _exec_ensure_rcs_ready,
    "close_alert_popup": _exec_close_alert_popup,
    "connect_tool": _exec_connect_tool,
    "wait_tool_window": _exec_wait_tool_window,
    "capture_screen": _exec_capture_screen,
}


def _save_cursor_preview(tool_window, align_xy, frame_wh, capture_dir: Path, tag: str):
    """커서가 안착한 live 화면을 재캡처해 align point 에 합성 마커를 그려 저장한다.

    mss 캡처는 OS 하드웨어 커서를 담지 않으므로, 커서가 이동한 좌표에 마커(십자선+원+
    'cursor' 라벨, feasibility 노랑 십자선과 구분되는 마젠타)를 그려 어디로 갔는지 자명히
    남긴다. 재캡처 크기가 매칭 프레임(frame_wh)과 다르면 좌표를 비례 보정한다.
    저장 경로(Path) 반환.
    """
    from PIL import ImageDraw

    image = capture_window(tool_window)
    iw, ih = image.size
    ax, ay = int(align_xy[0]), int(align_xy[1])
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    if (fw, fh) != (iw, ih) and fw > 0 and fh > 0:
        ax = int(round(ax * iw / fw))
        ay = int(round(ay * ih / fh))

    draw = ImageDraw.Draw(image)
    color = (255, 0, 255)
    r = 22
    draw.line([(ax - r, ay), (ax + r, ay)], fill=color, width=3)
    draw.line([(ax, ay - r), (ax, ay + r)], fill=color, width=3)
    draw.ellipse([(ax - r, ay - r), (ax + r, ay + r)], outline=color, width=3)
    draw.text((ax + r + 6, ay - 8), "cursor", fill=color)

    out_path = capture_dir / f"{tag}_rcs_cursor.jpg"
    save_debug_jpeg(image, out_path)
    return out_path


def _check_feasibility_and_preview(
    context: dict,
    result: "CycleResult",
    settings: Workflow3Settings,
    eqp_id: str,
    recipe_id: str,
) -> None:
    """캡처 직후(tool 창이 아직 열린 상태) 보정 가능성 판정 + (opt-in) align point 커서 미리보기.

    1) mark_align_feasibility 로 디스크 저장본을 판정해 _marked.jpg/_feasibility.json 을
       남기고 manifest 필드를 채운다(verdict/decision/align_xy/marked_path).
    2) reposition_preview_enabled 면 같은 align_xy 를 image_point_to_screen 으로 screen
       좌표로 변환해 커서만 옮긴 뒤(클릭 없음 = 장비 부작용 없음) 안착 화면을 재캡처한다.
       align_xy 는 캡처 프레임 좌표라 이미 live 좌표계 — DPI(rect/screenshot) 보정만 남는다.

    teardown(close)을 막지 않도록 모든 단계의 예외를 삼킨다. tool 창이 닫히기 전에
    호출돼야 한다(image→screen 변환·재캡처가 살아 있는 창을 필요로 함).
    """
    capture_path = Path(context["capture_path"])
    # live SEM box 검출용 VLM client(opt-in). 빌드 실패해도 feasibility 는 전체 창
    # 매칭으로 폴백하므로 None 으로 두고 진행한다(개발 PC/Flask 부재 안전).
    sem_box_client = None
    ocr_client = None
    if settings.sem_box_detect_enabled:
        try:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            # 짧은 timeout: 이 호출은 tool 창이 열린 채(teardown 전) 도므로, 느린/행
            # 걸린 VLM 이 close 를 최대 120s(기본) 막지 않도록 15s 로 묶는다. 초과하면
            # 검출 실패 → 전체 창 매칭 폴백(루프 진행 보장).
            sem_box_client = Workflow1VLMClient(
                settings.sem_box_vlm_service, timeout_sec=15.0
            )
            # PM 2단계 OCR(opt-in)일 때만 OCR client 추가 — crop 재독용(동일 짧은 timeout).
            if settings.pm_two_stage_ocr_enabled:
                ocr_client = Workflow1VLMClient(
                    settings.pm_ocr_service, timeout_sec=15.0
                )
        except Exception as exc:
            print(f"[WARNING] SEM box VLM client 생성 실패(전체 창 매칭 폴백): {exc}")
    try:
        from poc.workflow_3.align.diagnostics.feasibility_check import mark_align_feasibility

        feas = mark_align_feasibility(
            capture_path,
            eqp_id=eqp_id,
            recipe_id=recipe_id,
            cond_box_crop=settings.cond_box_crop,
            reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
            vlm_client=sem_box_client,
            ocr_client=ocr_client,
            pm_two_stage=settings.pm_two_stage_ocr_enabled,
        )
    except Exception as exc:
        print(f"[WARNING] feasibility 분석 실패(캡처는 유지): {exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="feasibility_error", level="warning",
            eqp_id=eqp_id, error=str(exc),
        )
        return

    # verdict 를 manifest 컬럼에 매핑(별도 컬럼 없이 기존 필드 재사용).
    result.outcome_status = feas.verdict
    result.key_decision = feas.decision
    if feas.align_xy is not None:
        result.best_xy = f"({feas.align_xy[0]},{feas.align_xy[1]})"
    if feas.marked_path is not None:
        result.outcome_path = str(feas.marked_path)

    # ---- (opt-in) align point 로 커서 이동 + 안착 화면 재캡처. ----
    if settings.reposition_preview_enabled:
        _run_reposition_preview(context, result, feas, settings)

    # ---- (opt-in) 모호/부재 verdict 면 zoom in/out ladder 보정탐색. ----
    # preview 와 독립 — feasibility 만 계산되면 verdict 로 발동을 판정한다.
    _run_zoom_ladder(context, result, feas, settings, sem_box_client, ocr_client)


def _run_reposition_preview(
    context: dict,
    result: "CycleResult",
    feas,
    settings: Workflow3Settings,
) -> None:
    """align point 로 커서만 옮긴 뒤(클릭 없음) 안착 화면을 재캡처해 좌표 매핑을 눈으로 검증한다.

    `_check_feasibility_and_preview` 에서 reposition_preview_enabled 일 때만 호출된다.
    tool 창이 닫히기 전에 와야 한다(image→screen 변환·재캡처가 살아 있는 창 필요).
    """
    tool_window = context.get("tool_window")
    if feas.align_xy is None or feas.frame_wh is None or tool_window is None:
        print(
            f"[INFO] align preview 생략: align_xy/frame_wh/tool_window 없음 "
            f"(verdict={feas.verdict})"
        )
        return
    if image_point_to_screen is None or move_cursor_to_screen is None:
        print("[INFO] align preview 생략: window/mouse util 비활성(개발 PC)")
        return

    ax, ay = int(feas.align_xy[0]), int(feas.align_xy[1])
    screen = image_point_to_screen(tool_window, {"x": ax, "y": ay}, image_size=feas.frame_wh)
    if screen is None:
        print("[WARNING] align preview: image→screen 변환 실패 - 이동 생략")
        return
    print(
        f"[INFO] align preview: verdict={feas.verdict} frame={feas.frame_wh} "
        f"align=({ax},{ay}) → screen=({screen['x']},{screen['y']})"
        f"{'' if settings.action_enabled else ' [dry-run]'}"
    )
    move_cursor_to_screen(screen, "align_preview", action_enabled=settings.action_enabled)
    if _PREVIEW_SETTLE_SEC > 0:
        time.sleep(_PREVIEW_SETTLE_SEC)
    try:
        cursor_path = _save_cursor_preview(
            tool_window, feas.align_xy, feas.frame_wh,
            Path(context["capture_path"]).parent, context["tag"],
        )
        result.notes.append(f"cursor_preview={cursor_path}")
        print(f"[INFO] 커서 안착 화면 재캡처: {cursor_path}")
    except Exception as exc:
        print(f"[WARNING] 커서 안착 화면 재캡처 실패: {exc}")


def _normalize_bbox(bbox):
    """SEM box bbox(dict l/t/r/b 또는 tuple)를 (l,t,r,b) int tuple 로 정규화. None→None."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return (int(bbox["left"]), int(bbox["top"]), int(bbox["right"]), int(bbox["bottom"]))
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _run_zoom_ladder(
    context: dict,
    result: "CycleResult",
    feas,
    settings: Workflow3Settings,
    sem_box_client,
    ocr_client,
) -> None:
    """모호/부재 verdict 일 때 live SEM box 안에서 wheel 로 fail 배율 기준 OUT(↓)·IN(↑)
    양방향을 한 칸씩 훑어 각 배율(rung)의 화면을 저장한다(클릭/recenter 없음 = 순수 wheel+캡처).

    zoom-out 만으로는 키를 못 찾으므로(좁은 FOV→넓게 보고, 다시 좁혀 정확한 점 확인)
    양방향이 필요하다. rematch 가 켜져 있으면 각 rung 에서 rcp 키를 재매칭해(mark_align_
    feasibility) 키가 또렷해지는 배율과 그 align point 를 표시(_marked.jpg)한다.

    핵심: wheel 은 반드시 **검출된 live SEM box 중심** 위에서 건다. 박스를 못 찾으면
    창 중심에 잘못 스크롤하지 않도록 탐색을 생략한다. 매 스크롤 전 커서를 박스로 다시
    옮기고(hover 안착) 창을 활성화해 wheel 이 박스 위에서 걸리게 한다.

    teardown(close)을 막지 않도록 전체를 try/except 로 감싼다. tool 창이 닫히기 전에
    호출돼야 한다(image→screen 변환·재캡처가 살아 있는 창을 필요로 함).
    """
    if not settings.zoom_probe_enabled:
        return
    if feas.verdict not in _ZOOM_PROBE_VERDICTS:
        print(f"[INFO] zoom ladder 생략: verdict={feas.verdict} (대상 아님)")
        return
    tool_window = context.get("tool_window")
    if feas.frame_wh is None or tool_window is None:
        print(f"[INFO] zoom ladder 생략: frame_wh/tool_window 없음 (verdict={feas.verdict})")
        return
    if image_point_to_screen is None or scroll_at_screen is None or move_cursor_to_screen is None:
        print("[INFO] zoom ladder 생략: window/mouse util 비활성(개발 PC)")
        return

    try:
        from poc.workflow_3.align.diagnostics.feasibility_check import mark_align_feasibility
        from poc.workflow_3.sem_monitor.sem_box_detect import (
            detect_sem_box,
            parse_pm_magnification,
        )

        eqp_id = context.get("eqp_id", "")
        recipe_id = context.get("recipe_id", "")
        capture_dir = Path(context["capture_path"]).parent
        tag = context["tag"]
        baseline_mag = parse_pm_magnification(feas.pm_text)

        def _box_center_screen(bbox):
            bbox = _normalize_bbox(bbox)
            if bbox is None:
                return None
            l, t, r, b = bbox
            return image_point_to_screen(
                tool_window, {"x": (l + r) // 2, "y": (t + b) // 2}, image_size=feas.frame_wh
            )

        method_cfg = (settings.zoom_method or "auto").lower()
        per = max(1, settings.zoom_probe_scrolls_per_step)
        dy_out = settings.zoom_probe_scroll_dy
        out_n = max(0, settings.zoom_probe_steps)
        in_n = max(0, settings.zoom_probe_in_steps)
        rungs: list[dict] = []

        # --- wheel 대상 = live SEM box 중심. feas 에 없으면 새 캡처로 1회 재검출. ---
        # pm_dropdown 강제 모드는 wheel 을 안 쓰므로 SEM box 중심이 없어도 진행한다
        # (PM 버튼은 pm_box 로 따로 찾는다).
        box = _normalize_bbox(feas.sem_box_bbox)
        if box is None and sem_box_client is not None:
            try:
                det0 = detect_sem_box(
                    capture_window(tool_window), sem_box_client,
                    ocr_client=ocr_client, two_stage=settings.pm_two_stage_ocr_enabled,
                )
                box = _normalize_bbox(det0.bbox_px)
            except Exception as exc:
                print(f"[WARNING] zoom ladder: 초기 SEM box 재검출 실패: {exc}")
        screen = _box_center_screen(box)
        if screen is None and method_cfg != "pm_dropdown":
            print(
                "[WARNING] zoom ladder 생략: live SEM box 위치 불명 -> wheel 대상 없음. "
                "창 중심 오스크롤 방지로 중단(ALIGN_FAIL_SEM_BOX_DETECT 활성/VLM 연결 확인)."
            )
            return

        # tool 창 포커스 — 일부 Windows 컨트롤은 포커스 없으면 입력을 무시한다.
        if activate_window is not None:
            try:
                activate_window(tool_window)
            except Exception:
                pass

        scr_txt = f"({screen['x']},{screen['y']})" if screen else "n/a"
        print(
            f"[INFO] zoom ladder 시작: method={method_cfg} verdict={feas.verdict} "
            f"box_center->screen={scr_txt} out={out_n} in={in_n} "
            f"dy={dy_out} per={per} rematch={settings.zoom_probe_rematch_enabled} "
            f"baseline_pm={feas.pm_text!r}({baseline_mag})"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )

        def _aim_and_scroll(dy, idx):
            # 매 스크롤 전 커서를 (갱신된) SEM box 중심으로 다시 옮겨 wheel 이 박스 위에서 걸리게.
            nonlocal screen
            if screen is not None:
                move_cursor_to_screen(
                    screen, f"zoom_ladder_aim{idx}", action_enabled=settings.action_enabled
                )
                if _ZOOM_AIM_SETTLE_SEC > 0:
                    time.sleep(_ZOOM_AIM_SETTLE_SEC)
                for _ in range(per):
                    scroll_at_screen(
                        screen, dy, "zoom_ladder", idx, action_enabled=settings.action_enabled
                    )
            if settings.zoom_probe_settle_sec > 0:
                time.sleep(settings.zoom_probe_settle_sec)

        def _capture_rung(label):
            nonlocal screen
            image = capture_window(tool_window)
            full_path = capture_dir / f"{tag}_rcs_zoom_{label}.jpg"
            save_debug_jpeg(image, full_path)
            rung = {
                "label": label, "full_path": str(full_path), "verdict": None,
                "decision": None, "score": None, "second_ratio": None,
                "pm_text": None, "magnification": None, "align_xy": None,
                "marked_path": None, "sembox_path": None,
            }
            new_box = None
            if settings.zoom_probe_rematch_enabled:
                try:
                    fr = mark_align_feasibility(
                        full_path, eqp_id=eqp_id, recipe_id=recipe_id,
                        cond_box_crop=settings.cond_box_crop,
                        reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
                        vlm_client=sem_box_client, ocr_client=ocr_client,
                        pm_two_stage=settings.pm_two_stage_ocr_enabled,
                    )
                    rung.update(
                        verdict=fr.verdict, decision=fr.decision, score=fr.score,
                        second_ratio=fr.second_ratio, pm_text=fr.pm_text,
                        magnification=parse_pm_magnification(fr.pm_text),
                        align_xy=list(fr.align_xy) if fr.align_xy is not None else None,
                        marked_path=str(fr.marked_path) if fr.marked_path is not None else None,
                    )
                    new_box = _normalize_bbox(fr.sem_box_bbox)
                except Exception as exc:
                    print(f"[WARNING] zoom ladder {label} 재매칭 실패: {exc}")
            elif sem_box_client is not None:
                # 재매칭 off → SEM box 만 검출(다음 조준 + ROI crop + PM 확인).
                try:
                    det = detect_sem_box(
                        image, sem_box_client, ocr_client=ocr_client,
                        two_stage=settings.pm_two_stage_ocr_enabled,
                    )
                    rung["pm_text"] = det.pm_text
                    rung["magnification"] = parse_pm_magnification(det.pm_text)
                    new_box = _normalize_bbox(det.bbox_px)
                except Exception as exc:
                    print(f"[WARNING] zoom ladder {label} SEM box 검출 실패: {exc}")

            # ROI crop 저장 (이번 rung 의 box, 없으면 직전 box).
            cbox = new_box or box
            if cbox is not None:
                try:
                    l, t, r, b = cbox
                    crop = image.crop((l, t, r, b))
                    sp = capture_dir / f"{tag}_rcs_zoom_{label}_sembox.jpg"
                    save_debug_jpeg(crop, sp)
                    rung["sembox_path"] = str(sp)
                except Exception as exc:
                    print(f"[WARNING] zoom ladder {label} ROI crop 실패: {exc}")

            # 다음 스크롤 조준 갱신(배율이 바뀌면 박스가 이동).
            ns = _box_center_screen(new_box) if new_box is not None else None
            if ns is not None:
                screen = ns
            rungs.append(rung)
            print(
                f"[INFO] zoom ladder {label}: verdict={rung['verdict']} "
                f"score={rung['score']} 2nd/best={rung['second_ratio']} "
                f"pm={rung['pm_text']!r} → {full_path.name}"
            )

        # --- 방식 분기. method=pm_dropdown 이면 wheel 생략하고 곧장 PM 버튼 드롭다운. ---
        method = "wheel"
        dropdown_meta = None

        if method_cfg == "pm_dropdown":
            method = "pm_dropdown"
            print("[INFO] zoom ladder: method=pm_dropdown 강제 -> wheel 생략, PM 버튼 클릭으로 진행.")
            dropdown_meta = _run_pm_dropdown_arms(
                tool_window, capture_dir, tag, feas, settings,
                sem_box_client, ocr_client, baseline_mag, out_n, in_n, _capture_rung,
            )
            _finish_zoom_ladder(
                capture_dir, tag, feas, settings, result,
                rungs, method, baseline_mag, dy_out, out_n, in_n, per, dropdown_meta,
            )
            return

        # --- OUT arm (배율↓). out1 후 PM 불변이면 wheel 무효 → PM 드롭다운 fallback. ---
        wheel_dead = False
        for i in range(1, out_n + 1):
            _aim_and_scroll(dy_out, i)
            _capture_rung(f"out{i}")
            if (
                method_cfg == "auto"
                and i == 1
                and settings.pm_dropdown_enabled
                and baseline_mag is not None
            ):
                m1 = rungs[-1].get("magnification")
                if m1 is not None and abs(m1 - baseline_mag) < 1e-6:
                    wheel_dead = True
                    print(
                        f"[WARNING] zoom ladder: out1 wheel 후 PM 불변({m1}=={baseline_mag}) "
                        f"-> wheel 이 배율을 안 바꿈. PM 드롭다운 fallback 전환."
                    )
                    break

        if wheel_dead:
            method = "pm_dropdown"
            # 방금 캡처한 wheel out1 rung(배율 불변 = baseline 중복)을 버린다 — 곧 드롭다운이
            # 진짜 낮은 배율의 out1 을 다시 캡처하므로 라벨/파일 중복을 막는다.
            if rungs:
                rungs.pop()
            dropdown_meta = _run_pm_dropdown_arms(
                tool_window, capture_dir, tag, feas, settings,
                sem_box_client, ocr_client, baseline_mag, out_n, in_n, _capture_rung,
            )
        else:
            # --- baseline 복귀(캡처 없이 반대 방향으로 OUT 만큼) — arm 전환. ---
            if out_n > 0 and in_n > 0:
                for i in range(1, out_n + 1):
                    _aim_and_scroll(-dy_out, i)
                if sem_box_client is not None:
                    try:
                        chk = detect_sem_box(
                            capture_window(tool_window), sem_box_client,
                            ocr_client=ocr_client, two_stage=settings.pm_two_stage_ocr_enabled,
                        )
                        back_mag = parse_pm_magnification(chk.pm_text)
                        if back_mag is not None and baseline_mag is not None and back_mag != baseline_mag:
                            print(
                                f"[WARNING] zoom ladder: baseline 복귀 PM {back_mag} != 시작 "
                                f"{baseline_mag} (wheel 비대칭 - IN arm 배율이 어긋날 수 있음)"
                            )
                        ns = _box_center_screen(_normalize_bbox(chk.bbox_px))
                        if ns is not None:
                            screen = ns
                    except Exception:
                        pass

            # --- IN arm (배율↑ = OUT 반대 부호). ---
            for i in range(1, in_n + 1):
                _aim_and_scroll(-dy_out, i)
                _capture_rung(f"in{i}")

        _finish_zoom_ladder(
            capture_dir, tag, feas, settings, result,
            rungs, method, baseline_mag, dy_out, out_n, in_n, per, dropdown_meta,
        )
    except Exception as exc:
        print(f"[WARNING] zoom ladder 실패(캡처는 유지): {exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="zoom_ladder_error", level="warning",
            eqp_id=context.get("eqp_id", ""), error=str(exc),
        )


def _finish_zoom_ladder(
    capture_dir, tag, feas, settings, result,
    rungs, method, baseline_mag, dy_out, out_n, in_n, per, dropdown_meta,
) -> None:
    """ladder rung 들을 best 선정 + sidecar JSON 저장 + manifest notes 로 마무리한다.

    wheel 경로와 pm_dropdown 강제 경로가 공통으로 호출한다.
    """
    present = [
        r for r in rungs
        if r.get("verdict") in ("possible", "ambiguous") and r.get("score") is not None
    ]
    best = max(present, key=lambda r: r["score"]) if present else None

    json_path = capture_dir / f"{tag}_zoom_ladder.json"
    try:
        payload = {
            "verdict_at_fail": feas.verdict,
            "baseline_pm_text": feas.pm_text,
            "baseline_magnification": baseline_mag,
            "method": method,
            "scroll_dy_out": dy_out, "out_steps": out_n, "in_steps": in_n,
            "scrolls_per_step": per, "rematch": settings.zoom_probe_rematch_enabled,
            "best_rung": best["label"] if best else None,
            "rungs": rungs,
        }
        if dropdown_meta is not None:
            payload["pm_dropdown"] = dropdown_meta
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result.notes.append(
            f"zoom_ladder={json_path} method={method} rungs={len(rungs)} "
            f"best={best['label'] if best else '-'}"
        )
        print(
            f"[INFO] zoom ladder 완료: method={method} {json_path} (rung {len(rungs)}, "
            f"best={best['label'] if best else '없음'})"
        )
    except Exception as exc:
        print(f"[WARNING] zoom ladder sidecar JSON 저장 실패: {exc}")


def _save_pm_dropdown_overlay(image, out_path, pm_box, btn, region) -> None:
    """PM 드롭다운 검증 overlay 저장.

    PM 숫자 박스(cyan, 화면에 이미 보이는 그 박스), 유도한 'PM 버튼' 클릭점(red 십자/원),
    드롭다운 crop 영역(yellow)을 그려 클릭이 'PM' 버튼에 정확히 맞는지 눈으로 검증한다.
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if pm_box:
        draw.rectangle(
            [int(pm_box["left"]), int(pm_box["top"]),
             int(pm_box["right"]), int(pm_box["bottom"])],
            outline=(0, 255, 255), width=3,
        )
    if region:
        l, t, r, b = region
        draw.rectangle([int(l), int(t), int(r), int(b)], outline=(255, 255, 0), width=2)
    if btn:
        x, y = int(btn["x"]), int(btn["y"])
        rad = 8
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(255, 0, 0), width=3)
        draw.line([x - rad - 5, y, x + rad + 5, y], fill=(255, 0, 0), width=2)
        draw.line([x, y - rad - 5, x, y + rad + 5], fill=(255, 0, 0), width=2)
        draw.text((x + rad + 4, y - 14), "PM btn click", fill=(255, 0, 0))
    save_debug_jpeg(canvas, out_path)


def _run_pm_dropdown_arms(
    tool_window,
    capture_dir,
    tag,
    feas,
    settings: Workflow3Settings,
    sem_box_client,
    ocr_client,
    baseline_mag,
    out_n: int,
    in_n: int,
    capture_rung,
) -> dict:
    """wheel 무효 시 'PM' 버튼 드롭다운으로 절대 배율을 골라 OUT/IN ladder 를 돈다.

    절차: (1) 현재 화면에서 PM 숫자 박스 검출 → 'PM' 버튼 클릭 → 드롭다운 오픈+캡처,
    (2) 최초 1회 PaddleOCR ``Spotting:`` 으로 행별 (배율, 클릭 박스) 읽기 → 값공간 목표 산출,
    (3) 각 목표 행 클릭(절대 배율 적용) → settle → ``capture_rung(label)`` 으로 캡처+재매칭.

    절대 선택이라 wheel 처럼 baseline 복귀가 필요 없다(드리프트 없음). 드롭다운은 PM
    버튼에 고정 앵커되어 같은 위치에 다시 열린다고 보고, 옵션 클릭 좌표는 첫 읽기 값을
    재사용한다(매번 재오픈하되 재읽기는 생략). 읽기 실패/옵션 부족이면 중단(엔지니어 처리).

    반환: zoom_ladder.json 의 ``pm_dropdown`` 섹션에 들어갈 메타(dict).
    """
    from poc.workflow_3.sem_monitor.pm_dropdown import (
        choose_step_targets,
        dropdown_region,
        pm_button_point,
        read_dropdown_options,
    )
    from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box

    meta = {"pm_options": [], "spotting_raw": "", "selections": [], "aborted": None}
    frame_wh = feas.frame_wh

    # spotting 읽기용 OCR client — 없으면 paddleocr-vl 로 1개 생성(실패 시 sem_box client 재사용).
    reader = ocr_client
    if reader is None:
        try:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            reader = Workflow1VLMClient(settings.pm_ocr_service, timeout_sec=15.0)
        except Exception as exc:
            print(f"[WARNING] PM 드롭다운: OCR client 생성 실패 -> sem_box client 로 읽기: {exc}")
            reader = sem_box_client

    def _open_dropdown(open_idx):
        """현재 화면에서 PM 버튼을 눌러 드롭다운을 열고 (crop, origin, pm_box) 반환(실패 None)."""
        img = capture_window(tool_window)
        pm_box = None
        try:
            det = detect_sem_box(
                img, sem_box_client, ocr_client=ocr_client,
                two_stage=settings.pm_two_stage_ocr_enabled,
            )
            pm_box = det.pm_box_px
        except Exception as exc:
            print(f"[WARNING] PM 드롭다운: PM 박스 검출 실패: {exc}")
        if pm_box is None:
            return None
        btn = pm_button_point(pm_box)
        btn_scr = (
            image_point_to_screen(tool_window, btn, image_size=frame_wh) if btn else None
        )
        if btn_scr is None:
            return None
        print(
            f"[INFO] PM 드롭다운 열기#{open_idx}: PM버튼 px={btn} -> screen={btn_scr}"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        click_at_screen(btn_scr, "pm_button", action_enabled=settings.action_enabled)
        if _PM_DROPDOWN_OPEN_SETTLE_SEC > 0:
            time.sleep(_PM_DROPDOWN_OPEN_SETTLE_SEC)
        shot = capture_window(tool_window)
        region = dropdown_region(pm_box, frame_wh)
        # 검증 overlay: PM 숫자 박스(cyan, 화면의 그것과 동일), 유도한 PM 버튼 클릭점(red),
        # 드롭다운 crop 영역(yellow) 을 그려 클릭이 'PM' 버튼에 맞는지 눈으로 확인.
        try:
            _save_pm_dropdown_overlay(
                shot, capture_dir / f"{tag}_pm_dropdown_open{open_idx}.jpg",
                pm_box, btn, region,
            )
        except Exception as exc:
            print(f"[WARNING] PM 드롭다운 overlay 저장 실패: {exc}")
            try:
                save_debug_jpeg(shot, capture_dir / f"{tag}_pm_dropdown_open{open_idx}.jpg")
            except Exception:
                pass
        if region is None:
            return None
        l, t, r, b = region
        return shot.crop((l, t, r, b)), (l, t), pm_box

    # 1) 첫 오픈 + 옵션 읽기.
    first = _open_dropdown(1)
    if first is None:
        meta["aborted"] = "pm_box_or_open_failed"
        print("[WARNING] PM 드롭다운 fallback 중단: PM 버튼/박스를 못 찾음.")
        return meta
    crop, origin, _pm_box = first
    options, raw_text = read_dropdown_options(crop, reader, crop_origin=origin)
    meta["spotting_raw"] = raw_text
    meta["pm_options"] = [{"value": o["value"], "text": o["text"]} for o in options]
    if options:
        print(
            f"[INFO] PM 드롭다운 옵션 {len(options)}개: "
            + ", ".join(f"{o['text']}({o['value']})" for o in options)
        )
    if len(options) < 2:
        meta["aborted"] = "too_few_options"
        print(
            "[WARNING] PM 드롭다운 fallback 중단: 옵션 2개 미만(읽기 실패 가능). "
            "spotting_raw 로 보정 필요."
        )
        return meta

    targets = choose_step_targets(options, baseline_mag, out_n, in_n)
    if not targets:
        meta["aborted"] = "no_targets"
        print(f"[WARNING] PM 드롭다운 fallback 중단: baseline={baseline_mag} 기준 목표 없음.")
        return meta

    def _click_option(opt):
        scr = image_point_to_screen(tool_window, opt["center"], image_size=frame_wh)
        if scr is None:
            return False
        print(
            f"[INFO] PM 옵션 클릭: {opt['text']}({opt['value']}) px={opt['center']} -> screen={scr}"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        click_at_screen(scr, f"pm_opt_{opt['text']}", action_enabled=settings.action_enabled)
        return True

    def _locate(opts, value):
        return min(opts, key=lambda o: abs(o["value"] - value)) if opts else None

    # 2) 각 목표: 첫 목표는 이미 열린 드롭다운(+읽은 옵션) 재사용, 이후는 매번 재오픈 후
    #    **재읽기**해 목표 배율 행의 현재 좌표를 다시 찾는다 — 선택으로 배율이 바뀌어
    #    드롭다운/PM 박스가 이동해도 stale 좌표로 오클릭하지 않게 한다.
    target_values = [(label, opt["value"]) for label, opt in targets]
    cur_options = options
    open_idx = 1
    for k, (label, value) in enumerate(target_values):
        if k > 0:
            open_idx += 1
            op = _open_dropdown(open_idx)
            if op is None:
                print(f"[WARNING] PM 드롭다운 재오픈 실패({label}) - 건너뜀.")
                continue
            crop_k, origin_k, _box_k = op
            cur_options, _raw_k = read_dropdown_options(crop_k, reader, crop_origin=origin_k)
        opt = _locate(cur_options, value)
        if opt is None:
            print(f"[WARNING] PM 드롭다운 재읽기 실패({label} target={value}) - 건너뜀.")
            continue
        if not _click_option(opt):
            print(f"[WARNING] PM 옵션 screen 변환 실패({label}) - 건너뜀.")
            continue
        meta["selections"].append(
            {"label": label, "value": opt["value"], "text": opt["text"]}
        )
        if settings.zoom_probe_settle_sec > 0:
            time.sleep(settings.zoom_probe_settle_sec)
        capture_rung(label)

    print(
        f"[INFO] PM 드롭다운 fallback 완료: 선택 {len(meta['selections'])}/{len(targets)} "
        f"(baseline={baseline_mag})"
    )
    return meta


def run_check_only_cycle(
    eqp_id: str,
    recipe_id: str,
    settings: Workflow3Settings,
    *,
    tag: str | None = None,
) -> CycleResult:
    """점검 전용 사이클 — 접속 → 첫 화면 1장 캡처 → tool 닫기.

    production `run_alarm_cycle` 의 경량 변형: 상시 녹화/SEM panel/CV 보정/engineer
    watch 를 모두 뺀다. 과거 데이터 수집(rcp/msr 는 office MES 가 align_images 에
    직접 적재, 최근 성공 S 이미지는 monitor 의 gather_success_async)은 사이클 밖에서
    이뤄진다. step 실패로 runner 가 중단돼도 tool 닫기·팝업 backstop 은 finally 가
    보장한다(러너가 중간에 죽어도 teardown 이 실행되게).
    """
    tag = tag or make_timestamp_tag()
    result = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag)

    if not RCS_MODULES_AVAILABLE:
        result.run_status = "rcs_unavailable"
        result.notes.append("RCS 모듈 비활성 — 감지/로그만")
        return result

    context: dict = {"eqp_id": eqp_id, "recipe_id": recipe_id, "tag": tag}
    runner = WorkflowRunner(
        settings,
        workflow_name=f"align_fail_check_{eqp_id}",
        log_name="work2",
        component_name=LOG_COMPONENT,
    )

    def executor(step, step_context):
        return _CHECK_STEP_EXECUTORS[step.step_id](step, step_context, settings)

    input_blocked = False
    try:
        # 자동 GUI 구간(접속~캡처~닫기) 동안 사용자 물리 입력 차단(opt-in).
        # engineer watch 가 없으므로 close 까지 차단 유지하고 finally 끝에서 해제.
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"align_fail_check {eqp_id}")
        run = runner.run(build_check_steps(eqp_id), context, executor)
        result.run_status = run.status
        result.run_dir = str(run.run_dir or "")
        for step_result in run.step_results:
            if step_result.status == "failed":
                result.failed_step = step_result.step_id
                result.failure_class = step_result.failure_class or ""
                break
        capture_path = context.get("capture_path")
        if capture_path is not None:
            result.outcome_status = "captured"
            result.outcome_path = str(capture_path)
            result.frame_count = 1
            # 캡처 성공 & tool 창이 아직 열린 동안(close 전) 보정 가능성 판정 +
            # (opt-in) align point 커서 미리보기 + 안착 화면 재캡처. rcp 자산 특정에
            # recipe_id 가 필요하므로 없으면 skip. live 창이 필요한 image→screen 변환·
            # 재캡처 때문에 finally 의 close 보다 반드시 먼저 와야 한다.
            if recipe_id and (
                settings.feasibility_mark_enabled
                or settings.reposition_preview_enabled
                or settings.zoom_probe_enabled
            ):
                _check_feasibility_and_preview(context, result, settings, eqp_id, recipe_id)
    except Exception as exc:
        result.run_status = "error"
        result.notes.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] 점검 사이클 예외: EQP_ID={eqp_id}, error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="check_cycle_error", level="error",
            eqp_id=eqp_id, error=str(exc),
        )
    finally:
        # teardown 보장 — tool 닫기 → 알림 팝업 backstop.
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            try:
                close_tool(eqp_id)
            except Exception as exc:
                print(f"[WARNING] tool 창 닫기 실패: {exc}")
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)
        # 입력 차단 해제 — 자동 구간 전체(닫기 포함) 종료 후. 예외 경로에서도 반드시 해제.
        if input_blocked:
            block_input(False, debug_label=f"align_fail_check {eqp_id}")
            input_blocked = False

    return result


__all__ = [
    "CycleResult",
    "build_check_steps",
    "build_cycle_steps",
    "run_alarm_cycle",
    "run_check_only_cycle",
]
