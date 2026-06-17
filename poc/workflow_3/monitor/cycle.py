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
    env_float,
    image_point_to_screen,
    make_timestamp_tag,
    move_cursor_to_screen,
)

LOG_COMPONENT = "align_fail_cycle"


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
    if not settings.reposition_preview_enabled:
        return
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
            tool_window, feas.align_xy, feas.frame_wh, capture_path.parent, context["tag"]
        )
        result.notes.append(f"cursor_preview={cursor_path}")
        print(f"[INFO] 커서 안착 화면 재캡처: {cursor_path}")
    except Exception as exc:
        print(f"[WARNING] 커서 안착 화면 재캡처 실패: {exc}")


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
                settings.feasibility_mark_enabled or settings.reposition_preview_enabled
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
