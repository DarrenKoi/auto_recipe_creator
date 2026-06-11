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
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.notify import close_alert_window, notify_correction_outcome
from poc.workflow_3.monitor.recording import RecordingSession
from poc.workflow_3.monitor.sem_controller import build_rcs_sem_monitor
from poc.workflow_3.runner.workflow_runner import WorkflowRunner
from poc.workflow_3.runner.workflow_types import (
    ConditionGroup,
    ConditionType,
    StepCondition,
    StepResult,
    WorkflowStep,
)
from poc.workflow_3.util import activate_window, make_timestamp_tag

LOG_COMPONENT = "align_fail_cycle"

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
    """④ tool 창 대기 — RCS 점유(select 팝업) 시 건드리지 않고 포기."""
    started_at = time.time()
    eqp_id = context["eqp_id"]
    window, title, backend = wait_for_remote_monitoring_window(
        eqp_id, max_attempts=settings.rcs_window_max_trials
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

    from poc.workflow_3.vision.align_fail_correct import CorrectionConfig, correct_align_fail_auto

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
    try:
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


__all__ = ["CycleResult", "build_cycle_steps", "run_alarm_cycle"]
