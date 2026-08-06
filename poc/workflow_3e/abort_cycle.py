"""측정 abort 사이클 — workflow_3 의 step executor/runner/teardown 을 재사용한다.

run_alarm_cycle 의 형제. 녹화/engineer-watch 는 없다(abort 가 곧 행동). 접속·창대기·증거
캡처는 workflow_3.monitor.cycle 의 기존 executor 를 그대로 쓰고, Stop/Abort 클릭만 새로
더한다. step 실패로 runner 가 중단돼도 cube 알림·tool 닫기·팝업 backstop 은 finally 가
보장한다.

단계:
  1. ensure_rcs_ready   — RCS 메인 창 확보 (재사용)
  2. close_alert_popup  — 감지 팝업 닫기 (재사용)
  3. connect_tool       — tool 더블클릭 접속 (재사용)
  4. wait_tool_window   — Remote Monitoring 창 대기 (재사용, 점유 처리 포함)
  5. capture_screen     — abort 전 증거 1장 (재사용)
  6. abort_measurement  — Stop/Abort 버튼 locate + (무장 시) 클릭 + 확인 (신규)
"""

import time

import numpy as np

from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.cycle import (
    CLOSE_TOOL_AVAILABLE,
    RCS_MODULES_AVAILABLE,
    CycleResult,
    _CHECK_CAPTURE_SETTLE_SEC,
    _ctx_set,
    _exec_capture_screen,
    _exec_close_alert_popup,
    _exec_connect_tool,
    _exec_ensure_rcs_ready,
    _exec_wait_tool_window,
    _make_result,
    _should_block_input,
    close_tool,
)
from poc.workflow_3.monitor.notify import close_alert_window
from poc.workflow_3.monitor.teardown import run_teardown
from poc.workflow_3.runner.workflow_runner import WorkflowRunner
from poc.workflow_3.runner.workflow_types import WorkflowStep
from poc.workflow_3.util import block_input, capture_window, click_at_screen, make_timestamp_tag
from poc.workflow_3e.abort_button import locate_abort_button, locate_abort_confirm
from poc.workflow_3e.notify import notify_abort_outcome

LOG_COMPONENT = "measurement_abort_cycle"


def build_abort_steps(eqp_id: str) -> list[WorkflowStep]:
    """측정 abort 사이클 step — 접속 -> 증거 캡처 -> Stop/Abort 버튼 클릭(게이트)."""
    return [
        WorkflowStep(
            step_id="ensure_rcs_ready", step_type="recover",
            target_description="RCS 메인 창 확보(전면화/재실행+재로그인)",
            success_criteria=_ctx_set("rcs_main_window"),
        ),
        WorkflowStep(
            step_id="close_alert_popup", step_type="cleanup",
            target_description="감지 알림 팝업 닫기(screenshot 오염 방지)",
        ),
        WorkflowStep(
            step_id="connect_tool", step_type="action",
            target_description=f"List 탭에서 tool 더블클릭: {eqp_id}",
            depends_on=["ensure_rcs_ready"],
        ),
        WorkflowStep(
            step_id="wait_tool_window", step_type="detect",
            target_description="Remote Monitoring 창 대기",
            depends_on=["connect_tool"], success_criteria=_ctx_set("tool_window"),
        ),
        WorkflowStep(
            step_id="capture_screen", step_type="action",
            target_description="abort 전 증거 화면 1장 캡처",
            depends_on=["wait_tool_window"], success_criteria=_ctx_set("capture_path"),
        ),
        WorkflowStep(
            step_id="abort_measurement", step_type="action",
            target_description="Stop/Abort 버튼 클릭으로 측정 중단(이중 게이트)",
            depends_on=["capture_screen"],
        ),
    ]


def _exec_abort_measurement(step, context, settings) -> "object":
    """Stop/Abort 버튼 locate + (무장 시) 클릭 + 확인. 이중 게이트로 보호한다.

    실제 클릭은 SAFE_MODE off **이고** abort_action_dry_run=False 일 때만. 그 외에는
    locate 좌표를 [DRY-RUN] 으로 로깅하고 클릭하지 않는다(notify-only 검증 경로).
    abort_outcome 문자열을 context 에 남겨 사이클이 manifest/notify 에 쓰게 한다.
    """
    started_at = time.time()
    eqp_id = context["eqp_id"]
    tool_window = context.get("tool_window")
    if tool_window is None:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_no_window", error_message="tool 창 없음 - abort 생략",
        )

    try:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.abort_button_vlm_service, timeout_sec=15.0)
    except Exception as exc:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_vlm_error", error_message=f"{type(exc).__name__}: {exc}",
        )

    frame = np.array(capture_window(tool_window))
    xy = locate_abort_button(frame_bgr=frame, client=client)
    if xy is None:
        context["abort_outcome"] = "abort_button_not_found"
        print(f"[WARNING] Abort 버튼을 찾지 못함 - 엔지니어 직접 처리 (EQP_ID={eqp_id})")
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_button_not_found", error_message="Stop/Abort 버튼 미검출",
        )

    armed = settings.action_enabled and not settings.abort_action_dry_run
    if not armed:
        context["abort_outcome"] = "abort_dry_run"
        print(
            f"[INFO] [DRY-RUN] Abort 버튼 검출 screen=({xy[0]},{xy[1]}) - 클릭 생략 "
            f"(SAFE_MODE/abort_dry_run 게이트). EQP_ID={eqp_id}"
        )
        return _make_result(step, "success", started_at, settings)

    # --- 무장 상태: 클릭 + 확인 다이얼로그 ---
    try:
        click_at_screen({"x": xy[0], "y": xy[1]}, "abort_button", action_enabled=True)
        if _CHECK_CAPTURE_SETTLE_SEC > 0:
            time.sleep(_CHECK_CAPTURE_SETTLE_SEC)
        confirm_frame = np.array(capture_window(tool_window))
        cxy = locate_abort_confirm(frame_bgr=confirm_frame, client=client)
        if cxy is not None:
            click_at_screen({"x": cxy[0], "y": cxy[1]}, "abort_confirm", action_enabled=True)
        context["abort_outcome"] = "aborted"
        print(f"[INFO] 측정 abort 실행: EQP_ID={eqp_id} button=({xy[0]},{xy[1]}) confirm={cxy}")
    except Exception as exc:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_click_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    return _make_result(step, "success", started_at, settings)


_ABORT_STEP_EXECUTORS = {
    "ensure_rcs_ready": _exec_ensure_rcs_ready,
    "close_alert_popup": _exec_close_alert_popup,
    "connect_tool": _exec_connect_tool,
    "wait_tool_window": _exec_wait_tool_window,
    "capture_screen": _exec_capture_screen,
    "abort_measurement": _exec_abort_measurement,
}


def _abort_teardown_steps(eqp_id, context, settings, *, input_blocked):
    """abort 사이클 teardown 단계 목록 - workflow_3 의 두 사이클과 같은 규약.

    첫 단계는 **항상** 입력 해제. 전제조건은 목록에서 빼지 않고 클로저 안에서
    판정한다(목록 길이/순서 고정 -> 순서 테스트 가능).
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"measurement_abort {eqp_id}")

    def _close_tool():
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)

    def _close_alert():
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return [
        ("input_unblock", _unblock),
        ("close_tool", _close_tool),
        ("close_alert", _close_alert),
    ]


def run_abort_cycle(
    eqp_id: str, recipe_id: str, settings, *, tag: str | None = None, detail: str = ""
) -> CycleResult:
    """측정 abort 알람 1건 사이클 — 접속 -> 증거 캡처 -> Stop/Abort 클릭(게이트) -> 닫기.

    detail 은 알람 정보(예: 연속 실패 수가 담긴 ALARM_NAME)로 cube 알림에 함께 실린다.
    step 실패로 runner 가 중단돼도 cube 알림·tool 닫기·팝업 backstop 은 finally 가 보장한다.
    예외는 삼켜 상위 슈퍼바이저 루프가 죽지 않게 한다.
    """
    tag = tag or make_timestamp_tag()
    result = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag)

    if not RCS_MODULES_AVAILABLE:
        result.run_status = "rcs_unavailable"
        result.notes.append("RCS 모듈 비활성 - 감지/로그만")
        notify_abort_outcome(
            eqp_id, recipe_id, None, detail=detail, enabled=settings.rich_notify_enabled
        )
        return result

    context: dict = {"eqp_id": eqp_id, "recipe_id": recipe_id, "tag": tag}
    runner = WorkflowRunner(
        settings, workflow_name=f"measurement_abort_{eqp_id}",
        log_name="work2", component_name=LOG_COMPONENT,
    )

    def executor(step, step_context):
        return _ABORT_STEP_EXECUTORS[step.step_id](step, step_context, settings)

    input_blocked = False
    try:
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"measurement_abort {eqp_id}")
        run = runner.run(build_abort_steps(eqp_id), context, executor)
        result.run_status = run.status
        result.run_dir = str(run.run_dir or "")
        for step_result in run.step_results:
            if step_result.status == "failed":
                result.failed_step = step_result.step_id
                result.failure_class = step_result.failure_class or ""
                break

        result.outcome_status = context.get("abort_outcome", "")
        if context.get("capture_path") is not None:
            result.outcome_path = str(context["capture_path"])
        notify_abort_outcome(
            eqp_id, recipe_id, result.outcome_status or None,
            capture_path=result.outcome_path, detail=detail,
            enabled=settings.rich_notify_enabled,
        )
    except Exception as exc:
        result.run_status = "error"
        result.notes.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] abort 사이클 예외: EQP_ID={eqp_id}, error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="abort_cycle_error", level="error",
            eqp_id=eqp_id, error=str(exc),
        )
    finally:
        failures = run_teardown(
            _abort_teardown_steps(eqp_id, context, settings, input_blocked=input_blocked),
            label=f"measurement_abort {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)

    return result


__all__ = ["build_abort_steps", "run_abort_cycle"]
