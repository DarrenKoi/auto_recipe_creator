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
from poc.workflow_3.util import (
    block_input,
    capture_window,
    click_at_screen,
    image_point_to_screen,
    make_timestamp_tag,
)
from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3e.abort_button import (
    ABORT_LABEL_POLICY_OFF,
    CONFIRM_BUTTON_LABELS,
    accepts_label,
    expected_labels_for_target,
    is_rehearsal_target,
    load_abort_label_policy,
    load_abort_target,
    locate_abort_confirm,
    locate_for_target,
    verify_button_label_at_point,
)
from poc.workflow_3e.notify import notify_abort_outcome

LOG_COMPONENT = "measurement_abort_cycle"
DEBUG_ARTIFACT_DIR = DEBUG_IMAGE_DIR / "measurement_abort"


def _make_abort_vlm_client(settings):
    """abort 버튼 locate 용 VLM client. 실패는 호출부가 abort_vlm_error 로 처리한다."""
    from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

    return Workflow1VLMClient(settings.abort_button_vlm_service, timeout_sec=15.0)


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
        client = _make_abort_vlm_client(settings)
    except Exception as exc:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_vlm_error", error_message=f"{type(exc).__name__}: {exc}",
        )

    # 클릭 대상: 검증 단계에는 Stop/Abort 대신 인접한 Queue 를 겨눈다(rehearsal).
    target = load_abort_target()
    rehearsal = is_rehearsal_target(target)
    context["abort_target"] = target

    image = capture_window(tool_window)
    # xy 는 **창-이미지 좌표**다. 라벨 확인은 이 좌표계에서 crop 하고, 클릭 직전에만
    # image_point_to_screen 으로 스크린 좌표를 만든다(두 좌표계를 섞지 않는다).
    xy = locate_for_target(
        target,
        window=tool_window,
        window_title=str(context.get("tool_window_title") or ""),
        backend=str(context.get("tool_window_backend") or ""),
        image=image,
    )
    if xy is None:
        context["abort_outcome"] = "abort_button_not_found"
        print(f"[WARNING] {target} 버튼을 찾지 못함 - 엔지니어 직접 처리 (EQP_ID={eqp_id})")
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_button_not_found", error_message=f"{target} 버튼 미검출",
        )

    # --- 라벨 확인 게이트: 그 좌표가 정말 Stop/Abort 인가 (좁은 crop OCR) ---
    # 무장 여부와 무관하게 항상 읽는다. notify-only 로 도는 동안 오피스가 게이트 판정을
    # 눈으로 볼 수 있어야, 무장 전에 라벨 집합/crop 크기를 교정할 수 있다.
    policy = load_abort_label_policy()
    verdict = None
    if policy != ABORT_LABEL_POLICY_OFF:
        verdict = verify_button_label_at_point(
            image,
            xy,
            expected_labels_for_target(target),
            debug_image_dir=DEBUG_ARTIFACT_DIR,
            timestamp_tag=str(context.get("tag") or make_timestamp_tag()),
            artifact_label=f"{target}_button_{eqp_id}",
        )
    verdict_status = getattr(verdict, "status", "skipped" if policy == ABORT_LABEL_POLICY_OFF else "unavailable")
    context["abort_label_verdict"] = verdict_status

    # rehearsal 대상이면 무장 여부와 무관하게 여기서 끝난다. 로케이트 + 라벨확인이라는
    # 검증 목적은 이미 달성했고, 실제 abort 가 아닌 버튼을 누를 이유는 없다.
    if rehearsal:
        context["abort_outcome"] = "abort_rehearsal"
        print(
            f"[INFO] [REHEARSAL:{target}] 버튼 검출 img=({xy[0]},{xy[1]}) "
            f"label={verdict_status}(policy={policy}) - 클릭 생략(검증 전용 대상). "
            f"실제 abort 는 MEAS_FAIL_ABORT_TARGET=abort. EQP_ID={eqp_id}"
        )
        return _make_result(step, "success", started_at, settings)

    armed = settings.action_enabled and not settings.abort_action_dry_run
    if not armed:
        context["abort_outcome"] = "abort_dry_run"
        print(
            f"[INFO] [DRY-RUN] {target} 버튼 검출 img=({xy[0]},{xy[1]}) "
            f"label={verdict_status}(policy={policy}) - 클릭 생략 "
            f"(SAFE_MODE/abort_dry_run 게이트). EQP_ID={eqp_id}"
        )
        return _make_result(step, "success", started_at, settings)

    if not accepts_label(verdict, policy):
        context["abort_outcome"] = "abort_label_unconfirmed"
        read_text = getattr(verdict, "raw_text", "")
        print(
            f"[WARNING] Abort 버튼 라벨 미확인({verdict_status}, policy={policy}) - "
            f"클릭 금지, 엔지니어 직접 처리. EQP_ID={eqp_id} "
            f"img=({xy[0]},{xy[1]}) read={read_text!r}"
        )
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_label_unconfirmed",
            error_message=f"Stop/Abort 라벨 확인 실패({verdict_status})",
        )

    # --- 무장 상태: 클릭 + 확인 다이얼로그 ---
    screen_xy = image_point_to_screen(
        tool_window, {"x": xy[0], "y": xy[1]}, image_size=image.size
    )
    if screen_xy is None:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_coord_error",
            error_message="창-이미지 -> 스크린 좌표 변환 실패",
        )

    try:
        click_at_screen(screen_xy, "abort_button", action_enabled=True)
        if _CHECK_CAPTURE_SETTLE_SEC > 0:
            time.sleep(_CHECK_CAPTURE_SETTLE_SEC)
        confirm_image = capture_window(tool_window)
        cxy = locate_abort_confirm(frame_bgr=np.array(confirm_image), client=client)
        if cxy is not None:
            # 확인 다이얼로그의 Yes/확인 도 같은 게이트를 통과해야 누른다. 막히면 다이얼로그가
            # 열린 채 남고 엔지니어가 마무리한다 - No/취소 를 잘못 누르는 것보다 낫다.
            cverdict = None
            if policy != ABORT_LABEL_POLICY_OFF:
                cverdict = verify_button_label_at_point(
                    confirm_image,
                    cxy,
                    CONFIRM_BUTTON_LABELS,
                    debug_image_dir=DEBUG_ARTIFACT_DIR,
                    timestamp_tag=str(context.get("tag") or make_timestamp_tag()),
                    artifact_label=f"abort_confirm_{eqp_id}",
                )
            context["abort_confirm_verdict"] = getattr(
                cverdict, "status", "skipped" if policy == ABORT_LABEL_POLICY_OFF else "unavailable"
            )
            confirm_screen = image_point_to_screen(
                tool_window, {"x": cxy[0], "y": cxy[1]}, image_size=confirm_image.size
            )
            if accepts_label(cverdict, policy) and confirm_screen is not None:
                click_at_screen(confirm_screen, "abort_confirm", action_enabled=True)
            else:
                print(
                    "[WARNING] abort 확인 버튼 라벨 미확인 - 클릭 생략(다이얼로그는 열린 채 "
                    f"엔지니어 처리). EQP_ID={eqp_id} confirm_img=({cxy[0]},{cxy[1]})"
                )
                cxy = None
        context["abort_outcome"] = "aborted"
        print(f"[INFO] 측정 abort 실행: EQP_ID={eqp_id} button_screen={screen_xy} confirm_img={cxy}")
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
