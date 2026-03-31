"""RCS 로그인 워크플로."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.workflow_1.login_rcs_common import (
    RCS_MAIN_WINDOW_TITLE_PREFIX,
    RCS_UPDATER_WINDOW_TITLE_PREFIX,
    WINDOW_TITLE_PREFIX,
    find_rcs_main_window,
    find_rcs_updater_window,
    find_login_window,
)
from poc.workflow_1.debug_artifacts import save_debug_jpeg
from poc.workflow_1.logger import log_work2_event
from poc.work2.util import (
    activate_window,
    capture_window,
    click_at_screen,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
)
from poc.workflow_1.workflow_config import WorkflowSettings, load_workflow_settings
from poc.workflow_1.workflow_runner import WorkflowRunner
from poc.workflow_1.workflow_types import (
    ConditionGroup,
    ConditionGroupType,
    ConditionType,
    StepCondition,
    StepResult,
    WorkflowStep,
)

try:
    from pynput.keyboard import Key, Controller as KeyboardController

    PYNPUT_KEYBOARD_AVAILABLE = True
except ImportError:
    PYNPUT_KEYBOARD_AVAILABLE = False
    KeyboardController = None
    Key = None
    print("[WARNING] pynput.keyboard 미설치 — 타이핑 동작은 로그만 출력됩니다.")

try:
    from poc.work2.util.mouse_utils import PYNPUT_MOUSE_AVAILABLE
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False

load_dotenv()


LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
DEFAULT_CREDENTIAL_USER_ID = "2067928_B"
DEFAULT_CREDENTIAL_PASSWORD = "1"
EXIT_SUCCESS = "success"
EXIT_WORKFLOW_ABORTED = "workflow_aborted"


def _load_login_targets():
    """login target 정의와 분석 함수를 지연 import 한다."""
    from poc.workflow_1.login_rcs_ui_venus_mai import (
        EXIT_SUCCESS as DETECT_SUCCESS,
        PREDEFINED_TARGETS,
        analyze_login_target,
    )

    return DETECT_SUCCESS, PREDEFINED_TARGETS, analyze_login_target


def load_login_credentials() -> dict[str, str]:
    """로그인 자격증명을 읽는다."""
    return {
        "userid_input": os.getenv("ACTION_LOGIN_USER_ID", DEFAULT_CREDENTIAL_USER_ID).strip(),
        "password_input": os.getenv(
            "ACTION_LOGIN_PASSWORD",
            DEFAULT_CREDENTIAL_PASSWORD,
        ).strip(),
    }


def build_login_workflow_steps(
    settings: WorkflowSettings,
    credentials: dict[str, str],
) -> list[WorkflowStep]:
    """로그인 워크플로 step 목록을 구성한다."""
    detect_success, targets, _analyze = _load_login_targets()
    _ = detect_success

    visible_login_window = ConditionGroup(
        conditions=[
            StepCondition(
                condition_type=ConditionType.WINDOW_VISIBLE,
                title_fragment=WINDOW_TITLE_PREFIX,
            ),
        ]
    )
    steps = [
        WorkflowStep(
            step_id="ensure_login_window",
            step_type="observe",
            target_description="find the RCS login dialog",
            preconditions=ConditionGroup(
                conditions=[StepCondition(condition_type=ConditionType.ALWAYS)]
            ),
            success_criteria=visible_login_window,
            safety_tier=0,
            depends_on=[],
        ),
    ]

    if settings.typing_enabled and credentials.get("userid_input"):
        steps.append(
            WorkflowStep(
                step_id="type_userid",
                step_type="type",
                target_key="userid_input",
                target_description=targets["userid_input"].description,
                preconditions=visible_login_window,
                success_criteria=ConditionGroup(
                    conditions=[
                        StepCondition(
                            condition_type=ConditionType.TEXT_APPEARED,
                            target_key="userid_input",
                            expected_text=credentials["userid_input"],
                            verify_method="typed_context",
                        ),
                    ]
                ),
                depends_on=["ensure_login_window"],
                input_text=credentials["userid_input"],
            )
        )

    if settings.typing_enabled and credentials.get("password_input"):
        password_dependencies = ["ensure_login_window"]
        if settings.typing_enabled and credentials.get("userid_input"):
            password_dependencies.append("type_userid")

        steps.append(
            WorkflowStep(
                step_id="type_password",
                step_type="type",
                target_key="password_input",
                target_description=targets["password_input"].description,
                preconditions=visible_login_window,
                success_criteria=ConditionGroup(
                    conditions=[
                        StepCondition(
                            condition_type=ConditionType.MASKED_TEXT_PRESENT,
                            target_key="password_input",
                            verify_method="typed_context",
                        ),
                    ]
                ),
                depends_on=password_dependencies,
                idempotent=False,
                input_text=credentials["password_input"],
                redact_input_text=True,
            )
        )

    click_dependencies = ["ensure_login_window"]
    if any(step.step_id == "type_password" for step in steps):
        click_dependencies.append("type_password")
    elif any(step.step_id == "type_userid" for step in steps):
        click_dependencies.append("type_userid")

    steps.append(
        WorkflowStep(
            step_id="click_login_button",
            step_type="click",
            target_key="login_button",
            target_description=targets["login_button"].description,
            preconditions=visible_login_window,
            success_criteria=ConditionGroup(
                conditions=[StepCondition(condition_type=ConditionType.ALWAYS)]
            ),
            depends_on=click_dependencies,
        )
    )
    steps.append(
        WorkflowStep(
            step_id="verify_updater_window",
            step_type="verify_only",
            target_description="verify that the updater or main RCS window appeared after login",
            preconditions=ConditionGroup(
                conditions=[StepCondition(condition_type=ConditionType.ALWAYS)]
            ),
            success_criteria=ConditionGroup(
                group_type=ConditionGroupType.ANY,
                conditions=[
                    StepCondition(
                        condition_type=ConditionType.WINDOW_APPEARED,
                        title_prefix=RCS_UPDATER_WINDOW_TITLE_PREFIX,
                        verify_method="window_title",
                    ),
                    StepCondition(
                        condition_type=ConditionType.WINDOW_APPEARED,
                        title_prefix=RCS_MAIN_WINDOW_TITLE_PREFIX,
                        verify_method="window_title",
                    ),
                ]
            ),
            safety_tier=1,
            depends_on=["click_login_button"],
        )
    )
    return steps


def _ensure_login_window_context(context: dict) -> tuple[object | None, str, str]:
    """현재 로그인 창 context 를 갱신한다."""
    login_window, window_title, backend = find_login_window()
    context["login_window"] = login_window
    context["window_title"] = window_title
    context["backend"] = backend
    context["login_window_visible"] = login_window is not None
    context["process_alive"] = login_window is not None
    return login_window, window_title, backend


def _capture_login_window(context: dict, step: WorkflowStep) -> tuple[object, str, str, object] | tuple[None, str, str, None]:
    """로그인 창을 foreground 로 올리고 캡처한다."""
    login_window, window_title, backend = _ensure_login_window_context(context)
    if login_window is None:
        return None, window_title, backend, None

    if not callable(activate_window) or not callable(foreground_window):
        print("[ERROR] window_utils unavailable - 창 활성화/foreground 불가")
        return login_window, window_title, backend, None

    if not activate_window(
        login_window,
        debug_label=f"{step.step_id} activate backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 활성화 실패: step={step.step_id}")
        return login_window, window_title, backend, None

    if not foreground_window(
        login_window,
        debug_label=f"{step.step_id} foreground backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 foreground 실패: step={step.step_id}")
        return login_window, window_title, backend, None

    if not callable(capture_window):
        print("[ERROR] capture_window unavailable - 스크린샷 캡처 불가")
        return login_window, window_title, backend, None

    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        return login_window, window_title, backend, None

    context["last_captured_image"] = image
    context["window_title_before"] = window_title
    return login_window, window_title, backend, image


def _maybe_save_capture(context: dict, filename: str, image, *, allow_save: bool) -> str | None:
    """run_dir 가 있으면 JPEG artifact 를 저장한다."""
    if not allow_save or image is None:
        return None

    run_dir = context.get("run_dir")
    if not isinstance(run_dir, Path):
        return None

    output_path = run_dir / filename
    save_debug_jpeg(image, output_path)
    return str(output_path)


def _clear_and_type(text: str, target_key: str, settings: WorkflowSettings) -> bool:
    """선택된 입력창 내용을 backspace 로 지운 뒤 새 문자열을 입력한다."""
    if not settings.action_enabled or not PYNPUT_KEYBOARD_AVAILABLE:
        if target_key == "password_input":
            print(
                "[INFO] [DRY-RUN] 비밀번호 입력 시퀀스 생략: "
                f"target={target_key}, chars={len(text)}, action_enabled={settings.action_enabled}"
            )
        else:
            print(
                f"[INFO] [DRY-RUN] 입력 시퀀스 생략: target={target_key}, "
                f"text={text!r}, action_enabled={settings.action_enabled}"
            )
        return True

    keyboard = KeyboardController()
    keyboard.press(Key.backspace)
    keyboard.release(Key.backspace)
    time.sleep(settings.post_type_backspace_settle_sec)

    for ch in text:
        keyboard.type(ch)
        time.sleep(settings.char_type_delay_sec)

    if target_key == "password_input":
        print(f"[INFO] 타이핑 완료: target={target_key}, chars={len(text)}")
    else:
        print(f"[INFO] 타이핑 완료: target={target_key}, text={text!r}")
    return True


def _build_base_result(
    step: WorkflowStep,
    started_at: float,
    settings: WorkflowSettings,
    *,
    status: str = "success",
    failure_class: str | None = None,
    error_message: str | None = None,
    detected_point: dict | None = None,
    screen_point: dict | None = None,
    verification_result: dict | None = None,
    before_screenshot: str | None = None,
    after_screenshot: str | None = None,
    window_title_before: str | None = None,
    window_title_after: str | None = None,
    artifact_redacted: bool | None = None,
    vlm_service_used: str = "ui-venus+mai-ui",
) -> StepResult:
    """StepResult 공통 생성기."""
    if artifact_redacted is None:
        artifact_redacted = step.redact_input_text

    return StepResult(
        step_id=step.step_id,
        status=status,
        failure_class=failure_class,
        attempt_count=1,
        strategy_used="phase1_direct",
        vlm_service_used=vlm_service_used,
        detected_point=detected_point,
        detected_bbox=None,
        screen_point=screen_point,
        verification_result=verification_result,
        before_screenshot=before_screenshot,
        after_screenshot=after_screenshot,
        error_message=error_message,
        elapsed_ms=(time.time() - started_at) * 1000,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        window_title_before=window_title_before,
        window_title_after=window_title_after,
        safe_mode=settings.safe_mode,
        artifact_redacted=artifact_redacted,
    )


def _wait_for_post_login_window(
    settings: WorkflowSettings,
) -> tuple[object | None, str, str, str]:
    """로그인 후 updater 또는 메인 RCS 창이 나타날 때까지 폴링한다."""
    deadline = time.time() + settings.login_verify_timeout_sec
    poll_interval_sec = max(0.1, settings.login_verify_poll_interval_sec)
    attempt = 0

    print(
        "[INFO] 로그인 후 창 대기 시작: "
        f"updater_prefix={RCS_UPDATER_WINDOW_TITLE_PREFIX!r}, "
        f"main_prefix={RCS_MAIN_WINDOW_TITLE_PREFIX!r}, "
        f"timeout={settings.login_verify_timeout_sec}s, "
        f"poll_interval={poll_interval_sec}s"
    )

    while time.time() < deadline:
        attempt += 1

        updater_window, updater_title, updater_backend = find_rcs_updater_window()
        if updater_window is not None:
            print(
                f"[INFO] 로그인 후 updater 창 발견 (attempt={attempt}): "
                f"title={updater_title!r}, backend={updater_backend}"
            )
            return updater_window, updater_title, updater_backend, "updater_window_found"

        main_window, main_title, main_backend = find_rcs_main_window()
        if main_window is not None:
            print(
                f"[INFO] 로그인 후 메인 창 발견 (attempt={attempt}): "
                f"title={main_title!r}, backend={main_backend}"
            )
            return main_window, main_title, main_backend, "main_window_found"

        remaining_sec = deadline - time.time()
        if remaining_sec <= 0:
            break
        time.sleep(min(poll_interval_sec, remaining_sec))

    print(
        "[WARNING] 로그인 후 창 타임아웃: "
        f"{settings.login_verify_timeout_sec}s 내 updater/main window 미발견"
    )
    return None, "", "", "post_login_window_not_found"


def execute_login_step(
    step: WorkflowStep,
    context: dict,
    settings: WorkflowSettings,
) -> StepResult:
    """로그인 워크플로 step 하나를 수행한다."""
    started_at = time.time()

    if step.step_type == "observe":
        login_window, window_title, backend = _ensure_login_window_context(context)
        if login_window is None:
            return _build_base_result(
                step,
                started_at,
                settings,
                status="failed",
                failure_class="login_window_not_found",
                error_message="먼저 open_rcs.py 로 로그인 창을 열어 두세요.",
                window_title_before=window_title,
            )

        print(f"[INFO] 로그인 창 발견: title={window_title!r}, backend={backend}")
        return _build_base_result(
            step,
            started_at,
            settings,
            window_title_before=window_title,
            window_title_after=window_title,
            vlm_service_used="",
        )

    if step.step_type == "verify_only":
        if not settings.action_enabled:
            return _build_base_result(
                step,
                started_at,
                settings,
                status="skipped",
                verification_result={"verified": False, "reason": "dry_run_skip"},
                vlm_service_used="window_title",
            )

        time.sleep(settings.post_login_wait_sec)
        post_login_window, post_login_title, _backend, verify_reason = _wait_for_post_login_window(
            settings
        )
        context["post_login_window"] = post_login_window
        context["post_login_title"] = post_login_title
        context["rcs_main_window"] = post_login_window
        context["rcs_main_title"] = post_login_title
        context["login_window_visible"] = post_login_window is None

        after_screenshot = None
        if post_login_window is not None and callable(capture_window):
            try:
                main_image = capture_window(post_login_window)
            except Exception:
                main_image = None
            after_screenshot = _maybe_save_capture(
                context,
                f"after_{step.step_id}.jpeg",
                main_image,
                allow_save=True,
            )

        if post_login_window is None:
            return _build_base_result(
                step,
                started_at,
                settings,
                status="failed",
                failure_class="verify_failed",
                error_message="로그인 버튼 클릭 후 RCS Updater 또는 메인 RCS 창 미확인",
                verification_result={"verified": False, "reason": verify_reason},
                after_screenshot=after_screenshot,
                window_title_after=post_login_title,
                vlm_service_used="window_title",
            )

        return _build_base_result(
            step,
            started_at,
            settings,
            verification_result={"verified": True, "reason": verify_reason},
            after_screenshot=after_screenshot,
            window_title_after=post_login_title,
            vlm_service_used="window_title",
        )

    login_window, window_title, backend, image = _capture_login_window(context, step)
    if login_window is None:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="login_window_not_found",
            error_message="로그인 창을 찾지 못했습니다.",
            window_title_before=window_title,
        )
    if image is None:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="capture_failed",
            error_message="로그인 창 캡처 또는 foreground 활성화 실패",
            window_title_before=window_title,
        )

    before_screenshot = _maybe_save_capture(
        context,
        f"before_{step.step_id}.jpeg",
        image,
        allow_save=not step.redact_input_text,
    )

    detect_success, targets, analyze_login_target = _load_login_targets()
    target_key = step.target_key or ""
    target_config = targets.get(target_key)
    if target_config is None:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="invalid_target",
            error_message=f"미정의 타겟: {target_key}",
            before_screenshot=before_screenshot,
            window_title_before=window_title,
        )

    print(f"[INFO] 타겟 탐지 시작: step={step.step_id}, target={target_key}")
    detection = analyze_login_target(
        login_window,
        window_title,
        backend,
        target_config,
        image=image,
    )
    if detection.exit_code != detect_success or detection.point is None:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="detect_failed",
            error_message=f"타겟 탐지 실패: {target_key}, exit_code={detection.exit_code}",
            before_screenshot=before_screenshot,
            window_title_before=window_title,
        )

    if not callable(image_point_to_screen):
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="act_failed",
            error_message="image_point_to_screen unavailable",
            detected_point=detection.point,
            before_screenshot=before_screenshot,
            window_title_before=window_title,
        )

    screen_point = image_point_to_screen(login_window, detection.point)
    if screen_point is None:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="act_failed",
            error_message=f"스크린 좌표 변환 실패: {target_key}",
            detected_point=detection.point,
            before_screenshot=before_screenshot,
            window_title_before=window_title,
        )

    context["active_target_key"] = target_key

    if step.step_type == "type":
        if not settings.action_enabled:
            if target_key == "password_input":
                print(
                    "[INFO] [DRY-RUN] 입력 step skip: "
                    f"target={target_key}, clicks=1+2, chars={len(step.input_text or '')}"
                )
            else:
                print(
                    "[INFO] [DRY-RUN] 입력 step skip: "
                    f"target={target_key}, clicks=1+2, text={(step.input_text or '')!r}"
                )
            return _build_base_result(
                step,
                started_at,
                settings,
                status="skipped",
                error_message="dry_run_skip",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
                window_title_after=window_title,
            )

        if not PYNPUT_MOUSE_AVAILABLE or not PYNPUT_KEYBOARD_AVAILABLE:
            print(
                "[INFO] 입력 step skip: "
                f"target={target_key}, pynput_mouse={PYNPUT_MOUSE_AVAILABLE}, "
                f"pynput_keyboard={PYNPUT_KEYBOARD_AVAILABLE}"
            )
            return _build_base_result(
                step,
                started_at,
                settings,
                status="skipped",
                error_message="input_device_unavailable",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
                window_title_after=window_title,
            )

        if not callable(click_at_screen):
            return _build_base_result(
                step,
                started_at,
                settings,
                status="failed",
                failure_class="act_failed",
                error_message="click_at_screen unavailable",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
            )

        click_at_screen(
            screen_point,
            target_key,
            click_count=1,
            action_enabled=settings.action_enabled,
        )
        time.sleep(settings.pre_type_click_settle_sec)
        click_at_screen(
            screen_point,
            target_key,
            click_count=2,
            action_enabled=settings.action_enabled,
        )
        time.sleep(settings.pre_type_double_click_settle_sec)
        typed_ok = _clear_and_type(step.input_text or "", target_key, settings)
        if not typed_ok:
            return _build_base_result(
                step,
                started_at,
                settings,
                status="failed",
                failure_class="act_failed",
                error_message=f"타이핑 실패: {target_key}",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
            )

        typed_values = context.setdefault("typed_values", {})
        typed_values[target_key] = step.input_text or ""
        context["focused_target_key"] = target_key
        time.sleep(settings.post_type_settle_sec)
    elif step.step_type == "click":
        if not settings.action_enabled:
            print(f"[INFO] [DRY-RUN] 클릭 step skip: target={target_key}, click_count=1")
            return _build_base_result(
                step,
                started_at,
                settings,
                status="skipped",
                error_message="dry_run_skip",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
                window_title_after=window_title,
            )

        if not PYNPUT_MOUSE_AVAILABLE:
            print(f"[INFO] 클릭 step skip: target={target_key}, pynput_mouse={PYNPUT_MOUSE_AVAILABLE}")
            return _build_base_result(
                step,
                started_at,
                settings,
                status="skipped",
                error_message="input_device_unavailable",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
                window_title_after=window_title,
            )

        if not callable(click_at_screen):
            return _build_base_result(
                step,
                started_at,
                settings,
                status="failed",
                failure_class="act_failed",
                error_message="click_at_screen unavailable",
                detected_point=detection.point,
                screen_point=screen_point,
                before_screenshot=before_screenshot,
                window_title_before=window_title,
            )

        time.sleep(settings.pre_click_settle_sec)
        click_at_screen(
            screen_point,
            target_key,
            click_count=1,
            action_enabled=settings.action_enabled,
        )
        time.sleep(settings.post_click_settle_sec)
        if target_key == "login_button":
            context["login_submit_attempted"] = True
    else:
        return _build_base_result(
            step,
            started_at,
            settings,
            status="failed",
            failure_class="unsupported_step_type",
            error_message=f"미지원 step_type: {step.step_type}",
            detected_point=detection.point,
            screen_point=screen_point,
            before_screenshot=before_screenshot,
            window_title_before=window_title,
        )

    after_screenshot = None
    if not step.redact_input_text and callable(capture_window):
        try:
            after_image = capture_window(login_window)
        except Exception:
            after_image = None
        after_screenshot = _maybe_save_capture(
            context,
            f"after_{step.step_id}.jpeg",
            after_image,
            allow_save=after_image is not None,
        )

    return _build_base_result(
        step,
        started_at,
        settings,
        detected_point=detection.point,
        screen_point=screen_point,
        before_screenshot=before_screenshot,
        after_screenshot=after_screenshot,
        window_title_before=window_title,
        window_title_after=window_title,
    )


def run_login_workflow(settings: WorkflowSettings | None = None):
    """RCS 로그인 워크플로를 실행한다."""
    resolved_settings = settings or load_workflow_settings()
    credentials = load_login_credentials()
    steps = build_login_workflow_steps(resolved_settings, credentials)

    context = {
        "typed_values": {},
        "process_exe_name": "RcsMainHD.exe",
    }
    runner = WorkflowRunner(
        resolved_settings,
        workflow_name="rcs_login",
        log_name=LOG_NAME,
        component_name=COMPONENT_NAME,
    )

    log_work2_event(
        component=COMPONENT_NAME,
        message="workflow_entry_started",
        log_name=LOG_NAME,
        step_ids=[step.step_id for step in steps],
        safe_mode=resolved_settings.safe_mode,
        action_enabled=resolved_settings.action_enabled,
        typing_enabled=resolved_settings.typing_enabled,
    )
    run = runner.run(
        steps,
        context,
        lambda step, step_context: execute_login_step(step, step_context, resolved_settings),
    )
    return run


def main() -> str:
    """워크플로 기반 로그인 실행 entrypoint."""
    started_at = time.time()
    run = run_login_workflow()

    if run.status == "completed":
        print(f"[INFO] 로그인 워크플로 완료: run_dir={run.run_dir}")
        print(f"[INFO] 총 소요={format_elapsed_ms(started_at)}")
        return EXIT_SUCCESS

    print(
        f"[WARNING] 로그인 워크플로 중단: status={run.status}, "
        f"last_step_index={run.current_step_index}, run_dir={run.run_dir}"
    )
    print(f"[INFO] 총 소요={format_elapsed_ms(started_at)}")
    return EXIT_WORKFLOW_ABORTED


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
