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
  * CycleNotifier.notify_outcome — status != corrected 면 cube 알림(outcome 요약)
  * engineer watch — 미보정 시 창이 닫히거나 watch 시간이 끝날 때까지 대기
    (녹화 스레드가 엔지니어 수동 조작을 계속 캡처)
  * finally: 결과 통보 backstop → 녹화 중지(manifest) → tool 닫기 → 알림 팝업 backstop

알림 순서 규약(결과-후-알림): 감지 즉시 cube 를 보내지 않는다. 접속 → 판정 →
보정 시도까지 끝낸 뒤 결과를 통보한다 — 알림을 보고 즉시 반응하는 엔지니어가
단일 RCS 커서를 두고 자동화와 경합하면 사이클이 `rcs_occupied` 로 깨지기 때문이다.
그 대가로 "사이클이 죽으면 아무 통보도 없다" 는 위험이 생기므로 두 장치를 둔다:
  * CycleNotifier 게이트 — 본문과 finally 가 모두 통보를 시도하고 1회만 발송.
  * watchdog — `notify_delay_sec` 를 넘기면 '진행 중' 을 1회 고지(무한 침묵 금지).
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR, DEBUG_IMAGE_DIR
from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.cycle_images import gather_and_report
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted
from poc.workflow_3.monitor.cycle_report import print_cycle_report
from poc.workflow_3.monitor.notify import (
    CORRECTED_UNVERIFIED,
    VIEW_ONLY_OBSERVATION,
    CycleNotifier,
    close_alert_window,
    notify_correction_outcome,
)
from poc.workflow_3.monitor.rcs_recovery import RECOVERED, recover_rcs_session
from poc.workflow_3.monitor.recording import RecordingSession
from poc.workflow_3.monitor.teardown import run_teardown
from poc.workflow_3.rcs.row_occupant import OCCUPIED_BY_OTHER, UNKNOWN
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
    capture_screen,
    capture_window,
    click_at_screen,
    collect_window_rows,
    env_float,
    find_window_by_title_prefix,
    image_point_to_screen,
    make_timestamp_tag,
    move_cursor_to_screen,
    scroll_at_screen,
)

LOG_COMPONENT = "align_fail_cycle"

# workflow_4 live graph view import 실패 경고를 한 번만 출력하기 위한 플래그.
_MIRROR_GRAPH_VIEW_WARNED = False

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
    prelude_dir: str = ""       # 접속 구간 화면 녹화(시연용, 기본 off).
    prelude_frame_count: int = 0
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


def _scan_rcs_processes(exe_path):
    """RCS 프로세스 목록을 조회한다 - **조회 자체가 불가하면 None('모름')**.

    `find_existing_rcs_processes` 는 psutil 이 없으면 빈 리스트를 돌려주므로 "안 돌고
    있다" 와 "알 수 없다" 가 구분되지 않는다. 그대로 넘기면 중복 실행 가드가 조용히
    무력화되므로 여기서 두 경우를 갈라 준다.
    """
    from poc.workflow_3.rcs.open_rcs import PSUTIL_AVAILABLE, find_existing_rcs_processes

    if not PSUTIL_AVAILABLE:
        # 이 한 줄이 없으면 "RCS 를 왜 안 띄웠나" 가 로그에서 사라진다 - 조회 불가는
        # 곧 재실행 보류이므로, 원인(psutil 미설치)을 여기서 이름으로 짚어 준다.
        print("[WARNING] psutil 미설치 - RCS 프로세스 조회 불가(재실행 보류). `uv sync` 필요.")
        return None
    try:
        return find_existing_rcs_processes(exe_path)
    except Exception as exc:
        print(f"[WARNING] RCS 프로세스 조회 실패(모름으로 처리): {exc}")
        return None


def _list_process_windows(pid):
    """PID 가 가진 보이는 top-level 창 목록. 조회 불가면 None(=판단 보류).

    창 없는 좀비 RCS 를 가려내기 위한 것이다. 조회가 안 될 때 빈 리스트를 주면 "창이
    없다"= 좀비로 오판해 멀쩡한 프로세스를 죽일 수 있으므로 예외를 올려 상위가
    '살아 있음' 으로 처리하게 둔다(classify_existing_processes 참고).
    """
    from poc.workflow_3.util.window_utils import collect_window_rows

    return collect_window_rows(process_id=int(pid))


def _terminate_process(pid) -> None:
    """PID 를 종료한다. psutil 이 없으면 예외를 올려 상위가 실패로 기록하게 한다."""
    import psutil

    psutil.Process(int(pid)).terminate()


def _open_list_tab(window, title, backend) -> bool:
    """메인 창의 List 탭을 클릭한다. 성공 여부만 돌려주고 예외는 올리지 않는다.

    복구 로그인 직후에만 쓴다 - 왜 필요한지는 `_exec_ensure_rcs_ready` 참고.
    """
    from poc.workflow_3.rcs.view_list_tab_rcs import (
        EXIT_SUCCESS,
        click_list_tab_in_main_window,
    )

    return click_list_tab_in_main_window(window, title, backend).exit_code == EXIT_SUCCESS


def _exec_ensure_rcs_ready(step, context, settings: Workflow3Settings) -> StepResult:
    """① RCS 메인 창 확보 — 떠 있으면 전면화, 없으면(옵션) 재실행+재로그인.

    복구 자체는 `monitor.rcs_recovery.recover_rcs_session` 이 하고 여기서는 협력자를
    묶어 넘기고 결과를 StepResult 로 옮긴다.

    **복구를 탄 경우에만 List 탭을 연다.** 다음 step 의 `connect_to_tool` 은 "현재
    List 탭에서" tool 행을 찾는다고 가정하는데, 복구 로그인은 `target_tool_name=""`
    으로 부르므로 `workflow_login.build_login_workflow_steps` 의 List 클릭 step 이 아예
    안 붙는다 - `click_list_tab`/`verify_list_tab_opened`/`open_target_tool` 이 전부
    `if normalized_tool_name:` 한 블록에 묶여 있어, "tool 은 열지 않는다" 는 계약이
    List 탭까지 같이 꺼 버린다. 복구를 안 탄 정상 경로에서는 누르지 않는다 - RCS 가
    이미 List 를 띄운 채 떠 있는 것이 정상이고, 알람마다 탭을 다시 누르면 VLM 왕복과
    클릭이 공짜로 늘 뿐 아니라 로케이터가 어긋나면 멀쩡하던 화면을 망친다.
    """
    started_at = time.time()
    window, title, backend = wait_for_rcs_main_window(timeout_sec=settings.connect_window_timeout_sec)
    if window is None and settings.rcs_recovery_enabled:
        print("[WARNING] RCS 메인 창 없음 - 재실행+재로그인 복구 시도(ALIGN_FAIL_RCS_RECOVERY=on)")
        from poc.workflow_3.rcs.open_rcs import launch_rcs
        from poc.workflow_3.rcs.workflow_login import run_login_workflow

        recovery = recover_rcs_session(
            settings,
            find_processes_fn=_scan_rcs_processes,
            launch_fn=launch_rcs,
            login_fn=run_login_workflow,
            wait_window_fn=wait_for_rcs_main_window,
            list_windows_fn=_list_process_windows,
            terminate_fn=_terminate_process,
        )
        if recovery.status != RECOVERED:
            return _make_result(
                step, "failed", started_at, settings,
                failure_class=recovery.status, error_message=recovery.error,
            )
        # 재실행까지 했는지 남긴다 - "로그인만 다시 했다" 와 "프로세스를 새로 띄웠다" 는
        # 오피스에서 원인이 다르다(전자는 세션 만료, 후자는 클라이언트 종료).
        print(
            f"[INFO] RCS 복구 성공: relaunched={recovery.launched} title={recovery.title!r}"
        )
        window, title, backend = recovery.window, recovery.title, recovery.backend
        # 재로그인 직후엔 List 탭이 안 열려 있다(위 docstring). 실패해도 step 은 죽이지
        # 않는다 - 창은 확보됐고, List 가 이미 열려 있었을 수도 있으며, 실제 판정은
        # 다음 step 의 connect 가 한다. 여기서 알람을 통째로 버리는 편이 더 비싸다.
        try:
            if _open_list_tab(window, title, backend):
                print("[INFO] 복구 후 List 탭 열기 완료")
            else:
                print("[WARNING] 복구 후 List 탭 열기 실패 - connect 단계에서 재판정")
        except Exception as exc:
            print(f"[WARNING] 복구 후 List 탭 열기 예외(connect 단계에서 재판정): {exc}")
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


def _read_row_tokens(image, box):
    """점유자 컬럼 crop 을 OCR 로 읽는다. 판독 실패는 None(= UNKNOWN 신호)."""
    from poc.workflow_3.vlm.label_verify import read_text_near_point, tokens_from_text

    read = read_text_near_point(
        image, box,
        debug_image_dir=DEBUG_IMAGE_DIR / "row_occupant",
        timestamp_tag=make_timestamp_tag(time.time()),
        artifact_label="row_occupant",
        log_name="row_occupant",
    )
    if not read.ok:
        return None
    return tokens_from_text(read.raw_text)


def _exec_connect_tool(step, context, settings: Workflow3Settings) -> StepResult:
    """③ tool 더블클릭 접속 — 알람당 1회만 느슨하게 시도(실패 시 엔지니어 직접).

    접속 직전 List 를 캡처해 두었다가, 행 좌표가 확정되면 그 자리의 점유자 컬럼을 읽어
    `context["occupancy"]` 를 채운다. 이미 승낙되어 팝업 없이 들어가는 경우(b)를
    잡기 위한 것이다 - 그 세션은 관전만 가능한데 겉보기에는 정상 접속과 같다.

    메인 창은 여기서 **한 번만** 찾아 `connect_to_tool` 에 넘긴다. 예전처럼 양쪽이 각자
    찾으면 창 열거와 포커스 활성화가 매 알람마다 두 번 일어나, 이 프로젝트에서 이미
    까다로운 foreground 경합을 공짜로 한 번 더 만든다.
    """
    from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    started_at = time.time()
    eqp_id = context["eqp_id"]
    action_enabled = settings.action_enabled and settings.connect_action_enabled

    main_window, main_title, main_backend = wait_for_rcs_main_window(
        timeout_sec=settings.connect_window_timeout_sec,
    )
    # 좌표와 이미지가 같은 순간의 것이어야 crop 이 맞는다. 접속 후에는 tool 창이 List 를
    # 덮을 수 있어 다시 잡을 수 없으므로 더블클릭 전에 찍어 둔다.
    list_image = None
    if main_window is not None:
        try:
            list_image = capture_window(main_window)
        except Exception as exc:
            print(f"[WARNING] 점유자 판독용 List 캡처 실패(점유 미상으로 진행): {exc}")

    try:
        result = connect_to_tool(
            eqp_id,
            action_enabled=action_enabled,
            main_window_timeout_sec=settings.connect_window_timeout_sec,
            main_window=main_window,
            main_window_title=main_title,
            main_window_backend=main_backend,
        )
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="connect_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    double_clicked = bool(getattr(result, "double_clicked", False))
    context["connect_result"] = result
    occupancy = read_occupancy(
        list_image,
        getattr(result, "tool_point_on_full_image", None),
        read_tokens_fn=_read_row_tokens,
    )
    context["occupancy"] = occupancy
    print(f"[INFO] List 점유 판독: occupancy={occupancy} (EQP_ID={eqp_id})")
    if not double_clicked:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="connect_not_clicked",
            error_message=f"tool 더블클릭 미수행(dry-run/인식 실패): action_enabled={action_enabled}",
        )
    return _make_result(step, "success", started_at, settings)


def _detect_wrong_tool_window(eqp_id: str) -> str:
    """목표 tool 이 아닌 Remote Monitoring 창이 떠 있으면 닫고 그 제목을 돌려준다.

    List 에서 옆 행을 더블클릭하면 엉뚱한 tool 창이 열리는데, teardown 의
    `close_tool(eqp_id)` 는 제목에 우리 eqp_id 가 있어야만 닫으므로 그 창은 그대로
    남는다(세션 누수). 여기서 발견 즉시 닫는다.

    닫기 실패나 예외는 삼킨다 - 이 함수는 진단/청소용이고 접속 실패 판정을 막으면 안 된다.
    """
    try:
        from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
        from poc.workflow_3.util import close_window
    except ImportError as exc:
        print(f"[WARNING] 오클릭 창 검사 불가(import 실패): {exc}")
        return ""

    try:
        stray_window, stray_title, _backend = find_remote_monitoring_window("")
    except Exception as exc:
        print(f"[WARNING] 오클릭 창 검사 실패: {exc}")
        return ""

    if stray_window is None or not stray_title:
        return ""
    if (eqp_id or "").strip().lower() in stray_title.lower():
        # 제목에 우리 ID 가 있는데 위에서 못 찾았다면 타이밍 문제 - 오클릭이 아니다.
        return ""

    print(f"[WARNING] 오클릭 감지: 다른 tool 창이 열림 title={stray_title!r} (target={eqp_id})")
    if close_window is not None:
        try:
            close_window(stray_window, debug_label=f"wrong_tool_opened {stray_title!r}")
        except Exception as exc:
            print(f"[WARNING] 오클릭 창 닫기 실패(수동 확인 필요): {exc}")
    return stray_title


def _find_tool_window(eqp_id: str):
    """제목에 eqp_id 를 가진 Remote Monitoring 창을 1회 탐색한다.

    찾으면 `(window, title, backend)` 를, 없으면 None 을 돌려준다. 튜플을 그대로 넘기는
    이유는 호출부가 같은 창을 다시 찾지 않게 하기 위해서다.
    """
    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window

    window, title, backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        return None
    return window, title, backend


def _close_select_popup() -> None:
    """점유 'select' 팝업을 닫는다 - 좌표 클릭이 아니라 창 핸들로.

    Cancel 을 좌표로 누르는 것은 로케이션이 방금 실패했을 수도 있는 시점에 같은 팝업을
    다시 겨냥하는 것이라, 확인 게이트를 통과하지 않은 클릭을 최악의 순간에 내보내게
    된다. 창 핸들은 이미 있으므로 VLM 도 좌표도 필요 없다.
    """
    from poc.workflow_3.monitor.occupied_popup import find_select_popup_window
    from poc.workflow_3.util import close_window

    try:
        popup = find_select_popup_window()
        if popup is not None and close_window is not None:
            close_window(popup, debug_label="select popup")
    except Exception as exc:
        print(f"[WARNING] select 팝업 닫기 실패(수동 확인 필요): {exc}")


def _run_share_request(settings: Workflow3Settings, tag: str):
    """share_request 의 주입점을 실제 VLM/OCR/클릭 구현으로 채워 호출한다.

    확인 실패 시 crop 과 OCR 원문이 debug_images 에 남는다 - strict 를 기본값으로 두는
    대가이며, Mac 에서는 팝업을 볼 수 없어 이 산출물이 실제 문구를 아는 유일한 경로다.
    """
    from poc.workflow_3.monitor.occupied_popup import (
        SELECT_TITLE,
        find_select_popup_window,
    )
    from poc.workflow_3.monitor.share_request import request_screen_share
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import analyze_window_target

    debug_dir = DEBUG_IMAGE_DIR / "share_request" / tag

    def _locate(image, target):
        result = analyze_window_target(
            None, SELECT_TITLE, "uia", target,
            debug_image_dir=debug_dir,
            log_name="share_request",
            component_name="share_request",
            artifact_prefix=target.key,
            image=image,
        )
        return result.point

    def _read_tokens(image, point, key):
        box = crop_box_around_point(
            point, image.width, image.height,
            left_ratio=0.30, right_ratio=0.30, half_height_ratio=0.06,
        )
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(time.time()),
            artifact_label=key,
            log_name="share_request",
        )
        return tokens_from_text(read.raw_text) if read.ok else []

    def _click(window, image, point, key):
        """이미지 픽셀 좌표를 스크린 좌표로 변환해 클릭한다.

        변환을 빼면 확인 게이트가 무의미해진다 - 라벨은 점 A 에서 읽고 클릭은 점 B 에
        떨어져, 오피스 125/150% 배율에서 하필 옆 라디오(강제 종료)를 누를 수 있다.
        변환 실패는 클릭하지 않고 예외로 올린다(fail-closed).
        """
        screen = image_point_to_screen(window, point, image_size=image.size)
        if screen is None:
            raise RuntimeError(f"공유 팝업 좌표 변환 실패: {key} point={point}")
        print(
            f"[INFO] 공유 팝업 클릭: {key} px={point} -> screen={screen}"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        click_at_screen(
            screen, f"share_{key}", action_enabled=settings.action_enabled
        )

    return request_screen_share(
        settings,
        locate_fn=_locate,
        read_tokens_fn=_read_tokens,
        click_fn=_click,
        capture_fn=capture_window,
        find_popup_fn=find_select_popup_window,
    )


def _handle_occupied_popup(step, context, settings: Workflow3Settings, started_at):
    """점유 'select' 팝업 감지 후 화면 공유를 요청하고 결과에 따라 분기한다.

    승낙되면 관전(view-only) 세션이므로 `context["occupancy"]` 를 점유로 확정한다 -
    이후 `_exec_run_correction` 이 보정을 건너뛰고 녹화만 하게 하는 신호다.
    """
    from poc.workflow_3.monitor.share_request import (
        ACCEPTED,
        STATUS_CONFIRM_FAILED,
        STATUS_REQUESTED,
        wait_share_response,
    )

    eqp_id = context["eqp_id"]

    if not settings.share_request_enabled:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="rcs_occupied_select",
            error_message=(
                "점유 'select'(공유/종료) 팝업 감지 - 공유 요청 비활성(설정), 접속 포기. "
                "다른 사용자 해제 후 cooldown 지나면 재시도."
            ),
        )

    def _give_up(share_result):
        """팝업을 닫고 실패 결과를 만든다 - 모든 포기 경로의 유일한 출구."""
        _close_select_popup()
        failure_class = (
            "rcs_share_confirm_failed"
            if share_result.status == STATUS_CONFIRM_FAILED
            else "rcs_occupied_select"
        )
        return _make_result(
            step, "failed", started_at, settings,
            failure_class=failure_class,
            error_message=(
                f"화면 공유 요청 결과: {share_result.status} "
                f"(radio={share_result.radio_verdict or '-'}, "
                f"button={share_result.button_verdict or '-'}"
                f"{', error=' + share_result.error if share_result.error else ''})"
            ),
        )

    share = _run_share_request(settings, context["tag"])
    if share.status != STATUS_REQUESTED:
        return _give_up(share)

    responded, found = wait_share_response(
        eqp_id, settings.share_wait_sec, find_window_fn=_find_tool_window
    )
    if responded != ACCEPTED or found is None:
        return _give_up(share)

    # 대기 루프가 이미 찾은 창을 그대로 쓴다 - 다시 찾으면 전체 창 열거와 포커스
    # 활성화가 한 번 더 일어난다(이미 전면에 있는 창을 낚아채는 셈).
    window, title, backend = found
    # 화면 공유는 관전만 가능하다. 여기서 확정해 두지 않으면 보정이 먹지 않는
    # 클릭을 하고도 'corrected' 로 보고한다.
    context["occupancy"] = OCCUPIED_BY_OTHER
    context["tool_window"] = window
    context["tool_window_title"] = title
    context["tool_window_backend"] = backend
    mode = ("보정 진행(설정)" if settings.correct_when_occupied else "관전·녹화만 수행")
    print(f"[INFO] 화면 공유 세션 진입 - {mode}: EQP_ID={eqp_id}")
    return _make_result(step, "success", started_at, settings)


def _exec_wait_tool_window(step, context, settings: Workflow3Settings) -> StepResult:
    """④ tool 창 대기 — RCS 점유(select 팝업) 시 화면 공유를 요청한다.

    점유 검출(opt-in): 매 시도 전 detect_select_popup 으로 'select' 팝업을 조기 감지해,
    떠 있으면 창 탐색을 더 돌지 않는다. 검출은 제목 열거 + (가능하면) VLM 확인의
    hybrid 이며, 실패해도 접속을 막지 않는다(전체 흐름은 기존 '미발견 → rcs_occupied'
    로 폴백).

    팝업이 감지되면 `_handle_occupied_popup` 이 화면 공유를 요청한다(2026-08-18). 승낙
    되면 관전 세션으로 진입해 엔지니어의 수동 작업을 녹화하고, 거절/무응답이면 기존처럼
    포기해 상위 루프가 cooldown 후 재시도한다.
    """
    started_at = time.time()
    eqp_id = context["eqp_id"]

    occupied = {"select": False}
    abort_check = None
    popup_client = None
    if settings.occupied_popup_detect_enabled:
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
        return _handle_occupied_popup(step, context, settings, started_at)
    if window is None:
        # 목표 tool 창이 없을 때, **다른** tool 의 Remote Monitoring 창이 떠 있으면
        # 그건 점유가 아니라 우리가 List 에서 옆 행을 더블클릭한 것이다(오클릭).
        # 이 둘은 증상이 같아 지금까지 전부 rcs_occupied 로 보고돼 구분이 안 됐다.
        stray_title = _detect_wrong_tool_window(eqp_id)
        if stray_title:
            return _make_result(
                step, "failed", started_at, settings,
                failure_class="wrong_tool_opened",
                error_message=(
                    f"목표 tool 창은 없고 다른 tool 창이 열림: title={stray_title!r} "
                    f"(target={eqp_id}) - List 오클릭. 열린 창은 닫았음."
                ),
            )
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


def start_prelude_recording(context: dict, settings: Workflow3Settings):
    """접속 구간(RCS 실행/로그인/tool 진입)을 **화면 전체**로 녹화한다 (기본 off).

    본 녹화는 tool 창 rect 를 찍으므로 창이 생기기 전 구간은 원리상 프레임이 없다.
    시연 영상이 "RCS 를 열어 tool 로 들어가는 장면"부터 시작하려면 그 구간만은
    창이 아니라 화면을 찍어야 한다.

    저장 위치는 본 녹화 폴더의 **하위** `prelude/` 다 - recording_filter 는
    `.../recording/*.jpg` 를 비재귀로 훑으므로, 창 rect 를 전제한 그 파이프라인에
    화면 전체 프레임이 섞여 들어가지 않는다.

    녹화 실패는 사이클을 죽이지 않는다(시연 보조물이지 보정 경로가 아니다).
    """
    if not settings.record_prelude_enabled:
        return None
    out_dir = _recording_dir_for(
        context["eqp_id"], context["recipe_id"], context["tag"]
    ) / "prelude"
    monitor_index = settings.prelude_monitor_index
    try:
        session = RecordingSession(
            None,
            out_dir,
            tag=f"{context['tag']}_pre",
            poll_sec=settings.prelude_poll_sec,
            heartbeat_sec=settings.recording_heartbeat_sec,
            change_min_px=settings.recording_change_min_px,
            max_sec=settings.prelude_max_sec,
            max_disk_mb=settings.prelude_max_disk_mb,
            jpeg_quality=settings.prelude_jpeg_quality,
            capture_fn=lambda: capture_screen(monitor_index),
            capture_source="screen",
        ).start()
        context["prelude_recording"] = session
        print(f"[INFO] 접속 구간 화면 녹화 시작(시연용): {out_dir}")
        return session
    except Exception as exc:
        print(f"[WARNING] prelude 녹화 시작 실패(사이클은 계속): {exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="prelude_start_failed", level="warning",
            error=str(exc),
        )
        return None


def stop_prelude_recording(context: dict, reason: str) -> int:
    """prelude 녹화를 멈춘다 (없으면 0). 두 번 불러도 안전 - 두 번째는 0.

    정상 흐름에선 tool 창이 뜬 시점에 멈추고, teardown 은 접속이 깨진 경로만 줍는다.
    그래서 결과 요약은 **여기서** context 에 적어 둔다 - 인계 시점에 멈춘 세션의
    프레임 수를 teardown 이 다시 알아낼 방법이 없기 때문이다.
    """
    session = context.pop("prelude_recording", None)
    if session is None:
        return 0
    frames = session.stop(reason)
    context["prelude_dir"] = str(session.out_dir)
    context["prelude_frame_count"] = len(frames)
    return len(frames)


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
        # tool 창 녹화가 떴으니 화면 전체 녹화는 여기서 끝난다 - 두 세션이 겹쳐 돌면
        # 같은 장면을 두 번 저장하고 폴링도 두 배가 된다.
        handed_over = stop_prelude_recording(context, "tool_window_open")
        if handed_over:
            print(f"[INFO] 접속 구간 화면 녹화 종료 - {handed_over} 프레임 인계")
    except Exception as exc:
        print(f"[WARNING] 녹화 시작 실패(사이클은 계속): {exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="recording_start_failed", level="warning",
            error=str(exc),
        )
    return _make_result(step, "success", started_at, settings)


def _exec_locate_sem_panel(step, context, settings: Workflow3Settings) -> StepResult:
    """⑥ SEM panel ROI → RCSSEMMonitor.

    panel ROI 는 live SEM box(VLM, check-only 에서 검증된 경로)를 우선 쓰고 실패 시
    landmark 로 폴백한다. 같은 검출의 PM 판독으로 OM/SEM 도 확정하는데, 판독이 모호하면
    (require_pm_mode) 보정을 진행하지 않는다 — 틀린 modality 는 틀린 template 을 뜻하고,
    그 좌표로 recenter 하면 화면이 엉뚱한 곳으로 간다.
    """
    started_at = time.time()
    controller_action = settings.action_enabled and not settings.correction_dry_run

    sem_box_client = None
    if settings.sem_box_detect_enabled:
        try:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            sem_box_client = Workflow1VLMClient(settings.sem_box_vlm_service)
        except Exception as exc:
            print(f"[WARNING] SEM box VLM 클라이언트 생성 실패 - landmark 경로만 시도: {exc}")

    # 실패 사유와 그때 본 화면을 사이클 debug 폴더에 남긴다. 이 폴더는 테이크 수집
    # (cycle_images)의 tag 키 소스라, 실패 화면이 자동으로 gathered/ 번들에 들어간다.
    panel_reasons: list = []
    fail_frame_path = (
        DEBUG_IMAGE_DIR / "align_fail_cycle" / context["tag"] / "sem_box_fail.jpg"
    )
    try:
        controller = build_rcs_sem_monitor(
            context["tool_window"],
            vlm_client=sem_box_client,
            pm_two_stage=settings.pm_two_stage_ocr_enabled,
            reason_sink=panel_reasons,
            fail_frame_path=fail_frame_path,
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
            # 정적 문자열이면 4가지 원인(클라이언트 부재/예외/미검출/박스이상)이 저널에서
            # 구분되지 않는다. build_rcs_sem_monitor 가 채운 실제 사유를 그대로 적는다.
            error_message=(
                f"SEM panel 확보 실패 - {panel_reasons[0]}" if panel_reasons
                else "SEM panel 확보 실패 - 사유 미기록"
            ),
        )
    if settings.require_pm_mode and not getattr(controller, "mode_hint", None):
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="pm_mode_unknown",
            error_message="PM 박스에서 OM/SEM 을 확정하지 못함 - 잘못된 template 매칭 방지 위해 보정 보류",
        )
    context["controller"] = controller
    context["sem_box_client"] = sem_box_client
    return _make_result(step, "success", started_at, settings)


def resolve_correction_outcome_status(
    occupancy: str, status: str, *, attempted: bool = True
) -> str:
    """점유 상태를 반영해 최종 outcome status 를 정한다.

    attempted=False    : 보정을 아예 시도하지 않았다(점유 관전). 관전 status.
    점유/불명 + corrected: 클릭이 실제로 먹었는지 확인할 수 없다. `correct_align_fail_auto`
                         는 open-loop 라 반영 여부를 화면으로 되읽지 않으므로, 여기서
                         'corrected' 로 두면 cube 가 생략되어(notify.py) 아무도 모르는
                         미보정이 남는다. 조용한 성공을 만들지 않는 것이 요점이다.
                         점유 중 보정(correct_when_occupied)은 화면 공유가 원래 view-only
                         라 이 불확실성이 unknown 보다 오히려 크므로 같은 강등을 받는다.
    그 외              : 그대로 둔다(실패/인계 경로가 담은 정보를 덮어쓰지 않는다).
    """
    if occupancy == OCCUPIED_BY_OTHER and not attempted:
        return VIEW_ONLY_OBSERVATION
    if occupancy in (UNKNOWN, OCCUPIED_BY_OTHER) and status == "corrected":
        return CORRECTED_UNVERIFIED
    return status


def _exec_run_correction(step, context, settings: Workflow3Settings) -> StepResult:
    """⑦ CV 보정 — RECIPE_ID 가 있고 correction_enabled 일 때만.

    다른 엔지니어가 점유 중이면(화면 공유 = view-only) 클릭이 장비에 먹지 않으므로
    보정을 시도하지 않고 관전만 한다. 점유 여부를 확정하지 못한 경우에는 보정을 하되
    결과 status 를 강등해 알림이 반드시 나가게 한다.
    """
    started_at = time.time()
    eqp_id, recipe_id = context["eqp_id"], context["recipe_id"]
    occupancy = context.get("occupancy", UNKNOWN)
    if occupancy == OCCUPIED_BY_OTHER and settings.correct_when_occupied:
        # 엔지니어와 조율해 제어를 넘겨받은 상황. 클릭이 먹었는지는 여전히 되읽지
        # 못하므로 결과는 아래 resolve 에서 corrected_unverified 로 강등된다.
        print("[WARNING] 다른 엔지니어 점유 중이지만 보정 진행 "
              "(ALIGN_FAIL_CORRECT_WHEN_OCCUPIED=1) - 클릭이 안 먹을 수 있음")
    elif occupancy == OCCUPIED_BY_OTHER:
        from poc.workflow_3.align.correction import CorrectionOutcome

        print("[INFO] 다른 엔지니어 점유 중 - 보정 건너뜀, 관전·녹화만 수행")
        context["outcome"] = CorrectionOutcome(
            status=VIEW_ONLY_OBSERVATION,
            path="observation",
            key_decision="",
            best_xy=None,
            ok_screen_xy=None,
            fallback=None,
        )
        return _make_result(step, "success", started_at, settings)

    if not settings.correction_enabled:
        return _make_result(step, "skipped", started_at, settings)
    if not recipe_id:
        print(f"[INFO] RECIPE_ID 없음 - 보정 생략, 엔지니어 직접 처리 (EQP_ID={eqp_id})")
        return _make_result(step, "skipped", started_at, settings)

    from poc.workflow_3.align.correction import CorrectionConfig, correct_align_fail_auto
    from poc.workflow_3.align.live_search import LiveSearchConfig

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
    context["correction_debug_dir"] = debug_dir
    # search-around 재설계: PM 드롭다운 절대 배율 + FOV 격자 sweep 주입점. None 이면
    # search_around 가 종전 legacy(휠+spiral) 경로를 그대로 쓴다.
    from poc.workflow_3.align.grid_search import GridSearchConfig

    grid_mag = _build_grid_mag_control(context, settings, debug_dir)
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
                # OK 자동 클릭(기본 off = reposition 까지만, OK 는 엔지니어).
                ok_click_enabled=settings.ok_click_enabled,
                # key 부재 시 spiral pan 탐색 위임 여부(off -> pan 없이 escalate).
                fallback_search_enabled=settings.fallback_search_enabled,
                # consensus 라우팅 설정(Workflow3Settings 에서 주입).
                consensus_enabled=settings.consensus_enabled,
                consensus_min_s=settings.consensus_min_s,
                consensus_max_events=settings.gather_max_events,
                consensus_sync_timeout_sec=settings.consensus_sync_timeout_sec,
            ),
            # legacy spiral 은 pan 상한만 주입(streak 는 LiveSearchConfig 가 예산+1 로 맞춘다).
            fallback_config=LiveSearchConfig(pan_budget=settings.search_pan_budget),
            dry_run=settings.correction_dry_run,
            debug_dir=debug_dir,
            eqp_id=eqp_id,
            recipe_name=recipe_id,
            grid_mag=grid_mag,
            grid_config=GridSearchConfig(
                radius_um=settings.search_radius_um,
                min_key_px=settings.search_min_key_px,
                pan_budget=settings.search_pan_budget,
                odom_tol_fov=settings.search_odom_tol_fov,
                max_chase=settings.search_max_chase,
            ),
        )
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="correction_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    # 점유 미확정이면 '보정 완료' 로 보고하지 않는다 - 알림이 생략되면 안 된다.
    # 여기까지 왔으면 보정을 실제로 시도한 것이다(점유 skip 은 위에서 early return).
    resolved = resolve_correction_outcome_status(occupancy, outcome.status, attempted=True)
    if resolved != outcome.status:
        print(
            f"[WARNING] 점유 여부 미확정({occupancy}) - status 강등: "
            f"{outcome.status} -> {resolved}"
        )
        outcome.status = resolved
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


_ACCESS_TITLE_TOKENS = tuple(
    token.strip().lower()
    for token in os.getenv(
        "ALIGN_FAIL_ACCESS_TITLES", "select,request,confirm,요청,공유,허용"
    ).split(",")
    if token.strip()
)


def _find_access_request_popup(baseline: set):
    """세션 중 새로 뜬 top-level 창 중 접근 요청 팝업으로 볼 만한 것을 찾는다.

    제목 문구를 아직 모르므로 두 가지를 함께 한다 - 후보 토큰으로 팝업을 고르고,
    baseline 에 없던 **모든** 새 제목을 콘솔에 남긴다(첫 실행에서 실제 문구를 아는
    유일한 경로. Mac 에서는 이 화면을 볼 수 없다).
    """
    if not callable(collect_window_rows) or not callable(find_window_by_title_prefix):
        return None
    try:
        rows = collect_window_rows()
    except Exception as exc:
        print(f"[WARNING] 접근 요청 감시용 창 열거 실패: {exc}")
        return None

    candidate = None
    for row in rows:
        title = (row.title or "").strip()
        if not title or title in baseline:
            continue
        baseline.add(title)
        print(f"[INFO] 세션 중 새 창 감지: title={title!r}")
        lowered = title.lower()
        if candidate is None and any(tok in lowered for tok in _ACCESS_TITLE_TOKENS):
            candidate = title
    if candidate is None:
        return None
    try:
        window, _title, _backend = find_window_by_title_prefix(candidate)
    except Exception as exc:
        print(f"[WARNING] 접근 요청 팝업 창 확보 실패: {exc}")
        return None
    return window


def _make_access_watcher(settings: Workflow3Settings, tag: str):
    """engineer watch 루프에서 매 주기 호출할 접근 요청 감시자를 만든다(off 면 None).

    응답하지 않으면 상대가 강제 종료로 우리 세션을 끊을 수 있다(오피스 확인). 그래서
    감시는 기본 on 이고, 실제 허용 클릭만 문구 확인 후 여는 opt-in 이다.
    """
    if not settings.access_request_watch_enabled:
        return None

    from poc.workflow_3.monitor.access_request import (
        STATUS_NOT_FOUND,
        grant_access_request,
    )
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import analyze_window_target

    debug_dir = DEBUG_IMAGE_DIR / "access_request" / tag
    baseline: set = set()
    try:
        if callable(collect_window_rows):
            baseline = {(row.title or "").strip() for row in collect_window_rows()}
    except Exception:
        baseline = set()

    def _locate(image, target):
        result = analyze_window_target(
            None, "", "uia", target,
            debug_image_dir=debug_dir,
            log_name="access_request",
            component_name="access_request",
            artifact_prefix=target.key,
            image=image,
        )
        return result.point

    def _read_tokens(image, point, key):
        box = crop_box_around_point(
            point, image.width, image.height,
            left_ratio=0.30, right_ratio=0.30, half_height_ratio=0.06,
        )
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(time.time()),
            artifact_label=key,
            log_name="access_request",
        )
        return tokens_from_text(read.raw_text) if read.ok else []

    def _click(window, image, point, key):
        screen = image_point_to_screen(window, point, image_size=image.size)
        if screen is None:
            raise RuntimeError(f"접근 요청 팝업 좌표 변환 실패: {key} point={point}")
        print(f"[INFO] 접근 요청 팝업 클릭: {key} px={point} -> screen={screen}")
        click_at_screen(screen, f"access_{key}", action_enabled=settings.action_enabled)

    def _watch():
        result = grant_access_request(
            settings,
            locate_fn=_locate,
            read_tokens_fn=_read_tokens,
            click_fn=_click,
            capture_fn=capture_window,
            find_popup_fn=lambda: _find_access_request_popup(baseline),
        )
        if result.status != STATUS_NOT_FOUND:
            log_work2_event(
                component=LOG_COMPONENT, message="access_request",
                level="info", status=result.status, verdict=result.verdict,
            )

    return _watch


def _engineer_watch(
    recording: RecordingSession,
    watch_sec: float,
    *,
    done_detector=None,
    poll_sec: float = 8.0,
    access_watcher=None,
    access_poll_sec: float = 2.0,
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
    next_access = 0.0
    while time.time() < deadline and recording.is_alive():
        if is_aborted():
            # 해제했는데 watch 상한(기본 수십 초~분)을 다 기다리면 "아무 조치도 못 하고
            # 끝날 때까지 기다린다" 는 원래 문제가 그대로 남는다.
            print(f"[WARNING] 긴급 해제됨({abort_reason()}) - engineer watch 조기 종료")
            break
        # 접근 요청은 상대가 오래 기다려 주지 않는다(무응답 시 강제 종료 가능).
        # done_detector 보다 촘촘히 본다.
        if access_watcher is not None and time.time() >= next_access:
            try:
                access_watcher()
            except Exception as exc:
                print(f"[WARNING] 접근 요청 감시 예외(무시, watch 계속): {exc}")
            next_access = time.time() + max(access_poll_sec, 0.5)
        if done_detector is not None and time.time() >= next_check:
            try:
                if done_detector():
                    print("[INFO] 측정 시작 감지(align 완료 추정) - engineer watch 조기 종료")
                    break
            except Exception as exc:
                print(f"[WARNING] done detector 예외(무시, cap 으로 진행): {exc}")
            next_check = time.time() + max(poll_sec, 0.0)
        time.sleep(2.0)
    reason = getattr(recording, "stop_reason", "")
    if reason == "window_gone":
        print("[INFO] 엔지니어가 Remote Monitoring 창을 닫음 - 명시적 완료로 watch 종료")
    elif reason:
        print(f"[INFO] 녹화 세션 종료로 watch 종료: reason={reason}")
    print("[INFO] engineer watch 종료")


def _teardown_steps(eqp_id, context, result, settings, *, input_blocked, recording):
    """알람 사이클 teardown 단계 목록 - 순서가 계약이다.

    첫 단계는 **항상** 입력 해제다: 뒤 단계가 전부 실패해도 엔지니어의 물리
    마우스/키보드는 풀려 있어야 한다. 각 단계의 전제조건은 목록에서 빼지 않고
    클로저 **안에서** 판정한다 - 목록 길이/순서를 일정하게 유지해야 순서 규약을
    테스트로 검사할 수 있다.
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"align_fail_cycle {eqp_id}")

    def _stop_recording():
        # stop() 과 결과 필드 갱신을 함께 감싼다 - 여기서 실패하면 두 필드가
        # 조용히 비는 대신 note 가 남고 나머지 teardown 은 계속된다.
        sess = recording if recording is not None else context.get("recording")
        if sess is None:
            return
        frames = sess.stop("cycle_teardown")
        result.recording_dir = str(sess.out_dir)
        result.frame_count = len(frames)

    def _stop_prelude():
        # 정상 흐름에선 tool 창이 뜬 시점에 이미 멈췄다. 여기 남는 건 접속이 깨진
        # 경로(rcs_occupied 등) - 그때도 화면 녹화 스레드가 살아 있으면 안 된다.
        stop_prelude_recording(context, "cycle_teardown")
        if context.get("prelude_dir"):
            result.prelude_dir = context["prelude_dir"]
            result.prelude_frame_count = context.get("prelude_frame_count", 0)

    def _close_tool():
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)

    def _close_alert():
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return [
        ("input_unblock", _unblock),
        ("recording_stop", _stop_recording),
        ("prelude_stop", _stop_prelude),
        ("close_tool", _close_tool),
        ("close_alert", _close_alert),
    ]


def _maybe_start_graph_mirror(settings: Workflow3Settings, steps, context: dict, graph_name: str):
    """live graph view(workflow_4 cycle mirror)를 시작한다 — opt-in, 기본 off.

    workflow_4 에 하드 의존을 만들지 않으려고 import 를 함수 안에서 가드한다.
    import 실패 시 경고 **1회** 후 비활성(사이클 동작 불변). 그래프는 runner 에
    넘기는 **바로 그 step 목록**으로 만들어 이름/순서가 production 과 어긋날 수
    없고, run_dir 은 runner.run() 이 `context["run_dir"]` 에 넣는 값을 mirror 가
    폴링마다 읽으므로 run() 전에 시작해도 된다(경로 예측 없음).
    """
    global _MIRROR_GRAPH_VIEW_WARNED
    if not getattr(settings, "graph_view_enabled", False):
        return None
    # 이 함수는 사이클의 try/finally **밖**에서 불리므로 어떤 예외도 밖으로 내보내면
    # 안 된다 - 새면 teardown 과 '알람당 cube 1회' 보장(7512b1d)을 통째로 건너뛴다.
    # 시각화 실패는 사이클 실패가 아니다: 경고 1회 후 None.
    try:
        from poc.workflow_4.adapters.workflow3_cycle import (
            CycleGraphMirror,
            build_step_chain_graph,
        )

        graph = build_step_chain_graph(
            graph_name, [(s.step_id, s.target_description) for s in steps]
        )
        mirror = CycleGraphMirror(
            graph,
            run_dir_fn=lambda: context.get("run_dir"),
            poll_sec=0.5,
            refresh_sec=1,
            autoopen=getattr(settings, "graph_view_autoopen", False),
        )
        mirror.start()
    except Exception as exc:
        if not _MIRROR_GRAPH_VIEW_WARNED:
            print(
                "[WARNING] workflow_4 live graph view 비활성(시작 실패, "
                f"1회만 경고): {type(exc).__name__}: {exc}"
            )
            _MIRROR_GRAPH_VIEW_WARNED = True
        return None
    print("[INFO] live graph view 시작 (workflow_4 mirror): 스냅샷은 runner run_dir 에 남는다")
    return mirror


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
    cycle_started_at = time.time()
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
    # 알람 1건의 cube 게이트 - 본문과 finally 가 모두 통보를 시도하고, 먼저 부른 쪽만
    # 실제로 나간다. watchdog 은 사이클이 오래 끌 때 '진행 중' 을 1회 고지한다.
    notifier = CycleNotifier(
        eqp_id, recipe_id,
        enabled=settings.rich_notify_enabled,
        reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
    )
    notifier.start_watchdog(settings.notify_delay_sec)
    mirror = None
    try:
        # live graph view (opt-in, 기본 off) — run() 전에 시작해 첫 step 부터 관찰한다.
        # try 안에 두는 이유: 여기서 던져도 finally 의 teardown/알림 보장이 지켜져야 한다.
        steps = build_cycle_steps(eqp_id)
        mirror = _maybe_start_graph_mirror(
            settings, steps, context, f"align_fail_cycle_{eqp_id}"
        )
        # 시연용 접속 구간 녹화 - **step 시작 전**에 켠다. ensure_rcs_ready 가 RCS 를
        # 재실행/재로그인하는 장면까지 담아야 "RCS 를 열어 tool 로 들어가는" 영상이 된다.
        prelude = start_prelude_recording(context, settings)
        if prelude is not None:
            result.prelude_dir = str(prelude.out_dir)

        # 자동 GUI 구간 동안 사용자 물리 입력 차단(opt-in) — foreground lock/클릭 방해 방지.
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"align_fail_cycle {eqp_id}")
        run = runner.run(steps, context, executor)
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
        # 접속 -> align fail 판정 -> 보정 시도까지 끝난 이 시점이 정책상 첫 통보다.
        recording_dir = str(recording.out_dir) if recording is not None else ""
        notifier.notify_outcome(
            outcome, recording_dir=recording_dir,
            failed_step=result.failed_step, failure_class=result.failure_class,
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
            access_watcher = None
            try:
                access_watcher = _make_access_watcher(settings, tag)
            except Exception as exc:
                print(f"[WARNING] 접근 요청 감시자 생성 실패(감시 없이 진행): {exc}")
            _engineer_watch(
                recording, settings.engineer_watch_sec,
                done_detector=done_detector,
                poll_sec=settings.engineer_done_poll_sec,
                access_watcher=access_watcher,
                access_poll_sec=settings.access_watch_poll_sec,
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
        # 결과 통보 backstop — 본문이 예외로 발송에 도달하지 못했으면 여기서 나간다.
        # teardown **앞**에 둔다: teardown 한 단계가 깨져도 엔지니어는 통보를 받아야
        # 한다. 이미 본문에서 보냈으면 게이트가 막아 중복은 생기지 않는다.
        sess = recording if recording is not None else context.get("recording")
        if notifier.notify_outcome(
            context.get("outcome"),
            recording_dir=str(sess.out_dir) if sess is not None else "",
            failed_step=result.failed_step, failure_class=result.failure_class,
        ):
            print(f"[WARNING] 사이클이 결과 통보 없이 종료 - backstop 발송: EQP_ID={eqp_id}")

        # teardown 은 run_teardown 이 단계별로 보호한다 - 한 단계가 던져도 입력
        # 해제/tool 닫기/팝업 backstop 은 반드시 실행된다.
        failures = run_teardown(
            _teardown_steps(
                eqp_id, context, result, settings,
                input_blocked=input_blocked, recording=recording,
            ),
            label=f"align_fail_cycle {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)

        # 이 테이크가 처리한 이미지를 한 폴더로 모은다. **teardown 뒤**여야
        # close_tool 이 남긴 crop 까지 들어오고 result 의 녹화 필드도 채워져 있다.
        # finally 안이어야 하는 이유는 녹화 스레드와 engineer watch 가 테이크마다
        # 존재하지 않기 때문이다 - 보정 성공(watch 없음)과 접속 단계 실패(녹화 없음)에
        # 훅을 걸면 그 테이크가 통째로 빠진다.
        gather_and_report(result, context, started_epoch=cycle_started_at)

        # tool 창을 닫고 빠져나온 직후 = 엔지니어가 화면을 되찾는 순간. 이 테이크가
        # 성공인지 판단할 신호를 여기서 한 장으로 낸다.
        print_cycle_report(
            result, context, elapsed_sec=time.time() - cycle_started_at
        )

        # live graph view mirror 를 멈추고 마지막 스냅샷(.md/.html)을 남긴다.
        if mirror is not None:
            mirror.stop(final=True)

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
            consensus_enabled=settings.consensus_enabled,
            consensus_min_s=settings.consensus_min_s,
            consensus_max_events=settings.gather_max_events,
            consensus_sync_timeout_sec=settings.consensus_sync_timeout_sec,
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

    # ---- 만성 모호(ambiguous) → align key 재등록 권고를 manifest/로그로 surface. ----
    if feas.reregister_recommended:
        sr = f"{feas.second_ratio:.3f}" if feas.second_ratio is not None else "-"
        note = (
            f"REREGISTER recipe={recipe_id} (chronic-ambiguous: 2nd/best={sr}); "
            f"align key 를 더 변별력 있는 영역으로 재등록 권고"
        )
        result.notes.append(note)
        print(f"[WARNING] {note}")
        log_work2_event(
            component=LOG_COMPONENT, message="reregister_recommended", level="warning",
            eqp_id=eqp_id, recipe_id=recipe_id, second_ratio=feas.second_ratio,
        )

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
                        consensus_enabled=settings.consensus_enabled,
                        consensus_min_s=settings.consensus_min_s,
                        consensus_max_events=settings.gather_max_events,
                        consensus_sync_timeout_sec=settings.consensus_sync_timeout_sec,
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


class _PMDropdownSelector:
    """'PM' 버튼 드롭다운으로 **절대 배율**을 고르는 actuator. zoom ladder(`_run_pm_dropdown_arms`)와
    grid search(`_build_grid_mag_control`)가 공유한다(2026-08-28 추출 - 본문은 오피스 검증본 그대로).

    위치 파이프라인(coarse->fine, 검증된 2단계 VLM 을 드롭다운에도 적용):
      * PM 버튼 = 2단계 VLM(캐시). 실패 시 PM 숫자박스 검출 -> 왼쪽 기하 폴백.
      * 드롭다운 영역 = coarse bbox -> crop_region_from_bbox 패딩(고정 비율 dropdown_region_below
        는 폴백). PM 앵커라 캐시.
      * 값 열거 = PaddleOCR ``Spotting:`` 1회로 배율 값공간.
      * 행 클릭 점 = 각 목표 값마다 fine(mai-ui)로 '값 <text> 행' 직접 그라운딩(실패 시 PaddleOCR
        박스 중심 폴백). VLM=영역/좌표, OCR=값 확인 규약.

    드롭다운은 **열면 바로 행을 눌러야 한다**(닫는 동작이 따로 없다). 그래서 ``list_values`` 는
    연 상태를 들고 있다가 다음 ``select`` 가 그 열린 목록에서 행을 누른다(grid search 의
    lazy options 규약). 선택 후 판독은 PM box OCR 이며 실패하면 None(명령값을 믿지 않는다).
    """

    def __init__(self, tool_window, capture_dir, tag, frame_wh, settings, sem_box_client, ocr_client,
                 *, action_enabled=None):
        self.tool_window = tool_window
        # 클릭 게이트. zoom ladder(진단)는 SAFE_MODE 만, 보정 경로는 correction_dry_run 도 본다
        # (이중 게이트 - 배율 변경도 보정 actuation 이다). 기본 None = settings.action_enabled.
        self.action_enabled = settings.action_enabled if action_enabled is None else bool(action_enabled)
        self.capture_dir = capture_dir
        self.tag = tag
        self.frame_wh = frame_wh
        self.settings = settings
        self.sem_box_client = sem_box_client
        self.ocr_client = ocr_client
        # spotting 읽기용 OCR client — 없으면 paddleocr-vl 로 1개 생성(실패 시 sem_box client 재사용).
        reader = ocr_client
        if reader is None:
            try:
                from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

                reader = Workflow1VLMClient(settings.pm_ocr_service, timeout_sec=15.0)
            except Exception as exc:
                print(f"[WARNING] PM 드롭다운: OCR client 생성 실패 -> sem_box client 로 읽기: {exc}")
                reader = sem_box_client
        self.reader = reader
        self._btn = None       # PM 버튼 위치 캐시(고정 UI - 재오픈마다 다시 안 찾음).
        self._region = None    # 드롭다운 영역 캐시.
        self.open_idx = 0
        self.options: list = []
        self._opened = None    # list_values 가 열어 둔 (crop, origin, shot).
        self._pm_box = None    # PM 숫자박스 캐시(고정 UI) - 되읽기는 첫 detect 이후 OCR crop 만.
        self.meta = {"pm_options": [], "spotting_raw": "", "selections": [], "aborted": None}

    def _locate_pm_button(self, image):
        if self._btn is not None:
            return self._btn
        pt = None
        try:
            from poc.workflow_3.vlm.ui_venus_mai_locator import (
                EXIT_SUCCESS,
                TargetConfig,
                analyze_window_target,
            )

            target = TargetConfig(
                key="pm_button",
                description=(
                    "the small 'PM' button (its label starts with the letter P followed "
                    "by M) in the SEM monitor toolbar, located just to the left of the "
                    "magnification value readout; clicking it opens a magnification dropdown"
                ),
            )
            res = analyze_window_target(
                self.tool_window, "", "uia", target,
                debug_image_dir=self.capture_dir, log_name="vlm_calls",
                component_name=LOG_COMPONENT, artifact_prefix=f"{self.tag}_pmbtn",
                image=image, timeout_sec=15.0,
            )
            if res.exit_code == EXIT_SUCCESS and res.point:
                pt = {"x": int(res.point["x"]), "y": int(res.point["y"])}
                print(f"[INFO] PM 버튼 2단계 VLM 위치: {pt}")
            else:
                print(f"[WARNING] PM 버튼 2단계 VLM 미검출(exit={res.exit_code}) -> 기하 폴백.")
        except Exception as exc:
            print(f"[WARNING] PM 버튼 2단계 VLM 실패(기하 폴백): {exc}")
        self._btn = pt
        return pt

    def _locate_dropdown_region(self, image, btn):
        """열린 드롭다운 리스트 영역을 coarse VLM bbox -> crop(l,t,r,b)로. 실패 시 기하 폴백."""
        from poc.workflow_3.sem_monitor.pm_dropdown import crop_region_from_bbox, dropdown_region_below

        if self._region is not None:
            return self._region
        region = None
        try:
            from poc.workflow_3.vlm.ui_venus_mai_locator import (
                EXIT_SUCCESS,
                TargetConfig,
                analyze_window_target,
            )

            target = TargetConfig(
                key="pm_dropdown_list",
                description=(
                    "the opened magnification dropdown list (a vertical list of "
                    "magnification value rows) that appeared just below the 'PM' button "
                    "in the SEM monitor toolbar"
                ),
            )
            res = analyze_window_target(
                self.tool_window, "", "uia", target,
                debug_image_dir=self.capture_dir, log_name="vlm_calls",
                component_name=LOG_COMPONENT, artifact_prefix=f"{self.tag}_pmdd",
                image=image, timeout_sec=15.0,
            )
            if res.exit_code == EXIT_SUCCESS and res.bbox:
                region = crop_region_from_bbox(res.bbox, self.frame_wh)
                if region is not None:
                    print(f"[INFO] PM 드롭다운 영역 2단계 VLM: bbox={res.bbox} -> crop={region}")
        except Exception as exc:
            print(f"[WARNING] PM 드롭다운 영역 VLM 실패(기하 폴백): {exc}")
        if region is None:
            region = dropdown_region_below(btn, self.frame_wh)
            print(f"[INFO] PM 드롭다운 영역 기하 폴백: {region}")
        self._region = region
        return region

    def _locate_row_point(self, image, value, text):
        """mai-ui(2단계)로 '값 <text> 행'의 클릭 점(full px)을 찾는다. 실패 시 None."""
        from poc.workflow_3.sem_monitor.pm_dropdown import row_target_description

        try:
            from poc.workflow_3.vlm.ui_venus_mai_locator import (
                EXIT_SUCCESS,
                TargetConfig,
                analyze_window_target,
            )

            target = TargetConfig(
                key=f"pm_row_{text}", description=row_target_description(value, text),
            )
            res = analyze_window_target(
                self.tool_window, "", "uia", target,
                debug_image_dir=self.capture_dir, log_name="vlm_calls",
                component_name=LOG_COMPONENT, artifact_prefix=f"{self.tag}_pmrow_{text}",
                image=image, timeout_sec=15.0,
            )
            if res.exit_code == EXIT_SUCCESS and res.point:
                return {"x": int(res.point["x"]), "y": int(res.point["y"])}
            print(f"[WARNING] PM 행 mai-ui 미검출({text}, exit={res.exit_code}) -> PaddleOCR 폴백.")
        except Exception as exc:
            print(f"[WARNING] PM 행 mai-ui 실패({text}, PaddleOCR 폴백): {exc}")
        return None

    def open_dropdown(self):
        """PM 버튼을 눌러 드롭다운을 열고 (crop, origin, pm_box, shot) 반환(실패 None).

        crop/origin 은 PaddleOCR 열거·폴백용(VLM 영역 기준), shot 은 mai-ui 행 그라운딩용 전체화면.
        """
        from poc.workflow_3.sem_monitor.pm_dropdown import pm_button_point
        from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box

        self.open_idx += 1
        open_idx = self.open_idx
        img = capture_window(self.tool_window)
        if self.frame_wh is None:
            self.frame_wh = img.size
        # PM 버튼: 2단계 VLM(캐시) 우선. 실패 시에만 숫자박스 검출 → 왼쪽 기하 폴백.
        btn = self._locate_pm_button(img)
        pm_box = None
        if btn is None:
            try:
                det = detect_sem_box(
                    img, self.sem_box_client, ocr_client=self.ocr_client,
                    two_stage=self.settings.pm_two_stage_ocr_enabled,
                )
                pm_box = det.pm_box_px
            except Exception as exc:
                print(f"[WARNING] PM 드롭다운: PM 박스 검출 실패: {exc}")
            btn = pm_button_point(pm_box)
        if btn is None:
            return None
        btn_scr = image_point_to_screen(self.tool_window, btn, image_size=self.frame_wh)
        if btn_scr is None:
            return None
        print(
            f"[INFO] PM 드롭다운 열기#{open_idx}: PM버튼 px={btn} -> screen={btn_scr}"
            f"{'' if self.action_enabled else ' [dry-run]'}"
        )
        if not click_at_screen(btn_scr, "pm_button", action_enabled=self.action_enabled):
            # 래치(abort)면 mouse_utils 가 클릭을 삼키고 False 를 준다 - 안 열린 드롭다운을
            # 읽으러 가지 않는다(ffd2d50 의 "래치는 actuation 지점에서 본다" 계약).
            if is_aborted():
                print("[WARNING] PM 드롭다운 열기 생략(긴급 해제 래치).")
                return None
        if _PM_DROPDOWN_OPEN_SETTLE_SEC > 0:
            time.sleep(_PM_DROPDOWN_OPEN_SETTLE_SEC)
        shot = capture_window(self.tool_window)
        # 드롭다운 영역 = 2단계 VLM(coarse) bbox. 실패 시 PM 버튼 아래 고정 비율 폴백.
        region = self._locate_dropdown_region(shot, btn)
        # 검증 overlay: PM 숫자 박스(cyan, 화면의 그것과 동일), 유도한 PM 버튼 클릭점(red),
        # 드롭다운 crop 영역(yellow) 을 그려 클릭이 'PM' 버튼에 맞는지 눈으로 확인.
        try:
            _save_pm_dropdown_overlay(
                shot, self.capture_dir / f"{self.tag}_pm_dropdown_open{open_idx}.jpg",
                pm_box, btn, region,
            )
        except Exception as exc:
            print(f"[WARNING] PM 드롭다운 overlay 저장 실패: {exc}")
            try:
                save_debug_jpeg(shot, self.capture_dir / f"{self.tag}_pm_dropdown_open{open_idx}.jpg")
            except Exception:
                pass
        if region is None:
            return None
        l, t, r, b = region
        return shot.crop((l, t, r, b)), (l, t), pm_box, shot

    def read_options(self, crop, origin):
        from poc.workflow_3.sem_monitor.pm_dropdown import read_dropdown_options

        options, raw_text = read_dropdown_options(crop, self.reader, crop_origin=origin)
        self.meta["spotting_raw"] = raw_text
        self.meta["pm_options"] = [{"value": o["value"], "text": o["text"]} for o in options]
        self.options = options
        if options:
            print(
                f"[INFO] PM 드롭다운 옵션 {len(options)}개: "
                + ", ".join(f"{o['text']}({o['value']})" for o in options)
            )
        return options

    def click_option(self, opt, crop, origin, shot, label) -> bool:
        """열린 드롭다운에서 opt 행을 누른다(mai-ui 우선, PaddleOCR 폴백) + settle. 실패 False."""
        from poc.workflow_3.sem_monitor.pm_dropdown import nearest_option, read_dropdown_options

        value, text = opt["value"], opt["text"]
        point = self._locate_row_point(shot, value, text)
        used = "mai-ui"
        if point is None:
            opts_k, _raw_k = read_dropdown_options(crop, self.reader, crop_origin=origin)
            o2 = nearest_option(opts_k, value)
            if o2 is not None:
                point, used = o2["center"], "paddle"
        if point is None:
            print(f"[WARNING] PM 행 위치 실패({label} {text}={value}) - 건너뜀.")
            return False
        scr = image_point_to_screen(self.tool_window, point, image_size=self.frame_wh)
        if scr is None:
            print(f"[WARNING] PM 옵션 screen 변환 실패({label}) - 건너뜀.")
            return False
        print(
            f"[INFO] PM 옵션 클릭({used}): {text}({value}) px={point} -> screen={scr}"
            f"{'' if self.action_enabled else ' [dry-run]'}"
        )
        if not click_at_screen(scr, f"pm_opt_{text}", action_enabled=self.action_enabled) and is_aborted():
            print(f"[WARNING] PM 옵션 클릭 생략({label}, 긴급 해제 래치).")
            return False
        self.meta["selections"].append(
            {"label": label, "value": value, "text": text, "locator": used}
        )
        if self.settings.zoom_probe_settle_sec > 0:
            time.sleep(self.settings.zoom_probe_settle_sec)
        return True

    # ---- grid search 주입점 (align/grid_search.MagnificationControl 의 options_fn / set_fn) ----

    def list_values(self) -> list:
        """드롭다운을 열어 배율 값 목록을 읽고 **열린 채** 둔다(다음 select 가 그 목록에서 누른다)."""
        op = self.open_dropdown()
        if op is None:
            print("[WARNING] PM 드롭다운: 열기 실패 -> 옵션 없음(grid search 는 legacy 로 degrade).")
            return []
        crop, origin, _pm_box, shot = op
        options = self.read_options(crop, origin)
        if not options:
            # 읽은 행이 없으면 누를 행도 없다 - 열린 채 두면 이후 클릭이 전부 드롭다운에 들어간다.
            # PM 버튼을 다시 눌러(토글) 닫기를 시도한다. 닫혔는지는 확인할 수 없어 경고만 남긴다.
            print("[WARNING] PM 드롭다운 옵션 0개 - PM 버튼 재클릭으로 닫기 시도(확인 불가).")
            btn = self._locate_pm_button(shot)
            btn_scr = image_point_to_screen(self.tool_window, btn, image_size=self.frame_wh) if btn else None
            if btn_scr is not None:
                click_at_screen(btn_scr, "pm_button_close", action_enabled=self.action_enabled)
            self._opened = None
            return []
        self._opened = (crop, origin, shot)
        return [float(o["value"]) for o in options]

    def select(self, target):
        """target 에 가장 가까운 행을 눌러 절대 배율을 바꾸고 PM OCR 로 되읽는다. 판독 실패 None."""
        from poc.workflow_3.sem_monitor.pm_dropdown import nearest_option
        from poc.workflow_3.sem_monitor.sem_box_detect import (
            detect_sem_box,
            parse_pm_magnification,
            read_pm_via_ocr,
        )

        if self._opened is None:
            op = self.open_dropdown()
            if op is None:
                return None
            crop, origin, _pm_box, shot = op
            if not self.options:
                self.read_options(crop, origin)
        else:
            crop, origin, shot = self._opened
        self._opened = None
        opt = nearest_option(self.options, float(target))
        if opt is None or not self.click_option(opt, crop, origin, shot, f"grid_{int(target)}"):
            return None
        try:
            shot = capture_window(self.tool_window)
            if self._pm_box is None:
                det = detect_sem_box(
                    shot, self.sem_box_client,
                    ocr_client=self.ocr_client, two_stage=self.settings.pm_two_stage_ocr_enabled,
                )
                self._pm_box = det.pm_box_px
                pm_text = det.pm_text
            else:
                pm_text = read_pm_via_ocr(shot, self._pm_box, self.reader)
            mag = parse_pm_magnification(pm_text)
        except Exception as exc:
            print(f"[WARNING] PM 배율 되읽기 실패(target={target}): {exc}")
            return None
        print(f"[INFO] PM 배율 선택 {opt['text']}({opt['value']}) -> 판독 {mag}")
        return None if mag is None else float(mag)


def _build_grid_mag_control(context: dict, settings: Workflow3Settings, capture_dir):
    """grid search 용 MagnificationControl. 조건이 안 되면 None -> search_around 가 legacy.

    Windows 유틸이 없는 개발 PC, search_mode=legacy, fallback off, tool 창 없음이 그 경우다.
    여기서는 VLM 호출도 캡처도 하지 않는다 - fallback 이 실제로 필요해질 때 selector 가 lazy 로
    드롭다운을 열고 첫 캡처에서 frame_wh 를 채운다(대부분의 사이클은 primary 로 끝난다).
    """
    tool_window = context.get("tool_window")
    if settings.search_mode != "grid" or not settings.fallback_search_enabled:
        return None
    if tool_window is None or image_point_to_screen is None or capture_window is None:
        print("[INFO] grid search 주입 생략(tool 창/Windows 유틸 없음) - legacy fallback.")
        return None
    from poc.workflow_3.align.grid_search import MagnificationControl

    sem_box_client = context.get("sem_box_client")
    if sem_box_client is None:
        try:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            sem_box_client = Workflow1VLMClient(settings.sem_box_vlm_service)
        except Exception as exc:
            print(f"[WARNING] grid search 주입 생략(VLM client 생성 실패): {exc} - legacy fallback.")
            return None
    sel = _PMDropdownSelector(
        tool_window, capture_dir, context.get("tag", ""), None, settings, sem_box_client, None,
        # 보정 actuation 의 이중 게이트(SAFE_MODE=0 **and** CORRECTION_DRY_RUN=0)를 배율 클릭에도 건다.
        action_enabled=settings.action_enabled and not settings.correction_dry_run,
    )
    return MagnificationControl(sel.list_values, sel.select)


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

    절차: (1) 드롭다운 오픈+캡처, (2) 최초 1회 PaddleOCR ``Spotting:`` 으로 행별 (배율, 클릭
    박스) 읽기 → 값공간 목표 산출, (3) 각 목표 행 클릭(절대 배율 적용) → settle →
    ``capture_rung(label)`` 으로 캡처+재매칭. 위치/클릭은 `_PMDropdownSelector` 가 한다.

    절대 선택이라 wheel 처럼 baseline 복귀가 필요 없다(드리프트 없음). 선택으로 배율이
    바뀌어도 mai-ui 가 매 목표마다 행을 다시 찾으므로 stale 좌표 오클릭이 없다. 읽기
    실패/옵션 부족이면 중단(엔지니어 처리).

    반환: zoom_ladder.json 의 ``pm_dropdown`` 섹션에 들어갈 메타(dict).
    """
    from poc.workflow_3.sem_monitor.pm_dropdown import choose_step_targets

    sel = _PMDropdownSelector(
        tool_window, capture_dir, tag, feas.frame_wh, settings, sem_box_client, ocr_client
    )
    meta = sel.meta

    # 1) 첫 오픈 + 옵션 읽기.
    first = sel.open_dropdown()
    if first is None:
        meta["aborted"] = "pm_box_or_open_failed"
        print("[WARNING] PM 드롭다운 fallback 중단: PM 버튼/박스를 못 찾음.")
        return meta
    crop, origin, _pm_box, shot0 = first
    options = sel.read_options(crop, origin)
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

    # 탐색 순서: 키를 찾았으면(ambiguous = 보이나 모호) 먼저 **zoom IN**(배율↑)으로 점을
    # 더 또렷이 핀하고, 키를 못 찾았으면(not_visible = 저점수) 먼저 **zoom OUT**(배율↓)으로
    # 넓게 보고 찾는다. (zoom-in 이 align 위치를 더 잘 잡는다는 경험.)
    outs = [t for t in targets if t[0].startswith("out")]
    ins = [t for t in targets if t[0].startswith("in")]
    targets = (outs + ins) if feas.verdict == "not_visible" else (ins + outs)
    meta["order"] = "out_first" if feas.verdict == "not_visible" else "in_first"
    print(
        f"[INFO] PM 드롭다운 탐색 순서: {meta['order']} (verdict={feas.verdict}) "
        f"-> {[lbl for lbl, _ in targets]}"
    )

    # 2) 각 목표 행 클릭. 첫 목표는 이미 열린 드롭다운(+읽은 crop/shot) 재사용, 이후는 매번 재오픈.
    cur = (crop, origin, shot0)
    for k, (label, opt0) in enumerate(targets):
        if k > 0:
            op = sel.open_dropdown()
            if op is None:
                print(f"[WARNING] PM 드롭다운 재오픈 실패({label}) - 건너뜀.")
                continue
            cur = (op[0], op[1], op[3])
        if not sel.click_option(opt0, cur[0], cur[1], cur[2], label):
            continue
        capture_rung(label)

    print(
        f"[INFO] PM 드롭다운 fallback 완료: 선택 {len(meta['selections'])}/{len(targets)} "
        f"(baseline={baseline_mag})"
    )
    return meta


def _check_teardown_steps(eqp_id, context, settings, *, input_blocked):
    """점검(check-only) 사이클 teardown 단계 목록 - 녹화가 없어 3단계다.

    첫 단계는 **항상** 입력 해제다. 과거 이 사이클만 해제를 마지막에 둬서,
    close_alert_window 가 던지면 엔지니어 입력이 잠긴 채 남는 결함이 있었다
    (F1). 순서는 test_teardown.py 가 검사한다.
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"align_fail_check {eqp_id}")

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
    mirror = None
    try:
        # live graph view (opt-in, 기본 off) — check 전용 5 step 그래프로 미러링.
        steps = build_check_steps(eqp_id)
        mirror = _maybe_start_graph_mirror(
            settings, steps, context, f"align_fail_check_{eqp_id}"
        )
        # 자동 GUI 구간(접속~캡처~닫기) 동안 사용자 물리 입력 차단(opt-in).
        # engineer watch 가 없으므로 close 까지 차단 유지하고 finally 끝에서 해제.
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"align_fail_check {eqp_id}")
        run = runner.run(steps, context, executor)
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
        # 입력 해제를 **첫 단계**로 올린다 - 과거엔 close_alert_window 뒤에 있어
        # 그게 던지면 엔지니어 입력이 잠긴 채 남았다(F1).
        failures = run_teardown(
            _check_teardown_steps(eqp_id, context, settings, input_blocked=input_blocked),
            label=f"align_fail_check {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)
        if mirror is not None:
            mirror.stop(final=True)

    return result


__all__ = [
    "CycleResult",
    "build_check_steps",
    "build_cycle_steps",
    "run_alarm_cycle",
    "run_check_only_cycle",
]
