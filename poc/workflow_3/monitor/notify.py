"""align fail 알림 — Windows 팝업 + cube rich notification(처리 결과 중심).

팝업 헬퍼(`_show_popup_windows`/`close_alert_window`)는 align_fail_alarm_record
에서 이동했다. cube 알림은 엔지니어 소유 `office_rich_notify` 모듈에 위임하며,
workflow_3 의 정책은 **처리 실패 시 알림** 이다 — CorrectionOutcome.status 가
"corrected" 가 아니면 outcome 요약을 실어 발송한다(자동화 초기에는 사실상 매번).

office_rich_notify 는 정위치(poc.workflow_3.monitor.office_rich_notify)에서
로드한다(없으면 cube 알림 비활성, 텍스트 로그만).
"""

import inspect
import threading
import time

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.integration_loader import load_office_integration

LOG_COMPONENT = "align_fail_notify"

# 알림 팝업 제목 — 표시(notify_align_fail)와 닫기(close_alert_window)가 같은 값을
# 써야 창을 찾을 수 있다.
ALERT_POPUP_TITLE = "CD-SEM Align Fail 감지"
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"


# ------------------------------------------------------------------
# office_rich_notify 로딩 (정위치).
# ------------------------------------------------------------------


def _load_rich_notify():
    """send_cube_align_fail_info 를 정위치에서 찾는다. 없으면 None."""
    integration = load_office_integration(
        "office_rich_notify",
        "poc.workflow_3.monitor.office_rich_notify",
        required_attrs=("send_cube_align_fail_info",),
    )
    if not integration.available:
        return None
    return integration.attrs["send_cube_align_fail_info"]


_SEND_CUBE_FN = _load_rich_notify()
RICH_NOTIFY_AVAILABLE = _SEND_CUBE_FN is not None
if not RICH_NOTIFY_AVAILABLE:
    print("[WARNING] office_rich_notify 모듈 없음 - cube 알림 비활성(텍스트 로그만).")


# ------------------------------------------------------------------
# Windows 팝업 (감지 즉시 알림 + record 전 닫기).
# ------------------------------------------------------------------


def show_popup_windows(title: str, message: str, *, timeout_sec: int = 60) -> None:
    """Windows MessageBox 를 데몬 스레드에서 띄운다 (루프 비차단).

    `timeout_sec` > 0 이면 해당 시간 후 팝업을 자동으로 닫는다(backstop).
    사이클이 정상이면 record 직전에 close_alert_window 로 먼저 닫는다.
    """
    try:
        import ctypes

        MB_ICONWARNING = 0x00000030
        MB_SYSTEMMODAL = 0x00001000
        MB_SETFOREGROUND = 0x00010000
        flags = MB_ICONWARNING | MB_SYSTEMMODAL | MB_SETFOREGROUND
        timeout_ms = max(0, timeout_sec) * 1000

        def _run():
            try:
                user32 = ctypes.windll.user32
                box_timeout = getattr(user32, "MessageBoxTimeoutW", None)
                if timeout_ms > 0 and box_timeout is not None:
                    # MessageBoxTimeoutW(hWnd, text, caption, type, langId, timeout_ms)
                    box_timeout(0, message, title, flags, 0, timeout_ms)
                else:
                    if timeout_ms > 0 and box_timeout is None:
                        print("[WARNING] MessageBoxTimeoutW 미지원 - 자동 종료 없이 표시")
                    user32.MessageBoxW(0, message, title, flags)
            except Exception as exc:
                print(f"[WARNING] Windows 팝업 실패: {exc}")

        threading.Thread(target=_run, daemon=True).start()
    except AttributeError:
        print(f"[INFO] 현재 OS 에서 MessageBox 미지원 - 콘솔 알림만: {title} | {message}")
    except Exception as exc:
        print(f"[WARNING] 팝업 표시 실패: {exc}")


def close_alert_window(title: str = ALERT_POPUP_TITLE, *, timeout_sec: float = 3.0) -> bool:
    """제목으로 알림 팝업(MessageBox) 창을 찾아 닫는다 (Windows 전용).

    팝업은 pywinauto 창이 아니라 ctypes FindWindowW + WM_CLOSE 로 닫는다. 같은
    제목 창이 여럿이면(연속 알림) 모두 닫을 때까지 짧게 반복한다. 비Windows/실패
    시 조용히 False.
    """
    try:
        import ctypes
    except Exception:
        return False

    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        print(f"[INFO] 현재 OS 에서 알림 창 닫기 미지원 - 생략: {title!r}")
        return False

    WM_CLOSE = 0x0010
    deadline = time.time() + max(0.0, timeout_sec)
    closed_any = False
    while True:
        try:
            hwnd = user32.FindWindowW(None, title)
        except Exception as exc:
            print(f"[WARNING] 알림 창 탐색 실패: {exc}")
            break
        if not hwnd:
            break
        try:
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            closed_any = True
        except Exception as exc:
            print(f"[WARNING] 알림 창 닫기 실패: {exc}")
            break
        if time.time() >= deadline:
            break
        time.sleep(0.2)

    if closed_any:
        print(f"[INFO] 알림 팝업 닫기 완료: {title!r}")
    else:
        print(f"[INFO] 닫을 알림 팝업 없음(이미 닫힘/미표시): {title!r}")
    return closed_any


def notify_align_fail_popup(
    eqp_id: str,
    alarm_time: str,
    alarm_name: str,
    recipe_id: str = "",
    operation_desc: str = "",
    lot_type_cd: str = "",
    *,
    timeout_sec: int = 60,
) -> None:
    """Align Fail 감지 시 Windows 팝업 알림."""
    message = (
        f"EQP_ID    : {eqp_id}\n"
        f"ALARM     : {alarm_name}\n"
        f"TIME      : {alarm_time}\n"
        f"RECIPE_ID : {recipe_id}\n"
        f"OPERATION : {operation_desc}\n"
        f"LOT_TYPE  : {lot_type_cd}\n\n"
        f"로그: {ALARM_LOG_PATH}"
    )
    show_popup_windows(ALERT_POPUP_TITLE, message, timeout_sec=timeout_sec)


# ------------------------------------------------------------------
# cube rich notification — 처리 결과 중심.
# ------------------------------------------------------------------


# 사이클 step id → 엔지니어가 읽을 단계 라벨. 다음 행동이 갈리는 지점이라 cube 에
# 싣는다: 접속 단계 실패면 tool 을 직접 열어야 하고, 보정 단계 실패면 이미 열린
# 창에서 align point 만 잡으면 된다. 목록에 없는 step 은 id 그대로 나간다.
_STEP_LABELS = {
    "ensure_rcs_ready": "RCS 준비(접속 전)",
    "close_alert_popup": "감지 팝업 닫기(접속 전)",
    "connect_tool": "tool 접속(List 탭 더블클릭)",
    "wait_tool_window": "tool 접속(Remote Monitoring 창 대기)",
    "start_recording": "녹화 시작",
    "locate_sem_panel": "SEM panel 인식(보정 준비)",
    "run_correction": "align 보정",
}


def _stage_note(failed_step: str, failure_class: str) -> str:
    """실패 step/failure_class 를 '실패단계=...' 한 줄로 만든다. 없으면 빈 문자열."""
    if not failed_step:
        return ""
    label = _STEP_LABELS.get(failed_step, "")
    stage = f"{label}[{failed_step}]" if label else failed_step
    if failure_class:
        stage = f"{stage}/{failure_class}"
    return f"실패단계={stage}"


def build_outcome_summary(
    outcome,
    *,
    recording_dir: str = "",
    reregister_ratio_threshold: float | None = None,
    failed_step: str = "",
    failure_class: str = "",
) -> str:
    """CorrectionOutcome 을 엔지니어용 한 줄 요약으로 만든다.

    outcome 이 None(보정 미수행: RECIPE_ID 없음, 사이클 중단 등)이어도 동작한다.
    matcher 모호도(second_ratio)가 있으면 값을 덧붙이고, reregister_ratio_threshold 가
    주어지고 그 값을 넘으면 '재등록 권장(모호 키)' 한 줄을 추가한다(임계 None=구 호출부면 권고 skip).
    """
    if outcome is None:
        parts = ["자동 보정 미수행(사이클 중단 또는 RECIPE_ID 없음) - 직접 확인 필요"]
    else:
        parts = []
        if outcome.status == "awaiting_engineer_ok":
            # 반자동 모드의 요구 행동을 맨 앞에 둔다 — status= 로 시작하면 엔지니어가
            # 무엇을 해야 하는지 알림 끝까지 읽어야 알 수 있다.
            parts.append("align point 로 이동 완료 - 위치 확인 후 OK 를 눌러주세요")
        parts += [f"status={outcome.status}", f"path={outcome.path}", f"decision={outcome.key_decision}"]
        if outcome.best_xy is not None:
            parts.append(f"best_xy={outcome.best_xy}")
        fallback = getattr(outcome, "fallback", None)
        if fallback is not None:
            parts.append(f"fallback={fallback.status}(pan {fallback.pan_count}회)")
            if fallback.best is not None:
                parts.append(f"최고후보 score={fallback.best.score:.3f}")
        if getattr(outcome, "error", None):
            parts.append(f"error={outcome.error}")
        second_ratio = getattr(outcome, "second_ratio", None)
        if second_ratio is not None:
            parts.append(f"second_ratio={second_ratio:.3f}")
            if reregister_ratio_threshold is not None and second_ratio > reregister_ratio_threshold:
                parts.append("재등록 권장(모호 키)")
    stage = _stage_note(failed_step, failure_class)
    if stage:
        # 요약 앞쪽에 둔다 - 엔지니어가 "어디까지 갔나" 를 먼저 알아야 다음 행동이 정해진다.
        parts.insert(1 if len(parts) > 1 else len(parts), stage)
    if recording_dir:
        parts.append(f"녹화={recording_dir}")
    return " | ".join(parts)


def _send_cube_async(eqp_id: str, recipe_id: str, summary: str) -> None:
    """office cube 함수를 데몬 스레드에서 호출한다(루프 비차단).

    office 함수가 summary 인자를 받으면 요약을 함께 보내고, 기존 2-인자 시그니처면
    생략한다(README: office 함수에 optional summary 추가 권장).
    """
    def _run():
        try:
            params = inspect.signature(_SEND_CUBE_FN).parameters
            if "summary" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                _SEND_CUBE_FN(eqp_id, recipe_id, summary=summary)
            else:
                _SEND_CUBE_FN(eqp_id, recipe_id)
        except Exception as exc:
            print(f"[WARNING] cube rich notify 예외: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def notify_correction_outcome(
    eqp_id: str,
    recipe_id: str,
    outcome,
    *,
    recording_dir: str = "",
    enabled: bool = True,
    reregister_ratio_threshold: float | None = None,
    failed_step: str = "",
    failure_class: str = "",
) -> None:
    """처리 실패 시 cube rich notification 을 비차단 발송한다.

    status == "corrected" 면 발송하지 않는다(성공은 로그만). office 함수가
    summary 인자를 받으면 outcome 요약을 함께 보내고, 기존 2-인자 시그니처면
    요약은 파일 로그에만 남긴다(README: office 함수에 optional summary 추가 권장).

    reregister_ratio_threshold 가 주어지면 모호 키(second_ratio>임계)를 판정한다:
    실패 경로는 summary 에 이미 권고가 실려 cube 에 나가고, corrected+모호는 cube spam
    없이 warning 파일 로그에 corrected_but_ambiguous audit 만 남긴다(재발 추적용).
    """
    status = getattr(outcome, "status", None) if outcome is not None else None
    summary = build_outcome_summary(
        outcome, recording_dir=recording_dir,
        reregister_ratio_threshold=reregister_ratio_threshold,
        failed_step=failed_step, failure_class=failure_class,
    )

    if status == "corrected":
        second_ratio = getattr(outcome, "second_ratio", None)
        ambiguous = (
            reregister_ratio_threshold is not None
            and second_ratio is not None
            and second_ratio > reregister_ratio_threshold
        )
        if ambiguous:
            print(f"[INFO] 자동 보정 성공이나 모호 키 - 재등록 권장(cube 생략): "
                  f"EQP_ID={eqp_id} | {summary}")
            log_work2_event(
                component=LOG_COMPONENT, message="corrected_but_ambiguous", level="warning",
                eqp_id=eqp_id, recipe_id=recipe_id,
                second_ratio=f"{second_ratio:.3f}", summary=summary,
            )
        else:
            print(f"[INFO] 자동 보정 성공 - cube 알림 생략: EQP_ID={eqp_id} | {summary}")
        return

    log_work2_event(
        component=LOG_COMPONENT, message="outcome_notify", level="warning",
        eqp_id=eqp_id, recipe_id=recipe_id, status=str(status), summary=summary,
    )
    if not enabled or not RICH_NOTIFY_AVAILABLE:
        print(f"[INFO] cube 알림 비활성 - 요약 로그만: EQP_ID={eqp_id} | {summary}")
        return

    _send_cube_async(eqp_id, recipe_id, summary)
    print(f"[INFO] cube 알림 발송(비차단): EQP_ID={eqp_id} | {summary}")


def send_progress_notify(eqp_id: str, recipe_id: str, elapsed_sec: float) -> None:
    """'자동 보정 진행 중' 중간 고지 — watchdog 전용(무한 침묵 방지).

    결과 알림이 아니므로 요구 행동을 담지 않는다. 자동화가 아직 tool 을 붙들고
    있다는 사실만 알려 엔지니어가 개입 시점을 판단하게 한다.
    """
    summary = (
        f"자동 보정 진행 중({elapsed_sec:.0f}s 경과) - 결과 알림이 곧 이어집니다. "
        f"지금 수동 조작하면 자동화와 충돌할 수 있습니다"
    )
    log_work2_event(
        component=LOG_COMPONENT, message="progress_notify", level="warning",
        eqp_id=eqp_id, recipe_id=recipe_id, elapsed_sec=f"{elapsed_sec:.1f}",
    )
    if not RICH_NOTIFY_AVAILABLE:
        print(f"[INFO] cube 알림 비활성 - 진행 고지 로그만: EQP_ID={eqp_id} | {summary}")
        return
    _send_cube_async(eqp_id, recipe_id, summary)
    print(f"[INFO] cube 진행 고지 발송(비차단): EQP_ID={eqp_id} | {summary}")


class CycleNotifier:
    """알람 1건의 cube 알림 게이트 — '정확히 1회' 발송 + 지연 watchdog.

    사이클은 본문(정상 종료)과 finally(예외 종료) 양쪽에서 결과를 통보하려 하므로,
    누가 먼저 부르든 첫 호출만 실제로 나가야 한다. 이 클래스가 그 판정을 소유한다.

    watchdog 은 `start_watchdog(delay_sec)` 로 건다. 그 시간까지 결과가 나오지 않으면
    '진행 중' 고지를 1회 보낸다 — 결과-후-알림 정책의 유일한 예외이며, 사이클이
    멈춰도 엔지니어가 영원히 모르는 상태를 막는 안전장치다.
    """

    def __init__(
        self,
        eqp_id: str,
        recipe_id: str,
        *,
        enabled: bool = True,
        reregister_ratio_threshold: float | None = None,
        timer_factory=threading.Timer,
    ):
        self.eqp_id = eqp_id
        self.recipe_id = recipe_id
        self.enabled = enabled
        self.reregister_ratio_threshold = reregister_ratio_threshold
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._outcome_sent = False
        self._progress_sent = False
        self._timer = None
        self._started_at = time.time()

    def start_watchdog(self, delay_sec: float) -> bool:
        """delay_sec 후 '진행 중' 고지를 보낼 watchdog 을 건다. 걸었으면 True.

        delay_sec <= 0 이거나 알림 자체가 꺼져 있으면 걸지 않는다.
        """
        if delay_sec <= 0 or not self.enabled:
            return False
        self._started_at = time.time()
        timer = self._timer_factory(delay_sec, self._fire_progress)
        timer.daemon = True
        self._timer = timer
        timer.start()
        return True

    def _fire_progress(self) -> None:
        """watchdog 만료 콜백 — 결과가 아직이면 진행 고지를 1회 보낸다.

        결과 발송과 경합할 수 있으므로(취소 직후 발화) 같은 락에서 판정한다.
        """
        with self._lock:
            if self._outcome_sent or self._progress_sent:
                return
            self._progress_sent = True
        send_progress_notify(
            self.eqp_id, self.recipe_id, time.time() - self._started_at,
        )

    def notify_outcome(
        self,
        outcome,
        *,
        recording_dir: str = "",
        failed_step: str = "",
        failure_class: str = "",
    ) -> bool:
        """결과 알림을 1회만 발송한다. 실제로 보냈으면 True, 이미 보냈으면 False."""
        with self._lock:
            if self._outcome_sent:
                return False
            self._outcome_sent = True
        self._cancel_timer()
        notify_correction_outcome(
            self.eqp_id, self.recipe_id, outcome,
            recording_dir=recording_dir, enabled=self.enabled,
            reregister_ratio_threshold=self.reregister_ratio_threshold,
            failed_step=failed_step, failure_class=failure_class,
        )
        return True

    def _cancel_timer(self) -> None:
        timer = self._timer
        if timer is not None:
            timer.cancel()
            self._timer = None


def send_detection_notify_async(eqp_id: str, recipe_id: str, *, enabled: bool = True) -> None:
    """감지 시점 cube 알림 — "지금 이 장비에 자동화가 들어간다"는 사전 고지.

    두 모니터(align_fail_monitor / align_fail_monitor_only_check)가 알람을 잡은 직후
    호출한다. 처리 *결과* 알림(notify_correction_outcome)과는 목적이 달라 둘 다 나간다.
    """
    if not enabled or not RICH_NOTIFY_AVAILABLE:
        return

    def _run():
        try:
            _SEND_CUBE_FN(eqp_id, recipe_id)
        except Exception as exc:
            print(f"[WARNING] cube rich notify 예외: {exc}")

    threading.Thread(target=_run, daemon=True).start()


__all__ = [
    "ALARM_LOG_PATH",
    "ALERT_POPUP_TITLE",
    "RICH_NOTIFY_AVAILABLE",
    "CycleNotifier",
    "build_outcome_summary",
    "close_alert_window",
    "notify_align_fail_popup",
    "notify_correction_outcome",
    "send_detection_notify_async",
    "send_progress_notify",
    "show_popup_windows",
]
