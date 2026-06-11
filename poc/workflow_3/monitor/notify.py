"""align fail 알림 — Windows 팝업 + cube rich notification(처리 결과 중심).

팝업 헬퍼(`_show_popup_windows`/`close_alert_window`)는 align_fail_alarm_record
에서 이동했다. cube 알림은 엔지니어 소유 `office_rich_notify` 모듈에 위임하며,
workflow_3 의 정책은 **처리 실패 시 알림** 이다 — CorrectionOutcome.status 가
"corrected" 가 아니면 outcome 요약을 실어 발송한다(자동화 초기에는 사실상 매번).

office_rich_notify 해석 순서는 alarm_source 와 동일한 2단 fallback:
  1. poc.workflow_3.monitor.office_rich_notify   (정위치)
  2. poc.workflow_1.office_rich_notify           (legacy 위치 — 복사 전 과도기)
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
# office_rich_notify 로딩 (2단 fallback).
# ------------------------------------------------------------------


def _load_rich_notify():
    """send_cube_align_fail_info 를 정위치 → legacy 순서로 찾는다. 없으면 None."""
    integration = load_office_integration(
        "office_rich_notify",
        (
            ("poc.workflow_3.monitor.office_rich_notify", False),
            ("poc.workflow_1.office_rich_notify", True),
        ),
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


def build_outcome_summary(
    outcome, *, recording_dir: str = "", reregister_ratio_threshold: float | None = None
) -> str:
    """CorrectionOutcome 을 엔지니어용 한 줄 요약으로 만든다.

    outcome 이 None(보정 미수행: RECIPE_ID 없음, 사이클 중단 등)이어도 동작한다.
    matcher 모호도(second_ratio)가 있으면 값을 덧붙이고, reregister_ratio_threshold 가
    주어지고 그 값을 넘으면 '재등록 권장(모호 키)' 한 줄을 추가한다(임계 None=구 호출부면 권고 skip).
    """
    if outcome is None:
        parts = ["자동 보정 미수행(사이클 중단 또는 RECIPE_ID 없음) - 직접 확인 필요"]
    else:
        parts = [f"status={outcome.status}", f"path={outcome.path}", f"decision={outcome.key_decision}"]
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
    if recording_dir:
        parts.append(f"녹화={recording_dir}")
    return " | ".join(parts)


def notify_correction_outcome(
    eqp_id: str,
    recipe_id: str,
    outcome,
    *,
    recording_dir: str = "",
    enabled: bool = True,
    reregister_ratio_threshold: float | None = None,
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
    print(f"[INFO] cube 알림 발송(비차단): EQP_ID={eqp_id} | {summary}")


def send_detection_notify_async(eqp_id: str, recipe_id: str, *, enabled: bool = True) -> None:
    """감지 시점 cube 알림(기존 동작 호환, 기본 루프에서는 미사용)."""
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
    "build_outcome_summary",
    "close_alert_window",
    "notify_align_fail_popup",
    "notify_correction_outcome",
    "send_detection_notify_async",
    "show_popup_windows",
]
