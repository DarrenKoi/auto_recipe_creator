"""align fail 알림 — Windows 팝업 + cube rich notification(처리 결과 중심).

팝업 헬퍼(`_show_popup_windows`/`close_alert_window`)는 align_fail_alarm_record
에서 이동했다. cube 알림은 엔지니어 소유 `office_rich_notify` 모듈에 위임하며,
workflow_3 의 정책은 **처리 실패 시 알림** 이다 — CorrectionOutcome.status 가
"corrected" 가 아니면 outcome 요약을 실어 발송한다(자동화 초기에는 사실상 매번).

office_rich_notify 는 정위치(poc.workflow_3.monitor.office_rich_notify)에서
로드한다(없으면 cube 알림 비활성, 텍스트 로그만).
"""

import inspect
import json
import os
import threading
import time
import uuid

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.monitor.integration_loader import load_office_integration

LOG_COMPONENT = "align_fail_notify"
# ponytail: 발송 정체 시 최대 64개만 유지한다. 슬롯 고갈을 막으려면 오피스 I/O timeout 이 필요하다.
_CUBE_SEND_SLOTS = threading.BoundedSemaphore(64)
# **반드시 가야 하는** 알림(사이클 결과, 엔지니어 조치 요청)의 outbox. 보내기 전에 파일로
# 적고 발송이 예외 없이 끝났을 때만 지운다 - 상한 도달/office 예외/프로세스 재시작 어느
# 쪽에서도 통보가 조용히 사라지지 않는다. 남은 파일은 pump 스레드 하나가 주기적으로 다시
# 보낸다(슬롯 해제에 맞춰 깨우는 방식은 '마지막 해제 직후 적재' 를 놓친다).
# 보장은 at-least-once 다: office 함수가 전송 후에 던지면 같은 알림이 한 번 더 갈 수 있다.
# 감지/진행 고지는 여기 넣지 않는다(늦게 가면 오히려 틀린 정보다).
_OUTBOX_DIR = LOG_DIR / "cube_outbox"
_OUTBOX_RETRY_SEC = 30.0
_OUTBOX_MAX_AGE_SEC = 24 * 3600.0     # 이보다 묵은 알림은 보내지 않고 error 로그로 남긴다.
_OUTBOX_LATE_SEC = 120.0              # 이보다 늦게 나가면 요약 앞에 지연 시간을 붙인다.
_OUTBOX_LOCK = threading.Lock()
_OUTBOX_IN_FLIGHT: set = set()
_OUTBOX_PUMP: threading.Thread | None = None
_OUTBOX_PUMP_ENABLED = True           # 테스트(poc/conftest.py)만 끈다.

# 알림 팝업 제목 — 표시(notify_align_fail)와 닫기(close_alert_window)가 같은 값을
# 써야 창을 찾을 수 있다.
ALERT_POPUP_TITLE = "CD-SEM Align Fail 감지"
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"

# 점유 tool 관련 outcome status (2026-08-18). 둘 다 "corrected" 가 아니므로 engineer
# watch(cycle.py) 와 cube 발송 분기의 기존 **정확 비교**를 그대로 통과한다 - 즉 알림이
# 나가고 녹화도 계속된다. 이것이 의도다: 점유자와 알람 담당자는 다른 사람일 수 있고,
# 점유자가 알람을 남긴 채 자리를 뜨면 생략된 알림은 아무에게도 도달하지 않는다.
VIEW_ONLY_OBSERVATION = "view_only_observation"   # 다른 엔지니어 점유 - 관전·녹화만.
CORRECTED_UNVERIFIED = "corrected_unverified"     # 점유 미상 - 보정했으나 반영 미확인.

# 접속 직후 화면에 align fail 다이얼로그가 있는지 먼저 본 결과 (2026-09-17). 알람 피드는
# 해제된 알람도 계속 돌려주므로(이벤트 로그), 큐에서 기다리는 동안 엔지니어가 이미 해결한
# tool 에 들어가 측정 중인 장비를 클릭하지 않으려는 것이다. 둘 다 클릭 없이 끝나지만
# "corrected" 가 아니므로 cube 는 나간다 - VLM 이 진짜 다이얼로그를 놓쳤다면 그 알림이
# 멈춘 장비를 사람에게 되돌려 주는 유일한 경로다.
ALIGN_FAIL_CLEARED = "align_fail_cleared"          # 다이얼로그 없음 - 이미 해결로 보고 종료.
ALIGN_FAIL_UNCONFIRMED = "align_fail_unconfirmed"  # 다른 창/판독 실패 - 확인 못 해 클릭 안 함.
# OK 뒤 다음 위치(wafer 당 OM/SEM 각 2~3 point)에서 추가 보정 상한을 넘겨 또 fail (2026-09-29).
NEXT_POINT_LIMIT = "escalated_next_point_limit"


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


# 자동 보정을 못 하고 끝난 status -> (원인, 엔지니어 요구 행동).
# CorrectionOutcome.status 는 **정확 비교 전용**(monitor 가 치환하는 status 가 있어
# 접두사 매칭이 새는 자리다)이라 fallback_* 도 접두사로 묶지 않고 4가지를 다 적는다.
# 여기 없는 status 는 요구 행동 줄 없이 종전처럼 status= 로만 나간다.
_UNCORRECTED_ACTIONS = {
    ALIGN_FAIL_CLEARED: (
        "접속 시 align fail 다이얼로그 없음(이미 해결된 것으로 보고 클릭 안 함)",
        "장비가 아직 멈춰 있으면 직접 확인해주세요",
    ),
    ALIGN_FAIL_UNCONFIRMED: (
        "접속 시 align fail 다이얼로그를 확인 못 함(다른 창이거나 판독 실패, 클릭 안 함)",
        "화면 확인 후 직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    NEXT_POINT_LIMIT: (
        "OK 뒤 다음 위치에서 align fail 이 추가 보정 상한을 넘겨 계속 발생",
        "화면 확인 후 남은 위치의 align point 를 직접 잡고 OK 를 눌러주세요",
    ),
    "escalated_invalid_geometry": (
        "저장 이미지와 live SEM 영역의 크기 비율이 맞지 않아 자동 보정 보류",
        "SEM 영상 영역과 배율을 확인한 뒤 직접 align point 를 잡아주세요",
    ),
    "no_assets": (
        "등록 align key 자산 없음(rcp/consensus 미확보)",
        "직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    "escalated_key_not_visible": (
        "align key 가 현재 화면에 없음(주변 탐색 비활성)",
        "직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    "fallback_exhausted": (
        "주변 탐색을 다 돌았으나 align key 미검출",
        "직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    "fallback_escalated": (
        "주변 탐색 중 점수가 계속 낮아 자동 판단 중단",
        "직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    "fallback_best_candidate": (
        "주변 탐색에서 후보는 나왔으나 확신 임계 미달",
        "화면 위치를 확인한 뒤 align point 를 잡고 OK 를 눌러주세요",
    ),
    "fallback_match": (
        "주변 탐색에서 align key 검출(자동 확정은 안 함)",
        "화면 위치를 확인한 뒤 align point 를 잡고 OK 를 눌러주세요",
    ),
    "escalated_ambiguous_key": (
        "align key 가 보이나 닮은 곳이 많아 단정 불가(만성 모호 - 재등록 대상)",
        "직접 align point 를 잡고 OK 를 눌러주세요",
    ),
    "escalated_reposition_unconverged": (
        "align point 로 여러 번 옮겼으나 중심에 맞지 않음(OK 안 누름)",
        "위치 확인 후 align point 를 잡고 OK 를 눌러주세요",
    ),
    "escalated_no_ok": (
        "align point 이동은 했으나 OK 버튼을 찾지 못함",
        "위치 확인 후 OK 를 눌러주세요",
    ),
    "ok_detect_error": (
        "OK 버튼 탐지 중 예외",
        "위치 확인 후 OK 를 눌러주세요",
    ),
}


def _match_index(outcome) -> str:
    """matcher 점수를 임계와 나란히 놓은 한 조각 — "왜 못 잡았나" 의 정량 근거.

    새로 계산하지 않는다. correct_align_fail 이 history 에 이미 적어 둔 paused_match
    레코드와 outcome 의 모호도 필드를 읽을 뿐이다. 그 레코드가 없으면(no_assets 처럼
    매칭 전에 끝난 경로) 빈 문자열이라 요약 모양이 종전 그대로다.
    """
    record = None
    for entry in reversed(getattr(outcome, "history", None) or []):
        if isinstance(entry, dict) and entry.get("stage") == "paused_match":
            record = entry
            break
    if record is None:
        return ""
    score, scale = record.get("score"), record.get("best_scale")
    if score is None or scale is None:
        return ""
    from poc.workflow_3.align.live_search import MIN_CONFIRM_SCALE
    from poc.workflow_3.align.matching.engine import STRUCTURE_POLICY

    parts = [
        f"매칭점수={score:.3f}(match임계 {STRUCTURE_POLICY.ensemble_match_threshold:.3f})",
        f"scale={scale:.2f}(최소 {MIN_CONFIRM_SCALE:.2f})",
    ]
    return " ".join(parts)


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
        elif outcome.status == VIEW_ONLY_OBSERVATION:
            parts.append("다른 엔지니어가 tool 점유 중 - 자동 보정 없이 관전·녹화만 수행")
        elif outcome.status == CORRECTED_UNVERIFIED:
            parts.append(
                "점유 여부 확인 불가 - 보정을 시도했으나 실제 반영 여부는 미확인, "
                "장비에서 직접 확인 필요"
            )
        else:
            # 자동 보정을 못 하고 끝난 경로 - 원인과 요구 행동을 맨 앞에 둔다.
            # 지금까지는 status=no_assets 처럼 코드값만 나가 엔지니어가 무엇을 해야
            # 하는지 알 수 없었다(이 알림이 유일한 통보라 그 자리에서 읽혀야 한다).
            reason_action = _UNCORRECTED_ACTIONS.get(outcome.status)
            if reason_action is not None:
                parts.append(f"자동 보정 불가: {reason_action[0]}")
                parts.append(reason_action[1])
        parts += [f"status={outcome.status}", f"path={outcome.path}", f"decision={outcome.key_decision}"]
        index = _match_index(outcome)
        if index:
            parts.append(index)
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


_OUTBOX_PUT_ATTEMPTS = 5


def _outbox_put(eqp_id: str, recipe_id: str, summary: str | None):
    """must-deliver 알림을 outbox 파일로 적는다. 실패하면 None(그 알림은 1회 시도로 강등).

    Windows 는 백신/색인기가 방금 쓴 파일을 잠깐 쥐면 rename 이 거부된다 - 짧게 다시
    시도하고, 그래도 안 되면 원자성 없이 직접 쓴다(반쯤 쓰인 파일은 pump 가 건너뛰고
    하루 뒤 치운다). 어떤 경우에도 예외를 올리지 않는다.
    """
    text = json.dumps(
        {"eqp_id": eqp_id, "recipe_id": recipe_id, "summary": summary, "created": time.time()},
        ensure_ascii=False,
    )
    path = _OUTBOX_DIR / f"{time.time():.3f}_{uuid.uuid4().hex[:8]}.json"
    tmp = path.with_suffix(".tmp")
    try:
        _OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        for _ in range(_OUTBOX_PUT_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return path
            except OSError:
                time.sleep(0.05)
        path.write_text(text, encoding="utf-8")
        _outbox_remove(tmp)
        return path
    except Exception as exc:
        print(f"[ERROR] cube outbox 기록 실패 - 이 알림은 재시도 없이 1회만 시도한다: "
              f"EQP_ID={eqp_id} | {summary} | {exc}")
        try:
            log_work2_event(
                component=LOG_COMPONENT, message="cube_outbox_write_failed", level="error",
                eqp_id=eqp_id, recipe_id=recipe_id, summary=summary, error=str(exc),
            )
        except Exception:
            pass
        return None


def _outbox_remove(path) -> None:
    """발송이 끝난 outbox 파일을 지운다. 예외를 올리지 않는다.

    Windows 는 백신/색인기가 파일을 잠깐 쥐면 삭제가 거부된다 - 짧게 다시 시도하고,
    끝내 못 지우면 그 알림은 다음 pump 주기에 한 번 더 나간다(유실보다 중복이 낫다).
    """
    for _ in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            time.sleep(0.1)
    print(f"[ERROR] cube outbox 파일 삭제 실패 - 같은 알림이 다시 나갈 수 있음: {path}")


def _ensure_outbox_pump() -> None:
    """outbox 재발송 스레드를 한 번만 띄운다. 예외를 올리지 않는다.

    스레드를 못 띄워도(자원 고갈) 호출자는 계속 가야 한다 - 이 함수는 사이클 finally 의
    결과 통보에서 불리고, 여기서 새는 예외는 그 뒤의 입력 해제/tool 닫기를 건너뛰게 한다.
    실패하면 핸들이 비어 있으므로 다음 호출이 다시 시도한다.
    """
    global _OUTBOX_PUMP
    if not _OUTBOX_PUMP_ENABLED:
        return
    try:
        with _OUTBOX_LOCK:
            if _OUTBOX_PUMP is not None and _OUTBOX_PUMP.is_alive():
                return

            def _pump():
                while True:
                    try:
                        retry_outbox_once()
                    except Exception as exc:
                        print(f"[WARNING] cube outbox 재발송 예외: {exc}")
                    time.sleep(_OUTBOX_RETRY_SEC)

            thread = threading.Thread(target=_pump, name="cube_outbox_pump", daemon=True)
            thread.start()
            _OUTBOX_PUMP = thread
    except Exception as exc:
        # 핸들은 start 성공 뒤에만 넣으므로 여기서 비울 것이 없다(lock 밖에서 비우면
        # 그 사이 다른 호출자가 띄운 pump 의 핸들을 지워 pump 가 둘이 된다).
        print(f"[ERROR] cube outbox 재발송 스레드 시작 실패(다음 발송 때 재시도): {exc}")


def start_cube_outbox(*, enabled: bool = True) -> None:
    """모니터 시작 시 한 번 부른다 - 재시작 전에 못 나간 알림을 새 알람을 기다리지 않고 보낸다.

    cube 가 꺼져 있거나 어댑터가 없으면 띄우지 않는다(파일은 그대로 남는다).
    """
    if not enabled or not RICH_NOTIFY_AVAILABLE:
        return
    try:
        pending = len(list(_OUTBOX_DIR.glob("*.json")))
    except OSError:
        pending = 0
    if pending:
        print(f"[WARNING] 이전 실행에서 못 나간 cube 알림 {pending}건 - 재발송합니다: {_OUTBOX_DIR}")
    _ensure_outbox_pump()


def _outbox_sweep_junk(now: float) -> None:
    """하루 넘게 남은 .tmp / 읽을 수 없는 파일을 치운다(디스크가 끝없이 차지 않게)."""
    for path in _OUTBOX_DIR.iterdir():
        if path in _OUTBOX_IN_FLIGHT:
            continue
        try:
            if now - path.stat().st_mtime <= _OUTBOX_MAX_AGE_SEC:
                continue
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))["created"]
                continue  # 정상 파일의 만료는 retry_outbox_once 가 로그와 함께 처리한다
        except OSError:
            continue
        except (ValueError, KeyError, TypeError):
            pass
        print(f"[ERROR] cube outbox 의 읽을 수 없는 파일 폐기: {path.name}")
        _outbox_remove(path)


def retry_outbox_once() -> int:
    """outbox 에 남은 알림을 한 번씩 다시 보낸다. 발송을 시작한 건수를 돌려준다."""
    if not _OUTBOX_DIR.is_dir():
        return 0
    started = 0
    _outbox_sweep_junk(time.time())
    for path in sorted(_OUTBOX_DIR.glob("*.json")):
        if path in _OUTBOX_IN_FLIGHT:
            continue  # 발송 중인 파일은 열지 않는다 - Windows 는 열린 파일을 못 지운다
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - float(data["created"])
            eqp_id, recipe_id, summary = data["eqp_id"], data["recipe_id"], data["summary"]
        except (OSError, ValueError, KeyError, TypeError):
            continue  # 방금 발송돼 지워졌거나 읽을 수 없는 파일(묵으면 sweep 이 치운다)
        if age > _OUTBOX_MAX_AGE_SEC:
            print(f"[ERROR] cube 알림 미발송 폐기({age / 3600:.0f}h 경과): EQP_ID={eqp_id} | {summary}")
            log_work2_event(
                component=LOG_COMPONENT, message="cube_outbox_expired", level="error",
                eqp_id=eqp_id, recipe_id=recipe_id, summary=summary,
            )
            _outbox_remove(path)
            continue
        if summary is not None and age > _OUTBOX_LATE_SEC:
            summary = f"[지연 {age / 60:.0f}분] {summary}"
        if _send_cube_async(eqp_id, recipe_id, summary, _entry=path):
            started += 1
    return started


def _send_cube_async(
    eqp_id: str, recipe_id: str, summary: str | None, *,
    must_deliver: bool = False, _entry=None,
) -> bool:
    """office cube 함수를 데몬 스레드에서 호출한다(루프 비차단).

    office 함수가 summary 인자를 받으면 요약을 함께 보내고, 기존 2-인자 시그니처면
    생략한다(README: office 함수에 optional summary 추가 권장).
    summary=None 은 감지 알림이며, 모든 발송 경로가 같은 동시 실행 상한을 쓴다.
    반환값은 스레드 시작 여부이며 외부 발송 완료를 의미하지 않는다.
    must_deliver=True 면 outbox 에 먼저 적는다 - 상한에 걸리거나 office 함수가 던져도
    파일이 남아 pump 가 다시 보낸다(그때 반환은 False - 아직 안 나갔다).
    _entry 는 pump 전용(이미 outbox 에 있는 파일의 재발송)이다.
    """
    retry = _entry is not None
    if must_deliver and _entry is None:
        # 파일을 먼저 남기고 pump 를 띄운다 - pump 시작이 실패해도 알림은 디스크에 있다.
        _entry = _outbox_put(eqp_id, recipe_id, summary)
        _ensure_outbox_pump()
    if _entry is not None:
        with _OUTBOX_LOCK:
            if _entry in _OUTBOX_IN_FLIGHT:
                return False
            _OUTBOX_IN_FLIGHT.add(_entry)
        # 발송 스레드는 파일을 지운 **뒤** in-flight 에서 뺀다 - 그래서 여기서 파일이 없으면
        # 그 사이에 이미 나간 것이다(pump 가 읽은 직후 발송이 끝난 경우의 중복 방지).
        if not _entry.exists():
            _OUTBOX_IN_FLIGHT.discard(_entry)
            return False

    slots = _CUBE_SEND_SLOTS
    if not slots.acquire(blocking=False):
        _OUTBOX_IN_FLIGHT.discard(_entry)
        if retry:
            return False  # pump 재시도 - 첫 보류 때 이미 기록했다(30s 마다 반복 로그 금지)
        if _entry is not None:
            print(f"[ERROR] cube 발송 상한 도달(발송 보류 - {_OUTBOX_RETRY_SEC:.0f}s 마다 재시도): "
                  f"EQP_ID={eqp_id} recipe={recipe_id} | {summary}")
        else:
            print(f"[WARNING] cube 발송 상한 도달(발송 생략): EQP_ID={eqp_id} recipe={recipe_id}")
        log_work2_event(
            component=LOG_COMPONENT, message="cube_sender_limit",
            level="error" if _entry is not None else "warning",
            eqp_id=eqp_id, recipe_id=recipe_id, summary=summary, queued=_entry is not None,
        )
        return False

    def _run():
        try:
            if summary is None:
                _SEND_CUBE_FN(eqp_id, recipe_id)
            else:
                params = inspect.signature(_SEND_CUBE_FN).parameters
                if "summary" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    _SEND_CUBE_FN(eqp_id, recipe_id, summary=summary)
                else:
                    _SEND_CUBE_FN(eqp_id, recipe_id)
            if _entry is not None:
                _outbox_remove(_entry)
                if retry:
                    print(f"[INFO] 보류된 cube 알림 재발송 완료: EQP_ID={eqp_id} | {summary}")
        except Exception as exc:
            if _entry is not None:
                print(f"[ERROR] cube rich notify 예외 - outbox 에 남겨 재시도: "
                      f"EQP_ID={eqp_id} error={exc}")
            else:
                print(f"[WARNING] cube rich notify 예외: {exc}")
        finally:
            slots.release()
            _OUTBOX_IN_FLIGHT.discard(_entry)

    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception as exc:
        slots.release()
        _OUTBOX_IN_FLIGHT.discard(_entry)
        print(f"[WARNING] cube 발송 스레드 시작 실패: {exc}")
        return False
    return True


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

    if _send_cube_async(eqp_id, recipe_id, summary, must_deliver=True):
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
    if _send_cube_async(eqp_id, recipe_id, summary):
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
        """결과 알림을 1회 처리한다. 최초 처리면 True, 중복이면 False(외부 발송 완료와 무관)."""
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


def notify_operator_action(eqp_id: str, recipe_id: str, summary: str, *, enabled: bool = True) -> None:
    """사이클 결과와 별개로 엔지니어 조치가 필요한 사실을 알린다(유실 금지).

    예: tool 창이 안 닫혀 세션이 남음, rcp 다운로더 정체. 결과 알림이 `corrected` 로
    생략된 알람에서도 나가야 하므로 CycleNotifier 의 1회 게이트를 거치지 않는다.
    teardown/finally 에서 불리므로 어떤 경우에도 예외를 올리지 않는다.
    """
    print(f"[ERROR] 엔지니어 조치 필요: EQP_ID={eqp_id} | {summary}")
    try:
        log_work2_event(
            component=LOG_COMPONENT, message="operator_action_required", level="error",
            eqp_id=eqp_id, recipe_id=recipe_id, summary=summary,
        )
        if enabled and RICH_NOTIFY_AVAILABLE:
            _send_cube_async(eqp_id, recipe_id, summary, must_deliver=True)
    except Exception as exc:
        print(f"[ERROR] 조치 알림 발송 예외: {exc}")


def notify_teardown_failures(eqp_id: str, recipe_id: str, failures, *, enabled: bool = True) -> None:
    """teardown 실패 중 엔지니어가 알아야 하는 것(tool 창이 열린 채 남음)을 알린다."""
    if any(name == "close_tool" for name, _ in failures):
        notify_operator_action(
            eqp_id, recipe_id,
            "tool 창 닫기 실패 - Remote Monitoring 창(세션)이 열린 채 남았습니다. 수동으로 닫아 주세요",
            enabled=enabled,
        )


def send_detection_notify_async(eqp_id: str, recipe_id: str, *, enabled: bool = True) -> None:
    """감지 시점 cube 알림 — "지금 이 장비에 자동화가 들어간다"는 사전 고지.

    두 모니터(align_fail_monitor / align_fail_monitor_only_check)가 알람을 잡은 직후
    호출한다. 처리 *결과* 알림(notify_correction_outcome)과는 목적이 달라 둘 다 나간다.
    """
    if not enabled or not RICH_NOTIFY_AVAILABLE:
        return

    _send_cube_async(eqp_id, recipe_id, None)


__all__ = [
    "ALARM_LOG_PATH",
    "ALERT_POPUP_TITLE",
    "CORRECTED_UNVERIFIED",
    "RICH_NOTIFY_AVAILABLE",
    "VIEW_ONLY_OBSERVATION",
    "CycleNotifier",
    "build_outcome_summary",
    "close_alert_window",
    "notify_align_fail_popup",
    "notify_correction_outcome",
    "notify_operator_action",
    "notify_teardown_failures",
    "send_detection_notify_async",
    "send_progress_notify",
    "start_cube_outbox",
    "show_popup_windows",
]
