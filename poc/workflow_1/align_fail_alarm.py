"""Align Fail 알람 감지 전용 스크립트.

1분 주기로 CD-SEM 알람을 조회하여 ALID=9006 (Align Fail) 이 감지되면:
  1. 텍스트 파일(`poc/workflow_1/logs/align_fail_alarms.txt`) 에 누적 기록
  2. Windows 팝업(MessageBox) 으로 알림 표시

CCTV/캡처/GUI 자동화 로직은 포함하지 않는다. 순수 감지 + 알림 전용.

사용법:
  uv run python poc/workflow_1/align_fail_alarm.py
"""

import threading
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from poc.workflow_1 import LOG_DIR
from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms
from poc.workflow_1.util import env_int

POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 60)
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"


def _alarm_rows_empty(rows) -> bool:
    """DataFrame 또는 iterable alarm rows 의 empty 상태를 판별한다."""
    if rows is None:
        return True
    if hasattr(rows, "empty"):
        try:
            return bool(rows.empty)
        except Exception:
            pass
    try:
        return len(rows) == 0
    except TypeError:
        return False


def _iter_alarm_rows(rows):
    """DataFrame / list[dict] / iterable row 를 순회 가능한 형태로 변환한다."""
    if rows is None:
        return []
    itertuples = getattr(rows, "itertuples", None)
    if callable(itertuples):
        return itertuples(index=False)
    if isinstance(rows, Mapping):
        return [rows]
    if isinstance(rows, Iterable):
        return rows
    return [rows]


def _row_value(row, *field_names: str):
    """row 객체에서 첫 번째로 발견한 필드 값을 반환한다."""
    if isinstance(row, Mapping):
        for field_name in field_names:
            if field_name in row:
                return row[field_name]
    for field_name in field_names:
        if hasattr(row, field_name):
            return getattr(row, field_name)
    return None


def append_alarm_record(eqp_id: str, alarm_time: str, alarm_name: str, alid: str) -> None:
    """감지된 Align Fail 을 텍스트 파일에 누적 기록한다."""
    ALARM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{detected_at} | EQP_ID={eqp_id} | ALID={alid} | "
        f"ALARM_NAME={alarm_name} | UTC9={alarm_time}\n"
    )
    with ALARM_LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(line)
    print(f"[INFO] 기록 완료 → {ALARM_LOG_PATH}")


def _show_popup_windows(title: str, message: str) -> None:
    """Windows MessageBox 를 데몬 스레드에서 띄운다 (루프 비차단)."""
    try:
        import ctypes

        MB_ICONWARNING = 0x00000030
        MB_SYSTEMMODAL = 0x00001000
        MB_SETFOREGROUND = 0x00010000
        flags = MB_ICONWARNING | MB_SYSTEMMODAL | MB_SETFOREGROUND

        def _run():
            try:
                ctypes.windll.user32.MessageBoxW(0, message, title, flags)
            except Exception as exc:
                print(f"[WARNING] Windows 팝업 실패: {exc}")

        threading.Thread(target=_run, daemon=True).start()
    except AttributeError:
        print(f"[INFO] 현재 OS 에서 MessageBox 미지원 — 콘솔 알림만: {title} | {message}")
    except Exception as exc:
        print(f"[WARNING] 팝업 표시 실패: {exc}")


def notify_align_fail(eqp_id: str, alarm_time: str, alarm_name: str) -> None:
    """Align Fail 감지 시 팝업 알림."""
    title = "CD-SEM Align Fail 감지"
    message = (
        f"EQP_ID: {eqp_id}\n"
        f"ALARM : {alarm_name}\n"
        f"TIME  : {alarm_time}\n\n"
        f"로그: {ALARM_LOG_PATH}"
    )
    _show_popup_windows(title, message)


def process_fail_rows(fails, already_handled: set[str]) -> int:
    """감지된 fails 에 대해 로깅 + 팝업을 수행한다. 새로 처리된 수를 반환."""
    newly_handled = 0
    for row in _iter_alarm_rows(fails):
        eqp_id = str(_row_value(row, "EQP_ID", "eqp_id", "tool_name") or "").strip()
        alarm_time = _row_value(row, "UTC9", "utc9", "alarm_time")
        alarm_name = str(
            _row_value(row, "ALARM_NAME", "alarm_name", "DESCRIPTION") or "Align Fail"
        ).strip()
        alid = str(_row_value(row, "ALID", "alarm_id", "al_id") or "9006").strip()

        if not eqp_id:
            print("[WARNING] EQP_ID 없는 Align Fail row 발견 — 건너뜀")
            continue

        alarm_key = f"{eqp_id}:{alarm_time}" if alarm_time else eqp_id
        if alarm_key in already_handled:
            continue

        print(
            f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
            f"ALID={alid}, 시각={alarm_time}"
        )
        append_alarm_record(eqp_id, str(alarm_time or ""), alarm_name, alid)
        notify_align_fail(eqp_id, str(alarm_time or ""), alarm_name)

        already_handled.add(alarm_key)
        newly_handled += 1

    return newly_handled


def monitor_loop() -> None:
    """메인 감지 루프 — 1분 주기 기본."""
    already_handled: set[str] = set()

    print(f"[INFO] Align Fail 감지 시작 (주기={POLL_INTERVAL_SEC}s)")
    print(f"[INFO] 누적 로그 파일: {ALARM_LOG_PATH}")

    while True:
        try:
            alarms = get_cdsem_alarms()
            fails = filter_align_fail(alarms)

            if _alarm_rows_empty(fails):
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} — Align Fail 없음")
            else:
                count = process_fail_rows(fails, already_handled)
                if count == 0:
                    print(
                        f"[INFO] {datetime.now().strftime('%H:%M:%S')} — "
                        f"모두 이전 감지분 (신규 없음)"
                    )
        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break
        except Exception as exc:
            print(f"[ERROR] 감지 루프 예외: {exc}")

        try:
            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break

    print("[INFO] 감지 종료")


if __name__ == "__main__":
    monitor_loop()
