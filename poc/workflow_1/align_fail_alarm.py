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
from poc.workflow_1.util import env_flag, env_int

POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 60)
POPUP_ENABLED_DEFAULT = True
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


def _collapse_rows_by_tool(fails) -> dict[str, dict]:
    """같은 EQP_ID 의 여러 알람 row 를 하나로 합친다 (첫 번째 row 채택)."""
    by_tool: dict[str, dict] = {}
    for row in _iter_alarm_rows(fails):
        eqp_id = str(_row_value(row, "EQP_ID", "eqp_id", "tool_name") or "").strip()
        if not eqp_id:
            print("[WARNING] EQP_ID 없는 Align Fail row 발견 — 건너뜀")
            continue
        if eqp_id in by_tool:
            continue
        by_tool[eqp_id] = {
            "eqp_id": eqp_id,
            "alarm_time": _row_value(row, "UTC9", "utc9", "alarm_time"),
            "alarm_name": str(
                _row_value(row, "ALARM_NAME", "alarm_name", "DESCRIPTION")
                or "Align Fail"
            ).strip(),
            "alid": str(_row_value(row, "ALID", "alarm_id", "al_id") or "9006").strip(),
        }
    return by_tool


def process_fail_rows(
    fails,
    active_tools: set[str],
    popup_enabled: bool = True,
) -> int:
    """EQP_ID 기준으로 edge-triggered 알림을 수행한다.

    - 같은 EQP_ID 에서 여러 알람 code 가 와도 하나로 취급.
    - 이전 poll 에서 이미 활성이던 EQP_ID 는 다시 알리지 않음.
    - 이번 poll 에 사라진 EQP_ID 는 `active_tools` 에서 제거 (복구 시 재알림 가능).

    `active_tools` 는 in-place 로 갱신된다. 새로 알린 개수를 반환.
    """
    by_tool = _collapse_rows_by_tool(fails)
    current_tools = set(by_tool.keys())

    new_tools = current_tools - active_tools
    cleared_tools = active_tools - current_tools

    for eqp_id in sorted(cleared_tools):
        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
    active_tools.difference_update(cleared_tools)

    newly_handled = 0
    for eqp_id in sorted(new_tools):
        info = by_tool[eqp_id]
        alarm_time = info["alarm_time"]
        alarm_name = info["alarm_name"]
        alid = info["alid"]

        print(
            f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
            f"ALID={alid}, 시각={alarm_time}"
        )
        append_alarm_record(eqp_id, str(alarm_time or ""), alarm_name, alid)
        if popup_enabled:
            notify_align_fail(eqp_id, str(alarm_time or ""), alarm_name)

        active_tools.add(eqp_id)
        newly_handled += 1

    return newly_handled


def monitor_loop(popup_enabled: bool | None = None) -> None:
    """메인 감지 루프 — 1분 주기 기본.

    popup_enabled:
      - True  → Align Fail 감지 시 Windows MessageBox 팝업 표시 (기본)
      - False → 텍스트 로그만 기록, 팝업 없음
      - None  → 환경변수 `ALIGN_FAIL_POPUP` 로 결정 (미설정 시 True)
    """
    if popup_enabled is None:
        popup_enabled = env_flag("ALIGN_FAIL_POPUP", POPUP_ENABLED_DEFAULT)

    active_tools: set[str] = set()

    print(f"[INFO] Align Fail 감지 시작 (주기={POLL_INTERVAL_SEC}s, 팝업={'on' if popup_enabled else 'off'})")
    print(f"[INFO] 누적 로그 파일: {ALARM_LOG_PATH}")
    print("[INFO] 같은 EQP_ID 의 중복 알람은 한 번만 알립니다 (해제 후 재감지 시 재알림).")

    while True:
        try:
            alarms = get_cdsem_alarms()
            fails = filter_align_fail(alarms)

            if _alarm_rows_empty(fails):
                if active_tools:
                    for eqp_id in sorted(active_tools):
                        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
                    active_tools.clear()
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} — Align Fail 없음")
            else:
                count = process_fail_rows(fails, active_tools, popup_enabled=popup_enabled)
                if count == 0:
                    print(
                        f"[INFO] {datetime.now().strftime('%H:%M:%S')} — "
                        f"신규 없음 (활성 {len(active_tools)}대 유지)"
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
