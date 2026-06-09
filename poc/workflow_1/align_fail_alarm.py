"""Align Fail 알람 감지 전용 스크립트.

1분 주기로 CD-SEM 알람을 조회하여 ALID=9006 (Align Fail) 이 감지되면:
  1. 텍스트 파일(`poc/workflow_1/logs/align_fail_alarms.txt`) 에 누적 기록
  2. Windows 팝업(MessageBox) 으로 알림 표시
  3. RECIPE_ID 가 있을 때만 해당 EQP_ID 장비로 RCS 접속(tool 더블클릭) — `workflow_select_tool` 위임 (기본 on)

RECIPE_ID 가 있으면 등록 recipe 의 자동 파이프라인(이미지 저장 → 접속 → live SEM
탐색)을 진행하므로 장비에 접속한다. RECIPE_ID 가 없으면 자동 접속하지 않고
엔지니어가 직접 접속한다. RCS 는 이미 로그인되어 있다고 가정한다. 장비 접속은
`ALIGN_FAIL_CONNECT_TOOL` 로 on/off, `ALIGN_FAIL_CONNECT_ACTION=off` 로 클릭 없는
dry-run 전환이 가능하다.

workflow_1 경계: 감지 + 알림 + 장비 접속(=Tool 화면 진입)까지 책임진다. 이후
align key CV 탐색은 workflow_2 가 다운로드된 이미지를 읽어 수행한다(파일로 디커플링).

사용법:
  uv run python poc/workflow_1/align_fail_alarm.py
"""

import threading
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

import pandas as pd

from poc.workflow_1 import LOG_DIR
from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms # pyright: ignore[reportMissingImports]
from poc.workflow_3.util import env_flag, env_int

# rich notify 는 선택 의존성. requests/PIL/ftplib 등이 빠져도 본체는 계속 동작해야 한다.
try:
    from poc.workflow_1.office_rich_notify import send_cube_align_fail_info # pyright: ignore[reportMissingImports]

    RICH_NOTIFY_AVAILABLE = True
except Exception as _rich_notify_import_exc:
    send_cube_align_fail_info = None
    RICH_NOTIFY_AVAILABLE = False
    print(f"[WARNING] office_rich_notify 모듈 로드 실패 - 텍스트 로그/팝업만 동작합니다: {_rich_notify_import_exc}")

# 장비 자동 접속(tool 더블클릭)도 선택 의존성. pywinauto/VLM 머신이 없는 환경
# (macOS 개발 등) 에서는 import 가 실패해도 감지/로그/팝업은 계속 동작해야 한다.
try:
    from poc.workflow_3.rcs.workflow_select_tool import connect_to_tool # pyright: ignore[reportMissingImports]

    CONNECT_TOOL_AVAILABLE = True
except Exception as _connect_tool_import_exc:
    connect_to_tool = None
    CONNECT_TOOL_AVAILABLE = False
    print(f"[WARNING] workflow_select_tool 로드 실패 - 장비 자동 접속 비활성화: {_connect_tool_import_exc}")

# Align fail 시 RCS tool 창을 1장 캡처해 captured_img_from_rcs 로 적재하는 코어는
# rcs_screenshot 에 분리돼 있다(독립 실행/테스트 가능). 윈도우/캡처 의존성이 없는
# 환경에서도 본체는 계속 동작해야 하므로 import 실패를 흡수한다.
try:
    from poc.workflow_3.rcs.rcs_screenshot import capture_and_close_rcs_window # pyright: ignore[reportMissingImports]

    CAPTURE_RCS_AVAILABLE = True
except Exception as _capture_rcs_import_exc:
    capture_and_close_rcs_window = None
    CAPTURE_RCS_AVAILABLE = False
    print(f"[WARNING] rcs_screenshot 로드 실패 - 화면 캡처 비활성화: {_capture_rcs_import_exc}")

POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 10)
# 감지 look-back 윈도우(초). poll 주기와 분리해, 알람 보고 지연이 있어도 놓치지 않게 한다.
DETECTION_WINDOW_SEC = env_int("ALIGN_FAIL_WINDOW_SEC", 60)
POPUP_ENABLED_DEFAULT = True
RICH_NOTIFY_ENABLED_DEFAULT = True
# Align Fail 감지 시 해당 EQP_ID 장비로 RCS 접속(tool 더블클릭) 을 시도할지. 기본 on.
CONNECT_TOOL_ENABLED_DEFAULT = True
# 접속 시 실제 더블클릭 수행 여부. off 면 인식/디버그 저장만 하고 클릭은 생략(dry-run).
CONNECT_TOOL_ACTION_DEFAULT = True
# 접속 시 메인 RCS 창 탐색 타임아웃(초). 한 번만 느슨하게 시도하고 실패하면 엔지니어가
# 직접 접속하는 정책이라 짧게 둔다(정상이면 즉시 발견, 실패 케이스에서만 루프 블록 단축).
CONNECT_TOOL_WINDOW_TIMEOUT_SEC = env_int("ALIGN_FAIL_CONNECT_WINDOW_TIMEOUT_SEC", 3)
# 팝업 자동 종료 시간(초). 0 이면 사용자가 닫을 때까지 유지. 기본 60초.
POPUP_TIMEOUT_SEC = env_int("ALIGN_FAIL_POPUP_TIMEOUT_SEC", 60)
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"

# 접속 후 RCS tool 창을 1장 캡처할지. 기본 on. (캡처 코어는 rcs_screenshot 에 있음.)
CAPTURE_RCS_ENABLED_DEFAULT = True


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


def filter_rows_within_window(rows: "pd.DataFrame", window_sec: int) -> "pd.DataFrame":
    """UTC9 가 현재 시각 기준 `window_sec` 이내인 row 만 남긴다.

    UTC9 가 비어있거나 파싱 불가한 row 는 제외한다.
    """
    if window_sec <= 0 or rows is None or rows.empty or "UTC9" not in rows.columns:
        return rows

    cutoff = datetime.now() - timedelta(seconds=window_sec)
    timestamps = pd.to_datetime(rows["UTC9"], errors="coerce")
    mask = timestamps.notna() & (timestamps >= cutoff)
    return rows[mask].reset_index(drop=True)


def append_alarm_record(
    eqp_id: str,
    alarm_time: str,
    alarm_name: str,
    alid: str,
    recipe_id: str = "",
    operation_desc: str = "",
    lot_type_cd: str = "",
) -> None:
    """감지된 Align Fail 을 텍스트 파일에 누적 기록한다."""
    ALARM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{detected_at} | EQP_ID={eqp_id} | ALID={alid} | "
        f"ALARM_NAME={alarm_name} | UTC9={alarm_time} | "
        f"RECIPE_ID={recipe_id} | OPERATION_DESC={operation_desc} | "
        f"LOT_TYPE_CD={lot_type_cd}\n"
    )
    with ALARM_LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(line)
    print(f"[INFO] 기록 완료 → {ALARM_LOG_PATH}")


def _show_popup_windows(title: str, message: str) -> None:
    """Windows MessageBox 를 데몬 스레드에서 띄운다 (루프 비차단).

    `POPUP_TIMEOUT_SEC` > 0 이면 해당 시간 후 팝업을 자동으로 닫는다.
    자동 종료는 undocumented `MessageBoxTimeoutW` 를 사용하며,
    없는 환경에서는 일반 `MessageBoxW`(수동 종료) 로 폴백한다.
    """
    try:
        import ctypes

        MB_ICONWARNING = 0x00000030
        MB_SYSTEMMODAL = 0x00001000
        MB_SETFOREGROUND = 0x00010000
        flags = MB_ICONWARNING | MB_SYSTEMMODAL | MB_SETFOREGROUND
        timeout_ms = max(0, POPUP_TIMEOUT_SEC) * 1000

        def _run():
            try:
                user32 = ctypes.windll.user32
                box_timeout = getattr(user32, "MessageBoxTimeoutW", None)
                if timeout_ms > 0 and box_timeout is not None:
                    # MessageBoxTimeoutW(hWnd, text, caption, type, langId, timeout_ms)
                    box_timeout(0, message, title, flags, 0, timeout_ms)
                else:
                    if timeout_ms > 0 and box_timeout is None:
                        print("[WARNING] MessageBoxTimeoutW 미지원 — 자동 종료 없이 표시")
                    user32.MessageBoxW(0, message, title, flags)
            except Exception as exc:
                print(f"[WARNING] Windows 팝업 실패: {exc}")

        threading.Thread(target=_run, daemon=True).start()
    except AttributeError:
        print(f"[INFO] 현재 OS 에서 MessageBox 미지원 — 콘솔 알림만: {title} | {message}")
    except Exception as exc:
        print(f"[WARNING] 팝업 표시 실패: {exc}")


def notify_align_fail(
    eqp_id: str,
    alarm_time: str,
    alarm_name: str,
    recipe_id: str = "",
    operation_desc: str = "",
    lot_type_cd: str = "",
) -> None:
    """Align Fail 감지 시 팝업 알림."""
    title = "CD-SEM Align Fail 감지"
    message = (
        f"EQP_ID    : {eqp_id}\n"
        f"ALARM     : {alarm_name}\n"
        f"TIME      : {alarm_time}\n"
        f"RECIPE_ID : {recipe_id}\n"
        f"OPERATION : {operation_desc}\n"
        f"LOT_TYPE  : {lot_type_cd}\n\n"
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
            "alarm_time": _row_value(row, "TIMESTAMP", "timestamp", "UTC9", "utc9", "alarm_time"),
            "alarm_name": str(
                _row_value(row, "ALARM_NAME", "alarm_name", "DESCRIPTION")
                or "Align Fail"
            ).strip(),
            "alid": str(_row_value(row, "ALID", "alarm_id", "al_id") or "9006").strip(),
            "recipe_id": str(_row_value(row, "RECIPE_ID", "recipe_id") or "").strip(),
            "operation_desc": str(
                _row_value(row, "OPERATION_DESC", "operation_desc") or ""
            ).strip(),
            "lot_type_cd": str(_row_value(row, "LOT_TYPE_CD", "lot_type_cd") or "").strip(),
        }
    return by_tool


def _send_rich_notify_async(eqp_id: str, recipe_id: str) -> None:
    """office rich notify 호출을 데몬 스레드로 비차단 실행한다."""
    if not RICH_NOTIFY_AVAILABLE:
        return

    def _run():
        try:
            send_cube_align_fail_info(eqp_id, recipe_id)
        except Exception as exc:
            print(f"[WARNING] office rich notify 예외: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def _connect_to_tool_sync(eqp_id: str, action_enabled: bool = True):
    """감지된 EQP_ID 장비로 RCS 접속(tool 더블클릭) 을 동기·1회 시도한다.

    정책: 알람당 한 번만 느슨하게 시도하고, 실패하면 엔지니어가 직접 접속한다.
    재시도/backoff 는 두지 않는다(접속 고도화는 추후 별도 작업).
    GUI 자동화는 직렬화되어야 하므로(동시에 두 tool 을 더블클릭할 수 없음) 비차단
    스레드가 아니라 poll 루프 안에서 순차로 수행하되, 메인 창 탐색 타임아웃을 짧게
    둬 실패 시 detection 루프를 오래 붙잡지 않는다. 예외는 삼켜 루프가 죽지 않게 한다.

    ToolSelectionResult(또는 None) 을 반환해, 후속 캡처가 실제 더블클릭 성공 여부를
    보고 진행할 수 있게 한다.
    """
    if not CONNECT_TOOL_AVAILABLE:
        return None
    try:
        return connect_to_tool(
            eqp_id,
            action_enabled=action_enabled,
            main_window_timeout_sec=CONNECT_TOOL_WINDOW_TIMEOUT_SEC,
        )
    except Exception as exc:
        print(f"[WARNING] 장비 자동 접속 예외: EQP_ID={eqp_id}, error={exc}")
        return None


def process_fail_rows(
    fails,
    active_tools: set[str],
    popup_enabled: bool = True,
    rich_notify_enabled: bool = True,
    connect_enabled: bool = True,
    connect_action_enabled: bool = True,
    capture_enabled: bool = True,
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
        alarm_time = str(info["alarm_time"] or "")
        alarm_name = info["alarm_name"]
        alid = info["alid"]
        recipe_id = info["recipe_id"]
        operation_desc = info["operation_desc"]
        lot_type_cd = info["lot_type_cd"]

        print(
            f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
            f"ALID={alid}, RECIPE_ID={recipe_id}, LOT_TYPE={lot_type_cd}, "
            f"시각={alarm_time}"
        )
        append_alarm_record(
            eqp_id, alarm_time, alarm_name, alid,
            recipe_id=recipe_id,
            operation_desc=operation_desc,
            lot_type_cd=lot_type_cd,
        )
        if popup_enabled:
            notify_align_fail(
                eqp_id, alarm_time, alarm_name,
                recipe_id=recipe_id,
                operation_desc=operation_desc,
                lot_type_cd=lot_type_cd,
            )
        if rich_notify_enabled:
            _send_rich_notify_async(eqp_id, recipe_id)
        # RECIPE_ID 가 있으면 등록 recipe 이므로 자동 파이프라인(이미지 저장 → 접속 →
        # live SEM 탐색)을 진행한다: 여기서 장비로 1회 접속 시도. RECIPE_ID 가 없으면
        # 자동 접속하지 않고 엔지니어가 직접 접속한다.
        if connect_enabled:
            if recipe_id:
                result = _connect_to_tool_sync(eqp_id, action_enabled=connect_action_enabled)
                # 접속(더블클릭)이 실제로 수행됐고 캡처가 켜져 있으면 RCS 화면을 1장
                # 박제하고 창을 닫는다. dry-run(클릭 생략) 이면 창이 안 떠 캡처도 생략.
                double_clicked = bool(getattr(result, "double_clicked", False))
                if capture_enabled and CAPTURE_RCS_AVAILABLE and double_clicked:
                    capture_and_close_rcs_window(eqp_id, recipe_id)
            else:
                print(
                    f"[INFO] RECIPE_ID 없음 — 자동 접속 생략, 엔지니어 직접 접속 필요 "
                    f"(EQP_ID={eqp_id})"
                )

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
    rich_notify_requested = env_flag("ALIGN_FAIL_RICH_NOTIFY", RICH_NOTIFY_ENABLED_DEFAULT)
    rich_notify_enabled = rich_notify_requested and RICH_NOTIFY_AVAILABLE
    if rich_notify_requested and not RICH_NOTIFY_AVAILABLE:
        print("[WARNING] ALIGN_FAIL_RICH_NOTIFY=on 이지만 office_rich_notify 모듈 로드 실패 - off 로 진행")

    connect_requested = env_flag("ALIGN_FAIL_CONNECT_TOOL", CONNECT_TOOL_ENABLED_DEFAULT)
    connect_enabled = connect_requested and CONNECT_TOOL_AVAILABLE
    connect_action_enabled = env_flag("ALIGN_FAIL_CONNECT_ACTION", CONNECT_TOOL_ACTION_DEFAULT)
    if connect_requested and not CONNECT_TOOL_AVAILABLE:
        print("[WARNING] ALIGN_FAIL_CONNECT_TOOL=on 이지만 workflow_select_tool 로드 실패 - off 로 진행")

    capture_requested = env_flag("ALIGN_FAIL_CAPTURE_RCS", CAPTURE_RCS_ENABLED_DEFAULT)
    capture_enabled = capture_requested and CAPTURE_RCS_AVAILABLE
    if capture_requested and not CAPTURE_RCS_AVAILABLE:
        print("[WARNING] ALIGN_FAIL_CAPTURE_RCS=on 이지만 rcs_screenshot 로드 실패 - off 로 진행")

    active_tools: set[str] = set()
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)

    print(
        f"[INFO] Align Fail 감지 시작 (주기={POLL_INTERVAL_SEC}s, "
        f"윈도우={DETECTION_WINDOW_SEC}s, "
        f"팝업={'on' if popup_enabled else 'off'}, "
        f"rich_notify={'on' if rich_notify_enabled else 'off'}, "
        f"장비접속={'on' if connect_enabled else 'off'}"
        f"{'(dry-run)' if connect_enabled and not connect_action_enabled else ''}, "
        f"화면캡처={'on' if capture_enabled else 'off'})"
    )
    print(f"[INFO] 누적 로그 파일: {ALARM_LOG_PATH}")
    print("[INFO] 같은 EQP_ID 의 중복 알람은 한 번만 알립니다 (해제 후 재감지 시 재알림).")

    while True:
        try:
            poll_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO] {poll_time} — 알람 조회 (최근 {DETECTION_WINDOW_SEC}s 윈도우)")
            alarms = get_cdsem_alarms()
            fails = filter_align_fail(alarms)
            fails = filter_rows_within_window(fails, DETECTION_WINDOW_SEC)

            if _alarm_rows_empty(fails):
                if active_tools:
                    for eqp_id in sorted(active_tools):
                        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
                    active_tools.clear()
                # idle 상태는 진입 시 한 번만 로깅 — poll 마다 찍지 않는다.
                if not idle_logged:
                    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} — Align Fail 없음")
                    idle_logged = True
            else:
                idle_logged = False
                count = process_fail_rows(
                    fails,
                    active_tools,
                    popup_enabled=popup_enabled,
                    rich_notify_enabled=rich_notify_enabled,
                    connect_enabled=connect_enabled,
                    connect_action_enabled=connect_action_enabled,
                    capture_enabled=capture_enabled,
                )
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
