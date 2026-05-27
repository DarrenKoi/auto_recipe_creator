"""Align Fail 알람 감지 + 자동 기록 사이클 스크립트.

`align_fail_alarm.py` 의 변형. 1분 주기로 CD-SEM 알람을 조회하여 ALID=9006
(Align Fail) 이 감지되면, EQP_ID 단위로 아래 **iteration** 을 수행한다:

  alarm(eqp_id) → ① tool 열기(더블클릭) → 알림(alert) 팝업 닫기
                → ② tool 화면 record(캡처) → ③ tool 창 닫기

알림 팝업은 SYSTEMMODAL/최상위라 RCS 모니터 screenshot 에 겹쳐 찍히므로, ② record
전에 먼저 닫는다. 캡처 후 tool 창도 닫아 한 사이클을 끝까지 마무리한다(연속 align
fail 에서 창/팝업이 쌓이지 않게). 캡처는 RECIPE_ID 가 있는 등록 recipe 일 때만
수행한다(저장 경로가 <eqp>/<class>/<recipe> 라서 RECIPE_ID 필요). RECIPE_ID 가
없으면 알림만 띄우고 엔지니어가 직접 처리한다.

각 단계 모듈:
  ① workflow_select_tool.connect_to_tool
  알림 닫기: FindWindowW + WM_CLOSE        (notify_align_fail 팝업 — record 전에)
  ② rcs_screenshot.record_rcs_window      (캡처만, 닫지 않음)
  ③ workflow_close_tool.close_tool         (제목의 tool id 로 창 찾아 닫기)

사용법:
  uv run python poc/workflow_1/align_fail_alarm_record.py
"""

import csv
import threading
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

import pandas as pd

from poc.workflow_1 import LOG_DIR
from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms # pyright: ignore[reportMissingImports]
from poc.workflow_1.util import env_flag, env_int

# rich notify 는 선택 의존성. requests/PIL/ftplib 등이 빠져도 본체는 계속 동작해야 한다.
try:
    from poc.workflow_1.office_rich_notify import send_cube_align_fail_info # pyright: ignore[reportMissingImports]

    RICH_NOTIFY_AVAILABLE = True
except Exception as _rich_notify_import_exc:
    send_cube_align_fail_info = None
    RICH_NOTIFY_AVAILABLE = False
    print(f"[WARNING] office_rich_notify 모듈 로드 실패 - 텍스트 로그/팝업만 동작합니다: {_rich_notify_import_exc}")

# 장비 자동 접속(tool 더블클릭). pywinauto/VLM 머신이 없는 환경(macOS 개발 등)에서는
# import 가 실패해도 감지/로그/팝업은 계속 동작해야 한다.
try:
    from poc.workflow_1.workflow_select_tool import connect_to_tool # pyright: ignore[reportMissingImports]

    CONNECT_TOOL_AVAILABLE = True
except Exception as _connect_tool_import_exc:
    connect_to_tool = None
    CONNECT_TOOL_AVAILABLE = False
    print(f"[WARNING] workflow_select_tool 로드 실패 - 장비 자동 접속 비활성화: {_connect_tool_import_exc}")

# tool 화면 record(캡처만, 닫지 않음). 윈도우/캡처 의존성이 없는 환경에서도 본체는
# 계속 동작해야 하므로 import 실패를 흡수한다.
try:
    from poc.workflow_1.rcs_screenshot import record_rcs_window # pyright: ignore[reportMissingImports]

    RECORD_RCS_AVAILABLE = True
except Exception as _record_rcs_import_exc:
    record_rcs_window = None
    RECORD_RCS_AVAILABLE = False
    print(f"[WARNING] rcs_screenshot 로드 실패 - 화면 record 비활성화: {_record_rcs_import_exc}")

# tool 창 닫기(제목의 tool id 매칭). 선택 의존성.
try:
    from poc.workflow_1.workflow_close_tool import close_tool # pyright: ignore[reportMissingImports]

    CLOSE_TOOL_AVAILABLE = True
except Exception as _close_tool_import_exc:
    close_tool = None
    CLOSE_TOOL_AVAILABLE = False
    print(f"[WARNING] workflow_close_tool 로드 실패 - tool 창 닫기 비활성화: {_close_tool_import_exc}")

POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 10)
# 감지 look-back 윈도우(초). poll 주기와 분리해, 알람 보고 지연이 있어도 놓치지 않게 한다.
DETECTION_WINDOW_SEC = env_int("ALIGN_FAIL_WINDOW_SEC", 60)
POPUP_ENABLED_DEFAULT = True
RICH_NOTIFY_ENABLED_DEFAULT = True
# Align Fail 감지 시 record 사이클(접속→캡처→닫기) 을 수행할지. 기본 on.
CONNECT_TOOL_ENABLED_DEFAULT = True
# 접속 시 실제 더블클릭 수행 여부. off 면 인식/디버그 저장만 하고 클릭은 생략(dry-run).
CONNECT_TOOL_ACTION_DEFAULT = True
# 접속 시 메인 RCS 창 탐색 타임아웃(초). 한 번만 느슨하게 시도하고 실패하면 엔지니어가
# 직접 접속하는 정책이라 짧게 둔다.
CONNECT_TOOL_WINDOW_TIMEOUT_SEC = env_int("ALIGN_FAIL_CONNECT_WINDOW_TIMEOUT_SEC", 3)
# 팝업 자동 종료 시간(초). 0 이면 사용자가 닫을 때까지. record 사이클이 끝나면 ④에서
# 명시적으로 닫지만, 사이클이 돌지 않는 경우(RECIPE_ID 없음 등)를 위한 backstop.
POPUP_TIMEOUT_SEC = env_int("ALIGN_FAIL_POPUP_TIMEOUT_SEC", 60)
# ④ 알림 창 닫기 탐색 타임아웃(초).
ALERT_CLOSE_TIMEOUT_SEC = env_int("ALIGN_FAIL_ALERT_CLOSE_TIMEOUT_SEC", 3)
# ② tool 창 탐색 최대 시도 횟수. RCS 가 다른 사용자에게 점유되면 tool 더블클릭 후
# 'select'(공유/종료 선택) 팝업이 떠 tool 창이 안 열린다. 이때 무한 폴링하지 않고
# 이 횟수만큼만 시도한 뒤 포기하고 다음 알람을 기다린다('select' 팝업은 건드리지 않음).
RCS_WINDOW_MAX_TRIALS = env_int("ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS", 10)
ALARM_LOG_PATH = LOG_DIR / "align_fail_alarms.txt"
# 나중에 examine 하기 쉽도록 align fail 한 건마다 eqp_id/recipe_id + 저장된 캡처
# 이미지 경로를 한 줄로 누적하는 CSV manifest.
RECORD_MANIFEST_PATH = LOG_DIR / "align_fail_records.csv"
RECORD_MANIFEST_COLUMNS = [
    "detected_at",
    "eqp_id",
    "recipe_id",
    "alid",
    "alarm_time",
    "alarm_name",
    "frame_count",
    "captured_dir",
    "frames",
]

# 알림 팝업 제목 — notify_align_fail 과 ④ 닫기에서 같은 값을 써야 창을 찾을 수 있다.
ALERT_POPUP_TITLE = "CD-SEM Align Fail 감지"

# record 사이클(접속→캡처→닫기) 수행 여부. 기본 on.
RECORD_CYCLE_ENABLED_DEFAULT = True
# 실행 중 PC 절전/디스플레이 끄기를 막을지. 기본 on. 마우스/키보드를 건드리지 않고
# Windows 전원 상태 API 로만 막으므로 RCS GUI 자동화와 충돌하지 않는다.
KEEP_AWAKE_ENABLED_DEFAULT = True


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


def _alarm_time_to_tag(alarm_time: str) -> str | None:
    """알람 UTC9 문자열을 캡처 폴더/파일명용 타임스탬프 태그로 변환한다.

    캡처 폴더가 캡처 wall-clock 이 아니라 실제 align fail 이벤트 시각으로 묶이도록,
    UTC9 를 파싱해 `make_timestamp_tag` 와 같은 `%y%m%d_%H%M%S` 형식으로 만든다.
    UTC9 는 이미 KST(UTC+9) wall-clock 이므로 epoch 변환(timezone 가정) 없이 파싱된
    값을 그대로 strftime 한다. 비어있거나 파싱 불가하면 None 을 반환해 호출부가 캡처
    시점으로 폴백하게 한다.
    """
    if not alarm_time:
        return None
    ts = pd.to_datetime(alarm_time, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return ts.strftime("%y%m%d_%H%M%S")


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


def append_record_manifest(
    eqp_id: str,
    recipe_id: str,
    frames,
    *,
    alid: str = "",
    alarm_time: str = "",
    alarm_name: str = "",
) -> None:
    """align fail 한 건의 메타 + 저장된 캡처 이미지 경로를 CSV manifest 에 한 줄 누적한다.

    eqp_id / recipe_id 와 이미지 경로를 함께 적어, 나중에 어떤 장비/recipe 의 어떤
    이미지가 어디에 저장됐는지 examine 하기 쉽게 한다(Excel/grep). 파일이 없으면
    헤더를 먼저 쓴다. 기록 실패는 삼켜 루프가 죽지 않게 한다.
    """
    frame_paths = [str(p) for p in (frames or [])]
    captured_dir = str(frames[0].parent) if frames else ""
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        RECORD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not RECORD_MANIFEST_PATH.exists()
            or RECORD_MANIFEST_PATH.stat().st_size == 0
        )
        with RECORD_MANIFEST_PATH.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(RECORD_MANIFEST_COLUMNS)
            writer.writerow([
                detected_at,
                eqp_id,
                recipe_id,
                alid,
                alarm_time,
                alarm_name,
                len(frame_paths),
                captured_dir,
                " | ".join(frame_paths),
            ])
        print(
            f"[INFO] record manifest 기록 → {RECORD_MANIFEST_PATH} "
            f"(EQP_ID={eqp_id}, frames={len(frame_paths)})"
        )
    except Exception as exc:
        print(f"[WARNING] record manifest 기록 실패: {exc}")


def _show_popup_windows(title: str, message: str) -> None:
    """Windows MessageBox 를 데몬 스레드에서 띄운다 (루프 비차단).

    `POPUP_TIMEOUT_SEC` > 0 이면 해당 시간 후 팝업을 자동으로 닫는다(backstop).
    record 사이클이 정상이면 ④에서 먼저 명시적으로 닫는다.
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


def _close_alert_window(title: str = ALERT_POPUP_TITLE, *, timeout_sec: float = ALERT_CLOSE_TIMEOUT_SEC) -> bool:
    """제목으로 알림 팝업(MessageBox) 창을 찾아 닫는다 (④, Windows 전용).

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
        print(f"[INFO] 현재 OS 에서 알림 창 닫기 미지원 — 생략: {title!r}")
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


def notify_align_fail(
    eqp_id: str,
    alarm_time: str,
    alarm_name: str,
    recipe_id: str = "",
    operation_desc: str = "",
    lot_type_cd: str = "",
) -> None:
    """Align Fail 감지 시 팝업 알림."""
    message = (
        f"EQP_ID    : {eqp_id}\n"
        f"ALARM     : {alarm_name}\n"
        f"TIME      : {alarm_time}\n"
        f"RECIPE_ID : {recipe_id}\n"
        f"OPERATION : {operation_desc}\n"
        f"LOT_TYPE  : {lot_type_cd}\n\n"
        f"로그: {ALARM_LOG_PATH}"
    )
    _show_popup_windows(ALERT_POPUP_TITLE, message)


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
            # 캡처 폴더 태그 전용 — 윈도우 필터(filter_rows_within_window)와 같은 UTC9
            # 컬럼만 쓴다(알람 시스템 기준 시각). 로그/매니페스트의 alarm_time 과 분리해
            # 폴더 태그를 알람 시각 한 가지로 일관되게 묶는다.
            "utc9": str(_row_value(row, "UTC9", "utc9") or "").strip(),
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
    """감지된 EQP_ID 장비로 RCS 접속(tool 더블클릭) 을 동기·1회 시도한다(①).

    정책: 알람당 한 번만 느슨하게 시도하고, 실패하면 엔지니어가 직접 접속한다.
    예외는 삼켜 루프가 죽지 않게 한다. ToolSelectionResult(또는 None) 을 반환한다.
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


def run_record_cycle(
    eqp_id: str,
    recipe_id: str,
    *,
    connect_action_enabled: bool = True,
    utc9: str = "",
) -> list:
    """한 알람에 대한 record 사이클: ① 열기 → ② record → ③ tool 닫기 → ④ 알림 닫기.

    ①에서 더블클릭이 실제로 수행됐을 때만 ②③를 진행한다(dry-run/실패 시 창이 안
    떠 캡처/닫기를 생략). RCS 가 다른 사용자에게 점유돼 'select' 팝업이 뜨면 tool
    창이 안 열리므로, ②의 창 탐색을 RCS_WINDOW_MAX_TRIALS 회로 제한하고 그래도 못
    찾으면 이번 알람은 건너뛰고 다음 알람을 기다린다('select' 팝업은 건드리지 않음).
    알림 팝업은 screenshot 에 겹치지 않도록 ② record 전에 닫는다.

    저장된 캡처 이미지 경로 목록(list[Path])을 반환한다(캡처 없으면 빈 리스트).
    """
    # ① tool 열기.
    result = _connect_to_tool_sync(eqp_id, action_enabled=connect_action_enabled)
    double_clicked = bool(getattr(result, "double_clicked", False))

    if not double_clicked:
        print(
            f"[INFO] tool 더블클릭 미수행(dry-run/실패) — record/tool 닫기 생략: "
            f"EQP_ID={eqp_id}"
        )
        _close_alert_window()  # 팝업 정리
        return []

    # 알림(alert) 팝업 닫기 — SYSTEMMODAL/최상위라 RCS 모니터 screenshot 에 겹쳐
    # 찍히므로, ② record(캡처) 전에 화면에서 치운다. ① 열기에 수 초가 걸려 그 사이
    # 팝업이 충분히 떠 있으므로 여기서 닫으면 누락 없이 닫힌다.
    _close_alert_window()

    # ② tool 화면 record (캡처만, 창은 닫지 않음). 창 탐색은 RCS_WINDOW_MAX_TRIALS 로 제한.
    # 캡처 폴더가 캡처 wall-clock 이 아니라 알람 시스템 기준 시각(UTC9)으로 묶이도록
    # tag 를 넘긴다. UTC9 비어있거나 파싱 불가하면 None → record_rcs_window 가 캡처
    # 시점으로 폴백.
    saved: list = []
    tool_window = None
    if RECORD_RCS_AVAILABLE:
        saved, tool_window, _title, _backend = record_rcs_window(
            eqp_id,
            recipe_id,
            window_max_attempts=RCS_WINDOW_MAX_TRIALS,
            tag=_alarm_time_to_tag(utc9),
        )
    else:
        print(f"[INFO] record 비활성 — 캡처 생략: EQP_ID={eqp_id}")

    if tool_window is not None:
        # ③ tool 창 닫기 (제목의 tool id 로 찾아 닫기). 창이 확인됐으니 빠르게 닫힌다.
        if CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)
        else:
            print(f"[INFO] close_tool 비활성 — tool 창 닫기 생략: EQP_ID={eqp_id}")
    elif RECORD_RCS_AVAILABLE:
        # 창을 못 찾음(RCS_WINDOW_MAX_TRIALS 회 시도) — RCS 점유(select 팝업) 가능성.
        # 'select' 팝업은 건드리지 않고(공유/종료는 사람 판단), 이번 알람은 포기한다.
        print(
            f"[WARNING] tool 창 미발견({RCS_WINDOW_MAX_TRIALS}회 시도) — RCS 가 다른 "
            f"사용자에게 점유됐을 수 있음(select 공유/종료 팝업). 이번 알람은 건너뛰고 "
            f"다음 알람을 기다립니다: EQP_ID={eqp_id}"
        )

    return saved


def process_fail_rows(
    fails,
    active_tools: set[str],
    popup_enabled: bool = True,
    rich_notify_enabled: bool = True,
    record_enabled: bool = True,
    connect_action_enabled: bool = True,
) -> int:
    """EQP_ID 기준으로 edge-triggered 알림 + record 사이클을 수행한다.

    - 같은 EQP_ID 에서 여러 알람 code 가 와도 하나로 취급.
    - 이전 poll 에서 이미 활성이던 EQP_ID 는 다시 처리하지 않음.
    - 이번 poll 에 사라진 EQP_ID 는 `active_tools` 에서 제거 (복구 시 재처리 가능).

    `active_tools` 는 in-place 로 갱신된다. 새로 처리한 개수를 반환.
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
        utc9 = info["utc9"]
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

        # RECIPE_ID 가 있는 등록 recipe 일 때만 record 사이클(열기→record→닫기→알림닫기).
        # 저장 경로가 <eqp>/<class>/<recipe> 라 RECIPE_ID 가 있어야 한다.
        frames: list = []
        if record_enabled:
            if recipe_id:
                frames = run_record_cycle(
                    eqp_id,
                    recipe_id,
                    connect_action_enabled=connect_action_enabled,
                    utc9=utc9,
                ) or []
            else:
                print(
                    f"[INFO] RECIPE_ID 없음 — record 사이클 생략, 엔지니어 직접 처리 "
                    f"(EQP_ID={eqp_id})"
                )

        # 나중에 examine 하기 쉽도록 한 줄짜리 record manifest 에 누적(이미지 없으면
        # frame 0 으로 기록되어 align fail 발생 자체는 남는다).
        # 매니페스트 alarm_time 은 캡처 폴더 태그와 같은 UTC9(알람 시스템 기준 시각)로
        # 적어, manifest 한 줄과 captured_dir 경로의 시각이 끝까지 일치하게 한다.
        append_record_manifest(
            eqp_id,
            recipe_id,
            frames,
            alid=alid,
            alarm_time=utc9,
            alarm_name=alarm_name,
        )

        active_tools.add(eqp_id)
        newly_handled += 1

    return newly_handled


def _set_keep_awake(enable: bool) -> None:
    """PC 절전/디스플레이 끄기를 막거나(enable=True) 원복(False)한다 (Windows 전용).

    SetThreadExecutionState 로 시스템/디스플레이 활성 요구를 건다. 마우스/키보드를
    전혀 건드리지 않으므로 RCS GUI 자동화(connect/close 클릭)와 충돌하지 않는다.
    화면 캡처가 검게 나오지 않도록 디스플레이도 켜둔다(ES_DISPLAY_REQUIRED). 한 번
    걸면 ES_CONTINUOUS 로 유지되며, 프로세스 종료 시 OS 가 자동 해제한다. 비Windows/
    실패 시 조용히 무시.
    """
    try:
        import ctypes
    except Exception:
        return

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if enable else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        print(f"[INFO] keep-awake {'ON (절전/화면 꺼짐 방지)' if enable else 'OFF (원복)'}")
    except AttributeError:
        # 비Windows — windll 없음.
        if enable:
            print("[INFO] 현재 OS 에서 keep-awake 미지원 — 생략")
    except Exception as exc:
        print(f"[WARNING] keep-awake 설정 실패: {exc}")


def monitor_loop(popup_enabled: bool | None = None) -> None:
    """메인 감지 루프 — 1분 주기 기본. 각 신규 Align Fail 마다 record 사이클 수행.

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

    # record 사이클은 접속(connect)이 가능해야 의미가 있다.
    record_requested = env_flag("ALIGN_FAIL_RECORD_CYCLE", RECORD_CYCLE_ENABLED_DEFAULT)
    record_enabled = record_requested and CONNECT_TOOL_AVAILABLE
    connect_action_enabled = env_flag("ALIGN_FAIL_CONNECT_ACTION", CONNECT_TOOL_ACTION_DEFAULT)
    if record_requested and not CONNECT_TOOL_AVAILABLE:
        print("[WARNING] ALIGN_FAIL_RECORD_CYCLE=on 이지만 workflow_select_tool 로드 실패 - off 로 진행")

    keep_awake = env_flag("ALIGN_FAIL_KEEP_AWAKE", KEEP_AWAKE_ENABLED_DEFAULT)
    if keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)

    print(
        f"[INFO] Align Fail 감지+record 시작 (주기={POLL_INTERVAL_SEC}s, "
        f"윈도우={DETECTION_WINDOW_SEC}s, "
        f"팝업={'on' if popup_enabled else 'off'}, "
        f"rich_notify={'on' if rich_notify_enabled else 'off'}, "
        f"record사이클={'on' if record_enabled else 'off'}"
        f"{'(dry-run)' if record_enabled and not connect_action_enabled else ''})"
    )
    print(f"[INFO] 누적 로그 파일: {ALARM_LOG_PATH}")
    print("[INFO] 각 신규 Align Fail: 열기 → 알림닫기 → record → tool 닫기. 중복 알람은 한 번만 처리.")

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
                    record_enabled=record_enabled,
                    connect_action_enabled=connect_action_enabled,
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

    if keep_awake:
        _set_keep_awake(False)
    print("[INFO] 감지 종료")


if __name__ == "__main__":
    monitor_loop()
