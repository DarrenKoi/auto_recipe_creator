"""Align Fail 모니터링 + CCTV 이동 + CH4 프레임 캡처 스크립트.

2분 주기로 CD-SEM 알람을 조회하여 ALID=9006 (Align Fail)이 감지되면:
  1. 사용자가 미리 열어 둔 RCS Tool List 창을 사용
  2. 해당 EQP_ID 의 Tool DVR(CCTV) 창을 열기
  3. Channel 4 를 확대
  4. `capture_window_frames_ch4.py` 를 사용해 최대 8분 프레임 캡처

사용법:
  uv run python poc/workflow_1/monitor_align_fail.py
"""

import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.util import env_int

LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

# ── 설정 (환경변수) ────────────────────────────────────────────────
POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 120)


def open_cctv_for_tool(eqp_id: str) -> bool:
    """현재 RCS 메인 창에서 대상 Tool 의 CCTV(DVR) 창을 연다."""
    from poc.workflow_1.login_rcs_common import wait_for_rcs_main_window
    from poc.workflow_1.workflow_select_tool_cctv import (
        DEFAULT_ACTION_ENABLED,
        EXIT_SUCCESS,
        select_tool_cctv_from_main_window,
    )

    print(f"[INFO] CCTV 창 열기 시작 (target_tool_name={eqp_id})")
    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print("[ERROR] RCS 메인 창을 찾지 못해 CCTV 열기를 중단합니다.")
        return False

    try:
        result = select_tool_cctv_from_main_window(
            main_window,
            window_title,
            backend,
            eqp_id,
            action_enabled=DEFAULT_ACTION_ENABLED,
        )
    except Exception as exc:
        print(f"[ERROR] CCTV 열기 중 예외: {exc}")
        return False

    if result.exit_code != EXIT_SUCCESS:
        print(
            f"[ERROR] CCTV 창 열기 실패: "
            f"target_tool_name={eqp_id}, exit_code={result.exit_code}"
        )
        return False

    print(
        f"[INFO] CCTV 창 열기 성공: target_tool_name={eqp_id}, "
        f"dvr_window_verified={result.dvr_window_verified}"
    )
    return True


def _find_tool_list_window():
    """현재 열려 있는 RCS Tool List 메인 창을 반환한다."""
    from poc.workflow_1.login_rcs_common import wait_for_rcs_main_window

    return wait_for_rcs_main_window()


def select_ch4_for_capture() -> bool:
    """열려 있는 DVR player 창에서 Channel 4 를 확대한다."""
    from poc.workflow_1.workflow_select_ch4_cctv import (
        DEFAULT_ACTION_ENABLED,
        EXIT_SUCCESS,
        _find_player_window,
        select_ch4_from_player_window,
    )

    print("[INFO] Channel 4 확대 시작")
    player_window, window_title, backend, process_name = _find_player_window()
    if player_window is None:
        print("[ERROR] DVR player 창을 찾지 못해 Channel 4 확대를 중단합니다.")
        return False

    try:
        result = select_ch4_from_player_window(
            player_window,
            window_title,
            backend,
            process_name,
            action_enabled=DEFAULT_ACTION_ENABLED,
            log_name=LOG_NAME,
            component_name=COMPONENT_NAME,
        )
    except Exception as exc:
        print(f"[ERROR] Channel 4 확대 중 예외: {exc}")
        return False

    if result.exit_code != EXIT_SUCCESS or not result.clicked:
        print(
            f"[ERROR] Channel 4 확대 실패: "
            f"exit_code={result.exit_code}, clicked={result.clicked}"
        )
        return False

    print("[INFO] Channel 4 확대 성공")
    return True


def capture_alarm_frames(eqp_id: str) -> bool:
    """Align Fail 대응용 CH4 프레임 캡처를 수행한다."""
    from poc.workflow_1.capture_window_frames_ch4 import capture_frames

    print(f"[INFO] CH4 프레임 캡처 시작 (target_tool_name={eqp_id})")
    try:
        exit_code = capture_frames()
    except Exception as exc:
        print(f"[ERROR] CH4 프레임 캡처 중 예외: {exc}")
        return False

    if exit_code != "success":
        print(
            f"[ERROR] CH4 프레임 캡처 실패: "
            f"target_tool_name={eqp_id}, exit_code={exit_code}"
        )
        return False

    print(f"[INFO] CH4 프레임 캡처 성공 (target_tool_name={eqp_id})")
    return True


def close_dvr_and_return_to_tool_list(
    tool_list_window,
    tool_list_title: str,
) -> bool:
    """DVR 창을 닫고 Tool List 창으로 포커스를 되돌린다."""
    from poc.workflow_1.util import activate_window
    from poc.workflow_1.workflow_select_ch4_cctv import _find_player_window

    player_window, player_title, _backend, process_name = _find_player_window()
    if player_window is None:
        print("[WARNING] 닫을 DVR player 창을 찾지 못했습니다. Tool List 복귀만 시도합니다.")
    else:
        try:
            activate_window(
                player_window,
                debug_label=f"close_dvr_{process_name or 'player'}",
            )
        except Exception:
            pass

        closed = False
        for method_name in ("close",):
            try:
                close_method = getattr(player_window, method_name, None)
                if callable(close_method):
                    close_method()
                    closed = True
                    break
            except Exception as exc:
                print(f"[WARNING] DVR 창 닫기 실패: method={method_name}, error={exc}")

        if not closed:
            try:
                player_window.type_keys("%{F4}")
                closed = True
            except Exception as exc:
                print(f"[WARNING] DVR Alt+F4 닫기 실패: error={exc}")

        time.sleep(0.7)
        print(
            f"[INFO] DVR 창 닫기 {'완료' if closed else '실패'}: "
            f"title={player_title!r}, process={process_name}"
        )

    if tool_list_window is None:
        tool_list_window, tool_list_title, _backend = _find_tool_list_window()

    if tool_list_window is None:
        print("[ERROR] Tool List 창 복귀 실패: 메인 RCS 창을 찾지 못했습니다.")
        return False

    refocused = activate_window(
        tool_list_window,
        debug_label=tool_list_title or "rcs_tool_list",
    )
    if not refocused:
        print(f"[ERROR] Tool List 창 활성화 실패: title={tool_list_title!r}")
        return False

    print(f"[INFO] Tool List 창 복귀 완료: title={tool_list_title!r}")
    return True


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


def monitor_loop():
    """메인 모니터링 루프.

    2분 주기로 알람을 조회하고, Align Fail 감지 시
    이미 열어 둔 Tool List 창에서 DVR -> CH4 -> 프레임 캡처를 수행한다.
    """
    already_handled: set[str] = set()

    print(f"[INFO] Align Fail 모니터링 시작 (주기={POLL_INTERVAL_SEC}s)")
    print("[INFO] 전제 조건: RCS Tool List 창을 미리 열어 두어야 합니다.")
    print("[INFO] 감지 시 동작: Tool DVR 열기 → CH4 확대 → 최대 8분 프레임 캡처")

    while True:
        try:
            alarms = get_cdsem_alarms()
            fails = filter_align_fail(alarms)

            if _alarm_rows_empty(fails):
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} — Align Fail 없음")
            else:
                for row in _iter_alarm_rows(fails):
                    eqp_id = str(_row_value(row, "EQP_ID", "eqp_id", "tool_name") or "").strip()
                    alarm_time = _row_value(row, "UTC9", "utc9", "alarm_time")
                    alarm_key = f"{eqp_id}:{alarm_time}" if alarm_time else eqp_id
                    tool_list_window = None
                    tool_list_title = ""

                    if not eqp_id:
                        print("[WARNING] EQP_ID 없는 Align Fail row 발견 — 건너뜀")
                        continue

                    if alarm_key in already_handled:
                        print(f"[INFO] {alarm_key} 이미 처리됨 — 건너뜀")
                        continue

                    print(f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
                          f"시각={alarm_time}")

                    log_work2_event(
                        component=COMPONENT_NAME,
                        message="align_fail_detected",
                        log_name=LOG_NAME,
                        eqp_id=eqp_id,
                        alarm_time=alarm_time,
                    )

                    tool_list_window, tool_list_title, _backend = _find_tool_list_window()
                    if tool_list_window is None:
                        print("[ERROR] Tool List 창을 찾지 못했습니다 — 다음 주기 재시도")
                        continue

                    try:
                        cctv_opened = open_cctv_for_tool(eqp_id)
                        if not cctv_opened:
                            print(f"[ERROR] {eqp_id} CCTV 창 열기 실패 — 다음 주기 재시도")
                            continue

                        ch4_selected = select_ch4_for_capture()
                        if not ch4_selected:
                            print(f"[ERROR] {eqp_id} Channel 4 확대 실패 — 다음 주기 재시도")
                            continue

                        captured = capture_alarm_frames(eqp_id)
                        if not captured:
                            print(f"[ERROR] {eqp_id} CH4 프레임 캡처 실패 — 다음 주기 재시도")
                            continue

                        already_handled.add(alarm_key)
                    finally:
                        returned = close_dvr_and_return_to_tool_list(
                            tool_list_window,
                            tool_list_title,
                        )
                        if not returned:
                            print("[WARNING] 후처리 완료 후 Tool List 창 복귀에 실패했습니다.")

        except KeyboardInterrupt:
            print("\n[INFO] 모니터링 중단 (Ctrl+C)")
            break
        except Exception as exc:
            print(f"[ERROR] 모니터링 루프 예외: {exc}")

        try:
            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n[INFO] 모니터링 중단 (Ctrl+C)")
            break

    print("[INFO] 모니터링 종료")


if __name__ == "__main__":
    monitor_loop()
