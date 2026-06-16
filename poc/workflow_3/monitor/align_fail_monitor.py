"""실시간 Align Fail 모니터링 루프 — workflow_3 진입점.

`workflow_1/align_fail_alarm_record.py` 를 전면 이전·확장한 production 루프:

  알람 폴링(ALID=9006) → [알람별 사이클: RCS 확보 → tool 접속 → 상시 녹화 →
  SEM panel 확보 → CV 보정] → 처리 실패 시 cube rich notification →
  엔지니어 수동 조작 녹화 대기 → tool 닫기 → 다음 알람 대기

사이클 내부는 `cycle.run_alarm_cycle`(WorkflowRunner step + 보장 teardown)이
담당하고, 본 모듈은 폴링/edge-trigger/로그/manifest 만 책임진다.

  * 같은 EQP_ID 의 중복 알람은 한 번만 처리(edge-trigger), 해제되면 재처리 가능.
  * RECIPE_ID 유무와 무관하게 사이클은 돈다(접속+녹화). 보정만 RECIPE_ID 필요.
  * 알람 소스는 office MES 모듈 또는 replay CSV (`alarm_source` 참고).

사용법:
  uv run python poc/workflow_3/monitor/align_fail_monitor.py

개발 PC dry-run:
  SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
    uv run python poc/workflow_3/monitor/align_fail_monitor.py
"""

import csv
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

import pandas as pd

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.monitor.alarm_source import load_alarm_source
from poc.workflow_3.monitor.cycle import CycleResult, run_alarm_cycle
from poc.workflow_3.monitor.notify import (
    ALARM_LOG_PATH,
    notify_align_fail_popup,
)
from poc.workflow_3.monitor.rcp_msr_gather import gather_rcp_msr
from poc.workflow_3.monitor.success_gather import gather_success_async

LOG_COMPONENT = "align_fail_monitor"

# 알람 1건당 사이클 결과를 한 줄로 누적하는 CSV manifest (Excel/grep 용).
CYCLE_MANIFEST_PATH = LOG_DIR / "align_fail_cycles.csv"
CYCLE_MANIFEST_COLUMNS = [
    "detected_at",
    "eqp_id",
    "recipe_id",
    "alid",
    "alarm_time",
    "alarm_name",
    "run_status",
    "failed_step",
    "failure_class",
    "outcome_status",
    "outcome_path",
    "key_decision",
    "best_xy",
    "frame_count",
    "recording_dir",
    "run_dir",
]


# ------------------------------------------------------------------
# 알람 row 헬퍼 (기존 검증 로직 유지).
# ------------------------------------------------------------------


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
    """알람 UTC9 문자열을 캡처/녹화 폴더용 타임스탬프 태그로 변환한다.

    폴더가 캡처 wall-clock 이 아니라 실제 align fail 이벤트 시각으로 묶이도록
    `%y%m%d_%H%M%S` 형식으로 만든다. 파싱 불가하면 None(호출부가 현재 시각 폴백).
    """
    if not alarm_time:
        return None
    ts = pd.to_datetime(alarm_time, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return ts.strftime("%y%m%d_%H%M%S")


def _collapse_rows_by_tool(fails) -> dict[str, dict]:
    """같은 EQP_ID 의 여러 알람 row 를 하나로 합친다 (첫 번째 row 채택)."""
    by_tool: dict[str, dict] = {}
    for row in _iter_alarm_rows(fails):
        eqp_id = str(_row_value(row, "EQP_ID", "eqp_id", "tool_name") or "").strip()
        if not eqp_id:
            print("[WARNING] EQP_ID 없는 Align Fail row 발견 - 건너뜀")
            continue
        if eqp_id in by_tool:
            continue
        by_tool[eqp_id] = {
            "eqp_id": eqp_id,
            "alarm_time": _row_value(row, "TIMESTAMP", "timestamp", "UTC9", "utc9", "alarm_time"),
            # 폴더 태그 전용 — 윈도우 필터와 같은 UTC9 컬럼만 쓴다(알람 시스템 기준 시각).
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


# ------------------------------------------------------------------
# 기록.
# ------------------------------------------------------------------


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


def append_cycle_manifest(
    info: dict,
    cycle: CycleResult,
) -> None:
    """알람 1건의 메타 + 사이클 결과를 CSV manifest 에 한 줄 누적한다.

    파일이 없으면 헤더를 먼저 쓴다. 기록 실패는 삼켜 루프가 죽지 않게 한다.
    """
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        CYCLE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not CYCLE_MANIFEST_PATH.exists()
            or CYCLE_MANIFEST_PATH.stat().st_size == 0
        )
        with CYCLE_MANIFEST_PATH.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(CYCLE_MANIFEST_COLUMNS)
            writer.writerow([
                detected_at,
                cycle.eqp_id,
                cycle.recipe_id,
                info["alid"],
                info["utc9"],
                info["alarm_name"],
                cycle.run_status,
                cycle.failed_step,
                cycle.failure_class,
                cycle.outcome_status,
                cycle.outcome_path,
                cycle.key_decision,
                cycle.best_xy,
                cycle.frame_count,
                cycle.recording_dir,
                cycle.run_dir,
            ])
        print(
            f"[INFO] cycle manifest 기록 → {CYCLE_MANIFEST_PATH} "
            f"(EQP_ID={cycle.eqp_id}, run={cycle.run_status}, outcome={cycle.outcome_status or '-'})"
        )
    except Exception as exc:
        print(f"[WARNING] cycle manifest 기록 실패: {exc}")


# ------------------------------------------------------------------
# keep-awake (기존 로직 유지).
# ------------------------------------------------------------------


def _set_keep_awake(enable: bool) -> None:
    """PC 절전/디스플레이 끄기를 막거나(enable=True) 원복(False)한다 (Windows 전용).

    SetThreadExecutionState 로 시스템/디스플레이 활성 요구를 건다. 마우스/키보드를
    전혀 건드리지 않으므로 RCS GUI 자동화와 충돌하지 않는다. 화면 캡처가 검게
    나오지 않도록 디스플레이도 켜둔다. 비Windows/실패 시 조용히 무시.
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
        if enable:
            print("[INFO] 현재 OS 에서 keep-awake 미지원 - 생략")
    except Exception as exc:
        print(f"[WARNING] keep-awake 설정 실패: {exc}")


# ------------------------------------------------------------------
# 알람 처리.
# ------------------------------------------------------------------


def process_fail_rows(
    fails,
    active_tools: set[str],
    settings: Workflow3Settings,
) -> int:
    """EQP_ID 기준 edge-triggered 로 신규 알람마다 사이클을 수행한다.

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

        print(
            f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
            f"ALID={info['alid']}, RECIPE_ID={info['recipe_id']}, "
            f"LOT_TYPE={info['lot_type_cd']}, 시각={alarm_time}"
        )
        append_alarm_record(
            eqp_id, alarm_time, info["alarm_name"], info["alid"],
            recipe_id=info["recipe_id"],
            operation_desc=info["operation_desc"],
            lot_type_cd=info["lot_type_cd"],
        )
        if settings.popup_enabled:
            notify_align_fail_popup(
                eqp_id, alarm_time, info["alarm_name"],
                recipe_id=info["recipe_id"],
                operation_desc=info["operation_desc"],
                lot_type_cd=info["lot_type_cd"],
                timeout_sec=settings.popup_timeout_sec,
            )

        # consensus 재료 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
        # 게이트(gather_enabled/recipe_id/downloader)는 gather_success_async 내부에서 판정.
        gather_success_async(eqp_id, info["recipe_id"], settings)

        # rcp/msr 1차 입력 — 사이클이 assets(보정)를 읽기 전에 **동기** 다운로드.
        # MES 가 align_images 트리에 직접 적재하면 downloader 부재로 자동 skip.
        # 게이트(rcp_msr_gather_enabled/recipe_id/downloader)는 gather_rcp_msr 내부에서 판정.
        gather_rcp_msr(eqp_id, info["recipe_id"], settings)

        # 알람별 사이클 — RECIPE_ID 유무와 무관하게 접속+녹화는 수행(보정만 RECIPE_ID 필요).
        # cube 알림(처리 실패 시)은 사이클 내부에서 outcome 기반으로 발송된다.
        if settings.cycle_enabled:
            cycle = run_alarm_cycle(
                eqp_id,
                info["recipe_id"],
                settings,
                tag=_alarm_time_to_tag(info["utc9"]),
            )
        else:
            cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
            cycle.run_status = "cycle_disabled"

        append_cycle_manifest(info, cycle)

        active_tools.add(eqp_id)
        newly_handled += 1

    return newly_handled


# ------------------------------------------------------------------
# 메인 루프.
# ------------------------------------------------------------------


def monitor_loop(settings: Workflow3Settings | None = None) -> None:
    """메인 감지 루프 — poll 주기마다 알람을 조회해 신규 Align Fail 사이클을 돌린다."""
    settings = settings or load_workflow3_settings()
    source = load_alarm_source(settings.alarm_source)

    if settings.keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)

    print(
        f"[INFO] Align Fail 모니터링 시작 (소스={source.kind}, "
        f"주기={settings.poll_interval_sec}s, 윈도우={settings.detection_window_sec}s, "
        f"팝업={'on' if settings.popup_enabled else 'off'}, "
        f"cube알림={'on' if settings.rich_notify_enabled else 'off'}, "
        f"사이클={'on' if settings.cycle_enabled else 'off'}, "
        f"보정={'on' if settings.correction_enabled else 'off'}"
        f"{'(dry-run)' if settings.correction_enabled and settings.correction_dry_run else ''})"
    )
    print(f"[INFO] 알람 로그: {ALARM_LOG_PATH}")
    print(f"[INFO] 사이클 manifest: {CYCLE_MANIFEST_PATH}")
    print(
        "[INFO] 각 신규 Align Fail: RCS 확보 → 접속 → 상시 녹화 → SEM panel → CV 보정 "
        "→ (실패 시 cube 알림 + 엔지니어 watch) → tool 닫기. 중복 알람은 한 번만 처리."
    )

    while True:
        try:
            poll_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO] {poll_time} - 알람 조회 (최근 {settings.detection_window_sec}s 윈도우)")
            alarms = source.poll()
            fails = source.filter_align_fail(alarms)
            fails = filter_rows_within_window(fails, settings.detection_window_sec)

            if _alarm_rows_empty(fails):
                if active_tools:
                    for eqp_id in sorted(active_tools):
                        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
                    active_tools.clear()
                if not idle_logged:
                    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - Align Fail 없음")
                    idle_logged = True
            else:
                idle_logged = False
                count = process_fail_rows(fails, active_tools, settings)
                if count == 0:
                    print(
                        f"[INFO] {datetime.now().strftime('%H:%M:%S')} - "
                        f"신규 없음 (활성 {len(active_tools)}대 유지)"
                    )
        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break
        except Exception as exc:
            print(f"[ERROR] 감지 루프 예외: {exc}")

        try:
            time.sleep(settings.poll_interval_sec)
        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break

    if settings.keep_awake:
        _set_keep_awake(False)
    print("[INFO] 감지 종료")


if __name__ == "__main__":
    monitor_loop()
