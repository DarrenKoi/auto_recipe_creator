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

from poc.workflow_3 import ALIGN_FAIL_ALID, LOG_DIR
from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.monitor.alarm_source import load_alarm_source
from poc.workflow_3.monitor.cycle import CycleResult, run_alarm_cycle
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted, start_abort_hotkey
from poc.workflow_3.monitor.notify import (
    ALARM_LOG_PATH,
    CORRECTED_UNVERIFIED,
    VIEW_ONLY_OBSERVATION,
    notify_align_fail_popup,
    send_detection_notify_async,
)
from poc.workflow_3.monitor.rcp_msr_gather import gather_rcp_msr
from poc.workflow_3.monitor.success_gather import gather_success_async
from poc.workflow_3.monitor.recovery_episode import alarm_fingerprint
from poc.workflow_3.util import make_timestamp_tag
from poc.workflow_3.logger import log_work2_event

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


def _collapse_rows_by_tool(
    fails,
    target_alid: str = ALIGN_FAIL_ALID,
) -> dict[str, dict]:
    """같은 EQP_ID 의 여러 알람 row 를 하나로 합친다 (첫 번째 row 채택).

    ALID 처리는 세 갈래:
      * 누락/NaN/빈 문자열 → 'unknown' 으로 stamp 후 계속 (관측 가능성 유지).
      * 있고 target_alid 와 같음 → 수락.
      * 있고 target_alid 와 다름 → target ALID 가 아니므로 건너뜀.
    두 번째로 보는 EQP_ID 는 조용히 무시(첫 row 채택) — 후속 row 가 더 좋은
    필드를 가져도 덮어쓰지 않는다.
    """
    by_tool: dict[str, dict] = {}
    for row in _iter_alarm_rows(fails):
        eqp_id = str(_row_value(row, "EQP_ID", "eqp_id", "tool_name") or "").strip()
        if not eqp_id:
            print("[WARNING] EQP_ID 없는 Align Fail row 발견 - 건너뜀")
            continue

        raw_alid = _row_value(row, "ALID", "alarm_id", "al_id")
        if (
            raw_alid is None
            or (isinstance(raw_alid, float) and pd.isna(raw_alid))
            or str(raw_alid).strip() == ""
        ):
            print(f"[WARNING] ALID 누락 - 'unknown' 으로 stamp 후 계속 (EQP_ID={eqp_id})")
            alid = "unknown"
        else:
            alid = str(raw_alid).strip()
            if alid != target_alid:
                print(
                    f"[WARNING] ALID={alid} (≠ {target_alid}) - "
                    f"target ALID 가 아니므로 건너뜀 (EQP_ID={eqp_id})"
                )
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
            "alid": alid,
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


# 점유(다른 사용자 사용 중)로 접속을 포기한 사이클의 failure_class — active 미등록 + cooldown.
_OCCUPIED_FAILURE_CLASSES = {"rcs_occupied", "rcs_occupied_select"}
# List 오클릭(다른 tool 창이 열림)도 재시도 대상 — 장비 탓이 아니라 우리 인식 실패라
# active 로 굳혀 버리면 이 알람은 영영 처리되지 않는다. cooldown 은 점유와 공유한다.
_MISCLICK_FAILURE_CLASSES = {"wrong_tool_opened"}
# 확인 게이트가 막아 공유 요청을 못 보낸 경우 - 장비 탓이 아니라 우리 인식 실패라
# 오클릭과 같은 성격이다(cooldown 후 재시도).
_SHARE_FAILURE_CLASSES = {"rcs_share_confirm_failed"}
_RETRY_LATER_FAILURE_CLASSES = (
    _OCCUPIED_FAILURE_CLASSES | _MISCLICK_FAILURE_CLASSES | _SHARE_FAILURE_CLASSES
)

# outcome 기반 재시도 - 사이클이 **완주했더라도** 성공으로 등록하면 안 되는 status.
# _cycle_failed 는 run_status/failed_step 만 보므로, 이 둘을 그대로 두면 active_tools 에
# 등록되어 알람이 해제될 때까지 재시도되지 않는다. 그러면 점유자가 tool 을 놓아준 뒤에도
# 우리는 돌아가지 않아 실제 보정이 돌 기회를 영영 잃는다.
_RETRY_LATER_OUTCOME_STATUSES = {VIEW_ONLY_OBSERVATION, CORRECTED_UNVERIFIED}

# failure_class -> 로그에 남길 사람이 읽을 사유. 미등록 class 는 점유 추정으로 본다.
_RETRY_LATER_REASONS = {
    "wrong_tool_opened": "List 오클릭(다른 tool 창 열림)",
    "rcs_share_confirm_failed": "공유 요청 라벨 확인 실패(클릭 안 함)",
}


def _cycle_failed(cycle) -> bool:
    """사이클이 정상 완료되지 못했는지 - 실패 cooldown 트리거 판정.

    True: 예외로 끝났거나(run_status='error') runner 가 step 실패로 중단(failed_step).
    False: 정상 수행. **correction fallback 은 실패가 아니다** - _exec_run_correction
    은 outcome.status 와 무관하게 success 를 반환하므로(cycle.py:468-475) fallback 은
    run_status='completed' 로 온다. 엔지니어 인계는 정상 경로이며 이미 알람 해제까지
    active_tools 에 머문다.
    """
    return cycle.run_status == "error" or bool(cycle.failed_step)


def _defer_retry(occupied_cooldown: dict, eqp_id: str, delay_sec: float,
                 reason: str, *, level: str = "INFO") -> None:
    """tool 을 active 로 굳히지 않고 cooldown 에 넣는다 (사유를 한 줄로 남긴다).

    실패/점유/오클릭/미확정 네 경로가 전부 같은 두 동작(만료시각 기록 + 사유 출력)을
    하므로 한곳에 모은다. 매 poll 재시도하면 단일 RCS 커서를 독점해 다른 알람을
    굶긴다(F2) - 그래서 어느 경로든 유예가 필요하다.
    """
    occupied_cooldown[eqp_id] = time.time() + delay_sec
    print(f"[{level}] EQP_ID={eqp_id} {reason} - active 미등록, {delay_sec:.0f}s 후 재시도")


def _should_retry_later(cycle) -> bool:
    """이 사이클을 active 로 굳히지 않고 cooldown 재시도로 보내야 하는가.

    사이클이 완주했어도(run_status='completed') outcome 이 '보정이 실제로 반영되었다'
    를 보장하지 못하면 성공으로 등록하지 않는다. 그래야 점유가 풀렸을 때 다시 붙는다.
    `_cycle_failed` 와 달리 **outcome** 을 본다 - 두 판정은 서로 다른 축이다.
    """
    return (cycle.outcome_status or "") in _RETRY_LATER_OUTCOME_STATUSES


def process_fail_rows(
    fails,
    active_tools: set[str],
    settings: Workflow3Settings,
    occupied_cooldown: dict | None = None,
    view_only_attempts: dict | None = None,
    episodes=None,
) -> int:
    """EQP_ID 기준 edge-triggered 로 신규 알람마다 사이클을 수행한다.

    - 같은 EQP_ID 에서 여러 알람 code 가 와도 하나로 취급.
    - 이전 poll 에서 이미 활성이던 EQP_ID 는 다시 처리하지 않음.
    - 이번 poll 에 사라진 EQP_ID 는 `active_tools` 에서 제거 (복구 시 재처리 가능).
    - 점유(select 팝업)/오클릭으로 포기했거나 사이클이 실패한(run_status='error' 또는
      failed_step) EQP_ID 는 `active_tools` 에 넣지 않고 `occupied_cooldown` 에 만료시각을
      기록해, cooldown 동안 재시도를 건너뛴다(만료 후 또는 알람 해제 후 재시도 가능).
      이 dict 는 점유/실패 두 사유를 함께 커버한다. `occupied_cooldown` 은
      {eqp_id: 만료 epoch} dict.
    - tool 1대의 처리 중 예외는 같은 poll 의 나머지 tool 처리를 막지 않는다(F5) -
      예외를 던진 tool 도 cooldown 에 등록해 다음 poll 에 같은 예외를 반복하지 않게 한다.

    `episodes` 는 Recovery Episode tracker(`recovery_episode.EpisodeTracker`) 또는 None.
    None 이면 Episode 수집을 하지 않으며 동작은 종전과 같다(기본 off 플래그).

    `active_tools`/`occupied_cooldown` 는 in-place 로 갱신된다. 새로 처리한 개수를 반환.
    """
    if occupied_cooldown is None:
        occupied_cooldown = {}
    if view_only_attempts is None:
        view_only_attempts = {}
    by_tool = _collapse_rows_by_tool(fails)
    current_tools = set(by_tool.keys())

    # 재시작 복구: 첫 poll 에 한 번만 capture tree 를 훑어 열린 Episode 를 되찾고,
    # 이번 알람 목록에 없는 것은 alarm_gone_during_restart 로 닫는다. tracker 가
    # 자체적으로 1회 실행을 보장하므로 매 poll 불러도 된다.
    if episodes is not None:
        episodes.resume_from_disk(
            alarm_fingerprint(info) for info in by_tool.values()
        )

    # cooldown 만료/알람해제 정리 → 남은 것은 '아직 점유 추정' 으로 이번 poll 에서 건너뜀.
    now = time.time()
    for eqp_id in list(occupied_cooldown):
        if eqp_id not in current_tools or now >= occupied_cooldown[eqp_id]:
            del occupied_cooldown[eqp_id]

    # 시도 카운터는 알람이 해제되면 리셋한다(active_tools/cooldown 과 같은 생애주기).
    # 만료 시각이 아니라 알람 유지 여부로만 지운다 - 같은 알람이 이어지는 동안의
    # 연속 횟수를 세는 것이 목적이기 때문이다.
    for eqp_id in list(view_only_attempts):
        if eqp_id not in current_tools:
            del view_only_attempts[eqp_id]
    cooling = current_tools & set(occupied_cooldown)

    new_tools = current_tools - active_tools - cooling
    cleared_tools = active_tools - current_tools

    for eqp_id in sorted(cleared_tools):
        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
    active_tools.difference_update(cleared_tools)

    # Episode 는 알람이 poll 에서 사라지는 순간 닫힌다 - active_tools 가 아니라
    # 현재 알람 목록이 기준이다(cooldown 중인 tool 은 알람이 살아 있어 안 닫힌다).
    if episodes is not None:
        episodes.close_cleared(current_tools)

    newly_handled = 0
    for eqp_id in sorted(new_tools):
        handle = None
        try:
            info = by_tool[eqp_id]
            alarm_time = str(info["alarm_time"] or "")

            # Episode 는 **첫 GUI step 전에** 열린다. 사이클이 예외로 끝나도
            # "이 알람을 건드렸다" 는 사실이 파일로 남아야 하기 때문이다.
            # 수집 off(episodes=None)면 tag 계산도 종전 그대로 둔다.
            tag = _alarm_time_to_tag(info["utc9"])
            if episodes is not None:
                handle = episodes.begin_attempt(
                    info, settings, tag=tag or make_timestamp_tag()
                )
                tag = handle.tag

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

            # 감지 시점 cube 사전 고지 — 기본 off. 켜면 알람 1건당 cube 가 2회 나간다
            # (여기 + 사이클 종료 후 outcome). 반자동 모드는 결과 알림이 항상 발송되므로
            # (awaiting_engineer_ok) 사전 고지는 중복으로 체감된다. 화면이 저절로 움직이는
            # 것을 미리 알려야 하는 상황에서만 ALIGN_FAIL_DETECTION_NOTIFY=1 로 켠다.
            send_detection_notify_async(
                eqp_id, info["recipe_id"],
                enabled=settings.rich_notify_enabled and settings.detection_notify_enabled,
            )

            # consensus 재료 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
            # 게이트(gather_enabled/recipe_id/downloader)는 gather_success_async 내부에서 판정.
            gather_success_async(eqp_id, info["recipe_id"], settings)

            # rcp/msr 1차 입력 — 사이클이 assets(보정)를 읽기 전에 **동기** 다운로드.
            # MES 가 align_images 트리에 직접 적재하면 downloader 부재로 자동 skip.
            # 게이트(rcp_msr_gather_enabled/recipe_id/downloader)는 gather_rcp_msr 내부에서 판정.
            gather_rcp_msr(eqp_id, info["recipe_id"], settings,
                           timeout_sec=settings.rcp_gather_timeout_sec)

            # 알람별 사이클 — RECIPE_ID 유무와 무관하게 접속+녹화는 수행(보정만 RECIPE_ID 필요).
            # cube 알림(처리 실패 시)은 사이클 내부에서 outcome 기반으로 발송된다.
            if settings.cycle_enabled:
                cycle = run_alarm_cycle(
                    eqp_id,
                    info["recipe_id"],
                    settings,
                    tag=tag,
                    # 수집 off 면 넘기지 않는다 - 사이클 폴더 규약이 종전 그대로여야 한다.
                    **(
                        {
                            "attempt_seq": handle.attempt_seq,
                            "episode_id": handle.episode_id,
                        }
                        if handle
                        else {}
                    ),
                )
            else:
                cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
                cycle.run_status = "cycle_disabled"

            if handle is not None:
                episodes.finish_attempt(handle, cycle)
            append_cycle_manifest(info, cycle)

            # 점유(select)로 포기한 경우: active 에 넣지 않고 cooldown 등록 → 만료 후 재시도.
            if cycle.failure_class in _RETRY_LATER_FAILURE_CLASSES:
                _defer_retry(
                    occupied_cooldown, eqp_id, settings.occupied_retry_cooldown_sec,
                    _RETRY_LATER_REASONS.get(cycle.failure_class, "점유(select) 추정"),
                )
            elif _should_retry_later(cycle):
                # 완주했지만 보정 반영이 보장되지 않은 사이클. active 로 굳히면 알람
                # 해제까지 재시도되지 않아, 점유가 풀려도 우리는 돌아가지 않는다.
                attempts = view_only_attempts.get(eqp_id, 0) + 1
                view_only_attempts[eqp_id] = attempts
                if attempts >= settings.share_max_attempts:
                    # 상한 도달 - 더 재시도하면 cooldown 마다 cube 가 나가고 단일 RCS
                    # 커서를 계속 점유한다. 엔지니어는 이미 상한만큼 통보받았다.
                    active_tools.add(eqp_id)
                    print(
                        f"[WARNING] EQP_ID={eqp_id} {cycle.outcome_status} "
                        f"{attempts}회 - 상한 도달, 재시도 중단(알람 해제까지 대기)"
                    )
                else:
                    _defer_retry(
                        occupied_cooldown, eqp_id, settings.failure_retry_cooldown_sec,
                        f"{cycle.outcome_status} "
                        f"({attempts}/{settings.share_max_attempts})",
                    )
            elif _cycle_failed(cycle):
                _defer_retry(
                    occupied_cooldown, eqp_id, settings.failure_retry_cooldown_sec,
                    f"사이클 실패(status={cycle.run_status}, "
                    f"step={cycle.failed_step or '-'})",
                    level="WARNING",
                )
            else:
                active_tools.add(eqp_id)
            newly_handled += 1
        except Exception as exc:
            # tool 1대의 예외가 같은 poll 의 나머지 tool 을 건너뛰게 하면 안 된다(F5).
            # 던진 tool 은 cooldown 에 넣어 다음 poll 에 같은 예외를 반복하지 않게 한다.
            if handle is not None:
                episodes.fail_attempt(handle, f"{type(exc).__name__}: {exc}")
            _defer_retry(
                occupied_cooldown, eqp_id, settings.failure_retry_cooldown_sec,
                f"처리 예외({type(exc).__name__}: {exc}) - 나머지 tool 은 계속",
                level="ERROR",
            )
            log_work2_event(
                component=LOG_COMPONENT, message="tool_process_error", level="error",
                eqp_id=eqp_id, error=str(exc),
            )

    return newly_handled


# ------------------------------------------------------------------
# 메인 루프.
# ------------------------------------------------------------------


def _describe_done_signals(settings: Workflow3Settings) -> str:
    """engineer watch 종료 신호 구성을 시작 로그 한 줄로 요약한다.

    "왜 watch 가 300s 를 다 채웠나 / 왜 일찍 끊겼나" 를 사후에 가르는 값이라, 그 세션이
    어떤 신호로 돌았는지 로그만 보고 판별되어야 한다.
    """
    if not settings.engineer_done_detect_enabled:
        return "off - cap/창닫힘까지 녹화"
    signals = ["분자 N 증가"]
    if settings.engineer_done_idle_sec > 0:
        signals.append(f"커서 {settings.engineer_done_idle_sec:.0f}s 정지")
    if settings.engineer_done_assist_enabled:
        signals.append("Assist 판독")
    return "on (" + " / ".join(signals) + ")"


def _run_rcs_preflight(settings: Workflow3Settings):
    """루프 진입 전 RCS 를 로그인 + List 탭까지 올려둔다. 실패해도 None/status 로 끝난다.

    `open_rcs -> workflow_login -> List 탭 -> 알람 대기` 순서를 여기서 만든다. 준비를
    안 하면 RCS 가 안 떠 있을 때 **첫 알람이 그 비용을 통째로 낸다** - 부팅 + 로그인 +
    List 탭이 알람 처리 시간에 붙고, 그동안 장비는 멈춰 있다.

    협력자 조립만 하고 판정은 `rcs_preflight.ensure_rcs_session_ready` 가 한다(그래야
    Mac 에서 시험된다). RCS 모듈이 없는 개발 PC(replay dry-run)에서는 조용히 건너뛴다 -
    기동 준비 때문에 Mac 검증 경로가 깨지면 안 된다.

    반환: PreflightOutcome, 또는 준비를 아예 돌리지 않았으면 None.
    """
    if not settings.rcs_preflight_enabled:
        print("[INFO] RCS 기동 준비 생략(ALIGN_FAIL_RCS_PREFLIGHT=0) - 알람 시 복구는 그대로.")
        return None

    try:
        from poc.workflow_3.monitor.cycle import (
            _list_process_windows,
            _scan_rcs_processes,
            _terminate_process,
        )
        from poc.workflow_3.monitor.rcs_preflight import ensure_rcs_session_ready
        from poc.workflow_3.monitor.rcs_recovery import recover_rcs_session
        from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
        from poc.workflow_3.rcs.open_rcs import launch_rcs
        from poc.workflow_3.rcs.view_list_tab_rcs import click_list_tab_in_main_window
        from poc.workflow_3.rcs.workflow_login import run_login_workflow
    except Exception as exc:
        print(f"[INFO] RCS 기동 준비 생략(Windows 전용 의존성 없음): {exc}")
        return None

    def _recover():
        return recover_rcs_session(
            settings,
            find_processes_fn=_scan_rcs_processes,
            launch_fn=launch_rcs,
            login_fn=run_login_workflow,
            wait_window_fn=wait_for_rcs_main_window,
            list_windows_fn=_list_process_windows,
            terminate_fn=_terminate_process,
        )

    def _open_list(window, title, backend):
        return click_list_tab_in_main_window(window, title, backend).exit_code

    try:
        return ensure_rcs_session_ready(
            settings,
            find_window_fn=wait_for_rcs_main_window,
            recover_fn=_recover,
            open_list_fn=_open_list,
        )
    except Exception as exc:
        # 준비가 어떤 이유로 깨져도 감시는 떠야 한다 - 알람 시 복구가 다시 시도한다.
        print(f"[WARNING] RCS 기동 준비 예외(감시는 계속): {exc}")
        return None


def monitor_loop(settings: Workflow3Settings | None = None) -> None:
    """메인 감지 루프 — poll 주기마다 알람을 조회해 신규 Align Fail 사이클을 돌린다."""
    settings = settings or load_workflow3_settings()
    source = load_alarm_source(settings.alarm_source)

    if settings.keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()
    occupied_cooldown: dict = {}  # {eqp_id: 재시도 가능 epoch} — 점유(select)로 포기한 tool.
    view_only_attempts: dict = {}  # {eqp_id: 연속 view-only/unverified 사이클 횟수}
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)
    # Recovery Episode tracker — 기본 off. None 이면 수집 경로가 통째로 비활성이다.
    episodes = None
    if settings.episode_collect_enabled:
        from poc.workflow_3.monitor.recovery_episode import EpisodeTracker

        episodes = EpisodeTracker()

    print(
        f"[INFO] Align Fail 모니터링 시작 (소스={source.kind}, "
        f"주기={settings.poll_interval_sec}s, 윈도우={settings.detection_window_sec}s, "
        f"팝업={'on' if settings.popup_enabled else 'off'}, "
        f"cube알림={'on' if settings.rich_notify_enabled else 'off'}"
        f"{'+사전고지' if settings.rich_notify_enabled and settings.detection_notify_enabled else ''}, "
        f"사이클={'on' if settings.cycle_enabled else 'off'}, "
        f"보정={'on' if settings.correction_enabled else 'off'}"
        f"{'(dry-run)' if settings.correction_enabled and settings.correction_dry_run else ''}"
        f"{'' if not settings.correction_enabled else (', OK클릭=on' if settings.ok_click_enabled else ', OK클릭=off(엔지니어)')})"
    )
    # 녹화 종료 조건은 사후에 "왜 300s 를 다 채웠나" 를 가르는 값이라 시작 로그에 남긴다.
    # done 감지가 off 면 창 닫힘 또는 watch cap 만이 종료 조건이다.
    print(
        f"[INFO] engineer watch: 상한={settings.engineer_watch_sec:.0f}s, "
        f"작업완료 감지={_describe_done_signals(settings)}"
    )
    print(f"[INFO] 알람 로그: {ALARM_LOG_PATH}")
    print(f"[INFO] 사이클 manifest: {CYCLE_MANIFEST_PATH}")
    # 로케이터 조합은 로그인/List 탭/tool 선택/PM 버튼을 한 번에 바꾸는 광역 스위치라,
    # 시작 로그에 실제 적용값을 남겨야 사후에 어떤 조합으로 돈 세션인지 판별된다.
    from poc.workflow_3.vlm.ui_venus_mai_locator import describe_locator_combo
    print(f"[INFO] VLM 로케이터 조합: {describe_locator_combo(settings.locator_combo)}")
    # 관리자 권한 진단 — production 도 BlockInput·강제 전면화에 의존하므로 비elevated 경고 필요.
    # window_utils 는 pywinauto 를 요구하므로 개발 PC(replay dry-run)에서는 없을 수 있다.
    # 진단 한 줄 때문에 dry-run 자체가 죽으면 안 된다.
    try:
        from poc.workflow_3.util.window_utils import print_elevation_status

        print_elevation_status()
    except ImportError as exc:
        print(f"[INFO] 권한 진단 생략(Windows 전용 의존성 없음): {exc}")
    print(
        "[INFO] 각 신규 Align Fail: RCS 확보 → 접속 → 상시 녹화 → SEM panel → CV 보정 "
        "→ (실패 시 cube 알림 + 엔지니어 watch) → tool 닫기. 중복 알람은 한 번만 처리."
    )

    # 알람 대기 전에 RCS 를 로그인 + List 탭까지 올려둔다(실행 -> 로그인 -> List -> 대기).
    # best-effort 라 실패해도 루프는 뜬다 - 알람이 오면 사이클의 ensure_rcs_ready 가
    # 같은 복구를 다시 시도하므로, 준비 실패가 곧 감시 중단이 되어서는 안 된다.
    # 긴급 해제 단축키 - preflight(첫 실제 마우스 조작) **전에** 띄운다.
    start_abort_hotkey(settings.abort_hotkey)

    _run_rcs_preflight(settings)

    while True:
        if is_aborted():
            print(f"[WARNING] 긴급 해제됨({abort_reason()}) - 감지 루프를 종료합니다.")
            break
        try:
            alarms = source.poll()
            fails = source.filter_align_fail(alarms)
            fails = filter_rows_within_window(fails, settings.detection_window_sec)

            if _alarm_rows_empty(fails):
                if active_tools:
                    for eqp_id in sorted(active_tools):
                        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
                    active_tools.clear()
                # 알람이 전부 사라진 poll 은 process_fail_rows 를 거치지 않는다 -
                # Episode clearance 는 이 분기에서도 닫아야 한다(빠뜨리면 열린 채
                # 남아 다음 알람이 같은 Episode 로 잘못 재개된다).
                if episodes is not None:
                    # 재시작 직후 첫 poll 이 빈 경우도 스캔을 거쳐야 한다 - 그러지
                    # 않으면 알람이 이미 풀린 고아 Episode 가 영영 열린 채 남는다.
                    episodes.resume_from_disk(())
                    episodes.close_cleared(())
                if not idle_logged:
                    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - Align Fail 없음")
                    idle_logged = True
            else:
                idle_logged = False
                process_fail_rows(
                    fails, active_tools, settings, occupied_cooldown,
                    view_only_attempts, episodes=episodes,
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


def _apply_live_mode_defaults() -> None:
    """실장비 운전 기본값(SAFE_MODE=0 + 보정 dry-run off)을 진입점에서 못박는다.

    이 모니터는 "실제로 보정하는 루프"가 존재 이유이므로, 매 실행마다 env 두 개를
    손으로 붙이지 않아도 실클릭 모드로 뜨게 한다. 대신 되돌릴 수 없는 조작을 기본으로
    켜는 셈이라 두 가지 안전장치를 함께 둔다.

      * setdefault 라 **실제 셸 env 가 항상 이긴다** - 점검만 하려면
        `SAFE_MODE=1 uv run python ...` 로 실행하면 클릭이 전부 막힌다.
      * 시작 시 배너로 현재 운전 모드를 크게 남긴다(로그만 봐도 그 세션이 실클릭이었는지
        판별되어야 한다).

    seed_env() 보다 **먼저** 불러야 한다. 그래야 오피스 PC 의 workflow_3_config.py 사본에
    남아 있을 수 있는 CORRECTION_DRY_RUN=1 이 이 기본값을 덮지 않는다(seed_env 도
    setdefault 라 먼저 잡은 쪽이 이긴다). 그 결과 우선순위는
    셸 env > 이 기본값 > workflow_3_config.py > config.py 기본값 이며,
    무시된 config 값은 seed_env 가 콘솔에 그대로 보고한다.
    """
    import os

    os.environ.setdefault("SAFE_MODE", "0")
    os.environ.setdefault("ALIGN_FAIL_CORRECTION_DRY_RUN", "0")

    safe_mode = os.environ.get("SAFE_MODE", "0")
    dry_run = os.environ.get("ALIGN_FAIL_CORRECTION_DRY_RUN", "0")
    live = safe_mode == "0" and dry_run == "0"
    print("=" * 70)
    if live:
        print("[WARNING] 실운전 모드: 실제 마우스 클릭이 발생합니다 "
              "(접속 더블클릭 + align point reposition).")
        hotkey = os.environ.get("ALIGN_FAIL_ABORT_HOTKEY", "<ctrl>+<alt>+q")
        print(f"[WARNING] 긴급 해제 단축키: {hotkey} - 누르면 마우스를 즉시 돌려받습니다.")
        print("[WARNING] 점검만 하려면 중단 후 'SAFE_MODE=1' 을 붙여 다시 실행하세요.")
    else:
        print(f"[INFO] 점검 모드: SAFE_MODE={safe_mode}, "
              f"ALIGN_FAIL_CORRECTION_DRY_RUN={dry_run} (셸 env 가 기본값을 덮었습니다)")
    print("=" * 70)


if __name__ == "__main__":
    # 실장비 운전 기본값을 먼저 못박고(셸 env 는 그대로 우선), 그 다음 실편집
    # workflow_3_config.py 의 나머지 토글을 env 로 브리지한다. 둘 다 setdefault 이며
    # load_workflow3_settings 가 env 를 읽기 전에 끝나야 적용된다.
    from poc.workflow_3.workflow_3_config_loader import seed_env

    _apply_live_mode_defaults()
    seed_env()
    monitor_loop()
