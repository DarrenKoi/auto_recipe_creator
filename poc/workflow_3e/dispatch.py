"""측정 abort 잡 디스패치 — EQP_ID 기준 edge-trigger 로 신규 알람마다 abort 사이클.

process_fail_rows(align)의 얇은 형제: 팝업/gather/correction 없이 run_abort_cycle 만
돌린다. workflow_3 의 edge-trigger/manifest 헬퍼를 그대로 재사용한다(core 무수정).
"""

import csv
from datetime import datetime

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.monitor.align_fail_monitor import (
    CYCLE_MANIFEST_COLUMNS,
    _OCCUPIED_FAILURE_CLASSES,
    _alarm_time_to_tag,
    _collapse_rows_by_tool,
    append_alarm_record,
)
from poc.workflow_3.monitor.cycle import CycleResult
from poc.workflow_3e.abort_cycle import run_abort_cycle

import time

# align manifest 와 같은 컬럼(CycleResult)을 쓰되 파일만 분리한다.
ABORT_MANIFEST_PATH = LOG_DIR / "measurement_abort_cycles.csv"


def append_abort_manifest(info: dict, cycle: CycleResult) -> None:
    """측정 abort 사이클 1건을 measurement_abort_cycles.csv 에 한 줄 누적한다.

    파일이 없으면 헤더를 먼저 쓴다. 기록 실패는 삼켜 루프가 죽지 않게 한다.
    """
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ABORT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not ABORT_MANIFEST_PATH.exists() or ABORT_MANIFEST_PATH.stat().st_size == 0
        )
        with ABORT_MANIFEST_PATH.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(CYCLE_MANIFEST_COLUMNS)
            writer.writerow([
                detected_at, cycle.eqp_id, cycle.recipe_id, info["alid"], info["utc9"],
                info["alarm_name"], cycle.run_status, cycle.failed_step, cycle.failure_class,
                cycle.outcome_status, cycle.outcome_path, cycle.key_decision, cycle.best_xy,
                cycle.frame_count, cycle.recording_dir, cycle.run_dir,
            ])
        print(
            f"[INFO] abort manifest 기록 -> {ABORT_MANIFEST_PATH} "
            f"(EQP_ID={cycle.eqp_id}, outcome={cycle.outcome_status or '-'})"
        )
    except Exception as exc:
        print(f"[WARNING] abort manifest 기록 실패: {exc}")


def process_abort_rows(meas_fails, aborted_tools: set, settings, abort_cooldown: dict | None = None) -> int:
    """측정 실패 임계 알람을 EQP_ID 기준 edge-trigger 로 처리해 abort 사이클을 돈다.

    - 같은 EQP_ID 의 중복 알람은 active 인 동안 1회만 처리.
    - 이번 poll 에 사라진 EQP_ID 는 aborted_tools 에서 제거(복구 시 재처리).
    - 점유(select)로 포기하면 active 미등록 + cooldown 등록(만료 후 재시도) — align 과 동일.

    잡 비활성(meas_fail_abort_enabled=False)이면 0. aborted_tools/abort_cooldown 은 in-place
    갱신. 새로 처리한 개수를 반환.
    """
    if not settings.meas_fail_abort_enabled:
        return 0
    if abort_cooldown is None:
        abort_cooldown = {}

    by_tool = _collapse_rows_by_tool(meas_fails)
    current_tools = set(by_tool.keys())

    now = time.time()
    for eqp_id in list(abort_cooldown):
        if eqp_id not in current_tools or now >= abort_cooldown[eqp_id]:
            del abort_cooldown[eqp_id]
    cooling = current_tools & set(abort_cooldown)
    for eqp_id in sorted(cooling):
        print(f"[INFO] EQP_ID={eqp_id} 점유 cooldown 중 - 이번 poll abort 재시도 건너뜀")

    new_tools = current_tools - aborted_tools - cooling
    cleared = aborted_tools - current_tools
    for eqp_id in sorted(cleared):
        print(f"[INFO] 측정 실패 알람 해제: EQP_ID={eqp_id}")
    aborted_tools.difference_update(cleared)

    handled = 0
    for eqp_id in sorted(new_tools):
        info = by_tool[eqp_id]
        print(
            f"[WARNING] 측정 실패 임계 감지: EQP_ID={eqp_id}, ALID={info['alid']}, "
            f"RECIPE_ID={info['recipe_id']}, LOT_TYPE={info['lot_type_cd']}, 시각={info['alarm_time']}"
        )
        append_alarm_record(
            eqp_id, str(info["alarm_time"] or ""), info["alarm_name"], info["alid"],
            recipe_id=info["recipe_id"], operation_desc=info["operation_desc"],
            lot_type_cd=info["lot_type_cd"],
        )
        if settings.cycle_enabled:
            cycle = run_abort_cycle(
                eqp_id, info["recipe_id"], settings, tag=_alarm_time_to_tag(info["utc9"])
            )
        else:
            cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
            cycle.run_status = "cycle_disabled"
        append_abort_manifest(info, cycle)

        if cycle.failure_class in _OCCUPIED_FAILURE_CLASSES:
            abort_cooldown[eqp_id] = time.time() + settings.occupied_retry_cooldown_sec
            print(
                f"[INFO] EQP_ID={eqp_id} 점유(select) 추정 - active 미등록, "
                f"{settings.occupied_retry_cooldown_sec:.0f}s 후 재시도"
            )
        else:
            aborted_tools.add(eqp_id)
        handled += 1

    return handled


__all__ = ["process_abort_rows", "append_abort_manifest", "ABORT_MANIFEST_PATH"]
