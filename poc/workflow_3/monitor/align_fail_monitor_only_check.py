"""실시간 Align Fail '점검 전용' 모니터링 루프 — workflow_3 보조 진입점.

`align_fail_monitor.py` 의 경량 변형이다. 알람마다 [접속 → 첫 화면 1장 캡처 →
tool 닫기] 만 수행하고(실제 reposition/OK 클릭 = 보정 actuation 은 안 함), 상시 녹화 /
SEM panel 확보 / engineer watch 는 전부 뺀다. 과거 데이터 수집은 그대로 유지한다:

  * rcp / msr 이미지: office MES 가 align_images 트리에 직접 적재(코드 개입 없음).
    물리 루트가 workflow_3 로 잡혀 있으면 캡처도 같은 트리에 모인다.
  * 최근 성공(S) align 이미지: monitor 의 `gather_success_async` 가 비차단 다운로드.

캡처 직후, tool 을 닫은 뒤 디스크 저장본으로 **보정 가능성** 을 정적 판정한다
(`run_check_only_cycle` 내부 → `align.diagnostics.feasibility_check.mark_align_feasibility`):
검증된 rcp align key 엔진으로 가능/불가/모호를 가리고, align_consensus_cache 의 최근
성공(S) event 수를 read-only 로 함께 표기해 캡처 옆에 `<tag>_rcs_marked.jpg` +
`<tag>_feasibility.json` 으로 남긴다. 엔지니어가 마킹 한 장으로 "이 fail 은 자동
보정이 됐을까" 를 눈으로 확인하는 평가/점검용이다.

용도: fail 시점 화면 박제 + 과거 데이터 수집 + 보정 가능성 마킹(데이터 수집/점검 모드).
실제 보정 actuation 은 production `align_fail_monitor.py` 가 담당한다. 폴링/edge-trigger/
로그/manifest 골격은 production 과 동일하며, 알람별 사이클만 `run_check_only_cycle` 로
교체한다.

사용법:
  uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py

개발 PC dry-run:
  SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
    uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py
"""

import csv
import time
from datetime import datetime

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.monitor.alarm_source import load_alarm_source

# 행 파싱/필터/기록 등 경로에 무관한 순수 헬퍼는 production 모니터에서 재사용한다
# (중복 유지보수를 피하기 위해). 사이클/manifest/루프만 점검 전용으로 따로 둔다.
from poc.workflow_3.monitor.align_fail_monitor import (
    CYCLE_MANIFEST_COLUMNS,
    _alarm_rows_empty,
    _alarm_time_to_tag,
    _collapse_rows_by_tool,
    _set_keep_awake,
    append_alarm_record,
    filter_rows_within_window,
)
from poc.workflow_3.monitor.cycle import CycleResult, run_check_only_cycle
from poc.workflow_3.monitor.notify import ALARM_LOG_PATH, notify_align_fail_popup
from poc.workflow_3.monitor.success_gather import (
    DOWNLOADER_AVAILABLE,
    gather_success_async,
)

LOG_COMPONENT = "align_fail_only_check"

# 점검 전용 manifest — production align_fail_cycles.csv 와 분리해 결과가 안 섞이게 한다.
CYCLE_MANIFEST_PATH = LOG_DIR / "align_fail_check_cycles.csv"


def append_cycle_manifest(info: dict, cycle: CycleResult) -> None:
    """알람 1건의 메타 + 점검 사이클 결과를 CSV manifest 에 한 줄 누적한다.

    파일이 없으면 헤더를 먼저 쓴다(컬럼은 production 과 동일). 기록 실패는 삼켜
    루프가 죽지 않게 한다.
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
            f"[INFO] check manifest 기록 → {CYCLE_MANIFEST_PATH} "
            f"(EQP_ID={cycle.eqp_id}, run={cycle.run_status}, outcome={cycle.outcome_status or '-'})"
        )
    except Exception as exc:
        print(f"[WARNING] check manifest 기록 실패: {exc}")


def process_fail_rows(
    fails,
    active_tools: set[str],
    settings: Workflow3Settings,
) -> int:
    """EQP_ID 기준 edge-triggered 로 신규 알람마다 점검 사이클을 수행한다.

    production `align_fail_monitor.process_fail_rows` 와 동일한 edge-trigger 규약
    이되, 알람별 사이클만 `run_check_only_cycle`(접속 → 캡처 → 닫기)로 바뀐다.
    `active_tools` 는 in-place 로 갱신되며, 새로 처리한 개수를 반환한다.
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

        # 과거 데이터 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
        # 게이트(gather_enabled/recipe_id/downloader)는 gather_success_async 내부에서 판정.
        gather_success_async(eqp_id, info["recipe_id"], settings)

        # 점검 전용 사이클 — 접속 → 첫 화면 1장 캡처 → tool 닫기 (보정/녹화 없음).
        if settings.cycle_enabled:
            cycle = run_check_only_cycle(
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


def _report_data_paths() -> None:
    """시작 시 데이터 입출력 루트를 절대경로로 찍어 'MES/다운로더가 코드와 다른 폴더에
    쓰는지' 를 즉시 드러낸다.

    캡처(captured_img_from_rcs)는 보이는데 rcp/msr(align_images) 과 consensus 캐시가
    비는 가장 흔한 증상의 1차 진단이다. 두 트리는 이 코드가 아니라 외부(office MES /
    success downloader)가 채우므로, 코드가 '읽는' 경로가 실제 '쓰는' 경로와 같은지
    확인하는 것이 핵심이다.
      * align_images: office MES 가 align_img_from_rcp/msr 를 적재(코드는 읽기만).
      * align_consensus_cache: success downloader 가 적재(없으면 gather 비활성).
    """
    # import 는 함수 안에서 — 모듈 로드 부작용/순환을 피하고 진단 비용을 시작 1회로 한정.
    from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR, ALIGN_IMAGES_DIR
    from poc.workflow_3.align.assets import iter_recipe_dirs

    img_root = ALIGN_IMAGES_DIR
    img_exists = img_root.is_dir()
    try:
        n_recipes = len(iter_recipe_dirs(img_root)) if img_exists else 0
    except Exception:
        n_recipes = -1  # glob 실패 — 경로 권한 등.
    print(
        f"[INFO] align_images 루트(코드가 rcp/msr 를 읽는 곳): {img_root} "
        f"(존재={'예' if img_exists else '아니오'}, "
        f"align_img_from_rcp 보유 recipe={n_recipes})"
    )
    if not img_exists:
        print(
            "[WARNING] align_images 루트가 없습니다. office MES 출력 위치와 "
            "ALIGN_IMAGES_DIR 가 다른지 확인하세요(rcp/msr 가 안 보이는 주원인). "
            "MES 가 과거 workflow_1/align_images 에 쓰면 env ALIGN_IMAGES_DIR 로 "
            "그 경로를 가리키거나 MES 출력 위치를 옮겨야 합니다."
        )
    elif n_recipes == 0:
        print(
            "[WARNING] align_images 루트는 있으나 align_img_from_rcp 를 가진 recipe 가 "
            "0개입니다. MES 가 정말 이 경로에 적재 중인지 확인하세요."
        )

    cache_root = ALIGN_CONSENSUS_CACHE_DIR
    print(
        f"[INFO] align_consensus_cache 루트(success S 이미지): {cache_root} "
        f"(존재={'예' if cache_root.is_dir() else '아니오'})"
    )
    print(
        f"[INFO] success downloader: "
        f"{'사용가능' if DOWNLOADER_AVAILABLE else '없음 → consensus gather 비활성(캐시 안 채워짐)'}"
    )


def monitor_loop(settings: Workflow3Settings | None = None) -> None:
    """점검 전용 메인 루프 — poll 주기마다 신규 Align Fail 을 캡처+닫기 처리한다."""
    settings = settings or load_workflow3_settings()
    source = load_alarm_source(settings.alarm_source)

    if settings.keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)

    print(
        f"[INFO] Align Fail 점검 모니터링 시작 (소스={source.kind}, "
        f"주기={settings.poll_interval_sec}s, 윈도우={settings.detection_window_sec}s, "
        f"팝업={'on' if settings.popup_enabled else 'off'}, "
        f"사이클={'on' if settings.cycle_enabled else 'off'}, "
        f"성공이미지수집={'on' if settings.gather_enabled else 'off'}, "
        f"보정가능성마킹={'on' if settings.feasibility_mark_enabled else 'off'})"
    )
    print(f"[INFO] 알람 로그: {ALARM_LOG_PATH}")
    print(f"[INFO] 점검 manifest: {CYCLE_MANIFEST_PATH}")
    _report_data_paths()
    print(
        "[INFO] 각 신규 Align Fail: RCS 확보 → 접속 → 첫 화면 1장 캡처 → tool 닫기 → "
        "보정 가능성 판정(_marked.jpg/_feasibility.json). "
        "과거 데이터(rcp/msr=MES 적재, 성공 S 이미지=gather)도 함께 수집. "
        "보정 actuation/녹화는 하지 않음(production 모니터 담당). 중복 알람은 한 번만 처리."
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
    print("[INFO] 점검 감지 종료")


if __name__ == "__main__":
    monitor_loop()
