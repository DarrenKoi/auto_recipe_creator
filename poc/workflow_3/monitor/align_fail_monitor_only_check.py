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
from poc.workflow_3.monitor.notify import (
    ALARM_LOG_PATH,
    notify_align_fail_popup,
    send_detection_notify_async,
)
from poc.workflow_3.monitor.rcp_msr_gather import (
    RCP_MSR_DOWNLOADER_AVAILABLE,
    gather_rcp_msr,
)
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


def _cycle_failed(cycle) -> bool:
    """사이클이 정상 완료되지 못했는지 - 실패 cooldown 트리거 판정.

    True: 예외로 끝났거나(run_status='error') runner 가 step 실패로 중단(failed_step).
    점유(select) 팝업도 step 실패로 오므로 여기 포함된다 - 점검 모니터는 production
    처럼 점유를 따로 분기하지 않고 같은 cooldown 으로 처리한다.
    False: 정상 수행. correction 계열 fallback 은 실패가 아니다(점검 사이클은 보정
    actuation 자체가 없다).
    """
    return cycle.run_status == "error" or bool(cycle.failed_step)


def process_fail_rows(
    fails,
    active_tools: set[str],
    settings: Workflow3Settings,
    cooldown: dict | None = None,
) -> int:
    """EQP_ID 기준 edge-triggered 로 신규 알람마다 점검 사이클을 수행한다.

    production `align_fail_monitor.process_fail_rows` 와 동일한 edge-trigger 규약
    이되, 알람별 사이클만 `run_check_only_cycle`(접속 → 캡처 → 닫기)로 바뀐다.

    `cooldown` 은 {eqp_id: 재시도 가능 epoch} - 사이클이 실패한 tool 을 매 poll
    재시도하면 직렬화된 단일 RCS 커서를 독점해 다른 알람을 굶긴다(F2). tool 1대의
    예외가 같은 poll 의 나머지를 건너뛰지 않게 본문은 tool 별로 보호한다(F5).

    `active_tools`/`cooldown` 은 in-place 로 갱신된다. 새로 처리한 개수를 반환.
    """
    if cooldown is None:
        cooldown = {}
    by_tool = _collapse_rows_by_tool(fails)
    current_tools = set(by_tool.keys())

    # cooldown 만료/알람해제 정리 → 남은 것은 이번 poll 에서 건너뛴다.
    now = time.time()
    for eqp_id in list(cooldown):
        if eqp_id not in current_tools or now >= cooldown[eqp_id]:
            del cooldown[eqp_id]
    cooling = current_tools & set(cooldown)
    for eqp_id in sorted(cooling):
        print(f"[INFO] EQP_ID={eqp_id} cooldown 중 - 이번 poll 재시도 건너뜀")

    new_tools = current_tools - active_tools - cooling
    cleared_tools = active_tools - current_tools

    for eqp_id in sorted(cleared_tools):
        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
    active_tools.difference_update(cleared_tools)

    newly_handled = 0
    for eqp_id in sorted(new_tools):
        try:
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

            # 감지 시점 cube rich notification — 점검 모드는 보정 actuation 이 없어
            # CorrectionOutcome 을 만들지 않으므로 outcome 기반 notify_correction_outcome
            # (production 전용) 대신 detection-time 변형을 쓴다. office_rich_notify 부재
            # /rich_notify_enabled=off 면 내부에서 조용히 skip(텍스트 로그만).
            send_detection_notify_async(
                eqp_id, info["recipe_id"], enabled=settings.rich_notify_enabled,
            )

            # 과거 데이터 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
            # 게이트(gather_enabled/recipe_id/downloader)는 gather_success_async 내부에서 판정.
            gather_success_async(eqp_id, info["recipe_id"], settings)

            # rcp/msr 1차 입력 — cycle 이 assets(feasibility)를 읽기 전에 **동기** 다운로드.
            # MES 가 align_images 트리에 직접 적재하면 downloader 부재로 자동 skip.
            # 게이트(rcp_msr_gather_enabled/recipe_id/downloader)는 gather_rcp_msr 내부에서 판정.
            # (Task 7 이 여기에 timeout_sec= 를 추가한다.)
            gather_rcp_msr(eqp_id, info["recipe_id"], settings,
                           timeout_sec=settings.rcp_gather_timeout_sec)

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

            if _cycle_failed(cycle):
                cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
                print(
                    f"[WARNING] EQP_ID={eqp_id} 점검 사이클 실패(status={cycle.run_status}, "
                    f"step={cycle.failed_step or '-'}) - active 미등록, "
                    f"{settings.failure_retry_cooldown_sec:.0f}s 후 재시도"
                )
            else:
                active_tools.add(eqp_id)
            newly_handled += 1
        except Exception as exc:
            # tool 1대의 예외가 같은 poll 의 나머지 tool 을 건너뛰게 하면 안 된다(F5).
            cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
            print(f"[ERROR] EQP_ID={eqp_id} 처리 예외 - 나머지 tool 계속: {exc}")

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
    from poc.workflow_3.util.window_utils import print_elevation_status

    # 관리자 권한(elevated) 여부 — UIPI 때문에 비elevated 면 사용자가 다른 앱을 쓰는
    # 중에 RCS 강제 전면화/BlockInput 이 조용히 실패한다(전면화 안 되는 숨은 주원인).
    print_elevation_status()

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
    print(
        f"[INFO] rcp/msr downloader: "
        f"{'사용가능 → 알람 시 align_images 트리로 동기 다운로드' if RCP_MSR_DOWNLOADER_AVAILABLE else '없음 → rcp/msr 은 office MES 직접 적재에 의존'}"
    )


def monitor_loop(settings: Workflow3Settings | None = None) -> None:
    """점검 전용 메인 루프 — poll 주기마다 신규 Align Fail 을 캡처+닫기 처리한다."""
    settings = settings or load_workflow3_settings()
    source = load_alarm_source(settings.alarm_source)

    if settings.keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()
    cooldown: dict = {}  # {eqp_id: 재시도 가능 epoch} — 사이클 실패로 쉬는 tool.
    idle_logged = False  # "Align Fail 없음" 은 idle 진입 시 한 번만 로깅 (poll 마다 X)

    print(
        f"[INFO] Align Fail 점검 모니터링 시작 (소스={source.kind}, "
        f"주기={settings.poll_interval_sec}s, 윈도우={settings.detection_window_sec}s, "
        f"팝업={'on' if settings.popup_enabled else 'off'}, "
        f"사이클={'on' if settings.cycle_enabled else 'off'}, "
        f"성공이미지수집={'on' if settings.gather_enabled else 'off'}, "
        f"보정가능성마킹={'on' if settings.feasibility_mark_enabled else 'off'}, "
        f"보정점미리보기(cursor)={'on' if settings.reposition_preview_enabled else 'off'}, "
        f"보정탐색(zoom in/out)={'on' if settings.zoom_probe_enabled else 'off'}"
        f"/method={settings.zoom_method})"
    )
    print(f"[INFO] 알람 로그: {ALARM_LOG_PATH}")
    print(f"[INFO] 점검 manifest: {CYCLE_MANIFEST_PATH}")
    # 조합 A/B 는 이 점검 모니터에서 먼저 돌리게 되므로, 어떤 조합의 산출물인지
    # 시작 로그에 남긴다(debug_images 하위 디렉터리명과 대조용).
    from poc.workflow_3.vlm.ui_venus_mai_locator import describe_locator_combo
    print(f"[INFO] VLM 로케이터 조합: {describe_locator_combo(settings.locator_combo)}")
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
                count = process_fail_rows(fails, active_tools, settings, cooldown)
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
    # 실편집 workflow_3_config.py 의 토글을 env 로 브리지(있으면). load_workflow3_settings
    # 가 env 를 읽기 전에 호출해야 적용된다. 실제 OS env 가 우선(setdefault).
    from poc.workflow_3.workflow_3_config_loader import seed_env

    seed_env()
    monitor_loop()
