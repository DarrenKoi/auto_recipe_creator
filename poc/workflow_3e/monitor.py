"""workflow_3e 통합 슈퍼바이저 — align fail + 측정 실패 abort 를 한 루프/한 프로세스로.

단일 RCS 커서를 공유하므로 두 잡은 직렬(blocking)로 처리된다. MES 는 한 번만 polling 하고
두 필터로 나눠 본다:
  * align fail(ALID=9006)  -> workflow_3 의 process_fail_rows (그대로 재사용)
  * 측정 실패 임계 알람      -> workflow_3e 의 process_abort_rows (신규)

align 경로의 폴링/edge-trigger/idle 로직은 workflow_3 의 헬퍼를 재사용하고, abort 분기만
더한다. abort 잡은 'abort can queue' 전제라 별도 lock 없이 한 프로세스 직렬로 충분하다.

    uv run python poc/workflow_3e/monitor.py

개발 PC dry-run (RCS 없이 두 잡 경로 확인):
    SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
      MEAS_FAIL_ALID=<alid> uv run python poc/workflow_3e/monitor.py
"""

import time
from datetime import datetime

from poc.workflow_3.monitor.alarm_source import load_alarm_source
from poc.workflow_3.monitor.align_fail_monitor import (
    CYCLE_MANIFEST_PATH,
    _alarm_rows_empty,
    _set_keep_awake,
    filter_rows_within_window,
    process_fail_rows,
)
from poc.workflow_3.monitor.notify import ALARM_LOG_PATH
from poc.workflow_3e.abort_button import (
    is_click_armed,
    is_rehearsal_target,
    load_abort_target,
)
from poc.workflow_3e.config import Workflow3eSettings, load_workflow3e_settings
from poc.workflow_3e.dispatch import ABORT_MANIFEST_PATH, process_abort_rows
from poc.workflow_3e.meas_alarm_source import MEAS_PROVIDER_AVAILABLE, measurement_fail_rows


def monitor_loop(settings: Workflow3eSettings | None = None) -> None:
    """통합 감지 루프 — poll 주기마다 align fail + 측정 실패 abort 를 함께 처리한다."""
    settings = settings or load_workflow3e_settings()
    source = load_alarm_source(settings.alarm_source)

    if settings.keep_awake:
        _set_keep_awake(True)

    active_tools: set[str] = set()            # align fail edge-trigger 상태.
    occupied_cooldown: dict = {}              # align 점유(select) 재시도 유예.
    aborted_tools: set[str] = set()           # 측정 실패 abort edge-trigger 상태.
    abort_cooldown: dict = {}                 # abort 점유 재시도 유예.
    idle_logged = False

    print(
        f"[INFO] 통합 모니터링 시작 (소스={source.kind}, 주기={settings.poll_interval_sec}s, "
        f"윈도우={settings.detection_window_sec}s, "
        f"보정={'on' if settings.correction_enabled else 'off'}"
        f"{'(dry-run)' if settings.correction_enabled and settings.correction_dry_run else ''})"
    )
    abort_target = load_abort_target()
    armed = is_click_armed(
        enabled=settings.meas_fail_abort_enabled,
        dry_run=settings.abort_action_dry_run,
        target=abort_target,
    )
    meas_src = "office provider" if MEAS_PROVIDER_AVAILABLE else f"ALID 필터({settings.meas_fail_alid or '미설정'})"
    if is_rehearsal_target(abort_target):
        mode = f" (REHEARSAL:{abort_target} - 클릭 없음)"
    elif not armed:
        mode = " (notify-only, dry-run)"
    else:
        mode = " [ARMED]"
    print(
        f"[INFO] 측정 실패 abort 잡: {'on' if settings.meas_fail_abort_enabled else 'off'}"
        f"{mode if settings.meas_fail_abort_enabled else ''}, 대상={abort_target}, 소스={meas_src}"
    )
    print(f"[INFO] 알람 로그: {ALARM_LOG_PATH}")
    print(f"[INFO] align manifest: {CYCLE_MANIFEST_PATH}")
    print(f"[INFO] abort manifest: {ABORT_MANIFEST_PATH}")
    if settings.meas_fail_abort_enabled and not MEAS_PROVIDER_AVAILABLE and not settings.meas_fail_alid:
        print(
            "[WARNING] 측정 실패 provider 없음 + MEAS_FAIL_ALID 미설정 - 측정 실패 abort 가 "
            "검출되지 않음. 오피스에서 office_meas_many_fails.py 구현 또는 MEAS_FAIL_ALID 설정 필요."
        )

    while True:
        try:
            poll_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO] {poll_time} - 알람 조회 (최근 {settings.detection_window_sec}s 윈도우)")
            alarms = source.poll()

            # --- align fail 잡 (workflow_3 재사용) ---
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
                count = process_fail_rows(fails, active_tools, settings, occupied_cooldown)
                if count == 0:
                    print(
                        f"[INFO] {datetime.now().strftime('%H:%M:%S')} - "
                        f"신규 없음 (활성 {len(active_tools)}대 유지)"
                    )

            # --- 측정 실패 abort 잡 (workflow_3e 신규: 전용 provider 우선, 없으면 ALID 필터) ---
            if settings.meas_fail_abort_enabled and (MEAS_PROVIDER_AVAILABLE or alarms is not None):
                meas = measurement_fail_rows(alarms, settings.meas_fail_alid)
                meas = filter_rows_within_window(meas, settings.detection_window_sec)
                if not _alarm_rows_empty(meas):
                    process_abort_rows(meas, aborted_tools, settings, abort_cooldown)
                elif aborted_tools:
                    for eqp_id in sorted(aborted_tools):
                        print(f"[INFO] 측정 실패 알람 해제: EQP_ID={eqp_id}")
                    aborted_tools.clear()

        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break
        except Exception as exc:
            print(f"[ERROR] 통합 감지 루프 예외: {exc}")

        try:
            time.sleep(settings.poll_interval_sec)
        except KeyboardInterrupt:
            print("\n[INFO] 감지 중단 (Ctrl+C)")
            break

    if settings.keep_awake:
        _set_keep_awake(False)
    print("[INFO] 통합 감지 종료")


if __name__ == "__main__":
    # 실편집 workflow_3_config.py 의 토글을 env 로 브리지(있으면). load_workflow3e_settings
    # 가 env 를 읽기 전에 호출해야 적용된다. 실제 OS env 가 우선(setdefault).
    # 빠뜨리면 파일에 적어둔 안전 토글(SAFE_MODE 등)이 이 진입점에서만 조용히 무시된다.
    from poc.workflow_3.workflow_3_config_loader import seed_env

    seed_env()
    monitor_loop()
