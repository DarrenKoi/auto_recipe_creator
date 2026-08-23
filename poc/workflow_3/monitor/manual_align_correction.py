"""수동 Align 보정 트리거 — 알람 대기 없이 알려진 tool/recipe 로 사이클을 돌린다.

`monitor/align_fail_monitor.py` 는 MES 알람(ALID=9006) 이 떨어질 때까지 기다린 뒤
사이클을 시작한다. 시험/디버깅 중에는 "알람이 의도적으로 발생한다"는 보장이 없어
대기 시간이 그대로 비용이 된다. 이 모듈은 EQP_ID/RECIPE_ID 를 환경변수로 받아
알람 폴링/edge-trigger 만 우회하고, 같은 `run_alarm_cycle` 을 한 번 돈다.

동작:
  1. RCS preflight (로그인 + List 탭) - 사이클의 ensure_rcs_ready 와 같은 복구 경로.
  2. 합성 info dict (ALID=9006, alarm_name="Manual Trigger") 로 알람 로그/manifest
     형식을 유지한다. 알람 시스템이 보낸 row 가 없으니 alarm_time/UTC9 는 wall-clock.
  3. rcp/msr 1차 입력 + consensus success gather (모니터와 동일한 pre-cycle 작업).
  4. `run_alarm_cycle(eqp_id, recipe_id, settings, tag=...)` - 사이클 본체.
  5. cycle manifest 한 줄 기록.

safety:
  * `SAFE_MODE=0` + `ALIGN_FAIL_CORRECTION_DRY_RUN=0` 기본값을 못박는다 (실운전).
    점검만 하려면 `SAFE_MODE=1` 으로 실행하면 클릭이 모두 막힌다.
  * 긴급 해제 단축키는 사이클 진입 전에 띄운다.

사용법 (venv 활성화 후, 저장소 루트에서):
  1) 이 파일 상단의 EQP_ID / RECIPE_ID 상수를 채운다.
  2) python poc/workflow_3/monitor/manual_align_correction.py

  이 프로젝트는 CLI 인자를 쓰지 않으므로 "인자"는 파일 상단 상수이거나 동명의
  환경변수다. 1회성으로 다른 장비를 돌릴 때만 env 를 붙이면 되고, env 가 상수를
  이긴다(어느 쪽이 쓰였는지는 시작 시 콘솔에 찍힌다).

    Windows cmd  : set MANUAL_CORRECTION_EQP_ID=MCD916
    PowerShell   : $env:MANUAL_CORRECTION_EQP_ID = "MCD916"
    macOS/Linux  : MANUAL_CORRECTION_EQP_ID=MCD916 python poc/workflow_3/monitor/manual_align_correction.py

인자 (상수 이름 = env 이름 MANUAL_CORRECTION_<상수>):
  EQP_ID           (필수)
  RECIPE_ID        (필수)
  CLASS_NAME       (선택 - asset 라우팅/로그용)
  TAG              (선택 - 미지정 시 wall-clock)
  MANUAL_CORRECTION_SKIP_PREFLIGHT=1  (점검 - List 탭이 이미 열려있다면. env 전용)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# venv 에서 `python poc/workflow_3/monitor/manual_align_correction.py` 로 직접 실행하면
# sys.path[0] 는 이 파일이 있는 monitor 폴더라 `poc` 패키지가 보이지 않는다. 저장소
# 루트를 먼저 얹어, 프로젝트를 pip 로 설치하지 않은 순수 venv 에서도 절대 import 가
# 풀리게 한다. editable 설치(uv sync)나 `python -m poc...` 실행에서는 중복 경로일
# 뿐이라 무해하다. 같은 규약: monitor/test_cycle_report.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3 import ALIGN_FAIL_ALID, LOG_DIR  # noqa: E402
from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.cycle import (
    CycleResult,
    _list_process_windows,
    _scan_rcs_processes,
    _terminate_process,
    run_alarm_cycle,
)
from poc.workflow_3.monitor.notify import (
    notify_align_fail_popup,
    send_detection_notify_async,
)
from poc.workflow_3.monitor.rcp_msr_gather import gather_rcp_msr
from poc.workflow_3.monitor.rcs_preflight import ensure_rcs_session_ready
from poc.workflow_3.monitor.rcs_recovery import recover_rcs_session
from poc.workflow_3.monitor.success_gather import gather_success_async
from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
from poc.workflow_3.rcs.open_rcs import launch_rcs
from poc.workflow_3.rcs.view_list_tab_rcs import click_list_tab_in_main_window
from poc.workflow_3.rcs.workflow_login import run_login_workflow
from poc.workflow_3.util.abort_switch import (
    abort_reason,
    is_aborted,
    start_abort_hotkey,
)
from poc.workflow_3.util import make_timestamp_tag
from poc.workflow_3.logger import log_work2_event

from poc.workflow_3.monitor.align_fail_monitor import (
    append_alarm_record,
    append_cycle_manifest,
)

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다 (셸 env 를 붙이지 않고 실행하려는 경우).
#
# 이 프로젝트는 CLI 인자(argparse/flag)를 쓰지 않는다. 그래서 "인자"는 아래 상수이거나
# 같은 이름의 환경변수 둘 중 하나다. 우선순위는 workflow_3_config.py 와 동일하게
#   실제 셸 env  >  아래 상수
# 이며, 어느 쪽이 쓰였는지 시작 시 콘솔에 찍힌다(둘이 어긋난 채 도는 것을 막는다).
# ===========================================================================

EQP_ID = "MCD513"
RECIPE_ID = "RJ1BXXX/RJ1B_ISOLINERPOLY_R1"   # 반드시 '<class>/<recipe>' 형태
CLASS_NAME = ""      # 선택. 알람 로그/팝업 표시용일 뿐이라 비워도 라우팅에 영향 없다
TAG = ""             # 선택. 산출물 폴더 tag. 비우면 wall-clock 으로 생성

LOG_COMPONENT = "manual_align_correction"

# 수동 트리거를 알람 row 와 구분하기 위한 sentinel. manifest/alarm log 에 그대로 남는다.
MANUAL_ALARM_NAME = "Manual Trigger"
MANUAL_OPERATION_DESC = "manual_trigger"

EXIT_OK = 0
EXIT_BAD_ARGS = 1
EXIT_PREFLIGHT_FAILED = 2


def _apply_live_mode_defaults() -> None:
    """프로덕션 모니터와 동일한 실운전 기본값을 못박는다.

    `align_fail_monitor._apply_live_mode_defaults` 와 같은 규칙이다. 이 진입점의
    존재 이유는 "tool 에서 바로 보정을 본다"이므로 dry-run 으로 뜨면 목적 자체가
    무너진다. 점검만 하려면 셸 env 로 `SAFE_MODE=1` 을 덮어쓰면 된다.
    """
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


def _resolve_arg(env_name: str, inline_value: str) -> tuple[str, str]:
    """하나의 인자를 `셸 env > 파일 상수` 순으로 고르고, 출처까지 돌려준다.

    출처를 같이 돌려주는 이유: 파일 상수를 고쳤는데 셸에 예전 env 가 남아 있으면
    조용히 다른 장비를 상대로 실클릭이 나간다. 어느 쪽이 이겼는지 콘솔에 찍어
    그 사고를 눈에 보이게 만든다. 우선순위는 workflow_3_config.py 와 같다.
    """
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value, "env"
    return (inline_value or "").strip(), "file"


def _load_trigger_args() -> tuple[str, str, str, str]:
    """실행 인자(파일 상수 또는 동명의 env)를 읽고 검증한다.

    EQP_ID/RECIPE_ID 는 필수. 없으면 사이클을 돌릴 식별자가 없으니 즉시 종료한다.
    """
    eqp_id, eqp_src = _resolve_arg("MANUAL_CORRECTION_EQP_ID", EQP_ID)
    recipe_id, recipe_src = _resolve_arg("MANUAL_CORRECTION_RECIPE_ID", RECIPE_ID)
    class_name, _ = _resolve_arg("MANUAL_CORRECTION_CLASS_NAME", CLASS_NAME)
    tag, _ = _resolve_arg("MANUAL_CORRECTION_TAG", TAG)

    if eqp_id or recipe_id:
        print(f"[INFO] 인자 출처: EQP_ID={eqp_src}, RECIPE_ID={recipe_src} "
              "(env 가 파일 상수를 이깁니다)")

    missing = [name for name, value in (
        ("EQP_ID", eqp_id),
        ("RECIPE_ID", recipe_id),
    ) if not value]
    if missing:
        print(f"[ERROR] 필수 인자 누락: {', '.join(missing)}")
        print("설정 방법 두 가지 - 아무거나 하나:")
        print("  [1] 이 파일 상단 상수를 고친다 (권장, 셸 무관)")
        print('        EQP_ID = "MCD916"')
        print('        RECIPE_ID = "MyRecipe"')
        print("      그리고: python poc/workflow_3/monitor/manual_align_correction.py")
        print("  [2] 셸 env 로 1회성 지정 (파일 상수를 덮어씀)")
        print("      Windows cmd:")
        print("        set MANUAL_CORRECTION_EQP_ID=MCD916")
        print("        set MANUAL_CORRECTION_RECIPE_ID=MyRecipe")
        print("      PowerShell:")
        print('        $env:MANUAL_CORRECTION_EQP_ID = "MCD916"')
        print('        $env:MANUAL_CORRECTION_RECIPE_ID = "MyRecipe"')
        raise SystemExit(EXIT_BAD_ARGS)

    return eqp_id, recipe_id, class_name, tag


def _build_synthetic_info(eqp_id: str, recipe_id: str, class_name: str) -> dict:
    """`_collapse_rows_by_tool` 출력 스키마와 같은 info dict 를 만든다.

    `append_alarm_record`/`append_cycle_manifest` 가 요구하는 키를 모두 채운다.
    실제 알람 row 가 없으니 시간 필드는 모두 wall-clock 으로 채우고, 알람 시스템이
    제공하던 `operation_desc`/`lot_type_cd` 는 비워둔다(필드가 없는 게 정상이므로).
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "eqp_id": eqp_id,
        "alarm_time": now_str,
        "utc9": now_str,
        "alarm_name": MANUAL_ALARM_NAME,
        "alid": ALIGN_FAIL_ALID,
        "recipe_id": recipe_id,
        "operation_desc": MANUAL_OPERATION_DESC,
        "lot_type_cd": class_name,  # 표시용(알람 로그/팝업). class 라우팅은 recipe_id 가 담당
    }


def _run_preflight(settings):
    """RCS 가 로그인 + List 탭까지 떠 있는지 보장한다.

    `align_fail_monitor._run_rcs_preflight` 와 같은 협력자 조립을 그대로 쓰되
    공개 API `ensure_rcs_session_ready` 를 직접 호출한다 (private 헬퍼 결합 회피).
    실패해도 None 을 돌려주고, 호출부가 사이클 안의 ensure_rcs_ready 가 복구를
    다시 시도한다는 사실을 로그에 남긴다.
    """
    if not settings.rcs_preflight_enabled:
        print("[INFO] RCS 기동 준비 생략(ALIGN_FAIL_RCS_PREFLIGHT=0).")
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
        outcome = ensure_rcs_session_ready(
            settings,
            find_window_fn=wait_for_rcs_main_window,
            recover_fn=_recover,
            open_list_fn=_open_list,
        )
    except Exception as exc:
        print(f"[WARNING] RCS 기동 준비 예외(사이클 안 복구에 의존): {exc}")
        return None

    status = getattr(outcome, "status", None)
    if status and status != "ready":
        print(
            f"[WARNING] RCS 기동 준비 미완료(status={status}) - "
            "사이클의 ensure_rcs_ready 가 다시 시도합니다."
        )
    return outcome


def _print_summary(cycle: CycleResult) -> None:
    """사이클 결과를 한 블록으로 요약한다.

    로그만 보고도 이 실행이 manual 트리거였는지, 보정이 반영되었는지, 어디로
    산출물을 남겼는지가 판별되어야 한다. 모니터가 쓰는 manifest 컬럼과 같은
    순서/키를 쓴다.
    """
    print("=" * 70)
    print(f"[INFO] manual cycle 완료: eqp_id={cycle.eqp_id}, recipe_id={cycle.recipe_id}")
    print(f"  run_status    = {cycle.run_status}")
    print(f"  failed_step   = {cycle.failed_step or '-'}")
    print(f"  failure_class = {cycle.failure_class or '-'}")
    print(f"  outcome       = {cycle.outcome_status or '-'}")
    print(f"  key_decision  = {cycle.key_decision or '-'}")
    print(f"  best_xy       = {cycle.best_xy or '-'}")
    print(f"  frame_count   = {cycle.frame_count}")
    print(f"  recording_dir = {cycle.recording_dir or '-'}")
    print(f"  run_dir       = {cycle.run_dir or '-'}")
    print(f"  log_dir       = {LOG_DIR}")
    print("=" * 70)


def main() -> int:
    """수동 트리거 1회 실행. 종료 코드: 0=사이클 완주, 1=인자 오류, 2=preflight 실패."""
    _apply_live_mode_defaults()
    from poc.workflow_3.workflow_3_config_loader import seed_env

    seed_env()

    eqp_id, recipe_id, _class_name, env_tag = _load_trigger_args()
    settings = load_workflow3_settings()
    tag = env_tag or make_timestamp_tag()

    print(
        f"[INFO] 수동 Align 보정 트리거: EQP_ID={eqp_id}, RECIPE_ID={recipe_id}, "
        f"tag={tag}, 사이클={'on' if settings.cycle_enabled else 'off'}, "
        f"보정={'on' if settings.correction_enabled else 'off'}"
        f"{'(dry-run)' if settings.correction_enabled and settings.correction_dry_run else ''}"
    )

    if not settings.cycle_enabled:
        print(
            "[WARNING] ALIGN_FAIL_RECORD_CYCLE=0 입니다 - 사이클이 비활성. "
            "수동 트리거가 의미 있으려면 활성화 후 다시 실행하세요."
        )
        return EXIT_PREFLIGHT_FAILED

    # 마우스/키보드 조작 진입 전에 abort 스위치를 먼저 띄운다 - 모니터의 순서를 그대로 따른다.
    start_abort_hotkey(settings.abort_hotkey)
    _run_preflight(settings)

    if is_aborted():
        print(f"[WARNING] 긴급 해제됨({abort_reason()}) - 사이클 진입 전 종료.")
        return EXIT_PREFLIGHT_FAILED

    info = _build_synthetic_info(eqp_id, recipe_id, _class_name)
    log_work2_event(
        component=LOG_COMPONENT, message="manual_trigger_start",
        level="info", eqp_id=eqp_id, recipe_id=recipe_id, tag=tag,
    )

    # 알람 로그 형식 유지 (manifest 와 짝을 이루도록 같은 row 를 남긴다).
    append_alarm_record(
        eqp_id, info["alarm_time"], info["alarm_name"], info["alid"],
        recipe_id=recipe_id,
        operation_desc=info["operation_desc"],
        lot_type_cd=info["lot_type_cd"],
    )
    if settings.popup_enabled:
        notify_align_fail_popup(
            eqp_id, info["alarm_time"], info["alarm_name"],
            recipe_id=recipe_id,
            operation_desc=info["operation_desc"],
            lot_type_cd=info["lot_type_cd"],
            timeout_sec=settings.popup_timeout_sec,
        )
    # 사전 큐브 고지 - 모니터의 detection_notify 와 같은 게이트.
    send_detection_notify_async(
        eqp_id, recipe_id,
        enabled=settings.rich_notify_enabled and settings.detection_notify_enabled,
    )

    # pre-cycle 데이터 - 모니터와 같은 비동기/동기 게이트를 gather_* 내부에서 판정한다.
    gather_success_async(eqp_id, recipe_id, settings)
    gather_rcp_msr(eqp_id, recipe_id, settings, timeout_sec=settings.rcp_gather_timeout_sec)

    # 본체. 예외는 run_alarm_cycle 안에서 잡혀 CycleResult.failed_step 으로 남는다.
    cycle = run_alarm_cycle(eqp_id, recipe_id, settings, tag=tag)
    append_cycle_manifest(info, cycle)

    if is_aborted():
        print(f"[WARNING] 사이클 진행 중 긴급 해제됨({abort_reason()}).")

    log_work2_event(
        component=LOG_COMPONENT, message="manual_trigger_done",
        level="info", eqp_id=eqp_id, recipe_id=recipe_id,
        run_status=cycle.run_status, outcome=cycle.outcome_status or "",
        failed_step=cycle.failed_step or "", failure_class=cycle.failure_class or "",
    )
    _print_summary(cycle)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
