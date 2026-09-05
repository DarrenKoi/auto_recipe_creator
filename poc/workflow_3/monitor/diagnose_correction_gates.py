"""보정이 실행되지 않는(마우스가 안 움직이는) 원인을 게이트 순서대로 짚는다.

RCS/VLM/실장비 없이 도는 **읽기 전용** 진단이다. 클릭도 접속도 하지 않는다.
`verify_consensus_path.py` 와 같은 성격 - 오피스에서 한 줄 실행하고 콘솔을 그대로
공유하면 원인 구간이 좁혀진다.

세 가지를 본다.

  1. **설정 게이트** - 마우스가 움직이려면 통과해야 하는 값들. 진입점
     (`align_fail_monitor.__main__`) 과 **같은 순서**로 env 를 세팅해서 읽는다:
     `_apply_live_mode_defaults()` -> `seed_env()`. 이 순서가 틀리면 dry_run 판정이
     통째로 달라진다(오피스 `workflow_3_config.py` 사본의 CORRECTION_DRY_RUN 이
     진입점 기본값을 덮는지 여부가 여기서 갈린다).
  2. **직전 사이클의 step 저널** - 어느 step 에서 멈췄는지가 사실로 적혀 있다.
     추측할 필요가 없는 유일한 증거다.
  3. **데이터/캘리브레이션** - rcp align 이미지와 SEM panel landmark 유무.

실행:
    uv run python poc/workflow_3/monitor/diagnose_correction_gates.py
"""

import json
import os
from pathlib import Path


_MARKS = {"ok": "OK", "block": "XX", "warn": "!!"}

# 보정 step 에 도달하기 전에 사이클을 세우는 failure_class -> 사람이 읽는 원인.
FAILURE_HINTS = {
    "rcs_unavailable": "RCS 메인 창을 못 찾음. 보정 이전 문제.",
    "rcs_recovery_error": "RCS 재실행/재로그인 실패.",
    "rcs_recovery_no_window": "재로그인했지만 창이 안 뜸(좀비 프로세스 가능).",
    "rcs_occupied": "다른 엔지니어 점유. 보정 안 함(관전만).",
    "wrong_tool_opened": "엉뚱한 tool 이 열림. row 클릭이 빗나감.",
    "panel_locate_error": "SEM panel 확보 중 예외.",
    "panel_not_found": "live SEM box 미검출 + landmark 미캘리브레이션. "
                       "보정 step 에 도달 못 함 -> 마우스 안 움직임.",
    "pm_mode_unknown": "PM 박스에서 OM/SEM 을 못 읽음. require_pm_mode=1 이라 보정 보류 "
                       "-> 마우스 안 움직임.",
    "correction_error": "보정 중 예외 발생.",
}


def _line(name, value, verdict) -> None:
    print(f"  [{_MARKS[verdict]}] {name:38} = {value}")


def report_settings_gates(settings) -> list[str]:
    """마우스가 움직이기 위한 설정 게이트를 찍고 차단 요인 목록을 돌려준다."""
    blockers: list[str] = []

    print("\n-- 1. 물리 출력 (막히면 호출은 되지만 커서가 안 움직인다) --")
    safe_mode = os.environ.get("SAFE_MODE", "(unset)")
    _line("SAFE_MODE (env)", safe_mode, "ok" if safe_mode == "0" else "block")
    if safe_mode != "0":
        blockers.append("SAFE_MODE!=0 - 모든 물리 마우스/키보드 출력이 차단된다")
    _line("action_enabled", settings.action_enabled, "ok" if settings.action_enabled else "block")
    if not settings.action_enabled:
        blockers.append("action_enabled=False")

    print("\n-- 2. 보정 actuation (막히면 move_to_point 호출 자체를 건너뛴다) --")
    _line("correction_dry_run", settings.correction_dry_run,
          "block" if settings.correction_dry_run else "ok")
    if settings.correction_dry_run:
        blockers.append(
            "correction_dry_run=True - correction.py 의 reposition 더블클릭을 건너뛴다"
        )
    _line("correction_enabled", settings.correction_enabled,
          "ok" if settings.correction_enabled else "block")
    if not settings.correction_enabled:
        blockers.append("correction_enabled=False - run_correction step 이 skipped")
    _line("ok_click_enabled", settings.ok_click_enabled,
          "ok" if settings.ok_click_enabled else "warn")
    if not settings.ok_click_enabled:
        print("        (반자동: reposition 더블클릭까지만 하고 OK 는 엔지니어가 누른다.")
        print("         이 값이 0이어도 reposition 커서는 움직여야 한다.)")

    print("\n-- 3. 보정 step 에 도달하기 위한 선행 게이트 --")
    _line("require_pm_mode", settings.require_pm_mode,
          "warn" if settings.require_pm_mode else "ok")
    if settings.require_pm_mode:
        print("        (PM 박스에서 OM/SEM 을 못 읽으면 pm_mode_unknown 으로 보정 보류.")
        print("         롤백: ALIGN_FAIL_REQUIRE_PM_MODE=0)")
    _line("sem_box_detect_enabled", settings.sem_box_detect_enabled,
          "ok" if settings.sem_box_detect_enabled else "block")
    if not settings.sem_box_detect_enabled:
        blockers.append("sem_box_detect_enabled=False - landmark 가 비면 panel 확보 불가")
    _line("correct_when_occupied", settings.correct_when_occupied,
          "ok" if settings.correct_when_occupied else "warn")
    _line("consensus_enabled", settings.consensus_enabled,
          "ok" if settings.consensus_enabled else "warn")
    return blockers


def report_data_gates(align_images_dir: Path, landmarks_dir: Path) -> list[str]:
    """rcp align 이미지와 landmark 캘리브레이션 유무를 찍는다."""
    blockers: list[str] = []

    print("\n-- 4. 데이터 (rcp 가 없으면 no_assets 로 끝난다) --")
    exists = align_images_dir.is_dir()
    _line("ALIGN_IMAGES_DIR", str(align_images_dir), "ok" if exists else "block")
    if not exists:
        blockers.append(f"ALIGN_IMAGES_DIR 없음: {align_images_dir}")
    else:
        eqps = sorted(p.name for p in align_images_dir.iterdir() if p.is_dir())
        _line("  eqp 폴더", f"{len(eqps)}개 {eqps[:5]}", "ok" if eqps else "block")
        if not eqps:
            blockers.append("align_images 트리가 비어 있다 - MES 출력 경로/ALIGN_IMAGES_DIR 확인")
        else:
            rcp = list(align_images_dir.rglob("align_img_from_rcp/*"))
            _line("  align_img_from_rcp 파일", len(rcp), "ok" if rcp else "block")
            if not rcp:
                blockers.append("align_img_from_rcp 가 비어 있다 -> 보정이 no_assets 로 끝난다")

    print("\n-- 5. panel 폴백 (SEM box 검출 실패 시 유일한 대안) --")
    marks = [p for p in landmarks_dir.glob("*") if p.name != ".gitkeep"] \
        if landmarks_dir.is_dir() else []
    _line("sem_panel_landmarks 템플릿", len(marks), "ok" if marks else "warn")
    if not marks:
        print("        (미캘리브레이션. live SEM box 검출이 실패하면 panel_not_found 로")
        print("         보정 step 에 도달하지 못한다 - 마우스가 전혀 안 움직이는 전형적 원인.)")
    return blockers


def report_last_run(runs_dir: Path, limit: int = 3) -> None:
    """직전 사이클들의 step 저널을 찍는다 - 어디서 멈췄는지의 유일한 사실 기록."""
    print("\n-- 6. 직전 사이클 step 저널 (여기에 답이 적혀 있다) --")
    if not runs_dir.is_dir():
        print(f"  [!!] 저널 폴더 없음: {runs_dir}")
        return
    runs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and "align_fail_cycle" in p.name),
        key=lambda p: p.name, reverse=True,
    )[:limit]
    if not runs:
        print("  [!!] align_fail_cycle 실행 기록이 없다")
        return
    for run in runs:
        print(f"\n  * {run.name}")
        steps = sorted(run.glob("step_*.json"))
        if not steps:
            print("      (step 기록 없음)")
            continue
        for path in steps:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                print(f"      {path.name}: 읽기 실패")
                continue
            klass = data.get("failure_class") or ""
            print(f"      {data.get('step_id', '?'):22} {data.get('status', '?'):9}"
                  f" {klass or '-'}")
            if klass and klass in FAILURE_HINTS:
                print(f"        -> {FAILURE_HINTS[klass]}")
            message = (data.get("error_message") or "").strip()
            if message:
                print(f"        msg: {message[:100]}")


def main() -> int:
    # 진입점(align_fail_monitor.__main__)과 같은 순서. 이 순서가 계약이다.
    os.environ.setdefault("SAFE_MODE", "0")
    os.environ.setdefault("ALIGN_FAIL_CORRECTION_DRY_RUN", "0")
    from poc.workflow_3.workflow_3_config_loader import seed_env

    seed_env()

    from poc.workflow_3 import ALIGN_IMAGES_DIR
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.sem_monitor.controller import DEFAULT_LANDMARKS_DIR

    settings = load_workflow3_settings()

    print("=" * 72)
    print("[진단] 보정 마우스가 움직이려면 아래가 전부 통과해야 한다")
    print("=" * 72)

    blockers = report_settings_gates(settings)
    blockers += report_data_gates(ALIGN_IMAGES_DIR, DEFAULT_LANDMARKS_DIR)
    report_last_run(Path(__file__).resolve().parent.parent / "logs" / "workflow_runs")

    print("\n" + "=" * 72)
    if blockers:
        print(f"[진단] 설정/데이터 차단 요인 {len(blockers)}건:")
        for item in blockers:
            print(f"   XX {item}")
    else:
        print("[진단] 설정/데이터 게이트는 통과. 원인은 런타임 쪽이다 -")
        print("       위 6번 step 저널의 failure_class 를 보라.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
