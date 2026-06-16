r"""[TEMP] office downloader 실측 진단 하니스 - rcp/msr + success.

`office_rcp_msr_downloader.py` 와 `office_success_downloader.py` 를 **실제 배포된 그대로**
(production 과 동일한 load_office_integration + make_* 팩토리 경로로) 로드해, 진짜
eqp_id/recipe_id 로 한 번 호출하고 디스크에 무엇이 떨어졌는지 인벤토리한다.

목적: "align_consensus_cache / align_img_from_rcp 가 왜 비나"를 한 번에 가른다.
  - 팩토리 로드 실패?            -> office integration status 로그로 드러남
  - fetch 가 빈 결과를 반환?      -> 반환 리스트/카운트가 0
  - 받긴 받는데 다른 곳에 씀?    -> 반환 경로 vs 인벤토리 경로 불일치

파일명이 temp_ 라 pytest 수집 대상이 아니다(test_ 였다면 수집됨). git 추적은 되므로
Mac 에서 편집 -> push -> 오피스 pull 후 실행하는 워크플로우에 맞는다. office_* 모듈은
gitignore 라 오피스 PC 에서만 실제 다운로드가 동작한다(개발 PC 에선 'missing' 으로 skip).

실행:
  uv run python poc/workflow_3/monitor/temp_test_office_downloaders.py

설정(아래 상수를 직접 편집하거나 env 로 override - CLAUDE.md: argparse 미사용):
  TEST_EQP_ID      장비 ID            (없으면 ALIGN_EQP_ID 사용)
  TEST_RECIPE_ID   '<class>/<recipe>' (없으면 ALIGN_CLASS_NAME/ALIGN_RECIPE_NAME 로 조합)
"""

import os
import shutil
from pathlib import Path

from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR, ALIGN_IMAGES_DIR, WORKFLOW_3_DIR
from poc.workflow_3.align.cond_file import cond_path_for
from poc.workflow_3.align.consensus_gather import (
    count_staged_events,
    gather_success_images,
)
from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.integration_loader import load_office_integration

# ── 여기에 실제 값을 넣어 실행 (env TEST_EQP_ID / TEST_RECIPE_ID 로도 override 가능) ──
EQP_ID = ""                 # 예: "ABCDEF01"
RECIPE_ID = ""              # 예: "MyClass/MyRecipe"  ('<class>/<recipe>' 형식)

# env override (env 가 우선). recipe_id 는 통값이 없으면 class/recipe 로 조합.
EQP_ID = os.environ.get("TEST_EQP_ID", "").strip() or os.environ.get("ALIGN_EQP_ID", "").strip() or EQP_ID
RECIPE_ID = os.environ.get("TEST_RECIPE_ID", "").strip() or RECIPE_ID
if not RECIPE_ID:
    _cls = os.environ.get("ALIGN_CLASS_NAME", "").strip()
    _rcp = os.environ.get("ALIGN_RECIPE_NAME", "").strip()
    if _cls and _rcp:
        RECIPE_ID = f"{_cls}/{_rcp}"


def _inventory(root, label):
    """root 아래 파일을 즉시 하위폴더별로 집계하고 cond.txt 수/샘플 이름을 찍는다."""
    root = Path(root)
    if not root.exists():
        print(f"[WARNING] {label}: 경로 없음 -> {root}")
        return 0
    files = [p for p in root.rglob("*") if p.is_file()]
    conds = [p for p in files if p.name == "cond.txt"]
    images = [p for p in files if p.name != "cond.txt"]
    print(f"[INFO] {label}: {root}")
    print(f"       파일 총 {len(files)}개 (이미지 {len(images)} / cond.txt {len(conds)})")

    # 즉시 하위폴더(예: align_img_from_rcp, align_img_from_msr, events)별 카운트.
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sub_files = [p for p in sub.rglob("*") if p.is_file() and p.name != "cond.txt"]
        sample = ", ".join(p.name for p in sub_files[:4])
        print(f"         {sub.name}/: 이미지 {len(sub_files)}개"
              + (f"  예: {sample}" if sample else "  (비어있음)"))
    return len(images)


def _check_cond_pairs(image_paths):
    """반환된 이미지마다 짝 cond.txt 가 존재하는지 확인(align reader 규약과 동기)."""
    missing = [p for p in image_paths if not cond_path_for(Path(p)).is_file()]
    if missing:
        print(f"[WARNING] cond.txt 없는 이미지 {len(missing)}/{len(image_paths)}개 "
              f"(예: {cond_path_for(Path(missing[0]))})")
    else:
        print(f"[INFO] cond.txt 짝 OK ({len(image_paths)}개 이미지 모두 존재)")


# ─────────────────────────── PART 1: rcp/msr downloader ───────────────────────────
def probe_rcp_msr():
    """office_rcp_msr_downloader 를 로드 -> download_rcp_msr 호출 -> align_images 트리 인벤토리."""
    print("\n" + "=" * 70)
    print("[PART 1] office_rcp_msr_downloader  (-> align_images 트리)")
    print("=" * 70)

    integ = load_office_integration(
        "office_rcp_msr_downloader",
        "poc.workflow_3.monitor.office_rcp_msr_downloader",
        required_attrs=("make_rcp_msr_downloader",),
    )
    if not integ.available:
        print("[FAIL] rcp/msr downloader 로드 실패 (위 status 로그 참고). "
              "개발 PC 면 정상(office_* 부재).")
        return False

    try:
        downloader = integ.attrs["make_rcp_msr_downloader"]()
    except Exception as exc:
        print(f"[FAIL] make_rcp_msr_downloader() 예외: {type(exc).__name__}: {exc}")
        return False

    dest_dir = ALIGN_IMAGES_DIR / EQP_ID / RECIPE_ID
    print(f"[INFO] 호출: download_rcp_msr(eqp_id={EQP_ID!r}, recipe_id={RECIPE_ID!r}, "
          f"dest_dir={dest_dir})")
    try:
        n = downloader.download_rcp_msr(EQP_ID, RECIPE_ID, dest_dir=dest_dir)
    except Exception as exc:
        print(f"[FAIL] download_rcp_msr 예외: {type(exc).__name__}: {exc}")
        return False

    print(f"[INFO] 반환 이미지 수(보고값) = {n}")
    found = _inventory(dest_dir, "디스크 인벤토리(dest_dir)")
    ok = found > 0
    print(f"[{'PASS' if ok else 'FAIL'}] rcp/msr: 디스크 이미지 {found}개 "
          f"(보고값 {n})" + ("" if ok else "  <- 비었음: fetch 0건 또는 다른 경로에 씀"))
    return ok


# ─────────────────────────── PART 2: success downloader ───────────────────────────
def probe_success():
    """office_success_downloader 를 로드 -> (a) download_recent_successes 원시 호출로
    StagedEvent 리스트를 직접 검사 -> (b) gather_success_images 로 실제 캐시 적재 확인."""
    print("\n" + "=" * 70)
    print("[PART 2] office_success_downloader  (-> align_consensus_cache)")
    print("=" * 70)

    integ = load_office_integration(
        "office_success_downloader",
        "poc.workflow_3.monitor.office_success_downloader",
        required_attrs=("make_success_downloader",),
    )
    if not integ.available:
        print("[FAIL] success downloader 로드 실패 (위 status 로그 참고). "
              "개발 PC 면 정상(office_* 부재).")
        return False

    try:
        downloader = integ.attrs["make_success_downloader"]()
    except Exception as exc:
        print(f"[FAIL] make_success_downloader() 예외: {type(exc).__name__}: {exc}")
        return False

    settings = load_workflow3_settings()
    max_events = settings.gather_max_events

    # (a) 원시 호출 - StagedEvent 리스트를 직접 본다(빈 리스트 vs 이미지 0 구분).
    probe_dir = WORKFLOW_3_DIR / "debug_images" / "_temp_success_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] (a) 원시 호출: download_recent_successes(recipe_id={RECIPE_ID!r}, "
          f"max_events={max_events}, dest_dir={probe_dir})")
    try:
        staged = downloader.download_recent_successes(
            RECIPE_ID, max_events=max_events, dest_dir=probe_dir
        )
    except Exception as exc:
        print(f"[FAIL] download_recent_successes 예외: {type(exc).__name__}: {exc}")
        return False

    staged = staged or []
    all_imgs = [p for ev in staged for p in getattr(ev, "image_paths", [])]
    print(f"[INFO] 반환 StagedEvent {len(staged)}개, 이미지 총 {len(all_imgs)}장")
    for ev in staged[:5]:
        eid = getattr(ev, "event_id", "?")
        imgs = getattr(ev, "image_paths", [])
        conds = getattr(ev, "cond_paths", [])
        on_disk = sum(1 for p in imgs if Path(p).is_file())
        print(f"         event={eid}: 이미지 {len(imgs)}장(디스크 존재 {on_disk}) "
              f"cond {len(conds)}개")
    if all_imgs:
        _check_cond_pairs(all_imgs)
    else:
        print("[WARNING] StagedEvent 가 없거나 image_paths 가 비었음 -> "
              "gather 가 reason=empty 로 빠지는 직접 원인. "
              "(recipe 에 최근 S 측정이 없거나 다운로더 fetch 로직이 빈 결과)")
    _inventory(probe_dir, "디스크 인벤토리(원시 staging)")

    # (b) production 과 동일한 orchestration 으로 실제 캐시에 적재(refresh 강제).
    print(f"\n[INFO] (b) gather_success_images -> 실제 캐시 {ALIGN_CONSENSUS_CACHE_DIR}")
    result = gather_success_images(
        EQP_ID, RECIPE_ID, downloader=downloader,
        max_events=max_events, refresh_ttl_sec=0,
    )
    print(f"[INFO] GatherResult: reason={result.reason} events={result.n_events} "
          f"images={result.n_images} events_dir={result.events_dir}")
    n_events, n_images = count_staged_events(EQP_ID, RECIPE_ID)
    _inventory(result.events_dir.parent, "디스크 인벤토리(실제 캐시 recipe 루트)")

    ok = result.reason == "ok" and n_images > 0
    print(f"[{'PASS' if ok else 'FAIL'}] success: 캐시 이미지 {n_images}개 "
          f"(reason={result.reason})"
          + ("" if ok else "  <- 비었음: 위 reason 으로 원인 판정"))
    return ok


def main():
    print("[INFO] office downloader 실측 진단 시작")
    print(f"[INFO] EQP_ID={EQP_ID!r}  RECIPE_ID={RECIPE_ID!r}")
    print(f"[INFO] ALIGN_IMAGES_DIR        = {ALIGN_IMAGES_DIR}")
    print(f"[INFO] ALIGN_CONSENSUS_CACHE_DIR = {ALIGN_CONSENSUS_CACHE_DIR}")

    if not EQP_ID or not RECIPE_ID:
        print("[ERROR] EQP_ID / RECIPE_ID 가 비었습니다. 파일 상단 상수를 편집하거나 "
              "env TEST_EQP_ID / TEST_RECIPE_ID 를 설정하세요.")
        print("  예) TEST_EQP_ID=ABCDEF01 TEST_RECIPE_ID=MyClass/MyRecipe \\")
        print("        uv run python poc/workflow_3/monitor/temp_test_office_downloaders.py")
        return 2
    if "/" not in RECIPE_ID:
        print(f"[WARNING] RECIPE_ID={RECIPE_ID!r} 에 '/' 가 없습니다. "
              "'<class>/<recipe>' 형식이어야 align_images 트리와 일치합니다.")

    r1 = probe_rcp_msr()
    r2 = probe_success()

    print("\n" + "=" * 70)
    print(f"[INFO] 결과 요약: rcp/msr={'PASS' if r1 else 'FAIL'}  "
          f"success={'PASS' if r2 else 'FAIL'}")
    print("=" * 70)
    return 0 if (r1 and r2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
