"""success gather 1회 검증 — office 에서 실제 DB stage 가 계약대로 되는지 확인.

office PC 전용 검증 스크립트(success downloader 가 없으면 즉시 종료).

**부작용 있음**: 실제로 S 이미지를 내려받아 consensus 캐시(events/)에 stage 한다.
읽기 전용 점검이 필요하면 align/diagnostics/verify_rcp_assets.py 를 쓸 것.

확인 항목: reason=ok / event 디렉토리별 S*.jpeg + 숨김폴더 cond(.<이미지명>/cond.txt) 짝 /
cond 파싱(crosshair 좌표, modality 추론 — msr cond 는 Scope 가 없어 msr_modality() 키로
가른다) / staging 정리. 통과하면 events/ 는 그대로 캐시로 남는다(추가 정리 불필요).

사용법 (venv 활성화 후, 저장소 루트에서):
  1) 아래 EQP_ID / CLASS_NAME / RECIPE_NAME 상수를 채운다.
  2) python poc/workflow_3/monitor/verify_success_gather.py

  CLI 인자는 쓰지 않는다. 1회성으로 다른 대상을 볼 때만 동명의 env
  (ALIGN_EQP_ID / ALIGN_CLASS_NAME / ALIGN_RECIPE_NAME)를 붙이면 되고, env 가 상수를
  이긴다 - assets 해석(resolve_assets_auto)과 같은 env 규약이다.
"""

import os
import sys
from pathlib import Path

# venv 에서 파일 경로로 직접 실행할 때 저장소 루트를 sys.path 에 얹는다.
# (monitor/manual_align_correction.py 와 같은 규약)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor.success_gather import (  # noqa: E402
    DOWNLOADER_AVAILABLE,
    _DOWNLOADER,
)
from poc.workflow_3.align.cond_file import load_cond, msr_modality  # noqa: E402
from poc.workflow_3.align.consensus_gather import gather_success_images  # noqa: E402

# ===========================================================================
# 검증 대상 - 여기만 고쳐서 쓴다 (동명의 ALIGN_* env 가 이 상수를 이긴다).
# CLASS_NAME 과 RECIPE_NAME 은 따로 준다 - 스크립트가 '<class>/<recipe>' 로 합친다.
# ===========================================================================

EQP_ID = "MCD513"
CLASS_NAME = "RJ1BXXX"
RECIPE_NAME = "RJ1B_ISOLINERPOLY_R1"


def main():
    if not DOWNLOADER_AVAILABLE:
        print("[ERROR] success downloader 없음 - office PC 에서 실행하세요.")
        return 1

    eqp_id = (os.environ.get("ALIGN_EQP_ID", "").strip() or EQP_ID).strip()
    class_name = (os.environ.get("ALIGN_CLASS_NAME", "").strip() or CLASS_NAME).strip()
    recipe_name = (os.environ.get("ALIGN_RECIPE_NAME", "").strip() or RECIPE_NAME).strip()
    if not eqp_id or not class_name or not recipe_name:
        print("[ERROR] 검증 대상이 비었습니다. 이 파일 상단의 EQP_ID / CLASS_NAME / "
              "RECIPE_NAME 상수를 채우거나, 동명의 ALIGN_* env 를 지정하세요.")
        print("  예: EQP_ID = \"MCD513\" / CLASS_NAME = \"RJ1BXXX\" / "
              "RECIPE_NAME = \"RJ1B_ISOLINERPOLY_R1\"")
        return 1
    recipe_id = f"{class_name}/{recipe_name}"
    print(f"[INFO] gather 1회 실행: EQP_ID={eqp_id} recipe={recipe_id}")

    res = gather_success_images(eqp_id, recipe_id, downloader=_DOWNLOADER)
    print(f"[INFO] reason={res.reason} events={res.n_events} images={res.n_images}")
    print(f"[INFO] events_dir={res.events_dir}")
    if res.reason != "ok":
        print("[ERROR] stage 실패 - reason 으로 downloader/DB 를 점검하세요. "
              "(empty=반환 리스트 비었음, error:=다운로드 예외, error:swap:=반환값/swap 문제)")
        return 1

    problems = 0
    for ev_dir in sorted(p for p in res.events_dir.iterdir() if p.is_dir()):
        images = sorted(ev_dir.glob("S*.jpeg")) + sorted(ev_dir.glob("S*.jpg"))
        pair_ok = len(images) > 0
        if not pair_ok:
            problems += 1
        print(f"[{'INFO' if pair_ok else 'ERROR'}] {ev_dir.name}: images={len(images)}")
        for img in images:
            cond = load_cond(img)  # .<이미지명>/cond.txt 를 해석.
            if cond is None:
                problems += 1
                print(f"[ERROR]   {img.name}: cond 없음 (.{img.name}/cond.txt 미존재)")
                continue
            modality = cond.scope or msr_modality(cond)
            cond_ok = cond.crosshair_xy is not None and modality is not None
            if not cond_ok:
                problems += 1
            print(f"[{'INFO' if cond_ok else 'ERROR'}]   {img.name}: "
                  f"crosshair={cond.crosshair_xy} modality={modality}")

    staging = res.events_dir.parent / ".events_staging"
    staging_clean = not staging.exists()
    if not staging_clean:
        problems += 1
    print(f"[{'INFO' if staging_clean else 'ERROR'}] staging 정리됨: {staging_clean}")

    if problems:
        print(f"[ERROR] 검증 실패 {problems}건")
        return 1
    print("[INFO] 검증 통과 - events/ 캐시가 계약대로 stage 되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
