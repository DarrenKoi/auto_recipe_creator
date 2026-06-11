"""success gather 1회 검증 — office 에서 실제 DB stage 가 계약대로 되는지 확인.

office PC 전용 검증 스크립트(success downloader 가 없으면 즉시 종료). 대상 recipe 는
assets 해석과 같은 env 규약으로 지정하고, CLI 인자는 쓰지 않는다:

    ALIGN_EQP_ID=<eqp> ALIGN_CLASS_NAME=<class> ALIGN_RECIPE_NAME=<recipe> \\
      uv run python poc/workflow_3/monitor/verify_success_gather.py

확인 항목: reason=ok / event 디렉토리별 S*.jpeg + 숨김폴더 cond(.<이미지명>/cond.txt) 짝 /
cond 파싱(crosshair 좌표, modality 추론 — msr cond 는 Scope 가 없어 msr_modality() 키로
가른다) / staging 정리. 통과하면 events/ 는 그대로 캐시로 남는다(추가 정리 불필요).
"""

import os

from poc.workflow_3.monitor.success_gather import DOWNLOADER_AVAILABLE, _DOWNLOADER
from poc.workflow_3.align.cond_file import load_cond, msr_modality
from poc.workflow_3.align.consensus_gather import gather_success_images


def main():
    if not DOWNLOADER_AVAILABLE:
        print("[ERROR] success downloader 없음 - office PC 에서 실행하세요.")
        return 1

    eqp_id = os.environ.get("ALIGN_EQP_ID", "").strip()
    class_name = os.environ.get("ALIGN_CLASS_NAME", "").strip()
    recipe_name = os.environ.get("ALIGN_RECIPE_NAME", "").strip()
    if not eqp_id or not class_name or not recipe_name:
        print("[ERROR] ALIGN_EQP_ID / ALIGN_CLASS_NAME / ALIGN_RECIPE_NAME env 를 지정하세요.")
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
