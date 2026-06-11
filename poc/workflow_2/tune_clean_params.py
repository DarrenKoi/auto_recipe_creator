"""golden 실데이터에서 (thickness · dilate · inpaint_radius) 최적 조합을 찾는다.

각 조합으로 샘플 이미지를 청소한 뒤,
  - ghost_residual : 주석 자리에 남은 잔상(낮을수록 잘 지움) — 1순위 지표.
  - mask_area      : 파괴(=hallucinate)한 면적 비율(낮을수록 원본 보존) — 동점 시 tie-break.
를 측정해 평균낸다. 추천 = '잔상이 충분히 낮으면서(최저 ± 허용오차) 마스크가 가장 작은' 조합.
'무조건 큰 마스크'가 아니라 **충분히 지우는 가장 작은 마스크**가 best 인 이유다.

CLI 인자 없음. env: ALIGN_GOLDEN_ROOT, TUNE_SAMPLE_N(기본 24), TUNE_RESID_TOL(기본 1.0).
    uv run python poc/workflow_2/tune_clean_params.py
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_1 import WORKFLOW_1_DIR
from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.assets import SUPPORTED_EXTS
from poc.workflow_3.align.clean_align_image import build_removal_mask, clean_image
from poc.workflow_2.clean_metrics import (
    build_eval_masks,
    ghost_residual,
    mask_area_fraction,
)
from poc.workflow_3.align.cond_file import load_cond

GOLDEN_ROOT = Path(
    os.getenv("ALIGN_GOLDEN_ROOT", str(WORKFLOW_1_DIR / "align_images_golden"))
)
SAMPLE_N = int(os.getenv("TUNE_SAMPLE_N", "24"))
RESID_TOL = float(os.getenv("TUNE_RESID_TOL", "1.0"))   # 추천 허용 잔상 여유(gray)

# 탐색 격자 (실선이 얇으니 thickness 는 작게, dilate 로 halo 흡수).
THICKNESS_GRID = (1, 2)
DILATE_GRID = (1, 2, 3, 4, 5)
RADIUS_GRID = (2, 3, 5)


def _sample_images(root: Path, n: int):
    """box/crosshair 가 있는 이미지를 데이터셋 전반에서 고르게 n장 고른다."""
    pool = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if any(p.startswith(".") for p in path.relative_to(root).parts):
            continue
        cond = load_cond(path)
        if cond and (cond.box_ltrb or cond.crosshair_xy):
            pool.append((path, cond))
    if len(pool) <= n:
        return pool
    step = len(pool) / float(n)                  # 균등 stride 샘플
    return [pool[int(i * step)] for i in range(n)]


def main():
    print(f"[INFO] golden root: {GOLDEN_ROOT}")
    if not GOLDEN_ROOT.is_dir():
        print(f"[ERROR] 루트 없음: {GOLDEN_ROOT}")
        raise SystemExit(1)

    samples = _sample_images(GOLDEN_ROOT, SAMPLE_N)
    if not samples:
        print("[ERROR] box/crosshair 있는 이미지를 못 찾음.")
        raise SystemExit(1)
    print(f"[INFO] 샘플 {len(samples)}장 × 조합 "
          f"{len(THICKNESS_GRID) * len(DILATE_GRID) * len(RADIUS_GRID)}개 평가")

    # 이미지별로 고정 eval 마스크 + gray 를 미리 캐시.
    cache = []
    for path, cond in samples:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        fp, bg = build_eval_masks(gray.shape[:2],
                                  box_ltrb=cond.box_ltrb, crosshair_xy=cond.crosshair_xy)
        cache.append((gray, cond, fp, bg))

    rows = []
    for t in THICKNESS_GRID:
        for d in DILATE_GRID:
            # 마스크 면적은 inpaint_radius 와 무관 → (t,d) 당 한 번만 측정.
            area = float(np.mean([
                mask_area_fraction(build_removal_mask(
                    gray.shape[:2], box_ltrb=cond.box_ltrb,
                    crosshair_xy=cond.crosshair_xy, thickness=t, dilate=d))
                for gray, cond, _, _ in cache
            ]))
            for r in RADIUS_GRID:
                resid = [
                    ghost_residual(
                        clean_image(gray, cond, thickness=t, dilate=d, inpaint_radius=r),
                        fp, bg)
                    for gray, cond, fp, bg in cache
                ]
                rows.append({"t": t, "d": d, "r": r,
                             "resid": float(np.mean(resid)), "area": area})

    rows.sort(key=lambda x: x["resid"])
    best_resid = rows[0]["resid"]
    # 추천: 잔상이 최저 ± 허용오차 안인 조합 중 마스크 면적이 가장 작은 것.
    within = [x for x in rows if x["resid"] <= best_resid + RESID_TOL]
    rec = min(within, key=lambda x: x["area"])

    print("\n[INFO] 잔상 낮은 순 상위 12개 (resid=잔상↓, area=파괴면적↓):")
    print("       thick dil rad | resid  area%")
    for x in rows[:12]:
        star = " <= 추천" if x is rec else ""
        print(f"       {x['t']:>5} {x['d']:>3} {x['r']:>3} | "
              f"{x['resid']:5.2f}  {x['area']*100:5.3f}{star}")
    print(f"\n[INFO] >>> 추천 조합: thickness={rec['t']} dilate={rec['d']} "
          f"inpaint_radius={rec['r']}  (resid={rec['resid']:.2f}, "
          f"area={rec['area']*100:.3f}%, 최저잔상={best_resid:.2f}±{RESID_TOL})")

    # 추천 조합으로 before|after 몇 장 저장(시각 확인).
    out_dir = DEBUG_IMAGE_DIR / "clean_tune" / time.strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (gray, cond, _, _) in enumerate(cache[:6]):
        out = clean_image(gray, cond, thickness=rec["t"], dilate=rec["d"],
                          inpaint_radius=rec["r"])
        cv2.imwrite(str(out_dir / f"sample{i}_before_after.jpg"),
                    np.hstack([gray, out]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[INFO] 추천 조합 before|after 샘플: {out_dir}")


if __name__ == "__main__":
    main()
