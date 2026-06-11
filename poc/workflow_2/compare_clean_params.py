"""여러 (thickness:dilate:inpaint_radius) 조합을 **눈으로** 비교하는 격자 이미지.

tune_clean_params.py 의 스칼라 지표(잔상 밝기차)는 over-inpaint(번짐)를 못 잡아
사람 눈과 어긋날 수 있다. 청소 품질은 결국 지각적 판단이므로, 같은 샘플들을 여러
조합으로 청소해 **주석 주변을 크게 확대**해 나란히 보여주고, 사람이 고르게 한다.

각 행 = 샘플 1장, 열 = [원본, 조합1, 조합2, ...]. 셀 위에 'tN dN rN' 라벨.
주석(box/crosshair) 주변만 crop 해 번짐/잔상이 잘 보이게 한다.

CLI 인자 없음. env:
  ALIGN_GOLDEN_ROOT, COMPARE_N(샘플 수, 기본 5),
  COMPARE_COMBOS(예 "1:3:2,1:2:2,2:5:5", 기본 아래 _DEFAULT_COMBOS)
    uv run python poc/workflow_2/compare_clean_params.py
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_1 import WORKFLOW_1_DIR
from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.assets import SUPPORTED_EXTS
from poc.workflow_3.align.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.align.cond_file import load_cond

GOLDEN_ROOT = Path(
    os.getenv("ALIGN_GOLDEN_ROOT", str(WORKFLOW_1_DIR / "align_images_golden"))
)
COMPARE_N = int(os.getenv("COMPARE_N", "5"))
_DEFAULT_COMBOS = "1:3:2,1:2:2,1:4:3,2:3:2,2:5:5"
TILE = 240          # 셀 한 변(px). crop 을 이 크기로 리사이즈해 격자 정렬.
CROP_PAD = 36       # box 주변 여유(px). crosshair 만 있으면 중심 window 사용.


def _parse_combos(s):
    combos = []
    for tok in s.split(","):
        t, d, r = (int(x) for x in tok.strip().split(":"))
        combos.append((t, d, r))
    return combos


def _crop_window(shape_hw, cond):
    """주석 주변 crop 박스 (y0, y1, x0, x1) 를 정한다."""
    h, w = shape_hw[:2]
    if cond.box_ltrb is not None:
        l, t = cursor_to_image(cond.box_ltrb[:2], OVERSAMPLE)
        r, b = cursor_to_image(cond.box_ltrb[2:], OVERSAMPLE)
        x0, y0 = int(l - CROP_PAD), int(t - CROP_PAD)
        x1, y1 = int(r + CROP_PAD), int(b + CROP_PAD)
    else:
        cx, cy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
        half = 130
        x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return y0, y1, x0, x1


def _cell(gray_crop, label):
    """crop 을 TILE 로 리사이즈하고 라벨 바를 올린 BGR 셀."""
    tile = cv2.resize(gray_crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (TILE, 20), (20, 20, 20), -1)
    cv2.putText(bgr, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)
    return bgr


def _sample_images(root, n):
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
    step = len(pool) / float(n)
    return [pool[int(i * step)] for i in range(n)]


def main():
    combos = _parse_combos(os.getenv("COMPARE_COMBOS", _DEFAULT_COMBOS))
    print(f"[INFO] golden root: {GOLDEN_ROOT}")
    print(f"[INFO] 비교 조합(t:d:r): {combos}")
    if not GOLDEN_ROOT.is_dir():
        print(f"[ERROR] 루트 없음: {GOLDEN_ROOT}")
        raise SystemExit(1)

    samples = _sample_images(GOLDEN_ROOT, COMPARE_N)
    if not samples:
        print("[ERROR] box/crosshair 있는 이미지를 못 찾음.")
        raise SystemExit(1)

    rows = []
    for path, cond in samples:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        y0, y1, x0, x1 = _crop_window(gray.shape, cond)
        cells = [_cell(gray[y0:y1, x0:x1], "original")]
        for (t, d, r) in combos:
            out = clean_image(gray, cond, thickness=t, dilate=d, inpaint_radius=r)
            cells.append(_cell(out[y0:y1, x0:x1], f"t{t} d{d} r{r}"))
        rows.append(np.hstack(cells))

    grid = np.vstack(rows)
    out_dir = DEBUG_IMAGE_DIR / "clean_compare" / time.strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "compare.jpg"
    cv2.imwrite(str(dst), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[INFO] 행=샘플 {len(rows)} · 열=[원본 + 조합 {len(combos)}]")
    print(f"[INFO] 비교 격자: {dst}")
    print("[INFO] 마음에 드는 열(t d r)을 알려주면 기본값으로 박아둔다.")


if __name__ == "__main__":
    main()
