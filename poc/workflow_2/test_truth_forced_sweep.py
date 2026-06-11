"""truth-forced sweep 진단 합성 self-test.

검증 (production threshold 가 아니라 *구조가 병목을 분리*하는지):
  - scale_band: 패턴을 1.2x 로 심으면 wide sweep(>1.0 포함)은 회복(best_scale≈1.2,
    scale_gain>0)하고 compare band(≤1.0)는 낮다 → diagnosis="scale_band_problem".
  - 반환 dict 의 핵심 필드가 채워지는지.
실행: uv run python poc/workflow_2/test_truth_forced_sweep.py
"""

import cv2
import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_2.align_similarity import _truth_forced


def _pattern(size=64) -> np.ndarray:
    img = np.full((size, size), 40, dtype=np.uint8)
    cv2.rectangle(img, (8, 8), (size - 8, size - 8), 235, 2)
    cv2.line(img, (8, 8), (size - 8, size - 8), 235, 2)
    cv2.rectangle(img, (size // 2 - 9, size // 2 - 9), (size // 2 + 9, size // 2 + 9), 255, 2)
    return img


def main() -> int:
    pat = _pattern(64)
    template = build_template(pat, recipe_id="T", version="om_center", key_type="om")

    # 1.2x 로 확대해 프레임 중앙에 심는다 (C4: 생산 band ≤1.0 은 놓치고 wide 만 회복).
    scaled = cv2.resize(pat, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)
    sh, sw = scaled.shape
    canvas = np.full((360, 480), 55, dtype=np.uint8)
    canvas = cv2.add(canvas, np.random.RandomState(2).randint(0, 16, canvas.shape).astype(np.uint8))
    cxh, cyh = 240, 180
    x0, y0 = cxh - sw // 2, cyh - sh // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = scaled

    t = _truth_forced(canvas, (cxh, cyh), {"om": template}, None)
    assert t is not None, "_truth_forced returned None"
    print("[truth]", {k: t[k] for k in ("valid", "err_px", "wide_chamfer", "wide_scale",
                                        "compare_chamfer", "scale_gain", "diagnosis")})
    print("[per_scale]", t["per_scale_chamfer"])

    ok = True
    for key in ("valid", "wide_chamfer", "wide_scale", "compare_chamfer", "scale_gain",
                "mean_dt_px", "tpl_edge_density", "per_scale_chamfer", "diagnosis"):
        if key not in t:
            print(f"[FAIL] 필드 누락: {key}")
            ok = False
    if not t["valid"]:
        print(f"[FAIL] truth 가 valid 여야 함 (err={t['err_px']}px)")
        ok = False
    if not (t["wide_scale"] > 1.0):
        print(f"[FAIL] 1.2x 패턴인데 wide_scale={t['wide_scale']} (>1.0 기대)")
        ok = False
    if not (t["scale_gain"] > 0):
        print(f"[FAIL] scale_gain={t['scale_gain']} (>0 기대: wide 가 compare 보다 높아야)")
        ok = False
    if t["diagnosis"] != "scale_band_problem":
        print(f"[WARN] diagnosis={t['diagnosis']} (scale_band_problem 기대 — 합성 임계 민감)")

    print("[INFO] PASS" if ok else "[INFO] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
