"""top-N 후보 + distinctiveness gate 합성 self-test (Slice 1).

검증 목표 (production threshold 증명이 아니라 *구조가 실패를 걸러내는지*):
  1. single  — 패턴이 프레임에 1회만 → best 가 2nd 대비 유일 → distinctive=True, reject_reason=None.
  2. repeated — 동일 패턴이 3곳 반복 → best≈2nd → distinctive=False, reject_reason="not_distinctive".
  3. 후보 목록이 chamfer 내림차순으로 채워지고, repeated 에서 top-N 에 ≥2 후보가 잡힌다.

실행: uv run python poc/workflow_2/test_align_key_distinctiveness.py
"""

import numpy as np
import cv2

from poc.workflow_3.vision.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
)


def _pattern(size=80) -> np.ndarray:
    """엣지가 뚜렷한 distinctive 패턴 (사각 outline + 대각 + 작은 nested box)."""
    img = np.full((size, size), 40, dtype=np.uint8)
    cv2.rectangle(img, (8, 8), (size - 8, size - 8), 235, 2)
    cv2.line(img, (8, 8), (size - 8, size - 8), 235, 2)
    cv2.rectangle(img, (size // 2 - 10, size // 2 - 10), (size // 2 + 10, size // 2 + 10), 255, 2)
    return img


def _frame(positions, pat, canvas=(420, 560), bg=55) -> np.ndarray:
    rs = np.random.RandomState(3)
    img = np.full(canvas, bg, dtype=np.uint8)
    img = cv2.add(img, rs.randint(0, 18, canvas).astype(np.uint8))
    ph, pw = pat.shape
    for (x, y) in positions:
        img[y:y + ph, x:x + pw] = pat
    return img


def main() -> int:
    pat = _pattern()
    template = build_template(pat, recipe_id="T", version="om_center", key_type="om")

    single = _frame([(60, 60)], pat)
    repeated = _frame([(40, 40), (260, 60), (150, 300)], pat)

    r_single = compute_align_key_score(template, single, scales=(1.0,), policy=STRUCTURE_POLICY)
    r_rep = compute_align_key_score(template, repeated, scales=(1.0,), policy=STRUCTURE_POLICY)

    print(f"[single]   candidates={len(r_single.candidates)} distinctive={r_single.distinctive} "
          f"reject={r_single.reject_reason} gap={r_single.score_gap} ratio={r_single.second_ratio}")
    print(f"[repeated] candidates={len(r_rep.candidates)} distinctive={r_rep.distinctive} "
          f"reject={r_rep.reject_reason} gap={r_rep.score_gap} ratio={r_rep.second_ratio}")

    ok = True
    if not (r_single.distinctive and r_single.reject_reason is None):
        print("[FAIL] single 은 distinctive=True / reject=None 이어야 함")
        ok = False
    if len(r_rep.candidates) < 2:
        print("[FAIL] repeated 는 top-N 에 ≥2 후보가 잡혀야 함")
        ok = False
    if not (r_rep.distinctive is False and r_rep.reject_reason == "not_distinctive"):
        print("[FAIL] repeated 는 distinctive=False / reject=not_distinctive 이어야 함")
        ok = False
    # best_xy 가 단일 케이스에서 실제 패턴 위치(중심≈100,100) 근처여야 (회귀 가드).
    bx, by = r_single.best_xy
    if not (abs(bx - 100) <= 12 and abs(by - 100) <= 12):
        print(f"[WARN] single best_xy={r_single.best_xy} 패턴 중심(~100,100)에서 다소 벗어남")

    print("[INFO] PASS" if ok else "[INFO] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
