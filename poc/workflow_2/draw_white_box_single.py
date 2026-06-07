"""레시피 숫자 4개로 정한 해석(5120-px Corner)을 **단일 깨끗한 이미지** 한 장에 그린다.

`draw_white_box_from_numbers.py` 가 27칸 대조표로 "어느 해석이 맞나"를 골랐다면,
이 스크립트는 그 결론(아래 NATIVE_SIZE·FORMAT)을 가지고 **white box 한 개만**
실제 보기 좋게 그려서 어떻게 생겼는지 눈으로 확인하는 용도다.

배경 두 가지
-----------
1) BACKGROUND_IMAGE 경로가 있으면 그 실제 이미지(예: IMAP0002 SEM) 위에 그린다.
   숫자는 NATIVE_SIZE(5120) 기준이므로, 배경 크기에 맞게 자동으로 스케일한다.
2) 없으면(맥 등 오피스 데이터 없음) 옅은 합성 배경에 그린다 — 모양/위치만 확인.

CLI 인자 없음. 아래 HARDCODE ZONE 만 고쳐서
`uv run python poc/workflow_2/draw_white_box_single.py` 로 실행.
"""

import os

import cv2
import numpy as np

# ============================================================================
# HARDCODE ZONE — 여기만 고치면 됩니다
# ============================================================================
FOUR_NUMBERS = (1600, 1600, 3520, 3520)   # 레시피 raw 의 후보 숫자 4개
NATIVE_SIZE = 5120                         # 그 숫자가 사는 원본 프레임 한 변(px)
FORMAT = "corner"                          # corner | center_size | origin_size
BACKGROUND_IMAGE = ""                      # 실제 이미지 경로(없으면 "" → 합성 배경)
VIEW_SIZE = 1024                           # 합성 배경일 때 저장(보기) 한 변(px)
# ============================================================================

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "white_box_single.jpg")


def quad_to_corner(nums, fmt):
    """4개 숫자(NATIVE_SIZE 기준 px)를 (left, top, right, bottom) 로 바꾼다."""
    a, b, c, d = (float(n) for n in nums)
    if fmt == "corner":                          # (left, top, right, bottom)
        return a, b, c, d
    if fmt == "center_size":                      # (cx, cy, w, h)
        return a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0
    if fmt == "origin_size":                      # (x, y, w, h)
        return a, b, a + c, b + d
    raise ValueError(f"unknown FORMAT: {fmt}")


def make_synthetic_background(w, h):
    """합성 배경: 옅은 그라데이션 + 10% 그리드 + 중앙(=align point) 십자선."""
    grad = np.linspace(40, 90, h, dtype=np.uint8).reshape(h, 1)
    img = cv2.cvtColor(np.repeat(grad, w, axis=1), cv2.COLOR_GRAY2BGR)
    for i in range(1, 10):
        x, y = round(w * i / 10.0), round(h * i / 10.0)
        cv2.line(img, (x, 0), (x, h - 1), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), (70, 70, 70), 1, cv2.LINE_AA)
    return img


def main():
    l0, t0, r0, b0 = quad_to_corner(FOUR_NUMBERS, FORMAT)

    # 배경 준비 + NATIVE_SIZE → 배경 크기 스케일 계산
    if BACKGROUND_IMAGE and os.path.exists(BACKGROUND_IMAGE):
        img = cv2.imread(BACKGROUND_IMAGE)
        if img is None:
            raise SystemExit(f"[ERROR] 이미지를 못 읽음: {BACKGROUND_IMAGE}")
        H, W = img.shape[:2]
        src = f"실제 이미지 {os.path.basename(BACKGROUND_IMAGE)} ({W}x{H})"
    else:
        W = H = VIEW_SIZE
        img = make_synthetic_background(W, H)
        src = f"합성 배경 ({W}x{H}) — 실제 이미지 없음"

    sx, sy = W / float(NATIVE_SIZE), H / float(NATIVE_SIZE)
    l, t, r, b = l0 * sx, t0 * sy, r0 * sx, b0 * sy
    bw, bh = r - l, b - t
    side_frac = (bw / W + bh / H) / 2.0

    # white box (흰색) — unique-area 단서
    thick = max(2, W // 400)
    cv2.rectangle(img, (int(round(l)), int(round(t))),
                  (int(round(r)), int(round(b))), (255, 255, 255), thick, cv2.LINE_AA)

    # box 중심(노랑) vs align point=프레임 중심(빨강 마커) 을 구분해서 표시
    bcx, bcy = int(round((l + r) / 2)), int(round((t + b) / 2))
    fcx, fcy = W // 2, H // 2
    cv2.drawMarker(img, (bcx, bcy), (0, 220, 220), cv2.MARKER_TILTED_CROSS,
                   W // 30, thick)                              # box 중심(노랑)
    cv2.drawMarker(img, (fcx, fcy), (0, 0, 255), cv2.MARKER_CROSS,
                   W // 18, thick)                              # align point(빨강)

    # 캡션 바
    cv2.rectangle(img, (0, 0), (W, 30), (20, 20, 20), -1)
    cap = (f"{FORMAT} {FOUR_NUMBERS}@{NATIVE_SIZE} -> box {bw:.0f}x{bh:.0f}px "
           f"({side_frac*100:.0f}% side) | white=box  red+=align point(center)")
    cv2.putText(img, cap, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_PATH, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[INFO] 배경: {src}")
    print(f"[INFO] box ltrb=({l:.0f},{t:.0f},{r:.0f},{b:.0f}) "
          f"wh=({bw:.0f},{bh:.0f}) side={side_frac:.3f}")
    print(f"[INFO] align point(frame center)=({fcx},{fcy})  box center=({bcx},{bcy})")
    print(f"[INFO] 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
