"""레시피 raw 의 후보 숫자 4개로 white box(align-key box)를 그려보는 테스트 스크립트.

목적
----
레시피 raw data 에서 white box 후보로 보이는 **숫자 4개**를 찾았다고 가정한다.
그 4개가 어떤 **단위(unit)** 이고 어떤 **기하 포맷(format)** 인지 모르는 상태에서,
**합성 이미지** 위에 가능한 모든 조합으로 box 를 그려 본다.

  단위(unit) 3가지
    - px        : 숫자가 곧 픽셀
    - norm[0,1] : 0~1 비율 → 변 길이에 곱함
    - norm[0,1000] : 0~1000 비율(VLM 내부 표준) → 변 길이/1000 을 곱함

  기하 포맷(format) 3가지
    - Corner      : (left, top, right, bottom)   # Pascal VOC
    - Center+Size : (cx, cy, w, h)               # YOLO 스타일
    - Origin+Size : (x, y, w, h)                 # OpenCV boundingRect

→ 3 단위 × 3 포맷 = **9 해석** 을, 512²·1024² **두 해상도에서 자동으로** 모두 그린다.
   (해상도를 손으로 줄 필요 없음 — 둘 다 시도한다. 픽셀 숫자는 한쪽 해상도에서만
    white box 지문에 맞고, 정규화 숫자는 양쪽에서 똑같이 맞는다 → 어느 단위인지 역추론.)

각 해석마다 white box 물리 지문( **중앙 · 정사각형 · 면적~10%(변0.316) 또는 선형10%(변0.10)** )
부합도를 점수화해 가장 그럴듯한 (해상도·단위·포맷) 조합을 표시한다.

실제 레시피 이미지는 필요 없다 — 배경은 합성한다.
CLI 인자 없음. 아래 FOUR_NUMBERS 만 고쳐서
`uv run python poc/workflow_2/draw_white_box_from_numbers.py` 로 실행.
"""

import os

import cv2
import numpy as np

# ============================================================================
# 여기만 고치면 됩니다 (HARDCODE ZONE) — 숫자 4개만!
# ============================================================================
FOUR_NUMBERS = (175, 175, 337, 337)   # 레시피 raw 에서 찾은 후보 숫자 4개
# ============================================================================

# 자동으로 둘 다 검사하는 후보 해상도(정사각). 손으로 줄 필요 없음.
CANDIDATE_SIZES = (512, 1024)
# 단위/포맷 후보 — 9 조합을 전수로 그린다.
UNITS = ("px", "norm1", "norm1000")
FORMATS = ("corner", "center_size", "origin_size")
TILE = 360                            # 격자 표시용 패널 한 변(px)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "white_box_from_numbers.jpg")

_UNIT_LABEL = {"px": "px", "norm1": "norm[0,1]", "norm1000": "norm[0,1000]"}
_FMT_LABEL = {"corner": "Corner", "center_size": "Center+Size", "origin_size": "Origin+Size"}


def make_synthetic_background(w, h):
    """배경 합성: 옅은 그라데이션 + 10% 그리드 + 중앙 십자선.

    실제 SEM 이미지를 흉내 내려는 게 아니라, box 좌표가 이미지 어디에
    떨어지는지 '눈금'을 주려는 용도다.
    """
    grad = np.linspace(40, 90, h, dtype=np.uint8).reshape(h, 1)
    img = cv2.cvtColor(np.repeat(grad, w, axis=1), cv2.COLOR_GRAY2BGR)
    for i in range(1, 10):                       # 10% 간격 그리드
        x, y = round(w * i / 10.0), round(h * i / 10.0)
        cv2.line(img, (x, 0), (x, h - 1), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), (70, 70, 70), 1, cv2.LINE_AA)
    cx, cy = w // 2, h // 2                       # 정중앙 십자선(white box 기준선)
    cv2.line(img, (cx, 0), (cx, h - 1), (0, 130, 130), 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (w - 1, cy), (0, 130, 130), 1, cv2.LINE_AA)
    return img


def to_pixels(nums, unit, w, h):
    """4개 숫자를 단위에 따라 픽셀로 환산한다(포맷 무관, 축 패턴 [x,y,x,y]).

    x 성분(0,2)에는 w 를, y 성분(1,3)에는 h 를 적용한다.
    """
    a, b, c, d = nums
    if unit == "px":
        return float(a), float(b), float(c), float(d)
    if unit == "norm1":                          # 0~1 비율
        return a * w, b * h, c * w, d * h
    if unit == "norm1000":                       # 0~1000 비율
        return a / 1000.0 * w, b / 1000.0 * h, c / 1000.0 * w, d / 1000.0 * h
    raise ValueError(f"unknown unit: {unit}")


def quad_to_corner(px_nums, fmt):
    """픽셀 환산된 4개를 (left, top, right, bottom) 픽셀로 바꾼다."""
    a, b, c, d = px_nums
    if fmt == "corner":                          # (left, top, right, bottom)
        return a, b, c, d
    if fmt == "center_size":                     # (cx, cy, w, h)
        return a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0
    if fmt == "origin_size":                     # (x, y, w, h)
        return a, b, a + c, b + d
    raise ValueError(f"unknown fmt: {fmt}")


def signature_score(l, t, r, b, w, h):
    """white box 물리 지문 부합도를 0~1 로 점수화한다.

    중앙성(0.4) + 정사각형(0.3) + 면적(0.3) 가중 평균.
    면적은 면적10%(변0.316)·선형10%(변0.10) 중 가까운 쪽 채택.
    """
    bw, bh = r - l, b - t
    if bw <= 0 or bh <= 0:
        return 0.0, "변 길이 음수/0 (포맷·단위 불일치)"

    bcx, bcy = (l + r) / 2.0, (t + b) / 2.0
    center_err = (abs(bcx - w / 2.0) / (w / 2.0) + abs(bcy - h / 2.0) / (h / 2.0)) / 2.0
    center_score = max(0.0, 1.0 - center_err)
    square_score = min(bw, bh) / max(bw, bh)

    side_frac = (bw / w + bh / h) / 2.0
    d_area, d_lin = abs(side_frac - 0.316), abs(side_frac - 0.10)
    area_dev = min(d_area, d_lin)
    area_kind = "면적10%(0.316)" if d_area <= d_lin else "선형10%(0.10)"
    area_score = max(0.0, 1.0 - area_dev / 0.316)

    score = 0.4 * center_score + 0.3 * square_score + 0.3 * area_score
    note = (f"중심오차 {center_err:.2f} · 정사각 {square_score:.2f} · "
            f"변비율 {side_frac:.3f}→{area_kind}")
    return score, note


def build_panel(nums, unit, fmt, size):
    """하나의 (단위·포맷·해상도) 해석으로 box 를 그린 TILE 크기 패널을 만든다."""
    w = h = size
    l, t, r, b = quad_to_corner(to_pixels(nums, unit, w, h), fmt)
    score, note = signature_score(l, t, r, b, w, h)

    img = make_synthetic_background(w, h)
    color = (0, 255, 0) if score >= 0.7 else (0, 165, 255) if score >= 0.4 else (0, 0, 255)
    li, ti, ri, bi = int(round(l)), int(round(t)), int(round(r)), int(round(b))
    cv2.rectangle(img, (li, ti), (ri, bi), color, max(2, size // 256), cv2.LINE_AA)
    cv2.drawMarker(img, (int(round((l + r) / 2)), int(round((t + b) / 2))),
                   color, cv2.MARKER_CROSS, size // 18, max(2, size // 256))

    # 격자 표시용으로 동일 크기 타일로 리사이즈한 뒤 텍스트를 올린다(글자 선명).
    tile = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (TILE, 22), (20, 20, 20), -1)
    head = f"{size} | {_UNIT_LABEL[unit]} | {_FMT_LABEL[fmt]}  s={score:.2f}"
    cv2.putText(tile, head, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, TILE - 18), (TILE, TILE), (20, 20, 20), -1)
    foot = f"ltrb=({l:.0f},{t:.0f},{r:.0f},{b:.0f}) wh=({r-l:.0f},{b-t:.0f})"
    cv2.putText(tile, foot, (5, TILE - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (220, 220, 220), 1, cv2.LINE_AA)
    return tile, score


def main():
    nums = FOUR_NUMBERS
    print(f"[INFO] 후보 숫자 {nums} — 해상도 {CANDIDATE_SIZES} × 단위 {len(UNITS)} × "
          f"포맷 {len(FORMATS)} = {len(CANDIDATE_SIZES) * len(UNITS) * len(FORMATS)} 해석")

    rows, results = [], []
    for size in CANDIDATE_SIZES:
        for unit in UNITS:                       # 각 (해상도·단위) → 포맷 3개를 한 줄로
            row_tiles = []
            for fmt in FORMATS:
                tile, score = build_panel(nums, unit, fmt, size)
                row_tiles.append(tile)
                results.append((score, size, unit, fmt))
                print(f"[INFO] {size} | {_UNIT_LABEL[unit]:13s} | {_FMT_LABEL[fmt]:12s} "
                      f"score={score:.2f}")
            rows.append(np.hstack(row_tiles))

    # 전수 랭킹 + 전역 최적
    results.sort(reverse=True)
    best_score, best_size, best_unit, best_fmt = results[0]
    if best_score >= 0.7:
        print(f"[INFO] >>> 가장 유력: {best_size}px · {_UNIT_LABEL[best_unit]} · "
              f"{_FMT_LABEL[best_fmt]} (score={best_score:.2f})")
        # 교차 스케일 해석: 같은 (단위·포맷)이 두 해상도에서 어떻게 변하나
        if best_unit == "px":
            print("[INFO]     단위=px 가 특정 해상도에서만 부합 → 숫자는 그 해상도의 "
                  "픽셀 저장(§5.4: 해상도 바뀌면 ~2배).")
        else:
            print("[INFO]     단위=정규화 가 부합 → 해상도와 무관(§5.4: 숫자 그대로). "
                  "nm 물리단위도 같은 성질(§4).")
    else:
        print(f"[WARNING] 어떤 (해상도·단위·포맷)도 white box 지문에 강하게 부합하지 않음 "
              f"(최고 {best_score:.2f}). 숫자가 box 가 아니거나 nm·반변(half-size) 저장일 수 "
              f"있음(§4,§8).")

    sheet = np.vstack(rows)
    cv2.imwrite(OUTPUT_PATH, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[INFO] 저장: {OUTPUT_PATH}  ({sheet.shape[1]}x{sheet.shape[0]})  "
          f"행=해상도×단위, 열={'/'.join(_FMT_LABEL[f] for f in FORMATS)}")


if __name__ == "__main__":
    main()
