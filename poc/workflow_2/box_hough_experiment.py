"""Hough 기반 흰 box 사각형 조립 실험 (throwaway) — align_point_correction 에 포팅 전 튜닝용.

진단 결과: 흰 box 는 흐릿해도 *곧은 축정렬 사각형*이라 Canny+HoughLinesP 가 4변을 잘 잡는다.
반면 top-hat contour + coverage 게이트는 윗변 끊김(5.png)·내부 겹침(4.png hole array)에 약하다.
여기서 "긴 H/V 직선을 군집화 → 2H+2V 로 사각형을 조립 → 네 변 edge 지지도로 채점" 을 튜닝한다.

실행: uv run python poc/workflow_2/box_hough_experiment.py
출력: debug_images/box_hough_experiment/<timestamp>/<stem>_hough_box.jpg
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR, WORKFLOW_2_DIR
from poc.workflow_1.util.time_utils import make_timestamp_tag

INPUT_DIR = WORKFLOW_2_DIR / "templates" / "created_by_myself"
OUTPUT_ROOT = DEBUG_IMAGE_DIR / "box_hough_experiment"

# 튜닝 상수.
CANNY_SIGMA = 1.2
CANNY_LO, CANNY_HI = 40, 120
HOUGH_THRESH = 70
HOUGH_MINLEN_RATIO = 0.15        # 최소 직선 길이(짧은 변 대비).
HOUGH_MAXGAP = 15
ANGLE_TOL_DEG = 10               # H/V 판정 허용 오차.
CLUSTER_TOL_RATIO = 0.012        # 같은 변으로 묶을 y(또는 x) 허용 간격(짧은 변 대비).
MIN_SIDE_RATIO = 0.10            # 사각형 짧은 변 하한(짧은 변 대비).
MAX_AREA_RATIO = 0.55
MIN_AREA_RATIO = 0.01
EDGE_MARGIN = 3
SIDE_SUPPORT_MIN = 0.55          # 네 변 각각 edge 지지도 하한.
CORNER_TOL_RATIO = 0.06          # 변 끝점이 코너에 닿았다고 볼 허용 거리(짧은 변 대비).


def _overlap(a_lo, a_hi, b_lo, b_hi) -> float:
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def _cluster(items, tol):
    """items: list[(key, span_lo, span_hi, length)] → key 로 군집화.

    반환: list[(key_weighted, span_lo_min, span_hi_max, total_length)] (key 오름차순).
    """
    if not items:
        return []
    items = sorted(items, key=lambda t: t[0])
    clusters = []
    cur = list(items[0])
    cur_wsum = items[0][0] * items[0][3]
    cur_len = items[0][3]
    for key, lo, hi, ln in items[1:]:
        if abs(key - (cur_wsum / max(cur_len, 1e-6))) <= tol:
            cur[1] = min(cur[1], lo)
            cur[2] = max(cur[2], hi)
            cur_wsum += key * ln
            cur_len += ln
        else:
            clusters.append((cur_wsum / max(cur_len, 1e-6), cur[1], cur[2], cur_len))
            cur = [key, lo, hi, ln]
            cur_wsum = key * ln
            cur_len = ln
    clusters.append((cur_wsum / max(cur_len, 1e-6), cur[1], cur[2], cur_len))
    return clusters


def detect_box_hough(gray: np.ndarray):
    """Canny+Hough 로 흰 box 사각형을 조립한다. 반환 (x,y,w,h) | None."""
    h, w = gray.shape[:2]
    short = min(h, w)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), CANNY_SIGMA), CANNY_LO, CANNY_HI)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESH,
                            minLineLength=int(HOUGH_MINLEN_RATIO * short), maxLineGap=HOUGH_MAXGAP)
    if lines is None:
        return None

    horiz, vert = [], []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ang = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        length = float(np.hypot(x2 - x1, y2 - y1))
        if ang < ANGLE_TOL_DEG or ang > 180 - ANGLE_TOL_DEG:
            y = (y1 + y2) / 2.0
            horiz.append((y, min(x1, x2), max(x1, x2), length))
        elif 90 - ANGLE_TOL_DEG < ang < 90 + ANGLE_TOL_DEG:
            x = (x1 + x2) / 2.0
            vert.append((x, min(y1, y2), max(y1, y2), length))

    tol = CLUSTER_TOL_RATIO * short
    Hc = _cluster(horiz, tol)   # (y, xlo, xhi, len)
    Vc = _cluster(vert, tol)    # (x, ylo, yhi, len)
    if len(Hc) < 2 or len(Vc) < 2:
        return None

    corner_tol = CORNER_TOL_RATIO * short
    best, best_score = None, 0.0
    for i in range(len(Hc)):
        for j in range(len(Hc)):
            ytop, txlo, txhi, _ = Hc[i]
            ybot, bxlo, bxhi, _ = Hc[j]
            if ybot - ytop < MIN_SIDE_RATIO * short:
                continue
            for k in range(len(Vc)):
                for m in range(len(Vc)):
                    xl, lylo, lyhi, _ = Vc[k]
                    xr, rylo, ryhi, _ = Vc[m]
                    if xr - xl < MIN_SIDE_RATIO * short:
                        continue
                    bw, bh = xr - xl, ybot - ytop
                    area = bw * bh
                    if not (MIN_AREA_RATIO * h * w <= area <= MAX_AREA_RATIO * h * w):
                        continue
                    if (xl < EDGE_MARGIN or ytop < EDGE_MARGIN
                            or xr > w - EDGE_MARGIN or ybot > h - EDGE_MARGIN):
                        continue
                    # 네 변 edge 지지도 — 각 직선 span 이 사각형 변을 얼마나 덮나.
                    top_s = _overlap(txlo, txhi, xl, xr) / bw
                    bot_s = _overlap(bxlo, bxhi, xl, xr) / bw
                    left_s = _overlap(lylo, lyhi, ytop, ybot) / bh
                    right_s = _overlap(rylo, ryhi, ytop, ybot) / bh
                    side_min = min(top_s, bot_s, left_s, right_s)
                    if side_min < SIDE_SUPPORT_MIN:
                        continue
                    # 코너 일치 — 수평선 끝이 좌/우 x 근처, 수직선 끝이 상/하 y 근처.
                    corner_ok = (
                        min(abs(txlo - xl), abs(bxlo - xl)) < corner_tol * 3
                        and min(abs(txhi - xr), abs(bxhi - xr)) < corner_tol * 3
                    )
                    score = side_min + (0.2 if corner_ok else 0.0) + 0.0001 * area
                    if score > best_score:
                        best_score = score
                        best = (int(round(xl)), int(round(ytop)), int(round(bw)), int(round(bh)))
    return best


def run():
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".png")
    hit = 0
    for p in images:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        box = detect_box_hough(gray)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if box:
            x, y, bw, bh = box
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (60, 220, 60), 3, cv2.LINE_AA)
            hit += 1
            print(f"[INFO] {p.name}: box={box}")
        else:
            cv2.putText(vis, "NO BOX", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 230), 2)
            print(f"[INFO] {p.name}: NO BOX")
        cv2.imwrite(str(out_dir / f"{p.stem}_hough_box.jpg"), vis,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[INFO] 검출 {hit}/{len(images)} → {out_dir}")


if __name__ == "__main__":
    run()
