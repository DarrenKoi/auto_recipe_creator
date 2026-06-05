"""rcp 흰 unique-area box 검출 *앙상블* — 여러 CV 검출기를 동시에 돌려 가장 'box 다운'
후보를 통합 점수로 골라낸다 (사용자 요청 2026-06-05: "multiple cv jobs at once, choose best").

배경:
  기존 단일 경로(`align_point_correction._detect_white_box`: photometric saturation island →
  top-hat+Otsu 폴백)는 실데이터에서 검출률이 약하다. 흰 box 는 *얇은 밝은 축정렬 사각형
  outline* 인데, 오피스 이미지에서는 (a) overlay 가 고정 채도 섬을 못 이루거나, (b) busy
  SEM 소자 패턴이 mask 에 끼어 사각형 게이트를 깨뜨린다. 한 방법이 못 잡아도 다른 방법은
  잡는 경우가 많다 → **여러 검출기를 병렬로 제안받고, 검출기-무관 품질 점수로 최선을 채택**.

설계 (research 근거: OpenCV table/grid line extraction, projection profile, HoughLinesP/LSD):
  각 검출기는 *후보 bbox 리스트* 만 내고(가벼운 sanity 만), 판별은 **단일 통합 scorer**가 한다.
  이렇게 해야 5개 검출기가 같은 잣대로 경쟁하고 best 를 뽑을 수 있다. scorer 는 기존
  `_rect_frame_ok`(yes/no 게이트)를 0~1 *연속 점수*로 승격한 것 — border 집중도·네 변 닫힘·
  stroke 대비·기하 prior 의 가중합. accept 임계 미달이면 None (→ 생산은 center fallback).

검출기 (front-end 만 다름, 같은 scorer 로 rerank):
  1. photometric  — overlay 고정 채도 섬 (기존, 깨끗한 고정값 overlay 에 강함).
  2. adaptive     — top-hat + Otsu (기존, 저대비 폴백).
  3. morphology   — 가로/세로 긴 선 커널 open → 직선만 남겨 wavy 소자 제거 (table-line 기법, 主력).
  4. projection   — 밝은 mask 의 행/열 합 peak → 축정렬 벽 2쌍 (no_island·faint box 에 강함).
  5. hough        — Canny + HoughLinesP 로 수평/수직 선 검출 → 최외곽으로 사각형 조립.

실행(점검): uv run python poc/workflow_2/white_box_detectors.py  (합성 hard-case self-test)
임계/커널 상수는 **오피스 실데이터 calibration 전 cold-start 값** — golden_localization_eval 의
박스 검출 진단(reason 히스토그램)으로 튜닝한다. 절대 픽셀 동일성 가정 없음.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, field

import cv2
import numpy as np

from poc.workflow_2.align_point_correction import (
    RCP_BOX_EDGE_MARGIN_PX,
    RCP_BOX_FRAME_BAND_RATIO,
    RCP_BOX_MAX_AREA_RATIO,
    RCP_BOX_MAX_ASPECT,
    RCP_BOX_MIN_AREA_RATIO,
    RCP_BOX_MIN_SIDE_RATIO,
    TOPHAT_KERNEL,
    _detect_white_box_adaptive,
    _detect_white_box_photometric,
)

# ------------------------------------------------------------------
# 튜닝 상수 (cold-start — 오피스 calibration 대상).
# ------------------------------------------------------------------

# morphology: 직선 커널 길이 = 이미지 변의 이 비율. 너무 길면 작은 box 변까지 지움(놓침),
# 너무 짧으면 wavy 소자 단편이 살아남음. box 최소 변(5%)보다 짧아야 변이 보존된다.
MORPH_LINE_FRAC = 0.03
MORPH_MIN_LINE = 12          # 커널 길이 하한(px) — 아주 작은 이미지 보호.
MORPH_CLOSE_K = 5            # 선 끊김 잇는 close 커널.

# projection: 행/열 합 profile 에서 '벽'으로 볼 peak 의 상대 임계(최댓값 대비).
PROJ_PEAK_REL = 0.35

# hough: Canny + HoughLinesP + 축정렬 각도 허용오차.
HOUGH_CANNY_LO = 50
HOUGH_CANNY_HI = 150
HOUGH_THRESH = 40
HOUGH_MIN_LEN_FRAC = 0.04
HOUGH_MIN_LINE = 12
HOUGH_MAX_GAP = 8
HOUGH_ANGLE_TOL_DEG = 8.0

# 후보 contour 에서 면적 상위 몇 개까지 scorer 에 넘길지(나머지는 어차피 저점).
CAND_TOP_K = 6

# 통합 품질 점수 가중치 (합=1.0). frame=border 집중도, side=네 변 닫힘, contrast=stroke 대비.
W_FRAME = 0.40
W_SIDE = 0.40
W_CONTRAST = 0.20

# contrast 정규화 — border-interior 밝기차가 이 비율(0~1)이면 만점.
CONTRAST_FULL = 0.30

# scorer 내부 bright 임계 — region 의 [40%, 99%] 분위로 동적범위를 잡고 그 사이 REL 지점.
# *절대 floor 없음* — faint/anti-alias box(예: stroke~150)도 채점 대상에 들도록(2026-06-05 self-test
# 회귀: _stroke_threshold 의 180 floor 가 faint box 를 통째로 0점 처리했음).
SCORE_STROKE_REL = 0.5
SCORE_MIN_DYN_RANGE = 15.0   # region 동적범위(hi-lo)가 이 미만이면 stroke 없음 → 0점.

# 앙상블 채택 임계 — best 후보 점수가 이 미만이면 'box 없음'으로 보고 None.
ENSEMBLE_ACCEPT_SCORE = 0.50


@dataclass
class BoxCandidate:
    """검출기 하나가 제안한 box 후보 + 통합 점수."""

    bbox: tuple        # (x, y, w, h)
    source: str        # 제안한 검출기 이름.
    score: float = 0.0  # 통합 품질 점수 0~1.
    parts: dict = field(default_factory=dict)  # 점수 구성요소(디버그).


# ------------------------------------------------------------------
# 통합 품질 scorer — 모든 검출기 후보를 같은 잣대로 0~1 채점.
# ------------------------------------------------------------------


def score_box_quality(gray: np.ndarray, bbox: tuple) -> tuple:
    """후보 bbox 가 얼마나 '얇은 밝은 축정렬 사각형 outline' 다운지 0~1 로 채점.

    기하 prior(면적·짧은 변·가장자리·종횡비)는 hard 게이트(벗어나면 0). 통과하면
    border 집중도(frame_ratio)·네 변 닫힘(side_cov)·stroke 대비(contrast)의 가중합.
    반환: (score, parts dict).
    """
    x, y, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    h, w = gray.shape[:2]
    total_area = float(h * w)

    def _zero(reason: str) -> tuple:
        return 0.0, {"reject": reason}

    if bw < 8 or bh < 8:
        return _zero("too_small")
    area_ratio = (bw * bh) / total_area
    if area_ratio < RCP_BOX_MIN_AREA_RATIO or area_ratio > RCP_BOX_MAX_AREA_RATIO:
        return _zero("area_out")
    if min(bw, bh) < RCP_BOX_MIN_SIDE_RATIO * min(h, w):
        return _zero("side_small")
    m = RCP_BOX_EDGE_MARGIN_PX
    if x < m or y < m or (x + bw) > (w - m) or (y + bh) > (h - m):
        return _zero("edge")
    aspect = max(bw, bh) / max(1, min(bw, bh))
    if aspect > RCP_BOX_MAX_ASPECT:
        return _zero("aspect")

    region = gray[y:y + bh, x:x + bw]
    hi = float(np.percentile(region, 99))
    lo = float(np.percentile(region, 40))
    if (hi - lo) < SCORE_MIN_DYN_RANGE:
        return _zero("flat")                      # 동적범위 없음 = stroke 없음.
    thr = lo + SCORE_STROKE_REL * (hi - lo)       # floor 없는 상대 임계(faint box 포용).
    bright = (region >= thr).astype(np.uint8)
    total = float(bright.sum())
    if total <= 0:
        return _zero("no_bright")

    band = max(2, int(round(RCP_BOX_FRAME_BAND_RATIO * min(bw, bh))))
    if bh <= 2 * band or bw <= 2 * band:
        return _zero("band")

    # (1) border 집중도 — 밝은 픽셀이 테두리 band 에 몰려 있는가(내부 비어있음 → outline).
    interior = bright[band:bh - band, band:bw - band]
    on_border = total - float(interior.sum())
    frame_ratio = on_border / total

    # (2) 네 변 닫힘 — 각 변 band 가 변 길이의 얼마나 이어지나(가장 약한 변 채택).
    top_cov = float(bright[:band, :].any(axis=0).mean())
    bot_cov = float(bright[bh - band:, :].any(axis=0).mean())
    left_cov = float(bright[:, :band].any(axis=1).mean())
    right_cov = float(bright[:, bw - band:].any(axis=1).mean())
    side_cov = min(top_cov, bot_cov, left_cov, right_cov)

    # (3) stroke 대비 — border 밝기 vs 내부 밝기.
    border_mask = np.ones((bh, bw), bool)
    border_mask[band:bh - band, band:bw - band] = False
    border_mean = float(region[border_mask].mean()) if border_mask.any() else 0.0
    interior_mean = float(region[~border_mask].mean()) if (~border_mask).any() else border_mean
    contrast = max(0.0, min(1.0, ((border_mean - interior_mean) / 255.0) / CONTRAST_FULL))

    score = W_FRAME * min(1.0, frame_ratio) + W_SIDE * side_cov + W_CONTRAST * contrast
    parts = {
        "score": round(float(score), 3),
        "frame_ratio": round(frame_ratio, 3),
        "side_cov_min": round(side_cov, 3),
        "side_cov": [round(top_cov, 2), round(bot_cov, 2), round(left_cov, 2), round(right_cov, 2)],
        "contrast": round(contrast, 3),
        "area_ratio": round(area_ratio, 4),
        "aspect": round(aspect, 2),
    }
    return float(score), parts


# ------------------------------------------------------------------
# 공통 헬퍼.
# ------------------------------------------------------------------


def _tophat_binary(gray: np.ndarray) -> "np.ndarray | None":
    """top-hat + Otsu 로 '국소 배경보다 밝은 얇은 구조' 이진 mask. 밝은 게 없으면 None."""
    k = np.ones((TOPHAT_KERNEL, TOPHAT_KERNEL), np.uint8)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    if int(th.max()) == 0:
        return None
    _t, mask = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def _bboxes_from_contours(contours: list, h: int, w: int, top_k: int = CAND_TOP_K) -> list:
    """contour 들에서 면적 sanity(min~max ratio) 통과한 bbox 를 면적순 상위 top_k 개."""
    total_area = float(h * w)
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = (bw * bh) / total_area
        if ar < RCP_BOX_MIN_AREA_RATIO or ar > RCP_BOX_MAX_AREA_RATIO:
            continue
        out.append((int(x), int(y), int(bw), int(bh)))
    out.sort(key=lambda b: -(b[2] * b[3]))
    return out[:top_k]


def _profile_wall_centers(profile: np.ndarray, rel: float) -> list:
    """1D profile 에서 rel*max 이상 구간(벽 후보)의 중심 좌표 리스트(좌→우/상→하)."""
    peak = float(profile.max())
    if peak <= 0:
        return []
    thr = max(1.0, rel * peak)
    on = profile >= thr
    centers = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            centers.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return centers


# ------------------------------------------------------------------
# 검출기들 — 각각 후보 bbox 리스트만 반환(판별은 scorer).
# ------------------------------------------------------------------


def detect_photometric(gray: np.ndarray) -> list:
    """기존 photometric saturation-island 검출 (단일 box → 리스트 wrap)."""
    box = _detect_white_box_photometric(gray)
    return [box] if box is not None else []


def detect_adaptive(gray: np.ndarray) -> list:
    """기존 top-hat+Otsu adaptive 검출 (단일 box → 리스트 wrap)."""
    box = _detect_white_box_adaptive(gray)
    return [box] if box is not None else []


def detect_morphology_lines(gray: np.ndarray) -> list:
    """가로/세로 긴 선 커널로 open → 직선만 남겨 wavy 소자 제거 후 contour bbox.

    OpenCV 의 table/grid line 추출 기법. box 의 4변(긴 직선)은 살고, 구불구불한 소자
    패턴(짧은 런)은 open 으로 사라진다 → 사각형 frame 만 깨끗이 남는 게 핵심.
    """
    mask = _tophat_binary(gray)
    if mask is None:
        return []
    h, w = gray.shape[:2]
    lh = max(MORPH_MIN_LINE, int(round(MORPH_LINE_FRAC * w)))
    lv = max(MORPH_MIN_LINE, int(round(MORPH_LINE_FRAC * h)))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (lh, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, lv)))
    lines = cv2.bitwise_or(horiz, vert)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                             np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _bboxes_from_contours(contours, h, w)


def detect_projection(gray: np.ndarray) -> list:
    """밝은 mask 의 행/열 합 profile 에서 축정렬 '벽' peak 2쌍 → bbox.

    축정렬 box 는 좌/우 벽이 열 합에 키 큰 좁은 peak 2개, 상/하 벽이 행 합에 2개로 나온다.
    최외곽 peak 쌍을 box 경계로 제안(여러 peak 면 scorer 가 옳은 조합을 가려냄).
    """
    mask = _tophat_binary(gray)
    if mask is None:
        return []
    h, w = gray.shape[:2]
    b = (mask > 0).astype(np.int32)
    xs = _profile_wall_centers(b.sum(axis=0), PROJ_PEAK_REL)
    ys = _profile_wall_centers(b.sum(axis=1), PROJ_PEAK_REL)
    if len(xs) < 2 or len(ys) < 2:
        return []
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return []
    return [(x0, y0, x1 - x0, y1 - y0)]


def detect_hough(gray: np.ndarray) -> list:
    """Canny + HoughLinesP 로 수평/수직 선 검출 → 최외곽 선으로 사각형 조립.

    얇은/끊긴/저대비 stroke 도 edge 로는 살아남아 선분으로 검출되는 경우가 많다(occlusion
    robust). 축정렬(±각도허용) 선만 골라 최외곽 수평 2 + 수직 2 의 bounding box 를 제안.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, HOUGH_CANNY_LO, HOUGH_CANNY_HI)
    min_len = max(HOUGH_MIN_LINE, int(round(HOUGH_MIN_LEN_FRAC * min(h, w))))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, HOUGH_THRESH,
                            minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP)
    if lines is None:
        return []
    xs_v, ys_h = [], []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ang = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        if ang < HOUGH_ANGLE_TOL_DEG or ang > 180.0 - HOUGH_ANGLE_TOL_DEG:
            ys_h.append((y1 + y2) / 2.0)        # 수평선 — y 위치.
        elif abs(ang - 90.0) < HOUGH_ANGLE_TOL_DEG:
            xs_v.append((x1 + x2) / 2.0)        # 수직선 — x 위치.
    if len(xs_v) < 2 or len(ys_h) < 2:
        return []
    x0, x1 = int(min(xs_v)), int(max(xs_v))
    y0, y1 = int(min(ys_h)), int(max(ys_h))
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return []
    return [(x0, y0, x1 - x0, y1 - y0)]


# ------------------------------------------------------------------
# 검출기 레지스트리 — 모듈식. 불필요한 방법은 ENABLED_DETECTORS 에서 한 줄 지우면 빠진다.
# ------------------------------------------------------------------
#
# DETECTOR_REGISTRY = 사용 가능한 모든 검출기(이름→함수). 새 방법은 여기 등록.
# ENABLED_DETECTORS = *실제로 돌릴* 검출기 이름 순서. 오피스 per-source 점수를 보고
#   일 안 하는 방법의 줄을 지우면(또는 주석 처리) 그 검출기가 앙상블에서 빠진다 —
#   코드 다른 곳은 손대지 않는다(사용자 요청 2026-06-05: 결과 기반 쉬운 제거).

DETECTOR_REGISTRY = {
    "photometric": detect_photometric,   # overlay 고정 채도 섬 (깨끗한 고정값 overlay).
    "adaptive": detect_adaptive,         # top-hat + Otsu 폴백 (저대비).
    "morphology": detect_morphology_lines,  # 가로/세로 선 커널 → wavy 소자 제거 (主력).
    "projection": detect_projection,     # 행/열 합 peak → 축정렬 벽 (no_island·faint).
    "hough": detect_hough,               # Canny+HoughLinesP → 최외곽 선 사각형 (occlusion robust).
}

ENABLED_DETECTORS = [
    "photometric",
    "adaptive",
    "morphology",
    "projection",
    "hough",
]


# ------------------------------------------------------------------
# 앙상블 — 모든 검출기 후보를 모아 통합 점수 best 채택.
# ------------------------------------------------------------------


def detect_white_box_ensemble_diagnose(gray: np.ndarray):
    """모든 검출기를 돌려 통합 점수 best box 를 채택. 실패 시 사유와 함께 반환.

    반환: (bbox | None, reason, per_source_best).
      reason  = 'ok:<source>(<score>)' 또는 'reject:best=<score>:<source>'.
      per_source_best = {검출기: 그 검출기 최고점수} — 어느 방법이 일하는지 진단.
    """
    best = None  # (score, bbox, source, parts)
    per_source = {}
    for name in ENABLED_DETECTORS:
        fn = DETECTOR_REGISTRY[name]
        try:
            bboxes = fn(gray)
        except Exception as exc:
            print(f"[WARNING] 검출기 {name} 실패: {exc}")
            bboxes = []
        src_best = 0.0
        for bb in bboxes:
            s, parts = score_box_quality(gray, bb)
            src_best = max(src_best, s)
            if best is None or s > best[0]:
                best = (s, bb, name, parts)
        per_source[name] = round(src_best, 3)

    if best is None or best[0] < ENSEMBLE_ACCEPT_SCORE:
        bs = round(best[0], 3) if best else 0.0
        src = best[2] if best else "none"
        return None, f"reject:best={bs}:{src}", per_source
    return best[1], f"ok:{best[2]}({best[0]:.2f})", per_source


def detect_white_box_ensemble(gray: np.ndarray):
    """앙상블 검출 — bbox 또는 None (drop-in for `_detect_white_box`)."""
    box, _reason, _src = detect_white_box_ensemble_diagnose(gray)
    return box


# ------------------------------------------------------------------
# 합성 hard-case self-test (Mac 점검용 — 실데이터 없이 파이프라인/판별 확인).
# ------------------------------------------------------------------


def _synth_box_image(*, faint: bool = False, wavy: bool = False, touch: bool = False) -> tuple:
    """흰 box 1개를 담은 합성 rcp 를 만든다. 반환: (gray, gt_bbox).

    faint=stroke 를 어둡게(저대비), wavy=배경에 구불구불 소자 라인(distractor) 추가,
    touch=box 를 가장자리 가깝게(예외 케이스). 검출기 강건성 회귀 가드용.
    """
    H, W = 320, 360
    bg = np.full((H, W), 70, np.uint8)
    rng = np.random.RandomState(7)
    bg = np.clip(bg.astype(np.int32) + rng.randint(-12, 12, (H, W)), 0, 255).astype(np.uint8)
    if wavy:
        for k in range(6):
            y = 40 + k * 40
            pts = np.array([[x, int(y + 12 * np.sin(x / 18.0 + k))] for x in range(10, W - 10, 3)])
            cv2.polylines(bg, [pts], False, 150, 1)
    bx, by, bw, bh = 120, 100, 120, 110
    if touch:
        bx, by = 4, 4
    stroke = 150 if faint else 255
    cv2.rectangle(bg, (bx, by), (bx + bw, by + bh), int(stroke), 2)
    # box 안쪽에 unique 패턴(저밝기) — 내부가 비어있지 않게.
    cv2.circle(bg, (bx + bw // 2, by + bh // 2), 18, 120, 1)
    return bg, (bx, by, bw, bh)


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def _run_selftest() -> int:
    cases = [
        ("clean", dict()),
        ("faint", dict(faint=True)),
        ("wavy_distractor", dict(wavy=True)),
        ("faint+wavy", dict(faint=True, wavy=True)),
    ]
    print("[INFO] === white_box ensemble self-test (합성 hard cases) ===")
    ok = 0
    for name, kw in cases:
        gray, gt = _synth_box_image(**kw)
        box, reason, per_src = detect_white_box_ensemble_diagnose(gray)
        iou = _iou(box, gt) if box is not None else 0.0
        hit = box is not None and iou >= 0.5
        ok += int(hit)
        srcs = " ".join(f"{k}={v}" for k, v in per_src.items())
        print(f"  {name:<16} {'HIT' if hit else 'MISS':<4} iou={iou:.2f} reason={reason}")
        print(f"      per-detector best: {srcs}")
    # no-box: 빈 배경은 None 이어야(거짓양성 0).
    blank = np.full((300, 320), 70, np.uint8)
    nb, nb_reason, _ = detect_white_box_ensemble_diagnose(blank)
    no_fp = nb is None
    print(f"  {'no_box':<16} {'OK' if no_fp else 'FALSE+!':<4} reason={nb_reason}")
    total_ok = ok + int(no_fp)
    print(f"[INFO] 결과: {total_ok}/{len(cases) + 1} 통과 "
          f"(hard case {ok}/{len(cases)} + no-box 거짓양성 {'없음' if no_fp else '있음!'})")
    return 0 if total_ok == len(cases) + 1 else 1


if __name__ == "__main__":
    raise SystemExit(_run_selftest())
