"""레시피 align-key 이미지와 SEM monitor 프레임을 매칭하는 classical CV 엔진.

설계 근거: ``docs/search_align_key.md`` §3, §7.

핵심 아이디어 두 가지를 합성한다.

* Chamfer 매칭 (구조 수준): 두 이미지를 edge map 으로 변환한 뒤
  distance transform 위에서 sliding-window 평균 거리로 점수화.
  픽셀 동일성에 의존하지 않으므로 공정 drift 에 견고.
* ORB + RANSAC (특징 수준): keypoint descriptor 매칭으로 sub-pixel/회전/스케일
  민감한 검증을 추가.

합성 점수: ``score = 0.6 * chamfer + 0.4 * orb_inlier_ratio`` (§7.3).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------
# 모듈 상수 — §7.3 의 임계값과 §7.2 의 multi-scale fallback 설정.
# ------------------------------------------------------------------

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSSIAN_SIGMA = 1.0
CANNY_T_LOW = 60
CANNY_T_HIGH = 160

# Chamfer 평균 거리 → 0~1 점수 변환의 감쇠 상수.
# tau=10 이면 mean_dt=5px → score≈0.61, mean_dt=10px → score≈0.37.
DT_TAU_PX = 10.0

# 메타가 없을 때 (§7.2 Case B) 시도할 5단계 스케일.
DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)

# ORB 파라미터.
ORB_N_FEATURES = 2000
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESH = 5.0
MIN_LOWE_MATCHES = 8

# 합성 점수 가중치 — §7.3.
CHAMFER_WEIGHT = 0.6
ORB_WEIGHT = 0.4

# 결정 임계값 — §7.3.
MATCH_THRESHOLD = 0.75
ADJUST_THRESHOLD = 0.55


# ------------------------------------------------------------------
# §7.6 의 인터페이스 dataclass.
# ------------------------------------------------------------------


@dataclass
class AlignKeyTemplate:
    """서버에서 받은 align key + 전처리 결과를 보존."""

    recipe_id: str
    version: str
    raw_image: np.ndarray
    edge_map: np.ndarray
    distance_transform: np.ndarray
    nm_per_pixel: float | None
    key_type: str | None
    fetched_at: datetime


@dataclass
class AlignKeyMatchResult:
    """compute_align_key_score 결과 + 시각화 페이로드."""

    score: float
    chamfer_score: float
    orb_inlier_ratio: float
    best_xy: tuple[int, int]
    best_scale: float
    decision: str
    debug_overlay: np.ndarray


# ------------------------------------------------------------------
# 전처리.
# ------------------------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """다양한 입력을 grayscale uint8 으로 정규화한다."""
    if image is None:
        raise ValueError("image is None")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def preprocess_for_matching(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """grayscale → CLAHE → Gaussian blur → Canny → distance transform.

    반환: (edge_map, distance_transform).

    distance_transform 은 *edge 까지의 거리* 이므로, edge 가 있는 위치에서 0,
    feature-less 영역에서 큰 값을 가진다. Chamfer 점수 계산에서 그대로 사용.
    """
    gray = _to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, ksize=(0, 0), sigmaX=GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_T_LOW, CANNY_T_HIGH)

    # distanceTransform 은 0이 *edge* 가 되도록 입력해야 한다 (값=0인 픽셀까지의 거리).
    inverted = cv2.bitwise_not(edges)
    dt = cv2.distanceTransform(inverted, cv2.DIST_L2, maskSize=5)
    return edges, dt.astype(np.float32)


def build_template(
    raw_image: np.ndarray,
    *,
    recipe_id: str,
    version: str,
    nm_per_pixel: float | None = None,
    key_type: str | None = None,
) -> AlignKeyTemplate:
    """레시피 raw 이미지를 1회 전처리하여 AlignKeyTemplate 으로 묶는다."""
    gray = _to_grayscale(raw_image)
    edges, dt = preprocess_for_matching(gray)
    return AlignKeyTemplate(
        recipe_id=recipe_id,
        version=version,
        raw_image=gray,
        edge_map=edges,
        distance_transform=dt,
        nm_per_pixel=nm_per_pixel,
        key_type=key_type,
        fetched_at=datetime.now(),
    )


# ------------------------------------------------------------------
# Chamfer 매칭.
# ------------------------------------------------------------------


def _chamfer_score_at_scale(
    template_edges: np.ndarray,
    frame_dt: np.ndarray,
    scale: float,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """단일 스케일에서 최적 위치와 점수를 계산.

    구현: cv2.matchTemplate(frame_dt, template_mask, TM_CCORR) 결과는
    각 offset 에서 sum(frame_dt * template_mask) 이다. edge 픽셀 수로
    나누면 평균 거리가 되고, exp(-mean_dt / tau) 로 0~1 점수화한다.

    반환: (score, (cx, cy), (tw, th))  — cx, cy 는 매칭된 템플릿 중심.
    """
    # 스케일 적용한 binary edge mask 만들기.
    if abs(scale - 1.0) < 1e-6:
        edges_scaled = template_edges
    else:
        new_w = max(8, int(round(template_edges.shape[1] * scale)))
        new_h = max(8, int(round(template_edges.shape[0] * scale)))
        edges_scaled = cv2.resize(
            template_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

    th, tw = edges_scaled.shape[:2]
    fh, fw = frame_dt.shape[:2]
    if th >= fh or tw >= fw:
        # 템플릿이 프레임보다 크면 매칭 불가.
        return 0.0, (fw // 2, fh // 2), (tw, th)

    template_mask = (edges_scaled > 0).astype(np.float32)
    edge_count = float(template_mask.sum())
    if edge_count <= 0:
        return 0.0, (fw // 2, fh // 2), (tw, th)

    # CCORR: result[y, x] = sum over (i, j) of frame_dt[y+i, x+j] * template_mask[i, j].
    result = cv2.matchTemplate(frame_dt, template_mask, cv2.TM_CCORR)
    # 작을수록 좋다 (edge 까지 평균 거리).
    min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
    mean_dt = float(min_val) / edge_count
    score = float(np.exp(-mean_dt / DT_TAU_PX))

    cx = int(min_loc[0] + tw // 2)
    cy = int(min_loc[1] + th // 2)
    return score, (cx, cy), (tw, th)


def compute_chamfer_score(
    template: AlignKeyTemplate,
    frame_dt: np.ndarray,
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> tuple[float, tuple[int, int], float, tuple[int, int]]:
    """multi-scale Chamfer 매칭. (best_score, (cx, cy), best_scale, (tw, th))."""
    best_score = 0.0
    best_xy = (frame_dt.shape[1] // 2, frame_dt.shape[0] // 2)
    best_scale = 1.0
    best_tsize = template.edge_map.shape[1], template.edge_map.shape[0]
    for scale in scales:
        score, xy, tsize = _chamfer_score_at_scale(template.edge_map, frame_dt, scale)
        if score > best_score:
            best_score = score
            best_xy = xy
            best_scale = scale
            best_tsize = tsize
    return best_score, best_xy, best_scale, best_tsize


# ------------------------------------------------------------------
# ORB + RANSAC.
# ------------------------------------------------------------------


def compute_orb_inlier_ratio(
    template_image: np.ndarray,
    frame_image: np.ndarray,
    *,
    max_features: int = ORB_N_FEATURES,
) -> tuple[float, int, int]:
    """ORB descriptor 매칭 + RANSAC homography 로 inlier 비율 계산.

    반환: (inlier_ratio, num_inliers, num_lowe_matches).
    Lowe ratio 통과 매칭이 8개 미만이면 (0.0, 0, n) 반환.
    """
    template_gray = _to_grayscale(template_image)
    frame_gray = _to_grayscale(frame_image)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_t, des_t = orb.detectAndCompute(template_gray, None)
    kp_f, des_f = orb.detectAndCompute(frame_gray, None)

    if des_t is None or des_f is None or len(kp_t) < 4 or len(kp_f) < 4:
        return 0.0, 0, 0

    # ORB 는 binary descriptor → BFMatcher with NORM_HAMMING.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des_t, des_f, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < MIN_LOWE_MATCHES:
        return 0.0, 0, len(good)

    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH
    )
    if mask is None:
        return 0.0, 0, len(good)

    num_inliers = int(mask.sum())
    ratio = num_inliers / float(len(good))
    return float(ratio), num_inliers, len(good)


# ------------------------------------------------------------------
# 합성 점수 + 결정 + 시각화.
# ------------------------------------------------------------------


def _decision_for_score(score: float) -> str:
    if score >= MATCH_THRESHOLD:
        return "match"
    if score >= ADJUST_THRESHOLD:
        return "adjust"
    return "low"


def _color_for_decision(decision: str) -> tuple[int, int, int]:
    # BGR.
    if decision == "match":
        return (0, 200, 0)
    if decision == "adjust":
        return (0, 200, 220)
    return (0, 0, 220)


def _crop_with_padding(
    frame: np.ndarray,
    cx: int,
    cy: int,
    tw: int,
    th: int,
    *,
    pad: float = 1.5,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Chamfer 의 best_xy 주변을 ORB 입력용으로 잘라낸다.

    반환: (cropped, (origin_x, origin_y)).
    """
    fh, fw = frame.shape[:2]
    half_w = int(tw * pad / 2)
    half_h = int(th * pad / 2)
    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(fw, cx + half_w)
    y1 = min(fh, cy + half_h)
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


def _render_overlay(
    frame: np.ndarray,
    *,
    cx: int,
    cy: int,
    tw: int,
    th: int,
    decision: str,
    score: float,
    chamfer: float,
    orb: float,
    scale: float,
) -> np.ndarray:
    """프레임 위에 매칭 박스 + 점수 텍스트를 그린 BGR 이미지를 반환."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()
    color = _color_for_decision(decision)

    x0 = max(0, cx - tw // 2)
    y0 = max(0, cy - th // 2)
    x1 = min(canvas.shape[1] - 1, cx + tw // 2)
    y1 = min(canvas.shape[0] - 1, cy + th // 2)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

    label1 = f"{decision.upper()}  score={score:.3f}"
    label2 = f"chamfer={chamfer:.3f} orb={orb:.3f} scale={scale:.2f}"

    # 검정 배경 박스 + 흰 글자로 가독성 확보.
    for i, text in enumerate((label1, label2)):
        ty = 18 + i * 18
        (tw_px, th_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (4, ty - th_px - 2), (8 + tw_px, ty + 4), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            text,
            (6, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def compute_align_key_score(
    template: AlignKeyTemplate,
    frame: np.ndarray,
    *,
    frame_nm_per_pixel: float | None = None,
    roi_hint: tuple[int, int, int, int] | None = None,
) -> AlignKeyMatchResult:
    """매 SEM 프레임마다 호출되는 메인 매칭 함수.

    ``frame_nm_per_pixel`` 이 양쪽 모두 주어지면 §7.2 Case A 단일 스케일로,
    그렇지 않으면 §7.2 Case B 의 multi-scale fallback 으로 동작.
    ``roi_hint`` 는 (x, y, w, h) — VLM 의 직전 이동 제안 영역에 한정해서
    탐색하고 싶을 때 사용한다 (현재 prototype 에서는 frame 자체를 잘라서 사용).
    """
    gray_frame = _to_grayscale(frame)

    # 스케일 결정 — ROI 검증에서 최소 크롭 크기 산출에 필요하므로 먼저.
    if (
        template.nm_per_pixel is not None
        and frame_nm_per_pixel is not None
        and frame_nm_per_pixel > 0
    ):
        single_scale = template.nm_per_pixel / frame_nm_per_pixel
        if single_scale <= 0:
            raise ValueError(
                f"resolved scale must be positive, got {single_scale} "
                f"(template.nm_per_pixel={template.nm_per_pixel}, "
                f"frame_nm_per_pixel={frame_nm_per_pixel})"
            )
        scales: tuple[float, ...] = (float(single_scale),)
    else:
        scales = DEFAULT_SCALES

    # 가능한 최소 템플릿 크기 (모든 스케일 중 최소). Chamfer 가 매칭하려면
    # 프레임/크롭이 이보다는 커야 한다.
    th0, tw0 = template.edge_map.shape[:2]
    min_scale = min(scales)
    min_th = max(8, int(round(th0 * min_scale)))
    min_tw = max(8, int(round(tw0 * min_scale)))

    # ROI hint 가 있으면 그 영역만 잘라서 그 안에서 매칭한다.
    roi_origin = (0, 0)
    if roi_hint is not None:
        if not (isinstance(roi_hint, tuple) and len(roi_hint) == 4):
            raise ValueError(
                f"roi_hint must be a 4-tuple (x, y, w, h), got {roi_hint!r}"
            )
        rx, ry, rw, rh = (int(v) for v in roi_hint)
        if rw <= 0 or rh <= 0:
            raise ValueError(
                f"roi_hint width/height must be positive, got w={rw}, h={rh}"
            )
        fh, fw = gray_frame.shape[:2]
        x0 = max(0, rx)
        y0 = max(0, ry)
        x1 = min(fw, rx + rw)
        y1 = min(fh, ry + rh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"roi_hint {(rx, ry, rw, rh)} does not intersect frame "
                f"of size {(fw, fh)}"
            )
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_w <= min_tw or crop_h <= min_th:
            raise ValueError(
                f"roi_hint crop {(crop_w, crop_h)} is smaller than the "
                f"smallest scaled template {(min_tw, min_th)} "
                f"(min_scale={min_scale:.3f}); widen the ROI or skip the hint"
            )
        gray_frame = gray_frame[y0:y1, x0:x1].copy()
        roi_origin = (x0, y0)

    _frame_edges, frame_dt = preprocess_for_matching(gray_frame)

    chamfer_score, (cx, cy), best_scale, (tw, th) = compute_chamfer_score(
        template, frame_dt, scales=scales
    )

    # ORB: Chamfer 의 best 위치를 중심으로 한 윈도우 vs 템플릿.
    if chamfer_score > 0.0 and tw > 0 and th > 0:
        crop, _crop_origin = _crop_with_padding(gray_frame, cx, cy, tw, th, pad=1.6)
        orb_ratio, _n_inliers, _n_matches = compute_orb_inlier_ratio(
            template.raw_image, crop
        )
    else:
        orb_ratio = 0.0

    score = CHAMFER_WEIGHT * chamfer_score + ORB_WEIGHT * orb_ratio
    decision = _decision_for_score(score)

    abs_xy = (cx + roi_origin[0], cy + roi_origin[1])

    overlay = _render_overlay(
        frame,
        cx=abs_xy[0],
        cy=abs_xy[1],
        tw=tw,
        th=th,
        decision=decision,
        score=score,
        chamfer=chamfer_score,
        orb=orb_ratio,
        scale=best_scale,
    )

    return AlignKeyMatchResult(
        score=float(score),
        chamfer_score=float(chamfer_score),
        orb_inlier_ratio=float(orb_ratio),
        best_xy=abs_xy,
        best_scale=float(best_scale),
        decision=decision,
        debug_overlay=overlay,
    )


# ------------------------------------------------------------------
# 디버그 저장 헬퍼 (프로토타입 단계에서 직접 사용).
# ------------------------------------------------------------------


def save_overlay_jpeg(overlay_bgr: np.ndarray, out_path: Path) -> None:
    """compute_align_key_score 의 debug_overlay 를 JPEG 로 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
