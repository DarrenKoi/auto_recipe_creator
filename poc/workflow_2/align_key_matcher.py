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

from dataclasses import dataclass, field
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

# 저배율(broad) 탐색용 scale 범위. recipe template 은 등록 시 고배율로 캡처되므로,
# zoom-out 한 live FOV 안에서는 align key 가 "miniature" 로 보인다. 이를 잡으려면
# template 을 크게 축소(0.15~0.5)해서 매칭해야 한다. DEFAULT_SCALES 와 분리해
# 두어 기존 smoke test (scale 2.0 negative) 의 보장을 깨지 않는다.
BROAD_SCALES = (0.15, 0.22, 0.3, 0.4, 0.5)


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
class AlignKeyCandidate:
    """top-N 후보 한 개 — Chamfer score map 의 NMS peak.

    score 는 후보 생성 시점의 Chamfer 점수. rerank 단계(Phase 3)에서 mi/contour 를 채워
    combined 로 재정렬할 수 있다. xy 는 매칭된 *템플릿 중심* (roi-local; 호출부에서 origin 가산).
    """

    score: float
    chamfer_score: float
    xy: tuple[int, int]
    scale: float
    template_size: tuple[int, int]
    orb_inlier_ratio: float = 0.0
    mi_score: float | None = None
    contour_score: float | None = None
    chamfer_rank: int | None = None   # MI rerank 전 원 chamfer 순위(0=chamfer-best). 재정렬돼도 보존.


@dataclass
class AlignKeyMatchResult:
    """compute_align_key_score 결과 + 시각화 페이로드.

    candidates 이하 필드는 top-N/distinctiveness 도입(2026-05-29)으로 추가된 *옵션* 필드.
    기존 호출부 호환을 위해 기본값을 가지며, 기존 필드(best_xy/score/decision)의 의미는 불변.
    reject_reason 은 현재 *advisory* — decision 을 바꾸지 않고 호출부가 참고/재판정에 쓴다.
    """

    score: float
    chamfer_score: float
    orb_inlier_ratio: float
    best_xy: tuple[int, int]
    best_scale: float
    decision: str
    debug_overlay: np.ndarray
    candidates: list = field(default_factory=list)   # list[AlignKeyCandidate], score 내림차순.
    second_score: float | None = None                # 2nd-best 후보의 chamfer.
    score_gap: float | None = None                   # best.chamfer - second.chamfer.
    second_ratio: float | None = None                # second.chamfer / best.chamfer (1.0 에 가까울수록 모호).
    distinctive: bool = True                          # best 가 2nd 대비 충분히 유일한가.
    reject_reason: str | None = None                  # "not_distinctive" | "no_candidates" | None.


@dataclass(frozen=True)
class MatchPolicy:
    """합성 점수 가중치 + 결정 임계값 묶음.

    Align fail 은 *대개 live key 가 등록 이미지와 다르게 보이기 때문에* 발생한다.
    이 경우 픽셀/feature 동일성에 의존하는 ORB 보다, edge 구조에 견고한 Chamfer 에
    더 무게를 둔 ``STRUCTURE_POLICY`` 가 적합하다. 기본값 ``DEFAULT_POLICY`` 는
    기존 §7.3 임계값을 그대로 보존하여 합성 smoke test 의 보장을 유지한다.
    """

    chamfer_weight: float = CHAMFER_WEIGHT
    orb_weight: float = ORB_WEIGHT
    match_threshold: float = MATCH_THRESHOLD
    adjust_threshold: float = ADJUST_THRESHOLD
    # top-N / distinctiveness (2026-05-29). best 가 2nd 대비 충분히 유일해야 distinctive.
    top_n: int = 8                    # NMS 후보 최대 개수.
    min_distinct_gap: float = 0.04    # best.chamfer - second.chamfer 가 이 미만이면 모호.
    max_second_ratio: float = 0.94    # second/best 가 이보다 크면 모호 (Lowe-ratio 의 template 판).
    # MI reranker (2026-06-01, opt-in). True 면 chamfer top-N 후보 *집합*은 그대로 두고(멤버십 게이트),
    # 각 후보 crop 의 MI(template.raw_image) 로 *순서만* 재정렬해 best 를 고른다. chamfer 가 dense-edge
    # distractor 에 1등을 뺏기는 topk_not_rank1 을 회복. DEFAULT/STRUCTURE 는 False(=기존 동작 보존).
    # 진단 A/B(align_similarity.py)에서 rerank_rank1_lift≥+0.10 검증 후 활성화 예정.
    mi_rerank: bool = False


# 기본 정책 — 기존 동작과 동일 (smoke test 호환).
DEFAULT_POLICY = MatchPolicy()

# Drift 에 견고한 구조 위주 정책 — 정적 비교(step 3)와 live search(step 4~7)에서 사용.
# Chamfer 비중을 높이고 임계값을 낮춰, 외형이 달라진 key 도 후보로 끌어올린다.
# 실데이터 calibration 전까지의 cold-start 값.
STRUCTURE_POLICY = MatchPolicy(
    chamfer_weight=0.8,
    orb_weight=0.2,
    match_threshold=0.62,
    adjust_threshold=0.40,
)


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


def _scaled_edges(template_edges: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return template_edges
    new_w = max(8, int(round(template_edges.shape[1] * scale)))
    new_h = max(8, int(round(template_edges.shape[0] * scale)))
    return cv2.resize(template_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _chamfer_score_map_at_scale(
    template_edges: np.ndarray,
    frame_dt: np.ndarray,
    scale: float,
) -> tuple[np.ndarray | None, tuple[int, int]]:
    """단일 스케일의 *전체 score map* 을 반환 (top-1 만이 아니라 후보 추출용).

    score_map[y, x] = exp(-mean_dt / tau) — 템플릿 top-left 가 (x, y) 일 때의 0~1 점수.
    매칭 불가(템플릿이 프레임보다 큼 / edge 없음) → (None, (tw, th)).
    """
    edges_scaled = _scaled_edges(template_edges, scale)
    th, tw = edges_scaled.shape[:2]
    fh, fw = frame_dt.shape[:2]
    if th >= fh or tw >= fw:
        return None, (tw, th)
    template_mask = (edges_scaled > 0).astype(np.float32)
    edge_count = float(template_mask.sum())
    if edge_count <= 0:
        return None, (tw, th)
    # CCORR: result[y, x] = sum over (i, j) of frame_dt[y+i, x+j] * template_mask[i, j].
    result = cv2.matchTemplate(frame_dt, template_mask, cv2.TM_CCORR)
    mean_dt_map = result / edge_count           # 작을수록 좋다 (edge 까지 평균 거리).
    score_map = np.exp(-mean_dt_map / DT_TAU_PX)
    return score_map.astype(np.float32), (tw, th)


def _chamfer_score_at_scale(
    template_edges: np.ndarray,
    frame_dt: np.ndarray,
    scale: float,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """단일 스케일 최적 위치/점수 — _chamfer_score_map_at_scale 의 top-1 wrapper (기존 동작 보존).

    반환: (score, (cx, cy), (tw, th)) — cx, cy 는 매칭된 템플릿 중심.
    """
    score_map, (tw, th) = _chamfer_score_map_at_scale(template_edges, frame_dt, scale)
    fh, fw = frame_dt.shape[:2]
    if score_map is None:
        return 0.0, (fw // 2, fh // 2), (tw, th)
    idx = int(np.argmax(score_map))
    y, x = np.unravel_index(idx, score_map.shape)
    score = float(score_map[y, x])
    return score, (int(x) + tw // 2, int(y) + th // 2), (tw, th)


def _extract_peaks(
    score_map: np.ndarray,
    tw: int,
    th: int,
    *,
    max_peaks: int,
    min_score: float,
    nms_radius: int,
) -> list[tuple[float, int, int]]:
    """score_map 에서 NMS 로 local maxima 를 뽑는다 → [(score, cx, cy)] (center 좌표).

    global argmax → 기록 → 주변 nms_radius 억제 → 반복. K 가 작아 비용 무시 가능.
    """
    work = score_map.copy()
    peaks: list[tuple[float, int, int]] = []
    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        y, x = np.unravel_index(idx, work.shape)
        s = float(work[y, x])
        if s < min_score:
            break
        peaks.append((s, int(x) + tw // 2, int(y) + th // 2))
        y0 = max(0, y - nms_radius); y1 = min(work.shape[0], y + nms_radius + 1)
        x0 = max(0, x - nms_radius); x1 = min(work.shape[1], x + nms_radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return peaks


def compute_chamfer_candidates(
    template: AlignKeyTemplate,
    frame_dt: np.ndarray,
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    top_n: int = 8,
    nms_radius_ratio: float = 0.5,
    min_score: float = 0.0,
) -> list["AlignKeyCandidate"]:
    """multi-scale Chamfer score map → NMS top-N 후보 (chamfer 내림차순).

    각 스케일에서 peak 를 뽑고, 스케일 간에는 center 거리 기반 global NMS 로 병합한다
    (같은 물리 위치가 여러 스케일에서 중복 잡히는 것 방지, 더 높은 점수 유지).
    """
    collected: list[tuple[float, int, int, float, int, int]] = []  # (score, cx, cy, scale, tw, th)
    for scale in scales:
        score_map, (tw, th) = _chamfer_score_map_at_scale(template.edge_map, frame_dt, scale)
        if score_map is None:
            continue
        nms_r = max(4, int(min(tw, th) * nms_radius_ratio))
        for s, cx, cy in _extract_peaks(
            score_map, tw, th, max_peaks=top_n, min_score=min_score, nms_radius=nms_r,
        ):
            collected.append((s, cx, cy, scale, tw, th))

    collected.sort(key=lambda t: t[0], reverse=True)
    # global NMS (스케일 교차) — center 거리가 가까우면 더 높은 점수만 남긴다.
    kept: list[tuple[float, int, int, float, int, int]] = []
    for item in collected:
        s, cx, cy, scale, tw, th = item
        merge_r = max(4, int(min(tw, th) * nms_radius_ratio))
        if any((abs(cx - k[1]) <= merge_r and abs(cy - k[2]) <= merge_r) for k in kept):
            continue
        kept.append(item)
        if len(kept) >= top_n:
            break

    return [
        AlignKeyCandidate(
            score=float(s), chamfer_score=float(s), xy=(cx, cy),
            scale=float(scale), template_size=(tw, th),
        )
        for (s, cx, cy, scale, tw, th) in kept
    ]


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
# MI reranker (chamfer = 멤버십 게이트, MI = 순서).
# 진단 하네스(align_similarity.py)의 _mi / _matched_crop 과 *동일* 해야 office A/B 의
# rerank_rank1_lift 수치가 production 동작과 일치한다. 변경 시 양쪽을 함께 맞춘다.
# ------------------------------------------------------------------


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """두 동일 크기 gray crop 의 mutual information (밝기/대비 drift 에 강건)."""
    a = a.ravel()
    b = b.ravel()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / max(hist.sum(), 1.0)
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    nz = pab > 0
    pa_pb = pa[:, None] * pb[None, :]
    return float((pab[nz] * np.log(pab[nz] / pa_pb[nz])).sum())


def _match_crop_for_mi(
    frame: np.ndarray, center_xy: tuple[int, int], tw: int, th: int, scale: float
) -> np.ndarray | None:
    """best 위치/스케일에서 frame crop 을 떼어 template 크기(tw, th)로 리사이즈 (MI 입력용).

    footprint = (tw*scale, th*scale) — tw/th 는 template.raw_image 크기, scale 은 후보 스케일.
    """
    cw = max(1, int(round(tw * scale)))
    ch = max(1, int(round(th * scale)))
    cx, cy = center_xy
    x0 = max(0, int(cx - cw // 2))
    y0 = max(0, int(cy - ch // 2))
    x1 = min(frame.shape[1], x0 + cw)
    y1 = min(frame.shape[0], y0 + ch)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)


def rerank_candidates_by_mi(
    template: AlignKeyTemplate,
    frame_gray: np.ndarray,
    candidates: list,
) -> list:
    """chamfer 후보 *집합*(멤버십)은 그대로, 각 후보 crop 의 MI 로 *순서만* 재정렬해 새 list 반환.

    각 후보의 ``mi_score`` 를 in-place 로 채운다. ``frame_gray`` 와 ``candidate.xy`` 는 *같은*
    좌표계여야 한다(compute_align_key_score 에서는 roi-local, abs 변환 *전*에 호출).
    crop 실패 후보는 MI=-inf 로 가장 뒤로 보낸다.
    """
    th_raw, tw_raw = template.raw_image.shape[:2]
    for c in candidates:
        crop = _match_crop_for_mi(frame_gray, c.xy, tw_raw, th_raw, c.scale)
        c.mi_score = (
            _mutual_information(template.raw_image, crop) if crop is not None else float("-inf")
        )
    return sorted(
        candidates,
        key=lambda c: c.mi_score if c.mi_score is not None else float("-inf"),
        reverse=True,
    )


# ------------------------------------------------------------------
# 합성 점수 + 결정 + 시각화.
# ------------------------------------------------------------------


def _decision_for_score(score: float, policy: MatchPolicy = DEFAULT_POLICY) -> str:
    if score >= policy.match_threshold:
        return "match"
    if score >= policy.adjust_threshold:
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
    scales: tuple[float, ...] | None = None,
    policy: MatchPolicy = DEFAULT_POLICY,
) -> AlignKeyMatchResult:
    """매 SEM 프레임마다 호출되는 메인 매칭 함수.

    ``frame_nm_per_pixel`` 이 양쪽 모두 주어지면 §7.2 Case A 단일 스케일로,
    그렇지 않으면 §7.2 Case B 의 multi-scale fallback 으로 동작.
    ``roi_hint`` 는 (x, y, w, h) — VLM 의 직전 이동 제안 영역에 한정해서
    탐색하고 싶을 때 사용한다 (현재 prototype 에서는 frame 자체를 잘라서 사용).

    ``scales`` 를 명시하면 nm_per_pixel 기반 자동 결정을 무시하고 그 범위로만
    매칭한다 (예: broad 탐색의 ``BROAD_SCALES``, confirm 단계의 좁은 범위).
    ``policy`` 는 합성 점수 가중치/임계값. 기본은 기존 동작과 동일하며,
    drift 에 견고한 매칭이 필요하면 ``STRUCTURE_POLICY`` 를 넘긴다.
    """
    gray_frame = _to_grayscale(frame)

    # 스케일 결정 — ROI 검증에서 최소 크롭 크기 산출에 필요하므로 먼저.
    if scales is not None:
        if not scales:
            raise ValueError("scales override must be a non-empty tuple")
        if any(s <= 0 for s in scales):
            raise ValueError(f"all scales must be positive, got {scales}")
        scales = tuple(float(s) for s in scales)
    elif (
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
        scales = (float(single_scale),)
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

    # top-N 후보 (NMS). best 후보 = candidates[0] 로, 기존 compute_chamfer_score 의 top-1 과 동일.
    candidates = compute_chamfer_candidates(
        template, frame_dt, scales=scales, top_n=policy.top_n,
    )

    if not candidates:
        # 매칭 불가 — 기존 동작(중앙, 점수 0)을 보존하고 reject_reason 만 남긴다.
        fh, fw = frame_dt.shape[:2]
        center = (fw // 2 + roi_origin[0], fh // 2 + roi_origin[1])
        t_h, t_w = template.edge_map.shape[:2]
        overlay = _render_overlay(
            frame, cx=center[0], cy=center[1], tw=t_w, th=t_h,
            decision="low", score=0.0, chamfer=0.0, orb=0.0, scale=1.0,
        )
        return AlignKeyMatchResult(
            score=0.0, chamfer_score=0.0, orb_inlier_ratio=0.0,
            best_xy=center, best_scale=1.0, decision="low", debug_overlay=overlay,
            candidates=[], reject_reason="no_candidates", distinctive=False,
        )

    # distinctiveness 는 *chamfer 집합* 기준(= 가장 강한 chamfer peak 이 2nd 대비 유일한가). MI rerank 가
    # best 를 바꿔도 이 신호는 chamfer 모호성 자체라 reorder 와 독립이어야 일관적이다(음수 gap 방지).
    # 동시에 각 후보에 원 chamfer 순위(chamfer_rank)를 새겨 재정렬 후에도 보존한다.
    chamfer_sorted = sorted(candidates, key=lambda c: c.chamfer_score, reverse=True)
    for rank, c in enumerate(chamfer_sorted):
        c.chamfer_rank = rank
    ch_best = chamfer_sorted[0]
    ch_second = chamfer_sorted[1] if len(chamfer_sorted) > 1 else None
    second_score = float(ch_second.chamfer_score) if ch_second is not None else None
    score_gap = (
        float(ch_best.chamfer_score - ch_second.chamfer_score) if ch_second is not None else None
    )
    second_ratio = (
        float(ch_second.chamfer_score / ch_best.chamfer_score)
        if ch_second is not None and ch_best.chamfer_score > 0 else None
    )
    distinctive = True
    reject_reason: str | None = None
    if ch_second is not None:
        if (score_gap is not None and score_gap < policy.min_distinct_gap) or (
            second_ratio is not None and second_ratio > policy.max_second_ratio
        ):
            distinctive = False
            reject_reason = "not_distinctive"

    # MI rerank (opt-in): 멤버십 불변, *순서만*. best/ORB/score/decision/best_xy 가 MI-best 에서 파생.
    # roi-local 좌표(abs 변환 전)에서 재정렬해야 crop 좌표계가 gray_frame 과 일치한다.
    if policy.mi_rerank:
        candidates = rerank_candidates_by_mi(template, gray_frame, candidates)

    best = candidates[0]
    cx, cy = best.xy
    best_scale = best.scale
    tw, th = best.template_size
    chamfer_score = best.chamfer_score

    # ORB: best 위치를 중심으로 한 윈도우 vs 템플릿 (mi_rerank 시 best=MI-best 에 재실행).
    if chamfer_score > 0.0 and tw > 0 and th > 0:
        crop, _crop_origin = _crop_with_padding(gray_frame, cx, cy, tw, th, pad=1.6)
        orb_ratio, _n_inliers, _n_matches = compute_orb_inlier_ratio(
            template.raw_image, crop
        )
    else:
        orb_ratio = 0.0
    best.orb_inlier_ratio = float(orb_ratio)

    score = policy.chamfer_weight * chamfer_score + policy.orb_weight * orb_ratio
    decision = _decision_for_score(score, policy)

    # 후보 좌표를 roi 절대 좌표로 환산 (best_xy 의미는 기존과 동일).
    abs_xy = (cx + roi_origin[0], cy + roi_origin[1])
    for c in candidates:
        c.xy = (c.xy[0] + roi_origin[0], c.xy[1] + roi_origin[1])

    overlay = _render_overlay(
        frame, cx=abs_xy[0], cy=abs_xy[1], tw=tw, th=th,
        decision=decision, score=score, chamfer=chamfer_score, orb=orb_ratio, scale=best_scale,
    )

    return AlignKeyMatchResult(
        score=float(score),
        chamfer_score=float(chamfer_score),
        orb_inlier_ratio=float(orb_ratio),
        best_xy=abs_xy,
        best_scale=float(best_scale),
        decision=decision,
        debug_overlay=overlay,
        candidates=candidates,
        second_score=second_score,
        score_gap=score_gap,
        second_ratio=second_ratio,
        distinctive=distinctive,
        reject_reason=reject_reason,
    )


# ------------------------------------------------------------------
# 디버그 저장 헬퍼 (프로토타입 단계에서 직접 사용).
# ------------------------------------------------------------------


def save_overlay_jpeg(overlay_bgr: np.ndarray, out_path: Path) -> None:
    """compute_align_key_score 의 debug_overlay 를 JPEG 로 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
