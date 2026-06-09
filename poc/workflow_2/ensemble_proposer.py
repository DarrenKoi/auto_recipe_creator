"""다중 구조 채널 ensemble proposer (C1 Canny + C2 Scharr + C3 orientation-binned).

기존 multi-scale chamfer 엔진(align_key_matcher)을 edge map 만 바꿔 재사용한다.
RRF(순위 기반, 스케일 무관)로 채널 후보를 융합해 top-N + shadow pool 을 낸다.
설계: docs/specs/2026-06-09-ensemble-proposer-design.md.
"""
import cv2
import numpy as np

from poc.workflow_2.align_key_matcher import (
    DT_TAU_PX, _scaled_edges, _to_grayscale, preprocess_for_matching,
)

# C2: gradient magnitude foreground 밀도를 C1 에 맞춘다(3~15% clamp).
SCHARR_R_MIN = 0.03
SCHARR_R_MAX = 0.15


def _scharr_edges(image: np.ndarray, r_c1: float) -> np.ndarray:
    """Scharr gradient magnitude 를 C1 밀도 매칭 percentile 로 이진화한 edge map(uint8 0/255).

    threshold = (1 - r) 백분위, r = clamp(r_c1, 3%~15%). Otsu 대신 밀도 매칭 →
    C1(Canny) 과 foreground ratio 를 맞춰 채널별 mean_dt 스케일을 동등하게.

    타이 브레이킹: 균일 이미지(magnitude=0) 에서도 clamp 하한으로 edge 가 생기도록
    미세 jitter(< 1e-6)를 더해 percentile 계산. 재현성을 위해 고정 시드 사용.
    실데이터에선 float32 정밀도가 jitter 를 흡수해 무영향(magnitude≈0 영역에서만 작동);
    실구조 밀도 < clamp 하한일 때만 부족분이 노이즈 픽셀로 채워진다(실 SEM 프레임엔 비발생).
    """
    gray = _to_grayscale(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    # 동점(exact-zero) 타이 브레이킹 — 균일 이미지에서도 clamp 하한 밀도 보장
    rng = np.random.default_rng(0)
    mag_j = mag + rng.uniform(0, 1e-6, mag.shape).astype(np.float32)
    r = float(min(SCHARR_R_MAX, max(SCHARR_R_MIN, r_c1)))
    thr = float(np.percentile(mag_j, 100.0 * (1.0 - r)))
    edges = (mag_j > thr).astype(np.uint8) * 255
    return edges


N_ORIENT_BINS = 8


def _orientation_bin_edges(image, n_bins=N_ORIENT_BINS, r_c1=None):
    """edge 픽셀을 gradient 방향(0~180° half-angle) n_bins 로 나눈 binary map 리스트.

    edge 위치는 C2 와 동일 밀도 매칭(_scharr_edges). 각 edge 픽셀을 unsigned gradient
    각도(0~180)로 bin 분류 → bin 별 0/255 map. polarity 불변(SEM/OM 밝기 반전 강건).
    """
    gray = _to_grayscale(image)
    if r_c1 is None:
        canny = preprocess_for_matching(gray)[0]
        r_c1 = float((canny > 0).mean())
    edges = _scharr_edges(gray, r_c1) > 0
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0          # 0~180 half-angle.
    bin_idx = np.minimum((ang / (180.0 / n_bins)).astype(np.int32), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = (edges & (bin_idx == b)).astype(np.uint8) * 255
        out.append(m)
    return out


def _directional_chamfer_score_map(template_gray, frame_gray, *, scale, n_bins=N_ORIENT_BINS):
    """방향 분할 chamfer score map = exp(-weighted_mean_dt/DT_TAU_PX).

    bin 별: same-bin frame edge 의 DT 위에 same-bin template edge 를 슬라이드한 mean_dt.
    bin 들을 template bin edge-count 가중 평균(weighted mean) → exp. min/sum 대신 weighted
    mean: 한 방향만 맞아도 과대평가(min)·edge 많은 bin 지배(sum) 회피, 기존 평균거리 규약 일관.
    """
    t_bins = _orientation_bin_edges(template_gray, n_bins)
    f_bins = _orientation_bin_edges(frame_gray, n_bins)
    num = None        # Σ_bin (edge_count_b * mean_dt_map_b)
    den = 0.0         # Σ_bin edge_count_b
    out_size = None
    for tb, fb in zip(t_bins, f_bins):
        tb_s = _scaled_edges(tb, scale)
        th, tw = tb_s.shape[:2]
        fh, fw = fb.shape[:2]
        if th >= fh or tw >= fw:
            return None, (tw, th)
        mask = (tb_s > 0).astype(np.float32)
        cnt = float(mask.sum())
        if cnt <= 0:
            continue
        f_dt = cv2.distanceTransform(cv2.bitwise_not(fb), cv2.DIST_L2, 5).astype(np.float32)
        mean_dt = cv2.matchTemplate(f_dt, mask, cv2.TM_CCORR) / cnt
        num = mean_dt * cnt if num is None else num + mean_dt * cnt
        den += cnt
        out_size = (tw, th)
    if num is None or den <= 0:
        return None, (out_size or (0, 0))
    weighted_mean_dt = num / den
    return np.exp(-weighted_mean_dt / DT_TAU_PX).astype(np.float32), out_size
