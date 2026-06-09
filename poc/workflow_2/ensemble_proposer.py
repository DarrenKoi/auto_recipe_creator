"""다중 구조 채널 ensemble proposer (C1 Canny + C2 Scharr + C3 orientation-binned).

기존 multi-scale chamfer 엔진(align_key_matcher)을 edge map 만 바꿔 재사용한다.
RRF(순위 기반, 스케일 무관)로 채널 후보를 융합해 top-N + shadow pool 을 낸다.
설계: docs/specs/2026-06-09-ensemble-proposer-design.md.
"""
import cv2
import numpy as np

from poc.workflow_2.align_key_matcher import _to_grayscale

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
