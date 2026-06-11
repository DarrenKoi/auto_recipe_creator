# poc/workflow_3/align/consensus_cv.py
"""consensus 빌드용 순수 CV 프리미티브 — bench(workflow_2)에서 bit-parity 포팅.

median(_consensus)·선명도(_edge_density/_lap_var)·crop 기하(_matched_crop)·
co-registration(_align_to_ref/coregister_crops)은 검증된 bench 로직 그대로다
(재구현 금지 — 표류 시 bit-parity/검증 수치 +0.442 가 깨진다). bench 출처:
align_similarity.py(_consensus/_edge_density/_lap_var/_matched_crop),
golden_consensus_eval_cond.py(_align_to_ref/coregister_crops/상수).
"""

import cv2
import numpy as np

# co-registration 상수 (golden_consensus_eval_cond.py:92-93).
COREG_ITERS = 2                 # ref median 다듬으며 원본 재정렬(보간 누적 방지).
COREG_MAX_SHIFT_FRAC = 0.3      # 추정 shift 가 변의 이 비율 초과면 spurious → 정렬 생략.


def _consensus(crops: list) -> np.ndarray:
    """동일 크기 gray crop 들의 median 이미지 = '현재 align 영역의 대표 모습'."""
    stack = np.stack([c.astype(np.float32) for c in crops])
    return np.median(stack, axis=0).astype(np.uint8)


def _edge_density(gray: np.ndarray) -> float:
    """Canny(60,160) edge 픽셀 비율 — matcher 전처리와 동일 임계."""
    if gray is None or gray.size == 0:
        return 0.0
    e = cv2.Canny(gray, 60, 160)
    return float((e > 0).mean())


def _lap_var(gray: np.ndarray) -> float:
    """Laplacian 분산 — 선명도 지표. median consensus blur 여부 확인용."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _matched_crop(frame: np.ndarray, center_xy, tw: int, th: int, scale: float):
    """center 위치/스케일에서 crop 을 떼어 template 크기로 리사이즈. 너무 작으면 None."""
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


def _align_to_ref(img, ref):
    """img 를 ref 에 sub-pixel 평행이동 정렬(phase correlation). 과도 shift 면 원본 반환."""
    h, w = ref.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(img.astype(np.float32), ref.astype(np.float32), win)
    if abs(dx) > COREG_MAX_SHIFT_FRAC * w or abs(dy) > COREG_MAX_SHIFT_FRAC * h:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def coregister_crops(crops):
    """crop 들을 공통 reference(다듬어진 median)에 sub-pixel 정렬해 median 을 또렷하게.

    매 iter ref=median(현재 정렬본)으로 갱신하되, 정렬은 항상 *원본*에서 한 번만 적용해
    보간 blur 누적을 막는다. crop 2장 미만이면 그대로 반환.
    """
    if len(crops) < 2:
        return crops
    aligned = list(crops)
    for _ in range(COREG_ITERS):
        ref = np.median(np.stack([a.astype(np.float32) for a in aligned]), 0).astype(np.uint8)
        aligned = [_align_to_ref(c, ref) for c in crops]
    return aligned
