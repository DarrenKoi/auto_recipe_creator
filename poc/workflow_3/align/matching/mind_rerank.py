"""MIND self-similarity rerank — ensemble selection 위 2차 순위 교정 신호.

workflow_2 registration verifier A/B(2026-07-20, 전체 67 recipe / 334점)에서 검증된
변경만 포팅한다:

  - B0(RRF proposer 순서) 대비 mind 재정렬 +0.039 (om +0.035, sem +0.041; promote/regress 23/10)
  - 운영 selection(prod = NCC rerank, +0.009)과 직교: prod_mind(sel⊕mind RRF) +0.042 > prod,
    SEM 은 +0.026 → +0.057 로 두 배 이상
  - 이득은 전부 순위 교정에서 나옴(golden GT 정수 격자상 sub-pixel 축 무효) → score-only,
    shift 를 내지 않는다(refined = 후보 그대로)

원리(Heinrich 2012 MIND 의 2-D 근사): 각 픽셀을 절대 밝기가 아니라 "자기 주변 8-이웃과
닮은 방식"(self-similarity)으로 부호화한다. offset r 마다 Gaussian patch SSD(x, x+r)를
채널 평균 분산 V 로 exp(-ssd/V) 정규화 → contrast/밝기 drift 에 불변에 가까워, 공정 변화로
외형이 변형된 SEM align key 에서 픽셀 유사도(NCC)가 놓치는 순위 오류를 교정한다.

설계 규칙 준수: 좌표는 언제나 기존 후보 중 하나(새 좌표 생성 금지), 전 후보 거부 시
기존 selection 그대로(안전 폴백). 척도가 다른 sel/mind score 는 합산하지 않고
RRF(순위 기반)로 결합한다 — 벤치의 prod_mind 의사-arm 과 bit-parity.

원본: poc/workflow_2/registration_lab.py 의 extract_candidate_crop / _mind_descriptor /
_zncc / mind_verify / rerank_by_verdicts / rrf_fuse_orders. 상수·게이트 동일하게 유지할 것
(벤치 재검증 시 두 구현이 같은 답을 내야 A/B 가 성립한다).
"""

import os

import cv2
import numpy as np

# Heinrich 2012 MIND 의 2-D 근사 — 8-이웃 offset self-similarity 채널.
MIND_OFFSETS = ((-2, -2), (0, -2), (2, -2), (-2, 0),
                (2, 0), (-2, 2), (0, 2), (2, 2))
MIND_PATCH_SIGMA = 1.5   # Gaussian patch distance 의 sigma.
MIND_MIN_SCORE = 0.10    # 채널 평균 ZNCC 하한 — 미만이면 유사 구조 없음으로 거부.
MIND_TRIM = 3            # 경계 artifact(offset shift 복제 경계) 제외 폭(px).
RRF_K = 8                # 순위 결합 상수 — 벤치 FUSE_RRF_K 와 동일.


def mind_rerank_enabled() -> bool:
    """kill switch — ALIGN_FAIL_MIND_RERANK=0 이면 비활성(기본 활성).

    call-time 에 읽으므로 프로세스 재시작 없이 테스트/운영에서 토글 가능하다.
    """
    val = os.getenv("ALIGN_FAIL_MIND_RERANK", "1").strip().lower()
    return val not in ("0", "false", "off")


def extract_candidate_crop(frame_gray, cand_xy, tpl_wh, scale):
    """후보 창(template*scale, cand_xy 중심)을 떼어 template 크기로 리사이즈.

    반환 (crop_uint8, ratio_xy) 또는 (None, None). 창이 frame 경계를 벗어나면 None
    (클램프 금지 — 기하 왜곡 방지). registration_lab.extract_candidate_crop 과 동일.
    """
    tw, th = int(tpl_wh[0]), int(tpl_wh[1])
    sc = float(scale) if scale else 1.0
    cw = max(1, int(round(tw * sc)))
    ch = max(1, int(round(th * sc)))
    cx, cy = float(cand_xy[0]), float(cand_xy[1])
    x0 = int(round(cx - cw / 2.0))
    y0 = int(round(cy - ch / 2.0))
    fh, fw = frame_gray.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + cw > fw or y0 + ch > fh:
        return None, None
    crop = frame_gray[y0:y0 + ch, x0:x0 + cw]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None, None
    interp = cv2.INTER_AREA if (cw >= tw and ch >= th) else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (tw, th), interpolation=interp)
    return resized, (cw / float(tw), ch / float(th))


def _mind_descriptor(img_u8):
    """MIND-like 2-D self-similarity descriptor stack (채널 = MIND_OFFSETS).

    각 offset r 에 대해 Gaussian patch SSD(x, x+r) 를 계산하고, 전 채널 평균을
    local variance 추정 V 로 써서 exp(-ssd/V) 로 정규화한다(Heinrich 2012 의 2-D 근사).
    """
    f = img_u8.astype(np.float32) / 255.0
    h, w = f.shape[:2]
    ssds = []
    for ox, oy in MIND_OFFSETS:
        m = np.float32([[1, 0, ox], [0, 1, oy]])
        shifted = cv2.warpAffine(f, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        ssds.append(cv2.GaussianBlur((f - shifted) ** 2, (0, 0), MIND_PATCH_SIGMA))
    v = np.maximum(np.mean(ssds, axis=0), 1e-8)
    return [np.exp(-s / v) for s in ssds]


def _zncc(a, b):
    """zero-mean NCC. 어느 한쪽이 상수(평탄)면 None."""
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return None
    return float((a * b).sum() / denom)


def mind_score(tpl_gray, crop_gray):
    """template vs 후보 crop 의 MIND descriptor 유사도. (score | None, reject_reason)."""
    d1 = _mind_descriptor(tpl_gray)
    d2 = _mind_descriptor(crop_gray)
    t = MIND_TRIM
    nccs = []
    for c1, c2 in zip(d1, d2):
        val = _zncc(c1[t:-t, t:-t], c2[t:-t, t:-t])
        if val is None:
            return None, "flat"
        nccs.append(val)
    score = float(np.mean(nccs))
    if not np.isfinite(score) or score < MIND_MIN_SCORE:
        return None, "low_score"
    return score, None


def mind_rerank_order(tpl_gray, frame_gray, cand_xy_scales):
    """후보들의 MIND score 재정렬 순열(index 리스트). 유효 점수가 없으면 None(폴백 신호).

    거부 후보(crop 밖 / 평탄 / 저점수)는 baseline 순서 그대로 유효 후보 뒤에 붙는다 —
    벤치 rerank_by_verdicts 와 동일 규약.
    """
    th, tw = tpl_gray.shape[:2]
    scores = []
    for xy, scale in cand_xy_scales:
        crop, _ratio = extract_candidate_crop(frame_gray, xy, (tw, th), scale)
        if crop is None:
            scores.append(None)
            continue
        s, _reason = mind_score(tpl_gray, crop)
        scores.append(s)
    valid = [i for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None
    rejected = [i for i, s in enumerate(scores) if s is None]
    return sorted(valid, key=lambda i: -scores[i]) + rejected


def rrf_fuse_orders(orders, n_cand, k=RRF_K):
    """순열들의 Reciprocal Rank Fusion. 동점은 낮은 index(=baseline 순위) 우선.

    sel(chamfer+NCC)과 mind score 는 척도가 달라 직접 합산할 수 없으므로 순위로만
    결합한다 — 벤치 rrf_fuse_orders 와 동일.
    """
    sc = [0.0] * int(n_cand)
    for order in orders:
        for r, idx in enumerate(order):
            sc[idx] += 1.0 / (k + r + 1)
    return sorted(range(int(n_cand)), key=lambda i: (-sc[i], i))
