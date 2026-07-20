"""후보 재정렬(rerank) 신호 — ensemble selection 위 2차 순위 교정. modality-aware.

workflow_2 registration verifier A/B(2026-07-20~21, 전체 67 recipe / 334점)에서 검증된
변경만 포팅한다. **modality 로 갈린다**(template.key_type):

  - **OM (prod_mind)**: sel(NCC selection) 순서 ⊕ mind 순서를 RRF 결합.
    검증: prod_mind d=+0.042 > prod(NCC 단독) d=+0.009, OM +0.021.
  - **SEM (route_sw)**: ecc **단독** 순위로 전환(결합 아님). 검증: route_sw 0.826 >
    route3(sel⊕mind⊕ecc 결합) 0.820 > prod_mind 0.817; SEM ecc 단독 0.775 >
    prod_mind SEM 0.759. RRF 로 섞으면 SEM 을 압도하는 ecc 가 희석되므로 전환이 낫다
    (한 심판이 특정 국면을 지배하면 결합 대신 전환). SEM p/r=22/8.

두 경로 공통: **이득은 전부 순위 교정에서 나온다**(golden GT 정수 격자상 sub-pixel 축
무효 — route_sw raw==ref 로 실증). 그래서 rerank 는 **순위만** 쓰고 좌표는 언제나 기존
후보 중 하나(새 좌표 생성 금지). 전 후보 거부 시 기존 selection 그대로(안전 폴백).

원리:
  - **MIND**(Heinrich 2012 의 2-D 근사): 각 픽셀을 절대 밝기가 아니라 "자기 주변 8-이웃과
    닮은 방식"(self-similarity)으로 부호화. offset r 마다 Gaussian patch SSD(x, x+r)를
    채널 평균 분산 V 로 exp(-ssd/V) 정규화 → contrast/밝기 drift 에 불변에 가까움.
  - **ECC**(findTransformECC, translation): 밝기 기반 국소 정합의 상관계수(cc)로 재채점.
    SEM(junction 구조·저텍스처)에서 NCC/mind 보다 순위를 잘 잡는다(실측). cc 만 순위에
    쓰고 shift 는 버린다(sub-pixel 무효).

원본: poc/workflow_2/registration_lab.py (extract_candidate_crop / _mind_descriptor /
_zncc / ecc_refine / rerank_by_verdicts / rrf_fuse_orders). 상수·게이트 동일하게 유지할 것
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

# ECC 상수 (registration_lab 와 동일 — cc 만 순위에 쓴다).
ECC_MOTION = cv2.MOTION_TRANSLATION
ECC_ITERS = 100
ECC_EPS = 1e-5
ECC_GAUSS_FILT = 5
ECC_MIN_CC = 0.05        # 수렴해도 cc 가 이 미만이면 신뢰 불가로 거부.


def _flag_on(name: str) -> bool:
    val = os.getenv(name, "1").strip().lower()
    return val not in ("0", "false", "off")


def mind_rerank_enabled() -> bool:
    """kill switch — ALIGN_FAIL_MIND_RERANK=0 이면 비활성(기본 활성).

    call-time 에 읽으므로 프로세스 재시작 없이 테스트/운영에서 토글 가능하다.
    """
    return _flag_on("ALIGN_FAIL_MIND_RERANK")


def ecc_rerank_enabled() -> bool:
    """kill switch — ALIGN_FAIL_ECC_RERANK=0 이면 SEM 경로에서 ecc 비활성(기본 활성)."""
    return _flag_on("ALIGN_FAIL_ECC_RERANK")


def is_sem_template(template) -> bool:
    """template 이 SEM modality 인가 — route_sw(SEM=ecc 단독) 분기 신호.

    key_type=='sem' 만 SEM 경로. om/box/checker/None 은 OM 경로(=현행 prod_mind, 안전 기본).
    """
    return str(getattr(template, "key_type", "") or "").strip().lower() == "sem"


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


def ecc_score(tpl_gray, crop_gray):
    """template vs 후보 crop 의 ECC 상관계수(cc). (cc | None, reject_reason).

    findTransformECC(translation)로 수렴 cc 만 취한다 — shift 는 순위 rerank 에 불필요
    (sub-pixel 무효, route_sw raw==ref). registration_lab.ecc_refine 의 점수부와 동일.
    """
    tpl_f = tpl_gray.astype(np.float32) / 255.0
    crop_f = crop_gray.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    try:
        cc, _warp = cv2.findTransformECC(
            tpl_f, crop_f, warp, ECC_MOTION, criteria, None, ECC_GAUSS_FILT)
    except cv2.error:
        return None, "ecc_no_converge"
    if not np.isfinite(cc) or cc < ECC_MIN_CC:
        return None, "low_score"
    return float(cc), None


def ecc_rerank_order(tpl_gray, frame_gray, cand_xy_scales):
    """후보들의 ECC cc 재정렬 순열(index 리스트). 유효 점수 없으면 None(폴백 신호).

    mind_rerank_order 와 동일 규약(거부 후보는 baseline 순서로 뒤에). SEM 경로 전용.
    """
    th, tw = tpl_gray.shape[:2]
    scores = []
    for xy, scale in cand_xy_scales:
        crop, _ratio = extract_candidate_crop(frame_gray, xy, (tw, th), scale)
        if crop is None:
            scores.append(None)
            continue
        s, _reason = ecc_score(tpl_gray, crop)
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
