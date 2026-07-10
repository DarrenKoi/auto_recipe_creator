"""top-K 후보 국소 registration verifier 실험 모듈 (lab-only, production 무수정).

`docs/study/cv/align_fail_cv_methods_research_ko.md` 의 P0-A(ECC) / P0-B(SIFT·AKAZE+RANSAC) /
P1-C(phase correlation) 를 벤치에서 A/B 하기 위한 부품 모음이다. 세 arm 모두 **verifier** 이지
proposer 가 아니다: 입력은 기존 consensus/ensemble proposer 가 낸 top-K 후보의
`(cand_xy, scale)` 이고, 출력은 후보별 재정렬 점수 + 정밀화된 template-center 좌표다.
전역 탐색을 하지 않으며, 새 좌표를 만들지 않는다(후보 crop 내부의 국소 보정만).

좌표 계약 (엔진과 동일):
  - cand_xy = 매칭된 template 중심(frame px). refined_xy 도 같은 의미.
  - align point = refined_xy + align_offset_xy * scale (offset 보정은 호출측 책임 —
    `align_similarity._gt_in_topk` 와 동일 규약).
  - crop 은 후보 창(template 크기 * scale)을 template 크기로 리사이즈해 비교하고,
    arm 이 낸 shift(리사이즈 px)는 실측 비율(ratio)로 frame px 에 되돌린다.

reject gate (모든 arm 공통):
  - 후보 창이 frame 을 벗어나면 crop_oob (baseline `_matched_crop` 은 클램프하지만
    registration 은 기하 왜곡이 치명적이라 거부한다).
  - |shift| 가 REG_MAX_SHIFT_FRAC * template 단변을 넘으면 shift_oob (후보 margin 밖).
  - arm 별 물리 게이트(ECC 수렴/최소 cc, phase 최소 response, feature 는 inlier 수·비율·
    spatial coverage·회전/스케일 범위). 거부된 후보는 baseline 순위를 유지한 채 뒤로 간다.

사용처: `golden_registration_eval_cond.py` (LOO harness 의 per-point hook 에서 호출).
live 경로(`workflow_3`)에는 어떤 import 도 넣지 않는다 — offline acceptance 전 금지.

실행 (합성 self-test, 데이터 불필요 — Mac/dev PC OK):
    uv run python poc/workflow_2/registration_lab.py
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# --- 공통 gate 상수 (오피스 A/B 후 보정 대상; env/CLI 없음, 코드 상수만) -----------------
REG_MAX_SHIFT_FRAC = 0.35    # |shift| <= 이 비율 * template 단변(리사이즈 px). 넘으면 후보 margin 밖.
REG_ARM_NAMES = ("ecc", "phase", "sift", "akaze")

# --- ECC (P0-A): intensity 기반 국소 정합. translation 모델부터(문서 권장 순서). ----------
ECC_MOTION = cv2.MOTION_TRANSLATION   # Euclidean/affine 확장은 translation 효과 확인 후.
ECC_ITERS = 100
ECC_EPS = 1e-5
ECC_GAUSS_FILT = 5                    # findTransformECC gaussFiltSize (노이즈 완화).
ECC_MIN_CC = 0.05                     # 수렴해도 cc 가 이 미만이면 신뢰 불가로 거부.

# --- phase correlation (P1-C): Fourier phase 의 sub-pixel translation. -------------------
PHASE_MIN_RESPONSE = 0.03             # single-peak 정도. 낮으면 다봉/노이즈 → 거부.

# --- SIFT/AKAZE + RANSAC limited-affine (P0-B) ------------------------------------------
FEAT_RATIO_TEST = 0.75                # Lowe ratio.
FEAT_MIN_MATCHES = 8                  # ratio 통과 매치 최소 수(RANSAC 입력).
FEAT_MIN_INLIERS = 6                  # limited-affine inlier 최소 수.
FEAT_MIN_INLIER_RATIO = 0.30
FEAT_REPROJ_TH = 3.0                  # RANSAC 재투영 임계(px).
FEAT_ROT_MAX_DEG = 15.0               # paused SEM FOV 에 물리적으로 허용되는 회전 상한.
FEAT_SCALE_LO = 0.75                  # 추정 uniform scale 허용 범위(후보 scale 은 이미 반영됨).
FEAT_SCALE_HI = 1.30
FEAT_COVER_GRID = 4                   # inlier spatial coverage: template 을 4x4 격자로.
FEAT_MIN_COVER_CELLS = 4              # 점유 격자 칸이 이 미만이면 한 조각 지지 → 거부(문서 §P0-B).


@dataclass
class RegVerdict:
    """후보 1개에 대한 verifier 판정.

    score 는 arm 내부 비교 전용(높을수록 좋음) — arm 간 절대 비교 금지(척도가 다름).
    refined_xy 는 frame px 의 template-center. ok=False 면 score/refined_xy 는 None.
    """

    arm: str
    ok: bool
    score: float | None = None
    refined_xy: tuple | None = None      # (x, y) frame px.
    shift_frame_px: float | None = None  # |refined - cand| (frame px).
    reject_reason: str | None = None
    runtime_ms: float = 0.0
    extra: dict = field(default_factory=dict)


# ====================================================================
# 후보 crop 추출 (공유 계약).
# ====================================================================


def extract_candidate_crop(frame_gray, cand_xy, tpl_wh, scale):
    """후보 창(template*scale, cand_xy 중심)을 떼어 template 크기로 리사이즈.

    반환 (crop_uint8, ratio_xy) 또는 (None, None). ratio_xy = (창폭/tw, 창높이/th) —
    리사이즈 px 의 shift 를 frame px 로 되돌리는 실측 비율(반올림 오차까지 반영).
    창이 frame 경계를 벗어나면 None (클램프 금지 — 기하 왜곡 방지).
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


# ====================================================================
# arm 구현 — 각 함수는 (ok, score, shift_xy, reject_reason, extra) 를 돌려준다.
# shift_xy 는 리사이즈(template) px: refined_center = cand_xy + shift * ratio.
# ====================================================================


def ecc_refine(tpl_gray, crop_gray):
    """ECC 국소 정합 (translation). cc 와 template-center 보정 shift 를 돌려준다.

    findTransformECC(template, input) 는 input(W(x)) ~= template(x) 인 W 를 찾으므로,
    crop 내용이 template 대비 d 만큼 밀려 있으면 W(x) = x - d 가 되고 shift = W(c) - c = -d
    = (truth - cand) 방향이다 (합성 self-test 로 검증되는 계약).
    """
    tpl_f = tpl_gray.astype(np.float32) / 255.0
    crop_f = crop_gray.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(
            tpl_f, crop_f, warp, ECC_MOTION, criteria, None, ECC_GAUSS_FILT)
    except cv2.error:
        return False, None, None, "ecc_no_converge", {}
    if not np.isfinite(cc) or cc < ECC_MIN_CC:
        return False, None, None, "low_score", {"cc": float(cc)}
    th, tw = tpl_gray.shape[:2]
    cx, cy = (tw - 1) / 2.0, (th - 1) / 2.0
    nx = warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2]
    ny = warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2]
    shift = (nx - cx, ny - cy)
    return True, float(cc), shift, None, {"cc": round(float(cc), 4)}


_HANN_CACHE: dict = {}   # {(w, h): window} — phaseCorrelate 용 Hann 창 재사용.


def _hann(w, h):
    win = _HANN_CACHE.get((w, h))
    if win is None:
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        _HANN_CACHE[(w, h)] = win
    return win


def phase_refine(tpl_gray, crop_gray):
    """phase correlation sub-pixel translation. response 를 점수로 쓴다.

    phaseCorrelate(src1=tpl, src2=crop) 는 crop = shift(tpl) 인 shift 를 돌려주고,
    crop(u) = tpl(u - shift) = tpl(u + d) 에서 shift = -d = (truth - cand) 방향
    (ECC 와 동일 계약; 합성 self-test 로 부호 고정).
    """
    th, tw = tpl_gray.shape[:2]
    tpl_f = tpl_gray.astype(np.float32)
    crop_f = crop_gray.astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(tpl_f, crop_f, _hann(tw, th))
    if not np.isfinite(response) or response < PHASE_MIN_RESPONSE:
        return False, None, None, "low_response", {"response": float(response)}
    return True, float(response), (float(dx), float(dy)), None, \
        {"response": round(float(response), 4)}


def _feature_detector(kind):
    if kind == "sift":
        return cv2.SIFT_create(), cv2.NORM_L2
    if kind == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"unknown feature kind: {kind}")


def _coverage_cells(pts, tpl_wh, grid=FEAT_COVER_GRID):
    """inlier template 점들이 점유한 격자 칸 수 — 한 stripe/모서리 몰림 검출(문서 §P0-B)."""
    tw, th = tpl_wh
    cells = set()
    for x, y in pts:
        gx = min(grid - 1, max(0, int(x / max(tw, 1) * grid)))
        gy = min(grid - 1, max(0, int(y / max(th, 1) * grid)))
        cells.add((gx, gy))
    return len(cells)


def feature_verify(tpl_gray, crop_gray, kind="sift"):
    """SIFT/AKAZE correspondence + RANSAC limited-affine (4-DoF) 기하 검증.

    점수 = inlier_count * inlier_ratio (둘 다에 단조). coverage/회전/스케일은 게이트로만.
    estimateAffinePartial2D 의 M 은 template -> crop 이므로 shift = M(center) - center
    (ECC 와 동일 방향 계약).
    """
    det, norm = _feature_detector(kind)
    kp1, des1 = det.detectAndCompute(tpl_gray, None)
    kp2, des2 = det.detectAndCompute(crop_gray, None)
    n_kp = {"n_kp_tpl": len(kp1 or ()), "n_kp_crop": len(kp2 or ())}
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return False, None, None, "few_keypoints", n_kp
    matcher = cv2.BFMatcher(norm)
    try:
        knn = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return False, None, None, "match_error", n_kp
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < FEAT_RATIO_TEST * n.distance]
    n_kp["n_matches"] = len(good)
    if len(good) < FEAT_MIN_MATCHES:
        return False, None, None, "few_matches", n_kp
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inlier_mask = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=FEAT_REPROJ_TH)
    if M is None or inlier_mask is None:
        return False, None, None, "no_transform", n_kp
    inl = inlier_mask.ravel().astype(bool)
    n_inl = int(inl.sum())
    ratio = n_inl / float(len(good))
    n_kp.update({"n_inliers": n_inl, "inlier_ratio": round(ratio, 3)})
    if n_inl < FEAT_MIN_INLIERS:
        return False, None, None, "few_inliers", n_kp
    if ratio < FEAT_MIN_INLIER_RATIO:
        return False, None, None, "low_inlier_ratio", n_kp
    est_scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot_deg = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    n_kp.update({"est_scale": round(est_scale, 3), "rot_deg": round(rot_deg, 2)})
    if not (FEAT_SCALE_LO <= est_scale <= FEAT_SCALE_HI) or abs(rot_deg) > FEAT_ROT_MAX_DEG:
        return False, None, None, "implausible_transform", n_kp
    th, tw = tpl_gray.shape[:2]
    cover = _coverage_cells(pts1[inl], (tw, th))
    n_kp["coverage_cells"] = cover
    if cover < FEAT_MIN_COVER_CELLS:
        return False, None, None, "low_coverage", n_kp
    proj = (pts1[inl] @ M[:, :2].T) + M[:, 2]
    n_kp["reproj_err"] = round(float(np.linalg.norm(proj - pts2[inl], axis=1).mean()), 3)
    cx, cy = (tw - 1) / 2.0, (th - 1) / 2.0
    nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    score = n_inl * ratio
    return True, float(score), (nx - cx, ny - cy), None, n_kp


_ARM_FUNCS = {
    "ecc": ecc_refine,
    "phase": phase_refine,
    "sift": lambda t, c: feature_verify(t, c, kind="sift"),
    "akaze": lambda t, c: feature_verify(t, c, kind="akaze"),
}


# ====================================================================
# 후보 1개 판정 + 후보 목록 재정렬.
# ====================================================================


def refine_candidate(arm, tpl_gray, frame_gray, cand_xy, scale):
    """후보 하나를 arm 으로 판정 — crop 추출 + arm 실행 + shift 게이트 + frame 재투영."""
    t0 = time.perf_counter()
    th, tw = tpl_gray.shape[:2]

    def _done(v):
        v.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return v

    fn = _ARM_FUNCS.get(arm)
    if fn is None:
        raise ValueError(f"unknown arm: {arm} (choose from {REG_ARM_NAMES})")
    crop, ratio = extract_candidate_crop(frame_gray, cand_xy, (tw, th), scale)
    if crop is None:
        return _done(RegVerdict(arm=arm, ok=False, reject_reason="crop_oob"))
    ok, score, shift, reason, extra = fn(tpl_gray, crop)
    if not ok:
        return _done(RegVerdict(arm=arm, ok=False, reject_reason=reason, extra=extra))
    max_shift = REG_MAX_SHIFT_FRAC * min(tw, th)
    if math.hypot(shift[0], shift[1]) > max_shift:
        extra["shift_resized_px"] = round(math.hypot(shift[0], shift[1]), 2)
        return _done(RegVerdict(arm=arm, ok=False, reject_reason="shift_oob", extra=extra))
    fx = float(cand_xy[0]) + shift[0] * ratio[0]
    fy = float(cand_xy[1]) + shift[1] * ratio[1]
    dist = math.hypot(fx - cand_xy[0], fy - cand_xy[1])
    return _done(RegVerdict(
        arm=arm, ok=True, score=score, refined_xy=(fx, fy),
        shift_frame_px=round(dist, 2), extra=extra))


def rerank_by_verdicts(verdicts):
    """verifier 점수로 후보 순서를 재정렬 — 거부 후보는 baseline 순서 그대로 뒤에.

    반환 (order, fallback). order = 후보 index 리스트(새 1순위가 앞). 유효 판정이
    하나도 없으면 fallback=True + baseline 순서(= 정확히 B0 동작으로 강등).
    """
    valid = [i for i, v in enumerate(verdicts) if v.ok]
    rejected = [i for i, v in enumerate(verdicts) if not v.ok]
    if not valid:
        return list(range(len(verdicts))), True
    order = sorted(valid, key=lambda i: -float(verdicts[i].score)) + rejected
    return order, False


# ====================================================================
# 합성 self-test — 데이터 없이 좌표/부호 계약과 게이트를 고정한다.
# ====================================================================

_TPL = 128          # self-test template 한 변.
_TOL_REFINE = 2.0   # refined center 가 truth 에 이 px 이내면 통과.


def _make_pattern(rng, size):
    """keypoint 가 풍부한 비주기 합성 패턴(사각/원/선 랜덤 배치 + 블러)."""
    img = np.full((size, size), 60, dtype=np.uint8)
    for _ in range(26):
        x, y = int(rng.integers(4, size - 22)), int(rng.integers(4, size - 22))
        w, h = int(rng.integers(6, 20)), int(rng.integers(6, 20))
        val = int(rng.integers(90, 250))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), val, -1)
        else:
            cv2.circle(img, (x + w // 2, y + h // 2), max(3, w // 2), val, -1)
    for _ in range(8):
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(img, p1, p2, int(rng.integers(120, 255)), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _make_scene(seed=7, pattern_scale=1.0):
    """합성 frame + template + truth 좌표 + decoy 좌표.

    frame 에는 truth 중심에 (pattern_scale 배율의) 패턴을, 다른 곳에 별개 decoy 패턴을
    붙인다. template 은 truth 패턴의 밝기/대비 변형판(등록-실측 drift 모사).
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(40, 70, size=(480, 640), dtype=np.uint8).copy()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    pattern = _make_pattern(rng, _TPL)
    decoy = _make_pattern(np.random.default_rng(seed + 999), _TPL)

    def _paste(img, patch, center):
        ph, pw = patch.shape[:2]
        x0, y0 = int(center[0] - pw // 2), int(center[1] - ph // 2)
        img[y0:y0 + ph, x0:x0 + pw] = patch

    truth = (200, 240)
    decoy_c = (480, 160)
    placed = pattern if pattern_scale == 1.0 else cv2.resize(
        pattern, (int(round(_TPL * pattern_scale)),) * 2, interpolation=cv2.INTER_AREA)
    _paste(frame, placed, truth)
    _paste(frame, decoy, decoy_c)
    # 등록-실측 drift 모사: 약한 대비/밝기 변화 + 미세 노이즈.
    tpl = np.clip(pattern.astype(np.float32) * 0.9 + 12.0, 0, 255).astype(np.uint8)
    noise = np.random.default_rng(seed + 1).normal(0, 2.0, tpl.shape)
    tpl = np.clip(tpl.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, tpl, truth, decoy_c


def _t_crop_geometry():
    frame, tpl, truth, _ = _make_scene()
    crop, ratio = extract_candidate_crop(frame, truth, (_TPL, _TPL), 1.0)
    assert crop is not None and crop.shape == (_TPL, _TPL)
    assert abs(ratio[0] - 1.0) < 1e-6 and abs(ratio[1] - 1.0) < 1e-6
    # 정확히 truth 에서 뜬 crop 은 (drift 전) 패턴과 강한 상관을 가져야 한다.
    ncc = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)[0, 0]
    assert ncc > 0.8, f"crop-template ncc={ncc:.3f}"
    # frame 밖으로 나가는 창은 거부(클램프 금지).
    oob, _ = extract_candidate_crop(frame, (10, 10), (_TPL, _TPL), 1.0)
    assert oob is None


def _t_arm_recovers_offset(arm):
    frame, tpl, truth, _ = _make_scene()
    cand = (truth[0] + 5, truth[1] - 3)   # 작은 위치 오차가 남은 후보.
    v = refine_candidate(arm, tpl, frame, cand, 1.0)
    assert v.ok, f"{arm}: reject {v.reject_reason} {v.extra}"
    err = math.hypot(v.refined_xy[0] - truth[0], v.refined_xy[1] - truth[1])
    assert err <= _TOL_REFINE, f"{arm}: refined err={err:.2f}px (cand err=5.8px)"


def _t_arm_prefers_truth_over_decoy(arm):
    frame, tpl, truth, decoy_c = _make_scene()
    v_true = refine_candidate(arm, tpl, frame, (truth[0] + 4, truth[1] + 2), 1.0)
    v_decoy = refine_candidate(arm, tpl, frame, decoy_c, 1.0)
    assert v_true.ok, f"{arm}: truth 후보 거부 {v_true.reject_reason}"
    assert (not v_decoy.ok) or (v_decoy.score < v_true.score), \
        f"{arm}: decoy 점수({v_decoy.score}) >= truth({v_true.score})"


def _t_scale_reprojection(arm):
    """frame 의 패턴이 0.8 배율일 때 후보 scale=0.8 로 crop -> shift 재투영이 truth 복원."""
    frame, tpl, truth, _ = _make_scene(seed=11, pattern_scale=0.8)
    cand = (truth[0] + 4, truth[1] + 4)
    v = refine_candidate(arm, tpl, frame, cand, 0.8)
    assert v.ok, f"{arm}(scale): reject {v.reject_reason} {v.extra}"
    err = math.hypot(v.refined_xy[0] - truth[0], v.refined_xy[1] - truth[1])
    assert err <= 2.5, f"{arm}(scale): refined err={err:.2f}px"


def _t_shift_gate():
    frame, tpl, truth, _ = _make_scene()
    far = (truth[0] + 70, truth[1] + 60)   # margin 밖 — ok 로 통과하면 안 된다.
    for arm in ("ecc", "phase"):
        v = refine_candidate(arm, tpl, frame, far, 1.0)
        assert not v.ok or v.shift_frame_px <= REG_MAX_SHIFT_FRAC * _TPL + 1, \
            f"{arm}: margin 밖 후보가 ok(shift={v.shift_frame_px})"


def _t_flat_crop_no_crash():
    tpl = _make_pattern(np.random.default_rng(3), _TPL)
    frame = np.full((400, 400), 128, dtype=np.uint8)   # 완전 평탄 — 정보 없음.
    for arm in REG_ARM_NAMES:
        v = refine_candidate(arm, tpl, frame, (200, 200), 1.0)
        assert not v.ok, f"{arm}: 평탄 crop 이 ok"


def _t_rerank():
    mk = lambda ok, score: RegVerdict(arm="ecc", ok=ok, score=score)
    order, fb = rerank_by_verdicts([mk(True, 0.2), mk(True, 0.9), mk(False, None)])
    assert order == [1, 0, 2] and not fb
    order, fb = rerank_by_verdicts([mk(False, None), mk(False, None)])
    assert order == [0, 1] and fb


def _run_selftests():
    tests = [("crop_geometry", _t_crop_geometry)]
    for arm in REG_ARM_NAMES:
        tests.append((f"{arm}_recovers_offset", lambda a=arm: _t_arm_recovers_offset(a)))
        tests.append((f"{arm}_truth_vs_decoy", lambda a=arm: _t_arm_prefers_truth_over_decoy(a)))
    for arm in ("ecc", "phase"):
        tests.append((f"{arm}_scale_reproject", lambda a=arm: _t_scale_reprojection(a)))
    tests += [("shift_gate", _t_shift_gate),
              ("flat_crop_no_crash", _t_flat_crop_no_crash),
              ("rerank", _t_rerank)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[INFO] [OK]   {name}")
        except Exception as exc:
            n_fail += 1
            print(f"[ERROR] [FAIL] {name}: {exc}")
    print(f"[INFO] registration_lab self-test: {len(tests) - n_fail}/{len(tests)} passed")
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _run_selftests() else 1)
