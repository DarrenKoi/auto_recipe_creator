"""ensemble 개선 실험장 (workflow_2). production 엔진(workflow_3/align)을 건드리지 않고
새 융합/주기성/[Phase2] 채널을 시험한다. 검증되면 workflow_3 으로 포팅한다.

drivers: golden_localization_eval_cond.py, golden_consensus_eval_cond.py.
재사용: 채널 solo 후보·primitive 는 workflow_3.ensemble_proposer 에서 import. 실험 부분
(rescore 대표 선택, template_periodicity, [Phase2] C4 'context' 채널)만 여기 포크한다.
기본 인자에서 compute_ensemble_candidates 는 workflow_3 ensemble 과 bit-parity.

실행 (인자 없음): uv run pytest poc/workflow_2/test_ensemble_lab.py
"""
import statistics

import cv2
import numpy as np

from poc.workflow_3.align.matching.ensemble import (
    EnsembleResult, RRF_K0, SHADOW_N, SOLO_TOP_K, _Cand, _channel_solo_candidates,
)
from poc.workflow_3.align.matching.engine import (
    DEFAULT_POLICY,
    DEFAULT_SCALES,
    _candidate_ncc,
    _extract_peaks,
    _finalize_match,
    _no_candidate_result,
    _prepare_match_inputs,
    _rescore_positions_to_candidates,
    _scaled_edges,
    _to_grayscale,
    preprocess_for_matching,
)


# Phase 1: template 내재 주기성(autocorrelation off-center peak). cold-start, 오피스 보정 예정.
PERIODICITY_EXCL_FRAC = 0.10      # zero-lag 제외 중심 반경 = min(h,w) 의 이 비율(=min_lag_frac 기본).
PERIODICITY_MAX_LAG_FRAC = 0.8    # 상한 lag 반경 = min(h,w) 의 이 비율. overlap→0 극단 lag 잡음 차단.
PERIODICITY_TAU = 0.5             # 이 값 초과면 template_periodic(=재등록 후보). 비교는 strict >. 합성 검증으로 선택.

# lab proposer 채널. 기본 3개는 workflow_3 production ensemble 과 bit-parity 를 유지한다.
LAB_DEFAULT_CHANNELS = ("canny", "scharr", "orient")
LAB_EDGE_NCC_CHANNEL = "edge_ncc"


def _norm_autocorr(g):
    """zero-mean g 의 *선형*(비순환) 자기상관 — overlap 정규화, 중심 lag0. 반환 (2h-1, 2w-1).

    순환 FFT 자기상관은 wrap-around 로 고립 특징/작은 window 에서 가짜 peak 를 만든다(decoy 는
    frame 상의 *선형* shift 라 순환이 아님). zero-pad 로 선형 상관을 구하고 각 lag 에서 겹친
    픽셀 수로 나눠(편향 제거) 멀리 떨어진 대칭 쌍도 제대로 본다. peak0 정규화는 호출부에서.
    """
    h, w = g.shape
    H, W = h + h, w + w
    Fg = np.fft.rfft2(g, s=(H, W))
    corr = np.fft.irfft2(Fg * np.conj(Fg), s=(H, W))           # 선형 자기상관(lag0 at [0,0]).
    ones = np.ones((h, w), dtype=np.float32)
    Fo = np.fft.rfft2(ones, s=(H, W))
    ovl = np.fft.irfft2(Fo * np.conj(Fo), s=(H, W))            # lag 별 겹친 픽셀 수.
    nac = corr / np.maximum(ovl, 1.0)                          # overlap 정규화(편향 제거).
    return np.fft.fftshift(nac)[1:H, 1:W]                      # lag0 → 중심(h-1, w-1).


def template_periodicity(template_gray, *, center_xy=None, win_frac=None,
                         min_lag_frac=PERIODICITY_EXCL_FRAC,
                         max_lag_frac=PERIODICITY_MAX_LAG_FRAC):
    """template 의 자기상관 기반 모호성 점수 [0,1] — 선형 autocorrelation 의 off-center peak 높이.

    높을수록 "유일하게 localize 되는 지점이 없음" → align key 모호(재등록 후보). 자기상관은
    *주기성*(grating/array)뿐 아니라 *대칭성*(반복/반사로 같은 모습이 또 나타남)도 감지한다 —
    둘 다 matcher 가 어느 지점인지 못 가리는 케이스라 함께 높게 나오는 것이 올바른 동작이다.
    0=유일/무특징. scale 무관(상대 비율). NCC-isolation 이 못 보는 모호성을 보강(Codex 리뷰 #1).

    Phase1-A 날카롭게(sharpen) — 전체 template·전체 lag·순환 자기상관은 무해한/가짜 자기유사성에
    희석돼 miss 예측력이 약했다(office 순환판 AUC 0.636). 세 가지로 실패 유발 조건에 집중한다:
      - 선형 overlap-정규화 자기상관(_norm_autocorr): wrap 가짜 peak 제거(decoy 는 선형 shift).
      - center_xy / win_frac: align point(center 템플릿은 기하 중심) 주변 정사각 window 로
        한정해 무관한 배경 자기유사성을 배제. win_frac=None 이면 전체 template.
      - min_lag_frac / max_lag_frac: zero-lag 부근(=tolerance 안이라 decoy 가 떠도 무해)과
        overlap→0 극단 lag(잡음)를 제외하는 환형(annulus) lag 영역만 본다.
    """
    g = _to_grayscale(template_gray)
    if win_frac is not None:                         # align point 주변 정사각 window 로 한정.
        h0, w0 = g.shape[:2]
        cx0 = (w0 // 2) if center_xy is None else int(center_xy[0])
        cy0 = (h0 // 2) if center_xy is None else int(center_xy[1])
        half = max(4, int(0.5 * win_frac * min(h0, w0)))
        g = g[max(0, cy0 - half):min(h0, cy0 + half),
              max(0, cx0 - half):min(w0, cx0 + half)]
    g = g.astype(np.float32)
    if g.size == 0:
        return 0.0
    g = g - g.mean()
    if g.std() < 1e-6:
        return 0.0                                   # 무특징 grey → 주기성 정의 안 됨.
    nac = _norm_autocorr(g)
    h, w = nac.shape
    cy, cx = h // 2, w // 2                           # = (orig_h-1, orig_w-1) = lag0.
    peak0 = nac[cy, cx]
    if peak0 <= 0:
        return 0.0
    nac = nac / peak0                                 # zero-lag = 1.0 정규화.
    short = min(g.shape)                              # lag 반경 척도 = (windowed) 짧은 변.
    r_in = max(1, int(min_lag_frac * short))
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    annulus = d2 > r_in * r_in                       # tolerance 안 lag 제외.
    if max_lag_frac is not None:
        r_out = max(r_in + 1, int(max_lag_frac * short))
        annulus &= d2 <= r_out * r_out               # overlap→0 극단 lag 제외.
    if not annulus.any():
        return 0.0
    return float(np.clip(nac[annulus].max(), 0.0, 1.0))


def peak_isolation_ratio(cands):
    """score 내림차순 후보의 peak 고립도(ambiguity) ratio = score_2nd / score_best ∈ [0,1].

    (B) match-time distinctiveness — periodicity(template 고유, AUC 천장 0.61)와 달리 *실제*
    template↔frame 매칭의 모호성을 직접 잰다. 0=top1 독주(distinctive·hit-like),
    1=top2 가 top1 동률(ambiguous·miss-like). 후보는 이미 NMS 로 공간 분리(_propose_topk)돼
    있어 2nd 는 non-local peak. <2개거나 best<=0 면 0.0(경쟁 peak 없음=모호 없음).
    flat score surface(반복 패턴, wrong_local_peak)에서 top1≈top2 → ratio↑ 로 실패를 직접 신호.
    정답 위치를 몰라도 계산되므로 production fail-time 모호 플래그로 그대로 쓸 수 있다(C4 검증판).
    """
    if len(cands) < 2:
        return 0.0
    s0 = float(cands[0].score)
    if s0 <= 0:
        return 0.0
    s1 = float(cands[1].score)
    return float(np.clip(s1 / s0, 0.0, 1.0))


PEAK_ISO_VARIANTS = ("ratio_top2", "ratio_median_rest", "count_near", "margin_abs", "ncc_ratio_top2")


def peak_isolation_variants(scores, *, ncc=None):
    """후보 점수에서 peak-isolation(ambiguity) 변형들 — 전부 '높을수록 miss-like'. 포팅 전 robust 비교용.

    scores: proposer 점수(내림차순; C1=chamfer, ensemble=RRF). ncc: 같은 후보 순서의 NCC 재점수(옵션).
    반환 dict(계산 불가 변형은 None):
      ratio_top2        = s1/s0            (현재 production primitive, scale-free)
      ratio_median_rest = median(s1..)/s0  (경쟁 peak 분포 전체 반영, outlier 강건)
      count_near        = #{si >= 0.95 s0} (top1 의 강한 경쟁자 수; 이산)
      margin_abs        = -(s0 - s1)       (절대 margin; scale-free 아님 → 보통 약함, 대조군)
      ncc_ratio_top2    = NCC 재정렬 top2/top1 (C4 production form; proposer 점수 무관 — drift 약함 검증)
    n==0 이면 전부 None. <2개면 ratio/count/margin=0.0(경쟁 peak 없음=distinctive).
    """
    out = {k: None for k in PEAK_ISO_VARIANTS}
    n = len(scores)
    if n == 0:
        return out
    s0 = float(scores[0])
    rest = [float(x) for x in scores[1:]]
    if n < 2 or s0 <= 0:
        out.update(ratio_top2=0.0, ratio_median_rest=0.0, count_near=0.0, margin_abs=0.0)
    else:
        out["ratio_top2"] = float(np.clip(rest[0] / s0, 0.0, 1.0))
        out["ratio_median_rest"] = float(np.clip(statistics.median(rest) / s0, 0.0, 1.0))
        out["count_near"] = float(sum(1 for x in rest if x >= 0.95 * s0))
        out["margin_abs"] = float(-(s0 - rest[0]))   # 높을수록(0 에 가까울수록) miss-like.
    if ncc is not None:
        nv = sorted((float(x) for x in ncc if x is not None), reverse=True)
        if len(nv) >= 2:
            out["ncc_ratio_top2"] = float(np.clip(nv[1] / nv[0], 0.0, 1.0)) if nv[0] > 0 else 1.0
        elif nv:
            out["ncc_ratio_top2"] = 0.0
    return out


def miss_predictor_stats(scores, missed):
    """predictor(score)가 miss(=True)를 예측하나 진단 (Phase1 보정: periodicity↔miss).

    scores[i]=예측값(periodicity 등), missed[i]=bool(정답이 top-k 밖). 같은 길이.
    반환 dict:
      n / n_miss / n_hit,
      mean_miss·mean_hit·median_miss·median_hit (그룹별 score; 빈 그룹은 None),
      auc = P(score_miss > score_hit) (동점 0.5; 0.5=무신호, 1.0=완전 분리) — Mann-Whitney,
      best_tau / youden_j / tpr / fpr : "score > tau 면 miss 예측" 의 Youden J=TPR-FPR 최대 tau.
    miss 또는 hit 그룹이 비면 분류가 정의되지 않으므로 auc/best_tau/youden_j/tpr/fpr = None.
    """
    miss_s = [s for s, m in zip(scores, missed) if m]
    hit_s = [s for s, m in zip(scores, missed) if not m]
    n_miss, n_hit = len(miss_s), len(hit_s)
    out = {
        "n": len(scores), "n_miss": n_miss, "n_hit": n_hit,
        "mean_miss": round(sum(miss_s) / n_miss, 4) if n_miss else None,
        "mean_hit": round(sum(hit_s) / n_hit, 4) if n_hit else None,
        "median_miss": round(statistics.median(miss_s), 4) if n_miss else None,
        "median_hit": round(statistics.median(hit_s), 4) if n_hit else None,
        "auc": None, "best_tau": None, "youden_j": None, "tpr": None, "fpr": None,
    }
    if not n_miss or not n_hit:
        return out
    # AUC = P(score_miss > score_hit), 동점은 0.5 (Mann-Whitney). 0.5=무신호.
    wins = 0.0
    for m in miss_s:
        for h in hit_s:
            wins += 1.0 if m > h else (0.5 if m == h else 0.0)
    out["auc"] = round(wins / (n_miss * n_hit), 4)
    # Youden tau sweep: "score > tau 면 miss" 의 J=TPR-FPR 최대. 후보 tau=고유 score 값.
    best = None  # (J, tau, tpr, fpr)
    for tau in sorted(set(scores)):
        tpr = sum(1 for s in miss_s if s > tau) / n_miss
        fpr = sum(1 for s in hit_s if s > tau) / n_hit
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, tau, tpr, fpr)
    out["youden_j"] = round(best[0], 4)
    out["best_tau"] = round(best[1], 4)
    out["tpr"] = round(best[2], 4)
    out["fpr"] = round(best[3], 4)
    return out


def parse_ensemble_channels(value=None, *, default=LAB_DEFAULT_CHANNELS):
    """env/상수 문자열을 lab ensemble 채널 tuple 로 정규화한다.

    빈 값이면 기본 3채널(``canny,scharr,orient``)을 돌려 bit-parity 를 보존한다.
    ``edge_ncc``/``edge-ncc``/``c4`` 는 opt-in C4 edge-only NCC proposer 로 매핑한다.
    알 수 없는 채널은 즉시 ``ValueError`` 를 내 eval 설정 오류를 조용히 숨기지 않는다.
    """
    if value is None or value == "":
        return tuple(default)
    if isinstance(value, str):
        raw = [p.strip().lower().replace("-", "_") for p in value.split(",")]
    else:
        raw = [str(p).strip().lower().replace("-", "_") for p in value]
    aliases = {
        "c1": "canny",
        "canny_dt": "canny",
        "c2": "scharr",
        "scharr_gradient": "scharr",
        "c3": "orient",
        "orientation": "orient",
        "directional": "orient",
        "c4": LAB_EDGE_NCC_CHANNEL,
        "edge": LAB_EDGE_NCC_CHANNEL,
        "edge_ncc": LAB_EDGE_NCC_CHANNEL,
        "ncc_edge": LAB_EDGE_NCC_CHANNEL,
    }
    allowed = set(LAB_DEFAULT_CHANNELS) | {LAB_EDGE_NCC_CHANNEL}
    out = []
    for item in raw:
        if not item:
            continue
        ch = aliases.get(item, item)
        if ch not in allowed:
            raise ValueError(f"unknown lab ensemble channel: {item!r}")
        if ch not in out:
            out.append(ch)
    return tuple(out or default)


def _edge_ncc_solo_candidates(template_gray, frame_gray, *, scales=DEFAULT_SCALES,
                              top_k=SOLO_TOP_K):
    """C4 edge-only NCC solo 후보.

    C1 과 같은 Canny edge 를 쓰되 distance transform 이 아니라 binary edge map 간
    ``TM_CCOEFF_NORMED`` peak 를 뽑는다. 같은 edge family 라서 도메인 drift 리스크가 낮고,
    Chamfer 와 달리 "edge 가 근처에만 있으면 OK"가 아니라 국소 edge 배치의 정규화 상관을 본다.
    기본 채널에는 들어가지 않으며 ``edge_ncc`` 를 명시했을 때만 실행된다.
    """
    t_edges, _ = preprocess_for_matching(_to_grayscale(template_gray))
    f_edges, _ = preprocess_for_matching(_to_grayscale(frame_gray))
    frame_edge = (f_edges > 0).astype(np.float32)
    fh, fw = frame_edge.shape[:2]
    collected = []
    for scale in scales:
        te = _scaled_edges(t_edges, scale)
        th, tw = te.shape[:2]
        if th >= fh or tw >= fw:
            continue
        tpl_edge = (te > 0).astype(np.float32)
        if float(tpl_edge.sum()) <= 0.0 or float(tpl_edge.std()) < 1e-6:
            continue
        score_map = cv2.matchTemplate(frame_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)
        score_map = np.nan_to_num(score_map, nan=-1.0, posinf=-1.0, neginf=-1.0)
        score_map = np.maximum(score_map, 0.0).astype(np.float32)
        if float(score_map.max()) <= 0.0:
            continue
        nms_r = max(4, int(min(tw, th) * 0.5))
        for s, cx, cy in _extract_peaks(
            score_map, tw, th, max_peaks=top_k, min_score=0.0, nms_radius=nms_r,
        ):
            collected.append(_Cand(xy=(cx, cy), score=float(s), scale=float(scale)))
    collected.sort(key=lambda c: c.score, reverse=True)
    return collected[:top_k]


def _channel_solo_candidates_lab(template_gray, frame_gray, channel, *,
                                 scales=DEFAULT_SCALES):
    """lab 채널 dispatch. C1/C2/C3 는 production primitive, C4 만 workflow_2 전용."""
    if channel in LAB_DEFAULT_CHANNELS:
        return _channel_solo_candidates(template_gray, frame_gray, channel, scales=scales)
    if channel == LAB_EDGE_NCC_CHANNEL:
        return _edge_ncc_solo_candidates(template_gray, frame_gray, scales=scales)
    raise ValueError(f"unknown lab ensemble channel: {channel!r}")


def _selection_gap(selection_scores):
    """selection score 내림차순 기준 best-second gap metadata 를 만든다."""
    vals = sorted((float(s) for s in selection_scores), reverse=True)
    if not vals:
        return None, None, None
    if len(vals) == 1:
        return None, None, vals[0]
    gap = vals[0] - vals[1]
    ratio = vals[1] / vals[0] if vals[0] > 0 else None
    return float(gap), (float(np.clip(ratio, 0.0, 1.0)) if ratio is not None else None), vals[0]


def rrf_fuse(channel_lists, *, k0=RRF_K0, match_radius=8, top_n=SHADOW_N, rescore_fn=None):
    """RRF 융합(포크). fused(c) = Σ_채널 1/(k0 + rank). 채널 간 후보는 center 거리 <=
    match_radius(Chebyshev) 면 동일 후보로 묶는다. 반환 list[_Cand](fused 내림차순, top_n).

    대표(xy/scale) 선택:
      - rescore_fn 없음(기본): 클러스터 멤버 중 raw score 최댓값 — workflow_3 _rrf_fuse 동작 보존.
      - rescore_fn 있음: 멤버 위치를 공통 yardstick rescore_fn(xy, scale)->float 로 재평가해 대표
        선택. 이종 채널(C4 NCC-isolation) 합류 시 raw score 비교 불가 문제 해소(Codex 리뷰 #3).
        rescore 실패(예외)/전무는 raw 대표 폴백.
    """
    clusters = []  # {"xy","score","scale","rrf","members":[(xy, score, scale)]}
    for ch_list in channel_lists:
        ranked = sorted(ch_list, key=lambda c: c.score, reverse=True)
        for rank, cand in enumerate(ranked, 1):
            hit = next((cl for cl in clusters
                        if abs(cl["xy"][0] - cand.xy[0]) <= match_radius
                        and abs(cl["xy"][1] - cand.xy[1]) <= match_radius), None)
            contrib = 1.0 / (k0 + rank)
            if hit is None:
                clusters.append({"xy": cand.xy, "score": cand.score, "scale": cand.scale,
                                 "rrf": contrib,
                                 "members": [(cand.xy, cand.score, cand.scale)]})
            else:
                hit["rrf"] += contrib
                hit["members"].append((cand.xy, cand.score, cand.scale))
                if cand.score > hit["score"]:           # raw 대표 추적(무 rescore/폴백 경로).
                    hit["xy"], hit["score"], hit["scale"] = cand.xy, cand.score, cand.scale
    if rescore_fn is not None:
        for cl in clusters:
            best = None  # (rescore, xy, scale)
            for (xy, _s, scale) in cl["members"]:
                try:
                    rs = float(rescore_fn(xy, scale))
                except Exception:
                    continue
                if best is None or rs > best[0]:
                    best = (rs, xy, scale)
            if best is not None:
                cl["xy"], cl["scale"] = best[1], best[2]   # 대표를 공통 yardstick 으로 교체.
    clusters.sort(key=lambda cl: cl["rrf"], reverse=True)
    return [_Cand(xy=cl["xy"], score=cl["rrf"], scale=cl["scale"]) for cl in clusters[:top_n]]


def compute_ensemble_candidates(template_gray, frame_gray, *,
                                channels=LAB_DEFAULT_CHANNELS,
                                top_n=8, shadow_n=SHADOW_N, k0=RRF_K0,
                                scales=DEFAULT_SCALES, rescore_fn=None):
    """lab ensemble — channel 선택 + rescore 대표. 기본 인자는 workflow_3
    compute_ensemble_candidates 와 bit-parity(parity 테스트로 고정).
    """
    channels = parse_ensemble_channels(channels)
    th, tw = _to_grayscale(template_gray).shape[:2]
    short = max(1, min(tw, th))
    match_r = max(8, int(0.05 * short))
    solo = {ch: _channel_solo_candidates_lab(template_gray, frame_gray, ch, scales=scales)
            for ch in channels}
    fused = rrf_fuse(list(solo.values()), k0=k0, match_radius=match_r,
                     top_n=shadow_n, rescore_fn=rescore_fn)
    return EnsembleResult(fused=fused, top_n_count=top_n, solo=solo)


def compute_align_key_score_lab(
    template,
    frame,
    *,
    frame_nm_per_pixel=None,
    roi_hint=None,
    scales=None,
    policy=DEFAULT_POLICY,
    channels=LAB_DEFAULT_CHANNELS,
):
    """workflow_2 전용 lab ensemble matcher.

    production ``compute_align_key_score_ensemble`` 과 같은 계약을 유지하되, proposer 채널을
    ``channels`` 로 바꿀 수 있게 한다. 기본 채널은 C1/C2/C3 이므로 결과 해석은 production
    ensemble 과 동일하고, ``edge_ncc`` 를 넣은 경우에만 C4 실험이 켜진다. 반환 객체에는
    기존 필드 외에 ``lab_selection_gap`` / ``lab_selection_second_ratio`` /
    ``lab_channels`` 속성을 덧붙여 absolute threshold 와 별도로 best-second gap 을 볼 수 있다.
    """
    channels = parse_ensemble_channels(channels)
    gray_frame, frame_dt, scales, roi_origin = _prepare_match_inputs(
        template,
        frame,
        frame_nm_per_pixel=frame_nm_per_pixel,
        roi_hint=roi_hint,
        scales=scales,
    )
    ens = compute_ensemble_candidates(
        template.raw_image, gray_frame, channels=channels, scales=scales, top_n=policy.top_n
    )
    positions = [(c.xy, c.scale) for c in ens.fused[:policy.top_n]]
    candidates = _rescore_positions_to_candidates(template, frame_dt, positions)
    if not candidates or all(c.chamfer_score <= 0.0 for c in candidates):
        result = _no_candidate_result(frame, frame_dt, template, roi_origin)
        result.lab_channels = channels
        result.lab_selection_gap = None
        result.lab_selection_second_ratio = None
        result.lab_selection_best = None
        return result

    best_cand = candidates[0]
    best_sel = -1.0e18
    selection_scores = []
    for cand in candidates:
        ncc = _candidate_ncc(template.raw_image, gray_frame, cand.xy, cand.scale)
        ncc_pos = max(0.0, ncc) if ncc is not None else 0.0
        sel = policy.rerank_chamfer_w * cand.chamfer_score + policy.rerank_ncc_w * ncc_pos
        selection_scores.append(sel)
        if sel > best_sel:
            best_sel = sel
            best_cand = cand

    gap, second_ratio, best_value = _selection_gap(selection_scores)
    candidates_sorted = sorted(candidates, key=lambda c: c.chamfer_score, reverse=True)
    result = _finalize_match(
        best_cand, candidates_sorted, frame, template, policy, roi_origin,
        chamfer_score=best_cand.chamfer_score, orb_ratio=0.0,
        score_override=best_sel,
        decision_thresholds=(policy.ensemble_match_threshold, policy.ensemble_adjust_threshold),
    )
    result.lab_channels = channels
    result.lab_selection_gap = gap
    result.lab_selection_second_ratio = second_ratio
    result.lab_selection_best = best_value
    return result
