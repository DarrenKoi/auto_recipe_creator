"""ensemble 개선 실험장 (workflow_2). production 엔진(workflow_3/vision)을 건드리지 않고
새 융합/주기성/[Phase2] 채널을 시험한다. 검증되면 workflow_3 으로 포팅한다.

drivers: golden_localization_eval_cond.py, golden_consensus_eval_cond.py.
재사용: 채널 solo 후보·primitive 는 workflow_3.ensemble_proposer 에서 import. 실험 부분
(rescore 대표 선택, template_periodicity, [Phase2] C4 'context' 채널)만 여기 포크한다.
기본 인자에서 compute_ensemble_candidates 는 workflow_3 ensemble 과 bit-parity.

실행 (인자 없음): uv run pytest poc/workflow_2/test_ensemble_lab.py
"""
import numpy as np

from poc.workflow_3.vision.ensemble_proposer import (
    EnsembleResult, RRF_K0, SHADOW_N, _Cand, _channel_solo_candidates,
)
from poc.workflow_3.vision.align_key_matcher import DEFAULT_SCALES, _to_grayscale


# Phase 1: template 내재 주기성(autocorrelation off-center peak). cold-start, 오피스 보정 예정.
PERIODICITY_EXCL_FRAC = 0.10   # zero-lag 제외 중심 반경 = min(h,w) 의 이 비율.
PERIODICITY_TAU = 0.5          # 이 값 초과면 template_periodic(=재등록 후보). 비교는 strict >. 합성 검증으로 선택.


def template_periodicity(template_gray):
    """template 의 자기상관 기반 모호성 점수 [0,1] — 원형 autocorrelation 의 off-center peak 높이.

    높을수록 "유일하게 localize 되는 지점이 없음" → align key 모호(재등록 후보). 원형 자기상관은
    *주기성*(grating/array)뿐 아니라 *대칭성*(반복/반사로 같은 모습이 또 나타남)도 감지한다 —
    둘 다 matcher 가 어느 지점인지 못 가리는 케이스라 함께 높게 나오는 것이 올바른 동작이다.
    0=유일/무특징. scale 무관(상대 비율). NCC-isolation 이 못 보는 모호성을 보강(Codex 리뷰 #1).
    """
    g = _to_grayscale(template_gray).astype(np.float32)
    g = g - g.mean()
    if g.std() < 1e-6:
        return 0.0                                   # 무특징 grey → 주기성 정의 안 됨.
    F = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))   # 2D 원형 자기상관(패딩 없음) — 주기성+대칭성 감지.
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    peak0 = ac[cy, cx]
    if peak0 <= 0:
        return 0.0
    ac = ac / peak0                                  # zero-lag = 1.0 정규화.
    r = max(1, int(PERIODICITY_EXCL_FRAC * min(h, w)))
    yy, xx = np.ogrid[:h, :w]
    outside = (yy - cy) ** 2 + (xx - cx) ** 2 > r * r
    if not outside.any():
        return 0.0
    return float(np.clip(ac[outside].max(), 0.0, 1.0))


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
                                channels=("canny", "scharr", "orient"),
                                top_n=8, shadow_n=SHADOW_N, k0=RRF_K0,
                                scales=DEFAULT_SCALES, rescore_fn=None):
    """lab ensemble — channel 선택 + rescore 대표. 기본 인자는 workflow_3
    compute_ensemble_candidates 와 bit-parity(parity 테스트로 고정).
    """
    th, tw = _to_grayscale(template_gray).shape[:2]
    short = max(1, min(tw, th))
    match_r = max(8, int(0.05 * short))
    solo = {ch: _channel_solo_candidates(template_gray, frame_gray, ch, scales=scales)
            for ch in channels}
    fused = rrf_fuse(list(solo.values()), k0=k0, match_radius=match_r,
                     top_n=shadow_n, rescore_fn=rescore_fn)
    return EnsembleResult(fused=fused, top_n_count=top_n, solo=solo)
