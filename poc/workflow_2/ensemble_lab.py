"""ensemble 개선 실험장 (workflow_2). production 엔진(workflow_3/vision)을 건드리지 않고
새 융합/주기성/[Phase2] 채널을 시험한다. 검증되면 workflow_3 으로 포팅한다.

drivers: golden_localization_eval_cond.py, golden_consensus_eval_cond.py.
재사용: 채널 solo 후보·primitive 는 workflow_3.ensemble_proposer 에서 import. 실험 부분
(rescore 대표 선택, template_periodicity, [Phase2] C4 'context' 채널)만 여기 포크한다.
기본 인자에서 compute_ensemble_candidates 는 workflow_3 ensemble 과 bit-parity.

실행 (인자 없음): uv run pytest poc/workflow_2/test_ensemble_lab.py
"""
from poc.workflow_3.vision.ensemble_proposer import (
    EnsembleResult, RRF_K0, SHADOW_N, _Cand, _channel_solo_candidates,
)
from poc.workflow_3.vision.align_key_matcher import DEFAULT_SCALES, _to_grayscale


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
