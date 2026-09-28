"""모호도(second_ratio)의 2nd 는 '다른 자리'여야 한다 - 같은 key 의 사본을 세지 않는다.

2026-09-28 오피스(manual_align_correction, OM): recipe/consensus 가 실 wafer 와 거의 같은데
decision=match score=0.874 인데도 2nd비=0.996 gap=0.004 -> engineer_review -> search-around ->
탐색의 accept 도 같은 게이트라 매 프레임 같은 비율로 거부 -> fallback_aborted.
ensemble 은 채널 간 병합 반경(짧은 변 5%)이 채널 내 NMS(50%)보다 훨씬 작아, 같은 key 가
다른 scale/채널로 수 px 어긋나 두 후보로 남는다.
"""

import cv2
import numpy as np

from poc.workflow_3.align.matching.engine import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score_ensemble,
)


def _key(sz=160):
    k = np.full((sz, sz), 90, np.uint8)
    cv2.rectangle(k, (20, 20), (70, 140), 200, -1)
    cv2.rectangle(k, (90, 30), (140, 60), 40, -1)
    cv2.line(k, (80, 90), (150, 150), 220, 6)
    cv2.circle(k, (115, 110), 18, 30, -1)
    return k


def _frame(positions, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.normal(100, 6, (768, 1024)).clip(0, 255).astype(np.uint8)
    for x, y in positions:
        frame[y:y + 160, x:x + 160] = _key()
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _match(frame):
    t = build_template(_key(), recipe_id="syn", version="v", key_type="om")
    return compute_align_key_score_ensemble(t, frame, policy=STRUCTURE_POLICY)


def test_unique_key_is_not_ambiguous():
    for seed in range(4):
        r = _match(_frame([(200 + 90 * seed, 150 + 60 * seed)], seed=seed))
        assert r.decision == "match"
        assert r.distinctive, f"seed={seed} second_ratio={r.second_ratio}"
        assert r.second_ratio < 0.9, f"seed={seed} second_ratio={r.second_ratio}"
        assert r.second_xy is not None
        assert r.second_ncc is not None and r.best_ncc > r.second_ncc


def test_real_lookalike_stays_ambiguous():
    r = _match(_frame([(150, 200), (650, 300)]))
    assert r.second_ratio > 0.98
    assert not r.distinctive
