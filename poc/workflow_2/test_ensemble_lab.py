import numpy as np
import cv2

from poc.workflow_2 import ensemble_lab as lab
from poc.workflow_3.vision import ensemble_proposer as ep


def _sq(size=200, box=(70, 70, 60, 60), bg=110, edge=230):
    img = np.full((size, size), bg, np.uint8)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 2)
    return img


def test_lab_ensemble_parity_with_workflow3():
    # lab 기본 채널/무 rescore = workflow_3 production ensemble 과 bit-parity (실험 baseline).
    frame = _sq(240, (120, 90, 60, 60))
    tpl = _sq(80, (10, 10, 60, 60))
    a = ep.compute_ensemble_candidates(tpl, frame, top_n=8, shadow_n=24)
    # channels 순서가 solo.values() -> rrf_fuse 입력 순서를 결정하므로
    # workflow_3 하드코딩 순서와 일치시켜야 bit-parity 성립.
    b = lab.compute_ensemble_candidates(
        tpl, frame, channels=("canny", "scharr", "orient"), top_n=8, shadow_n=24
    )
    pa = [(c.xy, round(c.score, 6), c.scale) for c in a.fused]
    pb = [(c.xy, round(c.score, 6), c.scale) for c in b.fused]
    assert pa == pb


def test_lab_rrf_fuse_rescore_fn_overrides_raw_representative():
    # 한 클러스터 두 멤버: raw score 는 A(20,20) 우세지만 yardstick 은 B(22,22) 선호 → 대표 B.
    A = [ep._Cand(xy=(20, 20), score=9.0, scale=0.6)]
    B = [ep._Cand(xy=(22, 22), score=1.0, scale=1.2)]

    def rescore(xy, scale):
        return 100.0 if xy == (22, 22) else 0.0

    fused = lab.rrf_fuse([A, B], k0=10, match_radius=5, top_n=1, rescore_fn=rescore)
    assert fused[0].xy == (22, 22) and fused[0].scale == 1.2


def test_lab_rrf_fuse_default_representative_unchanged():
    # rescore_fn 없으면 raw score 최댓값 멤버가 대표 — workflow_3 동작 보존.
    A = [ep._Cand(xy=(20, 20), score=9.0, scale=0.6)]
    B = [ep._Cand(xy=(22, 22), score=1.0, scale=1.2)]
    fused = lab.rrf_fuse([A, B], k0=10, match_radius=5, top_n=1)
    assert fused[0].xy == (20, 20) and fused[0].scale == 0.6


def test_template_periodicity_high_on_grating():
    g = np.zeros((120, 120), np.uint8)
    g[:, ::20] = 255                      # 주기 20px 수직 줄무늬
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    assert lab.template_periodicity(g) > lab.PERIODICITY_TAU


def test_template_periodicity_high_on_contact_array():
    a = np.full((120, 120), 110, np.uint8)
    for yy in range(12, 120, 24):
        for xx in range(12, 120, 24):
            cv2.circle(a, (xx, yy), 5, 230, -1)   # 주기 24px dot 격자
    assert lab.template_periodicity(a) > lab.PERIODICITY_TAU


def test_template_periodicity_low_on_unique_blob():
    b = np.full((120, 120), 110, np.uint8)
    cv2.circle(b, (60, 60), 12, 230, -1)          # 단일 유일 블롭
    # 단일 유일 블롭은 ~0.36 (tau=0.5 와 여유 약 0.14; 오피스 보정 시 재확인).
    assert lab.template_periodicity(b) < lab.PERIODICITY_TAU


def test_template_periodicity_high_on_symmetric_pair():
    # 동일한 두 블롭(반사/병진 대칭) → 자기상관 off-center peak 높음. matcher 가 둘을 구분 못 하므로
    # "유일 위치 없음"으로 높게 나오는 것이 올바른 동작(주기성 아닌 대칭성도 모호성 신호).
    s = np.full((120, 120), 110, np.uint8)
    cv2.circle(s, (30, 30), 12, 230, -1)
    cv2.circle(s, (90, 90), 12, 230, -1)
    assert lab.template_periodicity(s) > lab.PERIODICITY_TAU


def test_template_periodicity_zero_on_flat():
    f = np.full((100, 100), 128, np.uint8)        # 무특징 grey
    assert lab.template_periodicity(f) == 0.0


def test_miss_predictor_strong_signal():
    # miss 가 일관되게 높은 score → 완전 분리: AUC 1.0, Youden J 1.0 (TPR 1·FPR 0).
    scores = [0.8, 0.9, 0.85, 0.95, 0.1, 0.2, 0.15, 0.05]
    missed = [True, True, True, True, False, False, False, False]
    r = lab.miss_predictor_stats(scores, missed)
    assert r["n"] == 8 and r["n_miss"] == 4 and r["n_hit"] == 4
    assert r["auc"] == 1.0
    assert r["mean_miss"] > r["mean_hit"]
    assert r["youden_j"] == 1.0 and r["tpr"] == 1.0 and r["fpr"] == 0.0


def test_miss_predictor_no_signal():
    # 두 그룹 분포 동일 → AUC 0.5, Youden J 0 (예측력 없음).
    scores = [0.1, 0.9, 0.1, 0.9]
    missed = [True, True, False, False]
    r = lab.miss_predictor_stats(scores, missed)
    assert r["auc"] == 0.5
    assert r["youden_j"] == 0.0


def test_miss_predictor_reverse_signal_auc_below_half():
    # miss 가 오히려 낮은 score → AUC < 0.5 (역상관).
    scores = [0.1, 0.05, 0.9, 0.8]
    missed = [True, True, False, False]
    r = lab.miss_predictor_stats(scores, missed)
    assert r["auc"] < 0.5


def test_miss_predictor_empty_miss_group_returns_none():
    # miss 없음 → 분류 정의 불가 → auc/best_tau None.
    r = lab.miss_predictor_stats([0.2, 0.3], [False, False])
    assert r["n_miss"] == 0
    assert r["auc"] is None and r["best_tau"] is None
