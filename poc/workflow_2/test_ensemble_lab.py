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
