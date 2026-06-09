"""proposer recall@N / attribution 순수 헬퍼 테스트 — Mac 실행 가능."""
import poc.workflow_2.proposer_recall_ab as pab
from poc.workflow_2.ensemble_proposer import _Cand


def test_gt_rank_in_candidates_hits_within_tol():
    # GT=(100,100), offset=(0,0), short=200, tol=0.20 → 허용 40px. 2번째 후보가 hit.
    cands = [_Cand(xy=(180, 180), score=0.9), _Cand(xy=(115, 100), score=0.8)]
    rank = pab._gt_rank(cands, gt_xy=(100, 100), offset=(0, 0), short=200, tol=0.20)
    assert rank == 2


def test_gt_rank_none_when_all_far():
    cands = [_Cand(xy=(10, 10), score=0.9)]
    assert pab._gt_rank(cands, gt_xy=(200, 200), offset=(0, 0), short=200, tol=0.20) is None


def test_recall_at_counts_rank_within_n():
    ranks = [1, 3, None, 9, 16]
    # @8 → rank<=8 인 것: 1,3 → 2/5=0.4
    assert pab._recall_at(ranks, 8) == 0.4
    # @16 → 1,3,9,16 → 4/5=0.8
    assert pab._recall_at(ranks, 16) == 0.8


def test_offset_applied_before_compare():
    # offset (40,0) 적용 시 후보 (60,100)+offset=(100,100)=GT → hit.
    cands = [_Cand(xy=(60, 100), score=0.9)]
    assert pab._gt_rank(cands, gt_xy=(100, 100), offset=(40, 0), short=200, tol=0.20) == 1
