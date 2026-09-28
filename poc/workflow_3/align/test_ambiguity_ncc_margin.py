"""chamfer 2nd비가 모호해도 NCC 가 key 와 2nd 를 크게 가르면 act 하는 opt-in 게이트.

2026-09-29 오피스(consensus template): match score 0.720 인데 2nd비 0.981 > 0.98 로
engineer_review. 2nd 는 143px 옆(가로선을 따라 미끄러진 자리)이고 ncc key=0.612 / 2nd=-0.023.
넓고 낮은 key 는 가로선을 따라 chamfer 가 거의 안 떨어진다(aperture) - chamfer 2nd비만으로는
모호도를 못 잰다.
"""

import numpy as np

from poc.workflow_3.align.correction import key_visibility_gate
from poc.workflow_3.align.matching.engine import AlignKeyMatchResult


def _result(*, second_ratio=0.981, best_ncc=0.612, second_ncc=-0.023, decision="match"):
    return AlignKeyMatchResult(
        score=0.720, chamfer_score=0.829, orb_inlier_ratio=0.0, best_xy=(306, 215),
        best_scale=1.02, decision=decision, debug_overlay=np.zeros((4, 4, 3), np.uint8),
        distinctive=False, second_ratio=second_ratio, best_ncc=best_ncc, second_ncc=second_ncc,
    )


def _gate(result, ncc_margin):
    return key_visibility_gate(result, reregister_ratio_threshold=0.98, ncc_margin=ncc_margin)


def test_office_case_acts_when_ncc_separates_key_from_2nd():
    assert _gate(_result(), 0.3) == "act"


def test_off_by_default_keeps_engineer_review():
    assert _gate(_result(), None) == "engineer_review"


def test_small_ncc_margin_stays_ambiguous():
    # 진짜 닮은 이웃: 명암까지 비슷하다.
    assert _gate(_result(best_ncc=0.55, second_ncc=0.40), 0.3) == "engineer_review"


def test_inverted_polarity_fails_closed():
    # OM-D 극성 반전이면 key 자리의 ncc 가 음수다 - 차이가 커 보여도 act 하지 않는다.
    assert _gate(_result(best_ncc=-0.60, second_ncc=-0.95), 0.3) == "engineer_review"


def test_missing_ncc_stays_ambiguous():
    assert _gate(_result(best_ncc=None, second_ncc=None), 0.3) == "engineer_review"


def test_correction_acts_on_chamfer_ambiguous_key_when_ncc_margin_is_on():
    """합성 데모(2nd비 0.514, ncc key 0.868 / 2nd -0.023)를 임계 0.5 로 '모호' 로 만든다."""
    from poc.workflow_3.align.correction import (
        CorrectionConfig, _make_primary_demo, correct_align_fail,
    )
    from poc.workflow_3.align.test_correction import _FakeController

    def _run(margin):
        monitor, templates = _make_primary_demo(key_in_view=True)
        fake = _FakeController(monitor.capture(), monitor.capture_screen(), mode="SEM")
        outcome = correct_align_fail(
            fake, templates, ok_locator=lambda _s: (690, 560), dry_run=False,
            config=CorrectionConfig(reregister_ratio_threshold=0.5, fallback_search_enabled=False,
                                    ambiguity_ncc_margin=margin),
        )
        return outcome.status, len(fake.screen_clicks)

    assert _run(None) == ("escalated_ambiguous_key", 0)
    assert _run(0.3) == ("corrected", 1)
