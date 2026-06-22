"""consensus arm proposer 의 lab(edge_ncc) 라우팅 테스트.

목적: SEM recall_miss(진실이 후보 pool 에 아예 없음)는 consensus arm 의 proposer 문제다.
edge_ncc(C4) 레버는 그동안 rcp-only arm 에만 꽂혀 있었는데(그 arm 은 비어 있어 테스트 불가),
이제 _propose_topk(consensus arm 후보 생성)도 lab 경로를 타게 했다.

여기서는 *디스패치*(lab env 활성 시 lab ensemble 을 해당 채널로 호출하는가)만 검증한다 — lab
ensemble 의 CV 정확성은 test_ensemble_lab.py 에서 실동작으로 따로 검증한다. 후보 좌표 비교는
합성 프레임에서 chamfer 와 C4 가 같은 막대 위치를 우연히 반환해 비변별적이라, 채널 인자를
받는 spy 로 라우팅을 결정적으로 고정한다.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poc.workflow_2 import align_similarity as alsim
from poc.workflow_2 import ensemble_lab as el
from poc.workflow_3.align.matching.engine import build_template, preprocess_for_matching


def _bar_crop(size=64, x=30, width=4):
    img = np.zeros((size, size), np.uint8)
    img[:, x:x + width] = 255
    return img


def _textured_frame(size=256):
    img = np.full((size, size), 40, np.uint8)
    for x in range(20, size - 20, 40):
        img[:, x:x + 4] = 220
    return img


def _clear_lab_env(monkeypatch):
    for k in ("ALIGN_ENSEMBLE_LAB_MODE", "ALIGN_LAB_ENSEMBLE_CHANNELS", "ENSEMBLE_CHANNELS"):
        monkeypatch.delenv(k, raising=False)


class _FakeCand:
    def __init__(self):
        self.xy = (1, 2)
        self.score = 0.9
        self.scale = 1.0


def test_propose_topk_routes_to_lab_ensemble_with_resolved_channels(monkeypatch):
    """lab mode 활성 → _propose_topk 가 lab ensemble 을 lab_channels_from_env() 채널로 호출."""
    _clear_lab_env(monkeypatch)
    monkeypatch.setenv("ALIGN_ENSEMBLE_LAB_MODE", "edge_ncc")

    sentinel = [_FakeCand()]
    seen = {}

    def _spy(template_gray, frame_gray, *, channels, top_n, scales, **kw):
        seen["channels"] = tuple(channels)
        return el.EnsembleResult(fused=list(sentinel), top_n_count=top_n, solo={})

    monkeypatch.setattr(el, "compute_ensemble_candidates", _spy)

    tpl = build_template(_bar_crop(x=30), recipe_id="R", version="t", key_type="sem")
    got = alsim._propose_topk(tpl, _textured_frame(), None, scales=alsim.COMPARE_SCALES, topk=8)

    assert got == sentinel, "lab 활성인데 lab ensemble 후보를 반환하지 않음(라우팅 안 됨)"
    assert seen["channels"] == tuple(el.LAB_DEFAULT_CHANNELS) + (el.LAB_EDGE_NCC_CHANNEL,)


def test_propose_topk_does_not_call_lab_when_inactive(monkeypatch):
    """lab env 없음 → lab ensemble 호출 안 함(production/chamfer 경로 유지)."""
    _clear_lab_env(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("lab 비활성인데 lab proposer 가 호출됨")

    monkeypatch.setattr(el, "compute_ensemble_candidates", _boom)

    tpl = build_template(_bar_crop(x=30), recipe_id="R", version="t", key_type="sem")
    frame = _textured_frame()
    frame_dt = preprocess_for_matching(frame)[1]  # 비-lab(chamfer) 경로용(_gt_in_topk 와 동일).
    # 예외 없이 후보를 반환하면 lab 경로를 안 탄 것.
    out = alsim._propose_topk(tpl, frame, frame_dt, scales=alsim.COMPARE_SCALES, topk=8)
    assert isinstance(out, list)
