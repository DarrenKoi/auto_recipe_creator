"""_consensus_template_ab 의 modality 커버리지 회귀 테스트.

배경([[project_om_sem_positions_per_measurement]]): align 측정 1회당 OM 2장 / SEM 3장이라
dual-modality recipe 는 거의 항상 SEM 이 dominant 다. 기존 `_consensus_template_ab` 는
`Counter(mod).most_common(1)` 로 **dominant modality 한 종류만** 평가했기 때문에, dual recipe
에서 OM consensus 는 영영 측정되지 못했다(golden_combined 의 OM/SEM split 에서 OM consensus
arm 이 공백). 이 테스트는 두 modality 가 *각각* per_recipe row 로 나오는 것을 고정한다.

합성 crop + frame_loader 주입(함수의 설계된 파라미터)으로 Mac 에서 실CV 데이터 없이 결정적.
truth rank 는 보지 않는다 — proposer 가 후보를 1개라도 내면 row 가 생기므로 modality 커버리지만 검증.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # --extra dev 필요.

from pathlib import Path

from poc.workflow_2.align_similarity import _consensus_template_ab


def _bar_crop(size=64, x=30, width=4):
    """세로 흰 막대 한 줄이 박힌 sharp gray crop(consensus 재료)."""
    img = np.zeros((size, size), np.uint8)
    img[:, x:x + width] = 255
    return img


def _textured_frame(size=256):
    """여러 세로 막대로 edge 구조를 만든 프레임 — proposer 가 후보를 surface 하도록."""
    img = np.full((size, size), 40, np.uint8)
    for x in range(20, size - 20, 40):
        img[:, x:x + 4] = 220
    return img


def _frames(mod, n, *, xy=(128, 128)):
    return [{"path": Path(f"/synthetic/{mod}_{i}.png"), "xy": xy, "mod": mod,
             "crop": _bar_crop(x=30)} for i in range(n)]


def test_consensus_ab_evaluates_each_modality_not_just_dominant():
    """OM 3장 + SEM 4장(SEM dominant) → per_recipe 에 om·sem 두 row 모두 존재해야 한다."""
    by_recipe = {
        "EQP/CLS/RCP": {
            "s_frames": _frames("om", 3) + _frames("sem", 4),
            "e_paths": [],
            "rcp_tpls": {},
        }
    }
    frame = _textured_frame()
    res = _consensus_template_ab(by_recipe, min_s=3, frame_loader=lambda f: frame)

    assert res is not None, "dual-modality recipe 인데 결과가 None"
    mods = {r["modality"] for r in res["per_recipe"]}
    assert mods == {"om", "sem"}, f"두 modality 모두 평가돼야 하는데 {mods} 만 나왔다"
