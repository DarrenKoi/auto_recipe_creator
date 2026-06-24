"""Template-bank matcher (bench 전용 bit-parity fork — ensemble_lab.py 패턴).

workflow_3 엔진 primitive 를 import 만 하고 절대 수정/역import 하지 않는다.
heatmap(soft-voting, primary) + rrf(extra) 두 arm, 그리고 winner 의 GT-bucket 분류.
좌표 규약: 모든 candidate/winner xy = 템플릿 중심 frame 픽셀(엔진 불변식).
"""
from dataclasses import dataclass

import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.consensus_cv import coregister_crops


def bank_build(crops, *, recipe_id, modality, min_s, coregister=True):
    """N 개의 S crop 을 *개별* AlignKeyTemplate 로 빌드(median 합치기 없음).

    crops: 동일 크기 uint8 gray crop 리스트. len<min_s 면 [] (consensus min_s 동일 게이트).
    coregister=True 면 빌드 전 sub-pixel 정렬(consensus 와 동일 전처리, blur 없음).
    """
    if crops is None or len(crops) < min_s:
        return []
    members = coregister_crops(crops) if coregister else list(crops)
    bank = []
    for i, c in enumerate(members):
        tpl = build_template(
            np.ascontiguousarray(c), recipe_id=recipe_id, version=f"s{i}",
            key_type=modality, align_offset_xy=(0, 0),
        )
        bank.append(tpl)
    return bank
