"""Template-bank matcher (bench 전용 bit-parity fork — ensemble_lab.py 패턴).

workflow_3 엔진 primitive 를 import 만 하고 절대 수정/역import 하지 않는다.
heatmap(soft-voting, primary) + rrf(extra) 두 arm, 그리고 winner 의 GT-bucket 분류.
좌표 규약: 모든 candidate/winner xy = 템플릿 중심 frame 픽셀(엔진 불변식).
"""
from dataclasses import dataclass

import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.matching.engine import preprocess_for_matching, _chamfer_score_map_at_scale
from poc.workflow_3.align.consensus_cv import coregister_crops
from poc.workflow_2.align_similarity import COMPARE_SCALES, TOPK_CANDIDATES


@dataclass
class BankResult:
    """bank 매칭 결과. xy=템플릿중심 frame 픽셀(없으면 None)."""
    xy: "tuple[int, int] | None"
    score: float
    cand_xys: list
    cand_scores: list
    member_support: "list[int] | None"
    arm: str


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


def _accumulate_heatmap(bank, frame_dt, frame_shape, scales):
    """멤버×scale 의 dense chamfer score map 을 frame-중심 좌표계에 SUM 누적.

    score_map[y,x] 는 top-left 배치 점수 → 중심은 (x+tw//2, y+th//2). 그래서
    acc[th//2:th//2+sh, tw//2:tw//2+sw] += score_map 로 중심좌표계에 더한다.
    """
    H, W = frame_shape
    acc = np.zeros((H, W), dtype=np.float32)
    for tpl in bank:
        for s in scales:
            score_map, (tw, th) = _chamfer_score_map_at_scale(tpl.edge_map, frame_dt, s)
            if score_map is None:
                continue
            sh, sw = score_map.shape
            oy, ox = th // 2, tw // 2
            if oy + sh > H or ox + sw > W:
                continue
            acc[oy:oy + sh, ox:ox + sw] += score_map
    return acc


def _peaks_center(acc, *, nms_radius, max_peaks, min_score):
    """중심좌표계 누적맵에서 NMS argmax peak 추출 → [(score,cx,cy),...] 내림차순."""
    work = acc.copy()
    peaks = []
    r = max(1, int(nms_radius))
    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        cy, cx = divmod(idx, work.shape[1])
        sc = float(work[cy, cx])
        if sc <= min_score:
            break
        peaks.append((sc, cx, cy))
        y0, y1 = max(0, cy - r), min(work.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(work.shape[1], cx + r + 1)
        work[y0:y1, x0:x1] = -np.inf
    return peaks


def bank_match_heatmap(bank, gray, *, scales=COMPARE_SCALES, peak_nms_frac=0.5,
                       topk=TOPK_CANDIDATES):
    """PRIMARY arm: 멤버 dense 응답을 SUM → 전역 peak. per-member top-K 가 떨어뜨린
    참 peak 도 약하게 일관되면 합산으로 살아남는다(gt_not_in_topk 공략)."""
    if not bank:
        return BankResult(None, 0.0, [], [], None, "heatmap")
    frame_dt = preprocess_for_matching(gray)[1]
    acc = _accumulate_heatmap(bank, frame_dt, gray.shape[:2], scales)
    short = min(bank[0].raw_image.shape[:2])
    nms_radius = max(1, int(peak_nms_frac * short))
    peaks = _peaks_center(acc, nms_radius=nms_radius, max_peaks=topk, min_score=0.0)
    if not peaks:
        return BankResult(None, 0.0, [], [], None, "heatmap")
    cand_xys = [(cx, cy) for (_s, cx, cy) in peaks]
    cand_scores = [s for (s, _cx, _cy) in peaks]
    return BankResult(cand_xys[0], cand_scores[0], cand_xys, cand_scores, None, "heatmap")
