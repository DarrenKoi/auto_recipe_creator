"""프레임 가장자리에 걸친 align key 의 **이동 단서**(hint) - accept 가 아니다.

matching 엔진은 전 단계가 valid-mode(`cv2.matchTemplate`)라 template 창이 프레임 안에 통째로
들어가야만 점수가 난다. 검출 footprint 는 1 FOV 가 아니라 (프레임 - template)이고, key 가
모서리에 걸치면 눈에 보여도 low 다(2026-09-19 진단). 사람은 이때 보이는 조각을 더블클릭해
중심으로 데려온다 - 이 모듈이 그 조각의 위치를 낸다.

엔진 밖에 둔 이유: 부분 겹침 점수는 분포가 달라 calibrated 임계(0.6053/0.4727)와 workflow_2
lab bit-parity 를 깨면 안 된다. 그래서 여기 결과는 **어디로 옮길지** 에만 쓰고, 옮긴 뒤에는
정상 full-window matcher + `key_visibility_gate` 가 처음부터 다시 판정한다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from poc.workflow_3.align.matching.engine import (
    DT_TAU_PX,
    STRUCTURE_POLICY,
    AlignKeyTemplate,
    _extract_peaks,
    _ncc,
    _resize_template,
    _scaled_edges,
    _to_grayscale,
    preprocess_for_matching,
)

# 보이는 template edge 비율 하한. 0.4 = 창의 60% 까지 프레임 밖으로 나가도 본다.
# ponytail: 부분 겹침 점수는 미캘리브(오피스 데이터 없음) - 임계는 full-window adjust 값을
# 빌린 잠정치다. 헛 hint 가 잦으면 MIN_SEL 을 올리고, 비용 상한은 호출부의 이동 횟수(2)다.
MIN_VISIBLE = 0.4
MIN_EDGE_PX = 80      # 절대 하한 - 작은 반복 조각 하나가 '40%' 를 채우는 것을 막는다.
MIN_SEL = STRUCTURE_POLICY.ensemble_adjust_threshold
TOP_K = 5


@dataclass(frozen=True)
class PartialHint:
    """걸친 key 의 추정 align point(frame px, 프레임 밖일 수 있다) + 근거."""

    align_xy: tuple[int, int]
    sel: float       # 0.5*chamfer + 0.5*max(0, ncc) - 엔진 selection 과 같은 식, 보이는 겹침만.
    visible: float   # 보이는 template edge 비율.
    scale: float


def partial_key_hint(template: AlignKeyTemplate, frame: np.ndarray, scales) -> PartialHint | None:
    """창이 프레임 밖으로 **걸친** 위치만 채점해 가장 그럴듯한 하나를 낸다. 없으면 None.

    창이 통째로 안에 드는 위치는 엔진이 이미 판정했으므로 제외한다. DT 는 패딩 **전에**
    계산한다(패딩 경계가 가짜 edge 가 되지 않게). 평균은 보이는 edge 만으로 낸다.
    """
    gray = _to_grayscale(frame)
    _edges, dt = preprocess_for_matching(gray)
    fh, fw = gray.shape[:2]
    ones = np.ones_like(dt)
    best: PartialHint | None = None
    for scale in scales:
        mask = (_scaled_edges(template.edge_map, scale) > 0).astype(np.float32)
        th, tw = mask.shape[:2]
        n_edges = float(mask.sum())
        if n_edges < MIN_EDGE_PX:
            continue
        px, py = int(tw * (1 - MIN_VISIBLE)), int(th * (1 - MIN_VISIBLE))
        if px <= 0 or py <= 0:
            continue
        pad = dict(top=py, bottom=py, left=px, right=px, borderType=cv2.BORDER_CONSTANT, value=0)
        seen = cv2.matchTemplate(cv2.copyMakeBorder(ones, **pad), mask, cv2.TM_CCORR)
        sum_dt = cv2.matchTemplate(cv2.copyMakeBorder(dt, **pad), mask, cv2.TM_CCORR)
        score = np.exp(-(sum_dt / np.maximum(seen, 1.0)) / DT_TAU_PX).astype(np.float32)
        score[seen < max(MIN_EDGE_PX, MIN_VISIBLE * n_edges)] = -1.0
        if fw > tw and fh > th:
            score[py:py + fh - th + 1, px:px + fw - tw + 1] = -1.0  # 통째로 안 = 엔진 몫.
        tpl = _resize_template(template.raw_image, scale)
        for ch, cx, cy in _extract_peaks(score, tw, th, max_peaks=TOP_K, min_score=0.0,
                                         nms_radius=max(4, min(tw, th) // 2)):
            cx, cy = cx - px, cy - py                      # 패딩 좌표 -> frame 좌표(창 중심).
            x0, y0 = cx - tw // 2, cy - th // 2
            ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(fw, x0 + tw), min(fh, y0 + th)
            a = tpl[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            b = gray[iy0:iy1, ix0:ix1]
            ncc = max(0.0, _ncc(a, b)) if a.shape == b.shape and a.size else 0.0
            sel = 0.5 * ch + 0.5 * ncc
            if sel >= MIN_SEL and (best is None or sel > best.sel):
                ox, oy = template.align_offset_xy          # offset x scale 은 여기서 한 번만.
                vis = float(seen[cy + py - th // 2, cx + px - tw // 2] / n_edges)
                best = PartialHint((int(cx + round(ox * scale)), int(cy + round(oy * scale))),
                                   float(sel), vis, float(scale))
    return best
