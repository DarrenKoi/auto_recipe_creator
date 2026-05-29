"""Align fail recipe 전체의 from_msr 이미지에 대해 *정답 align point* 를 추정한다.

목표:
  - from_rcp/IMAP0001 (OM), IMAP0002 (SEM) 에는 엔지니어가 그려둔 *흰색 박스*
    overlay 가 있다 — 그 박스 안쪽이 "유일하게 식별 가능한 영역" 이고, align
    point 는 그 박스 중심이다. 따라서 박스를 검출 → 박스 *내부* 픽셀만 잘라낸
    것이 template, 그 template 중심이 정답 align point 가 된다.
  - from_msr/{S,E}##_A000X-01AP.* 각 이미지에서 **CV 가 결정한 정답 좌표**
    (corrected_xy = 매칭된 template 중심) 와, **도구가 이미 그려둔 crosshair
    좌표** (crosshair_xy) 를 함께 뽑아 둘 사이의 보정 벡터를 기록한다.
  - 파일명 접두 S/E 는 도구가 self-reported 한 라벨이다. *항상 의심한다.*
    S 라벨인데 보정 벡터가 큰 이미지는 ``suspect_success`` 로 따로 모은다.
  - 도구가 완전히 포기한 경우 msr 이미지에 crosshair 가 *전혀* 없을 수 있다.
    그 자체가 신호이므로 status="no_crosshair_drawn" 으로 기록만 하고, CV 가
    찾은 corrected_xy 는 그대로 남긴다 (live 단계에서 그쪽으로 이동시킬 후보).

라이브 SEM box 워크플로우로 옮기기 전에 정적 데이터로만 검증하기 위한 배치
테스트. RCS / SEM monitor 의존성 없음 — pure CV. Modality (OM vs SEM) 는 두
template 을 모두 돌려 점수가 높은 쪽을 채택한다. Scale bar OCR (PaddleOCR-VL)
은 (a) audit 힌트로 모든 row 에 기록, (b) race 가 비자신적 (ambiguous_modality
/ low_match_both) 한 경우에 한해 tiebreaker 로 winner 를 교체할 수 있다.
CV race winner 는 cv_winner_modality 로 따로 보존되어 audit 가능.

Align fail 처리 트리 (각 msr 이미지에 대해):
    1. Frame 전체 blur (Laplacian var 낮음) → status="msr_unrecognizable":
       정렬 위치 추정 불가 — 호출자는 다음 행동 (live 탐색 등) 으로 이동.
    2. rcp template 부재 → status="no_templates".
    3. 양쪽 template 점수 모두 낮음 → status="low_match_both": OCR 힌트로 tiebreak 가능.
    4. winner 가 frame 안의 다른 위치들과 비슷한 점수 (not distinctive, 1회 retry 후에도) →
       status="not_distinctive": 진짜 align 위치가 아닐 가능성 — live 탐색 필요.
    5. modality 점수차 작음 → status="ambiguous_modality": OCR 힌트로 tiebreak 가능.
    6. crosshair 미검출 → status="no_crosshair_drawn": 도구가 포기. CV 좌표는 유효.
    7. crosshair 검출 + 보정 거리 < 임계 → status="already_aligned".
    8. 그 외 → status="ok".

산출물:
    poc/workflow_2/debug_images/align_correction/<eqp>__<class>__<recipe>__<ts>/
      ├─ rcp_om_box_overlay.jpg
      ├─ rcp_sem_box_overlay.jpg
      ├─ overlay/<msr basename>_overlay.jpg
      ├─ results.jsonl
      └─ summary.json
    plus 한 batch 의 최상위 요약: align_correction/batch_summary_<ts>.json

모드:
    - **batch-all (기본)**: align_images/ 아래 모든 recipe leaf 를 최신순으로 처리.
    - **single_override**: ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME 셋이 모두
      설정되면 그 한 recipe 만 처리.

실행:
    uv run python poc/workflow_2/align_point_correction.py
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from poc.workflow_1.prompts import build_ocr_assist_prompt
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.vlm_client import Workflow1VLMClient
from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import (
    AlignFailAssets,
    iter_msr_images,
    iter_recipe_dirs,
    load_gray,
    resolve_assets,
    resolve_assets_auto,
)
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score,
)

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정한다.
# ====================================================================

# template 이 frame 안에 들어가도록 두를 replicate 여백 비율 (template 한 변 대비).
# compare_align_images.PAD_RATIO 와 동일한 값. _pad_frame 을 그대로 재현한다.
PAD_RATIO = 0.35

# 정적 비교용 scale band — recipe 와 측정 이미지가 거의 같은 배율이라는 가정.
COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)

# Modality margin 임계값 — OM/SEM 점수차가 이보다 작으면 ambiguous 로 표시.
AMBIGUOUS_MODALITY_MARGIN = 0.05

# crosshair 와 corrected_xy 가 이 이하면 "이미 정렬됨" 으로 간주 (px).
ALREADY_ALIGNED_PX = 3.0

# S 라벨이면서 보정 거리가 이보다 크면 의심한다 (px). 첫 데이터셋에서 튜닝 필요.
TRUST_SUSPECT_PX = 15.0

# crosshair 검출 — top-hat 커널 크기. 자연 SEM feature 두께(>4 px)를 죽이고
# 1~3 px 두께의 가는 선만 남기는 정도.
TOPHAT_KERNEL = 9

# crosshair 검출 — 프레임 한 변의 몇 % 이상을 가로질러야 crosshair 로 인정할지.
MIN_LINE_SPAN_RATIO = 0.40

# crosshair 검출 — projection peak 가 median 대비 몇 배 이상이어야 prominent 한지.
MIN_PEAK_OVER_MEDIAN = 3.0

# rcp 흰색 unique-area 박스 — 박스 stroke 두께를 피해 안쪽 픽셀만 template 으로 쓰기 위한 inset.
RCP_BOX_INSET_PX = 3

# rcp 박스 검출 시 후보 bbox 의 면적 하한/상한 (이미지 전체 면적 대비).
# 정상 box 는 이미지 중앙 부근에 작게 그려진다. 절반 가까이 차지하면 엔지니어 실수로 보고 거른다.
RCP_BOX_MIN_AREA_RATIO = 0.01
RCP_BOX_MAX_AREA_RATIO = 0.40

# rcp 박스 검출 시 후보 bbox 의 가장 짧은 변 길이 하한 (이미지 짧은 변 대비).
RCP_BOX_MIN_SIDE_RATIO = 0.05

# rcp 박스 검출 시 hollow-outline 필터 — bbox 내부 mask 충진율이 이 이하이면 outline 으로 본다.
# 채워진 wafer 형태/스케일바 사각형은 충진율이 높아 거르고, 1~3 px 두께 outline 만 통과시킨다.
RCP_BOX_MAX_FILL_RATIO = 0.40

# rcp 박스 후보 bbox 의 가로:세로 종횡비 상한 — 너무 가는 띠 (스케일바, axis label 등) 거르기.
RCP_BOX_MAX_ASPECT = 3.0

# rcp 박스는 이미지 가장자리에 *닿으면 안 된다* — 닿는 경우는 엔지니어가 잘못 그린 케이스.
# 이 px 만큼 가장자리에서 떨어져 있어야 정상 박스로 인정한다.
RCP_BOX_EDGE_MARGIN_PX = 2

# 박스 검출 실패시 fallback — 이미지 중심을 중심으로 *면적 기준 15%* 의 crop 을 template 으로 쓴다.
# 각 변은 sqrt(0.15) ≈ 0.387 비율. 전체 이미지를 쓰는 것보다 매칭이 unique 영역 (= 이미지 중심부)
# 에 집중되고, align_offset 은 (0,0) 그대로 유지되어 좌표 계산이 단순하다. 20% 는 여전히
# 너무 커서 인접한 noise 까지 포함됐던 사용자 피드백 반영 (2026-05-28).
RCP_FALLBACK_CENTER_CROP_AREA_RATIO = 0.15

# 도구가 그린 crosshair 를 *공간 prior* 로 사용해 CV 매칭의 ambiguity 를 깬다.
# free 검색 best 위치가 crosshair-derived 위치 (= crosshair - align_offset) 와 이만큼 떨어져 있으면
# prior-제한 ROI 로 다시 매칭해서 두 결과를 비교한다 (S=success 이미지에서 일관성 회복).
CROSSHAIR_PRIOR_DISAGREEMENT_PX = 30
# prior 결과를 prefer 하려면 prior 점수가 free 점수에서 이만큼 이내여야 한다.
# (prior 가 free 보다 약간 낮아도, 도구의 crosshair 는 *측정* 이므로 약간의 양보 가능.)
CROSSHAIR_PRIOR_SCORE_TOLERANCE = 0.10

# 도구가 그린 crosshair 라인 마스크 반폭 — 매칭 전에 frame 에서 inpaint 로 지운다.
# 실제 crosshair 두께가 1~3 px 이라 약 5 px 폭 mask (half=2 면 양쪽 합 5) 면 충분.
# Crosshair 가 길고 밝은 직선이라 Chamfer matcher 가 이를 wafer feature 로 오인해 lock-on 하는
# 사례를 막기 위함 — S 라벨에서 corrected_xy 가 엉뚱한 곳을 가리키던 원인.
CROSSHAIR_INPAINT_HALF_THICKNESS_PX = 2

# Distinctiveness — winner 영역을 마스킹하고 매칭을 다시 돌려 *2nd-best peak* 의 점수를 측정한다.
# distinctiveness_ratio = second_best.score / best.score.
# 1.0 에 가까울수록 "다른 곳도 거의 같은 점수" → 진짜 align 위치가 아님 (Lowe's ratio test 의 template 버전).
# 0.85 미만이면 distinctive — winner 채택.
# 0.85 이상이면 winner reject 후 다음 attempt 의 결과를 새 winner 로 시도 (최대 MAX_ATTEMPTS).
# attempt 마다 직전 winner 영역을 마스킹하여, "다른 곳"을 찾는 과정을 반복한다.
DISTINCTIVENESS_RATIO_MAX = 0.85
DISTINCTIVENESS_MAX_ATTEMPTS = 3  # 최대 3 candidates 평가 (ratio 계산은 인접 pair, 즉 2개 비교).

# msr frame 의 Laplacian 분산 (focus measure). 이보다 낮으면 "전체 blur 라 정렬 위치 추정
# 불가" 로 보고 매칭을 건너뛴다 — 호출자는 다음 행동(예: live 탐색)으로 이동한다.
# sem_box_detect.SHARPNESS_BLUR_THRESHOLD 와 같은 의미의 임계값. 실데이터 칼리브레이션 전 cold-start.
MIN_SHARPNESS_LAPVAR = 30.0

# 스케일 바 OCR — race winner 와 비교할 *보조 audit 힌트* + race 비자신 시에만 tiebreak.
SCALE_BAR_OCR_SERVICE = "paddleocr-vl-1.5"
# scale bar 는 이미지 *내부* 하단 오버레이 — 마지막 8% 만 잘라 보낸다 (PaddleOCR 환각 회피).
# 5% 는 강 ROI 지만 글자가 strip 위쪽에 걸리는 경우 누락 위험. 8% 가 안전 마진.
SCALE_BAR_CROP_RATIO = 0.08
SCALE_BAR_OCR_MAX_TOKENS = 256
# 100 µm 초과 → OM, 이하 → SEM (사용자 휴리스틱).
SCALE_BAR_OM_THRESHOLD_UM = 100.0
# race 자체로 결정이 어려운 status (ambiguous / low) 에서만 OCR 힌트를 tiebreaker 로 사용.
TIEBREAK_STATUSES = ("ambiguous_modality", "low_match_both")


# 결정/상태별 BGR 색.
_BGR_RED = (60, 60, 220)       # 도구가 그린 (현재) crosshair.
_BGR_GREEN = (80, 200, 80)     # CV 가 찾은 정답.
_BGR_YELLOW = (40, 200, 220)   # 보정 벡터.
_BGR_BLUE = (220, 140, 40)     # rcp 중심 crosshair.
_BGR_WHITE = (240, 240, 240)
_BGR_BLACK = (0, 0, 0)


# ====================================================================
# 데이터 클래스.
# ====================================================================


@dataclass
class _ModalityScore:
    """compute_align_key_score 결과를 JSON 직렬화 가능한 평탄 dict 로 풀어둔다.

    `out_of_frame=True` 는 매칭 위치가 _pad_frame 의 replicate-border 안에 있어
    pad 를 빼고 나니 원본 프레임 밖이라는 신호 — 실제 픽셀이 아니라 가짜 padding
    내용을 매칭한 것이므로 좌표는 클리핑된 값이고 신뢰도는 낮다.

    `used_crosshair_prior=True` 는 도구가 그린 crosshair 를 공간 prior 로 한 prior-ROI
    재매칭 결과를 채택했다는 의미. free 검색이 wrong-feature 에 락된 경우 (특히 S 라벨
    이미지) self-correct 하기 위한 메커니즘. `free_score`/`prior_score` 둘 다 기록해
    audit 가능하게 함.
    """

    score: float
    chamfer: float
    orb: float
    best_xy: tuple[int, int]   # 원본 frame 좌표계로 환원 + 프레임 안으로 클리핑한 좌표.
    best_scale: float
    decision: str
    out_of_frame: bool = False  # True 면 pad-border 매칭 — 좌표 신뢰 불가.
    used_crosshair_prior: bool = False
    free_score: float = 0.0
    prior_score: float | None = None
    prior_match_distance_px: float | None = None  # free 매치 중심에서 prior 까지의 거리 (px).
    # Distinctiveness (attempt-masking 방식) — best 위치가 frame 안에서 얼마나 유일하게 잘 맞는지.
    distinctive: bool = False
    distinctiveness_ratio: float | None = None  # best_chamfer / mean_sample_chamfer. 낮을수록 distinct.
    attempt_count: int = 1  # distinctiveness 실패 시 retry 횟수 (1 = 한 번에 성공, 2 = 1회 retry).
    # 엔진(top-N/NMS) distinctiveness — attempt-masking 과 *병렬 계측* (Item 2, A-safe).
    # redundant 하지만 1-pass 라 싸고, 같은 score map 의 인접 peak 기반. status 에는 보수적으로만 OR.
    engine_distinctive: bool = True
    engine_reject_reason: str | None = None     # "not_distinctive" | "no_candidates" | None.
    engine_second_ratio: float | None = None    # 2nd/best chamfer. 1.0 에 가까울수록 모호.
    engine_score_gap: float | None = None        # best - 2nd chamfer.
    engine_candidate_count: int = 0
    engine_scope: str = "free_full_frame"        # free_full_frame | prior_roi | masked_retry.

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "chamfer": self.chamfer,
            "orb": self.orb,
            "best_xy": list(self.best_xy),
            "best_scale": self.best_scale,
            "decision": self.decision,
            "out_of_frame": self.out_of_frame,
            "used_crosshair_prior": self.used_crosshair_prior,
            "free_score": self.free_score,
            "prior_score": self.prior_score,
            "prior_match_distance_px": self.prior_match_distance_px,
            "distinctive": self.distinctive,
            "distinctiveness_ratio": self.distinctiveness_ratio,
            "attempt_count": self.attempt_count,
            "engine_distinctive": self.engine_distinctive,
            "engine_reject_reason": self.engine_reject_reason,
            "engine_second_ratio": self.engine_second_ratio,
            "engine_score_gap": self.engine_score_gap,
            "engine_candidate_count": self.engine_candidate_count,
            "engine_scope": self.engine_scope,
        }


@dataclass
class _RcpTemplateBundle:
    """rcp template + 매칭 결과를 align point 로 환산할 때 더할 offset.

    엔지니어가 그린 흰 박스는 *unique area* 의 매칭 단서일 뿐이고, 실제 recipe 에
    기록된 align point 는 *이미지 중심* 이다. template (= 박스 안쪽 crop) 의 중심은
    박스 중심에 해당하므로, msr 에서 match 가 잡힌 위치 (= msr 에서의 박스 중심) 에
    `align_offset_xy` 를 더해야 msr 에서의 align point 좌표가 된다.

    fallback (박스 검출 실패) 경로에서는 template 이 전체 이미지이고 그 중심 = 이미지
    중심 = align point 이므로 align_offset_xy = (0, 0).
    """

    template: AlignKeyTemplate
    align_offset_xy: tuple[int, int]
    detected_box: tuple[int, int, int, int] | None
    inner_crop: tuple[int, int, int, int] | None


@dataclass
class RaceResult:
    """OM template 과 SEM template 을 동시에 돌려 점수가 높은 쪽을 채택한다."""

    winner: str                       # "om" | "sem" | "none"
    margin: float                     # winner.score - runner_up.score (없으면 winner.score).
    om: _ModalityScore | None
    sem: _ModalityScore | None

    @property
    def winner_xy(self) -> tuple[int, int] | None:
        winner_score = self._winner_score()
        return winner_score.best_xy if winner_score is not None else None

    @property
    def winner_score_value(self) -> float:
        s = self._winner_score()
        return s.score if s is not None else 0.0

    def _winner_score(self) -> _ModalityScore | None:
        if self.winner == "om":
            return self.om
        if self.winner == "sem":
            return self.sem
        return None


# ====================================================================
# 보조 — 파일명 파싱, padding.
# ====================================================================


_VISIT_RE = re.compile(r"A(\d+)", re.IGNORECASE)


def _parse_visit_order(name: str) -> int | None:
    """파일명에서 A000X 의 X 를 정수로 뽑는다. 없으면 None."""
    m = _VISIT_RE.search(name)
    return int(m.group(1)) if m else None


def _tool_label(name: str) -> str:
    """파일명 첫 글자 (S/E). 그 외는 '?'."""
    first = name[:1].upper()
    return first if first in ("S", "E") else "?"


def _pad_frame(frame: np.ndarray, template_shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """template 이 frame 안에 항상 들어가도록 replicate border 를 둘러 padding 한다.

    compare_align_images._pad_frame 과 동일한 로직 (PAD_RATIO=0.35).
    반환: (padded, pad_x, pad_y). best_xy 를 원본 좌표계로 환원할 때 빼주면 된다.
    """
    th, tw = template_shape[:2]
    pad_x = int(round(tw * PAD_RATIO))
    pad_y = int(round(th * PAD_RATIO))
    padded = cv2.copyMakeBorder(
        frame, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
    )
    return padded, pad_x, pad_y


# ====================================================================
# rcp 흰색 unique-area 박스 검출 + template crop.
# ====================================================================


def _detect_white_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """rcp 이미지에서 엔지니어가 그려둔 흰색 unique-area 박스의 bbox 를 추정한다.

    가정: 박스는 1~3 px 두께의 흰색 사각형 outline 이며 배경보다 확실히 밝다.
    1. Top-hat 으로 자연 feature 를 죽이고 얇고 밝은 구조만 남긴다.
    2. Otsu 이진화 → 짧은 dilation 으로 outline 의 모서리를 닫는다.
    3. 외곽 contour 의 boundingRect 중, 면적/짧은 변 길이 조건을 만족하는 가장 큰 것을
       채택한다 (scale bar 처럼 너무 작거나 프레임 전체에 가까운 후보는 거른다).

    반환: (x, y, w, h) bbox 또는 None.
    """
    h, w = gray.shape[:2]
    kernel = np.ones((TOPHAT_KERNEL, TOPHAT_KERNEL), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if int(tophat.max()) == 0:
        return None

    _thr, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    total_area = float(h * w)
    short_side = min(h, w)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < RCP_BOX_MIN_AREA_RATIO * total_area:
            continue
        if area > RCP_BOX_MAX_AREA_RATIO * total_area:
            continue
        if min(bw, bh) < RCP_BOX_MIN_SIDE_RATIO * short_side:
            continue
        # 가장자리 인접 거부 — 정상 box 는 항상 안쪽에 있어야 한다.
        margin = RCP_BOX_EDGE_MARGIN_PX
        if x < margin or y < margin or (x + bw) > (w - margin) or (y + bh) > (h - margin):
            continue
        # 종횡비 — 너무 길쭉한 띠 (스케일바, 축라벨, 줄 무늬) 거르기.
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > RCP_BOX_MAX_ASPECT:
            continue
        # Hollow-outline 검사 — 박스 *내부* 의 mask 충진율이 낮아야 한다 (1~3 px outline → 충진율 낮음).
        # findContours 의 boundingRect 는 외곽 walk 의 영역이라 contourArea/bbox 로는 hollowness 가 안 나옴.
        # 직접 binary 의 박스 영역을 보는 게 정확.
        inside = binary[y:y + bh, x:x + bw]
        fill_ratio = float(inside.mean()) if inside.size > 0 else 1.0
        if fill_ratio > RCP_BOX_MAX_FILL_RATIO:
            continue
        if area > best_area:
            best_area = area
            best = (int(x), int(y), int(bw), int(bh))
    return best


def _inner_crop_for_box(
    gray: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """흰색 박스 outline 픽셀을 피해, 박스 *안쪽* 영역만 잘라낸다.

    박스 stroke (보통 1~3 px) 자체가 template 에 들어가면 매칭이 흰색 픽셀을 찾으려
    해서 점수가 떨어진다. inset 만큼 안쪽으로 깎아 wafer 본연의 unique 패턴만 남긴다.

    반환: (cropped_gray, inner_bbox).
    """
    h, w = gray.shape[:2]
    x, y, bw, bh = box
    inset = RCP_BOX_INSET_PX
    x0 = max(0, x + inset)
    y0 = max(0, y + inset)
    x1 = min(w, x + bw - inset)
    y1 = min(h, y + bh - inset)
    if x1 - x0 < 8 or y1 - y0 < 8:
        # inset 으로 너무 작아지면 inset 없이 박스 통째로 사용.
        x0, y0, x1, y1 = x, y, min(w, x + bw), min(h, y + bh)
    return gray[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def _draw_rcp_overlay(
    gray: np.ndarray,
    *,
    bundle: _RcpTemplateBundle,
    out_path: Path,
    label: str,
) -> None:
    """rcp 이미지에 검출된 흰색 박스, inner crop (template), 그리고 *진짜 align point*
    (= 이미지 중심) 를 표시한다.

    파란 crosshair = 이미지 중심 = align point. 노란 박스 = 검출된 흰색 박스. 초록 박스
    = template (박스 안쪽 crop). 박스 중심에서 align point 로 시안색 화살표를 그려
    offset 을 시각화한다. 박스 미검출시에는 이미지 중심에 crosshair 만 (fallback 명시).
    """
    h, w = gray.shape[:2]
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_center = (w // 2, h // 2)

    if bundle.detected_box is not None:
        bx, by, bw, bh = bundle.detected_box
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), _BGR_YELLOW, 1, cv2.LINE_AA)
        if bundle.inner_crop is not None:
            ix, iy, iw, ih = bundle.inner_crop
            cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), _BGR_GREEN, 1, cv2.LINE_AA)
            tcx, tcy = ix + iw // 2, iy + ih // 2
        else:
            tcx, tcy = bx + bw // 2, by + bh // 2
        # 박스 중심 (template 중심) → align point (image center) 까지의 offset 시각화.
        cv2.arrowedLine(canvas, (tcx, tcy), image_center, (220, 200, 80), 1,
                        cv2.LINE_AA, tipLength=0.2)
        _draw_crosshair(canvas, image_center, _BGR_BLUE, length=18, thickness=2)
        dx, dy = bundle.align_offset_xy
        note = (
            f"{label}: box {bw}x{bh}, align point=image_center ({image_center[0]},{image_center[1]}), "
            f"offset (template->align)=({dx},{dy})"
        )
    elif bundle.inner_crop is not None:
        # Fallback: centered ~20% area crop (no engineer box). 시안색 사각형으로 표시해
        # box-detected 경로와 구분한다. align point 는 여전히 이미지 중심.
        ix, iy, iw, ih = bundle.inner_crop
        cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (220, 200, 80), 1, cv2.LINE_AA)
        _draw_crosshair(canvas, image_center, _BGR_BLUE, length=18, thickness=2)
        note = (
            f"{label}: white box NOT detected -> fallback {iw}x{ih} center crop "
            f"(~{RCP_FALLBACK_CENTER_CROP_AREA_RATIO * 100:.0f}% area), align=image_center"
        )
    else:
        _draw_crosshair(canvas, image_center, _BGR_BLUE, length=18, thickness=2)
        note = f"{label}: no template available"

    cv2.putText(canvas, note, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_WHITE, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _centered_area_crop_bbox(gray: np.ndarray, area_ratio: float) -> tuple[int, int, int, int]:
    """이미지 중심을 중심으로 하는 *면적 비율* 기반 crop bbox 를 (x, y, w, h) 로 돌려준다.

    각 변은 sqrt(area_ratio) 비율 — aspect 유지. 너무 작아서 매칭 불가능한 케이스를
    피하기 위해 변 길이의 하한을 32 px 로 둔다.
    """
    h, w = gray.shape[:2]
    side_ratio = float(np.sqrt(max(0.0, min(1.0, area_ratio))))
    cw = max(32, int(round(w * side_ratio)))
    ch = max(32, int(round(h * side_ratio)))
    cw = min(cw, w)
    ch = min(ch, h)
    x = max(0, (w - cw) // 2)
    y = max(0, (h - ch) // 2)
    return (x, y, cw, ch)


def _build_rcp_template(
    rcp_gray: np.ndarray,
    *,
    recipe_id: str,
    version: str,
    key_type: str,
    label: str,
) -> _RcpTemplateBundle:
    """rcp 이미지에서 unique-area 박스를 검출해, 그 안쪽 crop 으로 template 을 만들고
    align point (이미지 중심) 와 박스 중심 사이의 offset 을 함께 반환한다.

    Align point 는 *이미지 중심* 이지 박스 중심이 아니다. 박스는 단지 "이 영역이
    유일하게 식별 가능하다" 는 매칭 단서일 뿐. msr 에서 박스가 어디에 있는지 매칭으로
    찾으면 (= match center), 거기에 이 offset 을 더해야 msr 에서의 진짜 align point 가 된다.

    박스 검출 실패시 전체 이미지 대신 *이미지 중심 기준 ~20% area 크롭* 을 template 으로
    사용한다 — align point 가 이미지 중심이라는 사전 지식을 활용해 매칭이 의미 있는
    영역에만 집중하게 한다. crop 이 이미지 중심에 centered 되어 있으므로 offset = (0, 0).
    """
    h, w = rcp_gray.shape[:2]
    image_center = (w // 2, h // 2)

    detected_box = _detect_white_box(rcp_gray)
    if detected_box is None:
        fallback_bbox = _centered_area_crop_bbox(rcp_gray, RCP_FALLBACK_CENTER_CROP_AREA_RATIO)
        fx, fy, fw, fh = fallback_bbox
        fallback_gray = rcp_gray[fy:fy + fh, fx:fx + fw].copy()
        print(
            f"[WARNING] {label}: 흰색 unique-area 박스 검출 실패 — 이미지 중심 {fw}x{fh} "
            f"(≈{RCP_FALLBACK_CENTER_CROP_AREA_RATIO * 100:.0f}% area) 크롭을 template 으로 사용합니다."
        )
        template = build_template(
            fallback_gray, recipe_id=recipe_id, version=version, key_type=key_type,
        )
        return _RcpTemplateBundle(
            template=template,
            align_offset_xy=(0, 0),
            detected_box=None,
            inner_crop=fallback_bbox,
        )

    inner_gray, inner_bbox = _inner_crop_for_box(rcp_gray, detected_box)
    # template (= inner crop) 의 중심은 inner_bbox 의 중심 — 매칭이 잡아내는 위치.
    inner_x, inner_y, inner_w, inner_h = inner_bbox
    template_center_in_rcp = (inner_x + inner_w // 2, inner_y + inner_h // 2)
    align_offset_xy = (
        image_center[0] - template_center_in_rcp[0],
        image_center[1] - template_center_in_rcp[1],
    )
    print(
        f"[INFO] {label}: 박스 bbox={detected_box} inner={inner_bbox} "
        f"align_offset (template_center→image_center)={align_offset_xy}"
    )
    template = build_template(inner_gray, recipe_id=recipe_id, version=version, key_type=key_type)
    return _RcpTemplateBundle(
        template=template,
        align_offset_xy=align_offset_xy,
        detected_box=detected_box,
        inner_crop=inner_bbox,
    )


# ====================================================================
# msr 이미지의 기존 crosshair 검출 (top-hat + projection).
# ====================================================================


def _centroid_around(signal: np.ndarray, idx: int, band: int = 3) -> float:
    """argmax 주변 ±band 칸의 가중 평균으로 sub-pixel 좌표를 잡는다."""
    lo = max(0, idx - band)
    hi = min(signal.shape[0], idx + band + 1)
    window = signal[lo:hi].astype(np.float64)
    if window.sum() <= 0:
        return float(idx)
    coords = np.arange(lo, hi, dtype=np.float64)
    return float((coords * window).sum() / window.sum())


def _detect_existing_crosshair(
    gray: np.ndarray,
) -> tuple[tuple[int, int] | None, float, dict]:
    """msr 이미지에 도구가 그려둔 흰색/회색 crosshair 의 교점을 추정한다.

    1. Top-hat (커널 9x9) 으로 두꺼운 자연 feature 를 죽이고 가는 밝은 선만 남긴다.
    2. Otsu 이진화 후 row-sum / col-sum projection.
    3. 각 축의 argmax 가 (a) median 대비 prominent, (b) 프레임의 40% 이상 span,
       (c) 스케일바/축라벨 영역 (하단 10%, 좌측 5%) 밖에 있을 때만 crosshair 인정.
    4. 둘 다 통과하면 sub-pixel 보정 후 (cx, cy) 반환.

    crosshair 가 없거나 검출 실패한 경우 None 을 돌려준다 — 도구가 포기한 경우와
    검출이 헛친 경우를 downstream 에서 구분할 수 없으므로 status 하나로 묶는다.
    """
    h, w = gray.shape[:2]
    kernel = np.ones((TOPHAT_KERNEL, TOPHAT_KERNEL), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    if int(tophat.max()) == 0:
        return None, 0.0, {"reason": "empty_tophat"}

    _thr, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = (mask > 0).astype(np.uint8)

    row_sum = binary.sum(axis=1).astype(np.float32)   # shape (h,) — 각 y 에서 켜진 가로 픽셀 수.
    col_sum = binary.sum(axis=0).astype(np.float32)   # shape (w,) — 각 x 에서 켜진 세로 픽셀 수.

    # 스케일바/축라벨 영역 마스킹.
    row_keep = np.ones_like(row_sum, dtype=bool)
    row_keep[int(0.90 * h):] = False        # 하단 10% (scale bar)
    row_keep[: max(1, int(0.02 * h))] = False
    col_keep = np.ones_like(col_sum, dtype=bool)
    col_keep[: max(1, int(0.05 * w))] = False   # 좌측 5% (axis labels)
    col_keep[int(0.98 * w):] = False

    row_signal = np.where(row_keep, row_sum, 0.0)
    col_signal = np.where(col_keep, col_sum, 0.0)

    row_argmax = int(np.argmax(row_signal))
    col_argmax = int(np.argmax(col_signal))
    row_peak = float(row_signal[row_argmax])
    col_peak = float(col_signal[col_argmax])

    # median 은 *0보다 큰* 값만 가지고 잰다 (대부분 0인 분포에서 0-median 은 무의미).
    row_pos = row_signal[row_signal > 0]
    col_pos = col_signal[col_signal > 0]
    row_med = float(np.median(row_pos)) if row_pos.size else 0.0
    col_med = float(np.median(col_pos)) if col_pos.size else 0.0

    min_row_peak = MIN_LINE_SPAN_RATIO * w
    min_col_peak = MIN_LINE_SPAN_RATIO * h
    row_ok = (row_peak >= min_row_peak) and (row_peak >= max(MIN_PEAK_OVER_MEDIAN * row_med, 1.0))
    col_ok = (col_peak >= min_col_peak) and (col_peak >= max(MIN_PEAK_OVER_MEDIAN * col_med, 1.0))

    debug = {
        "row_peak": row_peak,
        "col_peak": col_peak,
        "row_median": row_med,
        "col_median": col_med,
        "min_row_peak": min_row_peak,
        "min_col_peak": min_col_peak,
        "row_argmax": row_argmax,
        "col_argmax": col_argmax,
        "row_ok": bool(row_ok),
        "col_ok": bool(col_ok),
    }

    if not (row_ok and col_ok):
        debug["reason"] = "axis_gate_failed"
        return None, 0.0, debug

    cy_sub = _centroid_around(row_signal, row_argmax)
    cx_sub = _centroid_around(col_signal, col_argmax)

    # confidence: 두 축 prominence 의 평균 (선 길이가 프레임을 얼마나 차지하나).
    confidence = min(1.0, 0.5 * (row_peak / max(w, 1) + col_peak / max(h, 1)))

    debug["reason"] = "ok"
    debug["sub_pixel"] = [cx_sub, cy_sub]
    return (int(round(cx_sub)), int(round(cy_sub))), float(confidence), debug


# ====================================================================
# Template race — OM/SEM 모두 돌려 점수 높은 쪽을 채택.
# ====================================================================


def _mask_region(
    frame_gray: np.ndarray,
    center_xy: tuple[int, int],
    half_w: int,
    half_h: int,
    fill_value: int = 128,
) -> np.ndarray:
    """frame 의 (center_xy 주변 half_w × half_h) 영역을 fill_value 로 덮은 사본 반환.

    Chamfer matcher 가 이 영역에서 다시 winner 를 뽑지 않도록 edge content 를 지운다.
    fill_value=128 (중간 회색) 은 CLAHE / Canny 통과 후 edge 가 거의 안 생긴다.
    """
    out = frame_gray.copy()
    cx, cy = center_xy
    fh, fw = out.shape[:2]
    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(fw, cx + half_w + 1)
    y1 = min(fh, cy + half_h + 1)
    out[y0:y1, x0:x1] = np.uint8(fill_value)
    return out


def _inpaint_crosshair(
    frame_gray: np.ndarray,
    crosshair_xy: tuple[int, int],
) -> np.ndarray:
    """도구가 그린 crosshair 의 가로/세로 직선을 inpaint 로 지운 frame 을 돌려준다.

    이 frame 은 CV 매칭 전용 — 시각화/JSONL 에 기록되는 원본은 변형하지 않는다.
    crosshair 가 frame 을 가로지르는 *thin bright line* 이라 Chamfer matcher 가 가짜
    edge 로 인식해 wrong-feature 에 lock-on 하는 경우를 방지한다 (특히 S 라벨에서
    corrected_xy 가 엉뚱한 곳을 가리키는 원인).
    """
    h, w = frame_gray.shape[:2]
    cx, cy = crosshair_xy
    mask = np.zeros((h, w), dtype=np.uint8)
    half = CROSSHAIR_INPAINT_HALF_THICKNESS_PX
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    mask[y0:y1, :] = 255  # 가로선 전체 폭
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    mask[:, x0:x1] = 255  # 세로선 전체 높이
    # INPAINT_TELEA: 빠르고 SEM 의 부드러운 텍스처에 잘 맞는다 (NS 보다 결과가 자연스러움).
    return cv2.inpaint(frame_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _match_with_prior_roi(
    template: AlignKeyTemplate,
    padded: np.ndarray,
    prior_center_in_padded: tuple[int, int],
) -> AlignKeyMatchResult | None:
    """compute_align_key_score 를 prior 중심의 좁은 ROI 로 제한해 다시 돌린다.

    ROI 폭/높이는 (smallest-scaled-template + 마진) 의 두 배 — 최소 매칭 가능 크기를
    만족시키면서, free 검색 대비 *충분히* 작은 영역에 가둔다. 매칭이 불가한 ROI
    (가장자리에 너무 가까워 잘렸을 때) 면 None.
    """
    th, tw = template.raw_image.shape
    min_scale = min(COMPARE_SCALES)
    # ROI 한 변이 smallest-scaled-template 보다 *확실히* 커야 matcher 가 통과.
    half_w = max(int(0.55 * tw) + 8, int(0.5 * min_scale * tw) + 12)
    half_h = max(int(0.55 * th) + 8, int(0.5 * min_scale * th) + 12)
    px, py = prior_center_in_padded
    ph, pw = padded.shape[:2]
    x0 = max(0, px - half_w)
    y0 = max(0, py - half_h)
    x1 = min(pw, px + half_w)
    y1 = min(ph, py + half_h)
    roi_w = x1 - x0
    roi_h = y1 - y0
    min_tw = max(8, int(round(tw * min_scale)))
    min_th = max(8, int(round(th * min_scale)))
    if roi_w <= min_tw or roi_h <= min_th:
        return None
    try:
        return compute_align_key_score(
            template,
            padded,
            roi_hint=(x0, y0, roi_w, roi_h),
            scales=COMPARE_SCALES,
            policy=STRUCTURE_POLICY,
        )
    except Exception as exc:
        print(f"[WARNING] prior-ROI 재매칭 실패 — free 검색 유지: {exc}")
        return None


def _match_against(
    bundle: _RcpTemplateBundle | None,
    frame_gray: np.ndarray,
    *,
    crosshair_xy: tuple[int, int] | None = None,
) -> _ModalityScore | None:
    """단일 rcp bundle 로 frame 을 매칭. bundle 이 없으면 None.

    Pipeline:
      1. free 검색: 매칭으로 frame 에서 template 중심 위치를 잡는다.
      2. 도구가 그린 crosshair 가 있고, free 검색의 match 중심이 crosshair-derived 위치
         (= crosshair - align_offset) 와 CROSSHAIR_PRIOR_DISAGREEMENT_PX 보다 멀면:
         prior 위치 주변 좁은 ROI 로 재매칭. prior 점수가 합리적 (>= adjust_threshold) 이고
         free 점수에서 CROSSHAIR_PRIOR_SCORE_TOLERANCE 안에 들면 prior 결과를 채택.
      3. 채택된 match 중심에 align_offset 을 더해 frame 에서의 align point 좌표.
      4. align point 가 frame 밖이면 out_of_frame=True 로 플래그하고 좌표를 클리핑.

    S (success) 라벨 이미지에서 free 검색이 wrong-feature 에 락된 경우, prior 가
    self-correct 시켜준다. E (fail) 라벨에서는 crosshair 자체가 틀린 위치이므로 prior
    점수가 낮게 나와 자동으로 free 검색으로 폴백된다.
    """
    if bundle is None:
        return None
    template = bundle.template
    # Crosshair 가 검출된 경우 그 직선을 frame 에서 지운 뒤 매칭 — 도구의 그림은 측정 데이터가
    # 아니라 annotation 이므로 matcher 가 봐서는 안 된다. 원본 frame_gray 는 변형하지 않는다.
    match_frame = _inpaint_crosshair(frame_gray, crosshair_xy) if crosshair_xy is not None else frame_gray
    dx_off, dy_off = bundle.align_offset_xy
    fh, fw = frame_gray.shape[:2]

    # 최대 DISTINCTIVENESS_MAX_ATTEMPTS 회 candidate 를 찾는다.
    # 각 attempt 는 직전 winner 영역을 마스킹한 frame 에서 free 검색 (attempt 0 만 crosshair prior 옵션).
    # 인접 두 attempt 의 점수비 = distinctiveness ratio:
    #   ratio = attempts[i+1].score / attempts[i].score 가 < THRESHOLD 면 attempts[i] 가 distinctive.
    # 처음으로 distinctive 한 attempt 를 채택. 모두 실패시 점수 최고 attempt 로 폴백 + distinctive=False.
    attempts: list[dict] = []
    current_frame = match_frame

    for attempt_idx in range(DISTINCTIVENESS_MAX_ATTEMPTS):
        padded, pad_x, pad_y = _pad_frame(current_frame, template.raw_image.shape)
        result_free = compute_align_key_score(
            template, padded, scales=COMPARE_SCALES, policy=STRUCTURE_POLICY,
        )
        fbx, fby = result_free.best_xy
        free_match_x = int(fbx - pad_x)
        free_match_y = int(fby - pad_y)

        attempt_used_prior = False
        attempt_prior_score: float | None = None
        attempt_prior_dist: float | None = None
        attempt_result = result_free
        attempt_match_x = free_match_x
        attempt_match_y = free_match_y

        # Crosshair prior 는 attempt 0 에서만 — retry 는 다른 위치를 찾는 게 목적이라 prior 가 무의미.
        if (
            attempt_idx == 0
            and crosshair_xy is not None
            and bundle.detected_box is not None
        ):
            prior_match_x = crosshair_xy[0] - dx_off
            prior_match_y = crosshair_xy[1] - dy_off
            attempt_prior_dist = float(np.hypot(
                free_match_x - prior_match_x, free_match_y - prior_match_y,
            ))
            if attempt_prior_dist > CROSSHAIR_PRIOR_DISAGREEMENT_PX:
                prior_result = _match_with_prior_roi(
                    template, padded,
                    (prior_match_x + pad_x, prior_match_y + pad_y),
                )
                if prior_result is not None:
                    attempt_prior_score = float(prior_result.score)
                    if (
                        prior_result.score >= STRUCTURE_POLICY.adjust_threshold
                        and prior_result.score + CROSSHAIR_PRIOR_SCORE_TOLERANCE >= result_free.score
                    ):
                        attempt_used_prior = True
                        attempt_result = prior_result
                        pbx, pby = prior_result.best_xy
                        attempt_match_x = int(pbx - pad_x)
                        attempt_match_y = int(pby - pad_y)

        attempts.append({
            "result": attempt_result,
            "free_score": float(result_free.score),
            "match_xy": (attempt_match_x, attempt_match_y),
            "score": float(attempt_result.score),
            "used_prior": attempt_used_prior,
            "prior_score": attempt_prior_score,
            "prior_dist": attempt_prior_dist,
        })

        # 마스크는 *다음* attempt 를 위한 준비 — 마지막 attempt 후에는 필요 없음.
        if attempt_idx < DISTINCTIVENESS_MAX_ATTEMPTS - 1:
            th_t, tw_t = template.raw_image.shape
            scale_for_mask = attempt_result.best_scale
            half_w = max(int(tw_t * scale_for_mask / 2), 16) + 4
            half_h = max(int(th_t * scale_for_mask / 2), 16) + 4
            current_frame = _mask_region(
                current_frame, (attempt_match_x, attempt_match_y), half_w, half_h,
            )

    # Distinctiveness 평가 — 인접 두 attempt 의 점수 비교.
    chosen_idx = 0
    chosen_distinct = False
    chosen_ratio: float | None = None
    for i in range(len(attempts) - 1):
        next_score = attempts[i + 1]["score"]
        cur_score = max(attempts[i]["score"], 1e-6)
        ratio = next_score / cur_score
        if ratio < DISTINCTIVENESS_RATIO_MAX:
            chosen_idx = i
            chosen_distinct = True
            chosen_ratio = ratio
            break
    else:
        # 어느 attempt 도 distinctive 하지 않음 — 점수 최고를 폴백으로 사용, not-distinctive 플래그.
        chosen_idx = max(range(len(attempts)), key=lambda k: attempts[k]["score"])
        chosen_distinct = False
        if len(attempts) >= 2:
            # 보고용 ratio: chosen vs 가장 가까운 비교 대상 (인덱스 + 1, 없으면 인덱스 - 1).
            cmp_idx = chosen_idx + 1 if chosen_idx + 1 < len(attempts) else chosen_idx - 1
            if 0 <= cmp_idx < len(attempts):
                chosen_ratio = attempts[cmp_idx]["score"] / max(attempts[chosen_idx]["score"], 1e-6)

    chosen = attempts[chosen_idx]
    final_result: AlignKeyMatchResult = chosen["result"]
    chosen_match_x, chosen_match_y = chosen["match_xy"]
    first_attempt = attempts[0]

    align_x = chosen_match_x + dx_off
    align_y = chosen_match_y + dy_off
    out_of_frame = (align_x < 0 or align_y < 0 or align_x >= fw or align_y >= fh)
    clipped_x = max(0, min(fw - 1, align_x))
    clipped_y = max(0, min(fh - 1, align_y))

    # 엔진(top-N/NMS) distinctiveness — chosen attempt 의 AlignKeyMatchResult 에서 추출 (getattr 로
    # 구버전 result 호환). scope: prior 채택이면 prior_roi(국소라 audit-only 권장), retry 면 masked_retry.
    if bool(chosen["used_prior"]):
        engine_scope = "prior_roi"
    elif chosen_idx > 0:
        engine_scope = "masked_retry"
    else:
        engine_scope = "free_full_frame"

    return _ModalityScore(
        score=float(final_result.score),
        chamfer=float(final_result.chamfer_score),
        orb=float(final_result.orb_inlier_ratio),
        best_xy=(clipped_x, clipped_y),
        best_scale=float(final_result.best_scale),
        decision=final_result.decision,
        out_of_frame=out_of_frame,
        used_crosshair_prior=bool(chosen["used_prior"]),
        free_score=first_attempt["free_score"],
        prior_score=first_attempt["prior_score"],
        prior_match_distance_px=first_attempt["prior_dist"],
        distinctive=chosen_distinct,
        distinctiveness_ratio=chosen_ratio,
        attempt_count=len(attempts),
        engine_distinctive=bool(getattr(final_result, "distinctive", True)),
        engine_reject_reason=getattr(final_result, "reject_reason", None),
        engine_second_ratio=getattr(final_result, "second_ratio", None),
        engine_score_gap=getattr(final_result, "score_gap", None),
        engine_candidate_count=len(getattr(final_result, "candidates", []) or []),
        engine_scope=engine_scope,
    )


def _race_templates(
    om_bundle: _RcpTemplateBundle | None,
    sem_bundle: _RcpTemplateBundle | None,
    frame_gray: np.ndarray,
    *,
    crosshair_xy: tuple[int, int] | None = None,
) -> RaceResult:
    """OM/SEM bundle 을 모두 돌려 점수가 더 높은 쪽을 winner 로 채택한다.

    scale bar OCR 을 쓰지 않고 점수 자체로 modality 를 결정한다 (계산은 두 배지만,
    OCR 환각/엣지 케이스로부터 자유롭다). _match_against 가 이미 align_offset 을
    적용하므로 winner.best_xy 는 msr 에서의 align point 좌표다. crosshair_xy 가 주어지면
    각 modality 에서 spatial prior 로 사용한다.
    """
    om = _match_against(om_bundle, frame_gray, crosshair_xy=crosshair_xy)
    sem = _match_against(sem_bundle, frame_gray, crosshair_xy=crosshair_xy)

    if om is None and sem is None:
        return RaceResult(winner="none", margin=0.0, om=None, sem=None)
    if om is None:
        return RaceResult(winner="sem", margin=sem.score, om=None, sem=sem)
    if sem is None:
        return RaceResult(winner="om", margin=om.score, om=om, sem=None)

    if om.score >= sem.score:
        return RaceResult(winner="om", margin=om.score - sem.score, om=om, sem=sem)
    return RaceResult(winner="sem", margin=sem.score - om.score, om=om, sem=sem)


# ====================================================================
# 보정 오버레이.
# ====================================================================


def _draw_crosshair(canvas: np.ndarray, xy: tuple[int, int], color, *, length: int = 18, thickness: int = 1) -> None:
    """주어진 위치에 십자 + 작은 원을 그린다 (이미지 전체를 가로지르지는 않는다)."""
    x, y = xy
    cv2.line(canvas, (x - length, y), (x + length, y), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y - length), (x, y + length), color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 4, color, thickness, cv2.LINE_AA)


def _build_unrecognizable_overlay(
    frame_gray: np.ndarray,
    *,
    sharpness: float,
    threshold: float,
    tool_label: str,
    visit_order: int | None,
) -> np.ndarray:
    """blur frame 전용 오버레이 — 가짜 corrected 마커를 그리지 않고 빨간 배너 + 안내 텍스트만.

    operator 가 overlay JPEG 만 보고 corrected 좌표가 있다고 착각하지 못하게 하기 위함.
    JSONL 의 corrected_xy=null 과 시각이 *일치* 해야 한다.
    """
    canvas = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    # 빨간 배너 (상단).
    header_h = 56
    banner = np.full((header_h, w, 3), (40, 40, 200), dtype=np.uint8)  # BGR 빨강 띠.
    visit_str = f"A{visit_order:04d}" if visit_order is not None else "A????"
    cv2.putText(
        banner,
        f"NO CORRECTION  label={tool_label} order={visit_str}",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BGR_WHITE, 2, cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        f"frame unrecognizable (lap_var={sharpness:.1f} < {threshold:.1f}) -> next action",
        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_WHITE, 1, cv2.LINE_AA,
    )

    # frame 위에 옅은 X 표시 — 시각적으로 '여기는 못 씀' 을 강조.
    line_color = (0, 0, 220)  # 진한 빨강
    cv2.line(canvas, (0, 0), (w - 1, h - 1), line_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (w - 1, 0), (0, h - 1), line_color, 1, cv2.LINE_AA)

    return np.vstack([banner, canvas])


def _build_correction_overlay(
    frame_gray: np.ndarray,
    *,
    crosshair_xy: tuple[int, int] | None,
    corrected_xy: tuple[int, int],
    modality: str,
    score: float,
    magnitude: float,
    status: str,
    tool_label: str,
    visit_order: int | None,
) -> np.ndarray:
    """현재 crosshair (빨강), 정답 (초록), 보정 벡터 (노랑) 를 한 장에 합친다."""
    canvas = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    if crosshair_xy is not None:
        _draw_crosshair(canvas, crosshair_xy, _BGR_RED, length=22, thickness=1)
        # 화면 전체를 가로지르는 가는 보조선 — 도구가 그린 위치와 그대로 비교.
        cv2.line(canvas, (0, crosshair_xy[1]), (canvas.shape[1] - 1, crosshair_xy[1]),
                 _BGR_RED, 1, cv2.LINE_AA)
        cv2.line(canvas, (crosshair_xy[0], 0), (crosshair_xy[0], canvas.shape[0] - 1),
                 _BGR_RED, 1, cv2.LINE_AA)

    _draw_crosshair(canvas, corrected_xy, _BGR_GREEN, length=22, thickness=2)

    if crosshair_xy is not None:
        cv2.arrowedLine(canvas, crosshair_xy, corrected_xy, _BGR_YELLOW, 2,
                        cv2.LINE_AA, tipLength=0.2)

    # 헤더 — 한글은 cv2 가 못 그리므로 ASCII 만.
    header_h = 56
    header = np.full((header_h, canvas.shape[1], 3), _BGR_BLACK, dtype=np.uint8)
    visit_str = f"A{visit_order:04d}" if visit_order is not None else "A????"
    cv2.putText(
        header,
        f"label={tool_label} order={visit_str} modality={modality.upper()} "
        f"score={score:.3f} status={status}",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BGR_WHITE, 1, cv2.LINE_AA,
    )
    detected_str = f"({crosshair_xy[0]},{crosshair_xy[1]})" if crosshair_xy else "MISSING"
    cv2.putText(
        header,
        f"detected={detected_str}  corrected=({corrected_xy[0]},{corrected_xy[1]})  "
        f"dist={magnitude:.1f}px",
        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_WHITE, 1, cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


# ====================================================================
# 한 msr 이미지 처리.
# ====================================================================


def _frame_sharpness(gray: np.ndarray) -> float:
    """frame 전체의 Laplacian 분산 (focus measure). 0 근처 → 전체 blur."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ====================================================================
# Scale bar OCR — 보조 audit 힌트 + race 비자신 시 tiebreaker.
# ====================================================================


# "5.0 um", "300 nm", "5µm" 등 매칭. µ/μ 두 종류 모두 허용.
_SCALE_BAR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(nm|um|µm|μm)\b",
    re.IGNORECASE,
)


def _parse_scale_bar_um(text: str) -> float | None:
    """OCR 텍스트에서 'X.X um' / 'X nm' 등을 찾아 µm 단위로 변환한다.

    여러 매치가 있으면 가장 큰 값을 채택 — scale bar 는 보통 OCR 결과 안에서 가장 큰
    숫자+단위 형태로 두드러지기 때문. 매치가 없으면 None.
    """
    if not text:
        return None
    candidates: list[float] = []
    for m in _SCALE_BAR_RE.finditer(text):
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        unit = m.group(2).lower()
        if unit == "nm":
            value /= 1000.0
        candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _modality_from_scale_um(um: float | None) -> str | None:
    """µm 값 → modality 힌트. 사용자 휴리스틱: > 100 µm 면 OM."""
    if um is None:
        return None
    return "om" if um > SCALE_BAR_OM_THRESHOLD_UM else "sem"


def _ocr_scale_bar(
    frame_gray: np.ndarray,
    *,
    ocr_client: Workflow1VLMClient | None,
) -> tuple[str, float | None, str | None]:
    """frame 하단 5% strip 을 PaddleOCR-VL 에 보내 (raw_text, um, modality_hint) 반환.

    실패 모드는 모두 ("", None, None) 으로 흡수 — 호출자는 client 가 없는/엔드포인트가
    먼저 떨어진 환경에서도 row 를 끝까지 만들 수 있어야 한다.

    layout→crop→recognize 룰 ([[project_paddleocr_vl_screenshot_hallucination]])
    준수 — 전체 프레임이 아니라 strip 만 보낸다.
    """
    if ocr_client is None:
        return "", None, None
    h, w = frame_gray.shape[:2]
    strip_top = int(h * (1.0 - SCALE_BAR_CROP_RATIO))
    strip = frame_gray[strip_top:h, :]
    if strip.size == 0 or strip.shape[0] < 4:
        return "", None, None

    rgb = cv2.cvtColor(strip, cv2.COLOR_GRAY2RGB)
    try:
        image_b64, _w, _h = encode_image_webp(Image.fromarray(rgb), quality=90)
        system_message, user_text = build_ocr_assist_prompt(width=0, height=0)
        response = ocr_client.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=SCALE_BAR_OCR_MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[WARNING] scale bar OCR 호출 실패: {exc}")
        return "", None, None

    raw_text = (getattr(response, "text", "") or "").strip()
    scale_um = _parse_scale_bar_um(raw_text)
    hint = _modality_from_scale_um(scale_um)
    return raw_text, scale_um, hint


def _process_msr_image(
    msr_path: Path,
    *,
    om_bundle: _RcpTemplateBundle | None,
    sem_bundle: _RcpTemplateBundle | None,
    overlay_dir: Path,
    is_current_sem: bool,
    eqp_id: str,
    class_name: str,
    recipe_id: str,
    ocr_client: Workflow1VLMClient | None,
) -> dict:
    """한 from_msr 이미지에 대해 결과 dict (JSONL row 한 줄 분량) 를 만든다.

    Align fail 처리 트리:
      - frame 이 너무 blur → status="msr_unrecognizable", corrected_xy=None.
        호출자는 정답을 찍지 말고 다음 행동 (live 탐색 등) 으로 이동한다.
      - template 부재 → status="no_templates".
      - 양쪽 template 모두 낮은 점수 → status="low_match_both" (보정 좌표는 emit, 신뢰 낮음).
      - modality 점수차 작음 → status="ambiguous_modality".
      - crosshair 검출 + 보정 거리 < 임계 → status="already_aligned".
      - crosshair 미검출 → status="no_crosshair_drawn".
      - 그 외 → status="ok".
    """
    print(f"[INFO] processing msr: {msr_path.name}")
    frame_gray = load_gray(msr_path)
    h, w = frame_gray.shape[:2]
    image_center = (w // 2, h // 2)

    tool_label = _tool_label(msr_path.name)
    visit_order = _parse_visit_order(msr_path.name)

    # Blur gate — 가장 먼저. frame 자체가 인식 불가면 매칭/십자검출 다 의미가 없다.
    # crosshair 검출도 *스킵* — blur frame 에 noise-driven 검출이 새어나가 JSONL 의 crosshair_xy 가
    # 가짜 좌표가 되는 걸 막는다. 오버레이도 가짜 corrected 마커를 그리지 않고 "NO CORRECTION" 배너만 표시.
    sharpness = _frame_sharpness(frame_gray)
    if sharpness < MIN_SHARPNESS_LAPVAR:
        overlay = _build_unrecognizable_overlay(
            frame_gray,
            sharpness=sharpness,
            threshold=MIN_SHARPNESS_LAPVAR,
            tool_label=tool_label,
            visit_order=visit_order,
        )
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{msr_path.stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[WARNING] {msr_path.name}: msr unrecognizable (lap_var={sharpness:.1f} "
              f"< {MIN_SHARPNESS_LAPVAR}) — 정렬 위치 추정 불가, 다음 행동 권장.")
        return {
            "recipe_id": recipe_id,
            "eqp_id": eqp_id,
            "class_name": class_name,
            "msr_image": msr_path.name,
            "visit_order": visit_order,
            "tool_label": tool_label,
            "is_current_sem": is_current_sem,
            "crosshair_xy": None,
            "crosshair_confidence": None,
            "sharpness_lapvar": sharpness,
            "winner_modality": "none",
            "winner_score": 0.0,
            "modality_margin": 0.0,
            "corrected_xy": None,
            "correction_vector_px": None,
            "correction_magnitude_px": None,
            "magnitude_basis": None,
            "om": None,
            "sem": None,
            "tool_label_suspect": False,
            "status": "msr_unrecognizable",
            "scale_bar_text": None,
            "scale_bar_um": None,
            "scale_bar_modality_hint": None,
            "scale_bar_matches_cv_winner": None,
            "scale_bar_matches_final_winner": None,
            "cv_winner_modality": "none",
            "tiebreak_applied": False,
            "winner_source": "cv",
            "overlay_path": str(overlay_path),
        }

    crosshair_xy, crosshair_conf, _crosshair_debug = _detect_existing_crosshair(frame_gray)
    race = _race_templates(om_bundle, sem_bundle, frame_gray, crosshair_xy=crosshair_xy)

    # OCR — race 결과를 *덮어쓰지 않고* 보조 힌트로만 기록 (단, 비자신 status 에서는 tiebreak).
    scale_bar_text, scale_bar_um, scale_bar_hint = _ocr_scale_bar(
        frame_gray, ocr_client=ocr_client,
    )

    # CV race 가 단독으로 결정한 winner 를 보존 — tiebreak 가 발생해도 audit 용으로 남긴다.
    cv_winner_modality = race.winner

    # status / 보정 좌표 결정.
    not_distinctive_source = None  # "attempt" | "engine" | "both" | None — not_distinctive 시 출처.
    if om_bundle is None and sem_bundle is None:
        status = "no_templates"
        corrected_xy = image_center
        magnitude_basis = "center"
        winner_modality = race.winner
        winner_score_value = race.winner_score_value
    elif race.winner == "none":
        status = "no_templates"
        corrected_xy = image_center
        magnitude_basis = "center"
        winner_modality = race.winner
        winner_score_value = race.winner_score_value
    else:
        corrected_xy = race.winner_xy or image_center
        winner_modality = race.winner
        winner_score_value = race.winner_score_value
        if crosshair_xy is None:
            magnitude_basis = "center"
        else:
            magnitude_basis = "crosshair"
        # 절대 점수가 낮은 게 더 큰 신호 — modality margin 보다 *우선해서* low_match_both 로 표시한다.
        # (점수가 둘 다 낮으면 margin 이 작은 건 당연하고, 그건 "둘 다 못 찾음" 이지 "어느 쪽인지 애매" 가 아니다.)
        winner_side = race.om if race.winner == "om" else race.sem
        winner_out_of_frame = bool(winner_side is not None and winner_side.out_of_frame)
        # attempt-masking distinctiveness (기존) OR 엔진 top-N distinctiveness (Item 2, A-safe).
        # 엔진 신호는 reject_reason=="not_distinctive" 만, prior_roi scope 는 국소라 audit-only(제외).
        winner_attempt_nd = bool(winner_side is not None and not winner_side.distinctive)
        winner_engine_nd = bool(
            winner_side is not None
            and winner_side.engine_reject_reason == "not_distinctive"
            and winner_side.engine_scope != "prior_roi"
        )
        if race.winner_score_value < STRUCTURE_POLICY.adjust_threshold or winner_out_of_frame:
            # pad-border 안에서 매칭된 경우 좌표 자체가 가짜이므로 점수와 무관하게 low_match_both.
            status = "low_match_both"
        elif winner_attempt_nd or winner_engine_nd:
            # 점수는 있지만 frame 안의 다른 위치들과 비슷하게 잘 맞음 — 진짜 align 위치가 아님.
            # 다음 행동: live 탐색이 필요. corrected_xy 는 best-guess 로 유지하지만 신뢰 표시는 낮춤.
            status = "not_distinctive"
            not_distinctive_source = (
                "both" if (winner_attempt_nd and winner_engine_nd)
                else ("attempt" if winner_attempt_nd else "engine")
            )
        elif (
            race.om is not None
            and race.sem is not None
            and race.margin < AMBIGUOUS_MODALITY_MARGIN
        ):
            status = "ambiguous_modality"
        else:
            status = "ok"

    # Tiebreak — race 가 비자신적인 status 일 때만 OCR 힌트로 winner 를 교체한다.
    # ok/already_aligned/no_crosshair_drawn 에서는 OCR 이 disagreement 만 기록.
    tiebreak_applied = False
    if (
        status in TIEBREAK_STATUSES
        and scale_bar_hint in ("om", "sem")
        and scale_bar_hint != race.winner
    ):
        hint_side = race.om if scale_bar_hint == "om" else race.sem
        if hint_side is not None:
            winner_modality = scale_bar_hint
            winner_score_value = hint_side.score
            corrected_xy = hint_side.best_xy
            tiebreak_applied = True
            print(
                f"[INFO] {msr_path.name}: race {cv_winner_modality}→{scale_bar_hint} "
                f"(status={status}, scale={scale_bar_um} µm) tiebreak applied"
            )

    # 보정 벡터 계산 — 기준은 crosshair (있으면) 또는 image center.
    ref_xy = crosshair_xy if crosshair_xy is not None else image_center
    dx = corrected_xy[0] - ref_xy[0]
    dy = corrected_xy[1] - ref_xy[1]
    magnitude = float(np.hypot(dx, dy))

    # crosshair-not-drawn 은 "도구가 포기" 의 강한 1차 신호 — CV 결정 status (ok/ambiguous_modality
    # /low_match_both) 보다 우선한다. msr_unrecognizable / no_templates / 이미 처리된 already_aligned
    # 는 더 강한 상태이므로 덮어쓰지 않는다.
    if crosshair_xy is None and status in ("ok", "ambiguous_modality", "low_match_both", "not_distinctive"):
        status = "no_crosshair_drawn"

    # already-aligned 는 crosshair 가 있고 거리 임계 이하인 경우에만.
    if crosshair_xy is not None and magnitude < ALREADY_ALIGNED_PX and status == "ok":
        status = "already_aligned"

    suspect = bool(tool_label == "S" and magnitude > TRUST_SUSPECT_PX)

    overlay = _build_correction_overlay(
        frame_gray,
        crosshair_xy=crosshair_xy,
        corrected_xy=corrected_xy,
        modality=winner_modality,
        score=winner_score_value,
        magnitude=magnitude,
        status=status,
        tool_label=tool_label,
        visit_order=visit_order,
    )
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{msr_path.stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 두 audit 비트로 분리: CV race 단독 vs 최종 winner (tiebreak 반영). 둘 다 hint 가 있을 때만 의미 있다.
    # tiebreak 가 일어났으면 final 쪽은 정의상 일치하므로, "CV vs OCR" 의 진짜 불일치는 _matches_cv_winner 로 본다.
    if scale_bar_hint is None:
        matches_cv_winner: bool | None = None
        matches_final_winner: bool | None = None
    else:
        matches_cv_winner = (
            scale_bar_hint == cv_winner_modality and cv_winner_modality in ("om", "sem")
        )
        matches_final_winner = (
            scale_bar_hint == winner_modality and winner_modality in ("om", "sem")
        )

    winner_source = "ocr_tiebreak" if tiebreak_applied else "cv"

    return {
        "recipe_id": recipe_id,
        "eqp_id": eqp_id,
        "class_name": class_name,
        "msr_image": msr_path.name,
        "visit_order": visit_order,
        "tool_label": tool_label,
        "is_current_sem": is_current_sem,
        "crosshair_xy": list(crosshair_xy) if crosshair_xy is not None else None,
        "crosshair_confidence": crosshair_conf,
        "sharpness_lapvar": sharpness,
        "winner_modality": winner_modality,
        "winner_score": winner_score_value,
        "winner_source": winner_source,
        "cv_winner_modality": cv_winner_modality,
        "tiebreak_applied": tiebreak_applied,
        "modality_margin": race.margin,
        "corrected_xy": list(corrected_xy),
        "correction_vector_px": [int(dx), int(dy)],
        "correction_magnitude_px": magnitude,
        "magnitude_basis": magnitude_basis,
        "om": race.om.to_dict() if race.om is not None else None,
        "sem": race.sem.to_dict() if race.sem is not None else None,
        "tool_label_suspect": suspect,
        "status": status,
        "not_distinctive_source": not_distinctive_source,
        "scale_bar_text": scale_bar_text or None,
        "scale_bar_um": scale_bar_um,
        "scale_bar_modality_hint": scale_bar_hint,
        "scale_bar_matches_cv_winner": matches_cv_winner,
        "scale_bar_matches_final_winner": matches_final_winner,
        "overlay_path": str(overlay_path),
    }


def _error_row(
    msr_path: Path,
    *,
    exc: BaseException,
    is_current_sem: bool,
    eqp_id: str,
    class_name: str,
    recipe_id: str,
) -> dict:
    """처리 실패한 msr 한 장의 row 를 정상 schema 로 만들어 batch 가 끊기지 않게 한다."""
    return {
        "recipe_id": recipe_id,
        "eqp_id": eqp_id,
        "class_name": class_name,
        "msr_image": msr_path.name,
        "visit_order": _parse_visit_order(msr_path.name),
        "tool_label": _tool_label(msr_path.name),
        "is_current_sem": is_current_sem,
        "crosshair_xy": None,
        "crosshair_confidence": None,
        "sharpness_lapvar": None,
        "winner_modality": "none",
        "winner_score": 0.0,
        "winner_source": "cv",
        "cv_winner_modality": "none",
        "tiebreak_applied": False,
        "modality_margin": 0.0,
        "corrected_xy": None,
        "correction_vector_px": None,
        "correction_magnitude_px": None,
        "magnitude_basis": None,
        "om": None,
        "sem": None,
        "tool_label_suspect": False,
        "status": "processing_error",
        "scale_bar_text": None,
        "scale_bar_um": None,
        "scale_bar_modality_hint": None,
        "scale_bar_matches_cv_winner": None,
        "scale_bar_matches_final_winner": None,
        "processing_error": {
            "type": type(exc).__name__,
            "message": str(exc)[:512],
        },
        "overlay_path": None,
    }


# ====================================================================
# Recipe 단위 처리.
# ====================================================================


def _process_recipe(
    assets: AlignFailAssets,
    out_root: Path,
    *,
    ocr_client: Workflow1VLMClient | None,
) -> dict:
    """한 recipe 의 모든 from_msr 이미지를 처리하고 summary dict 를 반환한다."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = lambda s: (s or "_").replace("/", "_").replace("\\", "_")  # noqa: E731
    out_dir = out_root / f"{safe(assets.eqp_id)}__{safe(assets.class_name)}__{safe(assets.recipe_name)}__{ts}"
    overlay_dir = out_dir / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # rcp template — 엔지니어가 그려둔 흰색 unique-area 박스를 검출, 그 안쪽 crop 으로 template 을 만들고
    # align_offset (image_center - template_center) 까지 묶어 bundle 로 들고 다닌다.
    om_bundle: _RcpTemplateBundle | None = None
    sem_bundle: _RcpTemplateBundle | None = None
    if assets.recipe_om is not None:
        om_gray = load_gray(assets.recipe_om)
        om_bundle = _build_rcp_template(
            om_gray, recipe_id=assets.recipe_id, version="rcp_om",
            key_type="om", label="OM (IMAP0001)",
        )
        _draw_rcp_overlay(
            om_gray, bundle=om_bundle,
            out_path=out_dir / "rcp_om_box_overlay.jpg", label="IMAP0001 OM",
        )
    if assets.recipe_sem is not None:
        sem_gray = load_gray(assets.recipe_sem)
        sem_bundle = _build_rcp_template(
            sem_gray, recipe_id=assets.recipe_id, version="rcp_sem",
            key_type="sem", label="SEM (IMAP0002)",
        )
        _draw_rcp_overlay(
            sem_gray, bundle=sem_bundle,
            out_path=out_dir / "rcp_sem_box_overlay.jpg", label="IMAP0002 SEM",
        )

    msr_images = iter_msr_images(assets)
    if not msr_images:
        print(f"[WARNING] from_msr 이미지 없음: {assets.recipe_dir}")

    results_path = out_dir / "results.jsonl"
    rows: list[dict] = []
    with results_path.open("w", encoding="utf-8") as fp:
        for msr_path in msr_images:
            try:
                row = _process_msr_image(
                    msr_path,
                    om_bundle=om_bundle,
                    sem_bundle=sem_bundle,
                    overlay_dir=overlay_dir,
                    is_current_sem=(assets.current_sem is not None and msr_path == assets.current_sem),
                    eqp_id=assets.eqp_id,
                    class_name=assets.class_name,
                    recipe_id=assets.recipe_id,
                    ocr_client=ocr_client,
                )
            except Exception as exc:
                # 한 장 실패가 batch 전체를 막지 않도록 — 동일 schema 로 처리 실패 row 를 남기고 다음 장으로.
                print(f"[ERROR] {msr_path.name}: 처리 중 예외 — {type(exc).__name__}: {exc}")
                row = _error_row(
                    msr_path,
                    exc=exc,
                    is_current_sem=(assets.current_sem is not None and msr_path == assets.current_sem),
                    eqp_id=assets.eqp_id,
                    class_name=assets.class_name,
                    recipe_id=assets.recipe_id,
                )
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 집계.
    status_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    ok_magnitudes: list[float] = []
    suspect_rows: list[dict] = []
    no_crosshair_rows: list[dict] = []
    unrecognizable_rows: list[dict] = []
    disagreement_rows: list[dict] = []
    tiebreak_rows: list[dict] = []
    error_rows: list[dict] = []
    crosshair_prior_rows: list[dict] = []
    not_distinctive_rows: list[dict] = []
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        modality_counts[row["winner_modality"]] = modality_counts.get(row["winner_modality"], 0) + 1
        if row["status"] == "ok":
            ok_magnitudes.append(row["correction_magnitude_px"])
        if row["tool_label_suspect"]:
            suspect_rows.append({
                "msr_image": row["msr_image"],
                "correction_magnitude_px": row["correction_magnitude_px"],
                "winner_modality": row["winner_modality"],
                "winner_score": row["winner_score"],
                "overlay_path": row["overlay_path"],
            })
        if row["status"] == "no_crosshair_drawn":
            no_crosshair_rows.append({
                "msr_image": row["msr_image"],
                "winner_modality": row["winner_modality"],
                "winner_score": row["winner_score"],
                "corrected_xy": row["corrected_xy"],
                "overlay_path": row["overlay_path"],
            })
        if row["status"] == "msr_unrecognizable":
            unrecognizable_rows.append({
                "msr_image": row["msr_image"],
                "sharpness_lapvar": row.get("sharpness_lapvar"),
                "tool_label": row["tool_label"],
                "overlay_path": row["overlay_path"],
            })
        # 진짜 "CV race vs scale-bar OCR" 불일치는 cv_winner 기준으로 본다 — tiebreak 적용 후
        # final_winner 와는 hint 가 일치하게 되므로 final 기준은 audit 가치가 없다.
        if row.get("scale_bar_matches_cv_winner") is False:
            disagreement_rows.append({
                "msr_image": row["msr_image"],
                "winner_modality": row["winner_modality"],
                "cv_winner_modality": row["cv_winner_modality"],
                "scale_bar_modality_hint": row.get("scale_bar_modality_hint"),
                "scale_bar_um": row.get("scale_bar_um"),
                "scale_bar_text": row.get("scale_bar_text"),
                "tiebreak_applied": row.get("tiebreak_applied"),
                "status": row["status"],
                "overlay_path": row["overlay_path"],
            })
        if row["status"] == "processing_error":
            error_rows.append({
                "msr_image": row["msr_image"],
                "processing_error": row.get("processing_error"),
            })
        # winner side 에서 prior 가 free 검색을 교체했는지 — S-image self-correction 의 핵심 audit.
        winner_modality = row.get("winner_modality")
        winner_payload = row.get(winner_modality) if winner_modality in ("om", "sem") else None
        if isinstance(winner_payload, dict) and winner_payload.get("used_crosshair_prior"):
            crosshair_prior_rows.append({
                "msr_image": row["msr_image"],
                "tool_label": row.get("tool_label"),
                "winner_modality": winner_modality,
                "free_score": winner_payload.get("free_score"),
                "prior_score": winner_payload.get("prior_score"),
                "prior_match_distance_px": winner_payload.get("prior_match_distance_px"),
                "corrected_xy": row.get("corrected_xy"),
                "crosshair_xy": row.get("crosshair_xy"),
                "overlay_path": row["overlay_path"],
            })
        if row["status"] == "not_distinctive":
            winner_modality_local = row.get("winner_modality")
            winner_payload_local = row.get(winner_modality_local) if winner_modality_local in ("om", "sem") else None
            wp = winner_payload_local if isinstance(winner_payload_local, dict) else {}
            not_distinctive_rows.append({
                "msr_image": row["msr_image"],
                "tool_label": row.get("tool_label"),
                "winner_modality": winner_modality_local,
                "not_distinctive_source": row.get("not_distinctive_source"),
                "attempt_ratio": wp.get("distinctiveness_ratio"),
                "attempt_count": wp.get("attempt_count"),
                "engine_second_ratio": wp.get("engine_second_ratio"),
                "engine_score_gap": wp.get("engine_score_gap"),
                "engine_reject_reason": wp.get("engine_reject_reason"),
                "engine_scope": wp.get("engine_scope"),
                "corrected_xy": row.get("corrected_xy"),
                "overlay_path": row["overlay_path"],
            })
        if row.get("tiebreak_applied"):
            tiebreak_rows.append({
                "msr_image": row["msr_image"],
                "cv_winner_modality": row["cv_winner_modality"],
                "final_winner_modality": row["winner_modality"],
                "scale_bar_um": row.get("scale_bar_um"),
                "status": row["status"],
                "overlay_path": row["overlay_path"],
            })

    summary = {
        "eqp_id": assets.eqp_id,
        "class_name": assets.class_name,
        "recipe_id": assets.recipe_id,
        "recipe_dir": str(assets.recipe_dir),
        "rcp_box": {
            "om_detected": om_bundle is not None and om_bundle.detected_box is not None,
            "om_bbox": list(om_bundle.detected_box) if om_bundle and om_bundle.detected_box else None,
            "om_inner_crop": list(om_bundle.inner_crop) if om_bundle and om_bundle.inner_crop else None,
            "om_align_offset_xy": list(om_bundle.align_offset_xy) if om_bundle else None,
            "sem_detected": sem_bundle is not None and sem_bundle.detected_box is not None,
            "sem_bbox": list(sem_bundle.detected_box) if sem_bundle and sem_bundle.detected_box else None,
            "sem_inner_crop": list(sem_bundle.inner_crop) if sem_bundle and sem_bundle.inner_crop else None,
            "sem_align_offset_xy": list(sem_bundle.align_offset_xy) if sem_bundle else None,
        },
        "total_msr_images": len(rows),
        "status_counts": status_counts,
        "modality_distribution": modality_counts,
        "ok_correction_magnitude_px": {
            "mean": float(np.mean(ok_magnitudes)) if ok_magnitudes else None,
            "median": float(np.median(ok_magnitudes)) if ok_magnitudes else None,
            "max": float(np.max(ok_magnitudes)) if ok_magnitudes else None,
        },
        "thresholds": {
            "TRUST_SUSPECT_PX": TRUST_SUSPECT_PX,
            "ALREADY_ALIGNED_PX": ALREADY_ALIGNED_PX,
            "AMBIGUOUS_MODALITY_MARGIN": AMBIGUOUS_MODALITY_MARGIN,
            "MIN_SHARPNESS_LAPVAR": MIN_SHARPNESS_LAPVAR,
            "SCALE_BAR_OM_THRESHOLD_UM": SCALE_BAR_OM_THRESHOLD_UM,
            "STRUCTURE_POLICY.adjust_threshold": STRUCTURE_POLICY.adjust_threshold,
            "STRUCTURE_POLICY.max_second_ratio": STRUCTURE_POLICY.max_second_ratio,
            "STRUCTURE_POLICY.min_distinct_gap": STRUCTURE_POLICY.min_distinct_gap,
            "DISTINCTIVENESS_RATIO_MAX": DISTINCTIVENESS_RATIO_MAX,
        },
        # not_distinctive 의 출처 분해 (attempt-masking vs 엔진 top-N). 오피스 첫 batch 에서
        # engine_only 가 튀면 status gate 를 재조정한다 (Item 2, A-safe 의 calibration 신호).
        "not_distinctive_source_counts": {
            src: sum(1 for r in not_distinctive_rows if r.get("not_distinctive_source") == src)
            for src in ("attempt", "engine", "both")
        },
        "ocr_client_initialized": ocr_client is not None,
        "suspect_success_images": suspect_rows,
        "no_crosshair_drawn_images": no_crosshair_rows,
        "unrecognizable_images": unrecognizable_rows,
        "modality_disagreement_images": disagreement_rows,
        "tiebreak_applied_images": tiebreak_rows,
        "processing_error_images": error_rows,
        "crosshair_prior_applied_images": crosshair_prior_rows,
        "not_distinctive_images": not_distinctive_rows,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"[INFO] processed {len(rows)} msr images → {out_dir}")
    print(f"[INFO] status counts: {status_counts}")
    if suspect_rows:
        print(f"[INFO] suspect successes (S 라벨이지만 보정 거리 큰 이미지): {len(suspect_rows)}")
        for s in suspect_rows:
            print(f"        - {s['msr_image']}  dist={s['correction_magnitude_px']:.1f}px  "
                  f"modality={s['winner_modality']}  score={s['winner_score']:.3f}")
    if no_crosshair_rows:
        print(f"[INFO] no-crosshair-drawn images (도구가 포기 추정): {len(no_crosshair_rows)}")
    if unrecognizable_rows:
        print(f"[INFO] unrecognizable images (전체 blur — 다음 행동 권장): {len(unrecognizable_rows)}")
    if tiebreak_rows:
        print(f"[INFO] tiebreak applied (race 비자신 + OCR 힌트로 modality 교체): {len(tiebreak_rows)}")
    if disagreement_rows:
        print(f"[INFO] modality 불일치 (CV race vs scale-bar OCR): {len(disagreement_rows)}")
    if error_rows:
        print(f"[INFO] processing errors (개별 장 실패, batch 계속): {len(error_rows)}")
    if crosshair_prior_rows:
        print(f"[INFO] crosshair prior 적용 (free 검색을 spatial prior 로 교체): {len(crosshair_prior_rows)}")
    if not_distinctive_rows:
        print(f"[INFO] not-distinctive 매치 (frame 의 다른 곳들과 비슷한 점수 — live 탐색 필요): {len(not_distinctive_rows)}")
    return summary


# ====================================================================
# 엔트리.
# ====================================================================


def _has_full_override() -> bool:
    """완전한 override (eqp + class + recipe 모두) 가 설정되었는지 확인.

    resolve_assets_auto 와 같은 룰: ALIGN_EQP_ID 가 있고 ALIGN_CLASS_NAME/ALIGN_RECIPE_NAME
    을 슬래시 단위로 풀었을 때 2단계 이상이 모이면 단일 recipe override 모드.
    """
    eqp = os.getenv("ALIGN_EQP_ID", "").strip()
    cls = os.getenv("ALIGN_CLASS_NAME", "").strip()
    rcp = os.getenv("ALIGN_RECIPE_NAME", "").strip()
    rel_parts = [
        part
        for segment in (cls, rcp)
        for part in segment.replace("\\", "/").strip("/").split("/")
        if part
    ]
    return bool(eqp and len(rel_parts) >= 2)


def _resolve_recipes_to_process() -> list[AlignFailAssets]:
    """처리할 recipe 목록을 결정한다.

    - 완전한 override 가 있으면 그 recipe 하나만.
    - 그 외에는 align_images/ 아래의 *모든* recipe leaf 폴더를 (최신순으로) 모두 처리.
      "내가 모든 recipe 에 대해 돌려보고 싶다" 시나리오용.
    """
    if _has_full_override():
        assets = resolve_assets_auto()
        return [assets] if assets is not None else []
    leaves = iter_recipe_dirs()
    print(f"[INFO] batch-all 모드: align_images 아래 recipe leaf {len(leaves)} 개 발견")
    return [resolve_assets(leaf) for leaf in leaves]


def run() -> str:
    started = time.time()
    recipes = _resolve_recipes_to_process()
    if not recipes:
        print("[ERROR] 처리할 align fail recipe 폴더를 찾지 못했습니다.")
        return "no_assets"

    out_root = DEBUG_IMAGE_DIR / "align_correction"
    out_root.mkdir(parents=True, exist_ok=True)

    # OCR client 는 batch 전체에서 한 번만 만들어 재사용 — 실패해도 진행 (Mac 등 proxy 부재 환경).
    ocr_client: Workflow1VLMClient | None = None
    try:
        ocr_client = Workflow1VLMClient(
            service_slug=SCALE_BAR_OCR_SERVICE, log_name="scale_bar_ocr",
        )
        print(f"[INFO] scale-bar OCR client 준비됨: {SCALE_BAR_OCR_SERVICE}")
    except Exception as exc:
        print(f"[WARNING] scale-bar OCR client 생성 실패 — 힌트 없이 진행: {exc}")

    per_recipe_summaries: list[dict] = []
    failed_recipes: list[dict] = []
    for idx, assets in enumerate(recipes, start=1):
        header = f"[{idx}/{len(recipes)}] {assets.eqp_id}/{assets.class_name}/{assets.recipe_name}"
        print(f"\n===== {header} =====")
        if assets.recipe_om is None and assets.recipe_sem is None:
            print(f"[WARNING] {header}: from_rcp 안에 IMAP0001/IMAP0002 둘 다 없음 — 건너뜀")
            failed_recipes.append({
                "recipe_dir": str(assets.recipe_dir),
                "reason": "no_templates",
            })
            continue
        try:
            summary = _process_recipe(assets, out_root, ocr_client=ocr_client)
            per_recipe_summaries.append({
                "recipe_dir": str(assets.recipe_dir),
                "out_dir": summary.get("out_dir"),
                "total_msr_images": summary.get("total_msr_images"),
                "status_counts": summary.get("status_counts"),
            })
        except Exception as exc:
            # recipe 한 개의 처리 자체가 실패해도 batch 가 멈추지 않는다.
            print(f"[ERROR] {header}: recipe 처리 예외 — {type(exc).__name__}: {exc}")
            failed_recipes.append({
                "recipe_dir": str(assets.recipe_dir),
                "reason": f"{type(exc).__name__}: {str(exc)[:256]}",
            })

    # batch 전체 요약.
    batch_ts = time.strftime("%Y%m%d_%H%M%S")
    batch_summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 2),
        "mode": "single_override" if _has_full_override() else "batch_all",
        "ocr_client_initialized": ocr_client is not None,
        "scale_bar_crop_ratio": SCALE_BAR_CROP_RATIO,
        "recipe_count": len(recipes),
        "processed_recipes": per_recipe_summaries,
        "failed_recipes": failed_recipes,
    }
    batch_path = out_root / f"batch_summary_{batch_ts}.json"
    batch_path.write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(
        f"\n[INFO] batch done in {batch_summary['duration_sec']}s — "
        f"processed={len(per_recipe_summaries)} failed={len(failed_recipes)}"
    )
    print(f"[INFO] batch summary: {batch_path}")

    # status 축 review 뷰 (by_status/ + index.html) 자동 생성 — 100+ recipe 를 폴더별로
    # 열지 않고 실패 유형별·worst-first 로 훑기 위함. review 실패가 batch 성공을 막지 않는다.
    if per_recipe_summaries:
        try:
            from poc.workflow_2.align_review import build_review
            build_review(out_root, batch_summary_path=batch_path)
        except Exception as exc:
            print(f"[WARNING] review 생성 실패 (batch 결과는 유효): {type(exc).__name__}: {exc}")
    if failed_recipes:
        for f in failed_recipes:
            print(f"        - FAILED {f['recipe_dir']}  reason={f['reason']}")
        # 일부 실패는 success 로 본다 — 다음 recipe 들이 정상 처리되었기 때문.
    if not per_recipe_summaries:
        return "all_failed"
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
