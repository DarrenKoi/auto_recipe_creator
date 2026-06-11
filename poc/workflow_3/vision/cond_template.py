"""cond.txt 기하 primitive — box-crop template + decoupled align offset (production).

office 검증(workflow_2 golden_localization_eval_cond, 2026-06-11): cond.box_ltrb 로 crop 한
box template(+분리된 offset)이 center-area crop 대비 모든 displacement bin 에서 localization
동반 상승(rank1 +0.16~0.18). 그 검증된 cond 기하 함수를 lab 에서 production 으로 byte-identical
승격한다([[project_align_cond_files_and_coords]], [[project_rcp_white_box_unique_area]]).

핵심 분리(decoupled offset): align point 는 *이미지 중심* 이지 box 중심이 아니다. box 는
유니크 영역 단서일 뿐이다. offset = image_center - box_center 를 crop 과 분리해 cond 기하로만
계산하고(검출 inner-crop 의 off-center 오염 제거), template 은 box stroke 를 inpaint 로 지운 뒤
box 내부를 *대칭* inset 해 만든다(crop 중심 == box 중심 → offset 과 일관).

좌표계: cond.txt cursor 좌표는 이미지 px 의 10배(OVERSAMPLE). 변환은 clean_align_image 의
cursor_to_image 를 재사용한다(중복 생성 금지). 의존 방향: lab → 이 모듈(prod), 역방향 금지.
"""

import numpy as np

from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import CondInfo

# box-없음 fallback 의 center-area crop 비율(검증된 center arm; lab CENTER_AREA_RATIO 동일 값).
CENTER_AREA_RATIO = 0.15

# --- cond box → template/offset 가드 상수 (lab byte-parity) ---
CROP_INSET_PX = 2       # inpaint 후 edge-smear 가 template 에 안 들어오게 하는 대칭 inset.
MIN_INNER_PX = 16       # 대칭 inset 후 box 내부 하한(미만이면 skip — 매칭 신호 불안정).
WARN_INNER_PX = 24      # 작은 box 경고 임계(skip 아님).
OFFSET_WARN = 0.25      # offset_norm(÷대각선) 경고 임계(box 가 중심에서 멂).
OFFSET_SKIP = 0.38      # offset_norm 하드 skip(=box≠center 가정 붕괴, 엔지니어 검토 필요).


def _cond_box_to_xywh(box_ltrb):
    """cond.box_ltrb(cursor frame, ×10) → 이미지 px (x, y, w, h)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t)))


def _cond_box_center(box_ltrb):
    """cond.box_ltrb → 이미지 px box 중심 (cx, cy) (정수 반올림 전 float)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (l + r) / 2.0, (t + b) / 2.0


def cond_align_offset(box_ltrb, shape_hw):
    """align point(이미지 중심) - box 중심. cond.txt 만으로 결정 → crop 과 분리(decoupled).

    crop 을 어떻게 잡든 align point 의 기하는 안 변한다. 이 분리가 원본의 결함 — 내용검출
    inner-crop 의 off-center 가 offset 을 오염시키던 경로 — 를 통째로 없앤다.
    """
    h, w = shape_hw[:2]
    bcx, bcy = _cond_box_center(box_ltrb)
    return (int(round(w / 2.0 - bcx)), int(round(h / 2.0 - bcy)))


def cond_offset_norm(box_ltrb, shape_hw):
    """|offset| 를 이미지 *대각선* 으로 정규화(crop 무관 척도)."""
    h, w = shape_hw[:2]
    dx, dy = cond_align_offset(box_ltrb, shape_hw)
    diag = float(np.hypot(w, h)) or 1.0
    return float(np.hypot(dx, dy) / diag)


def check_cond_box(box_ltrb, shape_hw):
    """cond box 가 template 으로 쓸만한지 가드. 반환 (status, reason, offset_norm).

    status: 'ok' | 'warn' | 'skip'. inner = min(box변) - 2·CROP_INSET_PX(대칭 inset 후).
    skip 우선순위: degenerate → out_of_bounds → too_small → offset_too_far.
    """
    h, w = shape_hw[:2]
    x, y, bw, bh = _cond_box_to_xywh(box_ltrb)
    onorm = cond_offset_norm(box_ltrb, shape_hw)
    if bw <= 0 or bh <= 0:
        return "skip", "box:degenerate", onorm
    if x < 0 or y < 0 or x + bw > w or y + bh > h:
        return "skip", "box:out_of_bounds", onorm
    inner = min(bw, bh) - 2 * CROP_INSET_PX
    if inner < MIN_INNER_PX:
        return "skip", "box:too_small", onorm
    if onorm > OFFSET_SKIP:
        return "skip", "offset:too_far", onorm
    if onorm > OFFSET_WARN:
        return "warn", "offset:far", onorm
    if inner < WARN_INNER_PX:
        return "warn", "box:small", onorm
    return "ok", "ok", onorm


def cond_template_crop(gray, cond, *, inset=CROP_INSET_PX):
    """cond box stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 해 template crop.

    대칭 inset → crop 중심 == box 중심 → cond_align_offset 과 정확히 일관.
    inset 후 너무 작아지면 inset 을 생략(작은 box 보호). 반환 (crop, (x0,y0,w,h)).

    **box stroke 만 지운다.** rcp cond 에 crosshair 가 있어도 그건 box 내부를 가로지르는
    *실제 내용* 이므로 inpaint 하면 매칭 신호가 깎인다 → crosshair_xy=None 으로 마스킹해
    box 테두리만 제거한다(msr 프레임의 crosshair 제거는 별개 — 거기선 distractor 라 지움).
    """
    box_only = CondInfo(scope=cond.scope, pixel=cond.pixel,
                        box_ltrb=cond.box_ltrb, crosshair_xy=None)
    cleaned = clean_image(gray, box_only)        # 튜닝된 1/1/2 로 box stroke 만 제거.
    x, y, bw, bh = _cond_box_to_xywh(cond.box_ltrb)
    h, w = gray.shape[:2]
    x0, y0 = max(0, x + inset), max(0, y + inset)
    x1, y1 = min(w, x + bw - inset), min(h, y + bh - inset)
    if x1 - x0 < MIN_INNER_PX or y1 - y0 < MIN_INNER_PX:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
    return cleaned[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def centered_area_crop(gray, area_ratio=CENTER_AREA_RATIO):
    """이미지 중심 기준 *면적 비율* crop (검증된 box-없음 fallback; offset 은 호출부에서 (0,0)).

    각 변 = sqrt(area_ratio) 비율(aspect 유지), 변 길이 하한 32 px. align point 가 이미지
    중심이라는 사전지식을 살려 매칭을 중심부에 집중시킨다. align_point_correction 의
    _centered_area_crop_bbox 와 동일 기하를 재사용한다(중복 방지; lazy import 로 로드 순서 무관).
    """
    from poc.workflow_3.vision.align_point_correction import _centered_area_crop_bbox

    x, y, cw, ch = _centered_area_crop_bbox(gray, area_ratio)
    return gray[y:y + ch, x:x + cw].copy()
