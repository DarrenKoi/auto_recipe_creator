"""live SEM box 위치 detector — offline(캡처 파일)과 online(RCS tool 창 방문 시)
양쪽이 공유하는 단일 진실 소스.

역할 분담(workflow_2 규칙): VLM(ui-venus) 은 SEM Monitor Box 를 coarse bbox 로만
제안하고, CV 가 좌표를 확정한다.
  1. VLM 이 coarse bbox 제안 (프롬프트는 ``vlm_sem_monitor_box`` 재사용).
  2. 네 변을 band 안에서 프레임 회색((170~190) 무채색) 의 '끊김 없는 긴 직선 run'
     으로 snap. 프레임 색이 약하면 Sobel projection peak 로 폴백.
  3. box 내부 Laplacian 분산으로 sharpness 측정 → total blur 후보 표시.

``detect_sem_box(image, client)`` 가 진입점이다. ``image`` 는 PIL Image 이므로
``capture_window()`` (Remote Monitoring System 창 캡처)의 반환값을 그대로 넣을 수
있고, 캡처 파일을 ``Image.open`` 한 것도 넣을 수 있다.

임계값/색 band 튜닝은 *이 파일에서* 한다(여기가 단일 소스). overlay·디버그
이미지·reference 저장 같은 부가 기능은 ``outline_live_sem_box.py`` 가 담당한다.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from poc.workflow_1.util import env_int
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.vlm_client import Workflow1VLMClient
from poc.workflow_2.vlm_sem_monitor_box import _run_sem_box_detection

# ====================================================================
# 튜닝 상수 — CLAUDE.md 규칙상 argparse 미사용, 상수/환경변수로만 조정.
# ====================================================================

# edge-snap 탐색 band — box 한 변 길이 대비 비율. 이 band 안에서만 변을 옮기므로
# VLM 이 크게 빗나가지 않는 한 엉뚱한 edge 로 도망가지 않는다.
EDGE_SNAP_BAND_RATIO = 0.06
EDGE_SNAP_BAND_MIN_PX = 6

# Laplacian 분산이 이 값 미만이면 'total blur → 클릭 금지' 후보로 표시.
# 실데이터로 보정 필요(콜드스타트 임계값).
SHARPNESS_BLUR_THRESHOLD = 60.0

# SEM box 외곽 프레임 색(사용자 관측): (170~190) 부근의 무채색 회색.
# edge-snap 시 이 색을 띤 '긴 직선 run' 을 1차 단서로 쓰고, band 안에 프레임 색이
# 충분치 않으면 Sobel gradient peak 로 폴백한다. 모두 콜드스타트값(실데이터 보정 필요).
GREY_FRAME_LO = 160          # 프레임 회색 밝기 하한.
GREY_FRAME_HI = 200          # 프레임 회색 밝기 상한.
GREY_FRAME_CHROMA_TOL = 20   # 채널 최대-최소 편차 허용치(무채색 판정).
GREY_FRAME_MIN_FRAC = 0.5    # 프레임 회색이 변 길이의 이 비율 이상 이어져야 색 단서를 신뢰.

# 환경변수 override (실데이터 보정 시 코드 수정 없이 조정).
GREY_FRAME_LO = env_int("SEM_BOX_GREY_LO", GREY_FRAME_LO)
GREY_FRAME_HI = env_int("SEM_BOX_GREY_HI", GREY_FRAME_HI)
GREY_FRAME_CHROMA_TOL = env_int("SEM_BOX_GREY_CHROMA_TOL", GREY_FRAME_CHROMA_TOL)


@dataclass
class SemBoxDetection:
    """한 이미지에 대한 live SEM box 검출 결과(좌표는 이미지 픽셀 기준)."""

    detected: bool              # VLM coarse 검출 성공 여부.
    width: int
    height: int
    bbox_px: dict | None        # CV edge-snap 확정 박스, 픽셀 좌표.
    bbox_1000: dict | None       # 창 크기 기준 0-1000 정규화(해상도 무관 비교용).
    sharpness: float | None     # box 내부 Laplacian 분산.
    blurry: bool                # sharpness < 임계값 → 클릭 금지 후보.
    mode_label: str | None      # 박스 상단 모드 라벨(Optics / OM 등).
    confidence: float | None    # VLM coarse confidence (좌표 신뢰엔 쓰지 않음).
    vlm_bbox_px: dict | None     # VLM coarse 박스(snap 전), 픽셀 좌표.


# ------------------------------------------------------------------
# 좌표 정규화.
# ------------------------------------------------------------------


def bbox_px_to_1000(bbox_px: dict, width: int, height: int) -> dict:
    """픽셀 bbox 를 이미지(창) 크기 기준 0-1000 정규화 bbox 로 바꾼다.

    해상도/창 크기가 달라도 같은 기준으로 비교할 수 있어, 저장된 reference 와
    현재 검출 박스를 맞대보는 production 모니터링에 쓴다.
    """
    def _n(value: float, size: int) -> int:
        return int(round(max(0.0, min(1000.0, value / max(1, size) * 1000.0))))

    return {
        "left": _n(bbox_px["left"], width),
        "top": _n(bbox_px["top"], height),
        "right": _n(bbox_px["right"], width),
        "bottom": _n(bbox_px["bottom"], height),
    }


# ------------------------------------------------------------------
# CV edge-snap — VLM coarse bbox 의 네 변을 프레임 회색/edge 로 정렬.
# ------------------------------------------------------------------


def grey_frame_mask(bgr: np.ndarray) -> np.ndarray:
    """프레임으로 쓰이는 밝은 무채색 회색 픽셀 mask(uint8 0/1).

    SEM box 외곽 프레임은 (170~190) 부근의 회색이다(사용자 관측). 채널 간 편차가
    작고(무채색) 밝기가 회색 band 안인 픽셀만 1 로 둔다. 컬러 텍스트/신호등 같은
    유채색 UI 요소나 어두운 라이브 영상은 자연히 0 이 된다.
    """
    bgr_i = bgr.astype(np.int16)
    lo = bgr_i.min(axis=2)
    hi = bgr_i.max(axis=2)
    chroma = hi - lo
    mean = bgr_i.mean(axis=2)
    mask = (
        (chroma <= GREY_FRAME_CHROMA_TOL)
        & (mean >= GREY_FRAME_LO)
        & (mean <= GREY_FRAME_HI)
    )
    return mask.astype(np.uint8)


def _longest_true_run(mask_1d: np.ndarray) -> int:
    """1D 불리언 배열에서 연속 True 의 최대 길이.

    프레임 한 변은 '끊김 없는 긴 회색 직선' 이다. 단순 합(sum)은 흩어진 회색
    픽셀(내부 텍스처·인접 패널)도 높게 세지만, 최대 연속 run 은 진짜 직선만
    크게 잡으므로 변 판정의 변별력이 훨씬 높다.
    """
    if not mask_1d.any():
        return 0
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int((ends - starts).max())


def snap_box_to_edges(gray: np.ndarray, grey_mask: np.ndarray, bbox: dict) -> dict:
    """``bbox`` 네 변을 각각 band 안에서 프레임 회색 run(1차) 또는 Sobel peak(폴백)로 옮긴다.

    프레임은 box 변을 가로지르는 '끊김 없는 긴 회색 직선' 이므로, band 안에서
    box 폭/높이에 걸친 *연속 회색 run* 이 가장 긴 행/열을 우선 고른다. 동률이면
    VLM 추정선에 가장 가까운 것을 골라 box 근처에 머무른다. 이 색 단서가
    약하면(겹친 컨트롤 패널·텍스처처럼 프레임이 안 보이면) 기존 Sobel gradient
    peak 로 폴백한다. band 밖으로는 나가지 않는다.
    """
    h, w = gray.shape[:2]
    left = int(np.clip(bbox["left"], 0, w - 2))
    top = int(np.clip(bbox["top"], 0, h - 2))
    right = int(np.clip(bbox["right"], left + 1, w - 1))
    bottom = int(np.clip(bbox["bottom"], top + 1, h - 1))

    box_w = right - left
    box_h = bottom - top
    band_x = max(EDGE_SNAP_BAND_MIN_PX, int(round(box_w * EDGE_SNAP_BAND_RATIO)))
    band_y = max(EDGE_SNAP_BAND_MIN_PX, int(round(box_h * EDGE_SNAP_BAND_RATIO)))

    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    methods: list[str] = []

    def _pick_closest(cands: np.ndarray, target: int) -> int:
        """동률 후보(band 내 인덱스) 중 ``target`` 에 가장 가까운 것."""
        return int(cands[int(np.argmin(np.abs(cands - target)))])

    def _snap_horizontal(edge_row: int) -> int:
        lo = max(0, edge_row - band_y)
        hi = min(h, edge_row + band_y + 1)
        if hi - lo < 2:
            methods.append("none")
            return edge_row
        # 1차: box 폭에 걸친 연속 회색 run 이 가장 긴 행(동률이면 VLM 추정선에 근접한 행).
        band = grey_mask[lo:hi, left:right].astype(bool)
        runs = np.array([_longest_true_run(band[i]) for i in range(band.shape[0])])
        best = int(runs.max())
        if best >= GREY_FRAME_MIN_FRAC * box_w:
            methods.append("grey")
            cands = np.flatnonzero(runs == best)
            return lo + _pick_closest(cands, edge_row - lo)
        # 폴백: 가로 edge 강도 합.
        methods.append("grad")
        strength = grad_y[lo:hi, left:right].sum(axis=1)
        return lo + int(np.argmax(strength))

    def _snap_vertical(edge_col: int) -> int:
        lo = max(0, edge_col - band_x)
        hi = min(w, edge_col + band_x + 1)
        if hi - lo < 2:
            methods.append("none")
            return edge_col
        band = grey_mask[top:bottom, lo:hi].astype(bool)
        runs = np.array([_longest_true_run(band[:, j]) for j in range(band.shape[1])])
        best = int(runs.max())
        if best >= GREY_FRAME_MIN_FRAC * box_h:
            methods.append("grey")
            cands = np.flatnonzero(runs == best)
            return lo + _pick_closest(cands, edge_col - lo)
        methods.append("grad")
        strength = grad_x[top:bottom, lo:hi].sum(axis=0)
        return lo + int(np.argmax(strength))

    new_top = _snap_horizontal(top)
    new_bottom = _snap_horizontal(bottom)
    new_left = _snap_vertical(left)
    new_right = _snap_vertical(right)

    # 변이 교차하지 않도록 정렬.
    if new_bottom <= new_top:
        new_top, new_bottom = top, bottom
    if new_right <= new_left:
        new_left, new_right = left, right

    print(f"[INFO] edge-snap 방법: T={methods[0]} B={methods[1]} L={methods[2]} R={methods[3]}")
    return {"left": new_left, "top": new_top, "right": new_right, "bottom": new_bottom}


def sharpness_in_box(gray: np.ndarray, bbox: dict) -> float:
    """box 내부의 Laplacian 분산(focus measure). 클수록 선명, 0 근처면 total blur."""
    left = int(np.clip(bbox["left"], 0, gray.shape[1] - 1))
    top = int(np.clip(bbox["top"], 0, gray.shape[0] - 1))
    right = int(np.clip(bbox["right"], left + 1, gray.shape[1]))
    bottom = int(np.clip(bbox["bottom"], top + 1, gray.shape[0]))
    roi = gray[top:bottom, left:right]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


# ------------------------------------------------------------------
# 진입점.
# ------------------------------------------------------------------


def detect_sem_box(image: "Image.Image", client: Workflow1VLMClient) -> SemBoxDetection:
    """PIL 이미지에서 live SEM box 위치를 검출한다(VLM coarse → CV snap → sharpness).

    ``image`` 는 ``capture_window()`` 의 반환값(또는 ``Image.open`` 한 캡처)을 그대로
    넣으면 된다. VLM coarse 가 박스를 못 잡으면 ``detected=False`` 로 돌려준다.
    """
    rgb = image.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    image_b64, vlm_w, vlm_h = encode_image_webp(rgb, quality=90)
    payload, vlm_bbox = _run_sem_box_detection(
        image_b64=image_b64, width=vlm_w, height=vlm_h, client=client
    )

    if vlm_bbox is None:
        return SemBoxDetection(
            detected=False, width=w, height=h, bbox_px=None, bbox_1000=None,
            sharpness=None, blurry=False,
            mode_label=payload.get("mode_label"), confidence=payload.get("confidence"),
            vlm_bbox_px=None,
        )

    grey_mask = grey_frame_mask(bgr)
    cv_bbox = snap_box_to_edges(gray, grey_mask, vlm_bbox)
    sharpness = sharpness_in_box(gray, cv_bbox)
    return SemBoxDetection(
        detected=True,
        width=w,
        height=h,
        bbox_px=cv_bbox,
        bbox_1000=bbox_px_to_1000(cv_bbox, w, h),
        sharpness=sharpness,
        blurry=sharpness < SHARPNESS_BLUR_THRESHOLD,
        mode_label=payload.get("mode_label"),
        confidence=payload.get("confidence"),
        vlm_bbox_px=vlm_bbox,
    )
