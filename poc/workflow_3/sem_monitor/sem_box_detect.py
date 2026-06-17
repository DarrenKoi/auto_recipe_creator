"""live SEM box 위치 detector + PM 박스 기반 OM/SEM 모드 판독 (workflow_3 포팅본).

workflow_2 의 검증된 detector(`poc/workflow_2/sem_box_detect.py` +
`poc/workflow_2/vlm_sem_monitor_box.py`)를 workflow_3 로 이식한 단일 소스다.
레이어 규칙상 workflow_3 는 workflow_2 를 import 하지 않으므로(역방향만 허용) CV/VLM
경로를 여기에 직접 둔다.

역할 분담(불변): VLM(ui-venus) 은 SEM Monitor Box 를 coarse bbox 로만 제안하고, CV 가
좌표를 확정한다.
  1. VLM 이 coarse bbox + PM 박스 텍스트를 제안.
  2. 네 변을 band 안에서 프레임 회색((170~190) 무채색) '끊김 없는 긴 직선 run' 으로 snap.
     프레임 색이 약하면 Sobel projection peak 로 폴백.
  3. box 내부 Laplacian 분산으로 sharpness 측정 → total blur 후보 표시.
  4. PM 박스 텍스트 → OM/SEM 모드 추론(`pm_text_to_mode`): 104/210 → OM, 'K' 포함 → SEM.

`detect_sem_box(image, client)` 가 진입점이다. ``image`` 는 PIL Image 이므로
``capture_window()`` 반환값을 그대로 넣을 수 있고, 캡처 파일을 ``Image.open`` 한 것도 된다.

임계값/색 band/PM OM 값 집합 튜닝은 *이 파일에서* 한다(여기가 단일 소스).
"""

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from poc.workflow_3.util import env_int
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

LOG_COMPONENT = "sem_box_detect"

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
GREY_FRAME_LO = 160          # 프레임 회색 밝기 하한.
GREY_FRAME_HI = 200          # 프레임 회색 밝기 상한.
GREY_FRAME_CHROMA_TOL = 20   # 채널 최대-최소 편차 허용치(무채색 판정).
GREY_FRAME_MIN_FRAC = 0.5    # 프레임 회색이 변 길이의 이 비율 이상 이어져야 색 단서를 신뢰.

# 환경변수 override (실데이터 보정 시 코드 수정 없이 조정).
GREY_FRAME_LO = env_int("SEM_BOX_GREY_LO", GREY_FRAME_LO)
GREY_FRAME_HI = env_int("SEM_BOX_GREY_HI", GREY_FRAME_HI)
GREY_FRAME_CHROMA_TOL = env_int("SEM_BOX_GREY_CHROMA_TOL", GREY_FRAME_CHROMA_TOL)


def _default_pm_om_values() -> frozenset:
    """PM 박스가 이 정수 값이면 OM 모드(사용자 관측: 104, 210). env 로 조정 가능.

    ``SEM_BOX_PM_OM_VALUES`` 에 콤마구분 정수(예: "104,210,420")를 주면 대체한다.
    """
    raw = os.environ.get("SEM_BOX_PM_OM_VALUES", "").strip()
    if not raw:
        return frozenset({104, 210})
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return frozenset(out) if out else frozenset({104, 210})


PM_OM_VALUES = _default_pm_om_values()


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
    pm_text: str | None         # PM 박스에서 읽은 텍스트(배율 readout) 원문.
    pm_mode: str | None         # PM 텍스트 → "OM" | "SEM" | None(모호).


# ------------------------------------------------------------------
# PM 박스 → OM/SEM 모드.
# ------------------------------------------------------------------


def pm_text_to_mode(pm_text) -> str | None:
    """PM 박스 텍스트(배율 readout) → "OM" | "SEM" | None.

    사용자 관측 규칙: PM 값이 104/210(PM_OM_VALUES) 같은 정수면 OM, 숫자에 'K' 가 붙은
    배율(예: '30K', '20 K')이면 SEM. 둘 다 아니면 None(모호 → 호출부가 점수 폴백).

    OCR 노이즈에 견고하도록 보수적으로 판정한다(오독 시 None=안전 폴백):
      * SEM 은 *숫자 뒤(공백 허용) K* 패턴만 인정 — 'PARK'/'OK' 같은 글자-only K 배제.
      * OM 은 공백 제거 후 *순수 숫자* 이며 PM_OM_VALUES 에 속할 때만 — 'PM104' 처럼
        글자가 섞이면 거부한다. (str 이 아닌 입력은 방어적으로 None.)
    """
    if not isinstance(pm_text, str):
        return None
    text = pm_text.strip().upper()
    if not text:
        return None
    if re.search(r"\d\s*K", text):
        return "SEM"
    compact = text.replace(" ", "")
    if compact.isdigit() and int(compact) in PM_OM_VALUES:
        return "OM"
    return None


# ------------------------------------------------------------------
# 좌표 정규화.
# ------------------------------------------------------------------


def bbox_px_to_1000(bbox_px: dict, width: int, height: int) -> dict:
    """픽셀 bbox 를 이미지(창) 크기 기준 0-1000 정규화 bbox 로 바꾼다."""
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
    작고(무채색) 밝기가 회색 band 안인 픽셀만 1 로 둔다.
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

    단순 합(sum)은 흩어진 회색 픽셀(내부 텍스처·인접 패널)도 높게 세지만, 최대
    연속 run 은 진짜 직선만 크게 잡으므로 변 판정의 변별력이 훨씬 높다.
    """
    if not mask_1d.any():
        return 0
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int((ends - starts).max())


def snap_box_to_edges(gray: np.ndarray, grey_mask: np.ndarray, bbox: dict) -> dict:
    """``bbox`` 네 변을 각각 band 안에서 프레임 회색 run(1차) 또는 Sobel peak(폴백)로 옮긴다."""
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

    methods: list = []

    def _pick_closest(cands: np.ndarray, target: int) -> int:
        """동률 후보(band 내 인덱스) 중 ``target`` 에 가장 가까운 것."""
        return int(cands[int(np.argmin(np.abs(cands - target)))])

    def _snap_horizontal(edge_row: int) -> int:
        lo = max(0, edge_row - band_y)
        hi = min(h, edge_row + band_y + 1)
        if hi - lo < 2:
            methods.append("none")
            return edge_row
        band = grey_mask[lo:hi, left:right].astype(bool)
        runs = np.array([_longest_true_run(band[i]) for i in range(band.shape[0])])
        best = int(runs.max())
        if best >= GREY_FRAME_MIN_FRAC * box_w:
            methods.append("grey")
            cands = np.flatnonzero(runs == best)
            return lo + _pick_closest(cands, edge_row - lo)
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
# VLM coarse — SEM Monitor Box bbox + PM 박스 텍스트.
# ------------------------------------------------------------------


def _sem_box_system_prompt() -> str:
    """SEM Monitor Box + PM 박스 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a Windows CD-SEM Tool application. "
        "Return strict JSON only. "
        "Locate the 'SEM Monitor Box' - a single rectangular region that shows the "
        "LIVE, actively-updating grayscale electron-microscope image of the wafer "
        "or sample being scanned. It looks like a dark, noisy, real-time SEM video "
        "feed showing wafer patterns, NOT a color photograph and NOT a static UI panel.\n"
        "Anchors for identification:\n"
        "  - At the TOP of this box there is a short text label that names the current "
        "monitoring mode. Typical values are 'Optics' or 'OM'. Find this label first; "
        "the SEM Monitor Box is the large rectangular live-image region directly "
        "underneath it.\n"
        "  - Floating control panels labelled 'Optics', 'Function', 'AMP', 'Next', "
        "'DDS' (and similar short SEM operation commands) are drawn ON TOP of this "
        "box, partially covering the live wafer image. These overlay panels are a "
        "STRONG SECONDARY CLUE that you are looking at the SEM Monitor Box, but they "
        "are NOT the box itself and the bbox must NOT shrink to fit around them.\n"
        "  - Near the SEM Monitor Box there is a small 'PM' box showing a short "
        "magnification readout (e.g. '104', '210', or a value containing 'K' such as "
        "'30K' or '20 K'). Read its text verbatim, including any space before 'K'.\n"
        "The returned bbox must enclose the FULL rectangle of the live wafer view, "
        "INCLUDING the parts that are currently hidden behind the overlay control "
        "panels. Estimate the underlying rectangle from the visible live-image edges "
        "and the position of the mode label at the top. "
        "Do NOT return a bbox of just the overlay control panels. "
        "Do NOT return a bbox of only the uncovered slivers of the live image. "
        "Do not include unrelated panels, separate toolbars, the window title bar, or "
        "other tabs of the application. "
        "If no live wafer-pattern region is visible at all, set panel_visible=false."
    )


def _sem_box_user_prompt() -> str:
    """SEM Monitor Box + PM 박스 탐지 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "panel_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "panel_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "mode_label": "Optics",\n'
        '  "pm_box_text": "104",\n'
        '  "overlay_panels_seen": ["Optics", "Function", "AMP", "Next", "DDS"],\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string explaining what you used to identify the region"\n'
        "}\n"
        "panel_bbox must tightly enclose the ENTIRE underlying live wafer-pattern "
        "rectangle as one box, INCLUDING any portion currently hidden behind the "
        "floating control panels overlaid on top of it. "
        "mode_label is the short text you actually read at the top of the SEM "
        "Monitor Box (expected values: 'Optics' or 'OM'); use null if you cannot "
        "read it. "
        "pm_box_text is the short magnification readout you actually read in the 'PM' "
        "box near the SEM Monitor Box (expected examples: '104', '210', '30K', "
        "'20 K'); read it verbatim including any space; use null if you cannot read it. "
        "overlay_panels_seen should list 1~5 of the floating control panel labels "
        "you actually read covering the live image (expected examples: 'Optics', "
        "'Function', 'AMP', 'Next', 'DDS'). "
        "If no live wafer-pattern region is visible, set panel_visible=false, "
        "panel_bbox=null, mode_label=null, pm_box_text=null, overlay_panels_seen=[]."
    )


def _run_sem_box_detection(
    *,
    image_b64: str,
    width: int,
    height: int,
    client: Workflow1VLMClient,
) -> tuple:
    """SEM monitor panel 의 bbox + PM 텍스트를 탐지한다. (payload, bbox_px|None) 반환."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_sem_box_system_prompt(),
        user_text=_sem_box_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("panel_visible") is not True:
        return parsed, None

    bbox_1000 = normalize_bbox_1000(parsed.get("panel_bbox"))
    if bbox_1000 is None:
        return parsed, None
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


# ------------------------------------------------------------------
# 진입점.
# ------------------------------------------------------------------


def detect_sem_box(image: "Image.Image", client: Workflow1VLMClient) -> SemBoxDetection:
    """PIL 이미지에서 live SEM box 위치 + PM 모드를 검출한다(VLM coarse → CV snap → sharpness).

    VLM coarse 가 박스를 못 잡으면 ``detected=False`` 로 돌려준다(이 때도 PM 텍스트는
    읽혔으면 채워 mode 폴백에 쓸 수 있게 한다).
    """
    rgb = image.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    image_b64, vlm_w, vlm_h = encode_image_webp(rgb, quality=90)
    payload, vlm_bbox = _run_sem_box_detection(
        image_b64=image_b64, width=vlm_w, height=vlm_h, client=client
    )

    # VLM 이 pm_box_text 를 문자열 대신 숫자/리스트로 줄 수 있다. 스칼라(숫자)는 문자열로
    # 정규화하고, list/dict/bool 등 비스칼라는 None 으로 버린다(.strip() 크래시 방지 +
    # 리스트 str 화로 인한 가짜 숫자 매칭 방지). pm_text_to_mode 도 내부에서 재방어한다.
    pm_raw = payload.get("pm_box_text")
    if isinstance(pm_raw, str):
        pm_text = pm_raw
    elif isinstance(pm_raw, (int, float)) and not isinstance(pm_raw, bool):
        pm_text = str(pm_raw)
    else:
        pm_text = None
    pm_mode = pm_text_to_mode(pm_text)

    if vlm_bbox is None:
        return SemBoxDetection(
            detected=False, width=w, height=h, bbox_px=None, bbox_1000=None,
            sharpness=None, blurry=False,
            mode_label=payload.get("mode_label"), confidence=payload.get("confidence"),
            vlm_bbox_px=None, pm_text=pm_text, pm_mode=pm_mode,
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
        pm_text=pm_text,
        pm_mode=pm_mode,
    )


__all__ = [
    "SemBoxDetection",
    "detect_sem_box",
    "pm_text_to_mode",
    "grey_frame_mask",
    "snap_box_to_edges",
    "sharpness_in_box",
    "bbox_px_to_1000",
    "PM_OM_VALUES",
]
