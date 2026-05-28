"""captured_img_from_rcs 의 RCS 스크린샷에서 'live SEM box' 외곽선을 정밀하게
그려보는 단독 테스트 스크립트.

목적(사용자 확정): VLM 클릭점 테스트에 들어가기 *전에*, "live SEM box 의 경계선을
좌표 혼동 없이 정확히 그릴 수 있는가" 를 먼저 눈으로 확인한다. 클릭/줌아웃/더블클릭
같은 후속 행동은 모두 이 box 좌표를 기준으로 하므로, 경계가 부정확하면 전부 어긋난다.

파이프라인 (역할 분담은 workflow_2 설계 규칙과 동일 — VLM 은 영역만, CV 가 좌표를 확정):
  1. VLM(ui-venus) 이 SEM Monitor Box 를 coarse bbox 로 제안한다.
     (프롬프트는 ``vlm_sem_monitor_box`` 의 것을 재사용)
  2. 그 bbox 네 변을 각각 band 안에서 프레임 회색((170~190) 무채색) 의 '긴 직선 run'
     으로 snap 한다. 프레임 색이 약하면 Sobel projection peak 로 폴백해 픽셀 단위로 정렬한다.
  3. box 내부의 Laplacian 분산으로 sharpness 를 재서, 'total blur → 클릭 금지' 후보를
     overlay 에 표시한다(클릭 대신 zoom-out/이동 판단의 1차 근거).

입력: ``ALIGN_IMAGES_ROOT/*/*/*/captured_img_from_rcs/<tag>/<tag>_rcs.jpg``
      (RCS_CAPTURE_DIR 환경변수로 임의 폴더를 줄 수도 있다.)
출력: ``debug_images/outline_live_sem_box/<tag>/`` 에 overlay JPEG + per-image JSON + summary.

실행:
    uv run python poc/workflow_2/outline_live_sem_box.py
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_1.util import env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.vlm_client import Workflow1VLMClient
from poc.workflow_2.vlm_sem_monitor_box import _run_sem_box_detection

load_dotenv()

LOG_NAME = "outline_live_sem_box"
CAPTURED_RCS_DIRNAME = "captured_img_from_rcs"

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수/환경변수로만 조정.
# ====================================================================

# 임의 캡처 폴더 직접 지정(재귀적으로 *_rcs.jpg 수집). 비우면 ALIGN_IMAGES_ROOT 자동 탐색.
RCS_CAPTURE_DIR_OVERRIDE = os.getenv("RCS_CAPTURE_DIR", "").strip()

# 처리할 최대 캡처 장수(VLM 호출 비용 상한). 0 이하면 전체 처리. mtime 최신순으로 자른다.
SAMPLE_LIMIT = env_int("RCS_OUTLINE_SAMPLE_LIMIT", 0)

DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME

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

# 프레임 회색 mask 를 별도 디버그 이미지(<stem>_greymask.jpg)로 저장할지.
# 색 band/임계값을 실데이터로 보정할 때 눈으로 확인하는 용도. 0 으로 끄면 생략.
SAVE_GREY_MASK_DEBUG = env_int("RCS_OUTLINE_SAVE_GREY_MASK", 1) == 1

# overlay 색상 (BGR).
_VLM_COLOR = (255, 0, 255)   # magenta — VLM coarse
_CV_COLOR = (255, 255, 0)    # cyan — CV-snapped
_OK_COLOR = (60, 200, 60)
_BLUR_COLOR = (60, 60, 230)


@dataclass
class OutlineReport:
    """한 캡처 이미지에 대한 SEM box 외곽선 결과."""

    image_path: str
    width: int
    height: int
    vlm_detected: bool
    vlm_bbox: dict | None       # coarse, 픽셀 좌표.
    cv_bbox: dict | None        # edge-snap 후, 픽셀 좌표.
    mode_label: str | None
    vlm_confidence: float | None
    sharpness: float | None     # box 내부 Laplacian 분산.
    blurry: bool                # sharpness < 임계값 → 클릭 금지 후보.
    overlay_path: str


# ------------------------------------------------------------------
# 입력 해석.
# ------------------------------------------------------------------


def _resolve_capture_paths() -> list[Path]:
    """처리할 *_rcs.jpg 캡처 경로들을 mtime 최신순으로 모은다."""
    if RCS_CAPTURE_DIR_OVERRIDE:
        root = Path(RCS_CAPTURE_DIR_OVERRIDE).expanduser()
        if not root.is_dir():
            print(f"[ERROR] RCS_CAPTURE_DIR 가 폴더가 아닙니다: {root}")
            return []
        paths = sorted(root.rglob("*_rcs.jpg"))
        if not paths:  # 네이밍이 다른 경우 일반 jpg 로 폴백.
            paths = sorted(root.rglob("*.jpg"))
        print(f"[INFO] RCS_CAPTURE_DIR 사용: {root} ({len(paths)} 장)")
    else:
        if not ALIGN_IMAGES_ROOT.is_dir():
            print(f"[ERROR] ALIGN_IMAGES_ROOT 가 없습니다: {ALIGN_IMAGES_ROOT}")
            return []
        pattern = f"*/*/*/{CAPTURED_RCS_DIRNAME}/*/*_rcs.jpg"
        paths = list(ALIGN_IMAGES_ROOT.glob(pattern))
        print(f"[INFO] ALIGN_IMAGES_ROOT 자동 탐색: {len(paths)} 장 발견")

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if SAMPLE_LIMIT > 0 and len(paths) > SAMPLE_LIMIT:
        print(f"[INFO] 최신 {SAMPLE_LIMIT} 장으로 제한 (전체 {len(paths)})")
        paths = paths[:SAMPLE_LIMIT]
    return paths


# ------------------------------------------------------------------
# CV edge-snap — VLM coarse bbox 의 네 변을 강한 직선 edge 로 정렬.
# ------------------------------------------------------------------


def _grey_frame_mask(bgr: np.ndarray) -> np.ndarray:
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


def _render_grey_mask_debug(bgr: np.ndarray, grey_mask: np.ndarray) -> np.ndarray:
    """프레임 회색으로 검출된 픽셀을 원본 위에 초록으로 덧칠한 디버그 이미지.

    원본을 어둡게 깔고 mask=1 인 픽셀만 초록으로 강조한다. 색 band/임계값이
    실제 프레임을 제대로 잡는지(또는 엉뚱한 UI 회색까지 잡는지) 눈으로 본다.
    """
    dim = (bgr * 0.4).astype(np.uint8)
    green = np.zeros_like(bgr)
    green[..., 1] = 255  # BGR 초록.
    sel = grey_mask.astype(bool)
    out = dim.copy()
    out[sel] = green[sel]
    return out


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


def _snap_box_to_edges(gray: np.ndarray, grey_mask: np.ndarray, bbox: dict) -> dict:
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


def _sharpness_in_box(gray: np.ndarray, bbox: dict) -> float:
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
# overlay 그리기.
# ------------------------------------------------------------------


def _draw_rect(img: np.ndarray, bbox: dict, color: tuple, label: str) -> None:
    p1 = (int(bbox["left"]), int(bbox["top"]))
    p2 = (int(bbox["right"]), int(bbox["bottom"]))
    cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
    cv2.putText(
        img, label, (p1[0] + 4, max(16, p1[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
    )


def _draw_overlay(
    bgr: np.ndarray,
    vlm_bbox: dict | None,
    cv_bbox: dict | None,
    sharpness: float | None,
    blurry: bool,
) -> np.ndarray:
    out = bgr.copy()
    if vlm_bbox is not None:
        _draw_rect(out, vlm_bbox, _VLM_COLOR, "VLM coarse")
    if cv_bbox is not None:
        _draw_rect(out, cv_bbox, _CV_COLOR, "CV snapped")
        cx = (cv_bbox["left"] + cv_bbox["right"]) // 2
        cy = (cv_bbox["top"] + cv_bbox["bottom"]) // 2
        cv2.drawMarker(out, (cx, cy), _CV_COLOR, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    # 상단 상태 배너 — sharpness 와 클릭 가능 여부.
    if sharpness is not None:
        verdict = "BLURRY (do NOT click)" if blurry else "sharp (clickable)"
        color = _BLUR_COLOR if blurry else _OK_COLOR
        cv2.putText(
            out, f"sharpness={sharpness:.1f}  {verdict}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
    return out


# ------------------------------------------------------------------
# 한 이미지 처리.
# ------------------------------------------------------------------


def _process_image(image_path: Path, client: Workflow1VLMClient, out_dir: Path) -> OutlineReport:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"이미지를 디코드하지 못했습니다: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    with Image.open(image_path) as pil_image:
        image_b64, vlm_w, vlm_h = encode_image_webp(pil_image.convert("RGB"), quality=90)

    payload, vlm_bbox = _run_sem_box_detection(
        image_b64=image_b64, width=vlm_w, height=vlm_h, client=client
    )

    grey_mask = _grey_frame_mask(bgr)
    if SAVE_GREY_MASK_DEBUG:
        cv2.imwrite(
            str(out_dir / f"{image_path.stem}_greymask.jpg"),
            _render_grey_mask_debug(bgr, grey_mask),
        )

    cv_bbox = None
    sharpness = None
    blurry = False
    if vlm_bbox is not None:
        cv_bbox = _snap_box_to_edges(gray, grey_mask, vlm_bbox)
        sharpness = _sharpness_in_box(gray, cv_bbox)
        blurry = sharpness < SHARPNESS_BLUR_THRESHOLD

    overlay = _draw_overlay(bgr, vlm_bbox, cv_bbox, sharpness, blurry)
    overlay_path = out_dir / f"{image_path.stem}_outline.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    return OutlineReport(
        image_path=str(image_path),
        width=w,
        height=h,
        vlm_detected=vlm_bbox is not None,
        vlm_bbox=vlm_bbox,
        cv_bbox=cv_bbox,
        mode_label=payload.get("mode_label"),
        vlm_confidence=payload.get("confidence"),
        sharpness=sharpness,
        blurry=blurry,
        overlay_path=str(overlay_path),
    )


def run() -> str:
    started = time.time()
    paths = _resolve_capture_paths()
    if not paths:
        print("[ERROR] 처리할 캡처 이미지가 없습니다.")
        return "no_captures"

    tag = make_timestamp_tag()
    out_dir = DEBUG_IMAGE_DIR / LOG_NAME / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE,
        model_name=DEFAULT_MODEL,
        log_name=LOG_NAME,
    )
    print(f"[INFO] SEM box 외곽선 테스트 시작: service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, {len(paths)} 장")

    reports: list[OutlineReport] = []
    for idx, path in enumerate(paths):
        try:
            report = _process_image(path, client, out_dir)
        except Exception as exc:
            print(f"[ERROR] 처리 실패: {path.name} ({exc})")
            continue
        reports.append(report)
        print(
            f"[INFO] {idx:02d} {path.name} vlm={'Y' if report.vlm_detected else 'N'} "
            f"mode={report.mode_label or '-'} "
            f"sharpness={report.sharpness if report.sharpness is None else round(report.sharpness, 1)} "
            f"blurry={report.blurry} cv_bbox={report.cv_bbox}"
        )

    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(reports),
        "vlm_detected": sum(1 for r in reports if r.vlm_detected),
        "blurry": sum(1 for r in reports if r.blurry),
        "sharpness_threshold": SHARPNESS_BLUR_THRESHOLD,
        "reports": [asdict(r) for r in reports],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[INFO] 완료: processed={len(reports)}/{len(paths)} "
        f"vlm_detected={summary['vlm_detected']} blurry={summary['blurry']} "
        f"elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if reports else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
