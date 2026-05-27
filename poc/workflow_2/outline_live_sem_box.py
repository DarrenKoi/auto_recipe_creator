"""captured_img_from_rcs 의 RCS 스크린샷에서 'live SEM box' 외곽선을 정밀하게
그려보는 단독 테스트 스크립트.

목적(사용자 확정): VLM 클릭점 테스트에 들어가기 *전에*, "live SEM box 의 경계선을
좌표 혼동 없이 정확히 그릴 수 있는가" 를 먼저 눈으로 확인한다. 클릭/줌아웃/더블클릭
같은 후속 행동은 모두 이 box 좌표를 기준으로 하므로, 경계가 부정확하면 전부 어긋난다.

파이프라인 (역할 분담은 workflow_2 설계 규칙과 동일 — VLM 은 영역만, CV 가 좌표를 확정):
  1. VLM(ui-venus) 이 SEM Monitor Box 를 coarse bbox 로 제안한다.
     (프롬프트는 ``vlm_sem_monitor_box`` 의 것을 재사용)
  2. 그 bbox 네 변을 각각 band 안에서 가장 강한 직선 edge(Sobel projection peak)로
     snap 해 픽셀 단위로 정렬한다.
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


def _snap_box_to_edges(gray: np.ndarray, bbox: dict) -> dict:
    """``bbox`` 네 변을 각각 band 안에서 Sobel projection peak 로 옮긴다.

    top/bottom 은 가로 edge(grad_y), left/right 는 세로 edge(grad_x) 를 본다.
    band 밖으로는 나가지 않으므로 VLM 추정 근처에 머무른다.
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

    def _snap_horizontal(edge_row: int) -> int:
        lo = max(0, edge_row - band_y)
        hi = min(h, edge_row + band_y + 1)
        if hi - lo < 2:
            return edge_row
        # 각 후보 행에서 box 폭에 걸친 가로 edge 강도 합.
        strength = grad_y[lo:hi, left:right].sum(axis=1)
        return lo + int(np.argmax(strength))

    def _snap_vertical(edge_col: int) -> int:
        lo = max(0, edge_col - band_x)
        hi = min(w, edge_col + band_x + 1)
        if hi - lo < 2:
            return edge_col
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

    cv_bbox = None
    sharpness = None
    blurry = False
    if vlm_bbox is not None:
        cv_bbox = _snap_box_to_edges(gray, vlm_bbox)
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
