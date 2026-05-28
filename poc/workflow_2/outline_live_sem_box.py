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
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR, WORKFLOW_2_DIR
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_1.util import env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.vlm_client import Workflow1VLMClient
from poc.workflow_2.sem_box_detect import (
    SHARPNESS_BLUR_THRESHOLD,
    bbox_px_to_1000,
    detect_sem_box,
    grey_frame_mask,
)

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

# 검출 파이프라인(VLM coarse → grey-snap → sharpness)과 튜닝 상수는 sem_box_detect 가
# 단일 소스로 보유한다. 본 모듈은 그 위에 overlay·디버그·reference 저장만 얹는다.

# 프레임 회색 mask 를 별도 디버그 이미지(<stem>_greymask.jpg)로 저장할지.
# 색 band/임계값을 실데이터로 보정할 때 눈으로 확인하는 용도. 0 으로 끄면 생략.
SAVE_GREY_MASK_DEBUG = env_int("RCS_OUTLINE_SAVE_GREY_MASK", 1) == 1

# SEM box 위치 reference 저장 — production 모니터가 '박스가 옮겨졌나/닫혔나' 를
# 이 reference(정상 위치)와 현재 검출 박스를 비교해 판정한다. 0 으로 끄면 생략.
# align_images 는 읽기 전용 입력이라 거기 쓰지 않고, eqp_id 별로 여기 별도 보관한다.
WRITE_SEM_BOX_REFERENCE = env_int("RCS_OUTLINE_WRITE_REFERENCE", 1) == 1
SEM_BOX_REFERENCE_DIR = WORKFLOW_2_DIR / "sem_box_references"

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
# greymask 디버그 렌더 — 검출/snap 자체는 sem_box_detect 가 담당.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# SEM box 위치 reference — production 모니터링(이동/닫힘 감지)의 기준값 저장/로드.
# ------------------------------------------------------------------


def _eqp_id_for_capture(image_path: Path) -> str | None:
    """캡처 경로에서 eqp_id 를 추출한다.

    레이아웃: ``<eqp_id>/<class>/<recipe>/captured_img_from_rcs/<tag>/<file>`` →
    captured_img_from_rcs 기준 3단계 위가 eqp_id. 경로로 못 구하면 ALIGN_EQP_ID
    환경변수, 그것도 없으면 None.
    """
    parts = image_path.parts
    if CAPTURED_RCS_DIRNAME in parts:
        idx = parts.index(CAPTURED_RCS_DIRNAME)
        if idx >= 3:
            return parts[idx - 3]
    return os.getenv("ALIGN_EQP_ID", "").strip() or None


def _build_reference(eqp_id: str, items: list["OutlineReport"], tag: str) -> dict:
    """한 eqp 의 detected 박스들에서 robust(중앙값) 위치 reference 를 만든다.

    위치는 sharpness 와 무관(프레임 위치는 blur 여부와 별개)하므로 detected+cv_bbox
    면 모두 표본으로 쓴다. 좌표별 중앙값으로 단일 프레임 오검출에 강건하게 만들고,
    표본 간 범위(spread)를 함께 적어 reference 신뢰도를 사람이 가늠하게 한다.
    """
    keys = ("left", "top", "right", "bottom")
    norms = [bbox_px_to_1000(r.cv_bbox, r.width, r.height) for r in items]
    median = {k: int(round(float(np.median([nb[k] for nb in norms])))) for k in keys}
    spread = {k: int(max(nb[k] for nb in norms) - min(nb[k] for nb in norms)) for k in keys}
    modes = [r.mode_label for r in items if r.mode_label]
    sharps = [r.sharpness for r in items if r.sharpness is not None]
    return {
        "eqp_id": eqp_id,
        "created_tag": tag,
        "coord_system": "relative_1000",
        "coord_note": "캡처된 tool 창 크기 기준 0-1000 정규화 (해상도/창 크기 달라도 비교 가능).",
        "bbox_1000": median,
        "spread_1000": spread,
        "sample_count": len(items),
        "ref_window_size": {
            "width": int(np.median([r.width for r in items])),
            "height": int(np.median([r.height for r in items])),
        },
        "mode_label": Counter(modes).most_common(1)[0][0] if modes else None,
        "sharpness_median": round(float(np.median(sharps)), 1) if sharps else None,
        "source_images": [Path(r.image_path).name for r in items],
        "note": (
            "outline_live_sem_box 가 생성한 SEM box 정상 위치. production 모니터가 "
            "현재 검출 박스와 비교해 이동/닫힘을 판정한다."
        ),
    }


def _write_sem_box_references(reports: list["OutlineReport"], out_dir: Path, tag: str) -> list[str]:
    """detected 박스들을 eqp_id 별로 묶어 위치 reference JSON 을 저장한다."""
    groups: dict[str, list["OutlineReport"]] = {}
    for report in reports:
        if not report.vlm_detected or not report.cv_bbox:
            continue
        eqp = _eqp_id_for_capture(Path(report.image_path)) or "unknown"
        groups.setdefault(eqp, []).append(report)

    if not groups:
        print("[WARNING] detected 박스가 없어 SEM box reference 를 저장하지 않습니다.")
        return []

    written: list[str] = []
    for eqp, items in groups.items():
        reference = _build_reference(eqp, items, tag)
        if eqp == "unknown":
            target = out_dir / "sem_box_reference_unknown.json"
            print(
                f"[WARNING] eqp_id 를 경로/ALIGN_EQP_ID 에서 못 구함 → {target} 에만 저장"
            )
        else:
            SEM_BOX_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            target = SEM_BOX_REFERENCE_DIR / f"{eqp}.json"
        target.write_text(
            json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(str(target))
        print(
            f"[INFO] SEM box reference 저장: eqp={eqp} samples={len(items)} "
            f"bbox_1000={reference['bbox_1000']} spread_1000={reference['spread_1000']} "
            f"-> {target}"
        )
    return written


def load_sem_box_reference(eqp_id: str) -> dict | None:
    """저장된 SEM box 위치 reference 를 로드한다(없으면 None).

    production 모니터의 read 진입점. 현재 검출한 박스를 reference['bbox_1000'] 과
    같은 0-1000 정규화로 바꿔 비교하면 이동/닫힘을 판정할 수 있다.
    """
    path = SEM_BOX_REFERENCE_DIR / f"{eqp_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    with Image.open(image_path) as pil_image:
        rgb = pil_image.convert("RGB")

    # 검출은 공유 detector(sem_box_detect)가 담당 — online RCS 방문 경로와 동일 로직.
    detection = detect_sem_box(rgb, client)

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if SAVE_GREY_MASK_DEBUG:
        cv2.imwrite(
            str(out_dir / f"{image_path.stem}_greymask.jpg"),
            _render_grey_mask_debug(bgr, grey_frame_mask(bgr)),
        )

    overlay = _draw_overlay(
        bgr, detection.vlm_bbox_px, detection.bbox_px, detection.sharpness, detection.blurry
    )
    overlay_path = out_dir / f"{image_path.stem}_outline.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    return OutlineReport(
        image_path=str(image_path),
        width=detection.width,
        height=detection.height,
        vlm_detected=detection.detected,
        vlm_bbox=detection.vlm_bbox_px,
        cv_bbox=detection.bbox_px,
        mode_label=detection.mode_label,
        vlm_confidence=detection.confidence,
        sharpness=detection.sharpness,
        blurry=detection.blurry,
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

    reference_files: list[str] = []
    if WRITE_SEM_BOX_REFERENCE:
        reference_files = _write_sem_box_references(reports, out_dir, tag)

    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(reports),
        "vlm_detected": sum(1 for r in reports if r.vlm_detected),
        "blurry": sum(1 for r in reports if r.blurry),
        "sharpness_threshold": SHARPNESS_BLUR_THRESHOLD,
        "reference_files": reference_files,
        "reports": [asdict(r) for r in reports],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[INFO] 완료: processed={len(reports)}/{len(paths)} "
        f"vlm_detected={summary['vlm_detected']} blurry={summary['blurry']} "
        f"references={len(reference_files)} elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if reports else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
