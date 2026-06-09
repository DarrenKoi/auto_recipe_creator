"""합성 reference(IMAP) 데모 — "white box 와 reference 이미지"를 내가 어떻게 이해하는지
사람이 눈으로 확인하기 위한 throwaway 스크립트.

내가 이해하는 align-fail reference (align_img_from_rcp/IMAP000x) 의 구조
-----------------------------------------------------------------------
1. 바탕 = **회색 SEM/OM FOV**. 순수 흑백이 아니라 feature-sparse 한 wafer 표면 +
   charging gradient + 자잘한 texture(분석을 방해하는 distractor blob 포함).
2. 그 안에 **실제로 매칭해야 할 fiducial 구조물**(box-in-box + 비대칭 코너/dot).
   인위적으로 만든 정렬 키지만, 바탕보다 *어두운* stroke 으로 찍힌다(밝지 않다).
3. 그 구조물 둘레에 엔지니어가 그려 넣은 **밝은(≈255) 흰색 box 주석**. 이건 wafer 의
   일부가 아니라 "여기가 align key 다"라고 표시한 *오버레이 마커*다. box 중심 ≈ 타깃
   align point. ([[align-fail-correction-model]])

왜 흰 box 를 crop 으로 잘라내야 하나
-------------------------------------
- box stroke(흰 선)는 wafer 에 없는 주석이라, live scene(현재 SEM)에는 대응물이 없다.
  scene 의 가장 밝은 흰 구조는 crosshair(=틀린 위치)뿐 → "제일 밝은 걸 매칭" 식 휴리스틱은
  reference box → scene crosshair 로 오인(false anchor)할 수 있다.
- 따라서 box 는 "어디를 자를지" 알려주는 가이드로만 쓰고, **테두리 안쪽(inset)** 만 잘라
  실제 fiducial 구조물만 template/VLM 입력으로 쓴다. 흰 선이 단 1px도 남으면 안 된다.

검출/크롭은 새로 짜지 않고 `align_point_correction.py` 의 기존 구현을 재사용한다:
  `_detect_white_box`  → 흰 box bbox (top-hat → Otsu → hollow-outline 필터)
  `_inner_crop_for_box`→ box stroke 를 RCP_BOX_INSET_PX 만큼 피한 내부 crop

실행
----
    uv run python poc/workflow_2/synth_white_box_demo.py

Mac/오프라인에서 그대로 돈다(실제 align_images 자산 불필요). 결과는
debug_images/synth_white_box_demo/<timestamp>/ 에 JPEG 로 저장된다.
"""

import os

# OpenBLAS/OMP 스레드 제한 — numpy/cv2 import 이전(Windows 다중코어 메모리 회피).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_point_correction import _detect_white_box, _inner_crop_for_box
from poc.workflow_2.test_align_key_match import (
    add_charging_gradient,
    add_random_blobs,
    make_synthetic_template,
    make_wafer_background,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag

# ====================================================================
# 설정 (CLI 인자 없음 — 상수로만).
# ====================================================================

FRAME_SIZE = (512, 768)        # (h, w) — SEM FOV 비율을 흉내.
STRUCT_SIZE = 128              # fiducial 구조물 한 변(px).
BOX_HALF = 88                  # 흰 box 중심에서 변까지 거리(px) — 구조물보다 살짝 크게.
BOX_STROKE = 2                 # 흰 box 선 두께(px) — 엔지니어가 그린 1~3px 가정.
WHITE = 255                    # 흰 box 밝기 — scene 에서 가장 밝은 주석.

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "synth_white_box_demo"

# overlay 색(BGR).
_BGR_YELLOW = (40, 220, 220)   # 검출된 흰 box.
_BGR_GREEN = (60, 200, 60)     # inner crop(테두리 제외 후 실제 template).
_BGR_BLUE = (220, 140, 40)     # box 중심 = 타깃 align point.


def _make_synthetic_reference() -> tuple[np.ndarray, tuple[int, int]]:
    """회색 FOV + 어두운 fiducial + 밝은 흰 box 주석을 합성해 reference(IMAP) 한 장을 만든다.

    반환: (grayscale_reference, box_center_xy).
    """
    h, w = FRAME_SIZE
    # 1) 회색 wafer 바탕 + charging gradient + distractor blob.
    bg = make_wafer_background(FRAME_SIZE, base=128)
    bg = add_charging_gradient(bg, max_delta=35)
    bg = add_random_blobs(bg, count=14)

    # 2) 어두운 fiducial 구조물(box-in-box)을 화면 중앙에 박아 넣는다(밝지 않다).
    struct = make_synthetic_template(size=STRUCT_SIZE, key_type="box")
    cx, cy = w // 2, h // 2
    x0, y0 = cx - STRUCT_SIZE // 2, cy - STRUCT_SIZE // 2
    # 바탕과 자연스럽게 섞이도록 어두운 부분만 덮어쓴다(밝은 바탕은 유지).
    region = bg[y0:y0 + STRUCT_SIZE, x0:x0 + STRUCT_SIZE]
    bg[y0:y0 + STRUCT_SIZE, x0:x0 + STRUCT_SIZE] = np.minimum(region, struct)

    # 3) 엔지니어 흰 box 주석을 구조물 둘레에 그린다(밝은 hollow rectangle).
    cv2.rectangle(bg, (cx - BOX_HALF, cy - BOX_HALF), (cx + BOX_HALF, cy + BOX_HALF),
                  WHITE, BOX_STROKE)
    return bg, (cx, cy)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[INFO] 저장: {path}")


def _draw_overlay(gray: np.ndarray, box, inner, box_center) -> np.ndarray:
    """원본 위에 검출 box(노랑)·inner crop(초록)·align point(파랑)을 그린다."""
    canvas = _to_bgr(gray)
    if box is not None:
        bx, by, bw, bh = box
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), _BGR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(canvas, "detected white box", (bx, max(14, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_YELLOW, 1, cv2.LINE_AA)
    if inner is not None:
        ix, iy, iw, ih = inner
        cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), _BGR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "inner crop (template)", (ix, iy + ih + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_GREEN, 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, box_center, _BGR_BLUE, cv2.MARKER_CROSS, 20, 2)
    cv2.putText(canvas, "align point", (box_center[0] + 12, box_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BGR_BLUE, 1, cv2.LINE_AA)
    return canvas


def _max_white_in_crop(crop: np.ndarray) -> int:
    """crop 내 최대 밝기 — 흰 box(≈255)가 남았는지 정량 확인용."""
    return int(crop.max()) if crop.size else 0


def run() -> str:
    tag = make_timestamp_tag()
    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 내가 이해한 reference 합성.
    ref, box_center = _make_synthetic_reference()
    _save_jpeg(_to_bgr(ref), out_dir / "reference_raw.jpg")
    print(f"[INFO] 합성 reference {ref.shape[1]}x{ref.shape[0]}  box_center={box_center}  "
          f"max_white={int(ref.max())}")

    # 2) 기존 구현으로 흰 box 검출.
    box = _detect_white_box(ref)
    print(f"[INFO] _detect_white_box → {box}")

    # 3) 흰 테두리를 inset 만큼 피해 내부만 crop.
    inner = None
    if box is not None:
        crop, inner = _inner_crop_for_box(ref, box)
        _save_jpeg(_to_bgr(crop), out_dir / "interior_crop.jpg")
        white_in_crop = _max_white_in_crop(crop)
        verdict = "OK(흰 선 없음)" if white_in_crop < 230 else "WARN(흰 선 잔존!)"
        print(f"[INFO] inner crop bbox={inner}  crop_size={crop.shape[1]}x{crop.shape[0]}  "
              f"max_white_in_crop={white_in_crop} → {verdict}")
    else:
        print("[WARNING] 흰 box 미검출 — fallback(centered crop) 경로가 필요한 케이스.")

    # 4) overlay 저장.
    overlay = _draw_overlay(ref, box, inner, box_center)
    _save_jpeg(overlay, out_dir / "detect_overlay.jpg")

    print(f"[INFO] 완료. 결과 폴더: {out_dir}")
    return "success"


if __name__ == "__main__":
    run()
