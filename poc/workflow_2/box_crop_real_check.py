"""실제와 흡사한 SEM 이미지(흰 box 삽입본)로 _detect_white_box + _inner_crop_for_box 를
검증하는 throwaway 체커.

합성(synth_white_box_demo.py)은 "박스가 가장 밝은 얇은 구조" 라는 이상적 가정에서만
돌았다. 실제 SEM 은 소자 패턴(밝은 가는 선)이 사방에 있어 박스보다 더 밝고 더 많을 수
있다 — 이 체커는 그 조건에서 검출/크롭이 버티는지(또는 어디서 깨지는지)를 사람이 overlay
로 보게 한다.

입력: poc/workflow_2/templates/whitebox_samples/*.png  (인터넷 SEM + 인위 삽입 흰 box)
출력: templates/box_crop_results/<timestamp>/  <stem>_overlay·_mask·_crop.jpg + report.json
      (crosshair_removal_results 와 동일한 flat 레이아웃)

실행
----
    uv run python poc/workflow_2/box_crop_real_check.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_3.vision.align_point_correction import (
    _detect_white_box,
    _inner_crop_for_box,
    _stroke_threshold,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag

INPUT_DIR = WORKFLOW_2_DIR / "templates" / "whitebox_samples"
OUTPUT_ROOT = WORKFLOW_2_DIR / "templates" / "box_crop_results"

_BGR_YELLOW = (40, 220, 220)   # 검출된 흰 box.
_BGR_GREEN = (60, 200, 60)     # inner crop.


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _build_box_mask(shape, box, inner) -> np.ndarray:
    """검출된 흰 box 의 *stroke ring*(바깥 box − 안쪽 crop)을 255 로 채운 uint8 mask.

    crosshair 의 full-span band mask 와 같은 역할(제거 대상 픽셀 표시)이지만, box 는
    inpaint 가 아니라 crop 으로 제거하므로 mask 는 검증·시각화 용도다.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bx, by, bw, bh = box
    mask[by:by + bh, bx:bx + bw] = 255       # 바깥 box 채움.
    ix, iy, iw, ih = inner
    mask[iy:iy + ih, ix:ix + iw] = 0         # 안쪽 crop 영역을 빼 stroke ring 만 남김.
    return mask


def _save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _overlay(gray: np.ndarray, box, inner) -> np.ndarray:
    canvas = _to_bgr(gray)
    if box is not None:
        bx, by, bw, bh = box
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), _BGR_YELLOW, 2, cv2.LINE_AA)
        cv2.putText(canvas, "detected box", (bx, max(16, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BGR_YELLOW, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "NO BOX DETECTED", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 230), 2, cv2.LINE_AA)
    if inner is not None:
        ix, iy, iw, ih = inner
        cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), _BGR_GREEN, 2, cv2.LINE_AA)
    return canvas


def _process(path: Path, out_dir: Path) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[ERROR] 디코드 실패: {path.name}")
        return {"file": path.name, "error": "decode_failed"}
    h, w = gray.shape[:2]

    box = _detect_white_box(gray)
    rec = {"file": path.name, "size": [w, h], "detected_box": list(box) if box else None}

    inner = None
    if box is not None:
        crop, inner = _inner_crop_for_box(gray, box)
        region = gray[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        thr = _stroke_threshold(region)
        crop_max = int(crop.max()) if crop.size else 0
        white_clean = crop_max < thr
        rec.update({
            "inner_crop": list(inner),
            "stroke_thr": thr,
            "crop_max": crop_max,
            "white_line_removed": white_clean,
        })
        mask = _build_box_mask(gray.shape, box, inner)
        _save_jpeg(mask, out_dir / f"{path.stem}_mask.jpg")
        _save_jpeg(_to_bgr(crop), out_dir / f"{path.stem}_crop.jpg")
        verdict = "OK(흰 선 제거)" if white_clean else "WARN(흰 선 잔존)"
        print(f"[INFO] {path.name} {w}x{h}  box={box}  inner={inner}  "
              f"thr={thr} crop_max={crop_max} → {verdict}")
    else:
        print(f"[WARNING] {path.name} {w}x{h}  흰 box 미검출 (소자 패턴에 묻혔을 가능성).")

    _save_jpeg(_overlay(gray, box, inner), out_dir / f"{path.stem}_overlay.jpg")
    return rec


def run() -> str:
    if not INPUT_DIR.is_dir():
        print(f"[ERROR] 입력 폴더 없음: {INPUT_DIR}")
        return "no_input"
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if not images:
        print(f"[ERROR] 이미지 없음: {INPUT_DIR}")
        return "no_images"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(images)}장 처리 → {out_dir}")

    records = [_process(p, out_dir) for p in images]
    detected = sum(1 for r in records if r.get("detected_box"))
    clean = sum(1 for r in records if r.get("white_line_removed"))
    print(f"[INFO] 검출 {detected}/{len(images)}  |  흰선제거 OK {clean}/{detected if detected else 0}")
    (out_dir / "report.json").write_text(
        json.dumps({"summary": {"total": len(images), "detected": detected, "clean": clean},
                    "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    run()
