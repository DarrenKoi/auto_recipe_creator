"""golden 트리에서 cond.txt 읽기 + white box / crosshair 제거를 일괄 검증한다.

오피스에서 다운로드한 ``align_images_golden`` 폴더를 통째로 훑어, 이미지마다
짝 cond.txt (.<파일명>/cond.txt) 를 파싱해 무엇이 읽혔는지 보고하고, box/crosshair
가 있으면 inpaint 로 지운 **before | mask | after** 비교 이미지를 저장한다.

CLI 인자 없음. 루트는 아래 GOLDEN_ROOT (환경변수 ALIGN_GOLDEN_ROOT 로 덮어쓰기).
    uv run python poc/workflow_2/verify_cond_cleaning.py

확인 포인트 (오피스):
  - cond.txt 가 dot-folder 째로 잘 전송됐나 (안 됐으면 "cond 없음" 다발).
  - box/crosshair 가 실제 주석 위에 정확히 놓이나 (좌표계/oversample 검증).
  - 1024-px 이미지에서도 /10 비율이 맞나 ([[project_align_cond_files_and_coords]]).
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_1 import WORKFLOW_1_DIR
from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import SUPPORTED_EXTS
from poc.workflow_2.clean_align_image import build_removal_mask, clean_image
from poc.workflow_2.cond_file import cond_path_for, load_cond

# 검증 대상 루트 (오피스 다운로드 폴더). 환경변수로 덮어쓸 수 있음.
GOLDEN_ROOT = Path(
    os.getenv("ALIGN_GOLDEN_ROOT", str(WORKFLOW_1_DIR / "align_images_golden"))
)


def _iter_images(root: Path):
    """루트 아래 모든 이미지(숨김 dot-folder 안은 제외)를 이름순으로 돌려준다."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue                       # .<name>/ 안의 파일은 건너뜀
        yield path


def _comparison(image, cond):
    """before | mask(빨강) | after 가로 결합 이미지를 만든다(시각 검증용)."""
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = build_removal_mask(
        bgr.shape[:2], box_ltrb=cond.box_ltrb, crosshair_xy=cond.crosshair_xy
    )
    overlay = bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    after = clean_image(bgr, cond)
    return np.hstack([bgr, overlay, after])


def main():
    out_dir = DEBUG_IMAGE_DIR / "cond_clean" / time.strftime("%y%m%d_%H%M%S")
    print(f"[INFO] golden root: {GOLDEN_ROOT}")
    if not GOLDEN_ROOT.is_dir():
        print(f"[ERROR] 루트 폴더가 없습니다. 다운로드 위치를 확인하거나 "
              f"ALIGN_GOLDEN_ROOT 로 지정하세요: {GOLDEN_ROOT}")
        raise SystemExit(1)

    n_img = n_cond = n_box = n_cross = n_written = n_nocond = 0
    for image_path in _iter_images(GOLDEN_ROOT):
        n_img += 1
        rel = image_path.relative_to(GOLDEN_ROOT)
        cond = load_cond(image_path)
        if cond is None:
            n_nocond += 1
            print(f"[WARNING] cond 없음: {rel}  (기대 위치: "
                  f"{cond_path_for(image_path).relative_to(GOLDEN_ROOT)})")
            continue
        n_cond += 1
        has_box = cond.box_ltrb is not None
        has_cross = cond.crosshair_xy is not None
        n_box += int(has_box)
        n_cross += int(has_cross)
        print(f"[INFO] {rel} | scope={cond.scope} pixel={cond.pixel} "
              f"box={cond.box_ltrb} crosshair={cond.crosshair_xy}")

        if not (has_box or has_cross):
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARNING] 이미지 디코드 실패: {rel}")
            continue
        # Pixel 과 실제 디코드 크기가 다르면 좌표계 경고.
        h, w = image.shape[:2]
        if cond.pixel and (w, h) != cond.pixel:
            print(f"[WARNING]   Pixel{cond.pixel} != 실제 이미지({w},{h}) — "
                  f"oversample 비율 재확인 필요")
        dst = out_dir / rel.parent / f"{image_path.stem}_cmp.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), _comparison(image, cond), [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_written += 1

    print(f"\n[INFO] 이미지 {n_img} · cond 있음 {n_cond} (없음 {n_nocond}) · "
          f"box {n_box} · crosshair {n_cross} · 비교본 저장 {n_written}")
    if n_written:
        print(f"[INFO] 비교본(before|mask|after): {out_dir}")
    if n_img and n_nocond == n_img:
        print("[ERROR] 모든 이미지에 cond.txt 가 없습니다 — dot-folder 전송 누락 의심 "
              "(zip/sync 가 '.' 폴더를 건너뛰었을 수 있음).")


if __name__ == "__main__":
    main()
