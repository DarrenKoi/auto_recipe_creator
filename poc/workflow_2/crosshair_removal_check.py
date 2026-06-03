"""crosshair 검출 → 제거(inpaint) 검증 (throwaway).

scene(현재 SEM)의 흰 십자선은 *틀린* 정렬 위치를 가리키는 도구 오버레이다. 동시에 scene
에서 가장 밝은 full-span 구조라, 매칭/VLM 이 align-key 대신 십자선에 끌릴 수 있는 false
anchor 다([[align-fail-correction-model]]). 그래서 매칭 전에 십자선을 지워 underlying
구조를 드러낸다.

박스(`box_crop_real_check.py`)와 달리 십자선은 *프레임 전체*를 가로지르는 가는 축정렬
선이라 검출이 명확하다 — 기존 `crosshair_detect.detect_crosshair`(절대 saturation +
방향성 morphology)를 그대로 재사용한다.

프로세스(박스와 동일한 흐름):
  1) detect_crosshair 로 십자 중심 (cx,cy) + 선 두께(band) 획득.
  2) full-span 가로/세로 band 를 mask 로 만든다(두께 = band + margin).
  3) cv2.inpaint 로 선을 지운다.
  4) 6개 샘플에 대해 overlay·mask·inpaint 결과를 저장해 사람이 검증.

실행: uv run python poc/workflow_2/crosshair_removal_check.py
출력: debug_images/crosshair_removal_check/<timestamp>/<stem>_*.jpg
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_2.crosshair_detect import detect_crosshair
from poc.workflow_1.util.time_utils import make_timestamp_tag

INPUT_DIR = WORKFLOW_2_DIR / "templates" / "crosshair_samples"
OUTPUT_ROOT = WORKFLOW_2_DIR / "templates" / "crosshair_removal_results"

# 선 두께(band) 위에 추가로 덮을 여유(px) — anti-alias 잔광 + 약간의 번짐 대비.
MASK_MARGIN_PX = 4
# band 추정이 없거나 비정상일 때 기본 반두께(px).
DEFAULT_HALF_PX = 3
INPAINT_RADIUS = 4
# 강화 검증 — 제거 band 의 Canny edge 밀도가 인접 평행 band 대비 이 배 이하여야 깔끔(seam/ghost 탐지).
# 잔존 밝은 픽셀 count 만으로는 어두운 이음새/고스트를 놓치므로, edge 밀도 비를 보조 지표로 쓴다.
EDGE_BAND_RATIO_MAX = 1.8


def _line_half_thickness(line_info: dict) -> int:
    """detect_crosshair debug 의 band(고-coverage 줄 수≈선 두께)에서 반두께를 구한다."""
    band = int(line_info.get("band", 0)) if isinstance(line_info, dict) else 0
    if band <= 0:
        return DEFAULT_HALF_PX
    return max(DEFAULT_HALF_PX, band // 2 + MASK_MARGIN_PX)


def _build_cross_mask(shape, cx: int, cy: int, half_h: int, half_v: int) -> np.ndarray:
    """full-span 가로(y=cy)·세로(x=cx) band 를 1 로 채운 uint8 mask."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[max(0, cy - half_h):min(h, cy + half_h + 1), :] = 255   # 가로선 전폭.
    mask[:, max(0, cx - half_v):min(w, cx + half_v + 1)] = 255   # 세로선 전높이.
    return mask


def _residual_along_lines(gray: np.ndarray, cx: int, cy: int, half: int, thr: int) -> int:
    """inpaint 후 십자선 자리에 thr 이상 밝은 픽셀이 남았는지(>0 이면 잔존)."""
    h, w = gray.shape[:2]
    band = np.concatenate([
        gray[max(0, cy - half):min(h, cy + half + 1), :].ravel(),
        gray[:, max(0, cx - half):min(w, cx + half + 1)].ravel(),
    ])
    return int((band >= thr).sum())


def _edge_band_ratio(img: np.ndarray, cx: int, cy: int, half_h: int, half_v: int) -> float:
    """제거 band 의 Canny edge 밀도 ÷ 인접 평행 band 의 edge 밀도 (가로·세로 중 큰 값).

    inpaint 가 깔끔하면 선 자리의 edge 밀도가 주변 텍스처와 비슷(비≈1). 이음새(seam)나
    고스트가 남으면 그 자리만 edge 가 튀어 비가 커진다 — 밝은-잔존-픽셀 count 가 0 이어도
    잡지 못하는 *어두운* 이음새를 보완하는 지표다.
    """
    edges = (cv2.Canny(img, 50, 150) > 0).astype(np.float32)
    h, w = edges.shape[:2]

    def density(y0: int, y1: int, x0: int, x1: int) -> float:
        sub = edges[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        return float(sub.mean()) if sub.size else 0.0

    eps = 1e-6
    # 가로선: 제거 band vs 위/아래 평행 band.
    off_h = max(6, 3 * half_h)
    rem_h = density(cy - half_h, cy + half_h + 1, 0, w)
    nb_h = 0.5 * (density(cy - half_h - off_h, cy + half_h + 1 - off_h, 0, w)
                  + density(cy - half_h + off_h, cy + half_h + 1 + off_h, 0, w))
    # 세로선: 제거 band vs 좌/우 평행 band.
    off_v = max(6, 3 * half_v)
    rem_v = density(0, h, cx - half_v, cx + half_v + 1)
    nb_v = 0.5 * (density(0, h, cx - half_v - off_v, cx + half_v + 1 - off_v)
                  + density(0, h, cx - half_v + off_v, cx + half_v + 1 + off_v))
    return max(rem_h / (nb_h + eps), rem_v / (nb_v + eps))


def _process(path: Path, out_dir: Path) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[ERROR] 디코드 실패: {path.name}")
        return {"file": path.name, "error": "decode_failed"}
    h, w = gray.shape[:2]

    res = detect_crosshair(gray)
    rec = {"file": path.name, "size": [w, h], "xy": list(res.xy) if res.xy else None,
           "confidence": round(res.confidence, 3), "reason": res.debug.get("reason")}

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if res.xy is None:
        cv2.putText(overlay, f"NO CROSSHAIR ({res.debug.get('reason')})", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 230), 2)
        cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), overlay,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[WARNING] {path.name} {w}x{h}  십자선 미검출 (reason={res.debug.get('reason')})")
        return rec

    cx, cy = res.xy
    half_h = _line_half_thickness(res.debug.get("h_line", {}))   # 가로선 두께.
    half_v = _line_half_thickness(res.debug.get("v_line", {}))   # 세로선 두께.
    mask = _build_cross_mask(gray.shape, cx, cy, half_h, half_v)
    inpainted = cv2.inpaint(gray, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # 검증 지표 1: inpaint 후 십자 자리의 밝은 잔존 픽셀(낮을수록 깔끔).
    sat = res.debug.get("sat_thresh", 215)
    half = max(half_h, half_v)
    resid_before = _residual_along_lines(gray, cx, cy, half, sat)
    resid_after = _residual_along_lines(inpainted, cx, cy, half, sat)
    clean_resid = resid_after <= max(5, int(0.05 * resid_before))

    # 검증 지표 2 (Codex 강화): inpaint 후 *재검출* — 선이 남아 있으면 다시 검출된다.
    re_res = detect_crosshair(inpainted)
    redetected = re_res.xy is not None

    # 검증 지표 3 (Codex 강화): 제거 band edge 밀도 / 인접 band — seam·ghost 탐지.
    edge_ratio = _edge_band_ratio(inpainted, cx, cy, half_h, half_v)

    removed_ok = clean_resid and (not redetected) and (edge_ratio <= EDGE_BAND_RATIO_MAX)
    rec.update({"half_h": half_h, "half_v": half_v,
                "resid_before": resid_before, "resid_after": resid_after,
                "clean_resid": clean_resid, "redetected": redetected,
                "edge_band_ratio": round(edge_ratio, 2), "removed_ok": removed_ok})

    # overlay: 검출된 full-span 선(초록).
    cv2.line(overlay, (0, cy), (w - 1, cy), (60, 220, 60), 1, cv2.LINE_AA)
    cv2.line(overlay, (cx, 0), (cx, h - 1), (60, 220, 60), 1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 5, (60, 220, 60), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(out_dir / f"{path.stem}_mask.jpg"), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(out_dir / f"{path.stem}_inpainted.jpg"), inpainted, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    verdict = "OK(제거됨)" if rec["removed_ok"] else "WARN(잔존)"
    print(f"[INFO] {path.name} {w}x{h}  xy=({cx},{cy}) conf={res.confidence:.2f}  "
          f"half=(h{half_h},v{half_v})  resid {resid_before}→{resid_after}  "
          f"redetect={'Y' if redetected else 'N'}  edge_ratio={edge_ratio:.2f} → {verdict}")
    return rec


def run() -> str:
    if not INPUT_DIR.is_dir():
        print(f"[ERROR] 입력 폴더 없음: {INPUT_DIR}")
        return "no_input"
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".png")
    if not images:
        print(f"[ERROR] 이미지 없음: {INPUT_DIR}")
        return "no_images"
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(images)}장 처리 → {out_dir}")

    records = [_process(p, out_dir) for p in images]
    detected = sum(1 for r in records if r.get("xy"))
    removed = sum(1 for r in records if r.get("removed_ok"))
    print(f"[INFO] 검출 {detected}/{len(images)}  |  제거 OK {removed}/{detected if detected else 0}")
    (out_dir / "report.json").write_text(
        json.dumps({"summary": {"total": len(images), "detected": detected, "removed": removed},
                    "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    run()
