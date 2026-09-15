"""라이브 SEM box 의 이동 모드(crosshair / L자 아이콘) 판별.

tool 창의 라이브 SEM box 옆에는 crosshair 아이콘과 L자 아이콘이 있고, **선택된 쪽이
초록으로 채워진다**(사용자 보고 2026-09-15). 모드에 따라 박스 안 이동 클릭 수가 다르다:
crosshair = 더블클릭, L자 = 싱글클릭. `sem_monitor.controller.RECENTER_CLICKS` 가 이
값을 쓴다(지금은 env `ALIGN_SEM_RECENTER_CLICKS` 로 오피스에서 맞춘다).

역할 분담은 저장소 규칙 그대로다 - VLM 은 두 아이콘의 **위치**만 찾고(요소당 호출
하나), 어느 쪽이 켜졌는지는 **픽셀 색**이 답한다. 아이콘엔 라벨이 없어 OCR 확인이
불가능하므로, 초록 채움 자체가 확인 게이트다: 두 bbox 중 초록 비율이 뚜렷이 높은 쪽만
활성으로 보고, 둘 다 초록이거나 둘 다 아니면 unknown 이다(추측하지 않는다).

단독 점검 (오피스, tool 창이 열려 있을 때; 클릭 없음):
  uv run python -m poc.workflow_3.sem_monitor.click_mode
  CLICK_MODE_IMAGE=/path/tool.jpg uv run python -m poc.workflow_3.sem_monitor.click_mode
"""

import os
import time

import numpy as np
from PIL import Image

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target

MODE_CROSSHAIR = "crosshair"   # 더블클릭 이동
MODE_L_SHAPE = "l_shape"       # 싱글클릭 이동
MODE_UNKNOWN = "unknown"

CLICKS_FOR_MODE = {MODE_CROSSHAIR: 2, MODE_L_SHAPE: 1}

# 초록 판정(HSV, OpenCV 규약 H 0-180). 아이콘 채움색을 모르는 상태의 넓은 띠 - 첫 오피스
# 실행의 green_ratio 값을 보고 좁힌다.
GREEN_H_MIN, GREEN_H_MAX = 35, 90
GREEN_S_MIN, GREEN_V_MIN = 80, 80
# 활성 판정: 초록 비율이 이 이상이고, 다른 아이콘보다 이 배수 이상 높아야 한다.
ACTIVE_MIN_RATIO = 0.10
ACTIVE_MARGIN = 2.0

ICON_TARGETS = {
    MODE_CROSSHAIR: TargetConfig(
        key="sem_crosshair_icon",
        description="the small crosshair (plus-shaped, '+') tool icon button next to the live "
                    "SEM image area, in the icon toolbar beside the live image. Not the "
                    "crosshair drawn inside the image itself.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=1.0,
        min_crop_width=120, min_crop_height=120,
    ),
    MODE_L_SHAPE: TargetConfig(
        key="sem_l_shape_icon",
        description="the small L-shaped (corner bracket) tool icon button next to the live "
                    "SEM image area, in the icon toolbar beside the live image, near the "
                    "crosshair icon.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=1.0,
        min_crop_width=120, min_crop_height=120,
    ),
}


def green_ratio(crop) -> float:
    """crop(PIL 또는 RGB ndarray)에서 '초록으로 채워진' 픽셀 비율(0~1)."""
    import cv2

    rgb = np.asarray(crop.convert("RGB") if isinstance(crop, Image.Image) else crop)
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= GREEN_H_MIN) & (hsv[..., 0] <= GREEN_H_MAX)
            & (hsv[..., 1] >= GREEN_S_MIN) & (hsv[..., 2] >= GREEN_V_MIN))
    return float(mask.mean())


def classify_mode(ratios: dict) -> str:
    """두 아이콘의 초록 비율로 모드를 정한다. 뚜렷하지 않으면 unknown."""
    cross = ratios.get(MODE_CROSSHAIR)
    lshape = ratios.get(MODE_L_SHAPE)
    if cross is None or lshape is None:
        return MODE_UNKNOWN
    if cross >= ACTIVE_MIN_RATIO and cross >= lshape * ACTIVE_MARGIN:
        return MODE_CROSSHAIR
    if lshape >= ACTIVE_MIN_RATIO and lshape >= cross * ACTIVE_MARGIN:
        return MODE_L_SHAPE
    return MODE_UNKNOWN


def _icon_box(result, image_size) -> dict | None:
    """로케이터 결과에서 아이콘 bbox. coarse bbox 가 없으면 fine point 주위 고정 상자."""
    if result.exit_code != "success" or result.point is None:
        return None
    if isinstance(result.bbox, dict):
        return result.bbox
    x, y = result.point["x"], result.point["y"]
    return {"left": max(0, x - 12), "top": max(0, y - 12),
            "right": min(image_size[0], x + 12), "bottom": min(image_size[1], y + 12)}


def detect_click_mode(image, *, artifact_dir=None) -> dict:
    """tool 창 이미지에서 이동 모드를 판별한다. 반환 dict 의 `mode` 는 세 값 중 하나."""
    artifact_dir = artifact_dir or (DEBUG_IMAGE_DIR / "click_mode" / str(time.time_ns()))
    report = {"mode": MODE_UNKNOWN, "icons": {}, "artifact_dir": str(artifact_dir)}
    ratios = {}
    for mode, target in ICON_TARGETS.items():
        result = None
        try:
            result = analyze_window_target(
                None, "tool window", "image", target, image=image,
                debug_image_dir=artifact_dir / "locator", log_name="click_mode",
                component_name="click_mode", artifact_prefix=mode,
            )
            box = _icon_box(result, image.size)
        except Exception as exc:
            print(f"[WARNING] {mode} 아이콘 로케이트 예외: {exc}")
            box = None
        info = {"box": box, "green_ratio": None, "exit_code": getattr(result, "exit_code", "error")}
        if box is not None:
            crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
            save_debug_jpeg(crop, artifact_dir / f"{mode}.jpg")
            ratios[mode] = info["green_ratio"] = green_ratio(crop)
        report["icons"][mode] = info
    report["mode"] = classify_mode(ratios)
    report["recenter_clicks"] = CLICKS_FOR_MODE.get(report["mode"])
    save_debug_json(artifact_dir / "result.json", report)
    print(f"[INFO] SEM box 이동 모드: {report['mode']} (clicks={report['recenter_clicks']}, "
          f"green={ {k: round(v, 3) for k, v in ratios.items()} })")
    return report


def main() -> int:
    """열린 tool 창(또는 저장 이미지)에서 모드만 판별한다. 클릭 없음."""
    from dotenv import load_dotenv
    load_dotenv()
    image_path = os.getenv("CLICK_MODE_IMAGE", "").strip()
    if image_path:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
    else:
        from poc.workflow_3.rcs.login_rcs_common import REMOTE_MONITORING_WINDOW_TITLE_PREFIX
        from poc.workflow_3.util import capture_window
        from poc.workflow_3.util.window_utils import find_window_by_title_prefix
        window, _, _ = find_window_by_title_prefix(REMOTE_MONITORING_WINDOW_TITLE_PREFIX)
        if window is None or not callable(capture_window):
            print("[ERROR] 열린 Remote Monitoring 창이 없습니다. tool 을 먼저 열어주세요.")
            return 2
        image = capture_window(window)
    report = detect_click_mode(image)
    return {MODE_CROSSHAIR: 0, MODE_L_SHAPE: 0, MODE_UNKNOWN: 1}[report["mode"]]


if __name__ == "__main__":
    raise SystemExit(main())
