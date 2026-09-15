"""라이브 SEM box 의 이동 모드(crosshair / L자 아이콘) 판별.

tool 창의 라이브 SEM box 옆에는 crosshair 아이콘과 L자 아이콘이 있고, **선택된 쪽이
초록으로 채워진다**(사용자 보고 2026-09-15). 모드에 따라 박스 안 이동 클릭 수가 다르다:
crosshair = 더블클릭, L자 = 싱글클릭. `sem_monitor.controller.RECENTER_CLICKS` 가 이
값을 쓴다(지금은 env `ALIGN_SEM_RECENTER_CLICKS` 로 오피스에서 맞춘다).

역할 분담은 저장소 규칙 그대로다 - VLM 은 두 아이콘의 **위치**만 찾고(요소당 호출
하나), 어느 쪽이 켜졌는지는 **픽셀 색**이 답한다. 아이콘은 **라이브 SEM box 바로
오른쪽에 세로로 나란히** 있다(사용자 보고 2026-09-15). 전체 창에서는 너무 작아 coarse
단계가 잡지 못하므로(1회차 오피스: 미검출), 오피스 검증된 `detect_sem_box` 로 박스를 먼저
잡고 그 오른쪽 strip 만 잘라 그 안에서 찾는다 - 그 crop 안에서는 아이콘이 크고 후보도
그 둘뿐이다. 박스를 못 잡으면 전체 창으로 폴백한다(종전 동작). 아이콘엔 라벨이 없어 OCR 확인이
불가능하므로, 초록 채움 자체가 확인 게이트다: 두 bbox 중 초록 비율이 뚜렷이 높은 쪽만
활성으로 보고, 둘 다 초록이거나 둘 다 아니면 unknown 이다(추측하지 않는다).

단독 점검 (오피스, tool 창이 열려 있을 때; 클릭 없음):
  uv run python -m poc.workflow_3.sem_monitor.click_mode
  CLICK_MODE_IMAGE=/path/tool.jpg uv run python -m poc.workflow_3.sem_monitor.click_mode

live 창에서 돌리면 판별 뒤 **커서를 crosshair -> L자 아이콘 중심으로 차례로 옮겨** 놓는다
(클릭 없음, 눈으로 위치 검증용). 끄려면 CLICK_MODE_MOVE_CURSOR=0, 체류 CLICK_MODE_HOVER_SEC.
"""

import os
import time

import numpy as np
from PIL import Image

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box
from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_SERVICE
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

# 아이콘 strip: 라이브 SEM box 오른쪽 경계부터 박스 폭의 이 비율만큼(최소 px). 세로는 박스와
# 같은 범위에 위아래 pad. 첫 오피스 실행의 strip.jpg 를 보고 맞춘다.
STRIP_WIDTH_RATIO = 0.18
STRIP_MIN_WIDTH_PX = 90
STRIP_VERTICAL_PAD_PX = 16

# 아이콘 열에서의 위치(위에서부터, 0-based). 사용자 확인 2026-09-15: crosshair 3번째, L자 5번째.
# 위치가 고정이라 strip 안에서는 VLM 없이 세로 blob 분할로 찾는다(VLM 은 분할 실패 시 폴백).
ICON_INDEX = {"crosshair": 2, "l_shape": 4}
MIN_ICON_RUNS = 5              # 분할된 아이콘 run 이 이보다 적으면 분할을 믿지 않는다.
ICON_MIN_HEIGHT_PX = 6         # 이보다 낮은 run 은 잡음(구분선 등)으로 버린다.
ICON_BG_DIFF = 40              # 배경(최빈색)과의 채널 차이가 이 이상이면 아이콘 픽셀.
ICON_ROW_MIN_FRACTION = 0.03   # 한 행에서 아이콘 픽셀이 이 비율 이상이어야 아이콘 행.

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
        description="the crosshair icon button (a plus sign '+' with a small circle at the "
                    "centre), the THIRD icon from the top in the vertical column of small tool "
                    "icons immediately to the RIGHT of the live SEM image. One of these icons "
                    "may be filled green. Return the icon button itself, not any crosshair "
                    "drawn inside the image.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=1.0,
        min_crop_width=96, min_crop_height=96,
    ),
    MODE_L_SHAPE: TargetConfig(
        key="sem_l_shape_icon",
        description="the L-shaped icon button (a corner bracket like the letter 'L'), the "
                    "FIFTH icon from the top in the vertical column of small tool icons "
                    "immediately to the RIGHT of the live SEM image, two below the crosshair "
                    "icon. One of these icons may be filled green.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=1.0,
        min_crop_width=96, min_crop_height=96,
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


def icon_strip_box(sem_box: dict, image_size) -> dict:
    """라이브 SEM box 오른쪽의 아이콘 strip(이미지 픽셀). 이미지 밖으로는 나가지 않는다."""
    width, height = image_size
    box_w = max(1, sem_box["right"] - sem_box["left"])
    strip_w = max(STRIP_MIN_WIDTH_PX, int(box_w * STRIP_WIDTH_RATIO))
    return {"left": min(width - 1, sem_box["right"]),
            "top": max(0, sem_box["top"] - STRIP_VERTICAL_PAD_PX),
            "right": min(width, sem_box["right"] + strip_w),
            "bottom": min(height, sem_box["bottom"] + STRIP_VERTICAL_PAD_PX)}


def segment_icon_runs(strip) -> list[dict]:
    """strip 을 위에서 아래로 훑어 아이콘 blob 의 bbox(strip 좌표) 목록을 낸다.

    배경은 strip 의 최빈색이고, 어느 채널이든 그 색과 ICON_BG_DIFF 이상 다르면 아이콘
    픽셀이다. 아이콘 행이 연속된 구간이 하나의 아이콘이며, 좌우는 그 구간 안의 아이콘
    픽셀 범위다. 아이콘 위치가 고정(3번째/5번째)이라 이 목록의 index 로 바로 고른다.
    """
    rgb = np.asarray(strip.convert("RGB") if isinstance(strip, Image.Image) else strip).astype(np.int16)
    if rgb.size == 0:
        return []
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]
    mask = (np.abs(rgb - bg) >= ICON_BG_DIFF).any(axis=2)
    row_on = mask.mean(axis=1) >= ICON_ROW_MIN_FRACTION
    runs, start = [], None
    for y, on in enumerate(list(row_on) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= ICON_MIN_HEIGHT_PX:
                cols = np.where(mask[start:y].any(axis=0))[0]
                runs.append({"left": int(cols[0]), "top": start,
                             "right": int(cols[-1]) + 1, "bottom": y})
            start = None
    return runs


def _locate_sem_box(image, client, artifact_dir) -> dict | None:
    """오피스 검증된 detect_sem_box 로 라이브 SEM box(px)를 잡는다. 실패는 None."""
    try:
        det = detect_sem_box(image, client)
        return det.bbox_px if isinstance(det.bbox_px, dict) else None
    except Exception as exc:
        print(f"[WARNING] live SEM box 검출 예외(전체 창으로 폴백): {exc}")
        return None


def detect_click_mode(image, *, client=None, artifact_dir=None) -> dict:
    """tool 창 이미지에서 이동 모드를 판별한다. 반환 dict 의 `mode` 는 세 값 중 하나."""
    artifact_dir = artifact_dir or (DEBUG_IMAGE_DIR / "click_mode" / str(time.time_ns()))
    report = {"mode": MODE_UNKNOWN, "icons": {}, "artifact_dir": str(artifact_dir)}
    client = client or Workflow1VLMClient(
        service_slug=os.getenv("ALIGN_FAIL_SEM_BOX_SERVICE", DEFAULT_SCREEN_ANALYSIS_SERVICE))
    sem_box = _locate_sem_box(image, client, artifact_dir)
    report["sem_box"] = sem_box
    if sem_box is not None:
        strip = icon_strip_box(sem_box, image.size)
        search = image.crop((strip["left"], strip["top"], strip["right"], strip["bottom"]))
        save_debug_jpeg(search, artifact_dir / "strip.jpg")
    else:
        strip = {"left": 0, "top": 0, "right": image.width, "bottom": image.height}
        search = image
        print("[WARNING] live SEM box 미검출 - 아이콘을 전체 창에서 찾는다(작아서 실패하기 쉬움)")
    report["strip"] = strip
    # 1차: 위치 고정 분할(VLM 없음). strip 이 있고 run 이 충분할 때만 믿는다.
    runs = segment_icon_runs(search) if sem_box is not None else []
    report["icon_runs"] = runs
    use_runs = len(runs) >= MIN_ICON_RUNS
    print(f"[INFO] 아이콘 열 분할: runs={len(runs)} -> {'위치 고정 사용' if use_runs else 'VLM 폴백'}")
    ratios = {}
    for mode, target in ICON_TARGETS.items():
        result = None
        box = None
        source = "segment" if use_runs else "vlm"
        try:
            if use_runs:
                box = dict(runs[ICON_INDEX[mode]])
            else:
                result = analyze_window_target(
                    None, "tool window", "image", target, image=search,
                    debug_image_dir=artifact_dir / "locator", log_name="click_mode",
                    component_name="click_mode", artifact_prefix=mode,
                )
                box = _icon_box(result, search.size)
            if box is not None:  # strip 좌표 -> 전체 이미지 좌표
                box = {"left": box["left"] + strip["left"], "top": box["top"] + strip["top"],
                       "right": box["right"] + strip["left"], "bottom": box["bottom"] + strip["top"]}
        except Exception as exc:
            print(f"[WARNING] {mode} 아이콘 로케이트 예외: {exc}")
            box = None
        info = {"box": box, "green_ratio": None, "source": source,
                "exit_code": getattr(result, "exit_code", "n/a")}
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
    if not image_path and os.getenv("CLICK_MODE_MOVE_CURSOR", "1") != "0":
        _hover_icons(window, image.size, report)
    return {MODE_CROSSHAIR: 0, MODE_L_SHAPE: 0, MODE_UNKNOWN: 1}[report["mode"]]


def _hover_icons(window, image_size, report) -> None:
    """검출한 아이콘 중심으로 커서만 옮긴다(클릭 없음). 좌표 변환은 DPI 보정 포함."""
    from poc.workflow_3.util import image_point_to_screen, move_cursor_to_screen
    if not callable(image_point_to_screen) or not callable(move_cursor_to_screen):
        print("[INFO] 커서 이동 생략(Windows 유틸 없음)")
        return
    hover_sec = float(os.getenv("CLICK_MODE_HOVER_SEC", "2.0"))
    for mode in (MODE_CROSSHAIR, MODE_L_SHAPE):
        box = report["icons"].get(mode, {}).get("box")
        if not box:
            print(f"[INFO] 커서 이동 생략: {mode} 아이콘 미검출")
            continue
        point = {"x": (box["left"] + box["right"]) // 2, "y": (box["top"] + box["bottom"]) // 2}
        screen = image_point_to_screen(window, point, image_size=image_size)
        if screen is None:
            print(f"[WARNING] 커서 이동 생략: {mode} 화면 좌표 변환 실패")
            continue
        move_cursor_to_screen(screen, f"click_mode_{mode}")
        print(f"[INFO] 커서 -> {mode} 아이콘 (image={point}, screen={screen}) {hover_sec}s 체류")
        time.sleep(hover_sec)


if __name__ == "__main__":
    raise SystemExit(main())
