"""라이브 SEM box 의 이동 모드(crosshair / L자 아이콘) 판별.

tool 창의 라이브 SEM box **오른쪽**에는 버튼이 세로로 한 줄 있고, crosshair 아이콘과 L자
아이콘이 그 안에 있다. **선택된 쪽이 초록으로 채워진다**(사용자 보고 2026-09-15). 모드에
따라 박스 안 이동 클릭 수가 다르다: crosshair = 더블클릭, L자 = 싱글클릭.
`sem_monitor.controller.RECENTER_CLICKS` 가 이 값을 쓴다(지금은 env
`ALIGN_SEM_RECENTER_CLICKS` 로 오피스에서 맞춘다).

버튼 열은 **맨 아래 `DDS` 부터** 위로 (사용자 확인 2026-09-15, 3회차):
  DDS, Next, ACD, AMS, 그림, AMP, (빈 버튼), #, L자, 네모, 십자(crosshair), =, ||
아이콘 자체를 VLM 에 묻는 두 차례 오피스 시도는 실패했다(전체 창에선 너무 작고, strip
분할은 열 위가 잘리거나 아이콘 일부만 잡혔다). 그래서 **글자 버튼을 앵커로** 쓴다 -
`DDS` 와 `AMS` 는 텍스트라 VLM 이 찾고 PaddleOCR 이 확인할 수 있으며(아이콘은 확인이
원리상 불가능), 두 앵커 사이 간격(3칸)으로 버튼 pitch 를 얻어 `DDS` 위로 k 칸을 센다:
L자 = 8칸 위, crosshair = 10칸 위. 어느 쪽이 켜졌는지는 그 자리의 **픽셀 색**이 답한다.
앵커 확인 실패/색 불명확은 unknown 이다(추측하지 않는다).

단독 점검 (오피스, tool 창이 열려 있을 때; 클릭 없음):
  uv run python -m poc.workflow_3.sem_monitor.click_mode
  CLICK_MODE_IMAGE=/path/tool.jpg uv run python -m poc.workflow_3.sem_monitor.click_mode

live 창에서 돌리면 판별 뒤 **커서를 crosshair -> L자 아이콘 중심으로 차례로 옮겨** 놓는다
(클릭 없음, 눈으로 위치 검증용). 끄려면 CLICK_MODE_MOVE_CURSOR=0, 체류 CLICK_MODE_HOVER_SEC.
산출물 `column.jpg` 에 13개 버튼의 예측 상자가 번호와 함께 그려진다.
"""

import os
import time

import numpy as np
from PIL import Image, ImageDraw

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box
from poc.workflow_3.vlm.flask_vlm import DEFAULT_SCREEN_ANALYSIS_SERVICE
from poc.workflow_3.vlm.label_verify import label_matches, read_text_near_point
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

MODE_CROSSHAIR = "crosshair"   # 더블클릭 이동
MODE_L_SHAPE = "l_shape"       # 싱글클릭 이동
MODE_UNKNOWN = "unknown"

CLICKS_FOR_MODE = {MODE_CROSSHAIR: 2, MODE_L_SHAPE: 1}

# 버튼 열, 맨 아래부터(0 = DDS). 사용자 확인 2026-09-15.
BUTTONS_FROM_BOTTOM = ["DDS", "Next", "ACD", "AMS", "picture", "AMP", "empty",
                       "hash", MODE_L_SHAPE, "square", MODE_CROSSHAIR, "equals", "pipes"]
ANCHOR_BOTTOM, ANCHOR_UPPER = "DDS", "AMS"   # 둘 다 글자 버튼 = OCR 확인 가능.
ANCHOR_GAP = BUTTONS_FROM_BOTTOM.index(ANCHOR_UPPER) - BUTTONS_FROM_BOTTOM.index(ANCHOR_BOTTOM)

# 검색 strip: 라이브 SEM box 오른쪽 경계부터 박스 폭의 이 비율(최소 px), 세로는 **창 전체**
# (2회차 오피스: 박스 높이로 자르면 열 위쪽이 잘린다).
STRIP_WIDTH_RATIO = 0.18
STRIP_MIN_WIDTH_PX = 90
MIN_PITCH_PX = 6               # 앵커 간 pitch 가 이보다 작으면 앵커가 잘못 잡힌 것.
ICON_HALF_H_RATIO = 0.42       # 예측 상자 세로 반높이 = pitch 비율(이웃 버튼을 안 물게).

# 초록 판정(HSV, OpenCV 규약 H 0-180). 첫 오피스 실행의 green_ratio 값을 보고 좁힌다.
GREEN_H_MIN, GREEN_H_MAX = 35, 90
GREEN_S_MIN, GREEN_V_MIN = 80, 80
ACTIVE_MIN_RATIO = 0.10        # 활성: 초록 비율이 이 이상이고 상대보다 ACTIVE_MARGIN 배 이상.
ACTIVE_MARGIN = 2.0

ANCHOR_TARGETS = {
    ANCHOR_BOTTOM: TargetConfig(
        key="sem_column_dds",
        description="the small button labelled 'DDS' at the very BOTTOM of the vertical column "
                    "of buttons immediately to the right of the live SEM image.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=2.0,
        min_crop_width=96, min_crop_height=160,
    ),
    ANCHOR_UPPER: TargetConfig(
        key="sem_column_ams",
        description="the small button labelled 'AMS' in the vertical column of buttons "
                    "immediately to the right of the live SEM image, three buttons above 'DDS' "
                    "(order from the bottom: DDS, Next, ACD, AMS).",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=2.0,
        min_crop_width=96, min_crop_height=160,
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


def icon_strip_box(sem_box: dict, image_size) -> dict:
    """라이브 SEM box 오른쪽의 버튼 열 strip(이미지 픽셀). 세로는 창 전체."""
    width, height = image_size
    box_w = max(1, sem_box["right"] - sem_box["left"])
    strip_w = max(STRIP_MIN_WIDTH_PX, int(box_w * STRIP_WIDTH_RATIO))
    return {"left": min(width - 1, sem_box["right"]), "top": 0,
            "right": min(width, sem_box["right"] + strip_w), "bottom": height}


def column_boxes(dds_box: dict, ams_box: dict) -> dict:
    """두 앵커 bbox 로 13개 버튼의 예측 bbox(같은 좌표계)를 낸다. pitch 이상이면 ValueError.

    x 범위는 DDS 버튼 폭, y 는 DDS 중심에서 k*pitch 위. 상자 반높이는 pitch 의 비율이라
    이웃 버튼을 물지 않는다(초록 판정이 옆 버튼에 오염되지 않게).
    """
    dds_cy = (dds_box["top"] + dds_box["bottom"]) / 2
    ams_cy = (ams_box["top"] + ams_box["bottom"]) / 2
    pitch = (dds_cy - ams_cy) / ANCHOR_GAP
    if pitch < MIN_PITCH_PX:
        raise ValueError(f"button pitch too small or inverted: {pitch:.1f}px")
    half_h = pitch * ICON_HALF_H_RATIO
    boxes = {}
    for k, name in enumerate(BUTTONS_FROM_BOTTOM):
        cy = dds_cy - k * pitch
        boxes[name] = {"left": int(dds_box["left"]), "top": int(round(cy - half_h)),
                       "right": int(dds_box["right"]), "bottom": int(round(cy + half_h))}
    return boxes


def _offset(box: dict, dx: int, dy: int) -> dict:
    return {"left": box["left"] + dx, "top": box["top"] + dy,
            "right": box["right"] + dx, "bottom": box["bottom"] + dy}


def _locate_anchor(name: str, strip_image, strip: dict, full_image, *, artifact_dir, ocr_client):
    """글자 버튼 하나를 strip 에서 찾고(요소당 호출 하나) 그 자리 라벨을 OCR 로 확인한다.

    반환은 전체 이미지 좌표의 bbox 이거나, 미검출/라벨 불일치면 None. 확인은 strict -
    앵커가 틀리면 13개 상자가 전부 틀리므로 lenient 통과는 여기서 의미가 없다.
    """
    result = analyze_window_target(
        None, "tool window", "image", ANCHOR_TARGETS[name], image=strip_image,
        debug_image_dir=artifact_dir / "locator", log_name="click_mode",
        component_name="click_mode", artifact_prefix=f"anchor_{name.lower()}",
    )
    if result.exit_code != "success" or result.point is None:
        print(f"[WARNING] 앵커 {name} 미검출: {result.exit_code}")
        return None
    if isinstance(result.bbox, dict):
        box = result.bbox
    else:
        x, y = result.point["x"], result.point["y"]
        box = {"left": max(0, x - 20), "top": max(0, y - 8),
               "right": min(strip_image.width, x + 20), "bottom": min(strip_image.height, y + 8)}
    box = _offset(box, strip["left"], strip["top"])
    read = read_text_near_point(
        full_image, box, debug_image_dir=artifact_dir, timestamp_tag=name.lower(),
        artifact_label=f"anchor_{name.lower()}", log_name="click_mode", client=ocr_client,
    )
    ok = read.ok and label_matches(read.tokens or read.raw_text.split(), name)
    print(f"[INFO] 앵커 {name}: box={box} OCR={read.raw_text!r} -> {'확인' if ok else '거부'}")
    return box if ok else None


def _locate_sem_box(image, client) -> dict | None:
    """오피스 검증된 detect_sem_box 로 라이브 SEM box(px)를 잡는다. 실패는 None."""
    try:
        det = detect_sem_box(image, client)
        return det.bbox_px if isinstance(det.bbox_px, dict) else None
    except Exception as exc:
        print(f"[WARNING] live SEM box 검출 예외: {exc}")
        return None


def detect_click_mode(image, *, client=None, ocr_client=None, artifact_dir=None) -> dict:
    """tool 창 이미지에서 이동 모드를 판별한다. 반환 dict 의 `mode` 는 세 값 중 하나."""
    artifact_dir = artifact_dir or (DEBUG_IMAGE_DIR / "click_mode" / str(time.time_ns()))
    report = {"mode": MODE_UNKNOWN, "icons": {}, "anchors": {}, "artifact_dir": str(artifact_dir)}
    client = client or Workflow1VLMClient(
        service_slug=os.getenv("ALIGN_FAIL_SEM_BOX_SERVICE", DEFAULT_SCREEN_ANALYSIS_SERVICE))
    sem_box = _locate_sem_box(image, client)
    report["sem_box"] = sem_box
    if sem_box is None:
        print("[WARNING] live SEM box 미검출 - 버튼 열을 찾을 기준이 없어 unknown")
        save_debug_json(artifact_dir / "result.json", report)
        return report
    strip = icon_strip_box(sem_box, image.size)
    strip_image = image.crop((strip["left"], strip["top"], strip["right"], strip["bottom"]))
    save_debug_jpeg(strip_image, artifact_dir / "strip.jpg")
    report["strip"] = strip

    anchors = {}
    for name in (ANCHOR_BOTTOM, ANCHOR_UPPER):
        try:
            anchors[name] = _locate_anchor(name, strip_image, strip, image,
                                           artifact_dir=artifact_dir, ocr_client=ocr_client)
        except Exception as exc:
            print(f"[WARNING] 앵커 {name} 로케이트 예외: {exc}")
            anchors[name] = None
    report["anchors"] = anchors
    if any(anchors[n] is None for n in (ANCHOR_BOTTOM, ANCHOR_UPPER)):
        save_debug_json(artifact_dir / "result.json", report)
        print("[INFO] SEM box 이동 모드: unknown (앵커 미확인)")
        return report

    try:
        boxes = column_boxes(anchors[ANCHOR_BOTTOM], anchors[ANCHOR_UPPER])
    except ValueError as exc:
        print(f"[WARNING] 버튼 열 기하 이상: {exc}")
        save_debug_json(artifact_dir / "result.json", report)
        return report
    report["column"] = boxes

    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for k, name in enumerate(BUTTONS_FROM_BOTTOM):
        b = boxes[name]
        draw.rectangle((b["left"], b["top"], b["right"], b["bottom"]), outline="red")
        draw.text((b["right"] + 3, b["top"]), f"{k}:{name}", fill="red")
    save_debug_jpeg(overlay.crop((strip["left"], 0, min(image.width, strip["right"] + 80), image.height)),
                    artifact_dir / "column.jpg")

    ratios = {}
    for mode in (MODE_CROSSHAIR, MODE_L_SHAPE):
        b = boxes[mode]
        crop = image.crop((b["left"], b["top"], b["right"], b["bottom"]))
        save_debug_jpeg(crop, artifact_dir / f"{mode}.jpg")
        ratios[mode] = green_ratio(crop)
        report["icons"][mode] = {"box": b, "green_ratio": ratios[mode],
                                 "steps_above_dds": BUTTONS_FROM_BOTTOM.index(mode)}
    report["mode"] = classify_mode(ratios)
    report["recenter_clicks"] = CLICKS_FOR_MODE.get(report["mode"])
    save_debug_json(artifact_dir / "result.json", report)
    print(f"[INFO] SEM box 이동 모드: {report['mode']} (clicks={report['recenter_clicks']}, "
          f"green={ {k: round(v, 3) for k, v in ratios.items()} })")
    return report


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


def main() -> int:
    """열린 tool 창(또는 저장 이미지)에서 모드만 판별한다. 클릭 없음."""
    from dotenv import load_dotenv
    load_dotenv()
    image_path = os.getenv("CLICK_MODE_IMAGE", "").strip()
    window = None
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
    if window is not None and os.getenv("CLICK_MODE_MOVE_CURSOR", "1") != "0":
        _hover_icons(window, image.size, report)
    return {MODE_CROSSHAIR: 0, MODE_L_SHAPE: 0, MODE_UNKNOWN: 1}[report["mode"]]


if __name__ == "__main__":
    raise SystemExit(main())
