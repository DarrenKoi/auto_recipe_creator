"""라이브 SEM box 의 이동 모드(crosshair / L자 아이콘) 판별.

tool 창의 라이브 SEM box **오른쪽**에는 버튼이 세로로 한 줄 있고, crosshair 아이콘과 L자
아이콘이 그 안에 있다. **선택된 쪽이 초록으로 채워진다**(사용자 보고 2026-09-15). 모드에
따라 박스 안 이동 클릭 수가 다르다: crosshair = 더블클릭, L자 = 싱글클릭.
`sem_monitor.controller.RECENTER_CLICKS` 가 이 값을 쓴다(지금은 env
`ALIGN_SEM_RECENTER_CLICKS` 로 오피스에서 맞춘다).

버튼 열은 **맨 아래 `DDS` 부터** 위로 (사용자 확인 2026-09-15, 3회차):
  DDS, Next, ACD, AMS, 그림, AMP, (빈 버튼), #, L자, 네모, 십자(crosshair), =, ||
**아이콘 직접 로케이트가 1차다**(9회차 사용자 지적): 열 전체 높이의 strip 에서 zoom 단계로
찾으면 버튼이 잡힌다(요소당 호출 하나, 열 안 순서를 설명에 넣는다). 못 찾은 아이콘은
**앵커 기하**로 채운다: 맨 위 `||` 와 그 아래 `=` 는 VLM 이 안정적으로 잡히므로(오피스
5~9회차) 둘의 간격이 곧 버튼 pitch 이고 `=` 가 원점이다 - crosshair = 1칸 아래, 네모 2칸,
L자 3칸. `DDS` 등 글자 버튼은 쓰지 않는다(비활성 회색 라벨은 OCR 이 못 읽고, 활성인 DDS
도 먼 끝이라 셈이 누적 오차를 가진다 - 사용자 결정 2026-09-15). 어느 쪽이 켜졌는지는 그
자리의 **픽셀 색**이 답한다. 아이콘도 앵커도 못 찾거나 색이 불명확하면 unknown 이다.

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
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

MODE_CROSSHAIR = "crosshair"   # 더블클릭 이동
MODE_L_SHAPE = "l_shape"       # 싱글클릭 이동
MODE_UNKNOWN = "unknown"

CLICKS_FOR_MODE = {MODE_CROSSHAIR: 2, MODE_L_SHAPE: 1}

# 버튼 열, 맨 아래부터(0 = DDS). 사용자 확인 2026-09-15.
BUTTONS_FROM_BOTTOM = ["DDS", "Next", "ACD", "AMS", "picture", "AMP", "empty",
                       "hash", MODE_L_SHAPE, "square", MODE_CROSSHAIR, "equals", "pipes"]
ANCHOR_TOP, ANCHOR_ORIGIN = "pipes", "equals"   # 인접한 두 앵커: 간격 = pitch, '=' 가 원점.
# 원점 보정(버튼 단위, +1 = 모든 상자를 한 칸 위로). 콘솔의 버튼별 초록 비율 표로 맞춘다.
ORIGIN_SHIFT = int(os.getenv("CLICK_MODE_ORIGIN_SHIFT", "0"))

# 검색 strip: 라이브 SEM box 오른쪽 경계부터 박스 폭의 이 비율(최소 px), 세로는 **창 전체**
# (2회차 오피스: 박스 높이로 자르면 열 위쪽이 잘린다).
STRIP_WIDTH_RATIO = 0.18
STRIP_MIN_WIDTH_PX = 90
MIN_PITCH_PX = 6               # 앵커 간 pitch 가 이보다 작으면 앵커가 잘못 잡힌 것.
ICON_HALF_H_RATIO = 0.42       # 예측 상자 세로 반높이 = pitch 비율(이웃 버튼을 안 물게).

# 초록 판정(HSV, OpenCV 규약 H 0-180). 7회차 오피스: 앵커는 맞는데 unknown - 채움이 면이
# 아니라 글리프 선이면 비율이 작다. 그래서 절대 하한은 낮게, 판정은 상대(배수)로 한다.
GREEN_H_MIN, GREEN_H_MAX = 30, 95
GREEN_S_MIN, GREEN_V_MIN = 50, 60
ACTIVE_MIN_RATIO = 0.02        # 활성: 초록 비율이 이 이상이고 상대보다 ACTIVE_MARGIN 배 이상.
ACTIVE_MARGIN = 2.0

ICON_TARGETS = {
    MODE_CROSSHAIR: TargetConfig(
        key="sem_icon_crosshair",
        description="the crosshair icon button (a '+' cross, may have a small circle at the "
                    "centre) in the vertical column of small buttons immediately to the RIGHT "
                    "of the live SEM image. Counting from the top of that column: '||', '=', "
                    "then this crosshair (3rd), then a square, then an L-shape. It may be "
                    "filled green. Return the button itself, not any crosshair drawn inside "
                    "the live image.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=2.0,
        min_crop_width=96, min_crop_height=160,
    ),
    MODE_L_SHAPE: TargetConfig(
        key="sem_icon_l_shape",
        description="the L-shaped icon button (a corner bracket like the letter 'L') in the "
                    "vertical column of small buttons immediately to the RIGHT of the live "
                    "SEM image. Counting from the top of that column: '||', '=', crosshair, "
                    "square, then this L-shape (5th), then a '#' grid icon below it. It may "
                    "be filled green.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=2.0,
        min_crop_width=96, min_crop_height=160,
    ),
}

ANCHOR_TARGETS = {
    ANCHOR_TOP: TargetConfig(
        key="sem_column_pipes",
        description="the small button at the very TOP of the vertical column of buttons "
                    "immediately to the right of the live SEM image. Its icon is two vertical "
                    "bars '||' (like a pause symbol). The button directly below it shows '='.",
        left_pad_ratio=1.0, right_pad_ratio=1.0, vertical_pad_ratio=2.0,
        min_crop_width=96, min_crop_height=160,
    ),
    ANCHOR_ORIGIN: TargetConfig(
        key="sem_column_equals",
        description="the small button showing '=' (two horizontal lines), the SECOND button "
                    "from the top of the vertical column of buttons immediately to the right "
                    "of the live SEM image, directly below the '||' button.",
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


def _cy(box: dict) -> float:
    return (box["top"] + box["bottom"]) / 2


def column_boxes(top_box: dict, eq_box: dict, origin_shift: int = 0) -> dict:
    """인접 앵커 `||`(위)와 `=`(원점)로 13개 버튼의 예측 bbox(같은 좌표계)를 낸다.

    pitch = `=` 중심 - `||` 중심(인접이라 정확). `=` 를 원점으로 아래로 센다 - crosshair 는
    바로 아래 1칸이라 오차가 누적되지 않는다. `origin_shift` 는 원점 보정(버튼 단위, +1 =
    전부 한 칸 위). x 범위는 `=` 버튼 폭, 상자 반높이는 pitch 의 비율이라 이웃 버튼을 물지
    않는다. pitch 이상(너무 작거나 뒤집힘)이면 ValueError.
    """
    pitch = _cy(eq_box) - _cy(top_box)
    if pitch < MIN_PITCH_PX:
        raise ValueError(f"button pitch too small or inverted: {pitch:.1f}px")
    half_h = pitch * ICON_HALF_H_RATIO
    origin_idx = BUTTONS_FROM_BOTTOM.index(ANCHOR_ORIGIN)
    boxes = {}
    for k, name in enumerate(BUTTONS_FROM_BOTTOM):
        cy = _cy(eq_box) + (origin_idx - k - origin_shift) * pitch
        boxes[name] = {"left": int(eq_box["left"]), "top": int(round(cy - half_h)),
                       "right": int(eq_box["right"]), "bottom": int(round(cy + half_h))}
    return boxes


def _offset(box: dict, dx: int, dy: int) -> dict:
    return {"left": box["left"] + dx, "top": box["top"] + dy,
            "right": box["right"] + dx, "bottom": box["bottom"] + dy}


def box_from_result(result, image_size, *, default_half_w: int, default_half_h: int) -> dict:
    """로케이터 결과 -> 상자. **중심은 fine point**, 크기만 coarse bbox 에서 빌린다.

    `TargetResult.bbox` 는 coarse 단계의 bbox 라 한 칸 아래 버튼에 걸리는 일이 있고, fine
    단계(zoom)가 그것을 바로잡은 점이 `point` 다(10회차 오피스: zoom overlay 는 정확한데
    column.jpg 만 한 칸 아래 = coarse bbox 를 그대로 쓴 탓). 그래서 위치는 point 만 믿는다.
    """
    x, y = int(result.point["x"]), int(result.point["y"])
    half_w, half_h = default_half_w, default_half_h
    if isinstance(result.bbox, dict):
        half_w = max(4, (result.bbox["right"] - result.bbox["left"]) // 2)
        half_h = max(4, (result.bbox["bottom"] - result.bbox["top"]) // 2)
    return {"left": max(0, x - half_w), "top": max(0, y - half_h),
            "right": min(image_size[0], x + half_w), "bottom": min(image_size[1], y + half_h)}


def _locate_icon(mode: str, strip_image, strip: dict, *, artifact_dir) -> dict | None:
    """아이콘 버튼 하나를 strip 에서 직접 찾는다(요소당 호출 하나). 전체 이미지 좌표 bbox 또는 None."""
    result = analyze_window_target(
        None, "tool window", "image", ICON_TARGETS[mode], image=strip_image,
        debug_image_dir=artifact_dir / "locator", log_name="click_mode",
        component_name="click_mode", artifact_prefix=f"icon_{mode}",
    )
    if result.exit_code != "success" or result.point is None:
        print(f"[WARNING] 아이콘 {mode} 직접 로케이트 실패: {result.exit_code}")
        return None
    box = _offset(box_from_result(result, strip_image.size, default_half_w=12, default_half_h=10),
                  strip["left"], strip["top"])
    print(f"[INFO] 아이콘 {mode} 직접 로케이트: box={box}")
    return box


def _locate_anchor(name: str, strip_image, strip: dict, *, artifact_dir) -> dict | None:
    """기호 버튼 앵커(`||`, `=`) 하나를 strip 에서 찾는다(요소당 호출 하나). 전체 이미지 좌표 bbox 또는 None."""
    result = analyze_window_target(
        None, "tool window", "image", ANCHOR_TARGETS[name], image=strip_image,
        debug_image_dir=artifact_dir / "locator", log_name="click_mode",
        component_name="click_mode", artifact_prefix=f"anchor_{name}",
    )
    if result.exit_code != "success" or result.point is None:
        print(f"[WARNING] 앵커 {name} 미검출: {result.exit_code}")
        return None
    box = _offset(box_from_result(result, strip_image.size, default_half_w=20, default_half_h=8),
                  strip["left"], strip["top"])
    print(f"[INFO] 앵커 {name}: box={box}")
    return box


def _locate_sem_box(image, client) -> dict | None:
    """오피스 검증된 detect_sem_box 로 라이브 SEM box(px)를 잡는다. 실패는 None."""
    try:
        det = detect_sem_box(image, client)
        return det.bbox_px if isinstance(det.bbox_px, dict) else None
    except Exception as exc:
        print(f"[WARNING] live SEM box 검출 예외: {exc}")
        return None


def detect_click_mode(image, *, client=None, artifact_dir=None) -> dict:
    """tool 창 이미지에서 이동 모드를 판별한다. 반환 dict 의 `mode` 는 세 값 중 하나."""
    artifact_dir = artifact_dir or (DEBUG_IMAGE_DIR / "click_mode" / str(time.time_ns()))
    report = {"mode": MODE_UNKNOWN, "diagnosis": "sem_box_missing", "icons": {}, "anchors": {},
              "artifact_dir": str(artifact_dir)}
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

    # 1차: 아이콘 직접 로케이트.
    direct = {}
    for mode in (MODE_CROSSHAIR, MODE_L_SHAPE):
        try:
            direct[mode] = _locate_icon(mode, strip_image, strip, artifact_dir=artifact_dir)
        except Exception as exc:
            print(f"[WARNING] 아이콘 {mode} 로케이트 예외: {exc}")
            direct[mode] = None
    report["direct"] = direct

    # 2차(폴백 + 교차검사): 앵커 기하 - `||` 와 `=`.
    anchors = {}
    for name in (ANCHOR_TOP, ANCHOR_ORIGIN):
        try:
            anchors[name] = _locate_anchor(name, strip_image, strip, artifact_dir=artifact_dir)
        except Exception as exc:
            print(f"[WARNING] 앵커 {name} 로케이트 예외: {exc}")
            anchors[name] = None
    report["anchors"] = anchors
    boxes = None
    if anchors[ANCHOR_TOP] is not None and anchors[ANCHOR_ORIGIN] is not None:
        try:
            boxes = column_boxes(anchors[ANCHOR_TOP], anchors[ANCHOR_ORIGIN], ORIGIN_SHIFT)
        except ValueError as exc:
            print(f"[WARNING] 버튼 열 기하 이상: {exc}")
    if boxes is not None:
        report["column"] = boxes
        pitch = _cy(anchors[ANCHOR_ORIGIN]) - _cy(anchors[ANCHOR_TOP])
        report["pitch_px"] = pitch
        print(f"[INFO] 버튼 pitch={pitch:.1f}px (||->= 인접), origin_shift={ORIGIN_SHIFT}")
        report["column_green"] = {}
        for k, name in reversed(list(enumerate(BUTTONS_FROM_BOTTOM))):
            b = boxes[name]
            g = green_ratio(image.crop((b["left"], b["top"], b["right"], b["bottom"])))
            report["column_green"][name] = g
            print(f"[INFO]   [{k:2d}] {name:<10} y={b['top']}-{b['bottom']} green={g:.3f}")
        overlay = image.copy().convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for k, name in enumerate(BUTTONS_FROM_BOTTOM):
            b = boxes[name]
            draw.rectangle((b["left"], b["top"], b["right"], b["bottom"]), outline="red")
            draw.text((b["right"] + 3, b["top"]), f"{k}:{name}", fill="red")
        for mode, b in direct.items():
            if b:
                draw.rectangle((b["left"], b["top"], b["right"], b["bottom"]), outline="blue")
        save_debug_jpeg(overlay.crop((strip["left"], 0, min(image.width, strip["right"] + 80), image.height)),
                        artifact_dir / "column.jpg")
        for mode, b in direct.items():
            if b:
                off = (_cy(b) - _cy(boxes[mode])) / pitch
                print(f"[INFO] {mode}: 직접 로케이트 vs 기하 예측 차이 = {off:+.2f}칸")
    else:
        report["diagnosis"] = "anchor_missing:" + ",".join(
            n for n in (ANCHOR_TOP, ANCHOR_ORIGIN) if anchors.get(n) is None) or "geometry"
        print(f"[WARNING] 앵커 기하 없음({report['diagnosis']}) - 직접 로케이트만으로 판정")

    chosen = {}
    for mode in (MODE_CROSSHAIR, MODE_L_SHAPE):
        if direct.get(mode) is not None:
            chosen[mode] = (direct[mode], "direct")
        elif boxes is not None:
            chosen[mode] = (boxes[mode], "geometry")
    if len(chosen) < 2:
        report["diagnosis"] = "icon_missing:" + ",".join(m for m in (MODE_CROSSHAIR, MODE_L_SHAPE) if m not in chosen)
        save_debug_json(artifact_dir / "result.json", report)
        print(f"[INFO] SEM box 이동 모드: unknown ({report['diagnosis']})")
        return report

    ratios = {}
    for mode in (MODE_CROSSHAIR, MODE_L_SHAPE):
        b, source = chosen[mode]
        crop = image.crop((b["left"], b["top"], b["right"], b["bottom"]))
        save_debug_jpeg(crop, artifact_dir / f"{mode}.jpg")
        ratios[mode] = green_ratio(crop)
        report["icons"][mode] = {"box": b, "green_ratio": ratios[mode], "source": source}
    report["mode"] = classify_mode(ratios)
    report["recenter_clicks"] = CLICKS_FOR_MODE.get(report["mode"])
    if report["mode"] != MODE_UNKNOWN:
        report["diagnosis"] = "ok"
    elif max(ratios.values()) < ACTIVE_MIN_RATIO:
        report["diagnosis"] = "no_green"        # 두 상자 모두 초록이 거의 없다: 색 띠/상자 위치 의심
    else:
        report["diagnosis"] = "ambiguous_green"  # 둘 다 초록: 상자가 이웃을 물었거나 채움색이 아닌 초록
    save_debug_json(artifact_dir / "result.json", report)
    print(f"[INFO] SEM box 이동 모드: {report['mode']} (clicks={report['recenter_clicks']}, "
          f"green={ {k: round(v, 3) for k, v in ratios.items()} }, diagnosis={report['diagnosis']})")
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
