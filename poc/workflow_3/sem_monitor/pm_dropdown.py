"""PM 버튼 드롭다운으로 배율(magnification)을 바꾸는 fallback 메커니즘.

배경: 이 tool 은 live SEM box 위 mouse wheel 로 배율이 바뀌지 않는다(오피스 확인).
배율을 바꾸려면 'PM' 버튼(배율 숫자 바로 왼쪽의 라벨)을 클릭해 뜨는 **드롭다운**에서
원하는 배율 값을 골라야 한다. zoom ladder 는 fail 시점 배율 기준으로 값공간에서
한 칸 낮은 값(OUT)·한 칸 높은 값(IN)을 차례로 골라 각 배율의 화면을 저장한다.

wheel 방식과의 차이:
  * wheel = 상대 이동(한 notch 씩, 누적 드리프트 가능) — 이 tool 에선 무효.
  * PM 드롭다운 = **절대 배율 선택**(목록의 값을 직접 지정) — 드리프트 없음, baseline
    복귀 불필요.

역할 분담(프로젝트 규약): VLM/OCR 은 '읽기/위치'만, 좌표/판정은 CV. 여기서는
PaddleOCR-VL 의 ``Spotting:`` 한 번으로 드롭다운 각 행의 (텍스트 + bbox)를 받아
배율 값과 클릭 박스를 동시에 얻는다(작은 crop 에만 적용 — 전체 스크린샷 OCR 환각 회피).

이 모듈은 좌표/파싱 순수 함수 + 드롭다운 읽기만 담당한다. 실제 클릭/캡처/재매칭은
``monitor/cycle.py`` 의 fallback ladder 가 이 함수들을 엮어 수행한다.
"""

import re

from poc.workflow_3.sem_monitor.sem_box_detect import parse_pm_magnification
from poc.workflow_3.util.env_utils import env_float, env_int
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_spotting_prompt

# --- 튜닝(오피스 RCS UI 에 맞춰 env 로 보정). ---
# PM 버튼('PM' 라벨)은 배율 숫자 박스 바로 왼쪽에 있다. 클릭 지점 x = 숫자박스.left -
# gap. gap 은 숫자박스 폭 대비 비율(없으면 최소값). 라벨이 더/덜 떨어져 있으면 보정.
PM_BUTTON_GAP_RATIO = env_float("ALIGN_FAIL_PM_BTN_GAP_RATIO", 0.9)
PM_BUTTON_MIN_GAP = env_int("ALIGN_FAIL_PM_BTN_MIN_GAP", 14)
# 드롭다운 crop 영역: PM 버튼/숫자 행을 기준으로 아래로 frame 높이의 이 비율만큼,
# 좌우로 숫자박스 폭의 이 배수만큼 pad 해서 잡는다.
PM_DROPDOWN_DOWN_RATIO = env_float("ALIGN_FAIL_PM_DD_DOWN_RATIO", 0.45)
PM_DROPDOWN_SIDE_PAD_RATIO = env_float("ALIGN_FAIL_PM_DD_SIDE_PAD_RATIO", 1.5)
PM_DROPDOWN_UP_PAD_RATIO = env_float("ALIGN_FAIL_PM_DD_UP_PAD_RATIO", 0.5)
# 드롭다운은 'PM 버튼 바로 아래'에 펼쳐진다 — 버튼 점 기준 crop(좌/우는 frame 폭 비율,
# 위는 버튼에서 살짝 내려서 시작, 아래로 frame 높이 비율).
PM_DD_LEFT_RATIO = env_float("ALIGN_FAIL_PM_DD_LEFT_RATIO", 0.04)
PM_DD_RIGHT_RATIO = env_float("ALIGN_FAIL_PM_DD_RIGHT_RATIO", 0.12)
PM_DD_TOP_GAP_RATIO = env_float("ALIGN_FAIL_PM_DD_TOP_GAP_RATIO", 0.0)


def pm_button_point(pm_box_px):
    """배율 숫자 박스(pm_box_px: l/t/r/b 픽셀)에서 'PM' 버튼 클릭 지점을 유도한다.

    'PM' 라벨은 숫자 바로 왼쪽 → x = left - gap, y = 숫자 박스 세로 중심.
    pm_box_px 가 없으면 None.
    """
    if not pm_box_px:
        return None
    try:
        l = int(pm_box_px["left"]); t = int(pm_box_px["top"])
        r = int(pm_box_px["right"]); b = int(pm_box_px["bottom"])
    except (KeyError, TypeError, ValueError):
        return None
    w = max(1, r - l)
    gap = max(PM_BUTTON_MIN_GAP, int(round(w * PM_BUTTON_GAP_RATIO)))
    x = max(0, l - gap)
    y = (t + b) // 2
    return {"x": x, "y": y}


def dropdown_region(pm_box_px, frame_wh):
    """PM 버튼 클릭 후 드롭다운이 펼쳐질 crop 영역 (l, t, r, b) 픽셀을 추정한다.

    드롭다운은 PM 행 근처에서 아래로 펼쳐진다고 가정한다. 현재 선택 행이 위에 올 수도
    있어 PM 행 위로도 약간(up_pad) 잡는다. frame 경계로 clamp. 추정이 빗나가도
    호출부가 _pm_dropdown.jpg 전체 캡처를 함께 저장하므로 눈으로 보정 가능하다.
    """
    if not pm_box_px or not frame_wh:
        return None
    try:
        fw, fh = int(frame_wh[0]), int(frame_wh[1])
        l0 = int(pm_box_px["left"]); t0 = int(pm_box_px["top"])
        r0 = int(pm_box_px["right"]); b0 = int(pm_box_px["bottom"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    w = max(1, r0 - l0)
    h = max(1, b0 - t0)
    btn = pm_button_point(pm_box_px)
    btn_x = btn["x"] if btn else l0
    side = int(round(w * PM_DROPDOWN_SIDE_PAD_RATIO))
    l = max(0, btn_x - side)
    r = min(fw, r0 + side)
    t = max(0, t0 - int(round(h * PM_DROPDOWN_UP_PAD_RATIO)))
    b = min(fh, t0 + int(round(fh * PM_DROPDOWN_DOWN_RATIO)))
    if r - l < 4 or b - t < 4:
        return None
    return (l, t, r, b)


def dropdown_region_below(button_xy, frame_wh):
    """'PM 버튼 바로 아래'에 펼쳐지는 드롭다운의 crop 영역 (l, t, r, b) 픽셀을 만든다.

    버튼 점(button_xy: {"x","y"} 풀프레임 px) 기준으로 좌/우는 frame 폭 비율, 위는 버튼에서
    살짝 내려서(top_gap) 시작, 아래로 frame 높이 비율만큼 잡는다. 버튼 위치를 2단계 VLM 으로
    정확히 얻은 뒤 이 함수로 그 아래만 OCR 한다(숫자박스 기준 추정보다 정확).
    """
    if not button_xy or not frame_wh:
        return None
    try:
        fw, fh = int(frame_wh[0]), int(frame_wh[1])
        bx, by = int(button_xy["x"]), int(button_xy["y"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    l = max(0, bx - int(round(fw * PM_DD_LEFT_RATIO)))
    r = min(fw, bx + int(round(fw * PM_DD_RIGHT_RATIO)))
    t = max(0, by + int(round(fh * PM_DD_TOP_GAP_RATIO)))
    b = min(fh, by + int(round(fh * PM_DROPDOWN_DOWN_RATIO)))
    if r - l < 4 or b - t < 4:
        return None
    return (l, t, r, b)


def crop_region_from_bbox(coarse_bbox, frame_wh, *, pad_x_ratio=0.4, pad_y_ratio=0.3):
    """VLM coarse bbox(열린 드롭다운 영역)를 패딩+frame clamp 한 crop (l,t,r,b) 픽셀로 만든다.

    드롭다운 리스트가 coarse bbox 보다 약간 크게 렌더될 수 있어 bbox 폭/높이 비율만큼
    여유를 둔다. 고정 비율 기하 추정(dropdown_region_below)을 대체하는, VLM 으로 위치를
    찾은 영역. bbox/frame 누락·degenerate 면 None.
    """
    if not coarse_bbox or not frame_wh:
        return None
    try:
        fw, fh = int(frame_wh[0]), int(frame_wh[1])
        l0 = int(coarse_bbox["left"]); t0 = int(coarse_bbox["top"])
        r0 = int(coarse_bbox["right"]); b0 = int(coarse_bbox["bottom"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    w = max(1, r0 - l0)
    h = max(1, b0 - t0)
    px = int(round(w * pad_x_ratio))
    py = int(round(h * pad_y_ratio))
    l = max(0, l0 - px)
    t = max(0, t0 - py)
    r = min(fw, r0 + px)
    b = min(fh, b0 + py)
    if r - l < 4 or b - t < 4:
        return None
    return (l, t, r, b)


_MAG_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?\s*[kK]?")


def read_dropdown_options(crop_image, ocr_client, *, crop_origin=(0, 0)):
    """드롭다운 crop(PIL)을 PaddleOCR ``Spotting:`` 으로 읽어 행별 옵션 목록을 돌려준다.

    반환: (options, raw_text)
      options = [{"value": float, "text": str, "center": {"x","y"}}, ...]
                center 는 **풀프레임 픽셀**(crop_origin 더해진 값), 배율 오름차순 정렬·중복 제거.
      raw_text = OCR 원문(오피스 디버그용 — 파싱 실패 시 이 값으로 보정).

    crop 좌표계: encode_image_webp 는 리사이즈하지 않으므로 spotting bbox 는 crop 픽셀과
    동일 좌표계 → center + crop_origin = 풀프레임 픽셀.
    """
    if crop_image is None or ocr_client is None:
        return [], ""
    try:
        crop_b64, _cw, _ch = encode_image_webp(crop_image, quality=90)
        system_msg, user_text = build_spotting_prompt()
        resp = ocr_client.chat_with_image_b64(
            image_b64=crop_b64, system_message=system_msg, user_text=user_text,
            image_mime="image/webp", temperature=0.0,
        )
        raw_text = (resp.text or "").strip()
    except Exception as exc:
        print(f"[WARNING] PM 드롭다운 spotting 실패: {exc}")
        return [], ""

    items = parse_spotting_items(raw_text)
    ox, oy = int(crop_origin[0]), int(crop_origin[1])
    cw, ch = crop_image.size

    # 좌표계 추정: spotting bbox 가 crop 픽셀인지 / 0-1 비율 / 0-1000 정규화인지.
    # encode_image_webp 는 리사이즈하지 않으므로 crop 픽셀이 기본 가정이지만, OCR 모델이
    # 정규화 좌표를 줄 수 있어(=클릭이 엉뚱한 위치로) bbox 최대값으로 보정한다.
    max_coord = 0.0
    for it in items:
        bb = it.get("bbox")
        if bb:
            max_coord = max(max_coord, float(bb.get("right", 0)), float(bb.get("bottom", 0)))
    if items and max_coord <= 1.5:
        sx, sy, space = float(cw), float(ch), "frac01"
    elif max_coord <= max(cw, ch) * 1.05:
        sx, sy, space = 1.0, 1.0, "crop_px"
    else:
        sx, sy, space = cw / 1000.0, ch / 1000.0, "norm1000"
    if items and space != "crop_px":
        print(f"[INFO] PM 드롭다운 좌표계 보정: {space} (bbox max={max_coord:.0f}, crop={cw}x{ch})")

    by_value = {}
    for it in items:
        text = (it.get("text") or "").strip()
        bbox = it.get("bbox")
        if not text or not bbox:
            continue
        # crop 로컬 중심(좌표계 보정 적용).
        lcx = (float(bbox["left"]) + float(bbox["right"])) / 2.0 * sx
        lcy = (float(bbox["top"]) + float(bbox["bottom"])) / 2.0 * sy
        # crop 영역 밖이면 좌표계 오류 가능 — 라이브 이미지 오클릭(recenter) 방지로 버린다.
        if not (0 <= lcx <= cw and 0 <= lcy <= ch):
            continue
        cx, cy = ox + int(round(lcx)), oy + int(round(lcy))
        # 행 텍스트가 'PM' 라벨/단위 등과 섞일 수 있어 토큰별로 배율을 뽑는다.
        for tok in _MAG_TOKEN_RE.findall(text):
            value = parse_pm_magnification(tok)
            # value<=0 은 'PM 0'/구분자/잡토큰에서 나온 가짜 옵션 — 배율이 될 수 없으므로 버린다.
            if value is None or value <= 0:
                continue
            # 같은 배율이 여러 토큰/행에 나오면 첫 검출만 유지(중복 제거).
            by_value.setdefault(
                value, {"value": value, "text": tok.strip(), "center": {"x": cx, "y": cy}}
            )
    options = sorted(by_value.values(), key=lambda o: o["value"])
    return options, raw_text


def nearest_option(options, value):
    """options(각 {"value": float, ...}) 중 value 와 가장 가까운 행을 고른다. 빈 목록이면 None.

    드롭다운 재읽기 후 목표 배율값에 해당하는 현재 행을 다시 찾을 때 쓴다(배율이 바뀌어
    행이 이동해도 값 기준으로 매칭).
    """
    if not options:
        return None
    return min(options, key=lambda o: abs(o["value"] - value))


def row_target_description(value, text):
    """mai-ui 2단계 그라운더가 찾을 '드롭다운 한 행' 설명 문자열을 만든다.

    coarse(ui-venus)->fine(mai-ui) 파이프라인에 넘길 타깃 설명. 라이브 SEM 이미지의
    배율 숫자와 헷갈리지 않게 '열린 magnification dropdown 의 행'이라는 맥락을 명시하고
    그 행의 배율 텍스트를 넣는다.
    """
    label = (str(text).strip() or str(value))
    return (
        f"the row showing the magnification value '{label}' in the opened "
        "magnification dropdown list that appeared below the PM button"
    )


def choose_step_targets(options, current_mag, out_steps, in_steps):
    """배율 값공간에서 OUT(낮은 값)·IN(높은 값) 목표 행들을 고른다.

    options 는 read_dropdown_options 의 오름차순 목록. current_mag 와 값이 가장 가까운
    행을 기준 인덱스로 잡고, OUT 은 한 칸씩 아래(낮은 값), IN 은 한 칸씩 위(높은 값)로
    범위 안에서만 고른다. 반환: [(label, option), ...] (예: ("out1", {...})).
    """
    if not options or current_mag is None:
        return []
    idx = min(
        range(len(options)),
        key=lambda i: abs(options[i]["value"] - current_mag),
    )
    targets = []
    for k in range(1, max(0, out_steps) + 1):
        j = idx - k
        if j >= 0:
            targets.append((f"out{k}", options[j]))
    for k in range(1, max(0, in_steps) + 1):
        j = idx + k
        if j < len(options):
            targets.append((f"in{k}", options[j]))
    return targets


__all__ = [
    "pm_button_point",
    "dropdown_region",
    "dropdown_region_below",
    "crop_region_from_bbox",
    "read_dropdown_options",
    "choose_step_targets",
    "nearest_option",
    "row_target_description",
]
