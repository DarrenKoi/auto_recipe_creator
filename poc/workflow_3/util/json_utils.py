"""VLM 응답 JSON 처리 유틸리티."""

import ast
import json
import re


_TRAILING_COMMA_PATTERN = re.compile(r",(\s*[}\]])")
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """markdown code fence 를 벗긴다."""
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_first_balanced_object(text: str) -> str:
    """문자열 안의 첫 balanced JSON object 부분만 추출한다."""
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _normalize_json_candidate(text: str) -> str:
    """JSON 파싱 전에 자주 섞이는 노이즈를 정리한다."""
    normalized = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .strip()
    )
    normalized = _TRAILING_COMMA_PATTERN.sub(r"\1", normalized)
    return normalized


def _try_parse_candidate(text: str) -> dict | None:
    """후보 문자열을 dict 로 파싱한다."""
    if not text:
        return None

    normalized = _normalize_json_candidate(text)
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(normalized)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def extract_json(text: str) -> dict:
    """VLM 응답 텍스트에서 JSON 객체를 추출한다."""
    candidates = [
        _strip_code_fence(text),
        _extract_first_balanced_object(text),
        text.strip(),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed = _try_parse_candidate(normalized)
        if parsed is not None:
            return parsed

    raise json.JSONDecodeError("JSON object not found", text, 0)


def coerce_float(value) -> float | None:
    """문자열/숫자를 float 로 변환한다. 변환 불가 시 None."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def normalize_bbox_1000(raw_bbox) -> dict | None:
    """모델 bbox 응답을 0-1000 기준 bbox dict 로 정규화한다."""
    if raw_bbox is None:
        return None

    if isinstance(raw_bbox, dict):
        if {"left", "top", "right", "bottom"} <= raw_bbox.keys():
            left = coerce_float(raw_bbox.get("left"))
            top = coerce_float(raw_bbox.get("top"))
            right = coerce_float(raw_bbox.get("right"))
            bottom = coerce_float(raw_bbox.get("bottom"))
        elif {"x", "y", "width", "height"} <= raw_bbox.keys():
            left = coerce_float(raw_bbox.get("x"))
            top = coerce_float(raw_bbox.get("y"))
            width = coerce_float(raw_bbox.get("width"))
            height = coerce_float(raw_bbox.get("height"))
            if None in {left, top, width, height}:
                return None
            right = left + width
            bottom = top + height
        elif {"x", "y", "w", "h"} <= raw_bbox.keys():
            left = coerce_float(raw_bbox.get("x"))
            top = coerce_float(raw_bbox.get("y"))
            width = coerce_float(raw_bbox.get("w"))
            height = coerce_float(raw_bbox.get("h"))
            if None in {left, top, width, height}:
                return None
            right = left + width
            bottom = top + height
        else:
            return None
    elif isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        left = coerce_float(raw_bbox[0])
        top = coerce_float(raw_bbox[1])
        right = coerce_float(raw_bbox[2])
        bottom = coerce_float(raw_bbox[3])
    else:
        return None

    if None in {left, top, right, bottom}:
        return None

    left = int(round(max(0.0, min(1000.0, left))))
    top = int(round(max(0.0, min(1000.0, top))))
    right = int(round(max(0.0, min(1000.0, right))))
    bottom = int(round(max(0.0, min(1000.0, bottom))))
    if right <= left or bottom <= top:
        return None

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _bbox_coord_to_pixel(value: float, axis_size: int) -> int:
    """0-1000 정규화 좌표 하나를 픽셀 좌표로 변환한다."""
    pixel = value / 1000.0 * (axis_size - 1)
    return max(0, min(int(round(pixel)), axis_size - 1))


def bbox_1000_to_pixels(bbox_1000: dict, img_w: int, img_h: int) -> dict:
    """0-1000 bbox 를 픽셀 bbox 로 변환한다."""
    left = _bbox_coord_to_pixel(bbox_1000["left"], img_w)
    top = _bbox_coord_to_pixel(bbox_1000["top"], img_h)
    right = max(left + 1, min(img_w, _bbox_coord_to_pixel(bbox_1000["right"], img_w) + 1))
    bottom = max(top + 1, min(img_h, _bbox_coord_to_pixel(bbox_1000["bottom"], img_h) + 1))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def bbox_center(bbox: dict) -> dict[str, int]:
    """bbox 중심 좌표를 반환한다."""
    center_x = int(round((bbox["left"] + bbox["right"] - 1) / 2))
    center_y = int(round((bbox["top"] + bbox["bottom"] - 1) / 2))
    return {"x": center_x, "y": center_y}


def _bbox_corners(raw_bbox):
    """다양한 bbox 표현에서 (left, top, right, bottom) raw 값을 뽑는다(clamp/스케일 없음)."""
    if isinstance(raw_bbox, dict):
        if {"left", "top", "right", "bottom"} <= raw_bbox.keys():
            return (
                coerce_float(raw_bbox.get("left")),
                coerce_float(raw_bbox.get("top")),
                coerce_float(raw_bbox.get("right")),
                coerce_float(raw_bbox.get("bottom")),
            )
        for wkey, hkey in (("width", "height"), ("w", "h")):
            if {"x", "y", wkey, hkey} <= raw_bbox.keys():
                left = coerce_float(raw_bbox.get("x"))
                top = coerce_float(raw_bbox.get("y"))
                width = coerce_float(raw_bbox.get(wkey))
                height = coerce_float(raw_bbox.get(hkey))
                if None in {left, top, width, height}:
                    return None
                return left, top, left + width, top + height
        return None
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        return tuple(coerce_float(v) for v in raw_bbox)
    return None


def bbox_to_pixels(raw_bbox, img_w: int, img_h: int, coord_system=None) -> dict | None:
    """모델 bbox 를 *coord_system 을 존중해* 픽셀 bbox 로 변환한다(없으면 None).

    ``bbox_1000_to_pixels`` 는 항상 0-1000 정규화로 가정하므로, 모델이 pixel/relative_1/
    percent 로 응답하면 잘못 스케일된다. 본 함수는 좌표값별로 ``_to_pixel_coordinate`` 를
    써 coord_system(+크기 휴리스틱)을 반영한다. 결과 bbox 는 [0, axis-1] 로 clamp 되고
    right>left, bottom>top 을 보장한다(아니면 None).
    """
    corners = _bbox_corners(raw_bbox)
    if corners is None or None in corners:
        return None
    cs = _normalize_coord_system(coord_system)
    left, _ = _to_pixel_coordinate(corners[0], img_w, cs)
    top, _ = _to_pixel_coordinate(corners[1], img_h, cs)
    right, _ = _to_pixel_coordinate(corners[2], img_w, cs)
    bottom, _ = _to_pixel_coordinate(corners[3], img_h, cs)
    if None in (left, top, right, bottom):
        return None
    if right <= left or bottom <= top:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _normalize_coord_system(value) -> str | None:
    """좌표계 문자열을 내부 표준값으로 정규화한다."""
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    aliases = {
        "pixel": "pixel",
        "pixels": "pixel",
        "absolute_pixel": "pixel",
        "absolute_pixels": "pixel",
        "relative_1000": "relative_1000",
        "normalized_1000": "relative_1000",
        "0_1000": "relative_1000",
        "0-1000": "relative_1000",
        "relative_1": "relative_1",
        "normalized_0_1": "relative_1",
        "normalized_1": "relative_1",
        "0_1": "relative_1",
        "0-1": "relative_1",
        "percent": "percent",
        "%": "percent",
    }
    return aliases.get(text)


def _to_pixel_coordinate(
    value,
    axis_size: int,
    coord_system: str | None = None,
) -> tuple[int | None, str]:
    """숫자/문자/정규화 좌표를 이미지 픽셀 좌표로 변환한다."""
    if axis_size <= 0 or value is None or isinstance(value, bool):
        return None, "invalid"

    is_percent = False
    looks_fractional = False

    if isinstance(value, int):
        numeric = float(value)
    elif isinstance(value, float):
        numeric = value
        looks_fractional = not value.is_integer()
    else:
        text = str(value).strip()
        if not text:
            return None, "invalid"
        if text.endswith("%"):
            text = text[:-1].strip()
            is_percent = True
        looks_fractional = "." in text or "e" in text.lower()
        try:
            numeric = float(text)
        except ValueError:
            return None, "invalid"

    max_index = axis_size - 1
    mode = "pixel"

    if coord_system == "pixel":
        mode = "pixel"
    elif coord_system == "relative_1000":
        numeric = (numeric / 1000.0) * max_index
        mode = "relative_1000"
    elif coord_system == "relative_1":
        numeric = numeric * max_index
        mode = "relative_1"
    elif coord_system == "percent" or is_percent:
        numeric = (numeric / 100.0) * max_index
        mode = "percent"
    elif 0.0 <= numeric <= 1.0 and looks_fractional:
        numeric = numeric * max_index
        mode = "normalized_0_1"
    elif 0.0 <= numeric <= max_index:
        mode = "pixel"
    elif 0.0 <= numeric <= 1000.0:
        numeric = (numeric / 1000.0) * max_index
        mode = "normalized_0_1000"
    else:
        mode = "pixel_clamped"

    coord = int(round(numeric))
    clamped = max(0, min(coord, max_index))
    if clamped != coord:
        mode = f"{mode}+clamped"
    return clamped, mode


def parse_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """VLM 응답 좌표를 픽셀 정수로 변환하고 범위를 보정한다."""
    coord_system = _normalize_coord_system(
        data.get("coord_system") or data.get("coordinate_system")
    )
    if coord_system:
        print(f"[INFO] coord_system={coord_system}")

    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"  [MISS] {key:20s} - VLM 응답에 없음")
            continue

        raw_x, raw_y = pt.get("x"), pt.get("y")
        x, x_mode = _to_pixel_coordinate(raw_x, img_w, coord_system)
        y, y_mode = _to_pixel_coordinate(raw_y, img_h, coord_system)
        if x is None or y is None:
            print(f"  [MISS] {key:20s} - 좌표 변환 실패 raw=({raw_x}, {raw_y})")
            continue

        data[key] = {"x": x, "y": y}
        print(
            f"  [RAW ] {key:20s} - raw=({raw_x}, {raw_y}) "
            f"-> px=({x}, {y}) [x:{x_mode}, y:{y_mode}]"
        )
    return data


def normalize_tool_text(text: str) -> str:
    """OCR 텍스트 비교를 위해 영숫자만 남기고 대문자로 정규화한다."""
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def normalize_lines(raw_text: str, max_items: int = 120) -> list[str]:
    """OCR 응답을 고유 줄 목록으로 정리한다."""
    if not raw_text.strip():
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_items:
            break
    return lines


def describe_bbox_reject(raw_bbox) -> str:
    """``normalize_bbox_1000`` 이 None 을 돌려준 이유를 한 토큰으로 말한다.

    미검출 로그가 "못 찾았다"만 찍으면 오피스에서 할 수 있는 일이 없다 - 모델이
    아예 답을 안 준 것(absent)과, 좌표를 줬는데 우리가 버린 것(degenerate/bad_shape)은
    고치는 방법이 정반대다: 앞은 프롬프트/화면 문제, 뒤는 파싱/규약 문제.
    None 이 아닌 값에는 "ok" 를 돌려준다(호출부의 방어용).
    """
    if raw_bbox is None:
        return "absent"          # 키 없음 또는 null - 모델이 '없다'고 답한 것.
    corners = None
    if isinstance(raw_bbox, dict):
        for keys in (("left", "top", "right", "bottom"),
                     ("x", "y", "width", "height"),
                     ("x", "y", "w", "h")):
            if set(keys) <= raw_bbox.keys():
                corners = [coerce_float(raw_bbox[k]) for k in keys]
                break
    elif isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        corners = [coerce_float(v) for v in raw_bbox]
    if corners is None:
        return "bad_shape"       # dict/list 모양이 규약과 다름.
    if None in corners:
        return "non_numeric"     # 좌표 자리에 숫자가 아닌 값.
    if any(v < 0 for v in corners):
        return "refusal"         # [-1,-1,-1,-1] 계열 = 모델의 명시적 거부.
    if normalize_bbox_1000(raw_bbox) is None:
        return "degenerate"      # 폭/높이가 0 이하 (clamp 후 붕괴 포함).
    return "ok"
