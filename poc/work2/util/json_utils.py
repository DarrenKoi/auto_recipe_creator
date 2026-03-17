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

    numeric: float
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
