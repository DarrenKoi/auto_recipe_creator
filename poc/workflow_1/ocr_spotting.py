"""PaddleOCR-VL `Spotting:` 응답에서 text+bbox 후보를 파싱한다.

`Spotting:` 태스크는 검출된 텍스트마다 좌표를 함께 돌려주지만, 응답 JSON 형태가
모델/버전에 따라 제각각이다 (dict bbox, [x1,y1,x2,y2], polygon, 중첩 wrapper 등).
이 모듈은 그 다양한 형태를 공통 `{"text", "bbox"}` 로 best-effort 정규화한다.

`poc/work2/tool_screen_spotting.py` 의 검증된 파서를 workflow_1 로 자립 이식한 것.
"""

import ast
import json


def _parse_json_like(text: str):
    """raw 텍스트를 JSON/파이썬 리터럴로 best-effort 파싱한다."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    candidates = [stripped]
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        candidates.append(stripped[start_obj:end_obj + 1])
    start_arr = stripped.find("[")
    end_arr = stripped.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        candidates.append(stripped[start_arr:end_arr + 1])

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    return None


def _extract_text_label(item: object) -> str:
    """spotting item 에서 텍스트 라벨을 추출한다."""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""

    for key in (
        "text",
        "content",
        "block_content",
        "transcription",
        "label",
        "word",
        "rec_text",
        "caption",
        "value",
    ):
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _is_number_list(values: object, length: int) -> bool:
    """지정 길이의 숫자 리스트인지 확인한다."""
    if not isinstance(values, list) or len(values) != length:
        return False
    return all(isinstance(value, (int, float)) for value in values)


def _polygon_to_bbox(points) -> dict | None:
    """polygon 좌표를 bbox 로 변환한다."""
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        if not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float)):
            return None
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None
    left = int(round(min(xs)))
    top = int(round(min(ys)))
    right = int(round(max(xs)))
    bottom = int(round(max(ys)))
    if right <= left or bottom <= top:
        return None
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _coerce_bbox(raw_box: object) -> dict | None:
    """여러 bbox 표현을 공통 {left, top, right, bottom} 으로 변환한다."""
    if raw_box is None:
        return None

    if _is_number_list(raw_box, 4):
        values = [int(round(float(value))) for value in raw_box]
        left, top, right, bottom = values
        if right <= left or bottom <= top:
            return None
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }

    if isinstance(raw_box, list) and raw_box and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in raw_box
    ):
        return _polygon_to_bbox(raw_box)

    if isinstance(raw_box, dict):
        if {"left", "top", "right", "bottom"} <= raw_box.keys():
            left = int(round(float(raw_box["left"])))
            top = int(round(float(raw_box["top"])))
            right = int(round(float(raw_box["right"])))
            bottom = int(round(float(raw_box["bottom"])))
            if right <= left or bottom <= top:
                return None
            return {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }
        if {"x1", "y1", "x2", "y2"} <= raw_box.keys():
            return _coerce_bbox([
                raw_box["x1"],
                raw_box["y1"],
                raw_box["x2"],
                raw_box["y2"],
            ])
        for key in ("bbox", "block_bbox", "box", "polygon", "points"):
            if key in raw_box:
                nested = _coerce_bbox(raw_box.get(key))
                if nested is not None:
                    return nested
    return None


_WRAPPER_KEYS = {
    "layoutParsingResults",
    "prunedResult",
    "spotting_res",
    "parsing_res_list",
    "result",
    "data",
    "items",
    "blocks",
    "detections",
    "texts",
}


def _collect_spotting_candidates(node: object, results: list[dict], hinted_text: str = "") -> None:
    """중첩된 응답 구조에서 spotting 후보들을 재귀적으로 수집한다."""
    if isinstance(node, list):
        for item in node:
            _collect_spotting_candidates(item, results, hinted_text=hinted_text)
        return

    if not isinstance(node, dict):
        return

    text = _extract_text_label(node) or hinted_text
    bbox = None
    for key in ("bbox", "block_bbox", "box", "polygon", "points"):
        if key in node:
            bbox = _coerce_bbox(node.get(key))
            if bbox is not None:
                break

    if bbox is None and text and len(node) == 1:
        only_value = next(iter(node.values()))
        bbox = _coerce_bbox(only_value)

    if text and bbox is not None:
        results.append({"text": text, "bbox": bbox})

    for key, value in node.items():
        if key in _WRAPPER_KEYS:
            _collect_spotting_candidates(value, results, hinted_text=hinted_text)
            continue

        if isinstance(value, (dict, list)):
            next_hint = ""
            if isinstance(value, list) and key not in _WRAPPER_KEYS:
                next_hint = str(key).strip()
            _collect_spotting_candidates(value, results, hinted_text=next_hint)


def _dedupe_spotting_items(items: list[dict]) -> list[dict]:
    """동일 텍스트/좌표 후보를 중복 제거한다."""
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for item in items:
        bbox = item["bbox"]
        signature = (
            item["text"],
            bbox["left"],
            bbox["top"],
            bbox["right"],
            bbox["bottom"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def parse_spotting_items(raw_text: str) -> list[dict]:
    """spotting raw 텍스트에서 text+bbox 후보를 추출한다.

    반환 형태: [{"text": str, "bbox": {"left","top","right","bottom"}}, ...]
    파싱 실패 시 빈 리스트.
    """
    parsed = _parse_json_like(raw_text)
    if parsed is None:
        return []

    results: list[dict] = []
    _collect_spotting_candidates(parsed, results)
    return _dedupe_spotting_items(results)


__all__ = ["parse_spotting_items"]
