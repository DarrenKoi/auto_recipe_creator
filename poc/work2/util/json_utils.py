"""VLM 응답 JSON 처리 유틸리티."""

import json


def extract_json(text: str) -> dict:
    """VLM 응답 텍스트에서 JSON 객체를 추출한다."""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return json.loads(text[start:end].strip())
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return json.loads(text[start : end + 1])
    return json.loads(text)


def parse_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """VLM 응답 좌표를 정수로 변환하고 범위를 검증한다."""
    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"  [MISS] {key:20s} - VLM 응답에 없음")
            continue
        raw_x, raw_y = pt.get("x", 0), pt.get("y", 0)
        x, y = int(raw_x), int(raw_y)
        data[key] = {"x": x, "y": y}
        suffix = ""
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            suffix = " <- OUT OF BOUNDS"
        print(f"  [RAW ] {key:20s} - raw=({raw_x}, {raw_y}) -> px=({x}, {y}){suffix}")
    return data
