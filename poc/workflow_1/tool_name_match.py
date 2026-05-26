"""OCR 혼동 문자에 강건한 tool 이름 매칭 (canonicalize + exact).

Spotting 으로 읽은 텍스트는 영숫자 ID 에서 자주 혼동된다 (O↔0, I↔1, B↔8 등).
편집거리 대신 혼동 문자를 하나의 대표 문자로 정규화(canonicalize)한 뒤 정확히
일치(token equality)하는지만 본다. 고정 길이 ID 에서는 이 방식이 fuzzy threshold
보다 오검출이 적다.
"""

# 혼동되기 쉬운 글자 → 대표 문자(주로 숫자). 보수적으로만 매핑한다.
# D→0, T→7, A→4 등은 의미 있는 접두(MCD 계열 등)를 뭉개므로 제외한다.
_CONFUSION_MAP = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "G": "6",
}


def canonicalize(text: str) -> str:
    """대문자화 후 영숫자만 남기고 혼동 문자를 대표 문자로 치환한다."""
    result: list[str] = []
    for ch in (text or "").upper():
        if not ch.isalnum():
            continue
        result.append(_CONFUSION_MAP.get(ch, ch))
    return "".join(result)


def _bbox_area(bbox: dict) -> int:
    """bbox 면적 (tie-break 용)."""
    width = max(0, bbox["right"] - bbox["left"])
    height = max(0, bbox["bottom"] - bbox["top"])
    return width * height


def best_match(items: list[dict], target_name: str) -> dict | None:
    """canonicalize + exact 기준으로 가장 적합한 spotting item 을 고른다.

    - item 의 텍스트를 공백 토큰으로 나눠 각 토큰을 canonicalize 한 값이 target 의
      canonical 값과 정확히 같으면 후보.
    - 줄 전체 canonical 이 target 과 같은 경우도 후보 (단일 토큰 라인).
    - 여러 후보면 bbox 가 가장 작은 것을 고른다 (한 행의 ID 텍스트가 가장 타이트).

    매칭 없으면 None.
    """
    canonical_target = canonicalize(target_name)
    if not canonical_target:
        return None

    candidates: list[dict] = []
    for item in items:
        text = item.get("text", "")
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            continue

        token_canons = {canonicalize(tok) for tok in str(text).split()}
        token_canons.add(canonicalize(text))
        if canonical_target in token_canons:
            candidates.append(item)

    if not candidates:
        return None
    return min(candidates, key=lambda it: _bbox_area(it["bbox"]))


__all__ = ["canonicalize", "best_match"]
