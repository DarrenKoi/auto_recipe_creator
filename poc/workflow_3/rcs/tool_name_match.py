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


def _distinct_row_count(items: list[dict]) -> int:
    """후보 bbox 들이 세로로 몇 개의 행 그룹을 이루는지 센다.

    같은 행을 OCR 이 여러 번(다른 텍스트로) 잡은 경우는 세로로 겹치므로 한 그룹.
    서로 다른 행에서 잡힌 경우는 세로로 떨어져 있어 별도 그룹이 된다.
    """
    boxes = sorted((item["bbox"] for item in items), key=lambda b: b["top"])
    clusters = 0
    cluster_bottom: int | None = None
    for bbox in boxes:
        if cluster_bottom is None or bbox["top"] >= cluster_bottom:
            clusters += 1
            cluster_bottom = bbox["bottom"]
        else:
            cluster_bottom = max(cluster_bottom, bbox["bottom"])
    return clusters


def best_match(items: list[dict], target_name: str) -> dict | None:
    """canonicalize + exact 기준으로 가장 적합한 spotting item 을 고른다.

    - item 의 텍스트를 공백 토큰으로 나눠 각 토큰을 canonicalize 한 값이 target 의
      canonical 값과 정확히 같으면 후보.
    - 줄 전체 canonical 이 target 과 같은 경우도 후보 (단일 토큰 라인).
    - 후보가 서로 다른 행(세로로 분리)에 2개 이상 있으면 모호하므로 매칭을 거부한다
      (혼동 문자 정규화로 인해 실제로 다른 tool 이 같은 canonical 이 된 경우).
      이때 None 을 돌려주어 상위에서 VLM grounding 으로 fallback 하게 한다.
    - 단일 행 안에서 여러 후보면 bbox 가 가장 작은 것을 고른다 (ID 텍스트가 가장 타이트).

    매칭 없거나 모호하면 None.
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
    if _distinct_row_count(candidates) > 1:
        print(
            f"[WARNING] tool 이름 매칭 모호: target={target_name!r} 가 서로 다른 "
            f"{_distinct_row_count(candidates)}개 행에서 검출됨 → 매칭 거부(VLM fallback)"
        )
        return None
    return min(candidates, key=lambda it: _bbox_area(it["bbox"]))


__all__ = ["canonicalize", "best_match"]
