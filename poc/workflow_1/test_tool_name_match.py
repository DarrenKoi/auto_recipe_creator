"""ocr_spotting 파서 + tool_name_match 매처 오프라인 검증 (RCS 불필요).

Mac 에서 `uv run python poc/workflow_1/test_tool_name_match.py` 로 실행한다.
"""

import json
import sys

from poc.workflow_1.ocr_spotting import parse_spotting_items
from poc.workflow_1.tool_name_match import best_match, canonicalize


def _check(name: str, condition: bool) -> bool:
    """단건 검증 결과를 출력하고 통과 여부를 반환한다."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def test_canonicalize() -> bool:
    """혼동 문자 정규화가 일관적인지 확인한다."""
    ok = True
    ok &= _check("canon MCD63O == MCD630", canonicalize("MCD63O") == canonicalize("MCD630"))
    ok &= _check("canon MCDG30 == MCD630", canonicalize("MCDG30") == canonicalize("MCD630"))
    ok &= _check("canon MCD680 != MCD630", canonicalize("MCD680") != canonicalize("MCD630"))
    ok &= _check("canon strips punctuation/space", canonicalize(" mcd-630 ") == canonicalize("MCD630"))
    ok &= _check("canon keeps D distinct from 0", canonicalize("MCD630") == "MCD630")
    return ok


def test_parse_dict_bbox() -> bool:
    """dict bbox 형태를 파싱한다."""
    raw = json.dumps([
        {"text": "MCD630", "bbox": {"left": 10, "top": 20, "right": 110, "bottom": 50}},
        {"text": "READY", "bbox": {"left": 200, "top": 20, "right": 260, "bottom": 50}},
    ])
    items = parse_spotting_items(raw)
    return _check("parse dict bbox -> 2 items", len(items) == 2)


def test_parse_xyxy_and_polygon() -> bool:
    """[x1,y1,x2,y2] 와 polygon 형태를 파싱한다."""
    ok = True
    raw_xyxy = json.dumps([{"text": "MCD631", "box": [5, 6, 95, 36]}])
    items_xyxy = parse_spotting_items(raw_xyxy)
    ok &= _check(
        "parse xyxy box",
        len(items_xyxy) == 1 and items_xyxy[0]["bbox"] == {"left": 5, "top": 6, "right": 95, "bottom": 36},
    )
    raw_poly = json.dumps([{"text": "MCD632", "polygon": [[0, 0], [100, 0], [100, 30], [0, 30]]}])
    items_poly = parse_spotting_items(raw_poly)
    ok &= _check(
        "parse polygon box",
        len(items_poly) == 1 and items_poly[0]["bbox"] == {"left": 0, "top": 0, "right": 100, "bottom": 30},
    )
    return ok


def test_parse_nested_wrapper() -> bool:
    """중첩 wrapper 구조에서 후보를 끌어낸다."""
    raw = json.dumps({
        "prunedResult": {
            "parsing_res_list": [
                {"text": "MCD630", "block_bbox": [10, 20, 110, 50]},
                {"rec_text": "MCD640", "bbox": {"left": 10, "top": 60, "right": 110, "bottom": 90}},
            ]
        }
    })
    items = parse_spotting_items(raw)
    texts = {it["text"] for it in items}
    return _check("parse nested wrapper -> MCD630 & MCD640", {"MCD630", "MCD640"} <= texts)


def test_best_match_confusions() -> bool:
    """OCR 혼동에도 정확한 행을 고르고, 인접 ID 는 제외한다."""
    items = [
        {"text": "MCD63O", "bbox": {"left": 10, "top": 20, "right": 110, "bottom": 50}},
        {"text": "MCD680", "bbox": {"left": 10, "top": 60, "right": 110, "bottom": 90}},
    ]
    ok = True
    hit = best_match(items, "MCD630")
    ok &= _check("best_match picks MCD63O for MCD630", hit is not None and hit["text"] == "MCD63O")

    only_neighbor = [{"text": "MCD680", "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 30}}]
    ok &= _check("best_match rejects MCD680 for MCD630", best_match(only_neighbor, "MCD630") is None)
    return ok


def test_best_match_bundled_token() -> bool:
    """ID 가 다른 텍스트와 한 줄에 묶여 있어도 토큰 일치로 잡는다."""
    items = [
        {"text": "● MCD630  RUN", "bbox": {"left": 10, "top": 20, "right": 240, "bottom": 50}},
    ]
    hit = best_match(items, "MCD630")
    return _check("best_match matches bundled-token line", hit is not None)


def test_best_match_smallest_bbox_tiebreak() -> bool:
    """동률이면 가장 타이트한 bbox 를 고른다."""
    items = [
        {"text": "MCD630", "bbox": {"left": 0, "top": 0, "right": 300, "bottom": 200}},
        {"text": "MCD630", "bbox": {"left": 10, "top": 20, "right": 110, "bottom": 50}},
    ]
    hit = best_match(items, "MCD630")
    return _check(
        "best_match tie-break smallest bbox",
        hit is not None and hit["bbox"] == {"left": 10, "top": 20, "right": 110, "bottom": 50},
    )


def test_best_match_refuses_ambiguous_rows() -> bool:
    """서로 다른 행에서 같은 canonical 이 2개 이상이면 매칭을 거부한다."""
    # 가령 target 'CDS5' 와 다른 tool 'CD55' 가 둘 다 'CD55' 로 정규화되어
    # 화면의 서로 다른 두 행에 등장 → 어느 행이 진짜인지 알 수 없으므로 거부.
    items = [
        {"text": "CDS5", "bbox": {"left": 10, "top": 20, "right": 110, "bottom": 50}},
        {"text": "CD55", "bbox": {"left": 10, "top": 120, "right": 110, "bottom": 150}},
    ]
    return _check("best_match refuses two distinct rows", best_match(items, "CDS5") is None)


def test_best_match_same_row_variants_ok() -> bool:
    """같은 행을 OCR 이 다른 텍스트로 두 번 잡아도(세로 겹침) 거부하지 않는다."""
    items = [
        {"text": "MCD630", "bbox": {"left": 10, "top": 20, "right": 110, "bottom": 50}},
        {"text": "MCD63O", "bbox": {"left": 12, "top": 22, "right": 108, "bottom": 48}},
    ]
    hit = best_match(items, "MCD630")
    return _check("best_match keeps same-row variants", hit is not None)


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_canonicalize,
        test_parse_dict_bbox,
        test_parse_xyxy_and_polygon,
        test_parse_nested_wrapper,
        test_best_match_confusions,
        test_best_match_bundled_token,
        test_best_match_smallest_bbox_tiebreak,
        test_best_match_refuses_ambiguous_rows,
        test_best_match_same_row_variants_ok,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] tool_name_match 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
