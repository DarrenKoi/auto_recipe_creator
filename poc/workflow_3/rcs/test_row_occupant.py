"""row_occupant 단위 테스트 - VLM/실장비 없이 Mac 에서 실행.

  uv run pytest poc/workflow_3/rcs/test_row_occupant.py
"""

from poc.workflow_3.rcs.row_occupant import (
    FREE,
    OCCUPIED_BY_OTHER,
    UNKNOWN,
    build_occupant_box,
    classify_occupancy,
    looks_like_occupant,
)


def test_occupant_id_shape():
    """사번형 ID: 영숫자 + 글자·숫자 혼재."""
    assert looks_like_occupant("KIM0234") is True
    assert looks_like_occupant("HYN1A2B") is True


def test_pure_word_is_not_occupant():
    """상태 문자열은 점유자가 아니다."""
    assert looks_like_occupant("Idle") is False
    assert looks_like_occupant("Status") is False


def test_pure_number_is_not_occupant():
    assert looks_like_occupant("12345") is False


def test_punctuated_is_not_occupant():
    """IP 나 시각은 구두점 때문에 탈락한다."""
    assert looks_like_occupant("10.1.2.3") is False
    assert looks_like_occupant("12:30:05") is False


def test_too_short_is_not_occupant():
    assert looks_like_occupant("A1") is False


def test_too_long_is_not_occupant():
    assert looks_like_occupant("A" * 20 + "1") is False


def test_blank_is_not_occupant():
    assert looks_like_occupant("") is False
    assert looks_like_occupant(None) is False


def test_occupied_when_occupant_token_present():
    assert classify_occupancy(True, ["KIM0234"]) == OCCUPIED_BY_OTHER


def test_occupied_when_occupant_mixed_with_noise():
    assert classify_occupancy(True, ["Idle", "KIM0234", "-"]) == OCCUPIED_BY_OTHER


def test_free_when_read_ok_and_no_occupant_token():
    """읽기는 성공했는데 점유자 모양 토큰이 없으면 비어 있는 것이다."""
    assert classify_occupancy(True, []) == FREE
    assert classify_occupancy(True, ["-"]) == FREE


def test_unknown_when_read_failed():
    """읽기 실패는 '비어 있음' 이 아니라 '모름' 이다. 이 구분이 모듈의 존재 이유다."""
    assert classify_occupancy(False, []) == UNKNOWN
    assert classify_occupancy(False, ["KIM0234"]) == UNKNOWN


def test_occupant_box_extends_right_of_row_point():
    """점유자 컬럼은 장비 ID 컬럼 오른쪽에 있다."""
    box = build_occupant_box({"x": 100, "y": 200}, 1920, 1080)
    assert box["right"] > box["left"]
    assert box["right"] > 100
    assert box["top"] < 200 < box["bottom"]


def test_occupant_box_clamped_to_image():
    """행이 가장자리에 있어도 이미지 밖으로 나가지 않는다."""
    box = build_occupant_box({"x": 1900, "y": 5}, 1920, 1080)
    assert box["right"] <= 1920
    assert box["top"] >= 0
    assert box["left"] >= 0


def test_occupant_box_has_minimum_height():
    """촘촘한 행에서도 crop 이 0 높이로 무너지지 않는다."""
    box = build_occupant_box({"x": 100, "y": 200}, 1920, 1080)
    assert box["bottom"] - box["top"] >= 16


# ------------------------------------------------------------------
# read_occupancy - crop -> OCR -> 분류 전체 흐름 (OCR 은 주입).
# ------------------------------------------------------------------


class _Image:
    """PIL 이미지 대역 - read_occupancy 는 width/height 만 본다."""

    width = 1920
    height = 1080


def test_read_occupancy_occupied():
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    result = read_occupancy(
        _Image(), {"x": 100, "y": 200},
        read_tokens_fn=lambda image, box: ["KIM0234"],
    )
    assert result == OCCUPIED_BY_OTHER


def test_read_occupancy_free():
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    result = read_occupancy(
        _Image(), {"x": 100, "y": 200},
        read_tokens_fn=lambda image, box: [],
    )
    assert result == FREE


def test_read_occupancy_none_tokens_is_unknown():
    """OCR 판독 실패(None)와 '읽었는데 비어 있음'([])은 반드시 갈려야 한다."""
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    result = read_occupancy(
        _Image(), {"x": 100, "y": 200},
        read_tokens_fn=lambda image, box: None,
    )
    assert result == UNKNOWN


def test_read_occupancy_missing_image_is_unknown():
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    assert read_occupancy(
        None, {"x": 1, "y": 1}, read_tokens_fn=lambda image, box: ["KIM0234"]
    ) == UNKNOWN


def test_read_occupancy_missing_point_is_unknown():
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    assert read_occupancy(
        _Image(), None, read_tokens_fn=lambda image, box: ["KIM0234"]
    ) == UNKNOWN


def test_read_occupancy_swallows_ocr_exception():
    """판독 예외가 사이클을 죽이면 안 된다 - UNKNOWN 으로 흡수."""
    from poc.workflow_3.rcs.row_occupant import read_occupancy

    def _boom(image, box):
        raise RuntimeError("ocr down")

    assert read_occupancy(_Image(), {"x": 1, "y": 1}, read_tokens_fn=_boom) == UNKNOWN


def test_shared_predicate_keeps_tool_id_behaviour():
    """공유 헬퍼로 바꾼 뒤에도 장비 ID 판정이 그대로여야 한다 (길이만 다름)."""
    from poc.workflow_3.rcs.tool_row_verify import _looks_like_tool_id

    assert _looks_like_tool_id("MCD427") is True
    assert _looks_like_tool_id("4DCDB807") is True
    assert _looks_like_tool_id("Status") is False
    assert _looks_like_tool_id("10.1.2.3") is False
