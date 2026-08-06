"""tool_row_verify 순수 로직 스모크 테스트 (VLM/RCS 없이 Mac 에서 돈다).

강조점: 확인 게이트가 **오탐으로 정상 클릭을 막지 않는지**. 옛 게이트가 비활성화된
이유가 바로 그것이었으므로, mismatch 판정은 진짜 다른 장비 ID 일 때만 나야 한다.

  uv run python poc/workflow_3/rcs/test_tool_row_verify.py
"""

import sys

from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_LENIENT,
    CONFIRM_POLICY_OFF,
    CONFIRM_POLICY_STRICT,
    RowVerdict,
    accepts,
    build_strip_box,
    classify_tokens,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """단언 결과를 기록한다."""
    if condition:
        PASSED.append(name)
        print(f"[INFO] PASS {name}")
    else:
        FAILED.append(name)
        print(f"[ERROR] FAIL {name} {detail}")


def test_confirms_exact_id() -> None:
    """정확히 목표 ID 를 읽으면 confirmed."""
    status, _ = classify_tokens(["MCDN01"], "MCDN01")
    check("exact_id_confirmed", status == "confirmed", f"status={status}")


def test_confirms_with_surrounding_text() -> None:
    """strip 에 상태 텍스트가 같이 잡혀도 목표 ID 가 있으면 confirmed."""
    status, _ = classify_tokens(["MCDN01", "On", "10.1.2.3"], "MCDN01")
    check("id_with_context_confirmed", status == "confirmed", f"status={status}")


def test_confusion_chars_still_confirm() -> None:
    """O/0, I/1 혼동은 canonicalize 가 흡수해 confirmed 여야 한다."""
    status, _ = classify_tokens(["MCDNO1"], "MCDN01")
    check("confusion_char_confirmed", status == "confirmed", f"status={status}")


def test_neighbor_row_is_mismatch() -> None:
    """다른 장비 ID 를 읽으면 mismatch (= 옆 행 클릭)."""
    status, token = classify_tokens(["MCDA01"], "MCDN01")
    check(
        "neighbor_row_mismatch",
        status == "mismatch" and token == "MCDA01",
        f"status={status} token={token}",
    )


def test_pure_words_are_not_mismatch() -> None:
    """순수 단어는 ID 로 오인되면 안 된다 (canonicalize 가 S->5 로 바꿔도).

    이 케이스가 깨지면 게이트가 정상 클릭을 막기 시작한다 - 옛 게이트의 실패 모드.
    """
    for tokens in (["Status"], ["Model"], ["Location"], ["Normal"]):
        status, token = classify_tokens(tokens, "MCDN01")
        check(
            f"word_not_mismatch_{tokens[0]}",
            status == "unreadable",
            f"status={status} token={token}",
        )


def test_empty_read_is_unreadable() -> None:
    """아무것도 못 읽으면 unreadable (거부가 아니라 보류)."""
    status, _ = classify_tokens([], "MCDN01")
    check("empty_unreadable", status == "unreadable", f"status={status}")


def test_empty_target_is_unreadable() -> None:
    """목표 이름이 비면 판정 불가."""
    status, _ = classify_tokens(["MCDN01"], "")
    check("empty_target_unreadable", status == "unreadable", f"status={status}")


def test_short_token_not_mismatch() -> None:
    """너무 짧은 영숫자 토큰은 장비 ID 로 보지 않는다."""
    status, _ = classify_tokens(["A1"], "MCDN01")
    check("short_token_unreadable", status == "unreadable", f"status={status}")


def test_different_length_id_is_mismatch() -> None:
    """길이가 달라도 다른 장비 ID 면 mismatch 다.

    실제 목록은 길이가 섞여 있다(MCD427 6자, 4DCDB807 8자). '목표와 같은 길이' 로만
    거르면 6자 목표 옆의 8자 ID 오클릭을 놓친다.
    """
    status, token = classify_tokens(["4DCDB807"], "MCD427")
    check(
        "cross_length_mismatch",
        # canonical: B->8 만 치환된다 (D 는 의미 있는 접두라 매핑 대상이 아니다).
        status == "mismatch" and token == "4DCD8807",
        f"status={status} token={token}",
    )


def test_confusable_pairs_from_office_list() -> None:
    """실제 목록의 한 글자 차이 쌍 - 옆 행 오클릭의 핵심 케이스."""
    pairs = [
        ("MCDN01", "MCDN02"),
        ("MCDC12", "MCDC22"),
        ("MCD427", "MCD717"),
        ("RKHV3101", "RMHV3301"),
    ]
    for target, neighbor in pairs:
        status, _ = classify_tokens([neighbor], target)
        check(
            f"pair_mismatch_{target}_vs_{neighbor}",
            status == "mismatch",
            f"status={status}",
        )
        status_self, _ = classify_tokens([target], target)
        check(f"pair_confirm_{target}", status_self == "confirmed", f"status={status_self}")


def test_target_wins_when_both_rows_in_strip() -> None:
    """strip 에 목표 행과 옆 행이 같이 걸리면 confirmed (정상 클릭을 막지 않는다)."""
    status, _ = classify_tokens(["MCDN02", "MCDN01"], "MCDN01")
    check("target_wins_over_neighbor", status == "confirmed", f"status={status}")


def test_digit_leading_id_confirms() -> None:
    """숫자로 시작하는 ID(4DCDB807, 6MCD3401)도 정상 판정된다."""
    for tool_name in ("4DCDB807", "6MCD3401", "6PCDL101"):
        status, _ = classify_tokens([tool_name], tool_name)
        check(f"digit_leading_confirm_{tool_name}", status == "confirmed", f"status={status}")


def test_ip_and_counts_not_mismatch() -> None:
    """옆 컬럼이 strip 에 걸려도 ID 로 오인되면 안 된다."""
    for tokens, label in (
        (["10.1.2.3"], "ip"),
        (["192.168.0.101"], "ip_long"),
        (["12345678"], "digits_only"),
        (["Connection"], "word_only"),
    ):
        status, token = classify_tokens(tokens, "MCDN01")
        check(f"noise_not_mismatch_{label}", status == "unreadable", f"status={status} token={token}")


def test_policy_matrix() -> None:
    """정책별 통과/거부 조합."""
    confirmed = RowVerdict("confirmed", "MCDN01")
    mismatch = RowVerdict("mismatch", "MCDN01")
    unreadable = RowVerdict("unreadable", "MCDN01")
    error = RowVerdict("error", "MCDN01")

    check("off_accepts_mismatch", accepts(mismatch, CONFIRM_POLICY_OFF))
    check("lenient_accepts_confirmed", accepts(confirmed, CONFIRM_POLICY_LENIENT))
    check("lenient_rejects_mismatch", not accepts(mismatch, CONFIRM_POLICY_LENIENT))
    check("lenient_accepts_unreadable", accepts(unreadable, CONFIRM_POLICY_LENIENT))
    check("lenient_accepts_error", accepts(error, CONFIRM_POLICY_LENIENT))
    check("strict_accepts_confirmed", accepts(confirmed, CONFIRM_POLICY_STRICT))
    check("strict_rejects_unreadable", not accepts(unreadable, CONFIRM_POLICY_STRICT))
    check("strict_rejects_error", not accepts(error, CONFIRM_POLICY_STRICT))
    check("none_verdict_accepted", accepts(None, CONFIRM_POLICY_STRICT))


def test_strip_box_geometry() -> None:
    """strip 은 클릭점을 감싸고 이미지 밖으로 나가지 않는다."""
    box = build_strip_box({"x": 300, "y": 500}, 1920, 1080)
    check(
        "strip_contains_point",
        box["left"] < 300 < box["right"] and box["top"] < 500 < box["bottom"],
        f"box={box}",
    )
    check("strip_is_row_shaped", (box["bottom"] - box["top"]) < (box["right"] - box["left"]), f"box={box}")

    edge = build_strip_box({"x": 5, "y": 3}, 1920, 1080)
    check(
        "strip_clamped_at_edge",
        edge["left"] == 0 and edge["top"] == 0 and edge["right"] <= 1920,
        f"box={edge}",
    )


def test_tool_row_crop_is_row_tight() -> None:
    """fine 모델에 주는 crop 이 실제로 행 하나 수준인지 (비율만 바꾸면 안 줄어든다).

    `_build_crop_box` 에는 여백 하한(vertical_pad_min_px)과 crop 높이 하한
    (min_crop_height)이 있어서, vertical_pad_ratio 만 낮추면 두 하한이 그대로
    살아나 crop 높이가 하나도 안 줄어든다. 이 테스트는 그 함정을 막는다.
    """
    from poc.workflow_3.rcs.workflow_select_tool import _tool_row_target
    from poc.workflow_3.vlm.ui_venus_mai_locator import _build_crop_box

    row_height = 24  # 오피스 list 행 대략 높이(논리 px)
    coarse_bbox = {"left": 40, "top": 500, "right": 130, "bottom": 500 + row_height}
    crop = _build_crop_box(coarse_bbox, 1920, 1080, _tool_row_target("MCDN01"))
    crop_height = crop["bottom"] - crop["top"]

    # 행 3개(= 위/아래 행이 통째로 들어옴) 미만이어야 한다.
    check(
        "tool_row_crop_under_3_rows",
        crop_height < row_height * 3,
        f"crop_height={crop_height} row_height={row_height}",
    )
    # 목표 행 자체는 온전히 들어 있어야 한다(너무 조여서 ID 가 잘리면 안 됨).
    check(
        "tool_row_crop_covers_target_row",
        crop["top"] <= coarse_bbox["top"] and crop["bottom"] >= coarse_bbox["bottom"],
        f"crop={crop} bbox={coarse_bbox}",
    )


def main() -> int:
    """모든 테스트를 돌리고 실패 수를 반환한다."""
    test_confirms_exact_id()
    test_confirms_with_surrounding_text()
    test_confusion_chars_still_confirm()
    test_neighbor_row_is_mismatch()
    test_pure_words_are_not_mismatch()
    test_empty_read_is_unreadable()
    test_empty_target_is_unreadable()
    test_short_token_not_mismatch()
    test_different_length_id_is_mismatch()
    test_confusable_pairs_from_office_list()
    test_target_wins_when_both_rows_in_strip()
    test_digit_leading_id_confirms()
    test_ip_and_counts_not_mismatch()
    test_policy_matrix()
    test_strip_box_geometry()
    test_tool_row_crop_is_row_tight()

    print("")
    print(f"[INFO] 통과 {len(PASSED)} / 실패 {len(FAILED)}")
    if FAILED:
        print(f"[ERROR] 실패 목록: {FAILED}")
    return len(FAILED)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
