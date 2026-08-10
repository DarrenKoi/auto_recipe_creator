"""abort_button locator 파싱/스케일 self-test (라이브 VLM 불필요).

가짜 client 가 캔드 JSON 을 돌려주게 해 bbox -> screen 좌표 변환 경로만 검증한다.

    uv run python poc/workflow_3e/test_abort_button.py
"""

import os

import numpy as np

from poc.workflow_3e.abort_button import (
    ABORT_LABEL_POLICY_LENIENT,
    ABORT_LABEL_POLICY_OFF,
    ABORT_LABEL_POLICY_STRICT,
    accepts_label,
    classify_button_tokens,
    load_abort_label_policy,
    locate_abort_button,
    locate_abort_confirm,
)


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, text):
        self._text = text

    def chat_with_image_b64(self, **kwargs):
        return _FakeResp(self._text)


def test_visible_center_relative_1000():
    """relative_1000 bbox 중심이 프레임 픽셀로 환산된다(1000x500 프레임)."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient(
        '{"abort_button_visible": true, "coord_system": "relative_1000", '
        '"abort_button_bbox": {"left": 400, "top": 800, "right": 600, "bottom": 900}, '
        '"confidence": 0.9}'
    )
    xy = locate_abort_button(frame_bgr=frame, client=client)
    # center rel (500, 850) -> px (500, 425)
    ok = xy is not None and abs(xy[0] - 500) <= 2 and abs(xy[1] - 425) <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] visible_center_relative_1000: xy={xy}")
    return ok


def test_not_visible_returns_none():
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient('{"abort_button_visible": false, "abort_button_bbox": null}')
    xy = locate_abort_button(frame_bgr=frame, client=client)
    ok = xy is None
    print(f"[{'PASS' if ok else 'FAIL'}] not_visible_returns_none: xy={xy}")
    return ok


def test_confirm_locator_shares_schema():
    """확인 다이얼로그 locator 도 같은 스키마 키로 파싱된다."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient(
        '{"abort_button_visible": true, "coord_system": "relative_1000", '
        '"abort_button_bbox": {"left": 100, "top": 100, "right": 300, "bottom": 200}}'
    )
    xy = locate_abort_confirm(frame_bgr=frame, client=client)
    # center rel (200, 150) -> px (200, 75)
    ok = xy is not None and abs(xy[0] - 200) <= 2 and abs(xy[1] - 75) <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] confirm_locator_shares_schema: xy={xy}")
    return ok


# ------------------------------------------------------------------
# 라벨 확인 게이트 - VLM 이 고른 점 주변을 OCR 로 읽어 'Stop 이 맞는지' 확인한다.
# (프로젝트 규칙: 좌표는 VLM 이, 확인은 OCR 이, 미확인이면 클릭 금지)
# ------------------------------------------------------------------

_ABORT_LABELS = ("Stop", "Abort", "중지", "정지")


def test_classify_confirmed_on_expected_label():
    v = classify_button_tokens(["Stop"], _ABORT_LABELS)
    ok = v.status == "confirmed" and v.matched_label == "Stop"
    print(f"[{'PASS' if ok else 'FAIL'}] classify_confirmed_on_expected_label: {v.status}/{v.matched_label}")
    return ok


def test_classify_confirmed_ignores_case_and_noise():
    """OCR 이 붙여온 주변 글자/대소문자는 허용된다(label_matches 의 포함 매칭)."""
    v = classify_button_tokens(["|STOP|"], _ABORT_LABELS)
    ok = v.status == "confirmed"
    print(f"[{'PASS' if ok else 'FAIL'}] classify_confirmed_ignores_case_and_noise: {v.status}")
    return ok


def test_classify_confirmed_on_korean_label():
    v = classify_button_tokens(["중지"], _ABORT_LABELS)
    ok = v.status == "confirmed" and v.matched_label == "중지"
    print(f"[{'PASS' if ok else 'FAIL'}] classify_confirmed_on_korean_label: {v.status}")
    return ok


def test_classify_mismatch_on_other_button():
    """읽히긴 했는데 다른 버튼 - 오클릭 직전 신호라 가장 중요한 케이스."""
    v = classify_button_tokens(["Pause"], _ABORT_LABELS)
    ok = v.status == "mismatch"
    print(f"[{'PASS' if ok else 'FAIL'}] classify_mismatch_on_other_button: {v.status}")
    return ok


def test_classify_unreadable_on_empty_tokens():
    v = classify_button_tokens([], _ABORT_LABELS)
    ok = v.status == "unreadable"
    print(f"[{'PASS' if ok else 'FAIL'}] classify_unreadable_on_empty_tokens: {v.status}")
    return ok


def test_strict_rejects_everything_but_confirmed():
    """기본 정책. abort 는 되돌릴 수 없어 '미확인 시 클릭 금지' 를 그대로 적용한다."""
    p = ABORT_LABEL_POLICY_STRICT
    ok = (
        accepts_label(classify_button_tokens(["Stop"], _ABORT_LABELS), p) is True
        and accepts_label(classify_button_tokens(["Pause"], _ABORT_LABELS), p) is False
        and accepts_label(classify_button_tokens([], _ABORT_LABELS), p) is False
    )
    print(f"[{'PASS' if ok else 'FAIL'}] strict_rejects_everything_but_confirmed")
    return ok


def test_lenient_rejects_only_mismatch():
    p = ABORT_LABEL_POLICY_LENIENT
    ok = (
        accepts_label(classify_button_tokens(["Stop"], _ABORT_LABELS), p) is True
        and accepts_label(classify_button_tokens(["Pause"], _ABORT_LABELS), p) is False
        and accepts_label(classify_button_tokens([], _ABORT_LABELS), p) is True
    )
    print(f"[{'PASS' if ok else 'FAIL'}] lenient_rejects_only_mismatch")
    return ok


def test_off_accepts_all_including_none():
    """게이트 비활성 - 검증 전 롤백 경로. verdict 가 아예 없어도 통과해야 한다."""
    p = ABORT_LABEL_POLICY_OFF
    ok = (
        accepts_label(classify_button_tokens(["Pause"], _ABORT_LABELS), p) is True
        and accepts_label(None, p) is True
    )
    print(f"[{'PASS' if ok else 'FAIL'}] off_accepts_all_including_none")
    return ok


def test_missing_verdict_rejected_under_strict():
    """OCR 자체가 못 돌면(client 생성 실패 등) verdict 는 None - strict 는 클릭 금지."""
    ok = accepts_label(None, ABORT_LABEL_POLICY_STRICT) is False
    print(f"[{'PASS' if ok else 'FAIL'}] missing_verdict_rejected_under_strict")
    return ok


def test_policy_defaults_to_strict_and_reads_env():
    prev = os.environ.pop("MEAS_FAIL_ABORT_LABEL_CONFIRM", None)
    try:
        default_ok = load_abort_label_policy() == ABORT_LABEL_POLICY_STRICT
        os.environ["MEAS_FAIL_ABORT_LABEL_CONFIRM"] = "  LENIENT  "
        env_ok = load_abort_label_policy() == ABORT_LABEL_POLICY_LENIENT
        os.environ["MEAS_FAIL_ABORT_LABEL_CONFIRM"] = "nonsense"
        bad_ok = load_abort_label_policy() == ABORT_LABEL_POLICY_STRICT
    finally:
        os.environ.pop("MEAS_FAIL_ABORT_LABEL_CONFIRM", None)
        if prev is not None:
            os.environ["MEAS_FAIL_ABORT_LABEL_CONFIRM"] = prev
    ok = default_ok and env_ok and bad_ok
    print(f"[{'PASS' if ok else 'FAIL'}] policy_defaults_to_strict_and_reads_env")
    return ok


def main():
    print("[INFO] abort_button self-test 시작")
    results = [
        test_visible_center_relative_1000(),
        test_not_visible_returns_none(),
        test_confirm_locator_shares_schema(),
        test_classify_confirmed_on_expected_label(),
        test_classify_confirmed_ignores_case_and_noise(),
        test_classify_confirmed_on_korean_label(),
        test_classify_mismatch_on_other_button(),
        test_classify_unreadable_on_empty_tokens(),
        test_strict_rejects_everything_but_confirmed(),
        test_lenient_rejects_only_mismatch(),
        test_off_accepts_all_including_none(),
        test_missing_verdict_rejected_under_strict(),
        test_policy_defaults_to_strict_and_reads_env(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
