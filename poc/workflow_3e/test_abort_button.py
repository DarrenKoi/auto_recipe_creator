"""abort_button locator 파싱/스케일 self-test (라이브 VLM 불필요).

가짜 client 가 캔드 JSON 을 돌려주게 해 bbox -> screen 좌표 변환 경로만 검증한다.

    uv run python poc/workflow_3e/test_abort_button.py
"""

import os

import numpy as np

from poc.workflow_3e.abort_button import (
    ABORT_BUTTON_LABELS,
    ABORT_LABEL_POLICY_LENIENT,
    ABORT_LABEL_POLICY_OFF,
    ABORT_LABEL_POLICY_STRICT,
    ABORT_TARGET_ABORT,
    ABORT_TARGET_QUEUE,
    LABEL_CROP_HALF_HEIGHT_RATIO,
    LABEL_CROP_LEFT_RATIO,
    LABEL_CROP_RIGHT_RATIO,
    QUEUE_BUTTON_LABELS,
    accepts_label,
    button_label_for_target,
    button_target_config,
    classify_button_tokens,
    expected_labels_for_target,
    is_click_armed,
    is_rehearsal_target,
    load_abort_label_policy,
    load_abort_target,
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


# ------------------------------------------------------------------
# 클릭 대상 전환 - 검증 중에는 Stop/Abort 대신 인접한 Queue 버튼을 겨눈다.
# 같은 창 같은 영역이라 로케이트+라벨확인 파이프라인은 동일하게 검증되지만, 오클릭
# 대가가 '측정 중단' 이 아니라 '큐 화면 열림' 이다.
# ------------------------------------------------------------------


def test_target_defaults_to_queue_rehearsal():
    """검증 단계 기본값 - 실제 abort 는 명시적으로 골라야 한다."""
    prev = os.environ.pop("MEAS_FAIL_ABORT_TARGET", None)
    try:
        default_ok = load_abort_target() == ABORT_TARGET_QUEUE
        os.environ["MEAS_FAIL_ABORT_TARGET"] = "  ABORT  "
        env_ok = load_abort_target() == ABORT_TARGET_ABORT
        os.environ["MEAS_FAIL_ABORT_TARGET"] = "nonsense"
        bad_ok = load_abort_target() == ABORT_TARGET_QUEUE
    finally:
        os.environ.pop("MEAS_FAIL_ABORT_TARGET", None)
        if prev is not None:
            os.environ["MEAS_FAIL_ABORT_TARGET"] = prev
    ok = default_ok and env_ok and bad_ok
    print(f"[{'PASS' if ok else 'FAIL'}] target_defaults_to_queue_rehearsal")
    return ok


def test_only_abort_target_is_not_rehearsal():
    ok = (
        is_rehearsal_target(ABORT_TARGET_QUEUE) is True
        and is_rehearsal_target(ABORT_TARGET_ABORT) is False
    )
    print(f"[{'PASS' if ok else 'FAIL'}] only_abort_target_is_not_rehearsal")
    return ok


def test_expected_labels_follow_target():
    ok = (
        expected_labels_for_target(ABORT_TARGET_QUEUE) == QUEUE_BUTTON_LABELS
        and expected_labels_for_target(ABORT_TARGET_ABORT) == ABORT_BUTTON_LABELS
    )
    print(f"[{'PASS' if ok else 'FAIL'}] expected_labels_follow_target")
    return ok


def test_queue_label_confirms_and_stop_does_not():
    """Queue 를 겨눴는데 Stop 이 읽히면 mismatch - 인접 버튼이라 반드시 구분돼야 한다."""
    good = classify_button_tokens(["Queue"], QUEUE_BUTTON_LABELS)
    bad = classify_button_tokens(["Stop"], QUEUE_BUTTON_LABELS)
    ok = good.status == "confirmed" and bad.status == "mismatch"
    print(f"[{'PASS' if ok else 'FAIL'}] queue_label_confirms_and_stop_does_not: {good.status}/{bad.status}")
    return ok



def test_button_geometry_matches_proven_bench():
    """버튼 로케이트/라벨확인 기하는 오피스 벤치에서 acc=1.000 으로 입증된 값과 같아야 한다.

    bench_tool_window_reader 가 Stop/Queue/PM 을 그 값으로 1.000 을 냈다. 여기서 임의로
    다른 값을 쓰면 '입증된 설정' 이라는 근거가 사라진다. 한쪽만 바뀌면 이 테스트가 깨진다.
    """
    from poc.workflow_3.rcs import bench_tool_window_reader as bench

    crop_ok = (
        LABEL_CROP_LEFT_RATIO == bench.LABEL_LEFT_RATIO
        and LABEL_CROP_RIGHT_RATIO == bench.LABEL_RIGHT_RATIO
        and LABEL_CROP_HALF_HEIGHT_RATIO == bench.LABEL_HALF_HEIGHT_RATIO
    )
    mine = button_target_config("Queue")
    theirs = bench._button_target("Queue")
    geom_ok = (
        mine.left_pad_ratio == theirs.left_pad_ratio
        and mine.right_pad_ratio == theirs.right_pad_ratio
        and mine.vertical_pad_ratio == theirs.vertical_pad_ratio
        and mine.min_crop_width == theirs.min_crop_width
        and mine.min_crop_height == theirs.min_crop_height
        and mine.vertical_pad_min_px == theirs.vertical_pad_min_px
    )
    ok = crop_ok and geom_ok
    print(f"[{'PASS' if ok else 'FAIL'}] button_geometry_matches_proven_bench: crop={crop_ok} target={geom_ok}")
    return ok


def test_target_label_maps_to_button_name():
    """대상 -> 실제로 찾을 버튼 이름(로케이터 hint 와 라벨확인이 같은 것을 가리켜야)."""
    ok = (
        button_label_for_target(ABORT_TARGET_QUEUE) == "Queue"
        and button_label_for_target(ABORT_TARGET_ABORT) == "Stop"
    )
    print(f"[{'PASS' if ok else 'FAIL'}] target_label_maps_to_button_name")
    return ok


def test_click_armed_requires_real_abort_target():
    """시작 배너의 [ARMED] 판정. rehearsal 대상이면 무장일 수 없다.

    target=queue + dry_run=0 조합에서 [ARMED] 를 찍으면, 검증 중에 '지금 진짜 누르는
    모드구나' 로 오독한다. 배너는 실제 클릭 가능성과 일치해야 한다.
    """
    ok = (
        is_click_armed(enabled=True, dry_run=False, target=ABORT_TARGET_ABORT) is True
        and is_click_armed(enabled=True, dry_run=False, target=ABORT_TARGET_QUEUE) is False
        and is_click_armed(enabled=True, dry_run=True, target=ABORT_TARGET_ABORT) is False
        and is_click_armed(enabled=False, dry_run=False, target=ABORT_TARGET_ABORT) is False
    )
    print(f"[{'PASS' if ok else 'FAIL'}] click_armed_requires_real_abort_target")
    return ok



def main():
    print("[INFO] abort_button self-test 시작")
    results = [
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
        test_target_defaults_to_queue_rehearsal(),
        test_only_abort_target_is_not_rehearsal(),
        test_expected_labels_follow_target(),
        test_queue_label_confirms_and_stop_does_not(),
        test_click_armed_requires_real_abort_target(),
        test_button_geometry_matches_proven_bench(),
        test_target_label_maps_to_button_name(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
