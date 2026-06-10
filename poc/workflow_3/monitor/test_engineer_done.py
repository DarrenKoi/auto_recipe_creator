"""engineer_done 감지기 합성 테스트 (Mac/dev, RCS·VLM 불요).

`uv run python poc/workflow_3/monitor/test_engineer_done.py` 로 실행한다.
"""

import sys

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.monitor.engineer_done import (
    extract_numerator,
    parse_point_1000,
    point_to_roi_ratios,
)
from poc.workflow_3.vlm.prompts.prompt_recipe_monitor_counter import (
    RECIPE_MONITOR_NUMERATOR_INSTRUCTION,
    build_recipe_monitor_counter_prompt,
)


def _check(name: str, condition: bool) -> bool:
    """단건 검증 결과를 출력하고 통과 여부를 반환한다."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def test_settings_defaults() -> bool:
    """engineer_done_* 필드가 기본값과 함께 존재한다 (기본 비활성)."""
    s = Workflow3Settings()
    ok = True
    ok &= _check("detect_enabled default False", s.engineer_done_detect_enabled is False)
    ok &= _check("poll_sec default 8.0", s.engineer_done_poll_sec == 8.0)
    ok &= _check("min_count default 2", s.engineer_done_min_count == 2)
    ok &= _check("change_min_px default 4", s.engineer_done_change_min_px == 4)
    ok &= _check("relocalize_after_miss default 3", s.engineer_done_relocalize_after_miss == 3)
    ok &= _check("roi_pad_x default 0.03", s.engineer_done_roi_pad_x == 0.03)
    ok &= _check("roi_pad_y default 0.02", s.engineer_done_roi_pad_y == 0.02)
    ok &= _check("vlm_service default ui-venus", s.engineer_done_vlm_service == "ui-venus-1.5-8b")
    ok &= _check("ocr_service default paddleocr", s.engineer_done_ocr_service == "paddleocr-vl-1.5")
    return ok


def test_counter_prompt() -> bool:
    """ui-venus 공식 단일요소 형식([x,y], [-1,-1] 거부)을 따른다."""
    system_message, user_text = build_recipe_monitor_counter_prompt()
    ok = True
    ok &= _check("system empty (official format)", system_message == "")
    ok &= _check("instruction embedded", RECIPE_MONITOR_NUMERATOR_INSTRUCTION in user_text)
    ok &= _check("point format requested", "[x,y]" in user_text)
    ok &= _check("refusal format requested", "[-1,-1]" in user_text)
    return ok


def test_parse_point_1000() -> bool:
    """ui-venus [x,y] 응답 파싱 — 거부/범위밖/없음은 None."""
    ok = True
    ok &= _check("valid point", parse_point_1000("[525, 550]") == (525, 550))
    ok &= _check("point in prose", parse_point_1000("the point is [10,20].") == (10, 20))
    ok &= _check("refusal -> None", parse_point_1000("[-1,-1]") is None)
    ok &= _check("out of range -> None", parse_point_1000("[1500, 200]") is None)
    ok &= _check("no point -> None", parse_point_1000("cannot find it") is None)
    ok &= _check("empty -> None", parse_point_1000("") is None)
    return ok


def test_point_to_roi_ratios() -> bool:
    """grounding 점(0-1000) -> 상대비율 ROI 확장 + 경계 clamp."""
    ok = True
    roi = point_to_roi_ratios(500, 500, 0.05, 0.05)
    ok &= _check("center roi", roi is not None and all(abs(a - b) < 1e-9 for a, b in zip(roi, (0.45, 0.45, 0.55, 0.55))))
    roi = point_to_roi_ratios(0, 0, 0.05, 0.05)
    ok &= _check("corner clamped", roi is not None and roi[0] == 0.0 and roi[1] == 0.0)
    ok &= _check("corner still has span", roi is not None and roi[2] > 0.0 and roi[3] > 0.0)
    return ok


def test_extract_numerator() -> bool:
    """OCR 텍스트에서 분자 정수 추출 (첫 연속 숫자열)."""
    ok = True
    ok &= _check("'2/350' -> 2", extract_numerator("2/350") == 2)
    ok &= _check("' 13 / 350 ' -> 13", extract_numerator(" 13 / 350 ") == 13)
    ok &= _check("bare '7' -> 7", extract_numerator("7") == 7)
    ok &= _check("no digits -> None", extract_numerator("abc") is None)
    ok &= _check("empty -> None", extract_numerator("") is None)
    return ok


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_settings_defaults,
        test_counter_prompt,
        test_parse_point_1000,
        test_point_to_roi_ratios,
        test_extract_numerator,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
