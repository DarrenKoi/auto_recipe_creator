"""engineer_done 감지기 합성 테스트 (Mac/dev, RCS·VLM 불요).

`uv run python poc/workflow_3/monitor/test_engineer_done.py` 로 실행한다.
"""

import sys

from poc.workflow_3.config import Workflow3Settings


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


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_settings_defaults,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
