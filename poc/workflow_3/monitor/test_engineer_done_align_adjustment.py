"""engineer_done_align_adjustment 감지기 합성 테스트 (Mac/dev, RCS·VLM 불요).

`uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py` 로 실행한다.
"""

import sys
import time as _time

import numpy as np
from PIL import Image

from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.monitor.cycle import _engineer_watch
from poc.workflow_3.monitor.engineer_done_align_adjustment import (
    EngineerDoneDetector,
    build_engineer_done_detector,
    extract_numerator,
    parse_point_1000,
    point_to_roi_ratios,
)
from poc.workflow_3.vlm.prompts.prompt_recipe_monitor_counter import (
    RECIPE_MONITOR_NUMERATOR_INSTRUCTION,
    build_recipe_monitor_counter_prompt,
)


# 기본 grounding 서비스 slug - Workflow3Settings 의 dataclass 기본값에서 읽는다.
_DEFAULT_VLM_SERVICE = Workflow3Settings.engineer_done_vlm_service


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
    ok &= _check("min_delta default 6", s.engineer_done_min_delta == 6)
    ok &= _check("change_min_px default 4", s.engineer_done_change_min_px == 4)
    ok &= _check("relocalize_after_miss default 3", s.engineer_done_relocalize_after_miss == 3)
    ok &= _check("roi_pad_x default 0.03", s.engineer_done_roi_pad_x == 0.03)
    ok &= _check("roi_pad_y default 0.02", s.engineer_done_roi_pad_y == 0.02)
    # 서비스 slug 는 A/B 로 왕복하는 값이라 리터럴 대신 dataclass 기본값과 비교한다
    # (모델을 바꿀 때마다 무관한 테스트가 따라 깨지면 전환 비용이 붙는다).
    ok &= _check(
        f"vlm_service default {_DEFAULT_VLM_SERVICE}",
        s.engineer_done_vlm_service == _DEFAULT_VLM_SERVICE,
    )
    ok &= _check("ocr_service default paddleocr", s.engineer_done_ocr_service == "paddleocr-vl-1.5")
    ok &= _check("reground_sec default 30.0", s.engineer_done_reground_sec == 30.0)
    return ok


def test_settings_env_load_path() -> bool:
    """load_workflow3_settings() env 경로로도 engineer_done 기본값이 동일하다."""
    import os

    from poc.workflow_3.config import load_workflow3_settings

    # 관련 env 가 비어 있는 상태를 보장한 뒤 로드한다 (있다면 임시 제거 후 복원).
    keys = [k for k in os.environ if k.startswith("ALIGN_FAIL_ENGINEER_DONE")]
    saved = {k: os.environ.pop(k) for k in keys}
    try:
        s = load_workflow3_settings()
    finally:
        os.environ.update(saved)
    ok = True
    ok &= _check("env path detect_enabled False", s.engineer_done_detect_enabled is False)
    ok &= _check("env path poll_sec 8.0", s.engineer_done_poll_sec == 8.0)
    ok &= _check("env path min_delta 6", s.engineer_done_min_delta == 6)
    ok &= _check(
        "env path services",
        s.engineer_done_vlm_service == _DEFAULT_VLM_SERVICE
        and s.engineer_done_ocr_service == "paddleocr-vl-1.5",
    )
    return ok


def test_counter_prompt() -> bool:
    """grounding 모델의 공식 단일요소 형식([x,y], [-1,-1] 거부)을 따른다."""
    system_message, user_text = build_recipe_monitor_counter_prompt()
    ok = True
    ok &= _check("system empty (official format)", system_message == "")
    ok &= _check("instruction embedded", RECIPE_MONITOR_NUMERATOR_INSTRUCTION in user_text)
    ok &= _check("point format requested", "[x,y]" in user_text)
    ok &= _check("refusal format requested", "[-1,-1]" in user_text)
    return ok


def test_parse_point_1000() -> bool:
    """grounding [x,y] 응답 파싱 — 거부/범위밖/없음은 None."""
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


def _frame(counter_value: int) -> Image.Image:
    """카운터 영역 픽셀이 counter_value 에 따라 달라지는 합성 tool 창 프레임.

    창 400x200. 카운터 셀은 x 190..230, y 100..120 부근 — grounding 점
    (525, 550) + pad (0.05, 0.05) 의 ROI 와 일치시킨다.
    """
    arr = np.zeros((200, 400, 3), dtype=np.uint8)
    arr[100:120, 190:190 + 4 * (counter_value + 1)] = 255
    return Image.fromarray(arr)


class _SeqCapture:
    """호출마다 프레임 시퀀스를 차례로 반환한다 (끝나면 마지막 프레임 반복)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def __call__(self):
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame


class _CountingFn:
    """반환값 시퀀스를 차례로 내놓으며 호출 횟수를 기록한다."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, *args):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def _settings(**overrides):
    """테스트용 설정 — ROI pad 를 합성 프레임 카운터 셀에 맞춘다.

    reground_sec=0.0: 테스트에선 grounding 거부 후 다음 호출에 바로 재시도.
    """
    base = dict(
        engineer_done_detect_enabled=True,
        engineer_done_roi_pad_x=0.05,
        engineer_done_roi_pad_y=0.05,
        engineer_done_min_delta=2,
        engineer_done_relocalize_after_miss=3,
        engineer_done_reground_sec=0.0,
    )
    base.update(overrides)
    return Workflow3Settings(**base)


def test_detector_static_no_ocr() -> bool:
    """정적 프레임(첫 샘플 포함)에서는 OCR 을 호출하지 않는다."""
    # grounding 1회 캡처 + 정적 crop 3회.
    capture = _SeqCapture([_frame(1), _frame(1), _frame(1), _frame(1)])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("all False on static", results == [False, False, False])
    ok &= _check("ground called once", ground.calls == 1)
    ok &= _check("ocr never called", ocr.calls == 0)
    return ok


def test_detector_two_read_confirm() -> bool:
    """변화 + OCR 2 -> 3: 첫 읽기는 확인 대기(False), 두 번째에 done."""
    capture = _SeqCapture([
        _frame(1),            # grounding 캡처
        _frame(1),            # baseline (첫 샘플, OCR 안 함)
        _frame(2),            # 변화 1 -> OCR '2' (last 없음 -> 확인 대기)
        _frame(3),            # 변화 2 -> OCR '3' (3>=2, 3>=2 -> done)
    ])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350", "3/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("baseline False", results[0] is False)
    ok &= _check("first read waits", results[1] is False)
    ok &= _check("second read done", results[2] is True)
    ok &= _check("ocr called twice", ocr.calls == 2)
    return ok


def test_detector_below_min_not_done() -> bool:
    """N < min_count 면 변화가 있어도 done 아님."""
    capture = _SeqCapture([_frame(0), _frame(0), _frame(1), _frame(2)])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["1/350", "1/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("below min stays False", results == [False, False, False])
    ok &= _check("ocr called for each change", ocr.calls == 2)
    return ok


def test_detector_ground_refusal_retries() -> bool:
    """grounding 거부(None) -> False 지만, 재정렬 중 카운터 blank 일 수 있어 재시도한다."""
    capture = _SeqCapture([_frame(1), _frame(2), _frame(3)])
    ground = _CountingFn([None, None, None])
    ocr = _CountingFn(["2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("refusal -> all False", results == [False, False, False])
    ok &= _check("ground retried each call (reground_sec=0)", ground.calls == 3)
    ok &= _check("ocr never called", ocr.calls == 0)
    return ok


def test_detector_reground_throttle() -> bool:
    """reground_sec 가 크면 거부 후 재시도가 throttle 된다 (VLM 호출 폭주 방지)."""
    capture = _SeqCapture([_frame(1), _frame(2), _frame(3)])
    ground = _CountingFn([None])
    ocr = _CountingFn(["2/350"])
    settings = _settings(engineer_done_reground_sec=3600.0)
    detector = EngineerDoneDetector(None, settings, capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("throttled refusal -> all False", results == [False, False, False])
    ok &= _check("ground called once (throttled)", ground.calls == 1)
    return ok


def test_detector_ground_blank_then_found() -> bool:
    """재정렬 중 blank(거부 2회) -> 측정 시작으로 카운터 등장 -> 정상 done 경로.

    오피스 관찰: re-align 진행 중에는 N/M 칸이 빈칸이라 VLM 이 거부한다.
    측정이 시작되면 숫자가 나타나므로 grounding 재시도가 성공해야 한다.
    """
    capture = _SeqCapture([
        _frame(1),            # grounding 시도 1 (blank 가정 -> 거부)
        _frame(1),            # grounding 시도 2 (거부)
        _frame(1),            # grounding 시도 3 (성공)
        _frame(1),            # baseline crop (첫 샘플)
        _frame(2),            # 변화 1 -> OCR '2'
        _frame(3),            # 변화 2 -> OCR '3' -> done
    ])
    ground = _CountingFn([None, None, (525, 550)])
    ocr = _CountingFn(["2/350", "3/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    results = [detector(), detector(), detector(), detector(), detector()]
    ok = True
    ok &= _check("blank phase all False", results[:3] == [False, False, False])
    ok &= _check("first read waits", results[3] is False)
    ok &= _check("second read done", results[4] is True)
    ok &= _check("ground called 3 times", ground.calls == 3)
    return ok


def test_detector_relocalize_after_miss() -> bool:
    """변화 후 OCR 연속 미검출이 임계에 닿으면 1회 재grounding 한다."""
    # 매 호출 프레임이 달라(계속 변화) OCR 이 그때마다 불리지만 빈 텍스트.
    capture = _SeqCapture([
        _frame(0),                       # grounding 1 캡처
        _frame(0), _frame(1), _frame(2), _frame(3),  # baseline + 변화 3회 (miss 3)
        _frame(4),                       # 재grounding 캡처
        _frame(4), _frame(5),            # 새 baseline + 변화
    ])
    ground = _CountingFn([(525, 550), (525, 550)])
    ocr = _CountingFn(["", "", "", "2/350"])
    detector = EngineerDoneDetector(None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr)
    for _ in range(6):
        detector()
    return _check("ground called twice (relocalize)", ground.calls == 2)


def test_tool_label_from_title() -> bool:
    """창 제목 -> debug 폴더용 tool 라벨 추출/정제."""
    from poc.workflow_3.monitor.engineer_done_align_adjustment import _tool_label_from_title

    ok = True
    ok &= _check(
        "title prefix stripped",
        _tool_label_from_title("Remote Monitoring System - MCD630") == "MCD630",
    )
    ok &= _check(
        "special chars sanitized",
        _tool_label_from_title("Remote Monitoring System - MC D/630 #2") == "MC_D_630__2",
    )
    ok &= _check("empty -> empty", _tool_label_from_title("") == "")
    return ok


def test_builder_gates() -> bool:
    """설정 off / tool_window 없음 -> None (고정 timeout 폴백)."""
    ok = True
    off = Workflow3Settings(engineer_done_detect_enabled=False)
    ok &= _check("disabled -> None", build_engineer_done_detector(object(), off) is None)
    on = _settings()
    ok &= _check("no window -> None", build_engineer_done_detector(None, on) is None)
    return ok


class _FakeRecording:
    """is_alive 만 흉내내는 fake (n번째 확인 후 사망 옵션)."""

    def __init__(self, alive_checks: int = 10**6):
        self.alive_checks = alive_checks
        self.checks = 0

    def is_alive(self) -> bool:
        self.checks += 1
        return self.checks <= self.alive_checks


def test_watch_early_exit_on_done() -> bool:
    """detector True -> cap 보다 훨씬 일찍 종료."""
    detector = _CountingFn([False, True])
    started = _time.time()
    _engineer_watch(_FakeRecording(), 60.0, done_detector=detector, poll_sec=0.0)
    elapsed = _time.time() - started
    ok = True
    ok &= _check("early exit well under cap", elapsed < 30.0)
    ok &= _check("detector called twice", detector.calls == 2)
    return ok


def test_watch_detector_exception_safe() -> bool:
    """detector 예외 -> 삼키고 recording 사망/cap 으로 정상 종료."""

    def boom():
        raise RuntimeError("detector crash")

    _engineer_watch(_FakeRecording(alive_checks=2), 60.0, done_detector=boom, poll_sec=0.0)
    return _check("watch survived detector exception", True)


def test_watch_no_detector_unchanged() -> bool:
    """detector 없음 -> 기존 동작(recording 사망 시 종료)."""
    recording = _FakeRecording(alive_checks=3)
    _engineer_watch(recording, 60.0, done_detector=None, poll_sec=0.0)
    return _check("exits on recording death", recording.checks >= 3)


def test_settings_use_delta_and_streak_not_absolute_count() -> bool:
    """절대값 기준(min_count)은 제거됐다.

    잔존 카운터 오탐의 근원이었다 - 이전 런의 350/350 이 떠 있으면 즉시 조건을 만족했다.
    """
    settings = load_workflow3_settings()
    ok = (
        settings.engineer_done_ok_streak == 6
        and settings.engineer_done_min_delta == 6
        and not hasattr(settings, "engineer_done_min_count")
    )
    print(f"[{'PASS' if ok else 'FAIL'}] settings_use_delta_and_streak_not_absolute_count")
    return ok


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_settings_defaults,
        test_settings_env_load_path,
        test_settings_use_delta_and_streak_not_absolute_count,
        test_counter_prompt,
        test_parse_point_1000,
        test_point_to_roi_ratios,
        test_extract_numerator,
        test_detector_static_no_ocr,
        test_detector_two_read_confirm,
        test_detector_below_min_not_done,
        test_detector_ground_refusal_retries,
        test_detector_reground_throttle,
        test_detector_ground_blank_then_found,
        test_detector_relocalize_after_miss,
        test_tool_label_from_title,
        test_builder_gates,
        test_watch_early_exit_on_done,
        test_watch_detector_exception_safe,
        test_watch_no_detector_unchanged,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
