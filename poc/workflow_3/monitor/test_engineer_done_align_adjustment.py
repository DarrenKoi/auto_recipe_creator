"""engineer_done_align_adjustment 감지기 합성 테스트 (Mac/dev, RCS·VLM 불요).

`uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py` 로 실행한다.
"""

import io
import sys
import time as _time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from poc.workflow_3 import util as w3util
from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.monitor.cycle import _engineer_watch
from poc.workflow_3.monitor.engineer_done_align_adjustment import (
    EngineerDoneDetector,
    NumeratorObservation,
    _make_assist_fn,
    build_engineer_done_detector,
    extract_numerator,
    parse_point_1000,
    point_to_roi_ratios,
)
from poc.workflow_3.sem_monitor import assist_score as asc
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
    ok &= _check(
        "assist_unusable_after default 3",
        s.engineer_done_assist_unusable_after == 3,
    )
    ok &= _check(
        "numerator_increase_reads default 3",
        s.engineer_done_numerator_increase_reads == 3,
    )
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
    """load_workflow3_settings() 가 priority 설정 env 이름을 읽는다."""
    import os

    from poc.workflow_3.config import load_workflow3_settings

    # 관련 env 가 비어 있는 상태를 보장한 뒤 로드한다 (있다면 임시 제거 후 복원).
    keys = [k for k in os.environ if k.startswith("ALIGN_FAIL_ENGINEER_DONE")]
    saved = {k: os.environ.pop(k) for k in keys}
    try:
        os.environ["ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER"] = "5"
        os.environ["ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS"] = "4"
        s = load_workflow3_settings()
    finally:
        for key in [k for k in os.environ if k.startswith("ALIGN_FAIL_ENGINEER_DONE")]:
            os.environ.pop(key)
        os.environ.update(saved)
    ok = True
    ok &= _check("env path detect_enabled False", s.engineer_done_detect_enabled is False)
    ok &= _check("env path poll_sec 8.0", s.engineer_done_poll_sec == 8.0)
    ok &= _check(
        "env path priority settings",
        s.engineer_done_assist_unusable_after == 5
        and s.engineer_done_numerator_increase_reads == 4,
    )
    ok &= _check(
        "env path services",
        s.engineer_done_vlm_service == _DEFAULT_VLM_SERVICE
        and s.engineer_done_ocr_service == "paddleocr-vl-1.5",
    )
    return ok


def test_settings_reject_non_positive_priority_thresholds() -> bool:
    """완료 임계값 3종은 env 에서 0/음수가 들어오면 설정 로드를 거부한다."""
    import os

    cases = [
        ("ALIGN_FAIL_ENGINEER_DONE_MIN_OK_ROWS", "0"),
        ("ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER", "-1"),
        ("ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS", "0"),
    ]
    keys = [name for name, _value in cases]
    saved = {name: os.environ.get(name) for name in keys}
    ok = True
    try:
        for name, value in cases:
            for key in keys:
                os.environ.pop(key, None)
            os.environ[name] = value
            try:
                load_workflow3_settings()
            except ValueError as exc:
                ok &= name in str(exc)
            else:
                ok = False
    finally:
        for name in keys:
            os.environ.pop(name, None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    _check("settings reject non-positive priority thresholds", ok)
    assert ok


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
        engineer_done_min_ok_rows=2,
        engineer_done_assist_unusable_after=1,
        engineer_done_numerator_increase_reads=2,
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
    """Assist unusable 상태에서 변화 + OCR 2 -> 4 두 표본은 fallback done."""
    capture = _SeqCapture([
        _frame(1),            # grounding + CV baseline (OCR 안 함)
        _frame(2),            # 변화 1 -> OCR '2'
        _frame(3),            # 변화 2 -> OCR '4' (두 번째 엄격 증가 표본 -> done)
    ])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350", "4/350"])
    detector = EngineerDoneDetector(
        None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr,
    )
    results = [detector(), detector(), detector()]
    ok = True
    ok &= _check("baseline False", results[0] is False)
    ok &= _check("first read waits", results[1] is False)
    ok &= _check("second read done", results[2] is True)
    ok &= _check("ocr called twice", ocr.calls == 2)
    return ok


def test_detector_below_min_not_done() -> bool:
    """같은 OCR 값은 엄격 증가 시퀀스를 만들지 못한다."""
    capture = _SeqCapture([_frame(0), _frame(1), _frame(2)])
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
    측정이 시작되면 숫자가 나타나므로 grounding 재시도가 성공하고 두 번의 엄격
    증가 OCR 표본으로 fallback 완료해야 한다.
    """
    capture = _SeqCapture([
        _frame(1),            # grounding 시도 1 (blank 가정 -> 거부)
        _frame(1),            # grounding 시도 2 (거부)
        _frame(1),            # grounding 시도 3 성공 + CV baseline
        _frame(2),            # 변화 1 -> OCR '2'
        _frame(3),            # 변화 2 -> OCR '4' (fallback done)
    ])
    ground = _CountingFn([None, None, (525, 550)])
    ocr = _CountingFn(["2/350", "4/350"])
    detector = EngineerDoneDetector(
        None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr,
    )
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

    def __init__(self, alive_checks: int = 10**6, stop_reason: str = ""):
        self.alive_checks = alive_checks
        self.stop_reason = stop_reason
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


def test_watch_logs_manual_completion_only_for_window_gone():
    """창 닫힘 종료만 엔지니어의 명시적 완료로 기록한다."""
    recording = _FakeRecording(alive_checks=0, stop_reason="window_gone")
    output = io.StringIO()
    with redirect_stdout(output):
        _engineer_watch(recording, 60.0, poll_sec=0.0)
    assert "엔지니어가 Remote Monitoring 창을 닫음" in output.getvalue()
    return True


def test_watch_does_not_call_max_sec_manual_completion():
    """녹화 상한 종료를 수동 창 닫힘으로 오인하지 않는다."""
    recording = _FakeRecording(alive_checks=0, stop_reason="max_sec")
    output = io.StringIO()
    with redirect_stdout(output):
        _engineer_watch(recording, 60.0, poll_sec=0.0)
    assert "엔지니어가 Remote Monitoring 창을 닫음" not in output.getvalue()
    assert "max_sec" in output.getvalue()
    return True


def test_settings_use_priority_signals() -> bool:
    """Assist 우선과 엄격 증가 fallback 설정만 노출한다."""
    settings = load_workflow3_settings()
    ok = (
        settings.engineer_done_min_ok_rows == 5
        and settings.engineer_done_assist_unusable_after == 3
        and settings.engineer_done_numerator_increase_reads == 3
    )
    print(f"[{'PASS' if ok else 'FAIL'}] settings_use_priority_signals")
    return ok


def _assist_count(ok_rows, *, red=False, status="usable"):
    """행 수/red 만 담은 신규 관측 (격자 없는 CV counting 판정용)."""
    return asc.AssistObservation(
        status=status, ok_row_count=ok_rows, has_red=red, reason="ok",
    )


def _count_detector(observations, *, min_ok_rows=5):
    return EngineerDoneDetector(
        None,
        _settings(engineer_done_min_ok_rows=min_ok_rows),
        capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
        assist_fn=_CountingFn(observations),
        numerator_fn=_CountingFn([
            NumeratorObservation(sampled=False, reason="no_change")
        ] * len(observations)),
    )


def test_done_when_enough_ok_rows_and_no_red():
    detector = _count_detector([_assist_count(5)])
    ok = detector() is True
    print(f"[{'PASS' if ok else 'FAIL'}] done_when_enough_ok_rows_and_no_red")
    return ok


def test_red_blocks_done_even_with_enough_rows():
    """보수적 정책(2026-08-13): 표는 새 측정마다 초기화되므로 red 는 이번 사이클의 실패다."""
    detector = _count_detector([_assist_count(7, red=True)])
    ok = detector() is False
    print(f"[{'PASS' if ok else 'FAIL'}] red_blocks_done_even_with_enough_rows")
    return ok


def test_not_done_below_min_ok_rows():
    detector = _count_detector([_assist_count(4)])
    ok = detector() is False
    print(f"[{'PASS' if ok else 'FAIL'}] not_done_below_min_ok_rows")
    return ok


def _assist_unusable():
    return asc.AssistObservation(status="unusable", reason="panel_unavailable")


def _priority_detector(*, assist, numerators):
    observations = [
        NumeratorObservation(
            sampled=True,
            value=value,
            reason="read" if value is not None else "ocr_miss",
        )
        for value in numerators
    ] or [NumeratorObservation(sampled=False, reason="no_change")]
    return EngineerDoneDetector(
        None,
        _settings(
            engineer_done_min_ok_rows=6,
            engineer_done_assist_unusable_after=3,
            engineer_done_numerator_increase_reads=3,
        ),
        capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
        assist_fn=assist,
        numerator_fn=_CountingFn(observations),
    )


def _run_numerator_sequence(values):
    assist = _CountingFn([_assist_unusable()] * len(values))
    detector = _priority_detector(assist=assist, numerators=values)
    return [detector() for _ in values][-1]


def test_numerator_fallback_requires_three_unusable_assist_observations():
    assist = _CountingFn([
        _assist_unusable(),
        _assist_unusable(),
        _assist_unusable(),
    ])
    detector = _priority_detector(assist=assist, numerators=[10, 11, 12])
    assert [detector(), detector(), detector()] == [False, False, True]


def test_invalid_numerator_sequences_do_not_finish():
    for values in ([10, 10, 11], [10, 12, None], [10, 9, 10]):
        assert not _run_numerator_sequence(values)


def test_ocr_miss_clears_prior_numerator_evidence():
    assist = _CountingFn([_assist_unusable()] * 5)
    detector = _priority_detector(
        assist=assist,
        numerators=[10, 11, None, 12, 13],
    )

    assert [detector() for _ in range(5)] == [False] * 5
    assert detector._numerator_sequence == [12, 13]


def test_unchanged_crop_preserves_partial_numerator_sequence():
    capture = _SeqCapture([_frame(0), _frame(1), _frame(1), _frame(2)])
    ocr = _CountingFn(["10/350", "11/350"])
    detector = EngineerDoneDetector(
        None,
        _settings(
            engineer_done_assist_unusable_after=99,
            engineer_done_numerator_increase_reads=3,
        ),
        capture_fn=capture,
        ground_fn=lambda _image: (525, 550),
        ocr_fn=ocr,
    )

    results = []
    sequences = []
    for _ in range(4):
        results.append(detector())
        sequences.append(list(detector._numerator_sequence))

    assert results == [False] * 4
    assert sequences == [[], [10], [10], [10, 11]]
    assert ocr.calls == 2


def test_usable_assist_resets_unusable_streak():
    assist = _CountingFn([
        _assist_unusable(),
        _assist_unusable(),
        _assist_count(1),
    ])
    detector = _priority_detector(assist=assist, numerators=[10, 11, 12])

    assert [detector(), detector(), detector()] == [False, False, False]
    assert detector._assist_unusable_streak == 0


def test_detector_captures_one_frame_per_poll():
    capture = _SeqCapture([
        Image.new("RGB", (400, 200), (0, 0, 0)),
        Image.new("RGB", (400, 200), (1, 1, 1)),
    ])
    detector = EngineerDoneDetector(
        None,
        _settings(),
        capture_fn=capture,
        assist_fn=_CountingFn([_assist_unusable(), _assist_unusable()]),
        numerator_fn=_CountingFn([
            NumeratorObservation(sampled=False, reason="no_change"),
        ]),
    )

    detector()
    detector()

    assert capture.calls == 2


def test_assist_primary_finishes_before_numerator_evaluation():
    numerator = _CountingFn([
        NumeratorObservation(sampled=True, value=10, reason="read"),
    ])
    detector = EngineerDoneDetector(
        None,
        _settings(engineer_done_min_ok_rows=6),
        capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
        assist_fn=_CountingFn([_assist_count(1), _assist_count(6)]),
        numerator_fn=numerator,
    )

    assert [detector(), detector()] == [False, True]
    assert numerator.calls == 1


def test_builder_keeps_assist_when_numerator_clients_fail():
    import poc.workflow_3.monitor.engineer_done_align_adjustment as module

    saved = (module._make_ground_fn, module._make_ocr_fn, module._make_assist_fn)
    try:
        module._make_ground_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("numerator grounding unavailable")
        )
        module._make_ocr_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("numerator OCR unavailable")
        )
        module._make_assist_fn = lambda *args, **kwargs: (
            lambda image: _assist_count(1)
        )
        detector = module.build_engineer_done_detector(object(), _settings())
    finally:
        module._make_ground_fn, module._make_ocr_fn, module._make_assist_fn = saved
    assert detector is not None


def test_detector_fails_closed_for_each_non_positive_priority_threshold():
    """설정 객체 검증을 우회한 잘못된 값도 detector 완료를 허용하지 않는다."""
    for name, value in (
        ("engineer_done_min_ok_rows", 0),
        ("engineer_done_assist_unusable_after", -1),
        ("engineer_done_numerator_increase_reads", 0),
    ):
        values = vars(_settings()).copy()
        values[name] = value
        invalid_settings = SimpleNamespace(**values)
        detector = EngineerDoneDetector(
            None,
            invalid_settings,
            capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
            assist_fn=_CountingFn([
                _assist_count(6),
                _assist_count(6),
                _assist_unusable(),
            ]),
            numerator_fn=_CountingFn([
                NumeratorObservation(sampled=True, value=10, reason="read"),
                NumeratorObservation(sampled=True, value=11, reason="read"),
                NumeratorObservation(sampled=True, value=12, reason="read"),
            ]),
        )

        assert [detector(), detector(), detector()] == [False, False, False]


def test_numerator_decisions_are_saved_for_every_evaluated_observation():
    """no-change/read/equal/miss도 run 폴더에 poll별 결정 JSON으로 남는다."""
    import json
    import tempfile

    observations = [
        NumeratorObservation(sampled=False, reason="no_change"),
        NumeratorObservation(sampled=True, value=10, reason="read"),
        NumeratorObservation(sampled=True, value=10, reason="read"),
        NumeratorObservation(sampled=True, value=None, reason="ocr_miss"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        detector = EngineerDoneDetector(
            None,
            _settings(engineer_done_assist_unusable_after=99),
            capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
            assist_fn=_CountingFn([_assist_unusable()] * len(observations)),
            numerator_fn=_CountingFn(observations),
            debug_dir=debug_dir,
        )
        assert [detector() for _ in observations] == [False] * len(observations)
        payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(debug_dir.glob("numerator_decision_*.json"))
        ]

    assert [payload["poll"] for payload in payloads] == [1, 2, 3, 4]
    assert [payload["reading"] for payload in payloads] == [
        "no_change", "read", "read", "ocr_miss",
    ]
    assert [payload["value"] for payload in payloads] == [None, 10, 10, None]
    assert [payload["sequence"] for payload in payloads] == [[], [10], [10], []]
    assert [payload["reset_reason"] for payload in payloads] == [
        None, None, "equal_or_decrease", "ocr_miss",
    ]


def test_numerator_decision_records_reground_reset():
    """새 ROI grounding은 기존 증가 sequence를 지우고 그 이유를 JSON에 남긴다."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        detector = EngineerDoneDetector(
            None,
            _settings(engineer_done_assist_unusable_after=99),
            capture_fn=lambda: _frame(1),
            ground_fn=lambda _image: (525, 550),
            ocr_fn=lambda _crop: "10/350",
            assist_fn=_CountingFn([_assist_unusable()]),
            debug_dir=debug_dir,
        )
        detector._numerator_sequence = [8, 9]

        assert detector() is False
        decision_path = debug_dir / "numerator_decision_001.json"
        payload = json.loads(decision_path.read_text(encoding="utf-8"))

    assert payload["reading"] == "no_change"
    assert payload["sequence"] == []
    assert payload["reset_reason"] == "reground"


def test_leftover_counter_does_not_fire():
    """잔존 분자 값이 반복되면 fallback 완료로 오인하지 않는다."""
    assist = _CountingFn([_assist_unusable()] * 3)
    detector = _priority_detector(assist=assist, numerators=[350, 350, 350])
    results = [detector() for _ in range(3)]
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] leftover_counter_does_not_fire: {results}")
    return ok


class _PanelFnHarness:
    """assist_score 의 패널 로케이트/판독을 갈아끼운다 (원복 보장).

    _make_assist_fn 이 함수 안에서 import 하므로 원본 모듈(asc) 속성을 패치해야
    가로챌 수 있다 - 클로저가 이미 잡은 참조를 바꾸는 게 아니다.
    """

    def __init__(self, *, locate_results, read_results=None):
        self._locate_results = list(locate_results)
        self._read_results = list(read_results or [])
        self.locate_calls = 0
        self._saved = {}

    def _locate(self, *args, **kwargs):
        self.locate_calls += 1
        value = self._locate_results.pop(0) if self._locate_results else None
        if isinstance(value, Exception):
            raise value
        return value

    def _read(self, _panel):
        value = self._read_results.pop(0) if self._read_results else _assist_count(0)
        if isinstance(value, Exception):
            raise value
        return value

    def __enter__(self):
        for name, value in (("locate_assist_panel", self._locate),
                            ("read_assist_state", self._read)):
            self._saved[name] = getattr(asc, name)
            setattr(asc, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(asc, name, value)
        return False


_PANEL_BOX = {"left": 0, "top": 0, "right": 20, "bottom": 20}


def _blank_frame():
    return Image.new("RGB", (20, 20), (240, 240, 240))


def test_assist_fn_exception_returns_unusable() -> bool:
    """Assist 판독 예외는 unusable 관측값으로 닫히고 호출자에게 전파되지 않는다."""
    with _PanelFnHarness(locate_results=[_PANEL_BOX],
                         read_results=[RuntimeError("read boom")]):
        assist_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        observation = assist_fn(_blank_frame())
    ok = observation.status == "unusable" and observation.reason == "exception"
    print(f"[{'PASS' if ok else 'FAIL'}] assist_fn_exception_returns_unusable")
    return ok


def test_assist_fn_locates_panel_only_once() -> bool:
    """패널 박스는 watch 당 1회만 잡는다 - 이후 폴링에 VLM 왕복이 없어야 한다."""
    with _PanelFnHarness(locate_results=[_PANEL_BOX],
                         read_results=[_assist_count(3), _assist_count(4)]) as harness:
        assist_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        first = assist_fn(_blank_frame())
        second = assist_fn(_blank_frame())
    ok = (
        harness.locate_calls == 1
        and first.ok_row_count == 3
        and second.ok_row_count == 4
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] assist_fn_locates_panel_only_once: "
        f"locate_calls={harness.locate_calls}"
    )
    return ok


def test_assist_fn_throttles_locate_retry_after_failure() -> bool:
    """로케이트 실패 후에는 reground_sec 안에 다시 VLM 을 부르지 않는다.

    throttle 이 없으면 실패할 때마다 15s timeout VLM 왕복이 매 폴링 반복돼 watch
    루프 전체가 막힌다.
    """
    with _PanelFnHarness(locate_results=[None, _PANEL_BOX]) as harness:
        assist_fn = _make_assist_fn(
            object(), _settings(engineer_done_reground_sec=300.0), debug_dir=None
        )
        first = assist_fn(_blank_frame())
        second = assist_fn(_blank_frame())
    ok = (
        harness.locate_calls == 1
        and first.reason == "panel_unavailable"
        and second.reason == "locate_throttled"
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] assist_fn_throttles_locate_retry_after_failure: "
        f"locate_calls={harness.locate_calls}, reasons=({first.reason}, {second.reason})"
    )
    return ok


def test_numerator_sequence_cleared_on_relocalize():
    """OCR miss 로 ROI 를 무효화하면 이전 분자 증가 시퀀스도 지운다."""
    capture = _SeqCapture([_frame(0), _frame(1), _frame(2)])
    detector = EngineerDoneDetector(
        None,
        _settings(
            engineer_done_assist_unusable_after=99,
            engineer_done_numerator_increase_reads=3,
            engineer_done_relocalize_after_miss=1,
        ),
        capture_fn=capture,
        ground_fn=lambda _image: (525, 550),
        ocr_fn=_CountingFn(["10/350", ""]),
    )
    detector()
    detector()
    had_sequence = detector._numerator_sequence == [10]
    detector()
    ok = had_sequence and detector._numerator_sequence == [] and detector._roi_ratios is None
    print(
        f"[{'PASS' if ok else 'FAIL'}] numerator_sequence_cleared_on_relocalize "
        f"(had_sequence={had_sequence})"
    )
    return ok


def test_lower_numerator_restarts_sequence():
    """낮아진 분자 값은 새 증가 시퀀스를 시작해 이전 높은 값을 이어 쓰지 않는다."""
    assist = _CountingFn([_assist_unusable()] * 3)
    detector = _priority_detector(assist=assist, numerators=[350, 5, 15])
    results = [detector(), detector(), detector()]
    ok = not any(results) and detector._numerator_sequence == [5, 15]
    print(
        f"[{'PASS' if ok else 'FAIL'}] lower_numerator_restarts_sequence: "
        f"{results} sequence={detector._numerator_sequence}"
    )
    return ok


def main() -> int:
    """전체 케이스를 실행하고 통과 여부를 반환한다."""
    tests = [
        test_settings_defaults,
        test_settings_env_load_path,
        test_settings_use_priority_signals,
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
        test_watch_logs_manual_completion_only_for_window_gone,
        test_watch_does_not_call_max_sec_manual_completion,
        test_leftover_counter_does_not_fire,
        test_assist_fn_exception_returns_unusable,
        test_assist_fn_locates_panel_only_once,
        test_assist_fn_throttles_locate_retry_after_failure,
        test_done_when_enough_ok_rows_and_no_red,
        test_red_blocks_done_even_with_enough_rows,
        test_not_done_below_min_ok_rows,
        test_numerator_sequence_cleared_on_relocalize,
        test_lower_numerator_restarts_sequence,
        test_assist_primary_finishes_before_numerator_evaluation,
        test_builder_keeps_assist_when_numerator_clients_fail,
        test_detector_captures_one_frame_per_poll,
        test_detector_fails_closed_for_each_non_positive_priority_threshold,
        test_invalid_numerator_sequences_do_not_finish,
        test_numerator_decision_records_reground_reset,
        test_numerator_decisions_are_saved_for_every_evaluated_observation,
        test_numerator_fallback_requires_three_unusable_assist_observations,
        test_ocr_miss_clears_prior_numerator_evidence,
        test_settings_reject_non_positive_priority_thresholds,
        test_unchanged_crop_preserves_partial_numerator_sequence,
        test_usable_assist_resets_unusable_streak,
    ]
    # 이 파일은 bool 반환형과 assert 형이 섞여 있다. assert 형은 통과 시 None 을
    # 돌려주므로 "falsy = 실패" 로 세면 멀쩡한 테스트가 실패로 잡힌다. 실패 신호는
    # 명시적 False 이거나 예외다.
    results = []
    for test in tests:
        try:
            results.append(test() is not False)
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            results.append(False)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
