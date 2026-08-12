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
    ALL_BLANK_RELOCATE_AFTER,
    EngineerDoneDetector,
    _make_assist_fn,
    build_engineer_done_detector,
    extract_numerator,
    parse_point_1000,
    point_to_roi_ratios,
)
from poc.workflow_3.sem_monitor import assist_score as asc
from poc.workflow_3.sem_monitor.assist_score import RowState
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
        engineer_done_ok_streak=2,
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
    """변화 + OCR 2 -> 4: 첫 읽기는 baseline 확정(False), 두 번째에 delta+streak 충족 -> done.

    옛 판정("2 -> 3, 비감소면 done")에서 새 판정(delta>=min_delta and streak>=ok_streak)
    으로 바뀌며 두 번째 읽기 값을 3에서 4로 올렸다 - delta(=4-2=2)가 _settings() 의
    min_delta=2 를 충족해야 하기 때문. rows_fn 은 ok_streak=2 를 충족하는 연속 정상
    2행을 공급한다(Assist streak 없이는 delta 만으론 done 이 안 된다).
    """
    capture = _SeqCapture([
        _frame(1),            # grounding 캡처
        _frame(1),            # baseline (첫 샘플, OCR 안 함)
        _frame(2),            # 변화 1 -> OCR '2' (baseline_n 확정)
        _frame(3),            # 변화 2 -> OCR '4' (delta=2>=2, streak=2>=2 -> done)
    ])
    ground = _CountingFn([(525, 550)])
    ocr = _CountingFn(["2/350", "4/350"])
    rows = _rows_all_ok(2)
    detector = EngineerDoneDetector(
        None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr,
        rows_fn=lambda: rows,
    )
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
    두 번째 OCR 값을 3 -> 4 로 올린 이유는 test_detector_two_read_confirm 과 같다
    (delta=4-2=2 가 _settings() 의 min_delta=2 를 충족해야 함). rows_fn 도 같은
    이유로 추가했다(streak 없이는 delta 만으론 done 이 안 됨).
    """
    capture = _SeqCapture([
        _frame(1),            # grounding 시도 1 (blank 가정 -> 거부)
        _frame(1),            # grounding 시도 2 (거부)
        _frame(1),            # grounding 시도 3 (성공)
        _frame(1),            # baseline crop (첫 샘플)
        _frame(2),            # 변화 1 -> OCR '2' (baseline_n 확정)
        _frame(3),            # 변화 2 -> OCR '4' (delta=2>=2, streak=2>=2 -> done)
    ])
    ground = _CountingFn([None, None, (525, 550)])
    ocr = _CountingFn(["2/350", "4/350"])
    rows = _rows_all_ok(2)
    detector = EngineerDoneDetector(
        None, _settings(), capture_fn=capture, ground_fn=ground, ocr_fn=ocr,
        rows_fn=lambda: rows,
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


def _rows_all_ok(count=7):
    cells = {"Addressing1": "black", "Addressing2": "blank", "Measurement": "black"}
    return [RowState(cells=dict(cells)) for _ in range(count)]


def _detector_with(counter_values, rows):
    """카운터 값을 순서대로 돌려주는 detector 를 만든다.

    브리프 원안은 capture_fn 이 매번 동일한 상수 이미지를 돌려줬다. 그러면
    `__call__` 의 기존 CV 변화게이트(첫 샘플은 무조건 OCR 미호출, 이후로도 프레임이
    안 바뀌면 미호출 - `test_detector_static_no_ocr` 가 지키는 기존 동작)에 걸려
    OCR 이 영원히 불리지 않아 delta/streak 판정 자체가 발화하지 않는다(테스트가
    "통과"는 하지만 검증하려는 로직을 실제로 거치지 않는 공허한 통과가 된다). 매
    호출 색조를 바꿔 실제로 매 회차 변화가 감지되게 한다 - counter_values/ocr_fn/
    rows_fn 은 브리프 그대로다.
    """
    settings = load_workflow3_settings()
    ocr_state = {"i": 0}

    def ocr_fn(_crop):
        idx = min(ocr_state["i"], len(counter_values) - 1)
        ocr_state["i"] += 1
        return f"{counter_values[idx]}/350"

    # 1000x500 - 기본 roi_pad(x=0.03,y=0.02)로 crop 해도 변화게이트의 4x 다운샘플
    # 후 픽셀 수가 min_changed_px(기본 4) 를 넘도록 충분히 크게 잡는다(작은 crop 은
    # 다운샘플 후 표본이 3개뿐이라 색이 달라도 절대 "changed" 로 잡히지 않는다).
    capture_state = {"i": -1}

    def capture_fn():
        capture_state["i"] += 1
        shade = (capture_state["i"] * 20) % 256
        return Image.new("RGB", (1000, 500), (shade, shade, shade))

    detector = EngineerDoneDetector(
        None, settings,
        capture_fn=capture_fn,
        ground_fn=lambda _img: (500, 500),
        ocr_fn=ocr_fn,
        rows_fn=lambda: rows,
    )
    return detector


def test_leftover_counter_does_not_fire():
    """watch 시작 시 7행 전부 검정 + 카운터가 안 움직이면 done 이 아니다.

    옛 판정(n >= 6 and n >= _last_n)이 즉시 True 를 내던 바로 그 상황이다.
    3회 호출 = 1) CV 게이트의 gray baseline 샘플(OCR 미호출) 2) 첫 실제 OCR
    (baseline_n=350 확정) 3) 같은 값 재확인(delta=0) - 세 번째 호출에서 비로소
    delta 로직이 실행되고 0 이 나와 통과한다.
    """
    detector = _detector_with([350, 350, 350], _rows_all_ok())
    results = [detector() for _ in range(3)]
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] leftover_counter_does_not_fire: {results}")
    return ok


def test_delta_reached_but_streak_short():
    """새 측정을 충분히 채워도(delta 충족) 연속 정상이 모자라면(streak) done 이 아니다.

    브리프 원안은 2회 호출이었으나, 1회차는 CV 게이트의 gray baseline 샘플(OCR
    미호출)·2회차는 그 뒤의 첫 실제 OCR 로 baseline_n 을 확정하는 자리라 항상
    False 다(값과 무관) - delta/streak 판정 자체가 아직 실행되지 않는다. 3회차를
    추가해야 delta=10(>=6)·streak=1(<6) 조합이 실제로 평가된다.
    """
    rows = _rows_all_ok(7)
    rows[-2].cells["Measurement"] = "red"   # 최신에서 두 번째가 실패 -> streak = 1
    detector = _detector_with([10, 20], rows)
    results = [detector(), detector(), detector()]
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] delta_reached_but_streak_short: {results}")
    return ok


def test_done_when_delta_and_streak_both_met():
    """delta 와 streak 을 모두 채우면 done.

    브리프 원안은 2회 호출로 두 번째에 done 을 기대했으나, 1회차(gray baseline,
    OCR 미호출)·2회차(첫 실제 OCR, baseline_n=10 확정 - 값과 무관하게 항상 False)
    까지는 구조적으로 delta 를 계산할 수 없다. 3회차에서 n=20 을 읽어 delta=10
    (>=6)·streak=7(>=6) 이 모두 만족되어 True 가 나온다.
    """
    detector = _detector_with([10, 20], _rows_all_ok())
    first = detector()      # gray baseline 샘플 -> False (OCR 미호출)
    second = detector()     # 첫 실제 OCR: n=10 -> baseline_n 확정 -> False
    third = detector()      # 두 번째 OCR: n=20, delta=10, streak=7 -> True
    ok = (first is False) and (second is False) and (third is True)
    print(
        f"[{'PASS' if ok else 'FAIL'}] done_when_delta_and_streak_both_met: "
        f"{first},{second},{third}"
    )
    return ok


def test_rows_fn_exception_returns_false() -> bool:
    """rows_fn 이 예외를 던져도 삼켜지고 streak 0(= 아직 아님)으로 처리돼 False 를 낸다.

    `_read_rows` 가 예외를 삼키지 않으면 폴링 루프까지 예외가 전파돼 detector 가 그
    시점에서 죽는다. delta/streak 판정이 실제로 실행되는 3회차까지 호출해 검증한다
    (1회차=CV 게이트 gray baseline, 2회차=첫 실제 OCR 로 baseline_n 확정).
    """
    settings = load_workflow3_settings()
    counter_values = [10, 20]
    ocr_state = {"i": 0}

    def ocr_fn(_crop):
        idx = min(ocr_state["i"], len(counter_values) - 1)
        ocr_state["i"] += 1
        return f"{counter_values[idx]}/350"

    capture_state = {"i": -1}

    def capture_fn():
        capture_state["i"] += 1
        shade = (capture_state["i"] * 20) % 256
        return Image.new("RGB", (1000, 500), (shade, shade, shade))

    def rows_fn():
        raise RuntimeError("rows boom")

    detector = EngineerDoneDetector(
        None, settings,
        capture_fn=capture_fn,
        ground_fn=lambda _img: (500, 500),
        ocr_fn=ocr_fn,
        rows_fn=rows_fn,
    )
    try:
        results = [detector() for _ in range(3)]
    except Exception as exc:
        print(f"[FAIL] rows_fn_exception_returns_false: 예외 전파됨 ({exc})")
        return False
    ok = not any(results)
    print(f"[{'PASS' if ok else 'FAIL'}] rows_fn_exception_returns_false: {results}")
    return ok


def _rows_of(verdicts):
    """verdict 목록을 RowState 목록으로 (Measurement 만으로 성부가 갈리게 구성)."""
    mapping = {
        "ok": {"Addressing1": "black", "Addressing2": "blank", "Measurement": "black"},
        "fail": {"Addressing1": "black", "Addressing2": "blank", "Measurement": "red"},
        "pending": {"Addressing1": "blank", "Addressing2": "blank", "Measurement": "blank"},
    }
    return [asc.RowState(cells=dict(mapping[v])) for v in verdicts]


class _RowsFnHarness:
    """_make_assist_fn 의 스텁 배선. locate/overlay 호출 횟수를 센다.

    _make_assist_fn 은 함수 본문 안에서 import 하므로 engineer_done_align_adjustment
    모듈 속성을 패치해도 가로채지 못한다 - 원본 모듈(assist_score, util)을 패치해야
    내부의 `from X import Y` 가 스텁에 바인딩된다.
    """

    def __init__(self, rows_seq, locate_ok=True):
        self.rows_seq = list(rows_seq)
        self.locate_ok = locate_ok
        self.locate_calls = 0
        self.overlay_calls = 0
        self._saved = {}

    def _locate(self, *a, **k):
        self.locate_calls += 1
        if not self.locate_ok:
            return None
        grid = []
        for row_idx in range(7):
            top = 1 + row_idx * 2
            grid.append([
                {"left": 1, "top": top, "right": 7, "bottom": top + 1},
                {"left": 12, "top": top, "right": 18, "bottom": top + 1},
            ])
        layout = SimpleNamespace(
            grid=grid,
            columns=("Addressing1", "Measurement"),
        )
        return ({"left": 0, "top": 0, "right": 20, "bottom": 20}, layout)

    def _read(self, image, layout):
        return self.rows_seq.pop(0) if self.rows_seq else []

    def _overlay(self, *a, **k):
        self.overlay_calls += 1

    def __enter__(self):
        for mod, name, fn in (
            (asc, "locate_assist_layout", self._locate),
            (asc, "read_row_states", self._read),
            (asc, "save_assist_overlay", self._overlay),
            (w3util, "capture_window", lambda win: Image.new("RGB", (20, 20))),
            (w3util, "crop_image", lambda img, box: img),
        ):
            self._saved[(mod, name)] = getattr(mod, name)
            setattr(mod, name, fn)
        return self

    def __exit__(self, *exc):
        for (mod, name), orig in self._saved.items():
            setattr(mod, name, orig)
        return False


def test_assist_fn_distinguishes_unusable_from_pending():
    image = Image.new("RGB", (20, 20), (240, 240, 240))
    with _RowsFnHarness([], locate_ok=False):
        failed_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        failed = failed_fn(image)

    pending_rows = _rows_of(["pending"] * 7)
    with _RowsFnHarness([pending_rows], locate_ok=True):
        pending_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        pending = pending_fn(image)

    assert failed.status == "unusable"
    assert failed.reason == "layout_unavailable"
    assert pending.status == "unusable"
    assert pending.reason == "measurement_unreadable"
    assert [row.verdict for row in pending.rows] == ["pending"] * 7
    return True


def test_assist_fn_ignores_addressing_fail_when_measurement_unreadable():
    """Addressing1 실패만으로 Measurement 미판독 프레임을 usable 로 올리지 않는다."""
    unreadable_rows = [
        RowState(cells={
            "Addressing1": "red",
            "Addressing2": "blank",
            "Measurement": "blank",
        }),
        RowState(cells={
            "Addressing1": "red",
            "Addressing2": "blank",
            "Measurement": "unknown",
        }),
    ]
    with _RowsFnHarness([unreadable_rows], locate_ok=True):
        assist_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        observation = assist_fn(Image.new("RGB", (20, 20), (240, 240, 240)))

    assert observation.status == "unusable"
    assert observation.reason == "measurement_unreadable"
    assert observation.rows == unreadable_rows
    assert [row.verdict for row in observation.rows] == ["fail", "fail"]
    return True


def test_rows_fn_locates_layout_only_once():
    """격자는 한 번만 잡고 캐시한다 - 폴링마다 VLM 을 부르면 안 된다."""
    ok = _rows_of(["ok"] * 3)
    with _RowsFnHarness([ok, ok, ok]) as h:
        fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        for _ in range(3):
            fn(Image.new("RGB", (20, 20)))
    passed = h.locate_calls == 1
    print(f"[{'PASS' if passed else 'FAIL'}] rows_fn_locates_layout_only_once: {h.locate_calls}")
    return passed


def test_rows_fn_warns_once_on_locate_failure():
    """로케이트가 계속 실패해도 경고는 한 번만 - watch 내내 반복되면 콘솔이 쓸모없어진다."""
    with _RowsFnHarness([], locate_ok=False) as h:
        fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            first, second, third = (fn(Image.new("RGB", (20, 20))) for _ in range(3))
    text = buf.getvalue()
    passed = (
        all(result.reason == "layout_unavailable" for result in (first, second, third))
        and text.count("[WARNING]") == 1
        and h.locate_calls == 3
    )
    print(f"[{'PASS' if passed else 'FAIL'}] rows_fn_warns_once_on_locate_failure: "
          f"warns={text.count('[WARNING]')} locates={h.locate_calls}")
    return passed


def test_rows_fn_throttles_locate_retry_after_failure():
    """(I5) 로케이트 실패 후 재시도는 reground_sec 로 throttle 돼야 한다.

    수정 전에는 실패마다 캐시를 안 하고 매 결정 폴링마다 2단계 VLM(15s timeout) +
    PaddleOCR(30s timeout) 왕복을 반복해 watch 루프를 막았다. reground_sec 를 크게
    잡으면 두 번째 호출은 throttle 되어 locate 를 다시 시도하지 않아야 한다.
    """
    settings = _settings(engineer_done_reground_sec=3600.0)
    with _RowsFnHarness([], locate_ok=False) as h:
        fn = _make_assist_fn(object(), settings, debug_dir=None)
        first, second, third = (fn(Image.new("RGB", (20, 20))) for _ in range(3))
    passed = (
        first.reason == "layout_unavailable"
        and second.reason == third.reason == "locate_throttled"
        and h.locate_calls == 1
    )
    print(f"[{'PASS' if passed else 'FAIL'}] rows_fn_throttles_locate_retry_after_failure: "
          f"locates={h.locate_calls}")
    return passed


def test_rows_fn_overlay_only_on_verdict_change():
    """오버레이는 판정이 바뀔 때만 - 폴링마다 저장하면 디스크가 찬다."""
    same = _rows_of(["ok", "ok"])
    changed = _rows_of(["ok", "fail"])
    with _RowsFnHarness([same, list(same), changed]) as h:
        fn = _make_assist_fn(object(), _settings(), debug_dir=Path("/tmp/nonexistent-overlay-dir"))
        for _ in range(3):
            fn(Image.new("RGB", (20, 20)))
    passed = h.overlay_calls == 2
    print(f"[{'PASS' if passed else 'FAIL'}] rows_fn_overlay_only_on_verdict_change: {h.overlay_calls}")
    return passed


def test_rows_fn_relocates_after_all_blank_streak():
    """전 행이 계속 빈칸이면 패널 이동으로 보고 격자를 다시 잡는다."""
    blanks = [_rows_of(["pending"] * 3) for _ in range(ALL_BLANK_RELOCATE_AFTER)]
    with _RowsFnHarness(blanks) as h:
        fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        results = [fn(Image.new("RGB", (20, 20))) for _ in range(ALL_BLANK_RELOCATE_AFTER)]
    passed = h.locate_calls == 1 and results[-1].reason == "measurement_unreadable"
    print(f"[{'PASS' if passed else 'FAIL'}] rows_fn_relocates_after_all_blank_streak: "
          f"locates={h.locate_calls} last={results[-1]}")
    return passed


def test_baseline_cleared_on_relocalize():
    """재grounding 하면 baseline 도 무효화한다.

    옛 구현은 _last_n 만 살려둬서 옛 ROI 값과 새 ROI 값을 비교했다. 같은 실수를 막는다.
    브리프 원안은 1회 호출 뒤 바로 리셋을 확인했으나, 1회차는 CV 게이트의 gray
    baseline 샘플이라 OCR 이 아예 안 불려 baseline_n 이 애초에 None 이다(리셋이
    실제로 뭔가를 지우는지 증명하지 못하는 공허한 검증) - 2회차를 더해 baseline_n
    이 진짜 값을 갖게 한 뒤에 리셋으로 지워지는지 확인한다.
    """
    detector = _detector_with([10, 20], _rows_all_ok())
    detector()              # gray baseline 샘플 -> baseline_n 여전히 None
    detector()              # 첫 실제 OCR -> baseline_n = 10 (None 아님)
    had_baseline = detector._baseline_n is not None
    detector._roi_ratios = None       # 재grounding 예약 상태를 흉내낸다
    detector._reset_baseline()
    ok = had_baseline and detector._baseline_n is None
    print(f"[{'PASS' if ok else 'FAIL'}] baseline_cleared_on_relocalize (had_baseline={had_baseline})")
    return ok


def test_baseline_adopts_lower_reading_as_new_run():
    """(I3) baseline 확정 후 더 낮은 값이 읽히면(카운터 역행) 그 값을 새 baseline 으로
    채택해야 한다.

    옛 구현은 baseline 이 한 번 확정되면(watch 시작 시 잔존 카운터를 잘못 흡수한
    경우 포함) 절대 바뀌지 않았다 - 이후 실제 새 측정이 낮은 값에서 다시 시작해도
    delta 가 영원히 음수가 되어(OCR 은 계속 성공하므로 재grounding 도 안 걸림) 이
    watch 내내 done 이 나올 수 없었다. counter_values=[350, 5, 15]: baseline=350
    확정 -> n=5(<350, 역행) -> baseline 재설정=5, delta=0(아직 done 아님) -> n=15,
    delta=10(>=6), streak=7(>=6) -> done.
    """
    detector = _detector_with([350, 5, 15], _rows_all_ok())
    buf = io.StringIO()
    with redirect_stdout(buf):
        first = detector()   # gray baseline 샘플
        second = detector()  # n=350 -> baseline 확정
        third = detector()   # n=5 -> 역행 감지, baseline=5 로 재설정
        fourth = detector()  # n=15 -> delta=10, streak=7 -> done
    text = buf.getvalue()
    ok = (
        first is False and second is False and third is False and fourth is True
        and detector._baseline_n == 5
        and "역행" in text
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] baseline_adopts_lower_reading_as_new_run: "
        f"{first},{second},{third},{fourth} baseline={detector._baseline_n}"
    )
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
        test_leftover_counter_does_not_fire,
        test_delta_reached_but_streak_short,
        test_done_when_delta_and_streak_both_met,
        test_rows_fn_exception_returns_false,
        test_assist_fn_distinguishes_unusable_from_pending,
        test_assist_fn_ignores_addressing_fail_when_measurement_unreadable,
        test_rows_fn_locates_layout_only_once,
        test_rows_fn_warns_once_on_locate_failure,
        test_rows_fn_throttles_locate_retry_after_failure,
        test_rows_fn_overlay_only_on_verdict_change,
        test_rows_fn_relocates_after_all_blank_streak,
        test_baseline_cleared_on_relocalize,
        test_baseline_adopts_lower_reading_as_new_run,
    ]
    results = [test() for test in tests]
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n[INFO] engineer_done 테스트: {passed}/{total} 통과")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
