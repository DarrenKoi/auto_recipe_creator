"""분자(N/M) per-read 기록이 attempt 폴더에 남는지 - fallback Verification 의 입력.

detector 의 boolean 반환은 `false` 와 `unknown` 을 구분하지 못하므로 Verification 입력이
될 수 없다. 그래서 **판독 자체**가 기록으로 남아야 하고, 그 기록이 OCR miss / 같음·감소 /
reground reset / strictly increasing 을 구분해야 한다.

`uv run pytest poc/workflow_3/monitor/test_numerator_records.py`
"""

import dataclasses
import json

from PIL import Image

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.engineer_done_align_adjustment import (
    NUMERATOR_DECISIONS,
    NUMERATOR_RECORDS_FILENAME,
    EngineerDoneDetector,
    NumeratorObservation,
    classify_numerator_decision,
)


def _settings(**overrides):
    base = dict(
        engineer_done_assist_enabled=False,
        engineer_done_numerator_increase_reads=3,
        engineer_done_idle_sec=0.0,
    )
    base.update(overrides)
    return dataclasses.replace(load_workflow3_settings(), **base)


def _detector(record_dir, numerator_values, **overrides):
    """주입된 분자 관측 시퀀스로 도는 detector (실장비/VLM 없음)."""
    values = list(numerator_values)

    def _numerator_fn(_image):
        return values.pop(0) if values else NumeratorObservation(sampled=False)

    return EngineerDoneDetector(
        object(), _settings(**overrides),
        capture_fn=lambda: Image.new("RGB", (32, 24), "white"),
        numerator_fn=_numerator_fn,
        cursor_fn=lambda: None,
        record_dir=record_dir,
    )


def _records(record_dir):
    path = record_dir / NUMERATOR_RECORDS_FILENAME
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ------------------------------------------------------------------
# 판정 분류 (순수 함수).
# ------------------------------------------------------------------


def test_decision_classes_are_closed_and_distinguish_the_four_cases():
    """네 경우가 서로 다른 이름을 얻고, 이름 집합은 닫혀 있다."""
    cases = {
        "not_sampled": dict(sampled=False, value=None, reset_reason=None, sequence=[]),
        "ocr_miss": dict(sampled=True, value=None, reset_reason="ocr_miss", sequence=[]),
        "equal_or_decrease": dict(sampled=True, value=4, reset_reason="equal_or_decrease",
                                  sequence=[4]),
        "reground_reset": dict(sampled=True, value=None, reset_reason="reground",
                               sequence=[]),
        "first_sample": dict(sampled=True, value=7, reset_reason=None, sequence=[7]),
        "strictly_increasing": dict(sampled=True, value=9, reset_reason=None,
                                    sequence=[7, 8, 9]),
    }
    for expected, kwargs in cases.items():
        assert classify_numerator_decision(**kwargs) == expected, kwargs
    assert set(cases) == set(NUMERATOR_DECISIONS)


def test_reground_reset_wins_over_a_readable_value():
    """재grounding 은 누적을 되돌린 사건이라 값이 읽혔어도 증가 근거가 되지 않는다."""
    assert classify_numerator_decision(
        sampled=True, value=12, reset_reason="reground", sequence=[12]
    ) == "reground_reset"


# ------------------------------------------------------------------
# 배선 - attempt 폴더에 JSONL 로 남는다.
# ------------------------------------------------------------------


def test_watch_writes_one_record_per_read_into_the_attempt_folder(tmp_path):
    """폴링 회차마다 1줄 - 값/시각/판정이 남는다."""
    detector = _detector(tmp_path, [
        NumeratorObservation(sampled=True, value=3, reason="ok"),
        NumeratorObservation(sampled=True, value=4, reason="ok"),
        NumeratorObservation(sampled=True, value=5, reason="ok"),
    ])
    results = [detector(), detector(), detector()]

    records = _records(tmp_path)
    assert len(records) == 3
    assert [r["poll"] for r in records] == [1, 2, 3]
    assert [r["value"] for r in records] == [3, 4, 5]
    assert [r["decision"] for r in records] == [
        "first_sample", "strictly_increasing", "strictly_increasing",
    ]
    assert all(r["observed_at"] for r in records)
    # 세 표본이 엄격히 증가했으니 detector 는 완료로 본다(반환 계약 무변경).
    assert results[-1] is True


def test_records_distinguish_miss_decrease_and_reground(tmp_path):
    """OCR miss / 같음·감소 / reground reset 이 기록에서 갈린다."""
    detector = _detector(tmp_path, [
        NumeratorObservation(sampled=True, value=5, reason="ok"),
        NumeratorObservation(sampled=True, value=None, reason="ocr_miss"),
        NumeratorObservation(sampled=True, value=5, reason="ok"),
        NumeratorObservation(sampled=True, value=5, reason="ok"),
        NumeratorObservation(sampled=True, value=6, reason="reground",
                             reset_reason="reground"),
    ])
    for _ in range(5):
        detector()

    decisions = [r["decision"] for r in _records(tmp_path)]
    assert decisions == [
        "first_sample", "ocr_miss", "first_sample", "equal_or_decrease", "reground_reset",
    ]
    # 어느 회차도 완료로 오인되지 않는다.
    assert all(r["done"] is False for r in _records(tmp_path))


def test_no_record_dir_means_no_file_and_unchanged_behaviour(tmp_path):
    """수집 off(record_dir 미지정)면 파일이 생기지 않고 판정은 그대로다."""
    detector = _detector(None, [
        NumeratorObservation(sampled=True, value=1, reason="ok"),
        NumeratorObservation(sampled=True, value=2, reason="ok"),
        NumeratorObservation(sampled=True, value=3, reason="ok"),
    ])
    results = [detector(), detector(), detector()]
    assert results[-1] is True
    assert list(tmp_path.rglob(NUMERATOR_RECORDS_FILENAME)) == []


def test_record_write_failure_does_not_break_detection(tmp_path):
    """기록이 깨져도 감지는 계속된다 - 기록은 보조물이지 판정 경로가 아니다."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")

    detector = _detector(blocker, [
        NumeratorObservation(sampled=True, value=1, reason="ok"),
        NumeratorObservation(sampled=True, value=2, reason="ok"),
        NumeratorObservation(sampled=True, value=3, reason="ok"),
    ])
    assert [detector(), detector(), detector()][-1] is True
