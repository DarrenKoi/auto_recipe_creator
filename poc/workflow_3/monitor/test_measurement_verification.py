"""Measurement Verification record + unknown-only reader stub.

`uv run pytest poc/workflow_3/monitor/test_measurement_verification.py`
"""

import json

import pytest
from PIL import Image

from poc.workflow_3.monitor.measurement_verification import (
    CROP_FILENAME,
    FAILURE,
    REASON_NOT_CALIBRATED,
    SOURCE_ANNOTATION,
    SOURCE_READER,
    SUCCESS,
    UNKNOWN,
    load_verification_record,
    read_measurement_stub,
    verification_record,
    write_verification_record,
)

_BOX = {"left": 2, "top": 2, "right": 30, "bottom": 20}


def _locate_ok(_image):
    return dict(_BOX)


def _save(crop, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, "JPEG")


# ------------------------------------------------------------------
# record 계약.
# ------------------------------------------------------------------


def test_three_values_round_trip_and_source_distinguishes_reader_from_annotation(tmp_path):
    """세 값이 구분되어 저장/복원되고 source 가 자동 판독과 사람 판독을 가른다."""
    cases = [
        (SUCCESS, SOURCE_ANNOTATION, "measurement_normalized"),
        (FAILURE, SOURCE_ANNOTATION, "red_rows_remain"),
        (UNKNOWN, SOURCE_READER, REASON_NOT_CALIBRATED),
    ]
    for index, (value, source, reason) in enumerate(cases):
        attempt_dir = tmp_path / f"attempt_{index + 1}"
        record = verification_record(
            value=value, reason=reason, source=source,
            baseline_ref=f"attempt_{index + 1}/recording/base.jpg",
            post_action_ref=f"attempt_{index + 1}/recording/after.jpg",
        )
        path = write_verification_record(attempt_dir, record)
        assert path.name == "measurement_verification.json"

        loaded = load_verification_record(path)
        assert loaded["value"] == value
        assert loaded["source"] == source
        assert loaded["reason"] == reason
        assert loaded["baseline_ref"].startswith("attempt_")
        assert loaded["post_action_ref"].startswith("attempt_")
        # 참조는 Episode-relative 만 - 절대 경로/부모 탈출이 없다.
        for ref in (loaded["baseline_ref"], loaded["post_action_ref"], loaded["evidence"]):
            assert not ref.startswith("/") and ".." not in ref.split("/"), ref


def test_unsupported_value_or_source_is_rejected(tmp_path):
    """3상태와 두 source 밖의 값은 만들 수도, 읽을 수도 없다."""
    with pytest.raises(ValueError):
        verification_record(value="maybe", reason="x", source=SOURCE_READER)
    with pytest.raises(ValueError):
        verification_record(value=UNKNOWN, reason="x", source="guess")

    path = tmp_path / "measurement_verification.json"
    path.write_text(json.dumps({"value": "recovered", "source": SOURCE_READER}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_verification_record(path)


# ------------------------------------------------------------------
# unknown-only stub.
# ------------------------------------------------------------------


def test_stub_is_unknown_for_any_input_and_persists_the_crop(tmp_path):
    """stub 은 어떤 화면에도 unknown 이고, 근거로 패널 crop 만 남긴다."""
    attempt_dir = tmp_path / "attempt_1"
    results = []
    for color in ("white", "black"):
        record = read_measurement_stub(
            Image.new("RGB", (64, 48), color),
            locate_fn=_locate_ok, save_crop_fn=_save, attempt_dir=attempt_dir,
            crop_ref_prefix="attempt_1",
        )
        results.append(record)

    for record in results:
        assert record["value"] == UNKNOWN
        assert record["reason"] == REASON_NOT_CALIBRATED
        assert record["source"] == SOURCE_READER
        assert record["evidence"] == f"attempt_1/{CROP_FILENAME}"
    # 화면 내용이 정반대여도 판정이 같다 = 픽셀을 보는 판독 로직이 없다.
    assert results[0]["value"] == results[1]["value"]
    assert results[0]["reason"] == results[1]["reason"]
    assert (attempt_dir / CROP_FILENAME).is_file()


def test_crop_failure_is_still_unknown_but_with_a_different_reason(tmp_path):
    """근거조차 못 남긴 경우와 '판독기가 아직 없다' 는 사후에 구분되어야 한다."""
    image = Image.new("RGB", (64, 48), "white")
    attempt_dir = tmp_path / "attempt_1"

    not_located = read_measurement_stub(
        image, locate_fn=lambda _img: None, save_crop_fn=_save, attempt_dir=attempt_dir
    )
    assert not_located["value"] == UNKNOWN
    assert not_located["reason"] != REASON_NOT_CALIBRATED
    assert "panel_not_located" in not_located["reason"]
    assert not_located["evidence"] == ""

    def _boom_locate(_img):
        raise RuntimeError("vlm down")

    locate_error = read_measurement_stub(
        image, locate_fn=_boom_locate, save_crop_fn=_save, attempt_dir=attempt_dir
    )
    assert locate_error["value"] == UNKNOWN
    assert "vlm down" in locate_error["reason"]

    def _boom_save(_crop, _path):
        raise OSError("disk full")

    save_error = read_measurement_stub(
        image, locate_fn=_locate_ok, save_crop_fn=_boom_save, attempt_dir=attempt_dir
    )
    assert save_error["value"] == UNKNOWN
    assert "disk full" in save_error["reason"]
    assert save_error["evidence"] == ""


def test_stub_never_reports_success_or_failure(tmp_path):
    """열 분리 판독 로직이 집에서 작성되지 않았다는 계약 - stub 은 success/failure 를 못 낸다."""
    values = {
        read_measurement_stub(
            Image.new("RGB", (64, 48), color),
            locate_fn=locate, save_crop_fn=_save, attempt_dir=tmp_path,
        )["value"]
        for color in ("white", "black", "red")
        for locate in (_locate_ok, lambda _img: None)
    }
    assert values == {UNKNOWN}


# ------------------------------------------------------------------
# 사이클 배선.
# ------------------------------------------------------------------


def test_cycle_writes_the_verification_record_into_the_attempt_folder(tmp_path, monkeypatch):
    """수집 on 이면 attempt 폴더에 record 가 남고 Episode 가 그것을 참조한다."""
    import dataclasses

    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(
        cycle, "capture_window", lambda _win: Image.new("RGB", (64, 48), "white")
    )
    monkeypatch.setattr(cycle, "_locate_measurement_panel", _locate_ok)
    settings = dataclasses.replace(load_workflow3_settings(), episode_collect_enabled=True)

    context = {"eqp_id": "EQP1", "recipe_id": "", "tag": "T1", "attempt_seq": 1,
               "tool_window": object()}
    cycle.write_attempt_verification(context, settings)

    path = (tmp_path / "EQP1" / "_unregistered" / "T1" / "attempt_1"
            / "measurement_verification.json")
    record = load_verification_record(path)
    assert record["value"] == UNKNOWN
    assert record["source"] == SOURCE_READER


def test_collection_off_writes_no_verification_record(tmp_path, monkeypatch):
    """수집 off 면 record 도 crop 도 없다."""
    import dataclasses

    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    settings = dataclasses.replace(load_workflow3_settings(), episode_collect_enabled=False)
    context = {"eqp_id": "EQP1", "recipe_id": "", "tag": "T1", "attempt_seq": 1,
               "tool_window": object()}
    cycle.write_attempt_verification(context, settings)
    assert list(tmp_path.rglob("measurement_verification.json")) == []
