"""엔트리포인트 테스트 - 입력 3파일 로드, degrade, 종료 상태."""

import json

from poc.workflow_3.workflow_extract.extract_workflow import run_extract


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _timeline_event(seq, t_sec, action="click", element="PM"):
    return {
        "seq": seq, "t_sec": t_sec, "action": action, "coords": {"x": 100, "y": 200},
        "element": element, "element_source": "ocr", "target_kind": "ui_control",
        "region": "ui", "generation": 0, "occlusion": "none", "text": None,
        "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }


def _session(tmp_path, events):
    out = tmp_path / "recording_filter"
    _write(out / "interaction_timeline.json",
           {"capture_dir": str(tmp_path / "recording"), "events": events})
    _write(out / "region_map.json", {"maps": [{"generation": 0, "live_box": None}]})
    _write(out / "change_events.json", {"events": []})
    return out


def test_missing_timeline_is_an_error(tmp_path):
    assert run_extract(input_dir=tmp_path) == "timeline_not_found"


def test_empty_timeline_is_not_success(tmp_path):
    """이벤트 0건은 조용한 성공이 아니다."""
    out = _session(tmp_path, [])
    assert run_extract(input_dir=out) == "no_events"


def test_writes_workflow_json_and_markdown(tmp_path):
    out = _session(tmp_path, [_timeline_event(0, 10.0), _timeline_event(1, 40.0, element="OK")])
    assert run_extract(input_dir=out) == "success"
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 2
    assert (out / "workflow.md").is_file()


def test_degrades_without_region_map(tmp_path):
    """region_map.json 이 없어도 실패하지 않고 R1 만 degrade 한다."""
    out = _session(tmp_path, [_timeline_event(0, 10.0)])
    (out / "region_map.json").unlink()
    assert run_extract(input_dir=out) == "success"


def test_workflow_json_records_settings(tmp_path):
    """임계값을 바꿔가며 재실행하므로 산출물이 자기 설정을 들고 있어야 한다."""
    out = _session(tmp_path, [_timeline_event(0, 10.0)])
    run_extract(input_dir=out)
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert "settings" in payload
    assert payload["settings"]["recenter_min_ratio"] == 0.40


def test_eqp_id_from_manual_layout_path(tmp_path):
    """eqp_id 는 고정 인덱스가 아니라 `_manual` 마커 바로 앞 컴포넌트에서 뽑는다.

    parts[-4] 같은 고정 음수 인덱스는 `<root>/<eqp>/_manual/<tag>/recording`
    레이아웃에서만 우연히 맞는다 - 다른 캡처 레이아웃에서는 틀린 라벨을 조용히
    낼 수 있으므로, 실제 `_manual` 레이아웃에서 정확한 eqp_id 를 뽑는지 고정한다.
    """
    out = tmp_path / "recording_filter"
    capture_dir = tmp_path / "EQP123" / "_manual" / "tag001" / "recording"
    _write(out / "interaction_timeline.json",
           {"capture_dir": str(capture_dir), "events": [_timeline_event(0, 10.0)]})
    _write(out / "region_map.json", {"maps": [{"generation": 0, "live_box": None}]})
    _write(out / "change_events.json", {"events": []})

    assert run_extract(input_dir=out) == "success"
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert payload["session"]["eqp_id"] == "EQP123"


def test_eqp_id_falls_back_to_unknown_without_manual_marker(tmp_path):
    """`_manual` 마커가 없는 경로(예: captured_img_from_rcs 레이아웃)는 "?" 로 정직하게 표시한다."""
    out = tmp_path / "recording_filter"
    capture_dir = tmp_path / "EQP1" / "CLASS" / "RECIPE" / "captured_img_from_rcs" / "tag001" / "recording"
    _write(out / "interaction_timeline.json",
           {"capture_dir": str(capture_dir), "events": [_timeline_event(0, 10.0)]})
    _write(out / "region_map.json", {"maps": [{"generation": 0, "live_box": None}]})
    _write(out / "change_events.json", {"events": []})

    assert run_extract(input_dir=out) == "success"
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert payload["session"]["eqp_id"] == "?"
