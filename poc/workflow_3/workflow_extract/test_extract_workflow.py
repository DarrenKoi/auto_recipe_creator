"""엔트리포인트 테스트 - 입력 3파일 로드, degrade, 종료 상태."""

import json

from poc.workflow_3.workflow_extract.extract_workflow import _exit_code, run_extract


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


def test_exit_code_maps_success_to_zero():
    """`__main__` 의 SystemExit 매핑을 상태 문자열이 아니라 종료 코드로 직접 고정한다.

    상태 문자열 검사만으로는 "success" 만 0, 나머지는 1" 이라는 규칙 자체가
    뒤집혀도(예: `SystemExit(0)` 로 고정하거나 새 상태를 성공 집합에 잘못 추가)
    잡아내지 못한다 - 그 회귀는 종료 코드를 직접 봐야만 드러난다.
    """
    assert _exit_code("success") == 0


def test_exit_code_maps_no_events_to_nonzero():
    """빈 절차서(no_events)는 반드시 0 이 아닌 종료 코드를 내야 한다."""
    assert _exit_code("no_events") == 1


def test_exit_code_maps_timeline_not_found_to_nonzero():
    """입력 자체가 없는 경우도 반드시 0 이 아닌 종료 코드를 내야 한다."""
    assert _exit_code("timeline_not_found") == 1


def test_malformed_timeline_reports_parse_failure_not_missing(tmp_path, capsys):
    """손상된 timeline 은 "파일이 없다"가 아니라 "있는데 못 읽었다"고 알려야 한다.

    _read_json 은 존재/손상 두 실패를 모두 None 으로 합치지만, 사람에게 보여줄
    진단은 갈라야 한다 - 있는 파일을 "없다"고 하면 사용자가 있지도 않은 파일을
    찾아 헤맨다. 상태 문자열은 두 경우 모두 timeline_not_found 로 동일하게
    유지한다(호출부 동작은 안 바꾸고 진단 문구만 바꾼다).
    """
    out = tmp_path / "recording_filter"
    out.mkdir(parents=True)
    (out / "interaction_timeline.json").write_text("{이것은 유효한 JSON 이 아님", encoding="utf-8")

    assert run_extract(input_dir=out) == "timeline_not_found"
    captured = capsys.readouterr()
    assert "읽지 못했습니다" in captured.out
    assert "이 없습니다" not in captured.out


def test_missing_timeline_reports_absent_not_parse_failure(tmp_path, capsys):
    """파일이 정말 없을 때는 기존 "없습니다 - 먼저 실행하세요" 안내를 그대로 유지한다."""
    assert run_extract(input_dir=tmp_path) == "timeline_not_found"
    captured = capsys.readouterr()
    assert "이 없습니다" in captured.out
    assert "읽지 못했습니다" not in captured.out


def test_malformed_region_map_reports_parse_failure_not_missing(tmp_path, capsys):
    """선택 입력(region_map.json)도 손상 시 "없음"이 아니라 "읽지 못함"으로 보고한다."""
    out = _session(tmp_path, [_timeline_event(0, 10.0)])
    (out / "region_map.json").write_text("not json at all {{{", encoding="utf-8")

    assert run_extract(input_dir=out) == "success"
    captured = capsys.readouterr()
    assert "region_map.json 이 있지만 읽지 못했습니다" in captured.out
