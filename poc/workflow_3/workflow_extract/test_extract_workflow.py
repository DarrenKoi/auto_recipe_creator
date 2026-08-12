"""엔트리포인트 테스트 - 입력 3파일 로드, degrade, 종료 상태."""

import json

from poc.workflow_3.recording_filter.region_gate import REGION_MAP_KEY
from poc.workflow_3.workflow_extract.extract_workflow import (
    _exit_code,
    _load_changes,
    _load_live_boxes,
    run_extract,
)


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
    _write(out / "region_map.json", {REGION_MAP_KEY: [{"generation": 0, "live_box": None}]})
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


def test_probable_close_evidence_is_not_emitted_as_workflow_click(tmp_path):
    """비재생 닫기 추정은 절차 동작이 아니며, 실제 클릭은 그대로 남는다."""
    probable_close = _timeline_event(
        0, 9.0, action="probable_close_click", element="Remote Monitoring close button"
    )
    probable_close["replayable"] = False
    real_click = _timeline_event(1, 10.0, element="OK")
    out = _session(tmp_path, [probable_close, real_click])

    assert run_extract(input_dir=out) == "success"

    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["action"] == "click"
    assert payload["steps"][0]["target"] == "OK"
    assert payload["steps"][0]["raw_events"] == [1]
    markdown = (out / "workflow.md").read_text(encoding="utf-8")
    assert "probable_close_click" not in markdown
    assert "Remote Monitoring close button" not in markdown


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
    _write(out / "region_map.json", {REGION_MAP_KEY: [{"generation": 0, "live_box": None}]})
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
    _write(out / "region_map.json", {REGION_MAP_KEY: [{"generation": 0, "live_box": None}]})
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


# ---------------------------------------------------------------------------
# 2026-08-11 최종 리뷰 C1/I1 - 생산자 산출물 <-> 소비자 리더 계약.
# ---------------------------------------------------------------------------

_LIVE_BOX = {"left": 200, "top": 100, "right": 1000, "bottom": 700}


def _real_region_map(tmp_path, live_box=None):
    """실제 build_region_maps 를 돌려 region_map.json 을 만든다(detect_sem_box 만 대역).

    C1 을 놓친 이유가 "픽스처가 소비자 쪽 키를 직접 적어 둔 것" 이므로, 이 테스트는
    문자열을 손으로 적지 않고 **생산자 함수가 실제로 쓴 파일**을 소비자에게 먹인다.
    """
    from PIL import Image

    import poc.workflow_3.sem_monitor.sem_box_detect as sem_box_detect_mod
    from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
    from poc.workflow_3.recording_filter.region_gate import FrameMeta, build_region_maps

    frame = tmp_path / "frame_0.jpg"
    Image.new("RGB", (1200, 800), "white").save(frame, format="JPEG")
    event = ChangeEvent(
        rank=0, frame_path=str(frame), prev_frame_path=str(frame), timestamp_sec=1.0,
        frame_index=0, change_bbox=dict(_LIVE_BOX), largest_blob_area_px=9000,
        changed_pixels=9000,
    )
    meta = FrameMeta(
        t_sec=1.0, rect={"left": 0, "top": 0, "right": 1200, "bottom": 800},
        occlusion="none", cursor_xy=[500, 400], cursor_in_window=True,
    )

    class _Detection:
        def __init__(self, bbox):
            self.detected = bbox is not None
            self.bbox_px = bbox

    original = sem_box_detect_mod.detect_sem_box
    sem_box_detect_mod.detect_sem_box = lambda _img, _client, **_kw: _Detection(live_box)
    try:
        out = tmp_path / "recording_filter"
        build_region_maps([event], [meta], object(), out)
    finally:
        sem_box_detect_mod.detect_sem_box = original
    return out


def test_producer_region_map_is_readable_by_consumer(tmp_path):
    """build_region_maps 가 쓴 region_map.json 을 _load_live_boxes 가 실제로 읽어야 한다.

    이것이 C1(생산자 "generations" vs 소비자 "maps") 을 잡는 테스트다 - 두 모듈이
    같은 키 상수를 공유하지 않으면 여기서 빈 dict 가 나온다.
    """
    out = _real_region_map(tmp_path, live_box=dict(_LIVE_BOX))
    boxes = _load_live_boxes(out)
    assert boxes == {0: _LIVE_BOX}, boxes


def test_load_live_boxes_warns_when_payload_has_zero_boxes(tmp_path, capsys):
    """파싱은 됐는데 박스가 0개면 조용히 빈 dict 를 돌려주면 안 된다(C1 을 숨긴 침묵)."""
    out = _real_region_map(tmp_path, live_box=None)   # detect 실패 -> live_box=None
    assert _load_live_boxes(out) == {}
    captured = capsys.readouterr()
    assert "live_box 가 0개" in captured.out


def _change_events_file(out, records):
    """filter_recording 의 실제 payload 생성기로 change_events.json 을 만든다."""
    from poc.workflow_3.recording_filter.filter_recording import _change_events_payload
    from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent

    events, gate_info = [], {}
    for rank, (t_sec, bbox, verdict, occlusion) in enumerate(records):
        events.append(ChangeEvent(
            rank=rank, frame_path=f"/tmp/f_{rank}.jpg", prev_frame_path=f"/tmp/f_{rank}p.jpg",
            timestamp_sec=t_sec, frame_index=rank, change_bbox=bbox,
            largest_blob_area_px=9000, changed_pixels=9000,
        ))
        if verdict is not None:
            gate_info[rank] = {
                "generation": 0, "region": "live_image",
                "occlusion": occlusion, "verdict": verdict,
            }
    _write(out / "change_events.json", {"events": _change_events_payload(events, gate_info)})


def test_producer_change_events_are_readable_by_consumer(tmp_path):
    """filter_recording 이 쓴 change_events.json 에서 candidate 만 골라 읽어야 한다."""
    out = tmp_path / "recording_filter"
    _change_events_file(out, [
        (10.4, dict(_LIVE_BOX), "candidate", "none"),
        (10.5, dict(_LIVE_BOX), "ambient", "none"),
        (10.6, dict(_LIVE_BOX), "candidate", "full"),
    ])
    loaded = _load_changes(out)
    assert [ev["timestamp_sec"] for ev in loaded] == [10.4], loaded


def _live_click(seq, t_sec):
    event = _timeline_event(seq, t_sec, element=None)
    event.update({"region": "live_image", "target_kind": "live_image",
                  "element_source": "none", "coords": {"x": 600, "y": 400}})
    return event


def _live_session(tmp_path, change_records):
    out = tmp_path / "recording_filter"
    _write(out / "interaction_timeline.json",
           {"capture_dir": str(tmp_path / "EQP1" / "_manual" / "tag" / "recording"),
            "events": [_live_click(0, 10.0)]})
    _write(out / "region_map.json",
           {REGION_MAP_KEY: [{"generation": 0, "live_box": dict(_LIVE_BOX)}]})
    _change_events_file(out, change_records)
    return out


def _steps(out):
    return json.loads((out / "workflow.json").read_text(encoding="utf-8"))["steps"]


def test_loaded_live_box_and_candidate_change_produce_double_click(tmp_path):
    """region_map.json -> live_box -> R1 -> double_click 경로 전체를 한 번에 건다.

    기존 R1 테스트는 live_boxes 를 직접 주입해 로더를 우회했기 때문에, R1 이 실제
    실행에서 100% 죽어 있다는 사실(C1)이 드러나지 않았다.
    """
    out = _live_session(tmp_path, [(10.4, dict(_LIVE_BOX), "candidate", "none")])
    assert run_extract(input_dir=out) == "success"
    step = _steps(out)[0]
    assert step["action"] == "double_click", step
    assert step["grouping_rule"] == "R1"
    assert step["coords_in_live_box"] == [0.5, 0.5]


def test_ambient_live_box_repaint_does_not_produce_double_click(tmp_path):
    """ambient(라이브 영상 자율 갱신)는 FOV 이동 근거가 될 수 없다(I1).

    ambient 전면 리페인트는 비율이 ~1.0 이라 recenter 시그니처와 겉모습이 같다 -
    걸러내지 않으면 평범한 영상 갱신이 `double_click(inferred)` 로 승격돼, 문서에
    엔지니어가 하지 않은 조작이 확신에 찬 얼굴로 실린다.
    """
    out = _live_session(tmp_path, [(10.4, dict(_LIVE_BOX), "ambient", "none")])
    assert run_extract(input_dir=out) == "success"
    step = _steps(out)[0]
    assert step["action"] == "click", step
    assert step["grouping_rule"] == "R5"


def test_change_events_without_verdict_disable_r1_with_warning(tmp_path, capsys):
    """판정 필드가 없는 예전 산출물은 "전부 통과"가 아니라 R1 비활성화 + 경고다."""
    out = _live_session(tmp_path, [(10.4, dict(_LIVE_BOX), None, None)])
    assert run_extract(input_dir=out) == "success"
    assert _steps(out)[0]["grouping_rule"] == "R5"
    assert "판정(verdict)이" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 낮은 우선순위 항목 - 세션 길이, frames/ 레이아웃, 빈 payload 진단.
# ---------------------------------------------------------------------------

def test_duration_includes_trailing_typing_burst_end():
    """마지막 조작이 긴 타이핑이면 세션 길이가 그 끝까지여야 한다.

    t_sec(시작)만 보면 구간 길이를 잃어 세션이 짧게 보고된다.
    """
    from poc.workflow_3.workflow_extract.extract_workflow import _event_end_sec

    typing = _timeline_event(0, 50.0, action="type_text")
    typing["t_sec_end"] = 95.5
    assert _event_end_sec(typing) == 95.5
    assert _event_end_sec(_timeline_event(1, 20.0)) == 20.0


def test_duration_uses_t_sec_end_end_to_end(tmp_path):
    """workflow.json 의 duration_sec 도 구간 끝을 반영해야 한다."""
    typing = _timeline_event(0, 50.0, action="type_text")
    typing["t_sec_end"] = 95.5
    out = _session(tmp_path, [typing])
    assert run_extract(input_dir=out) == "success"
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert payload["session"]["duration_sec"] == 95.5


def test_frame_size_found_in_nested_frames_dir(tmp_path):
    """프레임이 `frames/` 하위에 있어도 R2 용 프레임 크기를 찾아야 한다.

    filter_recording 은 두 레이아웃을 모두 받으므로(_resolve_frames_dir), 이쪽만
    루트를 보면 frames/ 레이아웃에서 R2 가 조용히 꺼진다.
    """
    from PIL import Image

    from poc.workflow_3.workflow_extract.extract_workflow import _frame_size

    capture = tmp_path / "recording"
    (capture / "frames").mkdir(parents=True)
    Image.new("RGB", (640, 480), "white").save(capture / "frames" / "f0.jpg", format="JPEG")
    assert _frame_size(capture) == (640, 480)


def test_empty_payload_is_not_reported_as_corrupted(tmp_path, capsys):
    """내용이 `{}` 인 파일은 "손상" 이 아니라 "비어 있음" 으로 보고해야 한다.

    정직한 진단을 담당하는 헬퍼가 유일하게 거짓말하던 자리다 - `if not payload`
    갈래는 파싱 실패와 빈 내용을 함께 받는다.
    """
    out = tmp_path / "recording_filter"
    _write(out / "interaction_timeline.json", {})

    assert run_extract(input_dir=out) == "timeline_not_found"
    captured = capsys.readouterr()
    assert "내용이 비어 있습니다" in captured.out
    assert "손상되었을 수 있습니다" not in captured.out
