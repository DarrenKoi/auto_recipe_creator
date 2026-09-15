"""List 점유 판정 및 클릭 전 게이트의 오프라인 회귀."""

from poc.workflow_3.check_tool_occupancy import classify_reading


def test_same_row_occupancy_requires_explicit_empty_cells():
    base = {"mc_id": "MCD630", "row_confirmed": True,
            "remote_text": "", "connection_user_text": ""}
    assert classify_reading(base, "MCD630") == "free"
    for remote in ("1", "2", "12"):
        assert classify_reading(dict(base, remote_text=remote), "MCD630") == "occupied_by_other"
    for user in ("kim", "12345", "홍길동"):
        assert classify_reading(dict(base, remote_text=None, connection_user_text=user), "MCD630") == "occupied_by_other"
    for change in ({"remote_text": None}, {"connection_user_text": None},
                   {"remote_text": "?"}, {"remote_text": "0"},
                   {"row_confirmed": "true"}, {"mc_id": "MCD631"},
                   {"remote_text": 1}):
        assert classify_reading(dict(base, **change), "MCD630") == "unknown"
    assert classify_reading({}, "MCD630") == "unknown"


def test_occupancy_gate_prevents_click_and_free_allows_it(monkeypatch, tmp_path):
    from PIL import Image
    from poc.workflow_3.rcs import workflow_select_tool as selection
    from poc.workflow_3 import check_tool_occupancy as checker

    image = Image.new("RGB", (400, 200))
    monkeypatch.setattr(selection, "_is_valid_main_window_title", lambda title: True)
    monkeypatch.setattr(selection, "window_rect_size", lambda window: (400, 200))
    monkeypatch.setattr(selection, "_locate_tool_via_vlm", lambda *a, **kw: ({
        "detection_source": "test", "full_image_point": {"x": 20, "y": 70},
        "verify_crop_box": {"left": 0, "top": 60, "right": 100, "bottom": 80},
        "matched_text": "MCD630",
    }, {}))
    monkeypatch.setattr(selection, "_save_tool_click_overlay", lambda *a, **kw: None)
    monkeypatch.setattr(selection, "foreground_window", lambda *a, **kw: True)
    monkeypatch.setattr(selection, "image_point_to_screen", lambda *a, **kw: {"x": 20, "y": 70})
    clicks = []
    monkeypatch.setattr(selection, "click_at_screen", lambda *a, **kw: clicks.append(kw) or True)
    for state in ("occupied_by_other", "unknown", "free"):
        monkeypatch.setattr(checker, "check_tool_occupancy", lambda *a, **kw: {"occupancy": state})
        result = selection.select_tool_from_main_window(
            object(), "RCS", "uia", "MCD630", image=image,
            require_occupancy_check=True, debug_image_dir=tmp_path,
            pre_click_settle_sec=0, post_double_click_settle_sec=0,
        )
        assert result.occupancy == state
        assert result.double_clicked is (state == "free")
        assert len(clicks) == (1 if state == "free" else 0)


def test_monitor_requests_gate_and_propagates_block(monkeypatch):
    from types import SimpleNamespace
    from poc.workflow_3.monitor import cycle
    from poc.workflow_3.rcs import login_rcs_common

    monkeypatch.setattr(login_rcs_common, "wait_for_rcs_main_window", lambda **kw: (object(), "RCS", "uia"))
    settings = SimpleNamespace(action_enabled=True, connect_action_enabled=True,
                               connect_window_timeout_sec=1, safe_mode=True)
    for state, failure in (("occupied_by_other", "rcs_occupied"),
                           ("unknown", "rcs_occupancy_unknown"), ("free", "success")):
        def connect(tool, **kwargs):
            assert tool == "MCD630"
            assert kwargs["require_occupancy_check"] is True
            return SimpleNamespace(occupancy=state, exit_code=failure, double_clicked=state == "free")
        monkeypatch.setattr(cycle, "connect_to_tool", connect)
        context = {"eqp_id": "MCD630"}
        result = cycle._exec_connect_tool(SimpleNamespace(step_id="connect_tool"), context, settings)
        assert context["occupancy"] == state
        assert result.status == ("success" if state == "free" else "failed")
        assert result.failure_class == (None if state == "free" else failure)


def test_vlm_failure_and_missing_fields_never_mean_free(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from poc.workflow_3 import check_tool_occupancy as checker

    monkeypatch.setattr(checker, "DEBUG_IMAGE_DIR", tmp_path)
    image = Image.new("RGB", (100, 100))
    for text in ('{}', 'not json', '{"mc_id":"MCD630","row_confirmed":true}'):
        client = SimpleNamespace(chat_with_image_b64=lambda **kw: SimpleNamespace(text=text))
        assert checker.check_tool_occupancy(image, "MCD630", client=client)["occupancy"] == "unknown"
    def unavailable(**kw):
        raise RuntimeError("offline")
    client = SimpleNamespace(chat_with_image_b64=unavailable)
    report = checker.check_tool_occupancy(image, "MCD630", client=client)
    assert report["occupancy"] == "unknown"
    assert "offline" in report["error"]


def test_fine_image_excludes_occupied_neighbor_and_keeps_distant_columns():
    from PIL import Image
    from poc.workflow_3.check_tool_occupancy import build_row_read_image

    image = Image.new("RGB", (1000, 200), "white")
    image.paste("blue", (0, 40, 1000, 60))  # MCDA01
    image.paste("red", (0, 60, 1000, 80))  # MCDA23: 바로 아래 점유 행
    layout = {"mc_id": "MCDA01", "row_top": 40, "row_bottom": 60,
              "columns": {"mc_id": [800, 880], "remote": [450, 550],
                          "connection_user": [900, 1000]}}
    fine = build_row_read_image(image, layout, "MCDA01")
    colors = {color for count, color in fine.getcolors(fine.width * fine.height)}
    assert (0, 0, 255) in colors
    assert (255, 0, 0) not in colors
    for change in ({"row_top": -1}, {"row_bottom": 110}, {"mc_id": "MCDA23"}):
        import pytest
        with pytest.raises(ValueError):
            build_row_read_image(image, dict(layout, **change), "MCDA01")


def test_coarse_then_fine_rejects_multiple_mc_ids(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from poc.workflow_3 import check_tool_occupancy as checker

    monkeypatch.setattr(checker, "DEBUG_IMAGE_DIR", tmp_path)
    layout = json.loads('{"headers": [{"name": "Remote", "x": 500}, {"name": "RCS IP", "x": 700}, {"name": "MC ID", "x": 840}, {"name": "Control User", "x": 900}, {"name": "Connection User", "x": 970}]}')
    from poc.workflow_3.vlm.label_verify import PointTextRead
    monkeypatch.setattr(checker, "locate_row_point", lambda *a: {"x": 840, "y": 50})
    monkeypatch.setattr(checker, "read_text_near_point", lambda *a, **kw: PointTextRead(ok=True, raw_text="MCDA01"))
    for ids, expected in ((["MCDA01"], "free"), (["MCDA01", "MCDA23"], "unknown"),
                          (["MCDA23"], "unknown")):
        responses = iter([layout, {"mc_id": "MCDA01", "row_confirmed": True,
                          "visible_mc_ids": ids, "remote_text": "", "connection_user_text": ""}])
        calls = []
        def chat(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=json.dumps(next(responses)))
        report = checker.check_tool_occupancy(Image.new("RGB", (1000, 200)), "MCDA01",
                                             client=SimpleNamespace(chat_with_image_b64=chat))
        assert report["occupancy"] == expected
        assert len(calls) == 2
        assert calls[0]["image_b64"] != calls[1]["image_b64"]


def test_wrong_paddle_id_stops_before_occupancy_read(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from poc.workflow_3 import check_tool_occupancy as checker
    from poc.workflow_3.vlm.label_verify import PointTextRead

    monkeypatch.setattr(checker, "DEBUG_IMAGE_DIR", tmp_path)
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"headers": [{"name": "Remote", "x": 500}, {"name": "RCS IP", "x": 700}, {"name": "MC ID", "x": 840}, {"name": "Control User", "x": 900}, {"name": "Connection User", "x": 970}]}')
    for raw in ("MC0916", "MCD916", "MCDA23 MCD916", ""):
        monkeypatch.setattr(checker, "read_text_near_point", lambda *a, **kw: PointTextRead(ok=True, raw_text=raw))
        report = checker.check_tool_occupancy(Image.new("RGB", (1000, 200)), "MCDA23",
            client=SimpleNamespace(chat_with_image_b64=chat), row_point={"x": 840, "y": 90})
        assert report["occupancy"] == "unknown"
        assert report["diagnosis"] == ("mc_id_unreadable" if not raw else "mc_id_mismatch")
        assert report["mc_id_ocr"]["raw_text"] == raw
    assert len(calls) == 4  # 컬럼 검출만; 점유 판독으로 진행하지 않는다.


def test_locator_maps_column_point_and_paddle_reads_target_band(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from poc.workflow_3 import check_tool_occupancy as checker

    monkeypatch.setattr(checker, "DEBUG_IMAGE_DIR", tmp_path)
    image = Image.new("RGB", (1000, 200), "white")
    image.paste("red", (0, 40, 1000, 60))  # 잘못 고르던 MCD916 행
    image.paste("blue", (0, 80, 1000, 100))  # 목표 MCDA23 행
    def locate(window, title, backend, tool_name, current_image, **kwargs):
        assert window is None and tool_name == "MCDA23"
        assert current_image is image  # 컬럼 strip 이 아니라 전체 List 이미지로 찾는다
        return {"full_image_point": {"x": 840, "y": 90}}, {"iters": []}
    monkeypatch.setattr(checker, "_locate_tool_via_vlm", locate)
    ocr_calls = []
    def ocr(**kwargs):
        assert kwargs["user_text"] == "OCR:"
        with Image.open(kwargs["image_path"]) as crop:
            red, green, blue = crop.convert("RGB").getpixel((20, 20))
            assert blue > 200 and red < 20  # y=90 밴드만 PaddleOCR로 전달
        ocr_calls.append(kwargs)
        return SimpleNamespace(text="MCDA23")
    responses = iter([
        json.loads('{"headers": [{"name": "Remote", "x": 500}, {"name": "RCS IP", "x": 700}, {"name": "MC ID", "x": 840}, {"name": "Control User", "x": 900}, {"name": "Connection User", "x": 970}]}'),
        {"mc_id": "MCDA23", "visible_mc_ids": ["MCDA23"], "row_confirmed": True,
         "remote_text": "1", "connection_user_text": "kim"},
    ])
    client = SimpleNamespace(chat_with_image_b64=lambda **kw: SimpleNamespace(text=json.dumps(next(responses))))
    report = checker.check_tool_occupancy(image, "MCDA23", client=client,
                                        ocr_client=SimpleNamespace(chat_with_image_path=ocr))
    assert report["occupancy"] == "occupied_by_other"
    assert report["diagnosis"] == "ok"
    assert report["row_point"] == {"x": 840, "y": 90}
    assert report["layout"]["row_top"] == 82
    assert len(ocr_calls) == 1


def test_columns_split_at_header_midpoints_and_scale_to_image_width():
    import pytest
    from poc.workflow_3.check_tool_occupancy import columns_from_headers

    headers = [{"name": "Remote", "x": 500}, {"name": "RCS IP", "x": 700},
               {"name": "MC ID", "x": 840}, {"name": "Control User", "x": 900},
               {"name": "Connection User", "x": 970}]
    assert columns_from_headers(headers, 2000) == {
        "remote": [0, 1200], "mc_id": [1540, 1740], "connection_user": [1870, 2000]}
    assert "control_user" not in columns_from_headers(headers, 2000)
    for bad in (None, [], [{"name": "MC ID"}], [{"name": 3, "x": 10}], [{"name": "MC ID", "x": "10"}]):
        with pytest.raises(ValueError):
            columns_from_headers(bad, 2000)
