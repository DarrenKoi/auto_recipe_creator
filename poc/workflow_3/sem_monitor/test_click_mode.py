"""crosshair / L자 아이콘 모드 판별 - DDS/|| 앵커 기하 + 초록 채움 판정 (Mac, VLM 불필요)."""

from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from poc.workflow_3.sem_monitor import click_mode as cm


def _icon(fill, size=24):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    ImageDraw.Draw(img).rectangle((4, 4, size - 5, size - 5), fill=fill)
    return img


def test_green_ratio_separates_filled_from_grey_icons():
    assert cm.green_ratio(_icon((0, 200, 60))) > 0.4
    assert cm.green_ratio(_icon((120, 120, 120))) == 0.0
    assert cm.green_ratio(_icon((30, 60, 200))) == 0.0   # 파랑은 초록이 아니다
    assert cm.green_ratio(Image.new("RGB", (0, 0))) == 0.0


def test_classify_mode_requires_a_clear_winner():
    assert cm.classify_mode({"crosshair": 0.5, "l_shape": 0.0}) == "crosshair"
    assert cm.classify_mode({"crosshair": 0.02, "l_shape": 0.4}) == "l_shape"
    assert cm.classify_mode({"crosshair": 0.4, "l_shape": 0.35}) == "unknown"  # 둘 다 초록
    assert cm.classify_mode({"crosshair": 0.0, "l_shape": 0.0}) == "unknown"   # 둘 다 아님
    assert cm.classify_mode({"crosshair": 0.5}) == "unknown"                    # 한쪽 미검출


def test_button_order_from_bottom_matches_user_report():
    assert cm.BUTTONS_FROM_BOTTOM[0] == "DDS" and cm.BUTTONS_FROM_BOTTOM[-1] == "pipes"
    assert cm.BUTTONS_FROM_BOTTOM[-2] == "equals"
    assert cm.BUTTONS_FROM_BOTTOM.index("l_shape") == 8
    assert cm.BUTTONS_FROM_BOTTOM.index("crosshair") == 10
    assert cm.ANCHOR_GAP == 12


def test_column_boxes_count_pitch_upward_from_dds_and_check_with_equals():
    dds = {"left": 300, "top": 400, "right": 340, "bottom": 420}     # cy=410
    top = {"left": 300, "top": 160, "right": 340, "bottom": 180}     # cy=170 -> pitch 20
    eq = {"left": 300, "top": 180, "right": 340, "bottom": 200}      # cy=190 = top + pitch
    boxes = cm.column_boxes(dds, top, eq)
    assert len(boxes) == 13
    cross, lshape = boxes["crosshair"], boxes["l_shape"]
    assert (cross["top"] + cross["bottom"]) / 2 == 410 - 10 * 20
    assert (lshape["top"] + lshape["bottom"]) / 2 == 410 - 8 * 20
    assert cross["left"] == 300 and cross["right"] == 340
    assert cross["bottom"] - cross["top"] < 20  # 이웃 버튼을 물지 않는다
    assert cm.column_boxes(dds, top) == boxes  # '=' 없이도 같은 기하
    with pytest.raises(ValueError):
        cm.column_boxes(dds, dds)                         # pitch 0
    with pytest.raises(ValueError):
        cm.column_boxes(top, dds)                         # 뒤집힘
    with pytest.raises(ValueError):                       # '||' 를 한 칸 아래 버튼으로 잘못 잡음
        cm.column_boxes(dds, {"left": 300, "top": 180, "right": 340, "bottom": 200}, eq)


def test_strip_spans_full_window_height_right_of_sem_box():
    strip = cm.icon_strip_box({"left": 100, "top": 50, "right": 600, "bottom": 450}, (700, 500))
    assert strip == {"left": 600, "top": 0, "right": 690, "bottom": 500}


def _column_image(active, pitch=20, dds_cy=440, x=310):
    """13개 버튼 열이 그려진 창 이미지. active 버튼만 초록."""
    image = Image.new("RGB", (400, 480), (235, 235, 235))
    for k, name in enumerate(cm.BUTTONS_FROM_BOTTOM):
        cy = dds_cy - k * pitch
        fill = (0, 200, 60) if name == active else (120, 120, 120)
        ImageDraw.Draw(image).rectangle((x, cy - 7, x + 30, cy + 7), fill=fill)
    return image


def _fake_pipeline(monkeypatch, *, dds_ok=True, top_offset=0, pitch=20, dds_cy=440):
    sem_box = {"left": 20, "top": 20, "right": 300, "bottom": 460}
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=sem_box))
    centers = {"dds": dds_cy, "pipes": dds_cy - 12 * pitch + top_offset, "equals": dds_cy - 11 * pitch}
    ocr_calls = []

    def locate(window, title, backend, target, **kw):
        assert kw["image"].size[1] == 480  # strip 은 창 전체 높이
        name = target.key.rsplit("_", 1)[1]
        cy = centers[name]
        return SimpleNamespace(exit_code="success", point={"x": 25, "y": cy},
                               bbox={"left": 10, "top": cy - 8, "right": 40, "bottom": cy + 8})
    monkeypatch.setattr(cm, "analyze_window_target", locate)

    def ocr(img, box, **kw):
        ocr_calls.append(kw["timestamp_tag"])
        assert kw["timestamp_tag"] == "dds"  # 기호 버튼은 OCR 하지 않는다
        assert box["right"] - box["left"] >= 30 + 2 * 12 and box["bottom"] - box["top"] >= 16 + 2 * 12  # 넉넉한 crop
        text = "DDS" if dds_ok else "ODS"
        return SimpleNamespace(ok=True, raw_text=text, tokens=[text])
    monkeypatch.setattr(cm, "read_text_near_point", ocr)
    return ocr_calls


def test_detect_reads_green_at_anchored_positions(monkeypatch, tmp_path):
    ocr_calls = _fake_pipeline(monkeypatch)
    report = cm.detect_click_mode(_column_image("l_shape"), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "l_shape" and report["recenter_clicks"] == 1
    assert ocr_calls == ["dds"]
    assert report["icons"]["crosshair"]["steps_above_dds"] == 10
    assert (tmp_path / "column.jpg").exists() and (tmp_path / "result.json").exists()
    report = cm.detect_click_mode(_column_image("crosshair"), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "crosshair" and report["recenter_clicks"] == 2


def test_anchor_label_mismatch_yields_unknown(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch, dds_ok=False)
    report = cm.detect_click_mode(_column_image("crosshair"), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "unknown" and report["anchors"]["DDS"] is None


def test_top_anchor_on_wrong_button_fails_pitch_check(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch, top_offset=20)  # '||' 대신 '=' 를 잡은 상황
    report = cm.detect_click_mode(_column_image("crosshair"), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "unknown" and "column" not in report


def test_missing_sem_box_yields_unknown_without_locating(monkeypatch, tmp_path):
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=None))

    def no_locate(*a, **kw):
        raise AssertionError("no locate without box")
    monkeypatch.setattr(cm, "analyze_window_target", no_locate)
    report = cm.detect_click_mode(Image.new("RGB", (100, 100)), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "unknown"
