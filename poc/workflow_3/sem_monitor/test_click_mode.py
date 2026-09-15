"""crosshair / L자 아이콘 모드 판별 - 초록 채움 색 판정과 unknown 규칙 (Mac, VLM 불필요)."""

from types import SimpleNamespace

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


def test_icon_strip_sits_right_of_sem_box_and_stays_inside_image():
    strip = cm.icon_strip_box({"left": 100, "top": 50, "right": 600, "bottom": 450}, (700, 500))
    assert strip == {"left": 600, "top": 34, "right": 690, "bottom": 466}
    edge = cm.icon_strip_box({"left": 100, "top": 0, "right": 680, "bottom": 500}, (700, 500))
    assert edge["right"] == 700 and edge["top"] == 0 and edge["bottom"] == 500


def _strip_with_icons(fills, bg=(235, 235, 235), icon=20, gap=12, width=60):
    """세로로 나란한 아이콘 열. fills[i] 가 i 번째 아이콘의 채움색."""
    strip = Image.new("RGB", (width, len(fills) * (icon + gap) + gap), bg)
    for i, fill in enumerate(fills):
        y = gap + i * (icon + gap)
        ImageDraw.Draw(strip).rectangle((18, y, 18 + icon - 1, y + icon - 1), fill=fill)
    return strip


def test_segment_icon_runs_finds_each_stacked_icon_in_order():
    grey = (120, 120, 120)
    runs = cm.segment_icon_runs(_strip_with_icons([grey, grey, (0, 200, 60), grey, grey, grey]))
    assert len(runs) == 6
    assert runs[2] == {"left": 18, "top": 12 + 2 * 32, "right": 38, "bottom": 12 + 2 * 32 + 20}
    assert all(runs[i]["bottom"] <= runs[i + 1]["top"] for i in range(5))
    assert cm.segment_icon_runs(Image.new("RGB", (60, 100), (235, 235, 235))) == []


def test_fixed_positions_pick_third_and_fifth_without_vlm(monkeypatch, tmp_path):
    grey = (120, 120, 120)
    image = Image.new("RGB", (400, 300), (235, 235, 235))
    sem_box = {"left": 20, "top": 20, "right": 300, "bottom": 280}
    # strip 은 x=300.., y=4.. ; 아이콘 6개 중 5번째(L자)만 초록
    image.paste(_strip_with_icons([grey, grey, grey, grey, (0, 200, 60), grey]), (310, 30))
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=sem_box))
    def no_vlm(*a, **kw):
        raise AssertionError("VLM must not be called when the icon column segments cleanly")
    monkeypatch.setattr(cm, "analyze_window_target", no_vlm)
    report = cm.detect_click_mode(image, client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "l_shape" and report["recenter_clicks"] == 1
    assert report["icons"]["crosshair"]["source"] == "segment"
    assert report["icons"]["crosshair"]["box"]["top"] == 30 + 12 + 2 * 32


def test_detect_click_mode_searches_the_strip_and_maps_boxes_back(monkeypatch, tmp_path):
    image = Image.new("RGB", (400, 300), (235, 235, 235))
    sem_box = {"left": 20, "top": 20, "right": 300, "bottom": 280}
    image.paste(_icon((0, 200, 60)), (310, 40))    # crosshair: 초록 (strip 안)
    image.paste(_icon((120, 120, 120)), (310, 80))  # L자: 회색
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=sem_box))
    strip_boxes = {"crosshair": {"left": 10, "top": 36, "right": 34, "bottom": 60},   # strip 좌표
                   "l_shape": {"left": 10, "top": 76, "right": 34, "bottom": 100}}

    def locate(window, title, backend, target, **kw):
        assert kw["image"].size == (90, 292)  # 박스 오른쪽 strip 만 넘긴다
        mode = "crosshair" if "crosshair" in target.key else "l_shape"
        b = strip_boxes[mode]
        return SimpleNamespace(exit_code="success", bbox=b,
                               point={"x": (b["left"] + b["right"]) // 2, "y": (b["top"] + b["bottom"]) // 2})
    monkeypatch.setattr(cm, "analyze_window_target", locate)
    report = cm.detect_click_mode(image, client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "crosshair" and report["recenter_clicks"] == 2
    assert report["icons"]["crosshair"]["source"] == "vlm"  # 아이콘 2개뿐 -> 분할 불신 -> VLM
    assert report["icons"]["crosshair"]["box"] == {"left": 310, "top": 40, "right": 334, "bottom": 64}
    assert (tmp_path / "strip.jpg").exists() and (tmp_path / "result.json").exists()


def test_without_sem_box_falls_back_to_whole_window(monkeypatch, tmp_path):
    image = Image.new("RGB", (400, 300), (235, 235, 235))
    image.paste(_icon((0, 200, 60)), (300, 20))
    image.paste(_icon((120, 120, 120)), (300, 60))
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=None))
    boxes = {"crosshair": {"left": 300, "top": 20, "right": 324, "bottom": 44},
             "l_shape": {"left": 300, "top": 60, "right": 324, "bottom": 84}}

    def locate(window, title, backend, target, **kw):
        assert kw["image"] is image
        b = boxes["crosshair" if "crosshair" in target.key else "l_shape"]
        return SimpleNamespace(exit_code="success", bbox=b, point={"x": 312, "y": 32})
    monkeypatch.setattr(cm, "analyze_window_target", locate)
    assert cm.detect_click_mode(image, client=object(), artifact_dir=tmp_path)["mode"] == "crosshair"


def test_locator_failure_yields_unknown_not_a_guess(monkeypatch, tmp_path):
    monkeypatch.setattr(cm, "detect_sem_box", lambda img, client: SimpleNamespace(bbox_px=None))
    monkeypatch.setattr(cm, "analyze_window_target",
                        lambda *a, **kw: SimpleNamespace(exit_code="refusal", bbox=None, point=None))
    report = cm.detect_click_mode(Image.new("RGB", (100, 100)), client=object(), artifact_dir=tmp_path)
    assert report["mode"] == "unknown" and report["recenter_clicks"] is None
