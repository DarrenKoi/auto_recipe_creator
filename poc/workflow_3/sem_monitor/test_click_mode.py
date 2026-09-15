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


def test_detect_click_mode_reads_color_at_located_icons(monkeypatch, tmp_path):
    image = Image.new("RGB", (400, 300), (235, 235, 235))
    image.paste(_icon((0, 200, 60)), (300, 20))    # crosshair: 초록
    image.paste(_icon((120, 120, 120)), (300, 60))  # L자: 회색
    boxes = {"crosshair": {"left": 300, "top": 20, "right": 324, "bottom": 44},
             "l_shape": {"left": 300, "top": 60, "right": 324, "bottom": 84}}

    def locate(window, title, backend, target, **kw):
        assert window is None and kw["image"] is image
        mode = "crosshair" if "crosshair" in target.key else "l_shape"
        b = boxes[mode]
        return SimpleNamespace(exit_code="success", bbox=b,
                               point={"x": (b["left"] + b["right"]) // 2, "y": (b["top"] + b["bottom"]) // 2})
    monkeypatch.setattr(cm, "analyze_window_target", locate)
    report = cm.detect_click_mode(image, artifact_dir=tmp_path)
    assert report["mode"] == "crosshair" and report["recenter_clicks"] == 2
    assert (tmp_path / "result.json").exists()


def test_locator_failure_yields_unknown_not_a_guess(monkeypatch, tmp_path):
    monkeypatch.setattr(cm, "analyze_window_target",
                        lambda *a, **kw: SimpleNamespace(exit_code="refusal", bbox=None, point=None))
    report = cm.detect_click_mode(Image.new("RGB", (100, 100)), artifact_dir=tmp_path)
    assert report["mode"] == "unknown" and report["recenter_clicks"] is None
