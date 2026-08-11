"""절차서 렌더러 테스트 - step 목록에서 한국어 markdown 을 만든다."""

from poc.workflow_3.workflow_extract.render import render_markdown

_SESSION = {"eqp_id": "MCD916", "tag": "20260811_150000",
            "capture_dir": "/x/recording", "total_events": 4, "duration_sec": 120.0}


def _step(seq, action, **kw):
    base = {
        "seq": seq, "action": action, "target": "PM", "target_kind": "ui_control",
        "value": None, "value_source": "none", "coords_in_live_box": None,
        "t_sec": [10.0, 12.0], "generation": 0, "grouping_rule": "R5",
        "inferred": False, "intent": None, "count": None, "raw_events": [seq],
        "frame": f"f_{seq}.jpg",
    }
    base.update(kw)
    return base


def test_renders_numbered_steps():
    md = render_markdown([_step(0, "click"), _step(1, "click", target="OK")], _SESSION)
    assert "1." in md and "2." in md
    assert "PM" in md and "OK" in md


def test_renders_typed_value():
    md = render_markdown(
        [_step(0, "type_text", target="Recipe Name", value="MCD916_A", value_source="ocr")],
        _SESSION,
    )
    assert "MCD916_A" in md


def test_marks_inferred_double_click():
    """추론된 더블클릭은 문서에서 추론임이 드러나야 한다."""
    md = render_markdown(
        [_step(0, "double_click", inferred=True, intent="fov_move",
               target_kind="live_image", target=None)],
        _SESSION,
    )
    assert "추론" in md


def test_footer_lists_limitations():
    """과신을 막기 위해 한계가 문서에 남아야 한다."""
    md = render_markdown([_step(0, "click")], _SESSION)
    for token in ("키 입력", "드래그", "스크롤"):
        assert token in md


def test_coverage_table_reports_rule_distribution():
    """어떤 규칙이 몇 개를 만들었는지 보여야 오작동 규칙을 지목할 수 있다."""
    md = render_markdown(
        [_step(0, "click", grouping_rule="R5"), _step(1, "click", grouping_rule="R5")],
        _SESSION,
    )
    assert "R5" in md


def test_empty_steps_still_renders_header():
    md = render_markdown([], _SESSION)
    assert "MCD916" in md


def test_typed_empty_value_is_shown_as_recovered_not_missing():
    """value="" 는 지워진 게 아니라 '복원된 빈 값' 이다 - falsy 체크로 숨기면 안 된다."""
    md = render_markdown(
        [_step(0, "type_text", target="Recipe Name", value="", value_source="ocr")],
        _SESSION,
    )
    assert "``" in md or "` `" in md or "`" in md
    # 값 부분이 실제로 렌더된 줄에 등장해야 한다(단순히 label 만 있는 게 아니라).
    lines = [l for l in md.splitlines() if "Recipe Name" in l]
    assert lines
    assert "->" in lines[0]


def test_seq_zero_renders_as_step_one():
    """seq=0 은 첫 step 이며 seq+1 로 '1.' 이 렌더돼야 한다(falsy 취급 금지)."""
    md = render_markdown([_step(0, "click")], _SESSION)
    assert "1. [10.0s]" in md
