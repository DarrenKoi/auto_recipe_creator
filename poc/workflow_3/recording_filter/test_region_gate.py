"""Stage 1.5 영역 게이트 - 세대 분할 / ambient 판정 / 사이드카 조인 테스트."""

import json

from poc.workflow_3.recording_filter.region_gate import (
    FrameMeta,
    assign_generations,
    gate_verdict,
    load_frame_meta,
    nearest_meta,
)


def _meta(t_sec, rect, occlusion="none", cursor_xy=None, cursor_in_window=False):
    return FrameMeta(
        t_sec=t_sec, rect=rect, occlusion=occlusion,
        cursor_xy=cursor_xy, cursor_in_window=cursor_in_window,
    )


_RECT_A = {"left": 0, "top": 0, "right": 1600, "bottom": 1000}
_RECT_B = {"left": 100, "top": 50, "right": 1700, "bottom": 1050}


def test_single_generation_when_rect_stable():
    """창이 안 움직이면 세대는 하나다."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_A), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 0, 0]


def test_new_generation_when_rect_changes():
    """창을 옮기면 그 시점부터 새 세대."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_B), _meta(0.4, _RECT_B)]
    assert assign_generations(metas) == [0, 1, 1]


def test_generation_increments_again_on_return():
    """원래 위치로 되돌아와도 새 세대다(지도 재검출이 필요하므로)."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, _RECT_B), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 1, 2]


def test_generation_ignores_missing_rect():
    """rect 가 없는 프레임은 직전 세대를 물려받는다(판정 불가로 쪼개지 않는다)."""
    metas = [_meta(0.0, _RECT_A), _meta(0.2, None), _meta(0.4, _RECT_A)]
    assert assign_generations(metas) == [0, 0, 0]


def test_generation_empty_input():
    assert assign_generations([]) == []


_LIVE_BOX = {"left": 400, "top": 200, "right": 1200, "bottom": 800}


def test_gate_ambient_when_change_inside_live_box_only():
    """라이브 박스 안에서만 변했고 커서가 밖이면 장비 자율 갱신."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, False, True) == "ambient"


def test_gate_candidate_when_change_touches_ui():
    """UI 영역에 걸치면 승격."""
    overlapping = {"left": 100, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(overlapping, _LIVE_BOX, False, True) == "candidate"


def test_gate_candidate_when_cursor_in_live_box():
    """커서가 라이브 박스 안이면 직접 조작 가능성이 있어 승격."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, True, True) == "candidate"


def test_gate_candidate_when_no_live_box():
    """박스를 못 찾은 세대는 게이트 없이 통과 - 오탐이 늘 뿐 데이터는 안 잃는다."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, None, False, True) == "candidate"


def test_gate_candidate_when_meta_missing():
    """사이드카가 없으면 커서 예외를 못 쓰므로 안전하게 전부 승격."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_verdict(inside, _LIVE_BOX, False, False) == "candidate"


def test_load_frame_meta_missing_file(tmp_path):
    """사이드카가 없으면 빈 목록(실패 아님)."""
    assert load_frame_meta(tmp_path) == []


def test_load_frame_meta_parses_lines(tmp_path):
    """JSONL 을 FrameMeta 로 읽는다. 깨진 줄은 건너뛴다."""
    lines = [
        json.dumps({"frame": "a", "t_sec": 0.0, "window_rect": _RECT_A,
                    "occlusion": "none", "cursor_screen_xy": [10, 20],
                    "cursor_in_window": True}),
        "{ broken json",
        json.dumps({"frame": "b", "t_sec": 0.4, "window_rect": _RECT_A,
                    "occlusion": "partial", "cursor_screen_xy": None,
                    "cursor_in_window": False}),
    ]
    (tmp_path / "frame_meta.jsonl").write_text("\n".join(lines), encoding="utf-8")
    metas = load_frame_meta(tmp_path)
    assert len(metas) == 2, metas
    assert metas[0].cursor_xy == [10, 20]
    assert metas[1].occlusion == "partial"


def test_nearest_meta_picks_closest_timestamp():
    """capture 순번과 파일 seq 가 어긋나므로 t_sec 최근접으로 조인한다."""
    metas = [_meta(0.0, _RECT_A), _meta(0.4, _RECT_A), _meta(1.0, _RECT_A)]
    assert nearest_meta(metas, 0.35).t_sec == 0.4
    assert nearest_meta(metas, 0.9).t_sec == 1.0
    assert nearest_meta([], 0.5) is None
