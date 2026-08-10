"""Stage 1.5 영역 게이트 - 세대 분할 / ambient 판정 / 사이드카 조인 테스트."""

import json

from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.region_gate import (
    META_MAX_JOIN_GAP_SEC,
    FrameMeta,
    RegionMap,
    apply_region_gate,
    assign_generations,
    gate_region,
    gate_verdict,
    load_frame_meta,
    nearest_meta,
    screen_point_to_frame,
)


def _meta(t_sec, rect, occlusion="none", cursor_xy=None, cursor_in_window=False):
    return FrameMeta(
        t_sec=t_sec, rect=rect, occlusion=occlusion,
        cursor_xy=cursor_xy, cursor_in_window=cursor_in_window,
    )


def _event(t_sec, change_bbox, rank=0):
    """테스트용 ChangeEvent - 경로는 존재할 필요 없다(apply_region_gate 는 열지 않는다)."""
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/region_gate_test_{rank}.jpg",
        prev_frame_path=f"/tmp/region_gate_test_prev_{rank}.jpg",
        timestamp_sec=t_sec, frame_index=rank, change_bbox=change_bbox,
        largest_blob_area_px=1000, changed_pixels=1000,
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


# ---------------------------------------------------------------------------
# 2026-08-10 리뷰 FINDING 1 - 커서 위치를 못 옮기면(rect 없음) "밖" 으로 단정하지
# 않고 candidate 로 강제해야 한다.
# ---------------------------------------------------------------------------

def test_apply_region_gate_forces_candidate_when_rect_missing_but_cursor_present():
    """cursor_xy 는 있는데 rect 가 없어 프레임 좌표로 옮길 수 없으면 candidate 다."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    metas = [_meta(0.0, None, cursor_xy=[9999, 9999])]
    region_maps = {0: RegionMap(generation=0, live_box=_LIVE_BOX)}
    events = [_event(0.0, inside)]

    results = apply_region_gate(events, metas, region_maps)

    assert len(results) == 1
    _event_out, generation, verdict, _occlusion, _region = results[0]
    assert generation == 0
    assert verdict == "candidate"


# ---------------------------------------------------------------------------
# 2026-08-10 리뷰 FINDING 2 - 최근접 조인에 최대 거리 상한을 둔다. 상한을 넘으면
# 사이드카가 없는 것과 동일하게 취급해야(죽은 writer 의 마지막 레코드에 세션 끝까지
# 계속 join 되는 것을 막는다) 한다.
# ---------------------------------------------------------------------------

def test_nearest_meta_none_when_beyond_max_gap():
    """가장 가까운 레코드라도 상한보다 멀면 조인하지 않는다."""
    metas = [_meta(0.0, _RECT_A), _meta(1.0, _RECT_A)]
    assert nearest_meta(metas, 1.0 + META_MAX_JOIN_GAP_SEC + 1.0) is None


def test_nearest_meta_returns_record_just_inside_max_gap():
    """상한 바로 안쪽이면 정상적으로 조인된다(정상 idle 간격을 오탐하지 않는다)."""
    metas = [_meta(0.0, _RECT_A)]
    t_sec = META_MAX_JOIN_GAP_SEC - 0.01
    result = nearest_meta(metas, t_sec)
    assert result is not None
    assert result.t_sec == 0.0


def test_apply_region_gate_stale_sidecar_forces_candidate_and_generation_fallback():
    """사이드카 마지막 기록보다 훨씬 뒤의 이벤트는 낡은 레코드/세대에 join 하지 않는다.

    세대가 도중에 바뀐(RECT_A -> RECT_B) 세션에서 writer 가 죽었다고 가정한다.
    상한을 넘긴 이벤트는 has_meta=False 로 candidate 강제되고, 세대도 죽기 직전의
    마지막 세대(1) 가 아니라 사이드카가 비었을 때와 같은 0 으로 폴백해야 한다.
    """
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    metas = [
        _meta(0.0, _RECT_A, cursor_xy=[10, 10]),
        _meta(0.5, _RECT_B, cursor_xy=[10, 10]),  # 여기서 세대 1 로 넘어간 뒤 writer 사망 가정.
    ]
    region_maps = {
        0: RegionMap(generation=0, live_box=_LIVE_BOX),
        1: RegionMap(generation=1, live_box=_LIVE_BOX),
    }
    far_t = 0.5 + META_MAX_JOIN_GAP_SEC + 5.0
    events = [_event(far_t, inside)]

    results = apply_region_gate(events, metas, region_maps)

    assert len(results) == 1
    _event_out, generation, verdict, occlusion, _region = results[0]
    assert generation == 0, "죽기 직전 세대(1)에 계속 join 하면 안 되고 폴백(0)해야 한다"
    assert verdict == "candidate"
    assert occlusion == "unknown"


# ---------------------------------------------------------------------------
# 2026-08-10 최종 리뷰 FINDING 2 - 화면 좌표 -> 프레임 좌표 변환은 DPI 배율을
# 보정해야 한다. 보정 전에는 라이브 박스 왼쪽/위쪽 대역에서 커서가 "박스 밖" 으로
# 오판돼 진짜 조작이 ambient 로 조용히 폐기됐다.
# ---------------------------------------------------------------------------

# rect 1600x1000 (논리 픽셀) 기준. 프레임은 배율에 따라 커진다.
_SCALE_RECT = {"left": 0, "top": 0, "right": 1600, "bottom": 1000}
# 라이브 박스는 프레임 픽셀 공간이다.
_SCALE_LIVE_BOX = {"left": 1000, "top": 200, "right": 1900, "bottom": 1100}


def test_screen_point_to_frame_is_identity_at_100_percent():
    """100% 배율(프레임 크기 == rect 크기)에서는 단순 뺄셈과 같다(무영향)."""
    point = screen_point_to_frame([1100, 300], _SCALE_RECT, (1600, 1000))
    assert point == (1100.0, 300.0), point


def test_screen_point_to_frame_corrects_125_percent_scaling():
    """125% 배율: 프레임 x=1100 인 커서는 rect 공간에서 880 이다.

    리뷰가 든 구체적 실패 - 보정 없이 880 을 그대로 쓰면 live_box(left=1000)
    밖으로 판정돼 라이브 영상 안 조작이 ambient 로 버려진다.
    """
    frame_wh = (2000, 1250)          # 1600x1000 * 1.25
    point = screen_point_to_frame([880, 240], _SCALE_RECT, frame_wh)
    assert point == (1100.0, 300.0), point
    fx, fy = point
    assert _SCALE_LIVE_BOX["left"] <= fx <= _SCALE_LIVE_BOX["right"]
    assert _SCALE_LIVE_BOX["top"] <= fy <= _SCALE_LIVE_BOX["bottom"]


def test_screen_point_to_frame_corrects_150_percent_scaling():
    """150% 배율에서도 프레임 좌표를 정확히 복원한다."""
    frame_wh = (2400, 1500)          # 1600x1000 * 1.5
    point = screen_point_to_frame([1000, 400], _SCALE_RECT, frame_wh)
    assert point == (1500.0, 600.0), point


def test_screen_point_to_frame_honors_rect_origin():
    """rect 원점이 (0,0) 이 아니어도 원점 이동 후 배율을 적용한다."""
    rect = {"left": 100, "top": 50, "right": 1700, "bottom": 1050}
    point = screen_point_to_frame([980, 290], rect, (2000, 1250))
    assert point == (1100.0, 300.0), point


def test_screen_point_to_frame_none_on_degenerate_rect():
    """rect 가 0/음수 크기이거나 프레임 크기를 모르면 판정 불가(None)."""
    assert screen_point_to_frame([10, 10], {"left": 0, "top": 0, "right": 0, "bottom": 0}, (100, 100)) is None
    assert screen_point_to_frame([10, 10], {"left": 100, "top": 0, "right": 0, "bottom": 100}, (100, 100)) is None
    assert screen_point_to_frame([10, 10], _SCALE_RECT, (0, 0)) is None
    assert screen_point_to_frame([10, 10], _SCALE_RECT, None) is None
    assert screen_point_to_frame(None, _SCALE_RECT, (100, 100)) is None


def test_apply_region_gate_keeps_cursor_inside_live_box_under_125_percent():
    """125% 배율에서 라이브 박스 안쪽 커서를 "밖" 으로 오판하지 않는다.

    보정 전 코드는 fx=880 < live_box.left=1000 으로 봐서 verdict=ambient 를 냈고
    (변화가 박스 안에만 있으므로) 이 이벤트를 통째로 버렸다.
    """
    change_inside = {"left": 1100, "top": 300, "right": 1400, "bottom": 600}
    metas = [_meta(0.0, _SCALE_RECT, cursor_xy=[880, 240])]
    region_maps = {0: RegionMap(generation=0, live_box=_SCALE_LIVE_BOX)}
    events = [_event(0.0, change_inside)]

    results = apply_region_gate(
        events, metas, region_maps, frame_size_fn=lambda _path: (2000, 1250)
    )

    _event_out, _generation, verdict, _occlusion, region = results[0]
    assert verdict == "candidate", "라이브 영상 직접 조작 가능성이라 승격해야 한다"
    assert region == "live_image"


def test_apply_region_gate_forces_candidate_when_frame_size_unreadable():
    """프레임 크기를 못 읽으면 커서를 "밖" 으로 단정하지 않고 candidate 로 강제한다."""
    change_inside = {"left": 1100, "top": 300, "right": 1400, "bottom": 600}
    metas = [_meta(0.0, _SCALE_RECT, cursor_xy=[880, 240])]
    region_maps = {0: RegionMap(generation=0, live_box=_SCALE_LIVE_BOX)}
    events = [_event(0.0, change_inside)]

    results = apply_region_gate(
        events, metas, region_maps, frame_size_fn=lambda _path: None
    )

    _event_out, _generation, verdict, _occlusion, _region = results[0]
    assert verdict == "candidate"


# ---------------------------------------------------------------------------
# 2026-08-10 최종 리뷰 FINDING 3 - region 은 verdict 파생이 아니라 기하 값이다.
# ---------------------------------------------------------------------------

def test_gate_region_live_image_when_change_contained():
    """변화가 라이브 박스 안에만 있으면 live_image."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_region(inside, _LIVE_BOX, False) == "live_image"


def test_gate_region_live_image_when_cursor_in_live_box():
    """커서가 라이브 박스 안이면(= ambient 예외로 살아남은 클릭) live_image."""
    overlapping = {"left": 100, "top": 300, "right": 700, "bottom": 500}
    assert gate_region(overlapping, _LIVE_BOX, True) == "live_image"


def test_gate_region_ui_when_change_outside_live_box():
    """라이브 박스가 있는데 변화도 커서도 밖이면 ui."""
    outside = {"left": 10, "top": 10, "right": 100, "bottom": 100}
    assert gate_region(outside, _LIVE_BOX, False) == "ui"


def test_gate_region_unknown_without_live_box():
    """지도를 못 만든 세대는 region 도 unknown 이다(ui 로 단정하지 않는다)."""
    inside = {"left": 500, "top": 300, "right": 700, "bottom": 500}
    assert gate_region(inside, None, False) == "unknown"
