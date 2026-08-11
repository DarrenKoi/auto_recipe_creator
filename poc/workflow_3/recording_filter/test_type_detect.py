"""Stage 2b 타이핑 구간 탐지 테스트 - 커서 정지 + 국소 반복 변화."""

from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.region_gate import FrameMeta
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.recording_filter.type_detect import (
    TypingBurst,
    find_typing_bursts,
    resolve_typing_events,
)

_RECT = {"left": 0, "top": 0, "right": 800, "bottom": 500}
_FIELD = {"left": 100, "top": 100, "right": 300, "bottom": 130}


def _ev(rank, t_sec, bbox=None):
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/t_{rank}.jpg", prev_frame_path=f"/tmp/t_prev_{rank}.jpg",
        timestamp_sec=t_sec, frame_index=rank, change_bbox=bbox or dict(_FIELD),
        # Stage 1 의 생존 조건은 최대 blob 면적 >= min_change_area_px(기본 5000, 1280폭
        # diff 공간)다 - 예전 픽스처의 500 은 Stage 1 이 애초에 내보낼 수 없는 값이라
        # "이 정도 작은 변화도 구간이 된다" 는 잘못된 인상을 준다(Stage 2b 는 이 필드를
        # 읽지 않으므로 판정에는 영향이 없다).
        largest_blob_area_px=9000, changed_pixels=9000,
    )


def _meta(t_sec, cursor_xy):
    return FrameMeta(
        t_sec=t_sec, rect=_RECT, occlusion="none",
        cursor_xy=cursor_xy, cursor_in_window=True,
    )


def _frame_size(_path):
    """프레임 픽셀 크기 주입 - 커서(화면 좌표) -> 프레임 좌표 변환에 필요하다.

    국소성 가드(2026-08-11 리뷰 C2)가 필드 기준점을 프레임 좌표로 잡으므로, 실제
    JPEG 없이 도는 단위 테스트도 크기를 알려줘야 한다. _RECT 와 같은 크기를 주면
    배율 1.0 이라 커서 화면 좌표가 그대로 프레임 좌표가 된다.
    """
    return (800, 500)


def test_finds_burst_when_cursor_still_and_change_localized():
    """커서가 멈춘 채 같은 영역이 4회 바뀌면 타이핑 구간 1개."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(4)]
    bursts = find_typing_bursts(events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size)
    assert len(bursts) == 1
    assert bursts[0].ranks == [0, 1, 2, 3]


def test_no_burst_when_cursor_moves():
    """커서가 움직이면 타이핑이 아니다(마우스 조작 중 화면 변화)."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200 + i * 50, 200]) for i in range(4)]
    assert find_typing_bursts(events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size) == []


def test_no_burst_below_min_events():
    """2건짜리 변화는 구간으로 인정하지 않는다(기본 임계 3)."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(2)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(2)]
    assert find_typing_bursts(events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size) == []


def test_idle_gap_splits_bursts():
    """변화가 idle 상한을 넘게 끊기면 별개 구간이 된다."""
    times = [10.0, 10.3, 10.6, 30.0, 30.3, 30.6]
    events = [_ev(i, t) for i, t in enumerate(times)]
    metas = [_meta(t, [200, 200]) for t in times]
    bursts = find_typing_bursts(events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size)
    assert [b.ranks for b in bursts] == [[0, 1, 2], [3, 4, 5]]


def test_no_burst_without_sidecar():
    """사이드카가 없으면 커서 정지를 알 수 없으므로 구간을 만들지 않는다."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    assert find_typing_bursts(events, [], RecordingFilterSettings(), frame_size_fn=_frame_size) == []


def test_roi_is_union_of_change_boxes():
    """구간 ROI 는 구성 change_bbox 의 합집합이어야 한다(글자가 오른쪽으로 늘어난다)."""
    boxes = [
        {"left": 100, "top": 100, "right": 150, "bottom": 130},
        {"left": 140, "top": 100, "right": 200, "bottom": 130},
        {"left": 190, "top": 100, "right": 260, "bottom": 130},
    ]
    events = [_ev(i, 10.0 + i * 0.3, boxes[i]) for i in range(3)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(3)]
    burst = find_typing_bursts(events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size)[0]
    assert burst.roi == {"left": 100, "top": 100, "right": 260, "bottom": 130}


class _StubOCR:
    """구간 시작/끝에 서로 다른 텍스트를 돌려주는 OCR 스텁."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def chat_with_image_b64(self, **kwargs):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1

        class _R:
            pass

        r = _R()
        r.text = '[{"text": "%s", "box": [0, 0, 10, 10]}]' % text if text else "[]"
        return r


def _burst(ranks=(0, 1, 2), start=10.0, end=12.0):
    return TypingBurst(
        ranks=list(ranks), start_t_sec=start, end_t_sec=end,
        roi={"left": 100, "top": 100, "right": 300, "bottom": 130},
        cursor_xy=[200, 200], frame_path="/tmp/t_0.jpg", end_frame_path="/tmp/t_2.jpg",
        anchor_xy=[200.0, 115.0], anchor_source="cursor", cursor_frame_xy=[200.0, 200.0],
    )


def _fake_loader(path):
    from PIL import Image

    return Image.new("RGB", (800, 500), "white")


def _click_event_for_focus(t_sec, rank=99):
    """포커스 클릭 역할의 ClickEvent 를 만든다(기본 rank=99, 라벨은 labels dict 로 전달)."""
    from poc.workflow_3.recording_filter.click_detect import ClickEvent

    change = _ev(rank, t_sec)
    return ClickEvent(
        change=change, is_click=True, status="click", cursor_visible=True,
        cursor_kind="sidecar", cursor_bbox=None, cursor_xy=[200, 110],
        click_window=None, changed_in_window_px=2000, confidence=1.0,
        evidence="", cursor_source="sidecar",
    )


def test_typing_event_recovers_value_from_ocr():
    """구간 끝 OCR 텍스트가 값이 된다."""
    ocr = _StubOCR(["", "MCD916_ALIGN_02"])
    events, _ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert len(events) == 1
    assert events[0]["action"] == "type_text"
    assert events[0]["text"] == "MCD916_ALIGN_02"
    assert events[0]["element_source"] == "ocr"


def test_caret_blink_rejected_when_text_unchanged():
    """시작/끝 텍스트가 같으면 캐럿 깜빡임이므로 이벤트를 만들지 않는다."""
    ocr = _StubOCR(["same", "same"])
    assert resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    ) == ([], set())


def test_focus_click_supplies_target_label():
    """구간 직전 클릭이 필드 이름을 준다.

    라벨은 ClickEvent 가 아니라 Stage 2c 의 labels dict(rank -> ElementLabel)에
    들어 있으므로 별도 인자로 넘긴다.
    """
    from poc.workflow_3.recording_filter.element_label import ElementLabel

    ocr = _StubOCR(["", "value"])
    click = _click_event_for_focus(t_sec=9.0)
    events, _ranks = resolve_typing_events(
        [_burst()], [click], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
        labels={99: ElementLabel(text="Recipe Name", source="ocr", confidence=1.0)},
    )
    assert events[0]["element"] == "Recipe Name"


def test_focus_click_uses_latest_qualifying_click():
    """포커스 창 안에 클릭이 둘이면 더 늦은(구간에 더 가까운) 클릭의 라벨이 이긴다.

    최초 클릭을 고르는 회귀가 있어도 클릭이 하나뿐인 테스트로는 드러나지 않으므로,
    서로 다른 시각의 두 후보를 넣어 "가장 늦은 클릭"을 실제로 검증한다.
    """
    from poc.workflow_3.recording_filter.element_label import ElementLabel

    ocr = _StubOCR(["", "value"])
    earlier = _click_event_for_focus(t_sec=8.0, rank=98)
    later = _click_event_for_focus(t_sec=9.5, rank=99)
    events, _ranks = resolve_typing_events(
        [_burst()], [earlier, later], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
        labels={
            98: ElementLabel(text="Wrong Field", source="ocr", confidence=1.0),
            99: ElementLabel(text="Recipe Name", source="ocr", confidence=1.0),
        },
    )
    assert events[0]["element"] == "Recipe Name"


def test_target_none_when_no_focus_click():
    """Tab 포커스 등으로 직전 클릭이 없으면 target 은 null 이다(추측 금지)."""
    ocr = _StubOCR(["", "value"])
    events, _ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert events[0]["element"] is None
    assert events[0]["text"] == "value"


def test_ocr_failure_yields_event_without_value():
    """OCR 이 던져도 구간은 남긴다(값만 비운다)."""

    class _Boom:
        def chat_with_image_b64(self, **kwargs):
            raise RuntimeError("ocr down")

    events, _ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=_Boom(), image_loader=_fake_loader,
    )
    assert events[0]["text"] is None
    assert events[0]["element_source"] == "none"


def test_both_reads_empty_still_emits_event_without_value():
    """양쪽 OCR 판독이 다 빈 문자열이면 캐럿이 아니라 OCR 실패로 취급해 남긴다.

    before == after 만으로 캐럿을 판정하면 '둘 다 빈 문자열'도 같은 취급을 받아
    조용히 버려진다 - 그러나 이는 ROI 정렬 오류/판독 실패일 수 있고, OCR 이
    값 복원의 유일한 경로이므로 이 실패를 캐럿과 혼동하면 안 된다.
    """
    ocr = _StubOCR(["", ""])
    events, _ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert len(events) == 1
    assert events[0]["text"] is None
    assert events[0]["element_source"] == "none"


def test_both_reads_nonempty_and_equal_still_rejected_as_caret():
    """양쪽이 비어 있지 않고 같으면 여전히 캐럿 깜빡임으로 버려야 한다(회귀 가드)."""
    ocr = _StubOCR(["same", "same"])
    assert resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    ) == ([], set())


def test_ocr_failure_event_has_unknown_target_kind():
    """OCR 실패 이벤트는 다른 장비로 이식 가능한지 알 수 없으므로 target_kind=unknown.

    element_source 가 ocr/vlm 이 아닐 때만 ui_control 이 아닌 unknown 을 돌려주는
    timeline.derive_target_kind 규칙을 이 모듈도 따라야 한다(하드코딩 금지).
    """

    class _Boom:
        def chat_with_image_b64(self, **kwargs):
            raise RuntimeError("ocr down")

    events, _ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=_Boom(), image_loader=_fake_loader,
    )
    assert events[0]["target_kind"] == "unknown"


# ---------------------------------------------------------------------------
# 2026-08-11 최종 리뷰 C2/E1 - 국소성 결속, 커서 드리프트, 빈 값 보존.
# ---------------------------------------------------------------------------

def test_no_burst_when_change_is_far_from_cursor():
    """커서가 멈춰 있어도 멀리서 반복되는 변화는 타이핑이 아니다(리뷰 C2).

    진행률/상태 패널 리페인트가 정확히 이 신호를 낸다 - 결속이 없으면 OCR 이
    그 패널의 숫자를 "입력값" 으로 복원해 확신에 찬 허구 step 을 만든다.
    """
    far = {"left": 600, "top": 400, "right": 700, "bottom": 450}
    events = [_ev(i, 10.0 + i * 0.3, dict(far)) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(4)]
    assert find_typing_bursts(
        events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size
    ) == []


def test_no_burst_when_anchor_cannot_be_established():
    """포커스 클릭도 없고 커서를 프레임 좌표로 옮길 수도 없으면 구간을 만들지 않는다.

    기준점이 없으면 그 값이 화면 어디에서 왔는지 보증할 수 없는데, 값은 절차서에
    confidence=1.0 으로 실린다 - 근거 없는 구간은 발행하지 않는 편이 낫다.
    """
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(4)]
    assert find_typing_bursts(
        events, metas, RecordingFilterSettings(), frame_size_fn=lambda _p: None
    ) == []


def test_focus_click_coords_anchor_the_field_even_when_cursor_moved_away():
    """포커스 클릭 좌표가 1순위 기준점이다(스펙 5.3: 필드 ROI = 클릭 좌표 주변).

    커서를 필드에서 조금 옮겨도(여전히 still_px 안) 클릭한 필드 근처의 변화는
    타이핑으로 남아야 한다.
    """
    events = [_ev(i, 10.0 + i * 0.3) for i in range(3)]
    metas = [_meta(10.0 + i * 0.3, [700, 450]) for i in range(3)]   # 커서는 필드에서 멀다
    click = _click_event_for_focus(t_sec=9.5)                       # cursor_xy=[200, 110]
    bursts = find_typing_bursts(
        events, metas, RecordingFilterSettings(),
        click_events=[click], frame_size_fn=_frame_size,
    )
    assert len(bursts) == 1
    assert bursts[0].anchor_source == "focus_click"
    assert bursts[0].anchor_xy == [200.0, 110.0]


def test_cursor_drift_is_measured_from_burst_start_not_previous_event():
    """스텝마다 조금씩 움직이는 커서는 "정지" 가 아니다(리뷰 C2 에 흡수된 드리프트 항목).

    한 스텝 5px(still_px=8 미만)씩 5회면 직전 비교로는 영원히 정지지만, 시작
    기준으로는 20px 를 이동했다 - 그 사이 ROI 도 함께 커져 다른 필드의 변화까지
    한 구간으로 빨아들인다.
    """
    events = [_ev(i, 10.0 + i * 0.3) for i in range(5)]
    metas = [_meta(10.0 + i * 0.3, [200 + i * 5, 200]) for i in range(5)]
    bursts = find_typing_bursts(
        events, metas, RecordingFilterSettings(), frame_size_fn=_frame_size
    )
    # 시작에서 8px 를 넘는 순간(3번째 이벤트, 누적 10px) 구간이 끊겨 최소 길이 3을
    # 채우지 못한다. 직전 비교였다면 5건이 한 구간으로 묶였다.
    assert [b.ranks for b in bursts] == [], bursts


def test_roi_union_area_cap_splits_runaway_burst():
    """ROI 합집합 면적 상한을 넘기면 같은 구간으로 이어붙이지 않는다."""
    boxes = [
        {"left": 100, "top": 100, "right": 150, "bottom": 130},
        {"left": 140, "top": 100, "right": 200, "bottom": 130},
        {"left": 190, "top": 100, "right": 260, "bottom": 130},
    ]
    events = [_ev(i, 10.0 + i * 0.3, boxes[i]) for i in range(3)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(3)]
    tight = RecordingFilterSettings(typing_roi_max_area_px=3000)   # 첫 두 개만 담긴다
    assert find_typing_bursts(events, metas, tight, frame_size_fn=_frame_size) == []


def test_cleared_field_keeps_empty_string_value_end_to_end():
    """지워진 필드(OCR 판독 성공, 결과 "")는 값 없음(None)이 아니라 빈 문자열이다.

    (리뷰 E1) `after or None` 이면 value=null 인데 value_source 는 "ocr" 로 남아
    스펙 8(값이 null 이면 출처도 none)과 모순되고, render/grouping 이 이미 갖춘
    "빈 문자열 보존" 처리가 도달 불가능한 죽은 코드가 된다. Stage 2b ->
    build_timeline -> group_events -> markdown 사슬 끝까지 확인한다.
    """
    from poc.workflow_3.recording_filter.timeline import build_timeline
    from poc.workflow_3.workflow_extract.grouping import GroupingContext, group_events
    from poc.workflow_3.workflow_extract.render import render_markdown
    from poc.workflow_3.workflow_extract.settings import WorkflowExtractSettings

    ocr = _StubOCR(["MCD916", ""])
    events, ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert events[0]["text"] == "", events[0]
    assert events[0]["element_source"] == "ocr", events[0]
    assert ranks == {0, 1, 2}, ranks

    timeline = build_timeline([], events)
    steps = group_events(timeline, GroupingContext(settings=WorkflowExtractSettings()))
    assert steps[0]["value"] == "", steps[0]
    assert steps[0]["value_source"] == "ocr", steps[0]
    assert "->" in render_markdown(steps, {"duration_sec": 12.0})


def test_consumed_ranks_excludes_caret_rejected_bursts():
    """캐럿으로 버린 구간의 rank 는 소비 목록에 들어가면 안 된다.

    들어가면 그 프레임의 진짜 클릭이 타임라인에서 사라진다 - 타이핑도 없고 클릭도
    없는, 조용히 삭제된 조작이 된다.
    """
    ocr = _StubOCR(["same", "same"])
    events, ranks = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert events == []
    assert ranks == set()
