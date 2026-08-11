"""STAGE 2b - 타이핑 구간을 찾아 OCR 로 입력값을 복원한다.

타이핑은 마우스가 멈춘 채 픽셀이 계속 바뀌는 유일한 조작이다 - 클릭(커서 이동 후
1회 국소 변화)의 정확한 반대다. 입력을 후킹하지 않으므로 키는 기록되지 않고,
화면에 렌더된 글자만 OCR 로 복원한다.

이 모듈은 두 단계로 나뉜다. `find_typing_bursts` (구현됨)는 커서 정지 + 국소
반복 변화만으로 후보 구간을 찾는다 - OCR 은 하지 않으며, 캐럿(텍스트 커서)
깜빡임을 의도적으로 걸러내지 않는다. 캐럿 깜빡임도 마우스가 멈춘 채 같은 좁은
영역이 반복적으로 바뀌는 동일한 신호를 내므로, 지금 단계에서는 실제 타이핑과
구분되지 않고 그대로 `TypingBurst` 로 올라온다. 후속 `resolve_typing_events`
(다음 태스크에서 이 모듈에 추가 예정)가 구간 시작/끝 시점의 필드 텍스트를 OCR 로
읽어 비교해, 텍스트가 같으면 캐럿 깜빡임으로 보고 그 구간을 버린다.
"""

from dataclasses import dataclass, field
from pathlib import Path

from poc.workflow_3.recording_filter.region_gate import nearest_meta


@dataclass
class TypingBurst:
    """타이핑으로 보이는 연속 변화 구간 1건."""

    ranks: list = field(default_factory=list)   # 구성 ChangeEvent 의 rank 목록
    start_t_sec: float = 0.0
    end_t_sec: float = 0.0
    roi: dict = field(default_factory=dict)     # 구성 change_bbox 의 합집합
    cursor_xy: list = field(default_factory=list)   # 프레임 좌표가 아니라 화면 좌표
    frame_path: str = ""        # 구간 시작 프레임
    end_frame_path: str = ""    # 구간 종료 프레임


def _union_box(a, b):
    """두 bbox 의 합집합을 만든다. 한쪽이 비면 다른 쪽을 그대로 돌려준다."""
    if not a:
        return dict(b) if b else {}
    if not b:
        return dict(a)
    return {
        "left": min(int(a["left"]), int(b["left"])),
        "top": min(int(a["top"]), int(b["top"])),
        "right": max(int(a["right"]), int(b["right"])),
        "bottom": max(int(a["bottom"]), int(b["bottom"])),
    }


def _cursor_moved(prev_xy, curr_xy, still_px) -> bool:
    """두 화면 좌표 사이 이동이 still_px 를 넘는지 본다."""
    if prev_xy is None or curr_xy is None:
        return True
    dx = float(curr_xy[0]) - float(prev_xy[0])
    dy = float(curr_xy[1]) - float(prev_xy[1])
    return (dx * dx + dy * dy) > (float(still_px) ** 2)


def _flush(current, settings, out):
    """모아둔 구간이 최소 길이를 넘으면 결과에 넣는다."""
    if current is not None and len(current.ranks) >= settings.typing_min_burst_events:
        out.append(current)


def find_typing_bursts(change_events, metas, settings) -> list:
    """커서 정지 + 국소 반복 변화로 타이핑 구간을 찾는다(OCR 없음).

    사이드카가 없으면 커서 정지를 알 수 없으므로 빈 목록을 돌려준다 - 알람 녹화는
    이 단계를 통째로 건너뛴다.
    """
    if not metas or not change_events:
        return []

    bursts: list = []
    current = None
    prev_cursor = None
    prev_t = None

    for event in change_events:
        meta = nearest_meta(metas, event.timestamp_sec)
        cursor = meta.cursor_xy if meta is not None else None
        if cursor is None:
            _flush(current, settings, bursts)
            current, prev_cursor, prev_t = None, None, None
            continue

        idle_broken = (
            prev_t is not None
            and (event.timestamp_sec - prev_t) > settings.typing_burst_idle_sec
        )
        if current is None or idle_broken or _cursor_moved(
            prev_cursor, cursor, settings.typing_cursor_still_px
        ):
            _flush(current, settings, bursts)
            current = TypingBurst(
                ranks=[event.rank], start_t_sec=event.timestamp_sec,
                end_t_sec=event.timestamp_sec, roi=dict(event.change_bbox or {}),
                cursor_xy=list(cursor), frame_path=event.frame_path,
                end_frame_path=event.frame_path,
            )
        else:
            current.ranks.append(event.rank)
            current.end_t_sec = event.timestamp_sec
            current.roi = _union_box(current.roi, event.change_bbox)
            current.end_frame_path = event.frame_path

        prev_cursor, prev_t = cursor, event.timestamp_sec

    _flush(current, settings, bursts)
    return bursts
