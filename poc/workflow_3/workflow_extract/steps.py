"""workflow step dict 를 만드는 단일 지점.

스키마 필드는 값이 없어도 항상 존재한다 - 소비자(render, 미래 재생기)가 키 유무로
분기하기 시작하면 스키마가 암묵적으로 갈라지기 때문이다.
"""


def make_step(events, *, action, rule, target=None, target_kind=None, value=None,
              value_source="none", coords_in_live_box=None, intent=None,
              count=None, inferred=False) -> dict:
    """구성 이벤트 목록에서 step dict 하나를 만든다.

    events 는 시간순이라고 가정한다(그룹핑 패스가 순서대로 넘긴다). raw_events 는
    원본 타임라인의 seq 를 그대로 든다 - 이것이 그룹핑을 되돌릴 수 있게 하는 유일한
    연결고리다.
    """
    first, last = events[0], events[-1]
    if target_kind is None:
        target_kind = first.get("target_kind") or "unknown"
    # 타이핑 이벤트는 구간이라 시작 시각 하나로는 길이를 잃는다. Stage 2b 가 실은
    # t_sec_end 를 우선 쓴다(클릭 이벤트에는 없으므로 t_sec 로 폴백).
    end_t = float(last.get("t_sec_end") or last["t_sec"])
    return {
        "seq": 0,   # 그룹핑 패스가 마지막에 다시 매긴다.
        "action": action,
        "target": target,
        "target_kind": target_kind,
        "value": value,
        "value_source": value_source,
        "coords_in_live_box": coords_in_live_box,
        "t_sec": [float(first["t_sec"]), end_t],
        "generation": int(first.get("generation") or 0),
        "grouping_rule": rule,
        "inferred": bool(inferred),
        "intent": intent,
        "count": count,
        "raw_events": [int(e["seq"]) for e in events],
        "frame": first.get("frame"),
    }
