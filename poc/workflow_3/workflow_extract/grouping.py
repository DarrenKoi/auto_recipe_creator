"""타임라인 이벤트를 의미 단위 step 으로 묶는다 - greedy 단일 패스.

좌->우로 한 번 훑으며 R1..R5 를 우선순위대로 시도하고, 먼저 맞는 규칙이 이벤트를
가져간다. 규칙이 틀려도 되돌릴 수 있도록 모든 이벤트는 정확히 하나의 step
raw_events 에 들어간다(패스 끝에서 assert 로 확인한다).
"""

from collections import Counter
from dataclasses import dataclass, field

from poc.workflow_3.workflow_extract.steps import make_step


@dataclass
class GroupingContext:
    """규칙이 참조하는 부수 입력. 없으면 해당 규칙이 degrade 한다."""

    settings: object
    live_boxes: dict = field(default_factory=dict)   # {generation: live_box}
    changes: list = field(default_factory=list)      # change_events.json 의 events
    frame_wh: tuple = None                           # (w, h). None 이면 R2 degrade.


def _rule_default(events, i, ctx):
    """R5 - 위 규칙에 안 걸린 이벤트를 1:1 step 으로 만든다."""
    event = events[i]
    return make_step(
        [event], action="click", rule="R5",
        target=event.get("element"), value=None,
    ), 1


_RULES = [_rule_default]


def group_events(events, ctx) -> list:
    """이벤트 목록을 step 목록으로 묶는다(시간순 정렬 후 greedy 단일 패스)."""
    ordered = sorted(events or [], key=lambda e: float(e["t_sec"]))
    steps = []
    i = 0
    while i < len(ordered):
        for rule in _RULES:
            result = rule(ordered, i, ctx)
            if result is not None:
                step, consumed = result
                steps.append(step)
                i += consumed
                break
        else:   # 모든 규칙이 None 을 돌려주는 일은 없어야 한다(R5 가 항상 잡는다).
            raise AssertionError(f"이벤트를 처리할 규칙이 없습니다: seq={ordered[i]['seq']}")

    for seq, step in enumerate(steps):
        step["seq"] = seq

    _assert_invariant(ordered, steps)
    return steps


def _assert_invariant(events, steps) -> None:
    """모든 이벤트가 정확히 한 번씩 raw_events 에 나타나는지 확인한다.

    이 불변식이 깨지면 잘못된 그룹핑을 되돌릴 수 없다 - 산출물을 내보내기 전에
    여기서 멈추는 편이 조용히 왜곡된 절차서를 내는 것보다 낫다.

    누락/중복을 각각 정확히 짚어야 한다. 단순히 `set(seen) - set(expected)` 만 보면
    "raw_events 에 있지만 애초에 입력에 없던 seq" 만 잡히고, 정작 흔한 실패 형태인
    "입력에는 있는 seq 가 두 step 에 겹쳐 들어간 경우"(seen 요소이자 expected 요소라
    두 집합의 차집합에서 사라짐)는 조용히 통과해 버린다. 그래서 중복은 카운트로 따로
    센다.
    """
    expected = sorted(int(e["seq"]) for e in events)
    seen = sorted(r for step in steps for r in step["raw_events"])
    if seen != expected:
        expected_set = set(expected)
        seen_set = set(seen)
        missing = sorted(expected_set - seen_set)
        unexpected = sorted(seen_set - expected_set)
        duplicated = sorted(seq for seq, count in Counter(seen).items() if count > 1)
        raise AssertionError(
            f"그룹핑 불변식 위반: 입력 {len(expected)} 건, raw_events {len(seen)} 건. "
            f"누락={missing}, 중복={duplicated}, 입력에 없는 seq={unexpected}"
        )
