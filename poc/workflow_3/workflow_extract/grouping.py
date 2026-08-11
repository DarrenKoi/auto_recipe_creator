"""타임라인 이벤트를 의미 단위 step 으로 묶는다 - greedy 단일 패스.

좌->우로 한 번 훑으며 R1..R5 를 우선순위대로 시도하고, 먼저 맞는 규칙이 이벤트를
가져간다. 규칙이 틀려도 되돌릴 수 있도록 모든 이벤트는 정확히 하나의 step
raw_events 에 들어간다(패스 끝에서 assert 로 확인한다).
"""

from collections import Counter
from dataclasses import dataclass, field

from poc.workflow_3.util import env_float
from poc.workflow_3.workflow_extract.steps import make_step

# R2 전용 - 피커가 오프너보다 "충분히 아래"인지 보는 최소 수직 간격(px). same_target_px
# (R4 의 "라벨 없을 때 동일 대상" 판정용)와 의미가 다르므로 별도 상수로 둔다 - 나중에
# 한쪽만 오피스 실측으로 튜닝하면 다른 쪽 경계가 조용히 같이 움직여선 안 된다.
# 값 12 는 두 경계 사이 절충치다: 이 UI 의 드롭다운 행 높이(~24px, 실측 아니라 추정)
# 보다는 작아야 진짜 첫 행 선택을 억지로 R5 로 떨어뜨리지 않고, 사람이 같은 버튼을
# 다시 누를 때의 재클릭 지터(보통 몇 px)보다는 커야 반복 클릭을 드롭다운으로
# 오판하지 않는다. 확정값 아님 - 오피스 첫 실측 녹화로 확인 필요.
_DROPDOWN_MIN_ROW_GAP_PX = env_float("WORKFLOW_EXTRACT_DROPDOWN_MIN_ROW_GAP_PX", 12.0)

# R2 전용 - 오프너에서 피커까지 허용하는 최대 수직 거리(px).
# (2026-08-11 리뷰 C3) dropdown_region_below 는 PM 드롭다운 **crop** 용 기하라
# 아래로 프레임 높이의 0.45 배(1080 에서 486px, 화면 절반)를 잡는다. 그 띠를 트리거로
# 쓰면 세로로 쌓인 평범한 폼에서 서로 다른 두 컨트롤 클릭이 한 개의 드롭다운 선택으로
# 뭉쳐지고, 두 번째 진짜 클릭은 문서에서 사라진다. 그래서 트리거 쪽에만 별도로
# "그럴듯한 리스트 높이" 상한을 둔다(crop 기하 자체는 건드리지 않는다).
# 값 120 은 이 UI 의 행 높이 추정(~24px) 기준 5행이다 - 실측 아님. 타이트하게 잡는
# 이유는 두 실패의 대가가 비대칭이기 때문이다: 놓치면 정직한 클릭 2개(R5)로
# degrade 하지만, 잘못 발화하면 없던 값이 생기고 진짜 클릭이 없어진다. 오피스에서
# 실제 리스트 높이를 재고 나면 env 로 올린다.
_DROPDOWN_MAX_DROP_PX = env_float("WORKFLOW_EXTRACT_DROPDOWN_MAX_DROP_PX", 120.0)


@dataclass
class GroupingContext:
    """규칙이 참조하는 부수 입력. 없으면 해당 규칙이 degrade 한다."""

    settings: object
    live_boxes: dict = field(default_factory=dict)   # {generation: live_box}
    changes: list = field(default_factory=list)      # change_events.json 의 events
    frame_wh: tuple = None                           # (w, h). None 이면 R2 degrade.


def box_overlap_ratio(bbox, live_box) -> float:
    """bbox 와 live_box 의 교집합 면적을 live_box 면적으로 나눈 비율."""
    if not bbox or not live_box:
        return 0.0
    left = max(int(bbox["left"]), int(live_box["left"]))
    top = max(int(bbox["top"]), int(live_box["top"]))
    right = min(int(bbox["right"]), int(live_box["right"]))
    bottom = min(int(bbox["bottom"]), int(live_box["bottom"]))
    if right <= left or bottom <= top:
        return 0.0
    live_area = (int(live_box["right"]) - int(live_box["left"])) * (
        int(live_box["bottom"]) - int(live_box["top"])
    )
    if live_area <= 0:
        return 0.0
    return ((right - left) * (bottom - top)) / float(live_area)


def normalized_in_live_box(coords, live_box):
    """클릭 좌표를 라이브 박스 내부 0~1 좌표로 바꾼다. 불가하면 None.

    창 위치/크기에 독립적이고, '좌표가 아니라 영상 내용에 의존한다'는 것을
    스키마 자체로 드러낸다.
    """
    if not coords or not live_box:
        return None
    width = int(live_box["right"]) - int(live_box["left"])
    height = int(live_box["bottom"]) - int(live_box["top"])
    if width <= 0 or height <= 0:
        return None
    nx = (float(coords["x"]) - int(live_box["left"])) / width
    ny = (float(coords["y"]) - int(live_box["top"])) / height
    return [round(nx, 4), round(ny, 4)]


def _has_recenter_change(t_sec, live_box, ctx) -> bool:
    """클릭 직후 창 안에서 라이브 박스 대부분이 다시 그려졌는지 본다."""
    for change in ctx.changes or []:
        delta = float(change.get("timestamp_sec") or 0.0) - float(t_sec)
        if delta < 0 or delta > ctx.settings.recenter_window_sec:
            continue
        if box_overlap_ratio(change.get("change_bbox"), live_box) >= ctx.settings.recenter_min_ratio:
            return True
    return False


def _rule_double_click(events, i, ctx):
    """R1 - 라이브 박스 클릭 + recenter 시그니처 = FOV 이동 더블클릭(추론).

    프레임 주기(~3-4fps)로는 두 번의 누름을 시간으로 분리할 수 없다. 대신 결과를
    본다 - recenter 는 라이브 박스 전체를 다시 그리고, 단발 클릭은 국소 변화만
    남긴다. 관측이 아니라 추론이므로 inferred=True 를 남긴다.
    """
    event = events[i]
    if event.get("action") != "click" or event.get("region") != "live_image":
        return None
    live_box = (ctx.live_boxes or {}).get(int(event.get("generation") or 0))
    if not live_box:
        return None
    if not _has_recenter_change(event["t_sec"], live_box, ctx):
        return None
    return make_step(
        [event], action="double_click", rule="R1", intent="fov_move", inferred=True,
        target=event.get("element"), target_kind="live_image",
        coords_in_live_box=normalized_in_live_box(event.get("coords"), live_box),
    ), 1


def _rule_default(events, i, ctx):
    """R5 - 위 규칙에 안 걸린 이벤트를 1:1 step 으로 만든다."""
    event = events[i]
    live_box = (ctx.live_boxes or {}).get(int(event.get("generation") or 0))
    normalized = None
    if event.get("region") == "live_image" and live_box:
        normalized = normalized_in_live_box(event.get("coords"), live_box)
    return make_step(
        [event], action="click", rule="R5",
        target=event.get("element"), value=None,
        coords_in_live_box=normalized,
    ), 1


def _point_in_region(coords, region) -> bool:
    """(l, t, r, b) 튜플 영역 안에 점이 있는지 본다."""
    if not coords or not region:
        return False
    left, top, right, bottom = region
    return left <= float(coords["x"]) <= right and top <= float(coords["y"]) <= bottom


def _rule_dropdown(events, i, ctx):
    """R2 - ui_control 클릭 직후 그 아래 영역 클릭 = 드롭다운 선택.

    기하는 sem_monitor.pm_dropdown.dropdown_region_below 를 그대로 쓴다. PM 드롭다운
    실행기가 이미 쓰는 함수라, 관측이 인식하는 드롭다운과 실행기가 수행할 수 있는
    드롭다운이 어긋날 수 없다. 다만 그 비율 상수는 PM 전용 보정이라 더 넓은
    드롭다운은 놓칠 수 있다(첫 실측 후 일반 비율셋 필요 여부를 판단한다).

    바로 다음 이벤트만 본다 - 사이에 다른 조작이 끼면 묶지 않는다. 비인접 이벤트를
    소비하면 사이 이벤트가 건너뛰어져 불변식이 깨진다.

    피커가 오프너보다 충분히(_DROPDOWN_MIN_ROW_GAP_PX 이상) 아래에 있지 않으면
    드롭다운으로 보지 않는다 - 방향성 검사이지 거리 검사가 아니다. dropdown_region_below
    는 top_gap=0 이라 영역 상단이 버튼 지점과 맞닿아 있어, 수직 간격을 걸지 않으면
    같은 버튼을 다시 누른 R4 반복 클릭 패턴(dx≈0, dy≈0)이 R2 에 선점당한다. 반경
    검사(dx²+dy²)를 쓰면 그 경계가 이 UI 의 드롭다운 첫 행 높이(~24px)와 겹쳐 진짜
    첫 항목 선택까지 오검(false reject)한다 - x 방향은 이미 아래 point_in_region 이
    영역 폭으로 걸러주므로 여기서는 y 방향만 본다.

    (2026-08-11 리뷰 C3) 여기에 가드 둘을 더 건다. 이 규칙에는 "드롭다운이 실제로
    열렸다"는 증거가 없고 기하만 있어서, 평범한 연속 클릭이 없던 선택을 만들어 내고
    두 번째 클릭을 삼켰다.
    1. 오프너와 피커가 **같은 대상**이면 거부한다 - 드롭다운이 자기 오프너의 라벨을
       고르는 일은 없다. 같은 버튼을 사람이 다시 누를 때의 지터(수십 px)가
       _DROPDOWN_MIN_ROW_GAP_PX(12px)만 넘으면 R4 반복 클릭이 R2 에 선점당했다.
       라벨/좌표 어느 쪽으로 판정되든 막히므로 행 높이에 의존하지 않는다.
    2. 수직 거리를 _DROPDOWN_MAX_DROP_PX 로 제한한다 - crop 기하의 0.45*높이 띠는
       화면 절반이라 세로로 쌓인 폼 전체가 후보가 된다.
    """
    from poc.workflow_3.sem_monitor.pm_dropdown import dropdown_region_below

    opener = events[i]
    if opener.get("action") != "click" or opener.get("target_kind") != "ui_control":
        return None
    if not opener.get("coords") or not ctx.frame_wh:
        return None
    if i + 1 >= len(events):
        return None
    picker = events[i + 1]
    if picker.get("action") != "click" or not picker.get("coords"):
        return None
    if float(picker["t_sec"]) - float(opener["t_sec"]) > ctx.settings.dropdown_max_sec:
        return None
    if same_target(opener, picker, ctx.settings):
        return None
    dy = float(picker["coords"]["y"]) - float(opener["coords"]["y"])
    if dy < _DROPDOWN_MIN_ROW_GAP_PX or dy > _DROPDOWN_MAX_DROP_PX:
        return None
    region = dropdown_region_below(
        {"x": int(opener["coords"]["x"]), "y": int(opener["coords"]["y"])}, ctx.frame_wh
    )
    if region is None or not _point_in_region(picker.get("coords"), region):
        return None
    return make_step(
        [opener, picker], action="select_from_dropdown", rule="R2",
        target=opener.get("element"), target_kind="ui_control",
        value=picker.get("element"),
        value_source=picker.get("element_source") or "none",
    ), 2


def same_target(a, b, settings) -> bool:
    """두 클릭이 같은 대상을 눌렀는지 본다.

    라벨이 둘 다 있으면 라벨로, 둘 다 없으면 좌표 근접으로 판정한다. 한쪽만 있는
    경우는 같다고 보지 않는다 - 같은 버튼을 두 번 눌렀는데 한 번만 OCR 이 성공한
    경우를 억지로 묶으면 묶임 여부가 OCR 운에 좌우되어 재현되지 않는다.
    """
    label_a = (a.get("element") or "").strip()
    label_b = (b.get("element") or "").strip()
    if label_a and label_b:
        return label_a == label_b
    if label_a or label_b:
        return False
    ca, cb = a.get("coords"), b.get("coords")
    if not ca or not cb:
        return False
    dx = float(ca["x"]) - float(cb["x"])
    dy = float(ca["y"]) - float(cb["y"])
    return (dx * dx + dy * dy) <= float(settings.same_target_px) ** 2


def _rule_type_text(events, i, ctx):
    """R3 - 타이핑 구간. 직전 필드 클릭이 있으면 포커스로 흡수한다.

    (2026-08-11 리뷰 I2) target_kind 는 하드코딩하지 않고 **타이핑 이벤트에서
    승계**한다(스펙 6: "타임라인에서 그대로 승계"). Stage 2b 는 OCR 이 라벨을
    못 읽으면 unknown 을 싣는데, 여기서 ui_control 로 덮으면 라벨을 읽은 적도 없는
    step 이 "다른 장비에서 라벨로 다시 찾을 수 있다"고 주장한다. 흡수 갈래에서는
    events[0] 이 클릭이라 make_step 의 first 기반 폴백이 클릭의 값을 집으므로
    타이핑 이벤트의 값을 명시적으로 넘긴다.
    """
    event = events[i]
    if event.get("action") == "type_text":
        return make_step(
            [event], action="type_text", rule="R3", target=event.get("element"),
            target_kind=event.get("target_kind") or "unknown", value=event.get("text"),
            value_source=event.get("element_source") or "none",
        ), 1

    if event.get("action") != "click" or i + 1 >= len(events):
        return None
    typing = events[i + 1]
    if typing.get("action") != "type_text":
        return None
    if float(typing["t_sec"]) - float(event["t_sec"]) > ctx.settings.focus_max_sec:
        return None
    return make_step(
        [event, typing], action="type_text", rule="R3",
        target=typing.get("element") or event.get("element"),
        target_kind=typing.get("target_kind") or "unknown", value=typing.get("text"),
        value_source=typing.get("element_source") or "none",
    ), 2


def _rule_click_repeat(events, i, ctx):
    """R4 - 같은 대상을 짧은 창 안에 여러 번 누른 것을 하나로 묶는다."""
    first = events[i]
    if first.get("action") != "click":
        return None
    group = [first]
    for j in range(i + 1, len(events)):
        nxt = events[j]
        if nxt.get("action") != "click":
            break
        if float(nxt["t_sec"]) - float(first["t_sec"]) > ctx.settings.repeat_window_sec:
            break
        if not same_target(first, nxt, ctx.settings):
            break
        group.append(nxt)
    if len(group) < ctx.settings.repeat_min_count:
        return None
    return make_step(
        group, action="click_repeat", rule="R4", target=first.get("element"),
        count=len(group),
    ), len(group)


_RULES = [
    _rule_double_click,   # R1
    _rule_dropdown,       # R2
    _rule_type_text,      # R3
    _rule_click_repeat,   # R4
    _rule_default,        # R5
]


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
                if consumed <= 0:
                    # consumed <= 0 이면 i 가 전진하지 않아 while 이 영원히 돈다 -
                    # 크래시도, 로그도 없이 그냥 멈춘 것처럼 보이는 최악의 실패
                    # 모드다. R1-R4 가 여러 개(N>=1) 이벤트를 소비하게 되면
                    # 규칙 구현 실수가 여기로 흘러들 수 있으므로, 전진을 보장하지
                    # 못하는 규칙은 어떤 규칙이 몇을 반환했는지 못박아 즉시 멈춘다.
                    raise AssertionError(
                        f"규칙이 이벤트를 소비하지 않았습니다: rule={getattr(rule, '__name__', rule)}, "
                        f"consumed={consumed!r} (양수여야 함), seq={ordered[i]['seq']}"
                    )
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
