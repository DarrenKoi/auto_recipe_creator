"""STAGE 2b - 타이핑 구간을 찾아 OCR 로 입력값을 복원한다.

타이핑은 마우스가 멈춘 채 픽셀이 계속 바뀌는 유일한 조작이다 - 클릭(커서 이동 후
1회 국소 변화)의 정확한 반대다. 입력을 후킹하지 않으므로 키는 기록되지 않고,
화면에 렌더된 글자만 OCR 로 복원한다.

이 모듈은 두 단계로 나뉜다. `find_typing_bursts` 는 커서 정지 + 국소 반복 변화만으로
후보 구간을 찾는다 - OCR 은 하지 않으며, 캐럿(텍스트 커서) 깜빡임을 의도적으로
걸러내지 않는다. 캐럿 깜빡임도 마우스가 멈춘 채 같은 좁은 영역이 반복적으로
바뀌는 동일한 신호를 내므로, 그 단계에서는 실제 타이핑과 구분되지 않고 그대로
`TypingBurst` 로 올라온다. `resolve_typing_events` 가 구간 시작/끝 시점의 필드
텍스트를 OCR 로 읽어 비교해, 텍스트가 같으면 캐럿 깜빡임으로 보고 그 구간을 버린다.
"""

from dataclasses import dataclass, field
from pathlib import Path

from poc.workflow_3.recording_filter.region_gate import (
    nearest_meta,
    read_frame_size,
    screen_point_to_frame,
)
from poc.workflow_3.recording_filter.timeline import derive_target_kind


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
    # 필드 기준점(프레임 좌표) - 국소성 판정의 원점. 포커스 클릭 좌표가 1순위,
    # 사이드카 커서를 프레임 좌표로 옮긴 값이 2순위다.
    anchor_xy: list = field(default_factory=list)
    anchor_source: str = "none"      # focus_click | cursor | none
    # 커서의 **프레임** 좌표. cursor_xy(화면 좌표)와 좌표계가 달라 별도 필드로 둔다 -
    # 타임라인의 click 이벤트 coords 가 프레임 좌표이므로 type_text 도 같아야 한다.
    cursor_frame_xy: list = field(default_factory=list)


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


def _cursor_moved(start_xy, curr_xy, still_px) -> bool:
    """구간 **시작** 커서에서 지금 커서까지의 이동이 still_px 를 넘는지 본다.

    (2026-08-11 리뷰 C2) 직전 이벤트와 비교하면 매 스텝 still_px 미만으로 조금씩
    움직이는 커서가 영원히 "정지"로 남는다 - 한 스텝 7px 씩 20 프레임이면 140px 를
    이동했는데도 구간이 끊기지 않는다. 정지의 기준점은 구간 시작이어야 한다.
    """
    if start_xy is None or curr_xy is None:
        return True
    dx = float(curr_xy[0]) - float(start_xy[0])
    dy = float(curr_xy[1]) - float(start_xy[1])
    return (dx * dx + dy * dy) > (float(still_px) ** 2)


def _box_center(box):
    """bbox 중심 (x, y). 없으면 None."""
    if not box:
        return None
    return (
        (float(box["left"]) + float(box["right"])) / 2.0,
        (float(box["top"]) + float(box["bottom"])) / 2.0,
    )


def _box_area(box) -> float:
    """bbox 면적(음수 폭/높이는 0 으로 클램프)."""
    if not box:
        return 0.0
    width = float(box["right"]) - float(box["left"])
    height = float(box["bottom"]) - float(box["top"])
    return max(0.0, width) * max(0.0, height)


def _near_anchor(box, anchor, max_px) -> bool:
    """bbox 중심이 기준점에서 max_px 안에 있는지 본다."""
    center = _box_center(box)
    if center is None or not anchor:
        return False
    dx = center[0] - float(anchor[0])
    dy = center[1] - float(anchor[1])
    return (dx * dx + dy * dy) <= float(max_px) ** 2


def _focus_click(start_t_sec, click_events, settings):
    """구간 시작 직전 focus_max_sec 안의 가장 늦은 클릭. 없으면 None.

    라벨(_focus_label)과 좌표(_burst_anchor)가 **같은 클릭**을 가리켜야 한다 -
    한쪽만 다른 클릭을 고르면 문서의 필드 이름과 값이 서로 다른 필드에서 온다.
    """
    best = None
    for ce in click_events or []:
        if not ce.is_click:
            continue
        gap = float(start_t_sec) - float(ce.timestamp_sec)
        if 0 <= gap <= settings.typing_focus_max_sec:
            if best is None or ce.timestamp_sec > best.timestamp_sec:
                best = ce
    return best


def _cursor_frame_point(event, meta, frame_wh_fn):
    """사이드카 커서(화면 좌표)를 프레임 좌표로 옮긴다. 불가하면 None."""
    if meta is None or meta.cursor_xy is None or meta.rect is None:
        return None
    point = screen_point_to_frame(meta.cursor_xy, meta.rect, frame_wh_fn(event.frame_path))
    if point is None:
        return None
    return (float(point[0]), float(point[1]))


def _flush(current, settings, out):
    """모아둔 구간이 최소 길이를 넘으면 결과에 넣는다."""
    if current is not None and len(current.ranks) >= settings.typing_min_burst_events:
        out.append(current)


def _start_burst(event, meta, cursor, click_events, settings, frame_wh_fn):
    """이 이벤트로 새 구간을 시작한다. 불가하면 (None, 사유) 를 돌려준다.

    (2026-08-11 리뷰 C2) 필드 기준점 없이는 "커서를 세워둔 채 화면 어딘가가 반복해
    바뀐다"와 "필드에 글자를 입력한다"를 구분할 수 없다. 스펙 5.3 은 필드 ROI 를
    **포커스 클릭 좌표 주변**으로 정의하므로 그 좌표를 1순위 기준점으로 쓰고,
    포커스 클릭이 없으면(Tab 포커스) 사이드카 커서의 프레임 좌표를 쓴다. 둘 다
    없으면 구간을 만들지 않는다 - 기준점 없는 구간은 화면 어디에서 온 값인지
    보증할 수 없고, 그 값이 절차서에 confidence=1.0 으로 실린다.
    """
    cursor_frame = _cursor_frame_point(event, meta, frame_wh_fn)
    focus = _focus_click(event.timestamp_sec, click_events, settings)
    if focus is not None and focus.cursor_xy:
        anchor = (float(focus.cursor_xy[0]), float(focus.cursor_xy[1]))
        anchor_source = "focus_click"
    elif cursor_frame is not None:
        anchor = cursor_frame
        anchor_source = "cursor"
    else:
        return None, "no_anchor"

    if not _near_anchor(event.change_bbox, anchor, settings.typing_roi_max_px):
        return None, "not_local"
    if _box_area(event.change_bbox) > settings.typing_roi_max_area_px:
        return None, "not_local"

    return TypingBurst(
        ranks=[event.rank], start_t_sec=event.timestamp_sec,
        end_t_sec=event.timestamp_sec, roi=dict(event.change_bbox or {}),
        cursor_xy=list(cursor), frame_path=event.frame_path,
        end_frame_path=event.frame_path,
        anchor_xy=[anchor[0], anchor[1]], anchor_source=anchor_source,
        cursor_frame_xy=list(cursor_frame) if cursor_frame is not None else [],
    ), ""


def _extends_burst(burst, event, settings) -> bool:
    """이 변화가 같은 필드의 연장인지 본다(기준점 근처 + ROI 면적 상한)."""
    if not _near_anchor(event.change_bbox, burst.anchor_xy, settings.typing_roi_max_px):
        return False
    union = _union_box(burst.roi, event.change_bbox)
    return _box_area(union) <= settings.typing_roi_max_area_px


def find_typing_bursts(
    change_events, metas, settings, *, click_events=None, frame_size_fn=None
) -> list:
    """커서 정지 + 필드 근처 반복 변화로 타이핑 구간을 찾는다(OCR 없음).

    사이드카가 없으면 커서 정지를 알 수 없으므로 빈 목록을 돌려준다 - 알람 녹화는
    이 단계를 통째로 건너뛴다.

    시간/커서 조건만으로는 부족하다(2026-08-11 리뷰 C2). 변화가 필드 기준점
    (포커스 클릭 좌표 또는 커서의 프레임 좌표) 근처여야 하고, 구간 ROI 합집합
    면적도 상한을 넘지 않아야 한다 - 그러지 않으면 커서를 세워둔 채 리페인트되는
    진행률 패널이 구간이 되고, OCR 이 그 패널의 숫자를 "입력값" 으로 복원한다.

    frame_size_fn 은 프레임 픽셀 크기 조회(커서 좌표계 변환용) 주입점이다.
    """
    if not metas or not change_events:
        return []

    size_fn = frame_size_fn or read_frame_size
    size_cache = {}

    def _frame_wh(path):
        if path not in size_cache:
            size_cache[path] = size_fn(path)
        return size_cache[path]

    bursts: list = []
    current = None
    prev_t = None
    dropped = {"no_anchor": 0, "not_local": 0}

    for event in change_events:
        meta = nearest_meta(metas, event.timestamp_sec)
        cursor = meta.cursor_xy if meta is not None else None
        if cursor is None:
            _flush(current, settings, bursts)
            current, prev_t = None, None
            continue

        idle_broken = (
            prev_t is not None
            and (event.timestamp_sec - prev_t) > settings.typing_burst_idle_sec
        )
        moved = current is not None and _cursor_moved(
            current.cursor_xy, cursor, settings.typing_cursor_still_px
        )
        left_field = current is not None and not _extends_burst(current, event, settings)
        if current is None or idle_broken or moved or left_field:
            _flush(current, settings, bursts)
            current, reason = _start_burst(
                event, meta, cursor, click_events, settings, _frame_wh
            )
            if reason:
                dropped[reason] += 1
        else:
            current.ranks.append(event.rank)
            current.end_t_sec = event.timestamp_sec
            current.roi = _union_box(current.roi, event.change_bbox)
            current.end_frame_path = event.frame_path

        prev_t = event.timestamp_sec

    _flush(current, settings, bursts)

    if dropped["not_local"]:
        print(
            f"[WARNING] 커서는 멈췄지만 필드 기준점에서 "
            f"{settings.typing_roi_max_px}px 를 벗어난 변화 {dropped['not_local']} 건을 "
            "타이핑 후보에서 제외했습니다(진행률/상태 패널 리페인트로 보입니다)."
        )
    if dropped["no_anchor"]:
        print(
            f"[WARNING] 필드 기준점(포커스 클릭/커서 프레임 좌표)을 정할 수 없어 변화 "
            f"{dropped['no_anchor']} 건을 타이핑 후보에서 제외했습니다."
        )
    return bursts


def _ocr_text_in_box(image, box, ocr_client) -> str:
    """box 영역의 텍스트를 PaddleOCR Spotting 으로 읽어 한 문자열로 잇는다.

    element_label 의 _read_with_ocr 는 '클릭 지점 최근접 1건'을 고르지만, 여기서는
    필드 전체 내용이 필요하므로 모든 항목을 x 순으로 잇는다. 정렬 실패는 삼킨다 -
    캐럿 판별(before != after)에는 순서가 아니라 결정성만 있으면 된다.
    """
    from poc.workflow_3.util import encode_image_webp
    from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
    from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_spotting_prompt

    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    crop_b64, _w, _h = encode_image_webp(crop, quality=90)
    system_msg, user_text = build_spotting_prompt()
    response = ocr_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    items = parse_spotting_items((response.text or "").strip())
    # parse_spotting_items 는 항상 {"text", "bbox"} 로 정규화해 돌려준다("box" 는 입력
    # 쪽에서만 허용되는 별칭이다). bbox 는 리스트가 아니라 left/top/right/bottom dict 다.
    try:
        items = sorted(items, key=lambda it: float(it["bbox"]["left"]))
    except Exception:
        pass
    return " ".join(str(it.get("text") or "").strip() for it in items).strip()


def _focus_label(burst, click_events, labels, settings):
    """구간 직전 focus_max_sec 안의 클릭에서 필드 라벨을 얻는다. 없으면 None.

    Tab/단축키로 포커스를 옮기면 직전 클릭이 없다. 그때는 라벨을 추측하지 않는다 -
    추측 라벨은 새 오차원이고, 값은 라벨 없이도 쓸모가 있다.
    """
    best = _focus_click(burst.start_t_sec, click_events, settings)
    if best is None:
        return None
    label = (labels or {}).get(best.rank)
    text = getattr(label, "text", "") if label is not None else ""
    return text or None


def _default_image_loader(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def resolve_typing_events(
    bursts, click_events, settings, *, ocr_client, image_loader=None, labels=None
):
    """구간별로 OCR 2회를 돌려 타임라인 스키마의 type_text 이벤트를 만든다.

    반환값은 `(events, consumed_ranks)` 다. consumed_ranks 는 **실제로 이벤트가 된**
    구간이 소비한 change event rank 집합이다(캐럿으로 버린 구간은 포함하지 않는다).
    호출부는 이 rank 들에서 나온 클릭을 타임라인에서 억제해야 한다 - 그러지 않으면
    타이핑 중 캐럿/글자 변화가 Stage 2a 의 ROI 임계도 함께 넘겨, 같은 구간이
    "값 입력" 1건 + "반복 클릭 N회" 로 두 번 보고된다(2026-08-11 리뷰 I4).

    before == after 이면서 둘 다 비어 있지 않은 구간만 캐럿 깜빡임으로 보고
    버린다. OCR 예외, 그리고 양쪽 다 빈 문자열(ROI 정렬 오류/판독 실패)은 같은
    실패로 취급해 값을 비운 채로 이벤트를 남긴다 - '여기서 무언가를 입력했다'는
    사실 자체가 절차의 일부이기 때문이다. target_kind 는 element_source 에서
    파생한다(`timeline.derive_target_kind`) - OCR 실패 이벤트는 다른 장비로
    이식 가능한지 알 수 없으므로 unknown 이 맞다.
    """
    loader = image_loader or _default_image_loader
    events = []
    consumed_ranks = set()
    for burst in bursts:
        before, after, source = "", "", "none"
        try:
            before = _ocr_text_in_box(loader(burst.frame_path), burst.roi, ocr_client)
            after = _ocr_text_in_box(loader(burst.end_frame_path), burst.roi, ocr_client)
            source = "ocr"
        except Exception as exc:
            print(f"[WARNING] 타이핑 구간 OCR 실패(값 없이 기록): {exc}")

        if source == "ocr" and before == after:
            if before == "" and after == "":
                # 양쪽 다 빈 문자열은 캐럿 깜빡임과 겉모습이 같지만 원인이 다르다 -
                # OCR 이 ROI 정렬 오류/판독 실패로 아무것도 못 읽은 것이다. 캐럿과
                # 혼동해 버리면 값 복원의 유일한 경로가 조용히 소실되므로, OCR 실패
                # 경로와 동일하게 값 없이 구간을 남긴다.
                print(
                    f"[WARNING] 타이핑 구간 OCR 이 양쪽 다 빈 텍스트를 읽었습니다 "
                    f"(t={burst.start_t_sec:.1f}s, ROI 정렬/판독 확인 필요) - 값 없이 기록"
                )
                source = "none"
            else:
                print(
                    f"[INFO] 캐럿 깜빡임으로 판단해 구간을 버립니다 "
                    f"(t={burst.start_t_sec:.1f}s, 텍스트 변화 없음)"
                )
                continue

        consumed_ranks.update(burst.ranks)
        events.append({
            "t_sec": burst.start_t_sec,
            "seq": 0,
            "action": "type_text",
            # 클릭 이벤트의 coords 와 같은 **프레임** 좌표만 싣는다. 화면 좌표
            # (burst.cursor_xy)를 그대로 쓰면 오피스 125/150% 배율에서 한 필드에
            # 두 좌표계가 섞인다. 변환 불가면 null 이 정직하다.
            "coords": {"x": int(burst.cursor_frame_xy[0]), "y": int(burst.cursor_frame_xy[1])}
            if burst.cursor_frame_xy else None,
            "element": _focus_label(burst, click_events, labels, settings),
            "element_source": source,
            "target_kind": derive_target_kind("ui", source),
            "region": "ui",
            "generation": 0,
            "occlusion": "unknown",
            # (2026-08-11 리뷰 E1) `after or None` 은 "입력했다가 지운 빈 필드"를
            # "값 없음"으로 붕괴시킨다 - 그러면서 element_source 는 "ocr",
            # confidence 는 1.0 으로 남아 value=null / value_source="ocr" 라는
            # 스펙 8 위반 조합이 산출물에 실린다. 판독이 성공했으면 빈 문자열도
            # 값이고, 실패했을 때만 null 이다.
            "text": after if source == "ocr" else None,
            "confidence": 1.0 if source == "ocr" else 0.0,
            "frame": Path(burst.end_frame_path).name,
            "source_frames": {"prev": burst.frame_path, "curr": burst.end_frame_path},
            "cursor_source": "sidecar",
            "t_sec_end": burst.end_t_sec,
        })
    return events, consumed_ranks
