"""STAGE 2a — VLM 커서 탐지 + cv2 ROI 변화로 마우스 클릭을 추출한다.

poc/workflow_2/vlm_cursor_click_filter.py 의 탐지 코어를 이식한다. 생존 프레임마다
VLM 커서 coarse bbox 를 1회 얻고, 커서 중심 정사각 ROI 안의 native 변화 픽셀이
임계 이상이면 클릭으로 본다. VLM client 는 주입(injection)이라 테스트는 오프라인.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from poc.workflow_3.recording_filter.cursor_prompt import (
    cursor_system_prompt,
    cursor_user_prompt,
)
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.region_gate import (
    nearest_meta,
    read_frame_size,
    screen_point_to_frame,
)
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.util import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


@dataclass
class ClickEvent:
    """Stage 2a 결과 1건 (ChangeEvent 확장)."""

    change: ChangeEvent
    is_click: bool
    status: str                 # click | no_click | cursor_unavailable
    cursor_visible: bool
    cursor_kind: str | None
    cursor_bbox: dict | None    # native px
    cursor_xy: list | None      # [x, y]
    click_window: dict | None
    changed_in_window_px: int
    confidence: float
    evidence: str
    cursor_source: str = "none"   # sidecar | vlm | none
    # Stage 2b 가 이 프레임을 타이핑 구간으로 가져갔으면 True - 타임라인에서는
    # type_text 하나로만 보고한다(중복 보고 방지, 2026-08-11 리뷰 I4).
    superseded_by_typing: bool = False

    # 편의 접근자 (timeline 에서 사용).
    @property
    def frame_path(self) -> str:
        return self.change.frame_path

    @property
    def prev_frame_path(self) -> str:
        return self.change.prev_frame_path

    @property
    def timestamp_sec(self) -> float:
        return self.change.timestamp_sec

    @property
    def rank(self) -> int:
        return self.change.rank


def _diff_mask(prev_path: Path, curr_path: Path, threshold: int):
    """native 해상도에서 prev/curr 변화 마스크(dilate 이진)를 만든다."""
    prev_gray = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr_gray = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
    if prev_gray is None or curr_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        target = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target, interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(thresh, kernel, iterations=2)


def _window_around(cx: int, cy: int, side: int, img_w: int, img_h: int) -> dict:
    """커서 중심을 둘러싼 정사각 ROI bbox 를 만든다."""
    half = max(1, side // 2)
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(img_w, cx + half)
    bottom = min(img_h, cy + half)
    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
    return {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)}


def _count_changed_in_window(mask, window: dict) -> int:
    """변화 마스크 안에서 window 영역의 변화 픽셀 수를 센다."""
    crop = mask[window["top"]:window["bottom"], window["left"]:window["right"]]
    if crop.size == 0:
        return 0
    return int(cv2.countNonZero(crop))


def resolve_sidecar_cursor(change, metas, frame_wh):
    """사이드카에서 이 프레임의 커서 프레임 좌표를 얻는다. 불가하면 None.

    수동 녹화 세션의 로컬 커서(GetCursorPos)가 프레임 안에 있으면 VLM 추정보다
    정확하고 콜이 들지 않는다. 알람 녹화는 사이드카가 없어 항상 None 이 나오고
    호출부가 기존 VLM 경로로 폴백한다.

    좌표 변환은 region_gate.screen_point_to_frame 을 쓴다 - 단순 뺄셈은
    오피스 125/150% 배율에서 좌표계를 섞는다(2026-08-10 FINDING 2).

    (2026-08-12) **프레임 밖으로 매핑되면 커서 관측이 아니다.** 로컬 포인터
    (GetCursorPos)와 프레임에 그려진 커서는 같은 것이 아니다 - RCS Remote
    Monitoring 창은 장비 화면을 비추는 뷰라, 커서 글리프는 그 영상 안에 그려져
    있고 우리 포인터가 창 위에 없어도 프레임에는 멀쩡히 보인다. 실제 오피스
    세션에서 포인터가 내내 창 밖(410/410)이었는데도 이 함수가 화면 밖 좌표를
    자신 있게 돌려주는 바람에, 호출부가 VLM 경로를 통째로 건너뛰고 빈 ROI 만
    세어 클릭이 237건 중 0건이 됐다(사이드카 도입 전 같은 작업은 약 50% 를
    잡았다). 범위를 벗어나면 None 을 돌려 VLM 폴백으로 되돌린다 - 틀린 확신보다
    비싼 관측이 낫다.
    """
    if not metas or not frame_wh:
        return None
    meta = nearest_meta(metas, change.timestamp_sec)
    if meta is None or meta.cursor_xy is None or meta.rect is None:
        return None
    point = screen_point_to_frame(meta.cursor_xy, meta.rect, frame_wh)
    if point is None:
        return None
    fx, fy = point
    frame_w, frame_h = int(frame_wh[0]), int(frame_wh[1])
    if not (0 <= fx < frame_w and 0 <= fy < frame_h):
        return None
    return [int(fx), int(fy)]


def _mask_regions(image, mask_boxes):
    """이미 정체가 드러난 오탐 영역을 프레임에서 지운다(중성 회색으로 덮는다).

    (2026-08-12) 프롬프트로 "저 손바닥 아이콘은 커서가 아니다" 라고 말해도, 창
    가장자리에서 커서가 X 로 바뀌는 등 **모델이 진짜 커서를 확신하지 못하는 순간**
    이면 화면에서 가장 커서처럼 생긴 것(= 늘 같은 자리에 있는 손바닥)으로 되돌아
    간다. 지시는 요청이지 제약이 아니다. 이미 정적 오탐으로 확인된 영역은 아예
    보이지 않게 만드는 편이 확실하다.

    검정이 아니라 중성 회색으로 덮는 이유는, 검은 사각형 자체가 어두운 UI 에서
    또 하나의 눈에 띄는 도형이 되어 모델의 시선을 끌기 때문이다.
    """
    if not mask_boxes:
        return image
    from PIL import ImageDraw

    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in mask_boxes:
        draw.rectangle(
            [int(box["left"]), int(box["top"]), int(box["right"]), int(box["bottom"])],
            fill=(128, 128, 128),
        )
    return masked


def _locate_cursor(client, frame_path: Path, mask_boxes=None):
    """프레임에서 커서 coarse bbox(native px) 를 탐지한다.

    mask_boxes 가 있으면 그 영역을 가린 뒤 질의한다(확인된 오탐 영역 제거).
    반환: (parsed_dict, cursor_px_bbox|None, img_w, img_h).
    """
    image = Image.open(frame_path).convert("RGB")
    image_b64, width, height = encode_image_webp(_mask_regions(image, mask_boxes))
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=cursor_system_prompt(),
        user_text=cursor_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return parsed, None, width, height
    bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
    if bbox_1000 is None:
        return parsed, None, width, height
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height), width, height


def _unavailable_event(change: ChangeEvent) -> ClickEvent:
    return ClickEvent(
        change=change, is_click=False, status="cursor_unavailable",
        cursor_visible=False, cursor_kind=None, cursor_bbox=None, cursor_xy=None,
        click_window=None, changed_in_window_px=0, confidence=0.0, evidence="",
    )


# 사이드카 좌표에는 bbox 가 없다. 오버레이(write_click_overlays)가 bbox 없는
# 이벤트를 건너뛰므로, 점 주위에 합성 bbox 를 만들어 클릭 오버레이가 계속
# 그려지게 한다 - 안 만들면 수동 세션의 오버레이가 통째로 사라진다.
SIDECAR_CURSOR_BBOX_PX = 32


def _sidecar_event(change, cursor_xy, frame_wh, settings) -> ClickEvent:
    """사이드카 커서로 ROI 변화를 세어 클릭을 판정한다(VLM 콜 없음)."""
    width, height = frame_wh
    mask = _diff_mask(
        Path(change.prev_frame_path), Path(change.frame_path), settings.click_diff_threshold
    )
    if mask is None:
        event = _unavailable_event(change)
        event.cursor_source = "sidecar"
        return event
    window = _window_around(
        cursor_xy[0], cursor_xy[1], settings.cursor_click_window_px, width, height
    )
    changed = _count_changed_in_window(mask, window)
    is_click = changed >= settings.click_min_changed_px
    return ClickEvent(
        change=change, is_click=is_click,
        status="click" if is_click else "no_click",
        cursor_visible=True, cursor_kind="sidecar",
        cursor_bbox=_window_around(
            cursor_xy[0], cursor_xy[1], SIDECAR_CURSOR_BBOX_PX, width, height
        ),
        cursor_xy=list(cursor_xy), click_window=window,
        changed_in_window_px=changed, confidence=1.0,
        evidence="sidecar cursor", cursor_source="sidecar",
    )


def flag_static_cursor_detections(results, settings) -> int:
    """여러 프레임에서 같은 자리에 계속 잡힌 "커서"를 정적 UI 아이콘으로 보고 무효화한다.

    (2026-08-12) 이 툴은 'Full Size' 버튼과 라이브 SEM 영상 사이에 **손바닥 모양
    아이콘**을 항상 그려 둔다. VLM 이 이걸 커서로 착각하면 모든 프레임에서 같은
    좌표를 확신 있게 돌려주는데, 산출물만 보면 성공과 구분되지 않는다. 게다가 그
    자리는 라이브 박스 테두리/버튼 리페인트로 실제 픽셀이 바뀌는 구역이라 ROI 변화
    임계를 넘겨 **없던 클릭을 만들어낸다** - 클릭 0건보다 나쁘다.

    판별은 VLM 없이 된다: 진짜 커서는 움직이고 정적 아이콘은 안 움직인다. 같은
    좌표(허용 오차 안)가 전체 탐지의 static_cursor_min_ratio 이상을 차지하고 최소
    반복 횟수를 넘으면 그 무리를 통째로 무효화한다.

    엔지니어가 마우스를 한 자리에 세워둔 채 같은 버튼을 반복 클릭하는 정상 상황과
    구분하기 위해 비율 하한을 높게 잡는다(기본 0.5) - 10분 세션의 절반을 1픽셀도
    안 움직이는 커서는 커서가 아니다. 무효화된 이벤트는 지우지 않고 status 를
    바꿔 감사 추적에 남긴다.

    반환: 오탐 무리 목록 [{"anchor", "count", "mask_box"}] (없으면 빈 목록).
    mask_box 는 그 무리의 커서 bbox 합집합에 여유를 준 영역으로, 재질의 때
    가릴 자리다.
    """
    detected = [r for r in results if r.cursor_xy and r.cursor_source == "vlm"]
    if len(detected) < settings.static_cursor_min_repeats:
        return []

    tolerance = settings.static_cursor_tolerance_px
    clusters = []          # [(anchor_xy, [event, ...]), ...]
    for event in detected:
        x, y = event.cursor_xy[0], event.cursor_xy[1]
        for anchor, members in clusters:
            if abs(anchor[0] - x) <= tolerance and abs(anchor[1] - y) <= tolerance:
                members.append(event)
                break
        else:
            clusters.append(((x, y), [event]))

    decoys = []
    for anchor, members in clusters:
        if len(members) < settings.static_cursor_min_repeats:
            continue
        # (2026-08-12) 이 창에는 오탐원이 셋이다(손바닥 아이콘 / 우상단 닫기 X /
        # 라이브 박스 좌상단 '>'). 폴백이 셋으로 갈리면 어느 하나도 과반을 넘지
        # 못해 "과반" 기준만으로는 전부 놓친다. 그래서 시간 폭을 함께 본다:
        # 정적 아이콘은 세션 내내 같은 자리에 나타나고, 같은 버튼을 반복 클릭하는
        # 정상 조작은 짧은 구간에 몰린다.
        span = _time_span_sec(members)
        by_ratio = len(members) / len(detected) >= settings.static_cursor_min_ratio
        by_span = span >= settings.static_cursor_min_span_sec
        if not (by_ratio or by_span):
            continue
        print(
            f"[WARNING] 정적 커서 후보 {anchor} 에서 {len(members)}/{len(detected)} 건 "
            f"(시간 폭 {span:.0f}s) - 커서가 아니라 고정 UI 그래픽(손바닥 아이콘 / "
            "우상단 닫기 X / 라이브 박스 좌상단 '>')일 가능성이 높아 무효화합니다."
        )
        for event in members:
            event.is_click = False
            event.status = "cursor_static_decoy"
        decoys.append({
            "anchor": anchor,
            "count": len(members),
            "mask_box": _union_bbox(
                [e.cursor_bbox for e in members if e.cursor_bbox],
                fallback_xy=anchor,
                pad=settings.static_cursor_mask_pad_px,
            ),
        })
    return decoys


def _time_span_sec(events) -> float:
    """이벤트 무리가 걸쳐 있는 시간 폭(초). 1건이면 0."""
    times = [float(e.timestamp_sec) for e in events]
    if len(times) < 2:
        return 0.0
    return max(times) - min(times)


def _union_bbox(boxes, *, fallback_xy, pad) -> dict:
    """bbox 목록의 합집합에 여유(pad)를 준다. 목록이 비면 좌표 주변 정사각형.

    여유를 주는 이유는 VLM bbox 가 글리프를 빠듯하게 잡을 때가 있어서다 - 아이콘
    가장자리가 몇 px 남으면 모델이 다시 그걸 붙잡는다.
    """
    if not boxes:
        return {
            "left": int(fallback_xy[0]) - pad, "top": int(fallback_xy[1]) - pad,
            "right": int(fallback_xy[0]) + pad, "bottom": int(fallback_xy[1]) + pad,
        }
    return {
        "left": min(int(b["left"]) for b in boxes) - pad,
        "top": min(int(b["top"]) for b in boxes) - pad,
        "right": max(int(b["right"]) for b in boxes) + pad,
        "bottom": max(int(b["bottom"]) for b in boxes) + pad,
    }


def _vlm_event(change, settings, *, client, mask_boxes=None) -> ClickEvent:
    """VLM 으로 커서를 찾아 ROI 변화로 클릭을 판정한 ClickEvent 를 만든다.

    mask_boxes 는 질의 전에 가릴 영역이다(확인된 정적 오탐 제거용). 실패는 모두
    이벤트로 흡수한다 - 한 프레임 때문에 세션 전체가 죽으면 안 된다.
    """
    try:
        parsed, cursor_px, width, height = _locate_cursor(
            client, Path(change.frame_path), mask_boxes=mask_boxes
        )
    except Exception as exc:
        print(f"[WARNING] 커서 탐지 실패(cursor_unavailable): {change.frame_path}: {exc}")
        return _unavailable_event(change)

    if cursor_px is None:
        return ClickEvent(
            change=change, is_click=False, status="no_click",
            cursor_visible=False, cursor_kind=parsed.get("cursor_kind"),
            cursor_bbox=None, cursor_xy=None, click_window=None,
            changed_in_window_px=0,
            confidence=float(parsed.get("confidence") or 0.0),
            evidence=str(parsed.get("evidence") or ""),
            cursor_source="vlm",
        )

    center = bbox_center(cursor_px)
    mask = _diff_mask(
        Path(change.prev_frame_path), Path(change.frame_path), settings.click_diff_threshold
    )
    if mask is None:
        return _unavailable_event(change)
    window = _window_around(
        center["x"], center["y"], settings.cursor_click_window_px, width, height
    )
    changed = _count_changed_in_window(mask, window)
    is_click = changed >= settings.click_min_changed_px
    return ClickEvent(
        change=change, is_click=is_click,
        status="click" if is_click else "no_click",
        cursor_visible=True, cursor_kind=parsed.get("cursor_kind"),
        cursor_bbox=cursor_px, cursor_xy=[center["x"], center["y"]],
        click_window=window, changed_in_window_px=changed,
        confidence=float(parsed.get("confidence") or 0.0),
        evidence=str(parsed.get("evidence") or ""),
        cursor_source="vlm",
    )


def _point_in_box(xy, box) -> bool:
    """좌표가 bbox 안인지 본다(둘 중 하나라도 없으면 False)."""
    if not xy or not box:
        return False
    return (
        box["left"] <= xy[0] <= box["right"] and box["top"] <= xy[1] <= box["bottom"]
    )


def _retry_decoy_events(results, decoys, settings, *, client, calls_used) -> int:
    """정적 오탐으로 무효화된 이벤트를 오탐 영역을 가린 채 다시 물어본다.

    (2026-08-12) 창 가장자리에서 커서가 X 모양으로 바뀌면 모델이 그것을 커서로
    확신하지 못하고, 화면에서 가장 커서처럼 생긴 것(늘 같은 자리의 손바닥 아이콘)
    으로 되돌아간다. 그 프레임들을 그냥 버리면 **가장자리 근처 조작만 골라서**
    타임라인에서 사라진다 - 무작위 손실이 아니라 계통적 편향이라 더 나쁘다.

    오탐 영역을 회색으로 덮고 한 번 더 물으면 모델에게는 되돌아갈 자리가 없어
    진짜 커서(X 포함)를 찾거나 "없음"이라고 답한다. 재질의 결과가 여전히 오탐
    영역 안이면 원래대로 무효 상태를 유지한다.

    남은 콜 예산(max_vlm_calls)을 넘기지 않는다. 반환: 실제로 쓴 콜 수.
    """
    mask_boxes = [d["mask_box"] for d in decoys]
    flagged = [r for r in results if r.status == "cursor_static_decoy"]
    if not flagged:
        return 0

    budget = None
    if settings.max_vlm_calls:
        budget = max(0, settings.max_vlm_calls - calls_used)
        if budget == 0:
            print(
                f"[WARNING] 콜 예산이 남지 않아 정적 오탐 {len(flagged)} 건의 재질의를 "
                "건너뜁니다(그대로 무효 유지)."
            )
            return 0

    used = 0
    recovered = 0
    for event in flagged:
        if budget is not None and used >= budget:
            print(f"[WARNING] 재질의 콜 예산 소진 - 남은 {len(flagged) - used} 건은 무효 유지")
            break
        retried = _vlm_event(event.change, settings, client=client, mask_boxes=mask_boxes)
        used += 1
        _sleep(settings.vlm_request_delay_sec)
        if retried.cursor_xy is None:
            continue                      # 가리고 나니 커서가 없다 - 무효 유지가 맞다.
        if any(_point_in_box(retried.cursor_xy, box) for box in mask_boxes):
            continue                      # 여전히 오탐 자리 - 신뢰하지 않는다.
        index = results.index(event)
        retried.cursor_source = "vlm_masked"
        results[index] = retried
        recovered += 1

    print(
        f"[INFO] 정적 오탐 재질의: {used} 콜로 {recovered} 건 회수 "
        f"(가린 영역 {len(mask_boxes)} 곳)."
    )
    return used


def detect_clicks(
    change_events: list[ChangeEvent],
    settings: RecordingFilterSettings,
    *,
    client,
    metas=None,
) -> list[ClickEvent]:
    """생존 프레임마다 커서를 찾아 ROI 변화로 클릭을 판정한다.

    metas 가 주어지면(수동 녹화 세션 사이드카) 프레임별로 먼저 사이드카 커서를
    시도한다 - 얻으면 VLM 을 부르지 않고 바로 판정한다. 사이드카가 없거나
    조인 불가면 오늘과 동일한 VLM 경로로 폴백한다(알람 녹화는 항상 이 경로).
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")

    results: list[ClickEvent] = []
    calls = 0
    sidecar_rejected = 0     # 사이드카는 있는데 커서가 프레임 밖 -> VLM 폴백.
    for change in change_events:
        if settings.max_vlm_calls and calls >= settings.max_vlm_calls:
            print(f"[WARNING] max_vlm_calls={settings.max_vlm_calls} 도달 -> 이후 생존 분류 중단")
            break

        frame_wh = read_frame_size(change.frame_path) if metas else None
        sidecar_xy = resolve_sidecar_cursor(change, metas, frame_wh)
        if sidecar_xy is not None:
            results.append(
                _sidecar_event(change, sidecar_xy, frame_wh, settings)
            )
            continue
        if metas:
            sidecar_rejected += 1

        event = _vlm_event(change, settings, client=client)
        calls += 1
        results.append(event)
        _sleep(settings.vlm_request_delay_sec)

    if settings.static_cursor_reject:
        decoys = flag_static_cursor_detections(results, settings)
        if decoys and settings.static_cursor_retry_masked:
            calls += _retry_decoy_events(
                results, decoys, settings, client=client, calls_used=calls
            )

    n_click = sum(1 for r in results if r.is_click)
    if sidecar_rejected:
        # 폴백은 비용(콜 수)이 달라지는 사건이라 조용히 넘어가지 않는다.
        print(
            f"[WARNING] 사이드카 커서가 프레임 밖이라 {sidecar_rejected} 건은 VLM 으로 "
            "찾았습니다 - RCS 창은 장비 화면을 비추는 뷰라 로컬 포인터가 창 밖이어도 "
            "프레임에는 커서가 그려져 있습니다(그 경우 VLM 이 유일한 관측입니다)."
        )
    print(f"[INFO] Stage 2a 완료: clicks={n_click} / processed={len(results)}")
    return results


def _sleep(delay_sec: float) -> None:
    """프록시 과부하 방지 대기(테스트는 0 으로 무효화)."""
    if delay_sec and delay_sec > 0:
        time.sleep(delay_sec)
