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

    수동 녹화 세션의 로컬 커서(GetCursorPos)는 엔지니어의 커서 그 자체라
    VLM 추정보다 정확하고 콜이 들지 않는다. 알람 녹화는 사이드카가 없어
    항상 None 이 나오고 호출부가 기존 VLM 경로로 폴백한다.

    좌표 변환은 region_gate.screen_point_to_frame 을 쓴다 - 단순 뺄셈은
    오피스 125/150% 배율에서 좌표계를 섞는다(2026-08-10 FINDING 2).
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
    return [int(fx), int(fy)]


def _locate_cursor(client, frame_path: Path):
    """프레임에서 커서 coarse bbox(native px) 를 탐지한다.

    반환: (parsed_dict, cursor_px_bbox|None, img_w, img_h).
    """
    image = Image.open(frame_path).convert("RGB")
    image_b64, width, height = encode_image_webp(image)
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

        try:
            parsed, cursor_px, width, height = _locate_cursor(client, Path(change.frame_path))
            calls += 1
        except Exception as exc:
            calls += 1
            print(f"[WARNING] 커서 탐지 실패(cursor_unavailable): {change.frame_path}: {exc}")
            results.append(_unavailable_event(change))
            _sleep(settings.vlm_request_delay_sec)
            continue

        if cursor_px is None:
            results.append(
                ClickEvent(
                    change=change, is_click=False, status="no_click",
                    cursor_visible=False, cursor_kind=parsed.get("cursor_kind"),
                    cursor_bbox=None, cursor_xy=None, click_window=None,
                    changed_in_window_px=0,
                    confidence=float(parsed.get("confidence") or 0.0),
                    evidence=str(parsed.get("evidence") or ""),
                    cursor_source="vlm",
                )
            )
            _sleep(settings.vlm_request_delay_sec)
            continue

        center = bbox_center(cursor_px)
        mask = _diff_mask(
            Path(change.prev_frame_path), Path(change.frame_path), settings.click_diff_threshold
        )
        if mask is None:
            results.append(_unavailable_event(change))
            _sleep(settings.vlm_request_delay_sec)
            continue
        window = _window_around(center["x"], center["y"], settings.cursor_click_window_px, width, height)
        changed = _count_changed_in_window(mask, window)
        is_click = changed >= settings.click_min_changed_px
        results.append(
            ClickEvent(
                change=change, is_click=is_click,
                status="click" if is_click else "no_click",
                cursor_visible=True, cursor_kind=parsed.get("cursor_kind"),
                cursor_bbox=cursor_px, cursor_xy=[center["x"], center["y"]],
                click_window=window, changed_in_window_px=changed,
                confidence=float(parsed.get("confidence") or 0.0),
                evidence=str(parsed.get("evidence") or ""),
                cursor_source="vlm",
            )
        )
        _sleep(settings.vlm_request_delay_sec)

    n_click = sum(1 for r in results if r.is_click)
    print(f"[INFO] Stage 2a 완료: clicks={n_click} / processed={len(results)}")
    return results


def _sleep(delay_sec: float) -> None:
    """프록시 과부하 방지 대기(테스트는 0 으로 무효화)."""
    if delay_sec and delay_sec > 0:
        time.sleep(delay_sec)
