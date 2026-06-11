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


def detect_clicks(
    change_events: list[ChangeEvent],
    settings: RecordingFilterSettings,
    *,
    client,
) -> list[ClickEvent]:
    """생존 프레임마다 커서를 찾아 ROI 변화로 클릭을 판정한다."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")

    results: list[ClickEvent] = []
    calls = 0
    for change in change_events:
        if settings.max_vlm_calls and calls >= settings.max_vlm_calls:
            print(f"[WARNING] max_vlm_calls={settings.max_vlm_calls} 도달 -> 이후 생존 분류 중단")
            break

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
