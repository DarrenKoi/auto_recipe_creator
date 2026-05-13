"""VLM 으로 커서 위치를 잡아내고, 변화 영역이 커서 근처인지 검사해
실제 mouse-click 이벤트만 추려내는 5장 샘플 프로브.

입력: `poc/workflow_2/filter_frames_by_change.py` 가 만든
`recordings/filter_frames_by_change/<tag>_<session>/change_events.json`
의 events 중 5개를 random sample 한다.

흐름:
  - curr_frame 을 WebP 인코딩 → VLM coarse cursor 호출
    (workflow_1.locate_cursor_in_captured_frames 의 _coarse_*_prompt 재사용)
  - cv2.absdiff(prev, curr) 를 native resolution 그레이스케일로 계산
    (threshold → dilate; filter_frames_by_change 와 같은 파이프라인)
  - 커서 중심 좌표를 중심으로 한 사각형 ROI 안의 변화 픽셀을 카운트한다.
  - changed_in_window 가 임계값 이상이면 "click event" 로 분류하고
    `click_events/` 폴더에 복사한다.
  - 각 샘플마다 overlay JPEG (cursor bbox + ROI 박스) 를 저장한다.

실행:
    uv run python poc/workflow_2/test_vlm_cursor_click_filter.py
"""

import json
import os
import random
import shutil
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_1.locate_cursor_in_captured_frames import (
    _coarse_system_prompt,
    _coarse_user_prompt,
)
from poc.workflow_1.util import env_float, env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "vlm_cursor_click_filter"
WORKFLOW_2_RECORDING_DIR = WORKFLOW_2_DIR / "recordings"
FILTER_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / "filter_frames_by_change"
DEFAULT_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / LOG_NAME

# ====================================================================
# 분석할 필터 결과 폴더 (filter_frames_by_change 가 만든 <tag>_<session>)
# 비워두면 가장 최신 폴더를 자동 선택한다.
# ====================================================================
FILTER_DIR_OVERRIDE = r""

DEFAULT_SAMPLE_COUNT = env_int("CURSOR_FILTER_SAMPLE_COUNT", 5)
DEFAULT_REQUEST_DELAY_SEC = env_float("TEST_VLM_REQUEST_DELAY_SEC", 1.0)
DEFAULT_DIFF_THRESHOLD = env_int("CURSOR_FILTER_DIFF_THRESHOLD", 25)
CURSOR_CLICK_WINDOW_PX = env_int("CURSOR_CLICK_WINDOW_PX", 200)
CLICK_MIN_CHANGED_PX = env_int("CLICK_MIN_CHANGED_PX", 1500)
DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME


def _resolve_filter_dir() -> Path | None:
    """분석할 filter_frames_by_change 결과 폴더를 결정한다."""
    override = (FILTER_DIR_OVERRIDE or "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            print(f"[INFO] FILTER_DIR_OVERRIDE 사용: {path}")
            return path.resolve()
        print(f"[ERROR] FILTER_DIR_OVERRIDE 디렉터리를 찾지 못했습니다: {path}")
        return None

    if not FILTER_OUTPUT_ROOT.exists():
        print(f"[ERROR] 필터 결과 루트가 없습니다: {FILTER_OUTPUT_ROOT}")
        return None

    candidates = sorted(
        (path for path in FILTER_OUTPUT_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[ERROR] 필터 결과 세트가 없습니다: {FILTER_OUTPUT_ROOT}")
        return None

    latest = candidates[0].resolve()
    print(f"[INFO] 최신 필터 결과 선택: {latest}")
    return latest


def _load_events(filter_dir: Path) -> list[dict]:
    """change_events.json 의 events 리스트를 로드한다."""
    path = filter_dir / "change_events.json"
    if not path.is_file():
        print(f"[ERROR] change_events.json 가 없습니다: {path}")
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not isinstance(events, list):
        print(f"[ERROR] events 가 list 가 아닙니다: {type(events)}")
        return []
    return events


def _sample_events(events: list[dict], count: int) -> list[dict]:
    """events 에서 최대 count 개를 random sample 한다."""
    if not events:
        return []
    if count <= 0 or count >= len(events):
        print(f"[INFO] 전체 events 사용: count={len(events)}")
        return list(events)
    sampled = random.sample(events, count)
    sampled.sort(key=lambda event: int(event.get("rank") or 0))
    print(f"[INFO] random sample: {count} / {len(events)}")
    return sampled


def _compute_diff_mask(prev_path: Path, curr_path: Path, threshold: int) -> "cv2.typing.MatLike | None":
    """native resolution 에서 prev/curr 의 변화 마스크 (dilated binary) 를 반환한다."""
    prev_gray = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr_gray = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
    if prev_gray is None or curr_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        target_shape = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target_shape, interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(thresh, kernel, iterations=2)


def _window_around(center_x: int, center_y: int, side: int, img_w: int, img_h: int) -> dict:
    """커서 중심을 둘러싼 정사각 ROI bbox 를 만든다."""
    half = max(1, side // 2)
    left = max(0, center_x - half)
    top = max(0, center_y - half)
    right = min(img_w, center_x + half)
    bottom = min(img_h, center_y + half)
    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
    return {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)}


def _count_changed_in_window(mask: "cv2.typing.MatLike", window: dict) -> int:
    """diff 마스크 안에서 window 영역의 변화 픽셀 수를 센다."""
    crop = mask[window["top"]:window["bottom"], window["left"]:window["right"]]
    if crop.size == 0:
        return 0
    return int(cv2.countNonZero(crop))


def _run_cursor_detection(
    *,
    image_b64: str,
    width: int,
    height: int,
    client: Workflow1VLMClient,
) -> tuple[dict, dict | None]:
    """프레임 전체에서 마우스 커서의 coarse bbox 를 탐지한다."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_coarse_system_prompt(),
        user_text=_coarse_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return parsed, None

    bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
    if bbox_1000 is None:
        return parsed, None
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


def _save_overlay(
    *,
    frame_path: Path,
    cursor_bbox: dict | None,
    click_window: dict | None,
    is_click_event: bool,
    output_path: Path,
) -> str:
    """cursor bbox + click ROI 를 frame 위에 함께 마킹한다."""
    with Image.open(frame_path) as image:
        elements: dict[str, dict] = {}
        if cursor_bbox is not None:
            elements["cursor"] = {"bbox": cursor_bbox, "center": bbox_center(cursor_bbox)}
        if click_window is not None:
            elements["click_window"] = {"bbox": click_window, "center": bbox_center(click_window)}
        colors = {
            "cursor": "cyan",
            "click_window": "lime" if is_click_event else "red",
        }
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors=colors,
            out_path=output_path,
        )
    return str(output_path)


def _build_output_dir(filter_dir: Path) -> Path:
    """이번 테스트 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    out_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{filter_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_timeline_text(results: list[dict]) -> str:
    """사람이 읽기 쉬운 분류 타임라인을 만든다."""
    lines = []
    for item in results:
        lines.append(
            f"[{float(item.get('timestamp_sec') or 0.0):>7.3f}s] "
            f"rank={int(item.get('rank') or 0):03d} "
            f"frame={int(item.get('frame_index') or 0):04d} "
            f"cursor={'Y' if item.get('cursor_bbox') else 'N'} "
            f"click={'Y' if item.get('is_click_event') else 'N'} "
            f"changed_in_window={int(item.get('changed_in_window') or 0):>6d} "
            f"focus={float(item.get('click_focus_ratio') or 0.0):.3f} "
            f"cursor_conf={item.get('cursor_confidence', '')}"
        )
    return "\n".join(lines) + "\n"


def run_test() -> str:
    """5장 random sample 에 대해 cursor 탐지 + click 분류를 실행한다."""
    started_at = time.time()
    filter_dir = _resolve_filter_dir()
    if filter_dir is None:
        return "filter_dir_not_found"

    events = _load_events(filter_dir)
    if not events:
        return "no_events"

    sampled = _sample_events(events, DEFAULT_SAMPLE_COUNT)
    if not sampled:
        return "no_sampled_events"

    output_dir = _build_output_dir(filter_dir)
    overlays_dir = output_dir / "overlays"
    results_dir = output_dir / "results"
    click_events_dir = output_dir / "click_events"
    for directory in (overlays_dir, results_dir, click_events_dir):
        directory.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE,
        model_name=DEFAULT_MODEL,
        log_name=LOG_NAME,
    )

    print(
        f"[INFO] VLM 분석 시작: service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"samples={len(sampled)}, request_delay_sec={DEFAULT_REQUEST_DELAY_SEC}, "
        f"window_px={CURSOR_CLICK_WINDOW_PX}, click_min_changed_px={CLICK_MIN_CHANGED_PX}"
    )

    results: list[dict] = []
    click_count = 0
    vlm_calls = 0

    for event in sampled:
        rank = int(event.get("rank") or 0)
        frame_index = int(event.get("frame_index") or 0)
        timestamp_sec = float(event.get("timestamp_sec") or 0.0)
        frame_path = Path(event.get("frame_path", ""))
        prev_frame_path = Path(event.get("prev_frame_path", ""))

        if not frame_path.is_file() or not prev_frame_path.is_file():
            print(f"[WARNING] 프레임 파일 누락: rank={rank}, frame={frame_path}, prev={prev_frame_path}")
            continue

        print(
            f"[INFO] VLM 분석: rank={rank:03d}, frame={frame_index:04d}, "
            f"ts={timestamp_sec:.3f}s, path={frame_path.name}"
        )

        try:
            with Image.open(frame_path) as image:
                image_b64, frame_w, frame_h = encode_image_webp(image, quality=90)
        except Exception as exc:
            print(f"[ERROR] WebP 인코딩 실패: rank={rank}, error={exc}")
            continue

        cursor_payload: dict = {}
        cursor_bbox: dict | None = None
        try:
            cursor_payload, cursor_bbox = _run_cursor_detection(
                image_b64=image_b64,
                width=frame_w,
                height=frame_h,
                client=client,
            )
        except Exception as exc:
            print(f"[ERROR] cursor detection 실패: rank={rank}, error={exc}")
        finally:
            vlm_calls += 1
            time.sleep(DEFAULT_REQUEST_DELAY_SEC)

        diff_mask = _compute_diff_mask(prev_frame_path, frame_path, DEFAULT_DIFF_THRESHOLD)
        total_changed = int(cv2.countNonZero(diff_mask)) if diff_mask is not None else 0

        click_window: dict | None = None
        changed_in_window = 0
        click_focus_ratio = 0.0
        is_click_event = False

        if cursor_bbox is not None and diff_mask is not None:
            center = bbox_center(cursor_bbox)
            click_window = _window_around(
                center.get("x", 0),
                center.get("y", 0),
                CURSOR_CLICK_WINDOW_PX,
                frame_w,
                frame_h,
            )
            changed_in_window = _count_changed_in_window(diff_mask, click_window)
            click_focus_ratio = changed_in_window / max(1, total_changed)
            is_click_event = changed_in_window >= CLICK_MIN_CHANGED_PX

        if is_click_event:
            click_count += 1
            dst = click_events_dir / f"{rank:03d}_{frame_path.name}"
            try:
                shutil.copy2(frame_path, dst)
            except Exception as exc:
                print(f"[ERROR] click event 복사 실패: rank={rank}, error={exc}")

        overlay_path = ""
        try:
            overlay_path = _save_overlay(
                frame_path=frame_path,
                cursor_bbox=cursor_bbox,
                click_window=click_window,
                is_click_event=is_click_event,
                output_path=overlays_dir / f"{rank:03d}_frame_{frame_index:04d}_overlay.jpg",
            )
        except Exception as exc:
            print(f"[ERROR] overlay 저장 실패: rank={rank}, error={exc}")

        result = {
            "rank": rank,
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "frame_path": str(frame_path),
            "prev_frame_path": str(prev_frame_path),
            "overlay_path": overlay_path,
            "cursor_payload": cursor_payload,
            "cursor_bbox": cursor_bbox or {},
            "cursor_confidence": cursor_payload.get("confidence"),
            "cursor_evidence": cursor_payload.get("evidence"),
            "click_window": click_window or {},
            "changed_in_window": changed_in_window,
            "total_changed_pixels": total_changed,
            "click_focus_ratio": round(click_focus_ratio, 4),
            "is_click_event": is_click_event,
        }
        save_debug_json(results_dir / f"{rank:03d}_frame_{frame_index:04d}.json", result)
        results.append(result)

    summary_payload = {
        "filter_dir": str(filter_dir),
        "total_events": len(events),
        "sampled": len(sampled),
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "request_delay_sec": DEFAULT_REQUEST_DELAY_SEC,
        "diff_threshold": DEFAULT_DIFF_THRESHOLD,
        "cursor_click_window_px": CURSOR_CLICK_WINDOW_PX,
        "click_min_changed_px": CLICK_MIN_CHANGED_PX,
        "vlm_calls": vlm_calls,
        "cursor_detected": sum(1 for r in results if r.get("cursor_bbox")),
        "click_events_classified": click_count,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "click_events_dir": str(click_events_dir),
    }
    save_debug_json(output_dir / "summary.json", summary_payload)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(results))

    print(
        f"[INFO] 완료: sampled={len(sampled)}, cursor_detected={summary_payload['cursor_detected']}, "
        f"click_events={click_count}, vlm_calls={vlm_calls}, "
        f"elapsed={format_elapsed_ms(started_at)}, output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run_test() == "success" else 1)
