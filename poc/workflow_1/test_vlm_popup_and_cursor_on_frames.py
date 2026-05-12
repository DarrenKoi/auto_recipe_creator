"""캡처된 Remote Monitoring System 프레임에서 두 가지 VLM 능력을 테스트한다.

1. 정렬 마크 관련 팝업("Click [OK] button ... alignment mark ...") 탐지
2. 마우스 커서 coarse bbox 탐지 + 오버레이 마킹

전체 프레임이 수천 장이라 Flask VLM 프록시가 과부하로 죽지 않도록,
먼저 cv2 absdiff 로 마우스 클릭으로 화면이 바뀐 프레임만 추려내고
(`change_events/` 폴더에 복사) 그 부분에 대해서만 VLM 호출을 보낸다.
모든 VLM 호출 사이에는 `TEST_VLM_REQUEST_DELAY_SEC` (기본 1.0초) 의 대기를 둔다.

기본 실행:
  uv run python poc/workflow_1/test_vlm_popup_and_cursor_on_frames.py
"""

import os
import shutil
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from PIL import Image

from poc.workflow_1 import RECORDING_DIR
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

LOG_NAME = "test_vlm_popup_and_cursor_on_frames"
DEFAULT_CAPTURE_ROOT = RECORDING_DIR / "capture_window_frames_tool"
DEFAULT_OUTPUT_ROOT = RECORDING_DIR / LOG_NAME

DEFAULT_REQUEST_DELAY_SEC = env_float("TEST_VLM_REQUEST_DELAY_SEC", 1.0)
DEFAULT_MIN_CHANGE_AREA_PX = env_int("TEST_VLM_MIN_CHANGE_AREA_PX", 5000)
DEFAULT_DIFF_THRESHOLD = env_int("TEST_VLM_DIFF_THRESHOLD", 25)
DEFAULT_DIFF_RESIZE_WIDTH = env_int("TEST_VLM_DIFF_RESIZE_WIDTH", 1280)
DEFAULT_MAX_EVENTS = env_int("TEST_VLM_MAX_EVENTS", 0)
DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME


def _resolve_capture_dir() -> Path | None:
    """분석할 캡처 프레임 세트 또는 frames 디렉터리를 결정한다."""
    raw_path = os.getenv("TEST_VLM_FRAMES_DIR", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] TEST_VLM_FRAMES_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    if not DEFAULT_CAPTURE_ROOT.exists():
        print(f"[ERROR] 캡처 프레임 디렉터리가 없습니다: {DEFAULT_CAPTURE_ROOT}")
        return None

    candidates = sorted(
        (path for path in DEFAULT_CAPTURE_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[ERROR] 캡처 프레임 세트가 없습니다: {DEFAULT_CAPTURE_ROOT}")
        return None

    latest = candidates[0].resolve()
    print(f"[INFO] 최신 캡처 세트 선택: {latest}")
    return latest


def _resolve_frames_dir(capture_dir: Path) -> Path | None:
    """실제 JPEG 프레임들이 있는 디렉터리를 결정한다."""
    if capture_dir.name == "frames":
        return capture_dir

    frames_dir = capture_dir / "frames"
    if frames_dir.is_dir():
        return frames_dir

    jpeg_files = sorted(capture_dir.glob("*.jpg"))
    if jpeg_files:
        return capture_dir

    print(f"[ERROR] frames 디렉터리 또는 JPEG 프레임이 없습니다: {capture_dir}")
    return None


def _parse_timestamp_sec_from_name(frame_path: Path) -> float:
    """frame 파일명에 포함된 밀리초 타임스탬프를 파싱한다."""
    stem = frame_path.stem
    for part in reversed(stem.split("_")):
        if part.endswith("ms") and part[:-2].isdigit():
            return round(int(part[:-2]) / 1000.0, 3)
    return 0.0


def _parse_frame_index_from_name(frame_path: Path) -> int:
    """frame 파일명에서 frame_NNNN 부분의 인덱스를 파싱한다. 실패 시 -1."""
    parts = frame_path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "frame" and parts[1].isdigit():
        return int(parts[1])
    return -1


def _collect_frame_paths(frames_dir: Path) -> list[Path]:
    """frames 디렉터리에서 JPEG 파일들을 파일명 정렬 순으로 반환한다.

    `frame_NNNN` 번호가 연속적이지 않을 수 있으므로 (사용자가 일부 삭제),
    인덱스 산술이 아닌 정렬 순서로만 인접 프레임을 판단한다.
    """
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def _load_diff_gray(image_path: Path, resize_width: int) -> "cv2.typing.MatLike | None":
    """cv2 로 그레이스케일 + 리사이즈 한 이미지를 로드한다. 실패 시 None."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    h, w = image.shape[:2]
    if resize_width > 0 and w > resize_width:
        new_h = max(1, int(round(h * (resize_width / w))))
        image = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return image


def _compute_change_stats(
    prev_gray: "cv2.typing.MatLike",
    curr_gray: "cv2.typing.MatLike",
    diff_threshold: int,
) -> dict:
    """두 그레이스케일 프레임 간 변화량 통계를 계산한다."""
    if prev_gray.shape != curr_gray.shape:
        target_shape = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target_shape, interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    changed_pixels = int(cv2.countNonZero(dilated))
    largest_blob_area = 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_blob_area:
            largest_blob_area = area

    return {
        "changed_pixels": changed_pixels,
        "largest_blob_area_px": largest_blob_area,
        "diff_height": int(dilated.shape[0]),
        "diff_width": int(dilated.shape[1]),
    }


def _detect_change_events(
    frame_paths: list[Path],
    *,
    diff_threshold: int,
    resize_width: int,
    min_change_area_px: int,
) -> list[dict]:
    """인접 프레임 비교로 클릭 변화 이벤트 후보를 추출한다."""
    events: list[dict] = []
    if len(frame_paths) < 2:
        return events

    prev_gray = _load_diff_gray(frame_paths[0], resize_width)
    if prev_gray is None:
        print(f"[WARNING] 첫 프레임 로드 실패: {frame_paths[0]}")
        return events

    rank = 0
    for prev_path, curr_path in zip(frame_paths[:-1], frame_paths[1:]):
        curr_gray = _load_diff_gray(curr_path, resize_width)
        if curr_gray is None:
            print(f"[WARNING] 프레임 로드 실패: {curr_path}")
            prev_gray = None
            continue
        if prev_gray is None:
            prev_gray = curr_gray
            continue

        stats = _compute_change_stats(prev_gray, curr_gray, diff_threshold)
        if stats["largest_blob_area_px"] >= min_change_area_px:
            events.append(
                {
                    "rank": rank,
                    "frame_index": _parse_frame_index_from_name(curr_path),
                    "timestamp_sec": _parse_timestamp_sec_from_name(curr_path),
                    "frame_path": str(curr_path.resolve()),
                    "prev_frame_path": str(prev_path.resolve()),
                    **stats,
                }
            )
            rank += 1

        prev_gray = curr_gray

    return events


def _copy_change_events(events: list[dict], out_dir: Path) -> None:
    """탐지된 변화 이벤트 프레임을 별도 폴더에 복사한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        src = Path(event["frame_path"])
        dst = out_dir / f"{event['rank']:03d}_{src.name}"
        shutil.copy2(src, dst)
        event["copied_path"] = str(dst.resolve())


def _popup_system_prompt() -> str:
    """팝업 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a Windows semiconductor metrology tool. "
        "Return strict JSON only. "
        "Detect whether a modal dialog or popup is currently displayed asking "
        "the operator to click an OK button regarding an alignment mark, "
        "alignment key, alignment failure, or similar alignment-related instruction. "
        "Do not invent popups."
    )


def _popup_user_prompt() -> str:
    """팝업 탐지 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "popup_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "popup_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "popup_text_excerpt": "short verbatim quote of the popup message, max 200 chars",\n'
        '  "mentions_alignment_mark": true,\n'
        '  "ok_button_present": true,\n'
        '  "confidence": 0.0\n'
        "}\n"
        "If no alignment-related popup is visible, set popup_visible=false, "
        "popup_bbox=null, popup_text_excerpt=null, mentions_alignment_mark=false, "
        "ok_button_present=false, confidence=0.0."
    )


def _run_popup_detection(
    *,
    image_b64: str,
    width: int,
    height: int,
    client: Workflow1VLMClient,
) -> tuple[dict, dict | None]:
    """프레임 전체에서 정렬 관련 팝업의 bbox 를 탐지한다."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_popup_system_prompt(),
        user_text=_popup_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("popup_visible") is not True:
        return parsed, None

    bbox_1000 = normalize_bbox_1000(parsed.get("popup_bbox"))
    if bbox_1000 is None:
        return parsed, None

    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


def _run_cursor_coarse_detection(
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
    popup_bbox: dict | None,
    cursor_bbox: dict | None,
    output_path: Path,
) -> str:
    """전체 프레임 위에 popup + cursor bbox 를 함께 마킹한다."""
    with Image.open(frame_path) as image:
        elements: dict[str, dict] = {}
        if popup_bbox is not None:
            elements["popup"] = {"bbox": popup_bbox, "center": bbox_center(popup_bbox)}
        if cursor_bbox is not None:
            elements["coarse_cursor"] = {"bbox": cursor_bbox, "center": bbox_center(cursor_bbox)}
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={"popup": "yellow", "coarse_cursor": "cyan"},
            out_path=output_path,
        )
    return str(output_path)


def _build_output_dir(capture_dir: Path) -> Path:
    """이번 테스트 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    out_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{capture_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_timeline_text(per_frame_results: list[dict]) -> str:
    """사람이 읽기 쉬운 탐지 타임라인을 만든다."""
    lines = []
    for item in per_frame_results:
        popup = item.get("popup_bbox") or {}
        cursor = item.get("cursor_bbox") or {}
        lines.append(
            f"[{float(item.get('timestamp_sec') or 0.0):>7.3f}s] "
            f"rank={int(item.get('rank') or 0):03d} "
            f"frame={int(item.get('frame_index') or 0):04d} "
            f"popup={'Y' if popup else 'N'} "
            f"cursor={'Y' if cursor else 'N'} "
            f"popup_conf={item.get('popup_confidence', '')} "
            f"cursor_conf={item.get('cursor_confidence', '')} "
            f"popup_text={(item.get('popup_text_excerpt') or '')[:60]!r}"
        )
    return "\n".join(lines) + "\n"


def run_test() -> str:
    """변화 이벤트 필터링 + VLM 팝업/커서 탐지를 실행한다."""
    started_at = time.time()
    capture_dir = _resolve_capture_dir()
    if capture_dir is None:
        return "capture_dir_not_found"

    frames_dir = _resolve_frames_dir(capture_dir)
    if frames_dir is None:
        return "frames_dir_not_found"

    frame_paths = _collect_frame_paths(frames_dir)
    if len(frame_paths) < 2:
        print(f"[ERROR] 변화 비교에 최소 2장이 필요합니다: found={len(frame_paths)} in {frames_dir}")
        return "not_enough_frames"

    output_dir = _build_output_dir(capture_dir)
    change_events_dir = output_dir / "change_events"
    overlays_dir = output_dir / "overlays"
    results_dir = output_dir / "results"
    for directory in (change_events_dir, overlays_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Stage 1/3: 프레임 수집 완료. "
        f"capture_dir={capture_dir}, frames_dir={frames_dir}, total_frames={len(frame_paths)}"
    )

    print(
        f"[INFO] Stage 2/3: cv2 변화 이벤트 필터 시작. "
        f"diff_threshold={DEFAULT_DIFF_THRESHOLD}, resize_width={DEFAULT_DIFF_RESIZE_WIDTH}, "
        f"min_change_area_px={DEFAULT_MIN_CHANGE_AREA_PX}"
    )
    events = _detect_change_events(
        frame_paths,
        diff_threshold=DEFAULT_DIFF_THRESHOLD,
        resize_width=DEFAULT_DIFF_RESIZE_WIDTH,
        min_change_area_px=DEFAULT_MIN_CHANGE_AREA_PX,
    )
    _copy_change_events(events, change_events_dir)
    save_debug_json(
        output_dir / "change_events.json",
        {
            "capture_dir": str(capture_dir),
            "total_frames": len(frame_paths),
            "diff_threshold": DEFAULT_DIFF_THRESHOLD,
            "diff_resize_width": DEFAULT_DIFF_RESIZE_WIDTH,
            "min_change_area_px": DEFAULT_MIN_CHANGE_AREA_PX,
            "events": events,
        },
    )
    print(
        f"[INFO] Stage 2/3 완료: change_events={len(events)} / total={len(frame_paths)}, "
        f"copied_to={change_events_dir}"
    )

    if not events:
        print("[WARNING] 변화 이벤트가 0건이라 VLM 호출을 건너뜁니다.")
        save_debug_json(
            output_dir / "summary.json",
            {
                "capture_dir": str(capture_dir),
                "total_frames": len(frame_paths),
                "change_events": 0,
                "vlm_service": DEFAULT_SERVICE,
                "vlm_model_name": DEFAULT_MODEL,
                "request_delay_sec": DEFAULT_REQUEST_DELAY_SEC,
                "popup_detected": 0,
                "cursor_detected": 0,
                "vlm_calls": 0,
                "elapsed": format_elapsed_ms(started_at),
                "output_dir": str(output_dir),
            },
        )
        return "no_change_events"

    max_events = DEFAULT_MAX_EVENTS if DEFAULT_MAX_EVENTS > 0 else len(events)
    target_events = events[:max_events]
    estimated_sec = len(target_events) * 2 * DEFAULT_REQUEST_DELAY_SEC
    print(
        f"[INFO] Stage 3/3: VLM 팝업+커서 탐지 시작. "
        f"service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"events_to_process={len(target_events)} (max={DEFAULT_MAX_EVENTS or 'all'}), "
        f"request_delay_sec={DEFAULT_REQUEST_DELAY_SEC}, "
        f"estimated_sleep_total_sec={estimated_sec:.1f}"
    )

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE,
        model_name=DEFAULT_MODEL,
        log_name=LOG_NAME,
    )

    per_frame_results: list[dict] = []
    popup_detected = 0
    cursor_detected = 0
    vlm_calls = 0

    for event in target_events:
        rank = int(event["rank"])
        frame_path = Path(event["frame_path"])
        frame_index = int(event.get("frame_index") or 0)
        timestamp_sec = float(event.get("timestamp_sec") or 0.0)

        if not frame_path.is_file():
            print(f"[WARNING] 프레임 파일 누락: {frame_path}")
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

        popup_payload: dict = {}
        popup_bbox: dict | None = None
        try:
            popup_payload, popup_bbox = _run_popup_detection(
                image_b64=image_b64,
                width=frame_w,
                height=frame_h,
                client=client,
            )
        except Exception as exc:
            print(f"[ERROR] popup detection 실패: rank={rank}, error={exc}")
        finally:
            vlm_calls += 1
            time.sleep(DEFAULT_REQUEST_DELAY_SEC)

        cursor_payload: dict = {}
        cursor_bbox: dict | None = None
        try:
            cursor_payload, cursor_bbox = _run_cursor_coarse_detection(
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

        if popup_bbox is not None:
            popup_detected += 1
        if cursor_bbox is not None:
            cursor_detected += 1

        overlay_path = ""
        try:
            overlay_path = _save_overlay(
                frame_path=frame_path,
                popup_bbox=popup_bbox,
                cursor_bbox=cursor_bbox,
                output_path=overlays_dir / f"{rank:03d}_frame_{frame_index:04d}_overlay.jpg",
            )
        except Exception as exc:
            print(f"[ERROR] overlay 저장 실패: rank={rank}, error={exc}")

        result = {
            "rank": rank,
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "frame_path": str(frame_path),
            "copied_path": event.get("copied_path", ""),
            "overlay_path": overlay_path,
            "popup_payload": popup_payload,
            "popup_bbox": popup_bbox or {},
            "popup_confidence": popup_payload.get("confidence"),
            "popup_text_excerpt": popup_payload.get("popup_text_excerpt"),
            "mentions_alignment_mark": popup_payload.get("mentions_alignment_mark"),
            "ok_button_present": popup_payload.get("ok_button_present"),
            "cursor_payload": cursor_payload,
            "cursor_bbox": cursor_bbox or {},
            "cursor_confidence": cursor_payload.get("confidence"),
        }
        result_path = results_dir / f"{rank:03d}_frame_{frame_index:04d}.json"
        save_debug_json(result_path, result)
        per_frame_results.append(result)

    summary_payload = {
        "capture_dir": str(capture_dir),
        "frames_dir": str(frames_dir),
        "total_frames": len(frame_paths),
        "change_events": len(events),
        "events_processed": len(per_frame_results),
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "request_delay_sec": DEFAULT_REQUEST_DELAY_SEC,
        "diff_threshold": DEFAULT_DIFF_THRESHOLD,
        "diff_resize_width": DEFAULT_DIFF_RESIZE_WIDTH,
        "min_change_area_px": DEFAULT_MIN_CHANGE_AREA_PX,
        "vlm_calls": vlm_calls,
        "popup_detected": popup_detected,
        "cursor_detected": cursor_detected,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
    }
    save_debug_json(output_dir / "summary.json", summary_payload)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(per_frame_results))

    print(
        f"[INFO] 완료: processed_events={len(per_frame_results)}, "
        f"popup_detected={popup_detected}, cursor_detected={cursor_detected}, "
        f"vlm_calls={vlm_calls}, elapsed={format_elapsed_ms(started_at)}, "
        f"output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run_test() == "success" else 1)
