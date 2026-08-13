"""창 종료 직전의 저신뢰 닫기 클릭 정황을 오프라인으로만 기록한다."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np


EVIDENCE_TEXT = "window_gone + top_right_change + cursor_vlm_missing"
CLOSE_REGION_WIDTH_RATIO = 0.10
CLOSE_REGION_HEIGHT_RATIO = 0.10
PROBABLE_CLOSE_CONFIDENCE = 0.35


def _load_stop_reason(capture_dir: Path) -> str:
    path = Path(capture_dir) / "recording_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    if not isinstance(manifest, Mapping):
        return ""
    return str(manifest.get("stop_reason") or "")


def _cursor_missing_for(change, click_events) -> bool:
    event = next((item for item in click_events if item.rank == change.rank), None)
    return bool(
        event is not None
        and event.cursor_source in {"vlm", "none"}
        and not event.cursor_visible
        and event.cursor_xy is None
        and event.status in {"no_click", "cursor_unavailable"}
    )


def _close_region(width, height):
    region_w = max(64, int(round(width * CLOSE_REGION_WIDTH_RATIO)))
    region_h = max(48, int(round(height * CLOSE_REGION_HEIGHT_RATIO)))
    return {
        "left": max(0, width - region_w),
        "top": 0,
        "right": width,
        "bottom": min(height, region_h),
    }


def _bbox_intersection(first: dict, second: dict) -> dict | None:
    left = max(int(first["left"]), int(second["left"]))
    top = max(int(first["top"]), int(second["top"]))
    right = min(int(first["right"]), int(second["right"]))
    bottom = min(int(first["bottom"]), int(second["bottom"]))
    if right <= left or bottom <= top:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _line_endpoint_distance(first, second) -> float:
    """두 선분의 끝점 조합 중 가장 가까운 거리를 반환한다."""
    first_points = ((first[0], first[1]), (first[2], first[3]))
    second_points = ((second[0], second[1]), (second[2], second[3]))
    return min(
        math.hypot(ax - bx, ay - by)
        for ax, ay in first_points
        for bx, by in second_points
    )


def _has_diagonal_pair(mask) -> bool:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=8,
        minLineLength=6,
        maxLineGap=3,
    )
    if lines is None:
        return False
    positive, negative = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        folded = ((angle + 90) % 180) - 90
        if 20 <= folded <= 70:
            positive.append((x1, y1, x2, y2))
        elif -70 <= folded <= -20:
            negative.append((x1, y1, x2, y2))
    return any(
        _line_endpoint_distance(first, second) <= 24
        for first in positive
        for second in negative
    )


def _difference_mask(prev_frame, curr_frame):
    if prev_frame.shape != curr_frame.shape:
        return None
    diff = cv2.absdiff(prev_frame, curr_frame)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    return mask


def infer_probable_close_click(capture_dir, candidate_events, click_events) -> dict | None:
    """세 보수적 gate가 모두 성립할 때만 비재생용 닫기 정황을 반환한다.

    이 함수는 녹화 후처리용 증거를 만들 뿐, ClickEvent나 실행 신호를 생성하지 않는다.

    candidate_events 는 마지막 원소만 본다. 호출부(run_filter)는 Stage 1.5 게이트
    생존 목록이 아니라 raw Stage 1 목록을 넘긴다 - 녹화의 진짜 마지막 변화가
    ambient/가림으로 빠졌거나 Stage 2a 상한에 잘렸을 수 있고, 생존 목록의 끝을
    쓰면 더 오래된 우상단 후보가 terminal 로 승격되기 때문이다. 그래서 이름이
    change_events 가 아니다. 넓은 입력이 gate 를 느슨하게 만들지는 않는다 -
    _cursor_missing_for 가 그 rank 의 cursor 결과가 없으면 fail closed 한다.
    """
    if _load_stop_reason(Path(capture_dir)) != "window_gone" or not candidate_events:
        return None

    change = candidate_events[-1]
    if not _cursor_missing_for(change, click_events):
        return None

    curr_frame = cv2.imread(str(change.frame_path), cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.imread(str(change.prev_frame_path), cv2.IMREAD_GRAYSCALE)
    if curr_frame is None or prev_frame is None:
        return None

    height, width = curr_frame.shape[:2]
    close_region = _close_region(width, height)
    candidate_box = _bbox_intersection(change.change_bbox, close_region)
    if candidate_box is None:
        return None

    mask = _difference_mask(prev_frame, curr_frame)
    if mask is None:
        return None
    candidate_mask = mask[
        candidate_box["top"]:candidate_box["bottom"],
        candidate_box["left"]:candidate_box["right"],
    ]
    if candidate_mask.size == 0 or not _has_diagonal_pair(candidate_mask):
        return None

    center_x = (candidate_box["left"] + candidate_box["right"]) // 2
    center_y = (candidate_box["top"] + candidate_box["bottom"]) // 2
    return {
        "t_sec": change.timestamp_sec,
        "seq": 0,
        "action": "probable_close_click",
        "coords": {"x": center_x, "y": center_y},
        "element": "Remote Monitoring close button",
        "element_source": "inferred",
        "target_kind": "ui_control",
        "region": "window_title_bar",
        "generation": 0,
        "occlusion": "unknown",
        "cursor_source": "inferred_after_vlm_miss",
        "text": None,
        "confidence": PROBABLE_CLOSE_CONFIDENCE,
        "frame": Path(change.frame_path).name,
        "source_frames": {
            "prev": Path(change.prev_frame_path).name,
            "curr": Path(change.frame_path).name,
        },
        "evidence": EVIDENCE_TEXT,
        "replayable": False,
        "candidate_box": candidate_box,
    }


def _draw_box(image, box: dict, color, label: str):
    left, top = int(box["left"]), int(box["top"])
    right, bottom = int(box["right"]) - 1, int(box["bottom"]) - 1
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    cv2.putText(
        image,
        label,
        (left, max(14, top - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def write_close_click_evidence(event: dict, change, out_dir: Path) -> list[Path]:
    """후보 프레임 오버레이와 타임라인 증거 JSON을 저장한다."""
    image = cv2.imread(str(change.frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"현재 프레임을 읽을 수 없습니다: {change.frame_path}")

    height, width = image.shape[:2]
    _draw_box(image, _close_region(width, height), (0, 200, 0), "top_right_close_region")
    _draw_box(image, change.change_bbox, (0, 165, 255), "change_bbox")
    _draw_box(image, event["candidate_box"], (0, 0, 255), "candidate_box")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "probable_close_click.jpg"
    json_path = out_dir / "probable_close_click.json"
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"증거 프레임을 저장할 수 없습니다: {image_path}")
    json_path.write_text(
        json.dumps(event, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [image_path, json_path]
