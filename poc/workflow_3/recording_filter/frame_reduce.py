"""STAGE 1 — cv2 absdiff 변화 이벤트로 녹화 프레임을 축소한다.

poc/workflow_2/filter_frames_by_change.py 의 엔진을 이식하면서, 가장 큰 변화
blob 의 *위치* bbox(change_bbox)를 native 픽셀로 함께 산출한다(현재 bench 는
면적만 계산하고 위치를 버린다). 이 bbox 는 Stage 2a 의 ROI 참고치이자
향후 Stage 2b(OCR-diff) 의 crop 영역으로 재사용된다.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2

from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


@dataclass
class ChangeEvent:
    """Stage 1 생존 프레임 1건."""

    rank: int                 # 0부터, 생존 순서
    frame_path: str           # 현재(curr) 프레임 절대경로
    prev_frame_path: str      # 직전 프레임 절대경로
    timestamp_sec: float      # 파일명 <elapsed_ms> 복원
    frame_index: int          # 파일명 seq (없으면 -1)
    change_bbox: dict         # native px {left,top,right,bottom}
    largest_blob_area_px: int
    changed_pixels: int


def _parse_timestamp_sec(frame_path: Path) -> float:
    """파일명 끝의 <ms> 토큰에서 초 단위 타임스탬프를 파싱한다."""
    for part in reversed(frame_path.stem.split("_")):
        if part.endswith("ms") and part[:-2].isdigit():
            return round(int(part[:-2]) / 1000.0, 3)
    return 0.0


def _parse_frame_index(frame_path: Path) -> int:
    """파일명에서 seq 정수를 파싱한다(<tag>_rcs_<seq>_...). 실패 시 -1."""
    parts = frame_path.stem.split("_")
    for i, part in enumerate(parts):
        if part == "rcs" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """frames 디렉터리의 JPEG 를 파일명 정렬 순으로 반환한다."""
    return sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def _load_diff_gray(image_path: Path, resize_width: int):
    """grayscale 로 로드하고 resize_width 로 다운스케일한다.

    반환: (resized_gray, native_w, native_h) 또는 None(로드 실패).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    native_h, native_w = image.shape[:2]
    if resize_width > 0 and native_w > resize_width:
        new_h = max(1, int(round(native_h * (resize_width / native_w))))
        image = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return image, native_w, native_h


def _largest_blob_stats(dilated):
    """dilate 된 이진 마스크에서 (면적, bbox) 를 반환한다. 변화 없으면 (0, zero-bbox)."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    best_label, best_area = -1, 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area, best_label = area, label
    if best_label < 0:
        return 0, {"left": 0, "top": 0, "right": 0, "bottom": 0}
    left = int(stats[best_label, cv2.CC_STAT_LEFT])
    top = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return best_area, {"left": left, "top": top, "right": left + width, "bottom": top + height}


def _compute_change(prev_gray, curr_gray, diff_threshold: int):
    """두 grayscale 프레임의 (changed_px, blob_area, blob_bbox(diff-space)) 를 구한다."""
    if prev_gray.shape != curr_gray.shape:
        target = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target, interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    changed_px = int(cv2.countNonZero(dilated))
    area, bbox = _largest_blob_stats(dilated)
    diff_h, diff_w = dilated.shape[:2]
    return changed_px, area, bbox, diff_w, diff_h


def _scale_bbox(bbox: dict, native_w: int, native_h: int, diff_w: int, diff_h: int) -> dict:
    """diff-space bbox 를 native 픽셀로 스케일한다."""
    sx = native_w / diff_w if diff_w else 1.0
    sy = native_h / diff_h if diff_h else 1.0
    return {
        "left": int(round(bbox["left"] * sx)),
        "top": int(round(bbox["top"] * sy)),
        "right": int(round(bbox["right"] * sx)),
        "bottom": int(round(bbox["bottom"] * sy)),
    }


def reduce_frames(frames_dir: Path, settings: RecordingFilterSettings) -> list[ChangeEvent]:
    """인접 프레임 변화로 ChangeEvent 목록을 만든다(변화 없는 프레임은 탈락)."""
    frames_dir = Path(frames_dir)
    frame_paths = collect_frame_paths(frames_dir)
    if len(frame_paths) < 2:
        print(f"[WARNING] 변화 비교에 최소 2장 필요: found={len(frame_paths)} in {frames_dir}")
        return []

    events: list[ChangeEvent] = []
    loaded = _load_diff_gray(frame_paths[0], settings.resize_width)
    if loaded is None:
        print(f"[WARNING] 첫 프레임 로드 실패: {frame_paths[0]}")
        return []
    prev_gray = loaded[0]
    rank = 0
    for prev_path, curr_path in zip(frame_paths[:-1], frame_paths[1:]):
        loaded = _load_diff_gray(curr_path, settings.resize_width)
        if loaded is None:
            print(f"[WARNING] 프레임 로드 실패: {curr_path}")
            prev_gray = None
            continue
        curr_gray, native_w, native_h = loaded
        if prev_gray is None:
            prev_gray = curr_gray
            continue

        changed_px, area, bbox_diff, diff_w, diff_h = _compute_change(
            prev_gray, curr_gray, settings.diff_threshold
        )
        if area >= settings.min_change_area_px:
            events.append(
                ChangeEvent(
                    rank=rank,
                    frame_path=str(curr_path.resolve()),
                    prev_frame_path=str(prev_path.resolve()),
                    timestamp_sec=_parse_timestamp_sec(curr_path),
                    frame_index=_parse_frame_index(curr_path),
                    change_bbox=_scale_bbox(bbox_diff, native_w, native_h, diff_w, diff_h),
                    largest_blob_area_px=area,
                    changed_pixels=changed_px,
                )
            )
            rank += 1
        prev_gray = curr_gray

    print(f"[INFO] Stage 1 완료: change_events={len(events)} / pairs={len(frame_paths) - 1}")
    return events
