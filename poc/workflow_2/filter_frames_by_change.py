"""캡처된 Tool 창 프레임에서 cv2 absdiff 기반 변화 이벤트만 추려낸다.

`poc/workflow_1/test_vlm_popup_and_cursor_on_frames.py` 의 Stage 2(필터)
부분만 떼어 낸 workflow_2 전용 테스트 하네스.

흐름:
  - 입력 폴더(`frames/*.jpg`) 의 모든 JPEG 를 파일명 순으로 정렬한다.
  - 인접 프레임끼리 그레이스케일 absdiff → threshold → dilate →
    connectedComponentsWithStats 를 돌려, 가장 큰 변화 blob 면적이
    `MIN_CHANGE_AREA_PX` 이상인 프레임만 "change event" 로 남긴다.
  - 추려진 프레임은 `recordings/filter_frames_by_change/<tag>/change_events/`
    에 rank 접두어와 함께 복사되고, `change_events.json` + `summary.json`
    이 함께 저장된다.

VLM 호출은 전혀 하지 않는다. 이후 단계에서 이 폴더를 입력으로
align-key matcher 또는 VLM 분석기를 별도 실행할 수 있도록 분리해 둔 것이다.

실행:
    uv run python poc/workflow_2/filter_frames_by_change.py
"""

import os
import shutil
import time
from pathlib import Path

import cv2

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_1 import RECORDING_DIR as WORKFLOW_1_RECORDING_DIR
from poc.workflow_1.debug_artifacts import save_debug_json
from poc.workflow_1.util import env_int, format_elapsed_ms, make_timestamp_tag

LOG_NAME = "filter_frames_by_change"
WORKFLOW_2_RECORDING_DIR = WORKFLOW_2_DIR / "recordings"
DEFAULT_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / LOG_NAME

# ====================================================================
# 분석할 캡처 폴더를 여기에 직접 적어 사용한다 (가장 우선 적용된다).
# 비워두면 workflow_2/recordings/ 의 가장 최신 서브폴더를 보고,
# 거기에도 없으면 workflow_1/recordings/capture_window_frames_tool/
# 의 가장 최근 세션을 자동 선택한다.
#
# `frames/` 하위 폴더가 있으면 그쪽을 쓰고, 그렇지 않으면 지정한
# 폴더 자체에서 *.jpg 를 찾는다.
# ====================================================================
CAPTURE_DIR_OVERRIDE = r""

DEFAULT_MIN_CHANGE_AREA_PX = env_int("FILTER_MIN_CHANGE_AREA_PX", 5000)
DEFAULT_DIFF_THRESHOLD = env_int("FILTER_DIFF_THRESHOLD", 25)
DEFAULT_DIFF_RESIZE_WIDTH = env_int("FILTER_DIFF_RESIZE_WIDTH", 1280)


def _candidate_session_dirs(root: Path) -> list[Path]:
    """root 하위에서 frames 또는 *.jpg 를 가진 서브폴더 후보를 모은다."""
    if not root.exists():
        return []
    out: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path == DEFAULT_OUTPUT_ROOT:
            continue
        if (path / "frames").is_dir() or any(path.glob("*.jpg")):
            out.append(path)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_capture_dir() -> Path | None:
    """분석할 캡처 프레임 세트 또는 frames 디렉터리를 결정한다."""
    override = (CAPTURE_DIR_OVERRIDE or "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            print(f"[INFO] CAPTURE_DIR_OVERRIDE 사용: {path}")
            return path.resolve()
        print(f"[ERROR] CAPTURE_DIR_OVERRIDE 디렉터리를 찾지 못했습니다: {path}")
        return None

    raw_path = os.getenv("FILTER_FRAMES_DIR", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] FILTER_FRAMES_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    for root in (WORKFLOW_2_RECORDING_DIR, WORKFLOW_1_RECORDING_DIR / "capture_window_frames_tool"):
        candidates = _candidate_session_dirs(root)
        if candidates:
            latest = candidates[0].resolve()
            print(f"[INFO] 최신 캡처 세트 선택: {latest}")
            return latest

    print(
        "[ERROR] 분석할 캡처 세트를 찾지 못했습니다. "
        f"확인 위치: {WORKFLOW_2_RECORDING_DIR}, "
        f"{WORKFLOW_1_RECORDING_DIR / 'capture_window_frames_tool'}"
    )
    return None


def _resolve_frames_dir(capture_dir: Path) -> Path | None:
    """실제 JPEG 프레임들이 있는 디렉터리를 결정한다."""
    if capture_dir.name == "frames":
        return capture_dir

    frames_dir = capture_dir / "frames"
    if frames_dir.is_dir():
        return frames_dir

    if any(capture_dir.glob("*.jpg")):
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
    """인접 프레임 비교로 변화 이벤트 후보를 추출한다."""
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


def _build_output_dir(capture_dir: Path) -> Path:
    """이번 필터 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    out_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{capture_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_filter() -> str:
    """cv2 변화 이벤트 필터링을 실행한다."""
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
    change_events_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Stage 1/2: 프레임 수집 완료. "
        f"capture_dir={capture_dir}, frames_dir={frames_dir}, total_frames={len(frame_paths)}"
    )

    print(
        f"[INFO] Stage 2/2: cv2 변화 이벤트 필터 시작. "
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
            "frames_dir": str(frames_dir),
            "total_frames": len(frame_paths),
            "diff_threshold": DEFAULT_DIFF_THRESHOLD,
            "diff_resize_width": DEFAULT_DIFF_RESIZE_WIDTH,
            "min_change_area_px": DEFAULT_MIN_CHANGE_AREA_PX,
            "events": events,
        },
    )

    summary_payload = {
        "capture_dir": str(capture_dir),
        "frames_dir": str(frames_dir),
        "total_frames": len(frame_paths),
        "change_events": len(events),
        "diff_threshold": DEFAULT_DIFF_THRESHOLD,
        "diff_resize_width": DEFAULT_DIFF_RESIZE_WIDTH,
        "min_change_area_px": DEFAULT_MIN_CHANGE_AREA_PX,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "change_events_dir": str(change_events_dir),
    }
    save_debug_json(output_dir / "summary.json", summary_payload)

    print(
        f"[INFO] 완료: change_events={len(events)} / total={len(frame_paths)}, "
        f"elapsed={format_elapsed_ms(started_at)}, copied_to={change_events_dir}"
    )
    return "success" if events else "no_change_events"


if __name__ == "__main__":
    raise SystemExit(0 if run_filter() in {"success", "no_change_events"} else 1)
