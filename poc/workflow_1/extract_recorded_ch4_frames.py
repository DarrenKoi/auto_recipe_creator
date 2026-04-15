"""녹화된 CH4 영상을 샘플 프레임으로 추출한다.

기본 실행:
  uv run python poc/workflow_1/extract_recorded_ch4_frames.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from poc.workflow_1 import DEBUG_IMAGE_DIR, WORKFLOW_1_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text
from poc.workflow_1.util import env_float, env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.video_frame_extractor import ExtractorConfig, VideoFrameExtractor

load_dotenv()

LOG_NAME = "extract_recorded_ch4_frames"
DEFAULT_RECORDING_ROOT = WORKFLOW_1_DIR / "recordings"
DEFAULT_FRAME_INTERVAL_SEC = env_float("CH4_EXTRACT_FRAME_INTERVAL_SEC", 0.25)
DEFAULT_MAX_FRAMES = env_int("CH4_EXTRACT_MAX_FRAMES", 0)
DEFAULT_OUTPUT_DIR = DEBUG_IMAGE_DIR / LOG_NAME
DEFAULT_OUTPUT_FORMAT = os.getenv("CH4_EXTRACT_OUTPUT_FORMAT", "jpg").strip() or "jpg"
DEFAULT_OUTPUT_QUALITY = env_int("CH4_EXTRACT_OUTPUT_QUALITY", 90)


def _resolve_video_path() -> Path | None:
    """추출할 동영상 파일 경로를 결정한다."""
    raw_path = os.getenv("CH4_EXTRACT_VIDEO_PATH", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
        print(f"[ERROR] CH4_EXTRACT_VIDEO_PATH 파일을 찾지 못했습니다: {path}")
        return None

    if not DEFAULT_RECORDING_ROOT.exists():
        print(f"[ERROR] recordings 디렉터리가 없습니다: {DEFAULT_RECORDING_ROOT}")
        return None

    candidates = sorted(
        (
            path for path in DEFAULT_RECORDING_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[ERROR] 추출할 동영상이 없습니다: {DEFAULT_RECORDING_ROOT}")
        return None

    latest = candidates[0].resolve()
    print(f"[INFO] 최신 동영상 선택: {latest}")
    return latest


def _build_output_dir(video_path: Path) -> Path:
    """이번 프레임 추출 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    out_dir = DEFAULT_OUTPUT_DIR / f"{tag}_{video_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_review_overlay(
    *,
    frame_path: str,
    frame_id: str,
    frame_number: int,
    timestamp_sec: float,
    change_score: float,
    output_dir: Path,
) -> str:
    """기본 정보가 포함된 리뷰용 전체 프레임 이미지를 저장한다."""
    with Image.open(frame_path) as image:
        review_img = image.convert("RGB")
        draw = ImageDraw.Draw(review_img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        header = (
            f"frame={frame_number}  ts={timestamp_sec:.2f}s  "
            f"change={change_score:.4f}"
        )
        draw.rectangle([(0, 0), (review_img.size[0], 30)], fill="black")
        draw.text((8, 6), header, fill="yellow", font=font)

        output_path = output_dir / f"{frame_id}_review.jpg"
        review_img.save(output_path, format="JPEG", quality=95)
        return str(output_path)


def _build_contact_sheet(
    *,
    frame_items: list[dict],
    output_path: Path,
    thumb_width: int = 320,
    columns: int = 3,
) -> str | None:
    """추출된 전체 프레임을 한 장의 컨택트시트로 저장한다."""
    if not frame_items:
        return None

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    thumbs: list[Image.Image] = []
    for item in frame_items:
        source_path = item.get("review_overlay_path") or item.get("frame_path")
        if not source_path:
            continue

        with Image.open(source_path) as image:
            base = image.convert("RGB")
            ratio = thumb_width / max(1, base.size[0])
            thumb_height = max(1, int(round(base.size[1] * ratio)))
            thumb = base.resize((thumb_width, thumb_height))

        label = (
            f"f={item['frame_number']} "
            f"t={item['timestamp_sec']:.2f}s "
            f"c={item['change_score']:.3f}"
        )
        labeled = Image.new("RGB", (thumb_width, thumb_height + 24), color="black")
        labeled.paste(thumb, (0, 24))
        draw = ImageDraw.Draw(labeled)
        draw.text((6, 4), label, fill="yellow", font=font)
        thumbs.append(labeled)

    if not thumbs:
        return None

    columns = max(1, columns)
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(thumb.size[0] for thumb in thumbs)
    cell_h = max(thumb.size[1] for thumb in thumbs)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), color=(32, 32, 32))

    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        sheet.paste(thumb, (col * cell_w, row * cell_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92)
    return str(output_path)


def _build_timeline_text(video_path: Path, frame_items: list[dict]) -> str:
    """사람이 읽기 쉬운 프레임 타임라인 텍스트를 만든다."""
    lines = [f"video={video_path}"]
    for item in frame_items:
        lines.append(
            f"[{item['timestamp_sec']:>7.2f}s] "
            f"frame={item['frame_number']} "
            f"change={item['change_score']:.4f}"
        )
    return "\n".join(lines) + "\n"


def extract_frames() -> str:
    """CH4 녹화 영상에서 샘플 프레임을 추출한다."""
    started_at = time.time()
    video_path = _resolve_video_path()
    if video_path is None:
        return "video_not_found"

    output_dir = _build_output_dir(video_path)
    frames_dir = output_dir / "frames"
    review_dir = output_dir / "review_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    extractor_config = ExtractorConfig(
        frame_interval=DEFAULT_FRAME_INTERVAL_SEC,
        output_format=DEFAULT_OUTPUT_FORMAT,
        quality=DEFAULT_OUTPUT_QUALITY,
    )
    max_frames = DEFAULT_MAX_FRAMES if DEFAULT_MAX_FRAMES > 0 else None

    print(
        f"[INFO] 프레임 추출 시작: path={video_path}, "
        f"frame_interval={DEFAULT_FRAME_INTERVAL_SEC}s, "
        f"max_frames={max_frames or 'all'}"
    )

    frame_items: list[dict] = []
    with VideoFrameExtractor(extractor_config) as extractor:
        metadata = extractor.open(video_path)
        print(
            f"[INFO] 비디오 메타데이터: duration={metadata.duration:.2f}s, "
            f"fps={metadata.fps:.2f}, size={metadata.width}x{metadata.height}, "
            f"codec={metadata.codec}"
        )

        for frame_data in extractor.extract_frames(max_frames=max_frames):
            frame_path = extractor.save_frame(frame_data, frames_dir)
            review_path = _save_review_overlay(
                frame_path=frame_path,
                frame_id=frame_data.frame_id,
                frame_number=int(frame_data.frame_number),
                timestamp_sec=float(frame_data.timestamp),
                change_score=float(frame_data.change_score or 0.0),
                output_dir=review_dir,
            )
            frame_items.append(
                {
                    "frame_id": frame_data.frame_id,
                    "frame_number": frame_data.frame_number,
                    "timestamp_sec": round(float(frame_data.timestamp), 3),
                    "change_score": round(float(frame_data.change_score or 0.0), 6),
                    "frame_type": frame_data.frame_type.value,
                    "frame_path": frame_path,
                    "review_overlay_path": review_path,
                }
            )

    summary = {
        "video_path": str(video_path),
        "frame_interval_sec": DEFAULT_FRAME_INTERVAL_SEC,
        "max_frames": max_frames,
        "sampled_frames": len(frame_items),
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "frame_results": frame_items,
    }
    summary["contact_sheet_path"] = _build_contact_sheet(
        frame_items=frame_items,
        output_path=output_dir / "contact_sheet.jpg",
    ) or ""
    save_debug_json(output_dir / "summary.json", summary)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(video_path, frame_items))

    print(
        f"[INFO] 프레임 추출 완료: sampled={len(frame_items)}, "
        f"elapsed={format_elapsed_ms(started_at)}, output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if extract_frames() == "success" else 1)
