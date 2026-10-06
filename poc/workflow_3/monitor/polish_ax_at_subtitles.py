"""`AX AT_rev4_분석자동화.mp4` 에 구간별 제목(좌상단) + 설명(하단) 자막을 입힌다.

자막 원본은 옆의 txt 다 - 문구/시간은 txt 만 고치고 다시 돌린다.
그리기는 `polish_demo_video` 것을 그대로 쓴다(포크 금지). 원본 크기/fps 를 유지하고,
프레임을 다시 인코딩하므로 오디오는 빠진다.

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_ax_at_subtitles.py
  -> video/AX AT_rev4_분석자동화_자막_<시각>.mp4
"""

import sys
import textwrap
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다.
# ===========================================================================
VIDEO_DIR = _REPO_ROOT / "video"
INPUT = "AX AT_rev4_분석자동화.mp4"
SUBTITLES = "AX AT_rev4_분석자동화_자막.txt"  # 줄마다 "MM:SS~MM:SS | 제목 | 설명"
OUTPUT = ""            # 비우면 video/AX AT_rev4_분석자동화_자막_<시각>.mp4
WRAP_CHARS = 40        # 설명 한 줄 최대 글자 수(넘으면 줄바꿈)
TITLE_MARGIN = 0.03    # 제목의 좌/상 여백(화면 폭 대비)
CRF = 18               # 낮을수록 고화질(파일 큼)


def _sec(stamp: str) -> float:
    minutes, seconds = stamp.strip().split(":")
    return int(minutes) * 60 + float(seconds)


def parse_subtitles(text: str) -> list[tuple]:
    """txt -> [(시작초, 끝초, 제목, 설명)]. 끝 시각은 그 초를 포함한다(00:06 = 7.0초 직전까지)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        span, title, body = (part.strip() for part in line.split("|", 2))
        start, end = span.split("~")
        rows.append((_sec(start), _sec(end) + 1.0, title, body))
    return rows


def draw_section(frame: np.ndarray, t: float, rows: list, wrap: int = WRAP_CHARS) -> None:
    """t 에 해당하는 구간의 제목(좌상단)과 설명(하단)을 프레임에 그린다."""
    row = polish.active_subtitle(t, rows)
    if row is None:
        return
    start, end, title, body = row
    level = polish.fade_level(t, start, end, polish.SUBTITLE_FADE_SEC)
    width = frame.shape[1]
    margin = int(width * TITLE_MARGIN)
    polish.overlay_rgba(frame, polish.subtitle_patch(title, width), margin, margin, level)
    polish.draw_subtitle(frame, "\n".join(textwrap.wrap(body, wrap, break_on_hyphens=False)), level)


def main() -> str:
    import imageio_ffmpeg

    src = VIDEO_DIR / INPUT
    if not src.is_file():
        raise FileNotFoundError(f"자막을 입힐 mp4 가 없습니다: {src}")
    rows = parse_subtitles((VIDEO_DIR / SUBTITLES).read_text(encoding="utf-8"))
    out_path = Path(OUTPUT or VIDEO_DIR / f"{src.stem}_자막_{time.strftime('%y%m%d_%H%M%S')}.mp4")

    reader = imageio_ffmpeg.read_frames(str(src))
    meta = next(reader)
    (fw, fh), fps = meta["size"], meta["fps"]
    writer = imageio_ffmpeg.write_frames(
        str(out_path), (fw, fh), fps=fps, codec="libx264", quality=None, macro_block_size=2,
        pix_fmt_out="yuv420p", ffmpeg_log_level="error",
        output_params=["-preset", "slow", "-crf", str(CRF), "-profile:v", "high",
                       "-movflags", "+faststart"],
    )
    writer.send(None)
    written = 0
    try:
        for raw in reader:
            frame = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3).copy()
            draw_section(frame, written / fps, rows)
            writer.send(frame)
            written += 1
    finally:
        reader.close()
        writer.close()
    print(f"[INFO] 자막 {len(rows)}구간, {written / fps:.1f}s ({fw}x{fh} {fps:g}fps) -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
