"""녹화 프레임(JPEG) 을 시연용 mp4 로 조립한다 - 오프라인, 루프와 무관.

`RecordingSession` 은 알람 사이클과 수동 녹화에서 tool 창을
`{tag}_rcs_{seq:04d}_{elapsed_ms:08d}ms.jpg` 형식으로 적재한다. 파일명에 경과시간이
박혀 있으므로 별도 메타 없이 **실제 시간축을 그대로 복원**할 수 있다.

왜 '순서대로 고정 fps' 가 아니라 리샘플링인가:
  녹화는 화면이 바뀔 때만 저장하는 change-triggered 샘플링이라 프레임 간격이
  불규칙하다. 파일 순서대로 같은 간격으로 붙이면 **빠른 조작 구간은 늘어지고 정지
  구간은 순식간에 지나가** 실제와 정반대인 영상이 나온다. 그래서 출력 프레임 k 를
  소스 시간축으로 역매핑한다(그 시각에 화면에 떠 있던 프레임을 고른다).

정지 구간 압축(`MAX_HOLD_SEC`):
  align fail 은 장비가 멈춘 상태라 프레임이 수십 초씩 정지한다. 그 구간을 실시간으로
  담으면 아무도 안 본다. 그래서 프레임 1장이 붙들 수 있는 시간에 상한을 두고, 잘려나간
  시간은 화면에 `>> +NNs skipped` 로 **정직하게 표시**한다(몰래 자르지 않는다).

사용법:
  uv run python poc/workflow_3/monitor/make_demo_video.py
  DEMO_VIDEO_INPUT_DIR=<recording 경로> uv run python poc/workflow_3/monitor/make_demo_video.py

  입력을 지정하지 않으면 ALIGN_IMAGES_DIR 아래에서 **가장 최근 recording 폴더**를
  자동으로 고른다(후보 목록도 함께 출력하므로 다른 것을 고르려면 env 로 지정).

주요 env:
  DEMO_VIDEO_INPUT_DIR    입력 recording 폴더 (미지정 시 자동 탐색)
  DEMO_VIDEO_OUTPUT       출력 파일 (기본 <input>/demo.mp4)
  DEMO_VIDEO_SPEED        1.0=실시간, 2.0=2배속 (기본 1.0)
  DEMO_VIDEO_FPS          출력 fps (기본 15)
  DEMO_VIDEO_MAX_HOLD_SEC 한 프레임 최대 유지 시간 = 정지 구간 압축 (기본 2.0, 0=압축 안 함)
  DEMO_VIDEO_START_SEC    이 경과시간 이후만 (기본 0)
  DEMO_VIDEO_END_SEC      이 경과시간 이전까지 (기본 0=끝까지)
  DEMO_VIDEO_MAX_WIDTH    가로 상한 (기본 1920, 0=원본)
  DEMO_VIDEO_OVERLAY      타임스탬프 burn-in (기본 1)
  DEMO_VIDEO_LABEL        좌상단 고정 라벨 (ASCII 만, 기본 없음)
  DEMO_VIDEO_TAIL_HOLD_SEC 마지막 프레임 유지 (기본 2.0)

주의: burn-in 문자는 cv2.putText 라 **ASCII 만** 렌더링된다(한글은 깨진다).
      한글 자막이 필요하면 편집 도구에서 얹을 것.
"""

import os
import re
import sys
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import ALIGN_IMAGES_DIR

# RecordingSession 이 쓰는 파일명 규약. seq 는 4자리 기준이지만 9999 를 넘으면
# 자릿수가 늘어나므로 하한만 걸고 받는다(경과시간은 항상 8자리 고정).
_FRAME_RE = re.compile(r"_(\d{4,})_(\d{8})ms\.jpg$", re.IGNORECASE)

# 자동 탐색 시 콘솔에 보여줄 최근 후보 수.
_CANDIDATE_PREVIEW = 5


def _env_float(name: str, default: float) -> float:
    """env 를 float 으로 읽는다 (빈값/파싱실패는 기본값)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARNING] {name}={raw!r} 파싱 실패 - 기본값 {default} 사용")
        return default


def _env_int(name: str, default: int) -> int:
    """env 를 int 로 읽는다 (빈값/파싱실패는 기본값)."""
    return int(_env_float(name, float(default)))


def _env_flag(name: str, default: bool) -> bool:
    """env 를 bool 로 읽는다 ('0'/'false'/'no' 만 거짓)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


# ------------------------------------------------------------------
# 입력 탐색.
# ------------------------------------------------------------------


def find_recording_dirs(root: Path) -> list[Path]:
    """root 아래의 recording 폴더를 최근 순으로 찾는다.

    알람 사이클(`captured_img_from_rcs/<tag>/recording`) 과 수동 녹화
    (`_manual/<tag>/recording`) 가 같은 이름을 쓰므로 한 번에 잡힌다.
    """
    if not root.exists():
        return []
    dirs = [p for p in root.rglob("recording") if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def resolve_input_dir() -> Path | None:
    """DEMO_VIDEO_INPUT_DIR 또는 가장 최근 recording 폴더를 고른다."""
    raw = os.environ.get("DEMO_VIDEO_INPUT_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_dir():
            print(f"[ERROR] 입력 폴더가 없습니다: {path}")
            return None
        return path

    print(f"[INFO] DEMO_VIDEO_INPUT_DIR 미지정 - {ALIGN_IMAGES_DIR} 에서 자동 탐색")
    candidates = find_recording_dirs(ALIGN_IMAGES_DIR)
    if not candidates:
        print(f"[ERROR] recording 폴더를 찾지 못했습니다: {ALIGN_IMAGES_DIR}")
        return None

    for idx, path in enumerate(candidates[:_CANDIDATE_PREVIEW]):
        frames = len(list(path.glob("*.jpg")))
        mark = "<- 선택" if idx == 0 else ""
        print(f"[INFO]   {frames:5d} frames  {path} {mark}")
    if len(candidates) > _CANDIDATE_PREVIEW:
        print(f"[INFO]   ... 외 {len(candidates) - _CANDIDATE_PREVIEW}개")
    print("[INFO] 다른 것을 쓰려면 DEMO_VIDEO_INPUT_DIR 로 지정하세요.")
    return candidates[0]


def scan_frames(input_dir: Path) -> list[tuple[float, Path]]:
    """recording 폴더의 프레임을 (경과초, 경로) 목록으로 읽는다.

    파일명 규약에 맞지 않는 jpg(디버그 산출물 등)는 조용히 건너뛴다.
    """
    items: list[tuple[float, int, Path]] = []
    for path in input_dir.glob("*.jpg"):
        matched = _FRAME_RE.search(path.name)
        if not matched:
            continue
        items.append((int(matched.group(2)) / 1000.0, int(matched.group(1)), path))
    items.sort(key=lambda item: (item[0], item[1]))
    return [(t_sec, path) for t_sec, _, path in items]


def trim_frames(
    frames: list[tuple[float, Path]], start_sec: float, end_sec: float
) -> list[tuple[float, Path]]:
    """경과시간 구간으로 프레임을 자른다 (end_sec<=0 이면 끝까지)."""
    if start_sec <= 0 and end_sec <= 0:
        return frames
    kept = [
        (t_sec, path)
        for t_sec, path in frames
        if t_sec >= start_sec and (end_sec <= 0 or t_sec <= end_sec)
    ]
    print(
        f"[INFO] 구간 트림: {start_sec:.1f}s ~ "
        f"{'끝' if end_sec <= 0 else f'{end_sec:.1f}s'} -> {len(kept)}/{len(frames)} 프레임"
    )
    return kept


# ------------------------------------------------------------------
# 시간축.
# ------------------------------------------------------------------


def build_timeline(
    frames: list[tuple[float, Path]], max_hold_sec: float, tail_hold_sec: float
) -> tuple[list[float], list[float], float, float]:
    """압축된 출력 시간축을 만든다.

    프레임 i 가 실제로 화면에 떠 있던 시간(다음 프레임까지의 간격)을 `max_hold_sec`
    으로 자른 것이 출력에서의 유지 시간이다. 반환값:

      starts      프레임 i 의 출력 시작 시각 (압축 후, 초)
      skipped     프레임 i 에서 잘려나간 시간 (초; 0 이면 손대지 않음)
      total       전체 출력 길이 (초, tail 포함)
      skipped_total 잘려나간 총 시간 (초)
    """
    starts = [0.0]
    skipped = []
    skipped_total = 0.0
    for idx in range(len(frames) - 1):
        gap = max(0.0, frames[idx + 1][0] - frames[idx][0])
        held = min(gap, max_hold_sec) if max_hold_sec > 0 else gap
        skipped.append(gap - held)
        skipped_total += gap - held
        starts.append(starts[-1] + held)
    skipped.append(0.0)  # 마지막 프레임은 다음이 없으므로 자를 것도 없다.
    total = starts[-1] + max(0.0, tail_hold_sec)
    return starts, skipped, total, skipped_total


# ------------------------------------------------------------------
# 이미지.
# ------------------------------------------------------------------


def read_image(path: Path):
    """JPEG 를 읽는다 (Windows 비ASCII 경로 대응).

    `cv2.imread` 는 Windows 에서 비ASCII 경로를 조용히 실패시키고 None 을 돌려준다.
    오피스 경로에 한글이 섞일 수 있으므로 바이트로 읽어 디코드한다.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        print(f"[WARNING] 프레임 읽기 실패({path.name}): {exc}")
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def target_size(image, max_width: int) -> tuple[int, int]:
    """출력 해상도를 정한다 (가로 상한 + 짝수 보정).

    mp4 코덱은 홀수 해상도를 거부하는 경우가 있어 짝수로 내림한다.
    """
    height, width = image.shape[:2]
    if 0 < max_width < width:
        height = int(round(height * max_width / width))
        width = max_width
    return width - (width % 2), height - (height % 2)


def draw_overlay(
    image, t_sec: float, index: int, total_frames: int, skipped_sec: float, label: str
):
    """경과시간/프레임번호/압축표시를 burn-in 한다 (ASCII 전용).

    cv2.putText 는 한글을 렌더링하지 못하므로 의도적으로 ASCII 만 쓴다.
    잘려나간 정지 시간을 함께 표시해 '몰래 편집한 영상' 이 되지 않게 한다.
    """
    height, width = image.shape[:2]
    scale = max(0.5, min(1.2, width / 1600.0))
    thickness = max(1, int(round(scale * 2)))
    text = f"t={t_sec:7.1f}s  f={index + 1}/{total_frames}"
    if skipped_sec >= 1.0:
        text += f"   >> +{skipped_sec:.0f}s skipped"

    baseline_y = height - int(18 * scale)
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(
        image,
        (int(10 * scale), baseline_y - text_h - int(10 * scale)),
        (int(20 * scale) + text_w, baseline_y + int(8 * scale)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image, text, (int(15 * scale), baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

    if label:
        cv2.putText(
            image, label, (int(15 * scale), int(40 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 255), thickness, cv2.LINE_AA,
        )
    return image


# ------------------------------------------------------------------
# 인코딩.
# ------------------------------------------------------------------


def open_writer(out_path: Path, fps: float, size: tuple[int, int]):
    """VideoWriter 를 연다 (mp4v 실패 시 MJPG/avi 로 폴백).

    오피스 PC 의 OpenCV 빌드에 따라 mp4v 가 없을 수 있다. 영상 자체를 못 만드는 것보다
    avi 라도 나오는 편이 낫고, 어느 쪽으로 떨어졌는지는 콘솔에 남긴다.
    """
    if not out_path.name.isascii():
        print("[WARNING] 출력 경로에 비ASCII 문자가 있습니다 - 인코딩 실패 시 ASCII 경로로 바꿔보세요")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, out_path
    writer.release()

    fallback = out_path.with_suffix(".avi")
    print(f"[WARNING] mp4v 인코더 사용 불가 - MJPG/avi 로 폴백: {fallback.name}")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        return writer, fallback
    writer.release()
    return None, out_path


def render(
    frames: list[tuple[float, Path]],
    out_path: Path,
    *,
    fps: float,
    speed: float,
    max_hold_sec: float,
    tail_hold_sec: float,
    max_width: int,
    overlay: bool,
    label: str,
) -> str:
    """프레임 목록을 영상으로 인코딩한다. 성공 시 'success'."""
    starts, skipped, total, skipped_total = build_timeline(
        frames, max_hold_sec, tail_hold_sec
    )
    source_span = frames[-1][0] - frames[0][0]
    out_duration = total / speed
    print(
        f"[INFO] 원본 {source_span:.1f}s ({len(frames)} frames) -> "
        f"압축 {starts[-1]:.1f}s (정지 {skipped_total:.1f}s 제거) "
        f"+ tail {tail_hold_sec:.1f}s -> 출력 {out_duration:.1f}s @ {speed:g}x"
    )

    first = read_image(frames[0][1])
    if first is None:
        return f"첫 프레임을 읽지 못했습니다: {frames[0][1]}"
    size = target_size(first, max_width)
    if size[0] <= 0 or size[1] <= 0:
        return f"해상도 계산 실패: {first.shape}"
    print(f"[INFO] 출력 해상도: {size[0]}x{size[1]}, fps={fps:g}")

    writer, actual_path = open_writer(out_path, fps, size)
    if writer is None:
        return "VideoWriter 를 열지 못했습니다 (코덱 없음)"

    total_out = max(1, int(round(out_duration * fps)))
    cached_index = -1
    cached_image = None
    written = 0
    try:
        for step in range(total_out):
            target_ct = step * speed / fps
            index = min(len(frames) - 1, max(0, bisect_right(starts, target_ct) - 1))
            if index != cached_index:
                image = read_image(frames[index][1])
                if image is None:
                    # 깨진 프레임 1장 때문에 영상 전체를 버리지 않는다 - 직전 화면을 유지.
                    if cached_image is None:
                        continue
                else:
                    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    cached_image = resized
                    cached_index = index
            if cached_image is None:
                continue

            frame = cached_image
            if overlay:
                frame = draw_overlay(
                    cached_image.copy(), frames[index][0], index, len(frames),
                    skipped[index], label,
                )
            writer.write(frame)
            written += 1
            if written % 300 == 0:
                print(f"[INFO]   인코딩 {written}/{total_out} 프레임")
    finally:
        writer.release()

    if written == 0:
        return "인코딩된 프레임이 없습니다 (모든 프레임 읽기 실패)"
    size_mb = actual_path.stat().st_size / (1024 * 1024) if actual_path.exists() else 0.0
    print(f"[INFO] 완료 -> {actual_path} ({written} frames, {size_mb:.1f} MB)")
    return "success"


def main() -> str:
    """recording 폴더를 시연 영상으로 만든다."""
    input_dir = resolve_input_dir()
    if input_dir is None:
        return "입력 폴더를 정하지 못했습니다"

    frames = scan_frames(input_dir)
    if not frames:
        return f"프레임을 찾지 못했습니다 (파일명 규약 불일치?): {input_dir}"
    print(f"[INFO] 입력: {input_dir} ({len(frames)} frames)")

    frames = trim_frames(
        frames,
        _env_float("DEMO_VIDEO_START_SEC", 0.0),
        _env_float("DEMO_VIDEO_END_SEC", 0.0),
    )
    if len(frames) < 2:
        return f"프레임이 부족합니다 ({len(frames)}장) - 트림 구간을 확인하세요"

    raw_output = os.environ.get("DEMO_VIDEO_OUTPUT", "").strip()
    out_path = Path(raw_output).expanduser() if raw_output else input_dir / "demo.mp4"

    return render(
        frames,
        out_path,
        fps=max(1.0, _env_float("DEMO_VIDEO_FPS", 15.0)),
        speed=max(0.1, _env_float("DEMO_VIDEO_SPEED", 1.0)),
        max_hold_sec=max(0.0, _env_float("DEMO_VIDEO_MAX_HOLD_SEC", 2.0)),
        tail_hold_sec=max(0.0, _env_float("DEMO_VIDEO_TAIL_HOLD_SEC", 2.0)),
        max_width=max(0, _env_int("DEMO_VIDEO_MAX_WIDTH", 1920)),
        overlay=_env_flag("DEMO_VIDEO_OVERLAY", True),
        label=os.environ.get("DEMO_VIDEO_LABEL", "").strip(),
    )


if __name__ == "__main__":
    result = main()
    if result != "success":
        print(f"[EXIT] {result}")
        sys.exit(1)
    sys.exit(0)
