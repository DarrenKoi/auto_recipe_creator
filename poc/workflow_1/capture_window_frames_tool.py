"""Remote Monitoring System 툴 창을 JPEG 프레임으로 저장한다.

`workflow_select_tool.py` 가 List 탭에서 대상 Tool 을 더블클릭한 직후 실행하여,
열린 'Remote Monitoring System - ...' 창을 일정 간격으로 5 분간 캡처한다.

기본 실행:
  uv run python poc/workflow_1/capture_window_frames_tool.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.workflow_1 import RECORDING_DIR
from poc.workflow_1.debug_artifacts import save_debug_jpeg, save_debug_json, save_debug_text
from poc.workflow_1.login_rcs_common import (
    REMOTE_MONITORING_WINDOW_TITLE_PREFIX,
    wait_for_remote_monitoring_window,
)
from poc.workflow_1.util import (
    WINDOW_UTILS_AVAILABLE,
    capture_window,
    env_float,
    env_int,
    format_elapsed_ms,
)
from poc.workflow_1.workflow_select_tool import DEFAULT_TARGET_TOOL_NAME, load_target_tool_name

load_dotenv()

LOG_NAME = "capture_window_frames_tool"
DEFAULT_OUTPUT_DIR = RECORDING_DIR / LOG_NAME
DEFAULT_FRAME_INTERVAL_MS = env_int("TOOL_CAPTURE_FRAME_INTERVAL_MS", 500)
DEFAULT_MAX_FRAMES = env_int("TOOL_CAPTURE_FRAME_MAX_FRAMES", 0)
DEFAULT_MAX_DURATION_SEC = env_float("TOOL_CAPTURE_MAX_DURATION_SEC", 300.0)
DEFAULT_JPEG_QUALITY = env_int("TOOL_CAPTURE_JPEG_QUALITY", 95)
DEFAULT_WINDOW_WAIT_SEC = env_float("TOOL_CAPTURE_WINDOW_WAIT_SEC", 30.0)
DEFAULT_WINDOW_POLL_SEC = env_float("TOOL_CAPTURE_WINDOW_POLL_SEC", 0.5)


def _sanitize_name(text: str) -> str:
    """파일명에 사용할 문자열을 정리한다."""
    safe_chars: list[str] = []
    for char in (text or "").strip():
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    safe = "".join(safe_chars).strip("-._")
    return safe or "tool"


def _build_output_dir(tool_name: str, window_title: str) -> Path:
    """이번 캡처 결과 디렉터리를 만든다."""
    tag = time.strftime("%y%m%d_%H%M%S", time.localtime())
    folder = f"{tag}_{_sanitize_name(tool_name)}_{_sanitize_name(window_title)}"
    out_dir = DEFAULT_OUTPUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_timeline_text(frame_items: list[dict]) -> str:
    """사람이 읽기 쉬운 캡처 타임라인을 만든다."""
    lines = []
    for item in frame_items:
        lines.append(
            f"[{item['timestamp_sec']:>7.3f}s] "
            f"frame={item['index']:04d} "
            f"path={item['frame_path']}"
        )
    return "\n".join(lines) + "\n"


def capture_frames() -> str:
    """Remote Monitoring System 툴 창을 일정 간격으로 캡처한다."""
    started_at = time.time()
    interval_sec = max(0.001, DEFAULT_FRAME_INTERVAL_MS / 1000.0)

    if os.name != "nt" or not WINDOW_UTILS_AVAILABLE:
        print(
            "[ERROR] 창 캡처는 Windows + window_utils 가 필요합니다. "
            f"os={os.name}, window_utils={WINDOW_UTILS_AVAILABLE}"
        )
        return "window_capture_unavailable"

    target_tool_name = load_target_tool_name(DEFAULT_TARGET_TOOL_NAME)
    print(
        f"[INFO] Remote Monitoring System 창 대기: "
        f"tool_name={target_tool_name!r}, title_prefix={REMOTE_MONITORING_WINDOW_TITLE_PREFIX!r}, "
        f"timeout={DEFAULT_WINDOW_WAIT_SEC}s"
    )

    tool_window, window_title, backend = wait_for_remote_monitoring_window(
        target_tool_name,
        timeout_sec=DEFAULT_WINDOW_WAIT_SEC,
        poll_interval_sec=DEFAULT_WINDOW_POLL_SEC,
    )
    if tool_window is None:
        print(
            f"[ERROR] Remote Monitoring System 창을 찾지 못했습니다: "
            f"tool_name={target_tool_name!r}"
        )
        return "tool_window_not_found"

    output_dir = _build_output_dir(target_tool_name, window_title)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] 툴 창 프레임 캡처 시작: title={window_title!r}, "
        f"backend={backend}, interval_ms={DEFAULT_FRAME_INTERVAL_MS}, "
        f"max_frames={DEFAULT_MAX_FRAMES or 'all'}, "
        f"max_duration_sec={DEFAULT_MAX_DURATION_SEC or 'unlimited'}, "
        f"output_dir={output_dir}"
    )

    frame_items: list[dict] = []
    frame_index = 0

    while True:
        if DEFAULT_MAX_FRAMES > 0 and frame_index >= DEFAULT_MAX_FRAMES:
            break

        elapsed_sec = time.time() - started_at
        if DEFAULT_MAX_DURATION_SEC > 0 and elapsed_sec >= DEFAULT_MAX_DURATION_SEC:
            break

        loop_started_at = time.time()
        try:
            image = capture_window(tool_window)
        except Exception as exc:
            print(f"[ERROR] 창 캡처 실패: frame={frame_index}, error={exc}")
            break

        frame_name = f"frame_{frame_index:04d}_{int(round(elapsed_sec * 1000)):08d}ms.jpg"
        frame_path = frames_dir / frame_name
        if DEFAULT_JPEG_QUALITY == 95:
            save_debug_jpeg(image, frame_path)
        else:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_image = image.convert("RGB") if image.mode != "RGB" else image
            rgb_image.save(frame_path, format="JPEG", quality=DEFAULT_JPEG_QUALITY)

        frame_items.append(
            {
                "index": frame_index,
                "timestamp_sec": round(elapsed_sec, 3),
                "frame_path": str(frame_path),
                "width": image.size[0],
                "height": image.size[1],
            }
        )
        frame_index += 1

        elapsed_loop = time.time() - loop_started_at
        sleep_sec = interval_sec - elapsed_loop
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    summary = {
        "tool_name": target_tool_name,
        "window_title": window_title,
        "backend": backend,
        "interval_ms": DEFAULT_FRAME_INTERVAL_MS,
        "max_frames": DEFAULT_MAX_FRAMES,
        "max_duration_sec": DEFAULT_MAX_DURATION_SEC,
        "captured_frames": len(frame_items),
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "frame_items": frame_items,
    }
    save_debug_json(output_dir / "summary.json", summary)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(frame_items))

    print(
        f"[INFO] 툴 창 프레임 캡처 완료: captured={len(frame_items)}, "
        f"elapsed={format_elapsed_ms(started_at)}, output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if capture_frames() == "success" else 1)
