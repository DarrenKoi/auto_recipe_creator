"""실시간으로 'Remote Monitoring System - ...' 창을 직접 캡처하면서
VLM 이 SEM Monitor Box 를 잘 찾는지 검증하는 라이브 프로브.

`vlm_sem_monitor_box.py` 는 미리 저장된 change_events.json 의 프레임을
샘플링하지만, 본 스크립트는 **지금 화면에 떠 있는 Tool 창** 을 일정 간격으로
직접 잡아 같은 SEM box 탐지 프롬프트를 돌린다.

흐름:
  1. 'Remote Monitoring System -' 제목 접두어로 Tool 창을 찾는다.
     (엔지니어가 직접 Tool 을 띄워 두면, 어떤 model 이든 매칭된다.)
  2. 창을 찾으면 우선 raw 캡처 1장을 저장해 "창 자체가 제대로 잡혔는지"
     눈으로 먼저 확인할 수 있게 한다.
  3. 이후 interval 마다 캡처 → WebP 인코딩 → VLM SEM box 탐지 →
     magenta bbox overlay 저장 을 반복한다.

Windows + window_utils 환경에서만 동작한다 (capture_window 필요).

실행:
    uv run python poc/workflow_2/vlm_sem_monitor_box_realtime.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_2.vlm_sem_monitor_box import (
    _run_sem_box_detection,
    _save_overlay,
)
from poc.workflow_1.debug_artifacts import save_debug_jpeg, save_debug_json, save_debug_text
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
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
    make_timestamp_tag,
)
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "vlm_sem_monitor_box_realtime"
WORKFLOW_2_RECORDING_DIR = WORKFLOW_2_DIR / "recordings"
DEFAULT_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / LOG_NAME

# ====================================================================
# 대상 Tool 이름. 비워두면 'Remote Monitoring System -' 접두어를 가진
# 아무 창이나 매칭한다 (엔지니어가 직접 띄운 창).
# 특정 model 만 잡고 싶으면 예: "MCD630".
# ====================================================================
TARGET_TOOL_NAME = os.getenv("SEM_BOX_TARGET_TOOL_NAME", "").strip()

DEFAULT_WINDOW_WAIT_SEC = env_float("SEM_BOX_WINDOW_WAIT_SEC", 30.0)
DEFAULT_WINDOW_POLL_SEC = env_float("SEM_BOX_WINDOW_POLL_SEC", 0.5)
DEFAULT_INTERVAL_SEC = env_float("SEM_BOX_REALTIME_INTERVAL_SEC", 3.0)
DEFAULT_MAX_ITERATIONS = env_int("SEM_BOX_REALTIME_MAX_ITER", 10)
DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME


def _build_output_dir(window_title: str) -> Path:
    """이번 실시간 테스트 결과 디렉터리를 만든다."""
    tag = make_timestamp_tag()
    safe_title = "".join(
        c if (c.isalnum() or c in {"-", "_", "."}) else "-" for c in (window_title or "")
    ).strip("-._") or "window"
    out_dir = DEFAULT_OUTPUT_ROOT / f"{tag}_{safe_title}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_timeline_text(results: list[dict]) -> str:
    """탐지 결과 타임라인 텍스트를 만든다."""
    lines = []
    for item in results:
        bbox = item.get("panel_bbox") or {}
        overlays = ", ".join(item.get("overlay_panels_seen") or [])
        lines.append(
            f"iter={int(item.get('iteration') or 0):03d} "
            f"ts={float(item.get('timestamp_sec') or 0.0):>7.3f}s "
            f"panel={'Y' if bbox else 'N'} "
            f"mode={item.get('mode_label') or '-':<8s} "
            f"conf={item.get('panel_confidence', '')} "
            f"overlays=[{overlays}] "
            f"bbox={bbox}"
        )
    return "\n".join(lines) + "\n"


def run_realtime() -> str:
    """실시간 Tool 창 캡처 + SEM monitor box 탐지 루프."""
    started_at = time.time()

    if os.name != "nt" or not WINDOW_UTILS_AVAILABLE:
        print(
            "[ERROR] 실시간 창 캡처는 Windows + window_utils 가 필요합니다. "
            f"os={os.name}, window_utils={WINDOW_UTILS_AVAILABLE}"
        )
        return "window_capture_unavailable"

    print(
        f"[INFO] Remote Monitoring System 창 대기: "
        f"tool_name={TARGET_TOOL_NAME!r}, "
        f"title_prefix={REMOTE_MONITORING_WINDOW_TITLE_PREFIX!r}, "
        f"timeout={DEFAULT_WINDOW_WAIT_SEC}s"
    )

    tool_window, window_title, backend = wait_for_remote_monitoring_window(
        TARGET_TOOL_NAME,
        timeout_sec=DEFAULT_WINDOW_WAIT_SEC,
        poll_interval_sec=DEFAULT_WINDOW_POLL_SEC,
    )
    if tool_window is None:
        print(
            "[ERROR] Remote Monitoring System 창을 찾지 못했습니다. "
            f"Tool 을 먼저 띄웠는지 확인하세요. (tool_name={TARGET_TOOL_NAME!r})"
        )
        return "tool_window_not_found"

    print(
        f"[INFO] 창 발견 OK: title={window_title!r}, backend={backend}"
    )

    output_dir = _build_output_dir(window_title)
    frames_dir = output_dir / "frames"
    overlays_dir = output_dir / "overlays"
    results_dir = output_dir / "results"
    for directory in (frames_dir, overlays_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 창이 제대로 잡혔는지 눈으로 먼저 확인할 수 있도록 raw 캡처 1장 저장.
    try:
        first_image = capture_window(tool_window)
        raw_path = frames_dir / "window_check_raw.jpg"
        save_debug_jpeg(first_image, raw_path)
        print(
            f"[INFO] 창 캡처 확인용 raw 저장: {raw_path} "
            f"(size={first_image.size[0]}x{first_image.size[1]})"
        )
    except Exception as exc:
        print(f"[ERROR] 창을 찾았으나 캡처에 실패했습니다: {exc}")
        return "capture_failed"

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE,
        model_name=DEFAULT_MODEL,
        log_name=LOG_NAME,
    )

    print(
        f"[INFO] 실시간 SEM box 탐지 시작: service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"interval_sec={DEFAULT_INTERVAL_SEC}, max_iter={DEFAULT_MAX_ITERATIONS}"
    )

    results: list[dict] = []
    panel_detected = 0
    vlm_calls = 0

    for iteration in range(1, DEFAULT_MAX_ITERATIONS + 1):
        loop_started_at = time.time()
        elapsed_sec = loop_started_at - started_at

        try:
            image = capture_window(tool_window)
        except Exception as exc:
            print(f"[ERROR] 창 캡처 실패 (iter={iteration}): {exc}")
            break

        frame_path = frames_dir / f"iter_{iteration:03d}.jpg"
        save_debug_jpeg(image, frame_path)

        try:
            image_b64, frame_w, frame_h = encode_image_webp(image, quality=90)
        except Exception as exc:
            print(f"[ERROR] WebP 인코딩 실패 (iter={iteration}): {exc}")
            continue

        payload: dict = {}
        panel_bbox: dict | None = None
        try:
            payload, panel_bbox = _run_sem_box_detection(
                image_b64=image_b64,
                width=frame_w,
                height=frame_h,
                client=client,
            )
        except Exception as exc:
            print(f"[ERROR] SEM box detection 실패 (iter={iteration}): {exc}")
        finally:
            vlm_calls += 1

        overlay_path = ""
        if panel_bbox is not None:
            panel_detected += 1
            try:
                overlay_path = _save_overlay(
                    frame_path=frame_path,
                    panel_bbox=panel_bbox,
                    output_path=overlays_dir / f"iter_{iteration:03d}_overlay.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패 (iter={iteration}): {exc}")
        else:
            print(f"[INFO] panel_visible=false (iter={iteration:03d})")

        result = {
            "iteration": iteration,
            "timestamp_sec": round(elapsed_sec, 3),
            "frame_path": str(frame_path),
            "overlay_path": overlay_path,
            "panel_payload": payload,
            "panel_bbox": panel_bbox or {},
            "panel_confidence": payload.get("confidence"),
            "panel_evidence": payload.get("evidence"),
            "mode_label": payload.get("mode_label"),
            "overlay_panels_seen": payload.get("overlay_panels_seen") or [],
        }
        save_debug_json(results_dir / f"iter_{iteration:03d}.json", result)
        results.append(result)

        print(
            f"[INFO] iter={iteration:03d} panel={'Y' if panel_bbox else 'N'} "
            f"mode={payload.get('mode_label') or '-'} "
            f"conf={payload.get('confidence')} bbox={panel_bbox or {}}"
        )

        sleep_sec = DEFAULT_INTERVAL_SEC - (time.time() - loop_started_at)
        if iteration < DEFAULT_MAX_ITERATIONS and sleep_sec > 0:
            time.sleep(sleep_sec)

    summary_payload = {
        "window_title": window_title,
        "backend": backend,
        "tool_name": TARGET_TOOL_NAME,
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "interval_sec": DEFAULT_INTERVAL_SEC,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "iterations_run": len(results),
        "vlm_calls": vlm_calls,
        "panel_detected": panel_detected,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
    }
    save_debug_json(output_dir / "summary.json", summary_payload)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(results))

    print(
        f"[INFO] 완료: iterations={len(results)}, panel_detected={panel_detected}, "
        f"vlm_calls={vlm_calls}, elapsed={format_elapsed_ms(started_at)}, "
        f"output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run_realtime() == "success" else 1)
