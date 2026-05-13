"""템플릿 없이 자연어 프롬프트만으로 VLM 이 "Image Operation Box" 를
찾아낼 수 있는지 검증하는 5장 샘플 프로브.

설명되는 박스 (Image Operation Box, 통칭 "SEM box"):
  - 두 부분이 옆으로 붙어 하나의 큰 블록을 이룬다:
    (A) Live SEM image area — 어둡고 실시간으로 갱신되는 wafer/시료 그레이스케일 영상.
    (B) Operation 버튼 패널 — 'AMS', 'ACD', 'Next', 'DDS' 등 SEM 조작 버튼이
        세로/격자 형태로 배치된 인접 패널.
  - VLM 은 두 영역을 하나의 직사각 bbox 로 함께 감싸야 한다.
    (예전 프롬프트는 버튼 패널만 잡아서 live SEM image 가 누락되는 문제가 있었다.)

입력은 `poc/workflow_2/filter_frames_by_change.py` 가 만든
`change_events.json` 의 events 에서 5개를 random sample 한다.

실행:
    uv run python poc/workflow_2/vlm_sem_monitor_box.py
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import WORKFLOW_2_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
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

LOG_NAME = "vlm_sem_monitor_box"
WORKFLOW_2_RECORDING_DIR = WORKFLOW_2_DIR / "recordings"
FILTER_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / "filter_frames_by_change"
DEFAULT_OUTPUT_ROOT = WORKFLOW_2_RECORDING_DIR / LOG_NAME

# ====================================================================
# 분석할 필터 결과 폴더 (filter_frames_by_change 가 만든 <tag>_<session>)
# 비워두면 가장 최신 폴더를 자동 선택한다.
# ====================================================================
FILTER_DIR_OVERRIDE = r""

DEFAULT_SAMPLE_COUNT = env_int("SEM_BOX_SAMPLE_COUNT", 5)
DEFAULT_REQUEST_DELAY_SEC = env_float("TEST_VLM_REQUEST_DELAY_SEC", 1.0)
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


def _sem_box_system_prompt() -> str:
    """Image Operation Box 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a Windows CD-SEM Tool application. "
        "Return strict JSON only. "
        "Locate the 'Image Operation Box' — a single rectangular region that contains TWO "
        "tightly-coupled parts arranged side-by-side and treated as one logical area:\n"
        "  (A) Live SEM image area: a dark, real-time-updating grayscale image of the wafer "
        "or sample being scanned. It looks like a noisy electron-microscope image, not a "
        "color photograph.\n"
        "  (B) Operation button panel: a vertical or grid arrangement of short-label buttons "
        "sitting flush against one edge of the live SEM image. Typical button labels you "
        "should expect to read here include 'AMS', 'ACD', 'Next', 'DDS', and similar short "
        "SEM operation commands.\n"
        "Return a bounding box that encloses BOTH the live SEM image area and the adjacent "
        "button panel together, as one rectangular block. "
        "Do NOT return only the button panel and do NOT return only the live SEM image — "
        "either one alone is the wrong answer. "
        "Do not include unrelated panels, separate toolbars, the window title bar, or other "
        "tabs of the application. "
        "If you cannot see both parts together, set panel_visible=false."
    )


def _sem_box_user_prompt() -> str:
    """Image Operation Box 탐지 사용자 프롬프트."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "panel_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "panel_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "image_area_side": "left | right",\n'
        '  "visible_buttons": ["AMS", "ACD", "Next", "DDS"],\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string explaining what you used to identify the region"\n'
        "}\n"
        "panel_bbox must tightly enclose the ENTIRE Image Operation Box — both the live SEM "
        "image area and the adjacent button panel together, as one rectangle. "
        "image_area_side tells which side of the bbox the live SEM image occupies relative "
        "to the button panel (use 'left' if the live image is on the left and the buttons "
        "are on the right; use 'right' for the opposite). "
        "visible_buttons should list 2~6 button labels you actually read inside the panel "
        "(expected examples: 'AMS', 'ACD', 'Next', 'DDS'). "
        "If the region is not visible as a single connected block, set panel_visible=false, "
        "panel_bbox=null, image_area_side=null, visible_buttons=[]."
    )


def _run_sem_box_detection(
    *,
    image_b64: str,
    width: int,
    height: int,
    client: Workflow1VLMClient,
) -> tuple[dict, dict | None]:
    """SEM monitor panel 의 bbox 를 탐지한다."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_sem_box_system_prompt(),
        user_text=_sem_box_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("panel_visible") is not True:
        return parsed, None

    bbox_1000 = normalize_bbox_1000(parsed.get("panel_bbox"))
    if bbox_1000 is None:
        return parsed, None
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


def _save_overlay(*, frame_path: Path, panel_bbox: dict, output_path: Path) -> str:
    """frame 위에 Image Operation Box bbox 를 마킹한다."""
    with Image.open(frame_path) as image:
        elements = {
            "image_operation_box": {"bbox": panel_bbox, "center": bbox_center(panel_bbox)},
        }
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={"image_operation_box": "magenta"},
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
    """탐지 결과 타임라인 텍스트를 만든다."""
    lines = []
    for item in results:
        bbox = item.get("panel_bbox") or {}
        buttons = ", ".join(item.get("visible_buttons") or [])
        lines.append(
            f"rank={int(item.get('rank') or 0):03d} "
            f"frame={int(item.get('frame_index') or 0):04d} "
            f"panel={'Y' if bbox else 'N'} "
            f"image_side={item.get('image_area_side') or '-':<7s} "
            f"conf={item.get('panel_confidence', '')} "
            f"buttons=[{buttons}] "
            f"bbox={bbox}"
        )
    return "\n".join(lines) + "\n"


def run_test() -> str:
    """5장 random sample 에 대해 SEM monitor box 탐지를 실행한다."""
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
    for directory in (overlays_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE,
        model_name=DEFAULT_MODEL,
        log_name=LOG_NAME,
    )

    print(
        f"[INFO] SEM monitor box 탐지 시작: service={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"samples={len(sampled)}, request_delay_sec={DEFAULT_REQUEST_DELAY_SEC}"
    )

    results: list[dict] = []
    panel_detected = 0
    vlm_calls = 0

    for event in sampled:
        rank = int(event.get("rank") or 0)
        frame_index = int(event.get("frame_index") or 0)
        timestamp_sec = float(event.get("timestamp_sec") or 0.0)
        frame_path = Path(event.get("frame_path", ""))

        if not frame_path.is_file():
            print(f"[WARNING] 프레임 파일 누락: rank={rank}, frame={frame_path}")
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
            print(f"[ERROR] SEM box detection 실패: rank={rank}, error={exc}")
        finally:
            vlm_calls += 1
            time.sleep(DEFAULT_REQUEST_DELAY_SEC)

        overlay_path = ""
        if panel_bbox is not None:
            panel_detected += 1
            try:
                overlay_path = _save_overlay(
                    frame_path=frame_path,
                    panel_bbox=panel_bbox,
                    output_path=overlays_dir / f"{rank:03d}_frame_{frame_index:04d}_overlay.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패: rank={rank}, error={exc}")
        else:
            print(f"[INFO] panel_visible=false (rank={rank:03d})")

        result = {
            "rank": rank,
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "frame_path": str(frame_path),
            "overlay_path": overlay_path,
            "panel_payload": payload,
            "panel_bbox": panel_bbox or {},
            "panel_confidence": payload.get("confidence"),
            "panel_evidence": payload.get("evidence"),
            "image_area_side": payload.get("image_area_side"),
            "visible_buttons": payload.get("visible_buttons") or [],
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
        "vlm_calls": vlm_calls,
        "panel_detected": panel_detected,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(output_dir),
    }
    save_debug_json(output_dir / "summary.json", summary_payload)
    save_debug_text(output_dir / "timeline.txt", _build_timeline_text(results))

    print(
        f"[INFO] 완료: sampled={len(sampled)}, panel_detected={panel_detected}, "
        f"vlm_calls={vlm_calls}, elapsed={format_elapsed_ms(started_at)}, "
        f"output_dir={output_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run_test() == "success" else 1)
