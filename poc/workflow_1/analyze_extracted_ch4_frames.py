"""추출된 CH4 프레임 세트를 VLM으로 분석한다.

기본 실행:
  uv run python poc/workflow_1/analyze_extracted_ch4_frames.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_1 import DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_marked_bboxes
from poc.workflow_1.flask_vlm import (
    DEFAULT_SCREEN_ANALYSIS_MODEL_NAME,
    DEFAULT_SCREEN_ANALYSIS_SERVICE,
)
from poc.workflow_1.util import env_float, env_int, format_elapsed_ms
from poc.workflow_1.util.json_utils import extract_json
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "analyze_extracted_ch4_frames"
DEFAULT_EXTRACTION_ROOT = DEBUG_IMAGE_DIR / "extract_recorded_ch4_frames"
DEFAULT_MIN_CHANGE_SCORE = env_float("CH4_ANALYZE_MIN_CHANGE_SCORE", 0.0)
DEFAULT_MAX_FRAMES = env_int("CH4_ANALYZE_MAX_FRAMES", 0)
DEFAULT_VLM_SERVICE = (
    os.getenv("CH4_ANALYZE_VLM_SERVICE", DEFAULT_SCREEN_ANALYSIS_SERVICE).strip()
    or DEFAULT_SCREEN_ANALYSIS_SERVICE
)
DEFAULT_VLM_MODEL_NAME = (
    os.getenv("CH4_ANALYZE_VLM_MODEL_NAME", DEFAULT_SCREEN_ANALYSIS_MODEL_NAME).strip()
    or DEFAULT_SCREEN_ANALYSIS_MODEL_NAME
)


def _resolve_extraction_dir() -> Path | None:
    """분석할 추출 결과 디렉터리를 결정한다."""
    raw_path = os.getenv("CH4_ANALYZE_FRAMES_DIR", "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] CH4_ANALYZE_FRAMES_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    if not DEFAULT_EXTRACTION_ROOT.exists():
        print(f"[ERROR] 추출 결과 디렉터리가 없습니다: {DEFAULT_EXTRACTION_ROOT}")
        return None

    candidates = sorted(
        (path for path in DEFAULT_EXTRACTION_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[ERROR] 분석할 추출 결과가 없습니다: {DEFAULT_EXTRACTION_ROOT}")
        return None

    latest = candidates[0].resolve()
    print(f"[INFO] 최신 추출 결과 선택: {latest}")
    return latest


def _load_extraction_summary(extraction_dir: Path) -> dict | None:
    """추출 단계 summary.json 을 읽는다."""
    summary_path = extraction_dir / "summary.json"
    if not summary_path.is_file():
        print(f"[ERROR] summary.json 이 없습니다: {summary_path}")
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] summary.json 파싱 실패: {exc}")
        return None


def _should_analyze_frame(frame_item: dict, analyzed_count: int) -> bool:
    """change_score 기준으로 프레임 분석 여부를 결정한다."""
    if analyzed_count == 0:
        return True
    if DEFAULT_MIN_CHANGE_SCORE <= 0:
        return True
    return float(frame_item.get("change_score") or 0.0) >= DEFAULT_MIN_CHANGE_SCORE


def _analysis_system_prompt() -> str:
    """프레임 분석용 시스템 프롬프트."""
    return (
        "You analyze sampled frames from a CCTV recording of a software UI. "
        "Return strict JSON only. "
        "Be conservative: do not invent clicks or typed text when evidence is weak. "
        "Use null or empty arrays when uncertain."
    )


def _analysis_user_prompt(
    *,
    frame_timestamp: float,
    frame_number: int,
    change_score: float,
    previous_frame_note: str,
) -> str:
    """프레임 분석용 사용자 프롬프트."""
    return (
        f"Sampled CCTV software frame.\n"
        f"frame_number={frame_number}\n"
        f"timestamp_sec={frame_timestamp:.2f}\n"
        f"change_score_vs_previous_sample={change_score:.4f}\n"
        f"previous_frame_note={previous_frame_note}\n\n"
        "Focus on actionable evidence from the current frame.\n"
        "Infer cautiously what likely happened since the previous sampled frame.\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "frame_summary": "short string",\n'
        '  "cursor_visible": true,\n'
        '  "cursor_position": {"x": 0, "y": 0},\n'
        '  "mouse_activity_since_previous_frame": "none|possible|clear|uncertain",\n'
        '  "clicked_targets": [\n'
        '    {"label": "string", "evidence": "string", "confidence": 0.0}\n'
        "  ],\n"
        '  "typed_text_visible": [\n'
        '    {"text": "string", "field_hint": "string", "new_since_previous_frame": "yes|no|uncertain"}\n'
        "  ],\n"
        '  "keyboard_activity_since_previous_frame": {\n'
        '    "detected": false,\n'
        '    "evidence": "string",\n'
        '    "visible_text": "string"\n'
        "  },\n"
        '  "ui_changes_since_previous_frame": ["string"],\n'
        '  "visible_buttons_or_fields": ["string"],\n'
        '  "confidence": 0.0\n'
        "}\n"
        "If cursor is not visible, set cursor_visible=false and cursor_position=null."
    )


def _coerce_pixel(value) -> int | None:
    """좌표 값을 int 픽셀로 변환한다."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(round(float(stripped)))
        except ValueError:
            return None
    return None


def _extract_cursor_point(analysis_payload: dict, image_size: tuple[int, int]) -> dict[str, int] | None:
    """VLM 응답에서 커서 좌표를 읽고 이미지 범위로 clamp 한다."""
    if not isinstance(analysis_payload, dict):
        return None
    if analysis_payload.get("cursor_visible") is not True:
        return None

    raw_point = analysis_payload.get("cursor_position")
    if not isinstance(raw_point, dict):
        return None

    x = _coerce_pixel(raw_point.get("x"))
    y = _coerce_pixel(raw_point.get("y"))
    if x is None or y is None:
        return None

    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return None

    return {
        "x": max(0, min(x, img_w - 1)),
        "y": max(0, min(y, img_h - 1)),
    }


def _save_cursor_overlay(
    *,
    frame_path: str,
    frame_id: str,
    analysis_payload: dict | None,
    overlays_dir: Path,
) -> str | None:
    """커서 위치가 있으면 overlay 이미지를 저장한다."""
    if not isinstance(analysis_payload, dict):
        return None

    with Image.open(frame_path) as image:
        cursor_point = _extract_cursor_point(analysis_payload, image.size)
        if cursor_point is None:
            return None

        radius = 12
        bbox = {
            "left": max(0, cursor_point["x"] - radius),
            "top": max(0, cursor_point["y"] - radius),
            "right": min(image.size[0], cursor_point["x"] + radius + 1),
            "bottom": min(image.size[1], cursor_point["y"] + radius + 1),
        }
        overlay_path = overlays_dir / f"{frame_id}_cursor_overlay.jpg"
        save_marked_bboxes(
            image.copy(),
            elements={
                "cursor": {
                    "bbox": bbox,
                    "center": cursor_point,
                }
            },
            colors={"cursor": "red"},
            out_path=overlay_path,
        )
        return str(overlay_path)


def _build_timeline_text(extraction_dir: Path, frame_results: list[dict]) -> str:
    """사람이 읽기 쉬운 분석 타임라인 텍스트를 만든다."""
    lines = [f"frames_dir={extraction_dir}"]
    for item in frame_results:
        analysis = item.get("analysis") or {}
        clicks = analysis.get("clicked_targets") or []
        typed = analysis.get("typed_text_visible") or []
        click_labels = ", ".join(
            str(entry.get("label") or "").strip()
            for entry in clicks
            if isinstance(entry, dict) and str(entry.get("label") or "").strip()
        ) or "-"
        typed_values = ", ".join(
            str(entry.get("text") or "").strip()
            for entry in typed
            if isinstance(entry, dict) and str(entry.get("text") or "").strip()
        ) or "-"
        lines.append(
            f"[{item['timestamp_sec']:>7.2f}s] "
            f"frame={item['frame_number']} "
            f"change={item['change_score']:.4f} "
            f"summary={analysis.get('frame_summary', '')}"
        )
        lines.append(f"  mouse={analysis.get('mouse_activity_since_previous_frame', 'uncertain')}")
        lines.append(f"  clicks={click_labels}")
        lines.append(f"  typed={typed_values}")
    return "\n".join(lines) + "\n"


def analyze_frames() -> str:
    """추출된 CH4 프레임 세트를 VLM으로 분석한다."""
    started_at = time.time()
    extraction_dir = _resolve_extraction_dir()
    if extraction_dir is None:
        return "frames_dir_not_found"

    extraction_summary = _load_extraction_summary(extraction_dir)
    if extraction_summary is None:
        return "summary_not_found"

    frame_items = extraction_summary.get("frame_results") or []
    if not frame_items:
        print(f"[ERROR] 분석할 프레임 정보가 없습니다: {extraction_dir}")
        return "frame_results_empty"

    analysis_dir = extraction_dir / "vlm_analysis"
    overlays_dir = analysis_dir / "cursor_overlays"
    results_dir = analysis_dir / "results"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    max_frames = DEFAULT_MAX_FRAMES if DEFAULT_MAX_FRAMES > 0 else None
    print(
        f"[INFO] 프레임 분석 시작: frames_dir={extraction_dir}, "
        f"min_change_score={DEFAULT_MIN_CHANGE_SCORE}, "
        f"max_frames={max_frames or 'all'}"
    )

    vlm_client = Workflow1VLMClient(
        service_slug=DEFAULT_VLM_SERVICE,
        model_name=DEFAULT_VLM_MODEL_NAME,
        log_name=LOG_NAME,
    )

    frame_results: list[dict] = []
    analyzed_count = 0
    skipped_count = 0
    previous_frame_note = "none"

    for frame_item in frame_items:
        if max_frames and analyzed_count >= max_frames:
            break

        frame_path = str(frame_item.get("frame_path") or "").strip()
        if not frame_path or not Path(frame_path).is_file():
            print(f"[WARNING] 프레임 파일 누락: {frame_path!r}")
            continue

        if not _should_analyze_frame(frame_item, analyzed_count):
            skipped_count += 1
            print(
                f"[INFO] 프레임 건너뜀: frame={frame_item.get('frame_number')}, "
                f"ts={float(frame_item.get('timestamp_sec') or 0.0):.2f}s, "
                f"change_score={float(frame_item.get('change_score') or 0.0):.4f}"
            )
            continue

        print(
            f"[INFO] 프레임 분석: frame={frame_item.get('frame_number')}, "
            f"ts={float(frame_item.get('timestamp_sec') or 0.0):.2f}s, "
            f"change_score={float(frame_item.get('change_score') or 0.0):.4f}"
        )
        response = vlm_client.chat_with_image_path(
            image_path=frame_path,
            system_message=_analysis_system_prompt(),
            user_text=_analysis_user_prompt(
                frame_timestamp=float(frame_item.get("timestamp_sec") or 0.0),
                frame_number=int(frame_item.get("frame_number") or 0),
                change_score=float(frame_item.get("change_score") or 0.0),
                previous_frame_note=previous_frame_note,
            ),
            image_mime="image/jpeg",
            temperature=0.0,
        )

        parsed_payload = None
        overlay_path = None
        try:
            parsed_payload = extract_json(response.text)
            overlay_path = _save_cursor_overlay(
                frame_path=frame_path,
                frame_id=str(frame_item.get("frame_id") or "frame"),
                analysis_payload=parsed_payload,
                overlays_dir=overlays_dir,
            )
        except json.JSONDecodeError:
            print(
                f"[WARNING] JSON 파싱 실패: frame={frame_item.get('frame_number')}, "
                "raw_response 를 그대로 저장합니다."
            )

        result = {
            "frame_id": frame_item.get("frame_id"),
            "frame_number": frame_item.get("frame_number"),
            "timestamp_sec": frame_item.get("timestamp_sec"),
            "change_score": frame_item.get("change_score"),
            "frame_path": frame_path,
            "review_overlay_path": frame_item.get("review_overlay_path") or "",
            "cursor_overlay_path": overlay_path or "",
            "raw_response_text": response.text,
            "analysis": parsed_payload or {},
        }
        frame_results.append(result)
        analyzed_count += 1

        result_path = results_dir / f"{frame_item.get('frame_id')}.json"
        save_debug_json(result_path, result)
        previous_frame_note = str(
            (result.get("analysis") or {}).get("frame_summary")
            or response.text[:300]
        ).strip() or "none"

    summary = {
        "frames_dir": str(extraction_dir),
        "vlm_service": DEFAULT_VLM_SERVICE,
        "vlm_model_name": DEFAULT_VLM_MODEL_NAME,
        "min_change_score": DEFAULT_MIN_CHANGE_SCORE,
        "max_frames": max_frames,
        "sampled_frames": len(frame_items),
        "analyzed_frames": analyzed_count,
        "skipped_frames": skipped_count,
        "elapsed": format_elapsed_ms(started_at),
        "output_dir": str(analysis_dir),
        "frame_results": frame_results,
    }
    save_debug_json(analysis_dir / "summary.json", summary)
    (analysis_dir / "timeline.txt").write_text(
        _build_timeline_text(extraction_dir, frame_results),
        encoding="utf-8",
    )

    print(
        f"[INFO] 프레임 분석 완료: analyzed={analyzed_count}, "
        f"skipped={skipped_count}, elapsed={format_elapsed_ms(started_at)}, "
        f"output_dir={analysis_dir}"
    )
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if analyze_frames() == "success" else 1)
