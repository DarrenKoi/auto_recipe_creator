"""recording_filter 엔트리포인트 — 입력 해석 → Stage 1+2a → 산출물 기록.

CLI 인자 없음(프로젝트 규칙). 입력은 env/모듈상수/자동탐색으로, 산출은 입력의
형제 recording_filter/ 폴더에 쓴다.

실행:
    uv run python poc/workflow_3/recording_filter/filter_recording.py
"""

import os
import shutil
import time
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.recording_filter.click_detect import detect_clicks
from poc.workflow_3.recording_filter.element_label import crop_box_around
from poc.workflow_3.recording_filter.frame_reduce import collect_frame_paths, reduce_frames
from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)
from poc.workflow_3.recording_filter.timeline import build_timeline, write_click_overlays
from poc.workflow_3.util import format_elapsed_ms

# 분석할 recording/ 폴더를 직접 적어 쓸 수 있다(가장 우선). 비우면 env/자동탐색.
INPUT_DIR_OVERRIDE = r""


def _resolve_input_dir() -> Path | None:
    """분석할 recording/ 폴더를 결정한다(override -> env -> 자동탐색)."""
    override = (INPUT_DIR_OVERRIDE or "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            print(f"[INFO] INPUT_DIR_OVERRIDE 사용: {path}")
            return path.resolve()
        print(f"[ERROR] INPUT_DIR_OVERRIDE 디렉터리를 찾지 못했습니다: {path}")
        return None

    env_path = os.getenv("RECORDING_FILTER_INPUT_DIR", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] RECORDING_FILTER_INPUT_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    # 등록(captured_img_from_rcs) + 미등록(_unregistered) 두 경로 형태 모두 탐색.
    candidates = sorted(
        [
            *ALIGN_IMAGES_DIR.glob("*/*/*/captured_img_from_rcs/*/recording"),
            *ALIGN_IMAGES_DIR.glob("*/_unregistered/*/recording"),
            *ALIGN_IMAGES_DIR.glob("*/_manual/*/recording"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0].resolve()
        print(f"[INFO] 최신 recording/ 자동 선택: {latest}")
        return latest
    print(f"[ERROR] 분석할 recording/ 폴더를 찾지 못했습니다(루트: {ALIGN_IMAGES_DIR}).")
    return None


def _resolve_frames_dir(capture_dir: Path) -> Path | None:
    """실제 JPEG 가 있는 디렉터리를 결정한다(capture_dir 직접 또는 frames/ 하위)."""
    if any(capture_dir.glob("*.jpg")) or any(capture_dir.glob("*.jpeg")):
        return capture_dir
    frames_dir = capture_dir / "frames"
    if frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
        return frames_dir
    print(f"[ERROR] JPEG 프레임이 없습니다: {capture_dir}")
    return None


def _resolve_output_dir(capture_dir: Path) -> Path:
    """산출 폴더를 결정한다(env override -> capture_dir 형제 recording_filter/)."""
    env_out = os.getenv("RECORDING_FILTER_OUTPUT_DIR", "").strip()
    if env_out:
        return Path(env_out).expanduser().resolve()
    return (capture_dir.parent / "recording_filter").resolve()


def _copy_change_events(change_events, out_dir: Path) -> None:
    """Stage 1 생존 프레임을 rank 접두로 복사한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev in change_events:
        src = Path(ev.frame_path)
        dst = out_dir / f"{ev.rank:03d}_{src.name}"
        shutil.copy2(src, dst)


def _change_events_payload(change_events) -> list[dict]:
    return [
        {
            "rank": ev.rank,
            "frame_path": ev.frame_path,
            "prev_frame_path": ev.prev_frame_path,
            "timestamp_sec": ev.timestamp_sec,
            "frame_index": ev.frame_index,
            "change_bbox": ev.change_bbox,
            "largest_blob_area_px": ev.largest_blob_area_px,
            "changed_pixels": ev.changed_pixels,
        }
        for ev in change_events
    ]


def run_filter(*, input_dir=None, settings: RecordingFilterSettings = None, client=None) -> str:
    """필터 파이프라인을 실행하고 상태 문자열을 반환한다."""
    started_at = time.time()
    settings = settings or load_recording_filter_settings()

    capture_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if capture_dir is None:
        return "input_not_found"
    frames_dir = _resolve_frames_dir(capture_dir)
    if frames_dir is None:
        return "frames_not_found"
    if len(collect_frame_paths(frames_dir)) < 2:
        print(f"[ERROR] 변화 비교에 최소 2장 필요: {frames_dir}")
        return "not_enough_frames"

    out_dir = _resolve_output_dir(capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 ----
    change_events = reduce_frames(frames_dir, settings)
    stage1_total = len(change_events)  # Stage 1.5 가 change_events 를 걸러 덮어쓰기 전 원본 건수.
    _copy_change_events(change_events, out_dir / "change_events")
    save_debug_json(
        out_dir / "change_events.json",
        {
            "capture_dir": str(capture_dir),
            "frames_dir": str(frames_dir),
            "min_change_area_px": settings.min_change_area_px,
            "diff_threshold": settings.diff_threshold,
            "resize_width": settings.resize_width,
            "events": _change_events_payload(change_events),
        },
    )

    # ---- Stage 1.5: 영역 게이트 ----
    from poc.workflow_3.recording_filter.region_gate import (
        apply_region_gate,
        build_region_maps,
        load_frame_meta,
    )

    metas = load_frame_meta(frames_dir)
    gate_info = {}
    if settings.region_gate_enabled:
        if client is None:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            client = Workflow1VLMClient(settings.vlm_service, model_name=settings.vlm_model)
        region_maps = build_region_maps(change_events, metas, client, out_dir)
        gated = apply_region_gate(change_events, metas, region_maps)
        for event, generation, verdict, occlusion in gated:
            gate_info[event.rank] = {
                "generation": generation,
                "region": "live_image" if verdict == "ambient" else "ui",
                "occlusion": occlusion,
                "verdict": verdict,
            }
        # ambient 와 가려진 프레임은 비싼 Stage 2a 에 태우지 않는다.
        change_events = [
            event for event, _g, verdict, occlusion in gated
            if verdict == "candidate" and occlusion != "full"
        ]
        print(f"[INFO] Stage 1.5 통과: {len(change_events)} 건이 Stage 2a 로 갑니다.")

    # ---- Stage 2a ----
    if client is None:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.vlm_service, model_name=settings.vlm_model)
    click_events = detect_clicks(change_events, settings, client=client)
    write_click_overlays(
        [ce for ce in click_events if ce.is_click], out_dir / "click_events"
    )

    # ---- Stage 2c: 요소 라벨링 ----
    from poc.workflow_3.recording_filter.element_label import label_element

    labels = {}
    if settings.element_label_enabled:
        from PIL import Image

        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        ocr_client = Workflow1VLMClient(settings.element_ocr_service)
        label_vlm = Workflow1VLMClient(settings.element_vlm_service)
        crops_dir = out_dir / "element_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for ce in click_events:
            if not ce.is_click or not ce.cursor_xy:
                continue
            frame_image = Image.open(ce.frame_path).convert("RGB")
            label = label_element(
                frame_image, (ce.cursor_xy[0], ce.cursor_xy[1]), settings,
                ocr_client=ocr_client, vlm_client=label_vlm,
            )
            labels[ce.rank] = label
            box = crop_box_around(
                ce.cursor_xy[0], ce.cursor_xy[1], settings.element_crop_px,
                frame_image.size[0], frame_image.size[1],
            )
            save_debug_jpeg(
                frame_image.crop((box["left"], box["top"], box["right"], box["bottom"])),
                crops_dir / f"{ce.rank:03d}_{label.source}.jpg",
            )
        n_labeled = sum(1 for lb in labels.values() if lb.source != "none")
        print(f"[INFO] Stage 2c 완료: 라벨 {n_labeled} / {len(labels)}")

    timeline = build_timeline(click_events, gate_info=gate_info, labels=labels)
    save_debug_json(
        out_dir / "interaction_timeline.json",
        {"capture_dir": str(capture_dir), "events": timeline},
    )

    truncated = len(change_events) - len(click_events)
    save_debug_json(
        out_dir / "summary.json",
        {
            "capture_dir": str(capture_dir),
            "output_dir": str(out_dir),
            "total_change_events": stage1_total,
            "processed_for_click": len(click_events),
            "clicks": sum(1 for ce in click_events if ce.is_click),
            "timeline_events": len(timeline),
            "vlm_calls": len(click_events),
            "truncated": truncated > 0,
            "skipped_due_to_cap": max(0, truncated),
            "max_vlm_calls": settings.max_vlm_calls,
            "generations": len({info["generation"] for info in gate_info.values()}) if gate_info else 0,
            "gate_passed": len(change_events),
            "labeled": sum(1 for lb in labels.values() if lb.source != "none"),
            "elapsed": format_elapsed_ms(started_at),
        },
    )

    print(
        f"[INFO] 완료: change_events={len(change_events)}, clicks="
        f"{sum(1 for ce in click_events if ce.is_click)}, out={out_dir}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return "success" if timeline else "no_clicks"


if __name__ == "__main__":
    result = run_filter()
    raise SystemExit(0 if result in {"success", "no_clicks"} else 1)
