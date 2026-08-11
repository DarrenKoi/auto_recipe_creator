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
from poc.workflow_3.recording_filter.element_label import crop_box_around, label_element
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


def _resolve_meta_dir(capture_dir: Path, frames_dir: Path) -> Path:
    """사이드카(frame_meta.jsonl) 를 찾을 디렉터리를 고른다(캡처 루트 우선).

    (2026-08-10 최종 리뷰 FINDING 8) FrameMetaWriter 는 사이드카를 **녹화 루트**에
    쓰는데 예전에는 frames_dir 에서만 찾았다. RecordingSession 이 JPEG 을 루트에
    바로 쓰기 때문에 지금은 두 경로가 같지만, frames/ 하위 폴더가 생기는 순간
    사이드카가 조용히 사라지고 모든 이벤트가 [INFO] 한 줄만 남긴 채 candidate 로
    degrade 한다. 루트를 먼저 보고, 없으면 frames_dir 로 폴백한다.
    """
    from poc.workflow_3.recording_filter.region_gate import FRAME_META_FILENAME

    if (capture_dir / FRAME_META_FILENAME).is_file():
        return capture_dir
    if frames_dir != capture_dir and (frames_dir / FRAME_META_FILENAME).is_file():
        print(f"[INFO] 사이드카를 frames 디렉터리에서 찾았습니다: {frames_dir}")
        return frames_dir
    return capture_dir


def _estimate_label_vlm_calls(labels) -> int:
    """Stage 2c 의 실제 호출 수를 추정한다(OCR 1콜, 폴백 시 VLM 1콜 추가).

    label_element 는 OCR 을 먼저 부르고 텍스트를 못 얻었을 때만 VLM 으로 넘어간다
    (element_label.py). 따라서 source=="ocr" 이면 1콜, 그 외(vlm/none)는 2콜이다.
    """
    return sum(1 if getattr(lb, "source", "none") == "ocr" else 2 for lb in labels.values())


def _copy_change_events(change_events, out_dir: Path) -> None:
    """게이트를 통과한 프레임을 rank 접두로 복사한다.

    (2026-08-10 최종 리뷰 FINDING 7) 예전에는 Stage 1 생존 프레임 전체를 복사해,
    10분 세션이면 1500~3000장(0.5~1GB)을 원본 옆에 그대로 늘렸고 그중 대부분은
    바로 뒤 게이트가 버렸다. change_events.json 은 여전히 Stage 1 전체를 담으므로
    감사 추적은 유지된다.
    """
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


def _label_one_click(ce, settings, *, ocr_client, vlm_client, crops_dir):
    """단일 클릭 이벤트의 요소 라벨을 계산하고 crop 을 저장한다.

    label_element 자체는 OCR/VLM 실패를 삼켜 source="none" 으로 돌아오지만,
    이 함수가 여는 프레임 파일(Image.open)이나 crop 저장(save_debug_jpeg)은
    방어되어 있지 않다 - 손상/누락된 프레임이면 예외를 그대로 던진다. 호출부
    (_label_click_events)가 이벤트 단위로 잡아야 한다는 계약이다.
    """
    from PIL import Image

    frame_image = Image.open(ce.frame_path).convert("RGB")
    label = label_element(
        frame_image, (ce.cursor_xy[0], ce.cursor_xy[1]), settings,
        ocr_client=ocr_client, vlm_client=vlm_client,
    )
    box = crop_box_around(
        ce.cursor_xy[0], ce.cursor_xy[1], settings.element_crop_px,
        frame_image.size[0], frame_image.size[1],
    )
    save_debug_jpeg(
        frame_image.crop((box["left"], box["top"], box["right"], box["bottom"])),
        crops_dir / f"{ce.rank:03d}_{label.source}.jpg",
    )
    return label


def _label_click_events(click_events, settings, crops_dir, *, ocr_client, vlm_client):
    """클릭 이벤트 목록을 순회하며 라벨링한다(Stage 2c 본체, (labels, label_errors) 반환).

    한 이벤트가 실패(프레임 손상/누락, crop 저장 실패 등)해도 나머지 이벤트 처리와
    interaction_timeline.json/summary.json 기록을 막지 않는다 - 재현 불가능한 실제
    녹화 세션에서 프레임 하나가 나쁘다고 세션 전체 라벨을 잃는 것은 가장 나쁜 실패
    형태다. 실패한 이벤트는 labels 에서 빠지고, build_timeline 이 문서화된 기본값
    (element=None, element_source="none", target_kind는 derive_target_kind 규칙대로)
    으로 채운다.
    """
    labels = {}
    label_errors = 0
    crops_dir.mkdir(parents=True, exist_ok=True)
    for ce in click_events:
        if not ce.is_click or not ce.cursor_xy:
            continue
        try:
            labels[ce.rank] = _label_one_click(
                ce, settings, ocr_client=ocr_client, vlm_client=vlm_client, crops_dir=crops_dir,
            )
        except Exception as exc:
            label_errors += 1
            print(
                f"[WARNING] Stage 2c 라벨링 실패(건너뜀, rank={ce.rank}, "
                f"frame={ce.frame_path}): {exc}"
            )
    return labels, label_errors


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

    metas = load_frame_meta(_resolve_meta_dir(capture_dir, frames_dir))
    gate_info = {}
    ambient_dropped = 0
    occluded_excluded = 0
    region_map_calls = 0
    if settings.region_gate_enabled:
        if client is None:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            # (FINDING 4) model_name 을 넘기지 않는다 - service slug 가 자기 모델을
            # 들고 있으므로, 넘기면 mai-ui 라우트에 ui-venus 모델명이 실려 나간다.
            client = Workflow1VLMClient(settings.vlm_service)
        region_maps = build_region_maps(change_events, metas, client, out_dir)
        region_map_calls = len(region_maps)
        gated = apply_region_gate(change_events, metas, region_maps)
        for event, generation, verdict, occlusion, region in gated:
            gate_info[event.rank] = {
                "generation": generation,
                "region": region,          # (FINDING 3) verdict 파생이 아니라 기하 값.
                "occlusion": occlusion,
                "verdict": verdict,
            }
        # ambient 와 가려진 프레임은 비싼 Stage 2a 에 태우지 않는다. 다만 "조용히"
        # 버리지는 않는다 - 사유별 건수를 세어 summary.json 에 남긴다(FINDING 1).
        kept = []
        for event, _generation, verdict, occlusion, _region in gated:
            if verdict != "candidate":
                ambient_dropped += 1
                continue
            if occlusion == "full":
                occluded_excluded += 1
                continue
            kept.append(event)
        change_events = kept
        if occluded_excluded:
            print(
                f"[WARNING] 창이 가려진(occlusion=full) 이벤트 {occluded_excluded} 건을 "
                "Stage 2a 에서 제외했습니다(summary.json 의 occluded_events_excluded)."
            )
        print(f"[INFO] Stage 1.5 통과: {len(change_events)} 건이 Stage 2a 로 갑니다.")

    # 게이트 통과분만 디스크에 복사한다(FINDING 7 - 원본 옆 GB 단위 중복 방지).
    _copy_change_events(change_events, out_dir / "change_events")

    # ---- Stage 2a ----
    if client is None:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.vlm_service)
    click_events = detect_clicks(change_events, settings, client=client, metas=metas)
    write_click_overlays(
        [ce for ce in click_events if ce.is_click], out_dir / "click_events"
    )

    # ---- Stage 2c: 요소 라벨링 ----
    labels = {}
    label_errors = 0
    if settings.element_label_enabled:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        ocr_client = Workflow1VLMClient(settings.element_ocr_service)
        label_vlm = Workflow1VLMClient(settings.element_vlm_service)
        labels, label_errors = _label_click_events(
            click_events, settings, out_dir / "element_crops",
            ocr_client=ocr_client, vlm_client=label_vlm,
        )
        n_labeled = sum(1 for lb in labels.values() if lb.source != "none")
        print(
            f"[INFO] Stage 2c 완료: 라벨 {n_labeled} / {len(labels)} "
            f"(건너뜀 {label_errors} 건)"
        )

    timeline = build_timeline(click_events, gate_info=gate_info, labels=labels)
    save_debug_json(
        out_dir / "interaction_timeline.json",
        {"capture_dir": str(capture_dir), "events": timeline},
    )

    truncated = len(change_events) - len(click_events)
    label_calls = _estimate_label_vlm_calls(labels)
    save_debug_json(
        out_dir / "summary.json",
        {
            "capture_dir": str(capture_dir),
            "output_dir": str(out_dir),
            "total_change_events": stage1_total,
            "processed_for_click": len(click_events),
            "clicks": sum(1 for ce in click_events if ce.is_click),
            "timeline_events": len(timeline),
            # (FINDING 6) 예전 "vlm_calls" 는 Stage 2a 만 세면서 전체처럼 읽혔다.
            # 스테이지별로 분해하고 합계를 따로 둔다(2c 는 OCR/VLM 폴백 규칙 기반 추정).
            "vlm_calls_stage1_5_region_map": region_map_calls,
            "vlm_calls_stage2a_cursor": len(click_events),
            "vlm_calls_stage2c_label_estimate": label_calls,
            "vlm_calls_total_estimate": region_map_calls + len(click_events) + label_calls,
            "truncated": truncated > 0,
            "skipped_due_to_cap": max(0, truncated),
            "max_vlm_calls": settings.max_vlm_calls,
            "generations": len({info["generation"] for info in gate_info.values()}) if gate_info else 0,
            "gate_passed": len(change_events),
            "ambient_events_dropped": ambient_dropped,
            "occluded_events_excluded": occluded_excluded,
            "labeled": sum(1 for lb in labels.values() if lb.source != "none"),
            "element_label_errors": label_errors,
            "elapsed": format_elapsed_ms(started_at),
        },
    )

    print(
        f"[INFO] 완료: change_events={len(change_events)}, clicks="
        f"{sum(1 for ce in click_events if ce.is_click)}, out={out_dir}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )

    # (FINDING 1) Stage 1 이 이벤트를 냈는데 게이트/가림이 전부 걷어갔다면 그것은
    # 성공이 아니다. 예전에는 timeline 이 비어도 "no_clicks"(exit 0) 라, 사이드카
    # 버그로 모든 프레임이 "full" 로 찍힌 세션이 조용히 성공처럼 끝났다.
    if stage1_total > 0 and not change_events:
        print(
            "[WARNING] 이벤트가 하나도 남지 않았습니다 - Stage 1 은 "
            f"{stage1_total} 건을 찾았지만 게이트({ambient_dropped} 건)/가림"
            f"({occluded_excluded} 건)이 전부 걷어냈습니다. 가림이 대부분이면 "
            "녹화 창 핸들/가림 판정을, ambient 가 대부분이면 영역 지도(region_map.json)를 "
            "먼저 보세요."
        )
        return "all_events_discarded"
    return "success" if timeline else "no_clicks"


if __name__ == "__main__":
    result = run_filter()
    raise SystemExit(0 if result in {"success", "no_clicks"} else 1)
