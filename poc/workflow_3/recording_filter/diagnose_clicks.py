"""Stage 2a 진단 — 클릭이 0 건인 이유를 VLM 콜 없이 특정한다.

filter_recording 이 click_events/ 와 element_crops/ 를 비워 두고 끝났을 때, 원인은
"Stage 1 이 아무것도 못 찾음" / "Stage 1.5 가 전부 걷어냄" / "Stage 2a 의 ROI 판정이
매번 임계 미달" 중 하나다. 세 가지는 조치가 완전히 다르므로(각각 녹화 문제 / 게이트
문제 / 임계·좌표 문제) 추정 대신 실제 숫자로 구분한다.

이 스크립트는 **판정만** 다시 계산한다(파일을 쓰지 않고 VLM 도 부르지 않는다).
Stage 2a 의 사이드카 경로는 원래도 VLM 을 쓰지 않으므로(click_detect.detect_clicks),
같은 입력에 대해 같은 답이 나온다.

실행:
    RECORDING_FILTER_INPUT_DIR=<recording 경로> \\
      uv run python poc/workflow_3/recording_filter/diagnose_clicks.py
"""

import json
from pathlib import Path

from poc.workflow_3.recording_filter.click_detect import (
    _count_changed_in_window,
    _diff_mask,
    _window_around,
    resolve_sidecar_cursor,
)
from poc.workflow_3.recording_filter.frame_reduce import collect_frame_paths, reduce_frames
from poc.workflow_3.recording_filter.region_gate import (
    load_frame_meta,
    nearest_meta,
    read_frame_size,
)
from poc.workflow_3.recording_filter.settings import load_recording_filter_settings

# 임계를 바꿨을 때 클릭이 몇 건이 되는지 함께 보여 준다(조치 방향 판단용).
_WINDOW_SWEEP = (200, 320, 480)
_THRESHOLD_SWEEP = (300, 800, 1500)


def _percentile(values, ratio: float) -> int:
    """정렬된 목록의 백분위 값을 고른다(numpy 없이)."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return int(ordered[index])


def _bbox_center(bbox):
    if not bbox:
        return None
    return (
        (int(bbox["left"]) + int(bbox["right"])) / 2.0,
        (int(bbox["top"]) + int(bbox["bottom"])) / 2.0,
    )


def _load_gate_verdicts(out_dir: Path) -> dict:
    """직전 실행의 change_events.json 에서 rank 별 게이트 판정을 읽는다(없으면 빈 dict)."""
    path = out_dir / "change_events.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] change_events.json 파싱 실패(게이트 판정 없이 진행): {exc}")
        return {}
    return {
        int(item.get("rank", -1)): item
        for item in raw.get("events", [])
        if item.get("verdict") is not None
    }


def diagnose(input_dir=None) -> dict:
    """Stage 1/1.5/2a 를 재현해 단계별 잔존 건수와 ROI 통계를 출력한다."""
    from poc.workflow_3.recording_filter.filter_recording import (
        _resolve_frames_dir,
        _resolve_input_dir,
        _resolve_meta_dir,
        _resolve_output_dir,
    )

    settings = load_recording_filter_settings()
    capture_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if capture_dir is None:
        return {"status": "input_not_found"}
    frames_dir = _resolve_frames_dir(capture_dir)
    if frames_dir is None:
        return {"status": "frames_not_found"}
    out_dir = _resolve_output_dir(capture_dir)

    n_frames = len(collect_frame_paths(frames_dir))
    events = reduce_frames(frames_dir, settings)
    metas = load_frame_meta(_resolve_meta_dir(capture_dir, frames_dir))
    gate = _load_gate_verdicts(out_dir)

    # Stage 1.5 판정은 직전 실행 산출물에서 읽는다(다시 계산하려면 VLM 이 필요하다).
    if gate:
        candidates = [ev for ev in events if (gate.get(ev.rank) or {}).get("verdict") == "candidate"]
        occluded = sum(
            1 for ev in events if (gate.get(ev.rank) or {}).get("occlusion") == "full"
        )
    else:
        print("[INFO] 이전 change_events.json 이 없어 게이트 판정 없이 Stage 1 전량을 봅니다.")
        candidates = list(events)
        occluded = 0

    joined = 0
    cursor_none = 0
    changed_px = []
    cursor_to_change_px = []
    sweep = {(w, t): 0 for w in _WINDOW_SWEEP for t in _THRESHOLD_SWEEP}

    for ev in candidates:
        frame_wh = read_frame_size(ev.frame_path) if metas else None
        cursor = resolve_sidecar_cursor(ev, metas, frame_wh)
        if cursor is None:
            cursor_none += 1
            meta = nearest_meta(metas, ev.timestamp_sec) if metas else None
            if metas and meta is None:
                pass  # 조인 실패(사이드카가 죽었거나 시간이 멀다).
            continue
        joined += 1
        mask = _diff_mask(
            Path(ev.prev_frame_path), Path(ev.frame_path), settings.click_diff_threshold
        )
        if mask is None or not frame_wh:
            continue
        width, height = frame_wh
        window = _window_around(cursor[0], cursor[1], settings.cursor_click_window_px, width, height)
        changed_px.append(_count_changed_in_window(mask, window))
        center = _bbox_center(ev.change_bbox)
        if center:
            cursor_to_change_px.append(
                int(round(((center[0] - cursor[0]) ** 2 + (center[1] - cursor[1]) ** 2) ** 0.5))
            )
        for side in _WINDOW_SWEEP:
            box = _window_around(cursor[0], cursor[1], side, width, height)
            count = _count_changed_in_window(mask, box)
            for threshold in _THRESHOLD_SWEEP:
                if count >= threshold:
                    sweep[(side, threshold)] += 1

    clicks_now = sum(1 for c in changed_px if c >= settings.click_min_changed_px)
    print("")
    print(f"[INFO] frames={n_frames}  stage1_events={len(events)}  "
          f"gate_candidates={len(candidates)}  occluded_full={occluded}")
    print(f"[INFO] sidecar_records={len(metas)}  cursor_joined={joined}  cursor_missing={cursor_none}")
    if changed_px:
        print(
            f"[INFO] ROI 변화픽셀 (window={settings.cursor_click_window_px}px): "
            f"min={min(changed_px)} p50={_percentile(changed_px, 0.5)} "
            f"p90={_percentile(changed_px, 0.9)} max={max(changed_px)} "
            f"/ 임계={settings.click_min_changed_px}"
        )
    if cursor_to_change_px:
        print(
            f"[INFO] 커서<->변화중심 거리 px: min={min(cursor_to_change_px)} "
            f"p50={_percentile(cursor_to_change_px, 0.5)} "
            f"p90={_percentile(cursor_to_change_px, 0.9)} max={max(cursor_to_change_px)}"
        )
    print("[INFO] window x threshold 별 클릭 건수:")
    for side in _WINDOW_SWEEP:
        row = "  ".join(f"t={t}:{sweep[(side, t)]:4d}" for t in _THRESHOLD_SWEEP)
        print(f"       window={side:4d}px  {row}")
    print(
        f"[DIGEST] frames={n_frames} stage1={len(events)} gate={len(candidates)} "
        f"joined={joined} nocursor={cursor_none} clicks={clicks_now} "
        f"roi_p50={_percentile(changed_px, 0.5)} dist_p50={_percentile(cursor_to_change_px, 0.5)}"
    )
    return {
        "status": "ok",
        "frames": n_frames,
        "stage1_events": len(events),
        "gate_candidates": len(candidates),
        "cursor_joined": joined,
        "cursor_missing": cursor_none,
        "clicks": clicks_now,
    }


if __name__ == "__main__":
    diagnose()
