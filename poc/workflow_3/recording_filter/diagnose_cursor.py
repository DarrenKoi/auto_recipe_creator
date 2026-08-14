"""커서 좌표계 진단 — 매핑된 커서가 맞는지, 변화가 라이브 박스에서 새는지 가른다.

diagnose_clicks 가 "커서 ROI 에 변화가 없고(roi_p50=0) 실제 변화는 수천 px 떨어져
있다" 를 보여 준 다음 단계다. 그 증상의 원인은 정반대 조치를 요구하는 둘 중 하나다:

  H1 라이브 SEM 영상의 자율 갱신이 Stage 1.5 를 통과했다(커서는 정상, 이벤트가 가짜).
      -> 게이트를 고쳐야 한다. 임계를 낮추면 가짜 클릭만 늘어난다.
  H2 화면->프레임 커서 변환이 어긋났다(이벤트는 진짜, ROI 만 엉뚱한 곳).
      -> 좌표 변환을 고쳐야 한다. 게이트를 손대면 증상이 가려진다.

둘을 가르는 관찰값은 "변화 bbox 가 라이브 박스와 겹치는 비율" 과 "매핑된 커서가
프레임 안 어디에 찍히는가" 다. 숫자로 답이 안 나면 오버레이 이미지를 눈으로 보면
된다(빨강 십자 = 매핑된 커서, 노랑 = ROI, 마젠타 = 변화 blob, 시안 = 라이브 박스).

파일은 진단 폴더에만 쓰고 파이프라인 산출물은 건드리지 않는다. VLM 0 콜.

실행:
    RECORDING_FILTER_INPUT_DIR=<recording 경로> \\
      uv run python poc/workflow_3/recording_filter/diagnose_cursor.py
"""

import json
from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes
from poc.workflow_3.recording_filter.click_detect import _window_around, resolve_sidecar_cursor
from poc.workflow_3.recording_filter.frame_reduce import reduce_frames
from poc.workflow_3.recording_filter.region_gate import (
    REGION_MAP_KEY,
    _MetaIndex,
    assign_generations,
    load_frame_meta,
    nearest_meta,
    read_frame_size,
)
from poc.workflow_3.recording_filter.settings import load_recording_filter_settings

# 눈으로 확인할 오버레이 장수(전량을 그리면 진단이 또 하나의 대량 산출물이 된다).
_OVERLAY_SAMPLES = 8


def _load_live_boxes(out_dir: Path) -> dict:
    """직전 실행의 region_map.json 에서 세대별 라이브 박스를 읽는다."""
    path = out_dir / "region_map.json"
    if not path.is_file():
        print(f"[WARNING] region_map.json 이 없습니다: {path}")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] region_map.json 파싱 실패: {exc}")
        return {}
    boxes = {}
    for item in raw.get(REGION_MAP_KEY, []):
        boxes[int(item.get("generation", 0))] = item.get("live_box")
    return boxes


def _overlap_ratio(inner, outer) -> float:
    """inner bbox 면적 중 outer 와 겹치는 비율(0~1). outer 가 없으면 0."""
    if not inner or not outer:
        return 0.0
    left = max(int(inner["left"]), int(outer["left"]))
    top = max(int(inner["top"]), int(outer["top"]))
    right = min(int(inner["right"]), int(outer["right"]))
    bottom = min(int(inner["bottom"]), int(outer["bottom"]))
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area = max(1, (int(inner["right"]) - int(inner["left"])) * (int(inner["bottom"]) - int(inner["top"])))
    return inter / area


def _point_in_box(x, y, box) -> bool:
    if not box:
        return False
    return box["left"] <= x <= box["right"] and box["top"] <= y <= box["bottom"]


def diagnose(input_dir=None) -> dict:
    """커서 매핑과 라이브 박스 누출을 함께 진단한다."""
    from poc.workflow_3.recording_filter.filter_recording import (
        _resolve_frames_dir,
        _resolve_input_dir,
        _resolve_meta_dir,
        _resolve_output_dir,
    )
    from PIL import Image

    settings = load_recording_filter_settings()
    capture_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if capture_dir is None:
        return {"status": "input_not_found"}
    frames_dir = _resolve_frames_dir(capture_dir)
    if frames_dir is None:
        return {"status": "frames_not_found"}
    out_dir = _resolve_output_dir(capture_dir)
    diag_dir = out_dir / "diag_cursor"

    events = reduce_frames(frames_dir, settings)
    metas = load_frame_meta(_resolve_meta_dir(capture_dir, frames_dir))
    live_boxes = _load_live_boxes(out_dir)
    generations = assign_generations(metas)
    meta_index = _MetaIndex(metas, generations)

    if not events:
        print("[ERROR] Stage 1 이벤트가 없습니다.")
        return {"status": "no_events"}

    # ---- 좌표계 자체 점검: rect 크기 vs 프레임 크기(배율), 커서의 창 내부 여부 ----
    frame_wh = read_frame_size(events[0].frame_path)
    sample_meta = nearest_meta(metas, events[0].timestamp_sec) if metas else None
    if sample_meta and sample_meta.rect and frame_wh:
        rect = sample_meta.rect
        rect_w = int(rect["right"]) - int(rect["left"])
        rect_h = int(rect["bottom"]) - int(rect["top"])
        print(
            f"[INFO] rect={rect_w}x{rect_h}  frame={frame_wh[0]}x{frame_wh[1]}  "
            f"scale=({frame_wh[0] / max(1, rect_w):.3f}, {frame_wh[1] / max(1, rect_h):.3f})"
        )
    in_window = sum(1 for m in metas if m.cursor_in_window)
    print(f"[INFO] 사이드카 cursor_in_window: {in_window} / {len(metas)}")
    for gen, box in sorted(live_boxes.items()):
        print(f"[INFO] live_box gen{gen}: {box}")

    # ---- 이벤트별 통계 ----
    n = 0
    cursor_out_of_frame = 0
    cursor_in_live = 0
    change_overlap_live = 0     # 변화 blob 이 라이브 박스와 절반 이상 겹침 = ambient 성격.
    change_contained_live = 0   # 게이트가 실제로 ambient 로 강등하는 조건(완전 포함).
    cursor_xs, cursor_ys = [], []
    samples = []

    for ev in events:
        wh = read_frame_size(ev.frame_path)
        cursor = resolve_sidecar_cursor(ev, metas, wh)
        if cursor is None or not wh:
            continue
        n += 1
        width, height = wh
        cx, cy = cursor
        cursor_xs.append(cx)
        cursor_ys.append(cy)
        if not (0 <= cx < width and 0 <= cy < height):
            cursor_out_of_frame += 1
        _meta, gen = meta_index.lookup(ev.timestamp_sec)
        live = live_boxes.get(gen)
        if _point_in_box(cx, cy, live):
            cursor_in_live += 1
        ratio = _overlap_ratio(ev.change_bbox, live)
        if ratio >= 0.5:
            change_overlap_live += 1
        if ratio >= 0.999:
            change_contained_live += 1
        if len(samples) < _OVERLAY_SAMPLES and ev.rank % max(1, len(events) // _OVERLAY_SAMPLES) == 0:
            samples.append((ev, cursor, live, wh))

    print("")
    print(f"[INFO] 커서 매핑 이벤트 {n} 건")
    if cursor_xs:
        print(
            f"[INFO] 매핑 커서 x: min={min(cursor_xs)} max={max(cursor_xs)} / "
            f"y: min={min(cursor_ys)} max={max(cursor_ys)} (프레임 {width}x{height})"
        )
    print(f"[INFO] 프레임 밖으로 매핑된 커서: {cursor_out_of_frame} 건")
    print(f"[INFO] 커서가 라이브 박스 안: {cursor_in_live} 건")
    print(
        f"[INFO] 변화 blob 이 라이브 박스와 50%+ 겹침: {change_overlap_live} 건 "
        f"/ 완전 포함(게이트가 ambient 로 강등): {change_contained_live} 건"
    )

    # ---- 오버레이: 숫자로 안 갈리면 눈으로 ----
    diag_dir.mkdir(parents=True, exist_ok=True)
    for ev, cursor, live, wh in samples:
        try:
            image = Image.open(ev.frame_path).convert("RGB")
            elements = {
                "cursor": {
                    "bbox": _window_around(cursor[0], cursor[1], 24, wh[0], wh[1]),
                    "center": {"x": cursor[0], "y": cursor[1]},
                },
                "roi": {"bbox": _window_around(
                    cursor[0], cursor[1], settings.cursor_click_window_px, wh[0], wh[1]
                )},
                "change": {"bbox": ev.change_bbox},
            }
            colors = {"cursor": "red", "roi": "yellow", "change": "magenta"}
            if live:
                elements["live"] = {"bbox": live}
                colors["live"] = "cyan"
            save_marked_bboxes(
                image, elements, colors, diag_dir / f"{ev.rank:03d}_cursor.jpg"
            )
        except Exception as exc:
            print(f"[WARNING] 오버레이 저장 실패(rank={ev.rank}): {exc}")
    print(f"[INFO] 확인용 오버레이 {len(samples)} 장: {diag_dir}")
    print(
        f"[DIGEST] mapped={n} out_of_frame={cursor_out_of_frame} cursor_in_live={cursor_in_live} "
        f"overlap50={change_overlap_live} contained={change_contained_live} "
        f"in_window={in_window}/{len(metas)}"
    )
    return {"status": "ok", "mapped": n, "cursor_in_live": cursor_in_live}


if __name__ == "__main__":
    diagnose()
