"""STAGE 1.5 - 영역 게이트: 라이브 SEM 영상의 자율 변화를 이벤트에서 걷어낸다.

알람 사이클은 장비가 멈춰 화면이 정적이라 "변화 = 사람의 조작"이 성립했다. 엔지니어가
수동 조작하는 동안은 라이브 SEM 영상이 계속 갱신되므로 그 전제가 깨진다. 이 스테이지는
프레임을 live_image(라이브 박스) 와 ui(나머지) 로만 나눠, 라이브 박스 안에서만 일어난
변화를 ambient 로 강등한다.

비용 설계가 핵심이다 - 영역 지도는 **세대당 한 번만** VLM(detect_sem_box)을 쓰고,
프레임 단위 게이팅은 순수 기하 비교라 VLM 0콜이다. 세션 길이가 아니라 세대 수에만
비용이 비례한다.

레이아웃 세대(generation): 창을 옮기거나 리사이즈하면 좌표계가 통째로 바뀐다. 창 rect
가 달라지는 시점마다 새 세대를 열어 지도를 다시 뽑는다. 장비 A/B/C 의 레이아웃 차이는
detect_sem_box 가 그 장비의 실제 프레임을 보고 찾으므로 자동으로 흡수된다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

FRAME_META_FILENAME = "frame_meta.jsonl"


@dataclass
class FrameMeta:
    """사이드카 1줄 - 프레임과 같은 시각의 창 상태."""

    t_sec: float
    rect: dict | None
    occlusion: str
    cursor_xy: list | None
    cursor_in_window: bool


@dataclass
class RegionMap:
    """한 레이아웃 세대의 영역 지도."""

    generation: int
    live_box: dict | None   # 프레임 픽셀 기준 {left,top,right,bottom}. None = 검출 실패.


def load_frame_meta(capture_dir) -> list:
    """frame_meta.jsonl 을 읽어 FrameMeta 목록으로 만든다. 없으면 빈 목록."""
    path = Path(capture_dir) / FRAME_META_FILENAME
    if not path.is_file():
        print(f"[INFO] 사이드카 없음 - 커서/가림 신호 없이 진행합니다: {path}")
        return []
    metas = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except Exception:
            continue
        metas.append(FrameMeta(
            t_sec=float(raw.get("t_sec") or 0.0),
            rect=raw.get("window_rect"),
            occlusion=str(raw.get("occlusion") or "unknown"),
            cursor_xy=raw.get("cursor_screen_xy"),
            cursor_in_window=bool(raw.get("cursor_in_window")),
        ))
    metas.sort(key=lambda m: m.t_sec)
    print(f"[INFO] 사이드카 {len(metas)} 건 로드: {path}")
    return metas


def nearest_meta(metas, t_sec):
    """t_sec 에 가장 가까운 FrameMeta 를 돌려준다(없으면 None).

    capture 호출 순번과 저장된 프레임 seq 는 어긋난다(변화 없는 샘플은 저장되지
    않는다). 그래서 순번이 아니라 시각으로 조인한다.
    """
    if not metas:
        return None
    return min(metas, key=lambda m: abs(m.t_sec - float(t_sec)))


def _rect_key(rect):
    if not rect:
        return None
    return (int(rect["left"]), int(rect["top"]), int(rect["right"]), int(rect["bottom"]))


def assign_generations(metas) -> list:
    """FrameMeta 목록에 레이아웃 세대 번호를 매긴다.

    rect 가 바뀌는 지점마다 세대가 하나 늘어난다. rect 가 없는 프레임은 직전 세대를
    물려받는다(판정 불가를 이유로 세대를 쪼개면 지도만 늘고 이득이 없다).
    """
    generations = []
    current = 0
    last_key = None
    for meta in metas:
        key = _rect_key(meta.rect)
        if key is not None:
            if last_key is not None and key != last_key:
                current += 1
            last_key = key
        generations.append(current)
    return generations


def _boxes_overlap(a, b) -> bool:
    return not (
        a["right"] <= b["left"] or a["left"] >= b["right"]
        or a["bottom"] <= b["top"] or a["top"] >= b["bottom"]
    )


def _box_contains(outer, inner) -> bool:
    return (
        inner["left"] >= outer["left"] and inner["top"] >= outer["top"]
        and inner["right"] <= outer["right"] and inner["bottom"] <= outer["bottom"]
    )


def gate_verdict(change_bbox, live_box, cursor_in_live, has_meta) -> str:
    """변화 bbox 를 ambient / candidate 로 판정한다.

    ambient 는 "라이브 박스 안에서만 변했고 커서는 밖" 인 경우뿐이다. 나머지는 전부
    candidate 로 승격한다 - 조용히 이벤트를 잃는 것보다 오탐이 낫다.
    """
    if live_box is None:
        return "candidate"          # 지도 없음 - 게이트 무효화.
    if not has_meta:
        return "candidate"          # 커서 예외를 쓸 수 없음 - 안전 쪽으로.
    if cursor_in_live:
        return "candidate"          # 라이브 영상 직접 조작 가능성.
    if change_bbox and _box_contains(live_box, change_bbox):
        return "ambient"
    return "candidate"


def build_region_maps(events, metas, client, out_dir) -> dict:
    """세대별 영역 지도를 만든다(세대당 detect_sem_box 1회 + 확인용 오버레이 저장)."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")
    from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box

    generations = assign_generations(metas)
    gen_by_time = list(zip([m.t_sec for m in metas], generations))
    maps = {}
    out_dir = Path(out_dir)

    for event in events:
        generation = _generation_for(gen_by_time, event.timestamp_sec)
        if generation in maps:
            continue
        try:
            image = Image.open(event.frame_path).convert("RGB")
            detection = detect_sem_box(image, client)
            live_box = detection.bbox_px if detection.detected else None
        except Exception as exc:
            print(f"[WARNING] 세대 {generation} 영역 지도 검출 실패(게이트 없이 통과): {exc}")
            live_box = None
            image = None
        maps[generation] = RegionMap(generation=generation, live_box=live_box)
        if image is not None and live_box is not None:
            try:
                save_marked_bboxes(
                    image, {"live_image": {"bbox": live_box}},
                    {"live_image": "cyan"},
                    out_dir / f"region_map_gen{generation}.jpg",
                )
            except Exception as exc:
                print(f"[WARNING] 영역 지도 오버레이 저장 실패: {exc}")

    # 스펙 8.2 의 region_map.json - 오피스에서 세대별 박스를 텍스트로 대조할 수 있어야 한다.
    from poc.workflow_3.debug_artifacts import save_debug_json

    save_debug_json(
        out_dir / "region_map.json",
        {
            "generations": [
                {"generation": gen, "live_box": region_map.live_box}
                for gen, region_map in sorted(maps.items())
            ]
        },
    )
    print(f"[INFO] 영역 지도 {len(maps)} 세대 확보(VLM 호출 = 세대 수).")
    return maps


def _generation_for(gen_by_time, t_sec) -> int:
    """시각으로 세대 번호를 찾는다(사이드카 없으면 항상 0)."""
    if not gen_by_time:
        return 0
    return min(gen_by_time, key=lambda pair: abs(pair[0] - float(t_sec)))[1]


def apply_region_gate(events, metas, region_maps) -> list:
    """변화 이벤트마다 (event, generation, verdict, occlusion) 을 계산한다."""
    generations = assign_generations(metas)
    gen_by_time = list(zip([m.t_sec for m in metas], generations))
    has_meta = bool(metas)
    results = []
    for event in events:
        generation = _generation_for(gen_by_time, event.timestamp_sec)
        region_map = region_maps.get(generation)
        live_box = region_map.live_box if region_map else None
        meta = nearest_meta(metas, event.timestamp_sec)
        cursor_in_live = False
        occlusion = "unknown"
        if meta is not None:
            occlusion = meta.occlusion
            if meta.cursor_xy and meta.rect and live_box:
                # 화면 좌표 -> 프레임 좌표로 옮긴 뒤 라이브 박스 포함 여부를 본다.
                fx = int(meta.cursor_xy[0]) - int(meta.rect["left"])
                fy = int(meta.cursor_xy[1]) - int(meta.rect["top"])
                cursor_in_live = (
                    live_box["left"] <= fx <= live_box["right"]
                    and live_box["top"] <= fy <= live_box["bottom"]
                )
        verdict = gate_verdict(event.change_bbox, live_box, cursor_in_live, has_meta)
        results.append((event, generation, verdict, occlusion))

    n_ambient = sum(1 for _e, _g, v, _o in results if v == "ambient")
    print(f"[INFO] Stage 1.5 완료: ambient={n_ambient} / 전체={len(results)}")
    return results
