"""클릭 이벤트를 시간순 InteractionEvent 타임라인으로 병합하고 오버레이를 기록한다.

스키마는 미래 타이핑(Stage 2b)과 공용이다(element/text 필드 예약). build_timeline 은
typing_events 인자를 미리 받아 추가 시 재설계가 없도록 한다.
"""

from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def derive_target_kind(region, element_source) -> str:
    """region + 라벨 출처로 이식 가능성 종류를 정한다.

    ui_control 은 다른 장비에서 라벨로 다시 찾을 수 있고, live_image 는 좌표가 아니라
    영상 내용에 의존해 CV 재해석이 필요하다. 파생 규칙이 바뀌어도 원본 region 이
    남아 있어 다시 계산할 수 있다.
    """
    if region == "live_image":
        return "live_image"
    if region == "ui" and element_source in {"ocr", "vlm"}:
        return "ui_control"
    return "unknown"


def build_timeline(click_events, typing_events=None, *, gate_info=None, labels=None) -> list[dict]:
    """클릭(+미래 타이핑) 이벤트를 시간순 정렬된 dict 목록으로 만든다.

    gate_info / labels 는 rank 키의 dict 다(없으면 기본값으로 채운다). 알람 사이클
    녹화처럼 Stage 1.5/2c 를 돌리지 않은 입력도 그대로 처리된다.
    """
    gate_info = gate_info or {}
    labels = labels or {}
    events: list[dict] = []
    for ce in click_events:
        if ce.status != "click" or not ce.is_click:
            continue
        coords = {"x": ce.cursor_xy[0], "y": ce.cursor_xy[1]} if ce.cursor_xy else None
        gate = gate_info.get(ce.rank) or {}
        label = labels.get(ce.rank)
        element_source = label.source if label is not None else "none"
        region = str(gate.get("region") or "unknown")
        events.append(
            {
                "t_sec": ce.timestamp_sec,
                "seq": 0,
                "action": "click",
                "coords": coords,
                "element": (label.text if label is not None and label.text else None),
                "element_source": element_source,
                "target_kind": derive_target_kind(region, element_source),
                "region": region,
                "generation": int(gate.get("generation") or 0),
                "occlusion": str(gate.get("occlusion") or "unknown"),
                "text": None,              # 예약: 타이핑 텍스트 (Stage 2b)
                "confidence": ce.confidence,
                "frame": Path(ce.frame_path).name,
                "source_frames": {
                    "prev": Path(ce.prev_frame_path).name,
                    "curr": Path(ce.frame_path).name,
                },
            }
        )
    for te in (typing_events or []):
        events.append(te)  # 이미 동일 스키마 dict 라고 가정(Stage 2b).

    events.sort(key=lambda e: e["t_sec"])
    for i, event in enumerate(events):
        event["seq"] = i
    return events


def write_click_overlays(click_events, out_dir: Path) -> list[Path]:
    """클릭 프레임에 커서 bbox + ROI 박스를 그려 별도 폴더에 저장한다."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ce in click_events:
        if not ce.is_click or ce.cursor_bbox is None:
            continue
        image = Image.open(ce.frame_path).convert("RGB")
        elements = {
            "cursor": {
                "bbox": ce.cursor_bbox,
                "center": {"x": ce.cursor_xy[0], "y": ce.cursor_xy[1]} if ce.cursor_xy else None,
            },
            "roi": {"bbox": ce.click_window},
        }
        colors = {"cursor": "red", "roi": "yellow"}
        out_path = out_dir / f"{ce.rank:03d}_{Path(ce.frame_path).name}"
        save_marked_bboxes(image, elements, colors, out_path)
        written.append(out_path)
    print(f"[INFO] 클릭 오버레이 {len(written)} 장 기록: {out_dir}")
    return written
