"""Stage 4 crop 기하: bbox -> margin 추가 -> clamp -> 잘라 저장.

영역을 잘라내는 것은 순수 CV 라 집에서 테스트된다(자른 crop 의 *재인식* 만 모델
필요). pipeline_plan.md Stage 4 규칙: 전송 전에 작은 margin 을 더하고, parent
screenshot 좌표와 함께 crop metadata 를 저장한다.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CropMeta:
    """한 crop 의 메타데이터(부모 좌표 보존)."""

    region_id: str
    region_type: str
    parent_bbox: tuple = field(default_factory=tuple)  # (left, top, right, bottom) 원본
    crop_bbox: tuple = field(default_factory=tuple)     # margin 포함, frame 으로 clamp
    crop_path: str = ""
    crop_wh: tuple = field(default_factory=tuple)       # (width, height)

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "parent_bbox": list(self.parent_bbox),
            "crop_bbox": list(self.crop_bbox),
            "crop_path": self.crop_path,
            "crop_wh": list(self.crop_wh),
        }


def compute_crop_box(
    bbox: tuple, frame_wh: tuple, margin_ratio: float = 0.06
) -> tuple | None:
    """bbox 에 margin 을 더하고 frame 범위로 clamp 한 (l, t, r, b) 를 반환.

    유효하지 않은 bbox(너비/높이 <= 0)면 None.
    margin 은 각 변에 box 해당 차원의 margin_ratio 만큼 더한다.
    """
    left, top, right, bottom = (int(v) for v in bbox)
    frame_w, frame_h = int(frame_wh[0]), int(frame_wh[1])
    if frame_w <= 0 or frame_h <= 0:
        return None
    if right <= left or bottom <= top:
        return None

    mx = int(round((right - left) * margin_ratio))
    my = int(round((bottom - top) * margin_ratio))

    l = max(0, left - mx)
    t = max(0, top - my)
    r = min(frame_w, right + mx)
    b = min(frame_h, bottom + my)
    if r <= l or b <= t:
        return None
    return (l, t, r, b)


def crop_region(
    image_path: str | Path,
    region_id: str,
    region_type: str,
    bbox: tuple,
    out_dir: str | Path,
    *,
    margin_ratio: float = 0.06,
) -> CropMeta | None:
    """원본 이미지에서 region 을 잘라 JPEG 로 저장하고 CropMeta 를 반환.

    bbox 가 유효하지 않거나 이미지 열기에 실패하면 None.
    """
    from PIL import Image

    image_path = Path(image_path)
    try:
        with Image.open(image_path) as img:
            frame_wh = (img.width, img.height)
            box = compute_crop_box(bbox, frame_wh, margin_ratio)
            if box is None:
                return None
            crop = img.crop(box)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # 디버그 로컬 저장은 JPEG (repo 컨벤션)
            crop_path = out_dir / f"{region_id}_{region_type}.jpg"
            crop.convert("RGB").save(crop_path, format="JPEG", quality=90)
            crop_wh = (crop.width, crop.height)
    except Exception as exc:
        print(f"[WARNING] crop 실패(region={region_id}): {exc}")
        return None

    return CropMeta(
        region_id=region_id,
        region_type=region_type,
        parent_bbox=tuple(int(v) for v in bbox),
        crop_bbox=box,
        crop_path=str(crop_path),
        crop_wh=crop_wh,
    )


def map_charts_to_crop_paths(charts, crop_metas: list[dict]) -> dict:
    """chart evidence(cNNN)에 chart crop 경로를 순서 대응시킨다(순수).

    crop 은 layout region id(rNNN) 기준으로 저장되지만 Chart 에는 bbox 가 없어
    좌표 대응이 불가능하다 -> "layout 의 k번째 chart crop <-> k번째 chart" 순서
    대응(marp/crop_map.py 와 동일 규칙; 여기는 추출 시점의 in-memory CropMeta 용).

    반환: {chart_region_id -> crop_path}. 개수가 어긋나면 앞에서부터만 대응.
    """
    chart_metas = sorted(
        (m for m in crop_metas
         if (m.get("region_type") or "").strip().lower() == "chart"
         and (m.get("crop_path") or "").strip()),
        key=lambda m: str(m.get("region_id") or ""),
    )
    lookup: dict = {}
    for chart, meta in zip(charts, chart_metas):
        lookup[chart.region_id] = str(meta["crop_path"])
    if chart_metas and charts and len(chart_metas) != len(charts):
        print(
            f"[WARNING] chart/crop 개수 불일치: charts={len(charts)}, "
            f"crops={len(chart_metas)} - 앞에서부터 순서 대응"
        )
    return lookup


__all__ = ["CropMeta", "compute_crop_box", "crop_region", "map_charts_to_crop_paths"]
