"""crop 파일 <-> chart region 자동 대응 (status.md 계획 8번).

Stage 4 crop refine 은 layout region id(rNNN) 기준으로
`<이미지폴더>/_crops/<screenshot_id>/rNNN_chart.jpg` 를 저장하지만, evidence 의
Chart 는 OCR 순번 기반 cNNN id 를 갖는다. Chart 에는 bbox 가 없어 좌표 대응이
불가능하므로 "layout 의 k번째 chart crop <-> evidence 의 k번째 chart" 순서
대응을 쓴다. 대부분의 슬라이드는 chart 가 0~1개라 순서 대응으로 충분하고,
개수가 어긋나면 경고를 남기고 앞에서부터만 대응한다(값 창작/오대응 방지).

대응 소스 우선순위:
    1) crops.json  — extract_screenshot 이 저장하는 CropMeta 목록(bbox 포함)
    2) 파일명 스캔 — `rNNN_chart.*` 패턴(과거 실행 산출물 호환)
"""

import json
from pathlib import Path


def parse_crop_filename(name: str) -> tuple[str, str] | None:
    """'r002_chart.jpg' -> ('r002', 'chart'). 패턴이 아니면 None(순수)."""
    stem = Path(name).stem
    region_id, sep, region_type = stem.partition("_")
    if not sep or not region_id or not region_type:
        return None
    return region_id, region_type


def chart_crop_paths(crops_dir: Path) -> list[str]:
    """스크린샷 1장의 crop 폴더에서 chart crop 경로를 layout 순서(rNNN)로 반환."""
    crops_dir = Path(crops_dir)
    entries: list[tuple[str, str]] = []

    crops_json = crops_dir / "crops.json"
    if crops_json.exists():
        try:
            metas = json.loads(crops_json.read_text(encoding="utf-8"))
            for meta in metas:
                if not isinstance(meta, dict):
                    continue
                if (meta.get("region_type") or "").strip().lower() != "chart":
                    continue
                path = (meta.get("crop_path") or "").strip()
                if path:
                    entries.append((str(meta.get("region_id") or ""), path))
        except Exception as exc:
            print(f"[WARNING] crops.json 파싱 실패({crops_dir}): {exc}")

    if not entries:
        for path in crops_dir.glob("*_chart.*"):
            parsed = parse_crop_filename(path.name)
            if parsed is not None:
                entries.append((parsed[0], str(path)))

    entries.sort(key=lambda e: e[0])  # rNNN 은 zero-pad 라 문자열 정렬 = 순서
    return [path for _, path in entries]


def map_chart_crops(result, crops_dir) -> dict:
    """ExtractionResult 1장의 charts(cNNN)에 crop 경로를 순서 대응시킨다.

    반환: {chart_region_id -> crop 경로} (evidence_to_marp 의 crop_lookup 형식).
    존재하지 않는 파일은 제외한다.
    """
    paths = chart_crop_paths(Path(crops_dir))
    lookup: dict = {}
    for chart, path in zip(result.charts, paths):
        if Path(path).exists():
            lookup[chart.region_id] = path
    if paths and result.charts and len(paths) != len(result.charts):
        print(
            f"[WARNING] chart/crop 개수 불일치({Path(crops_dir).name}): "
            f"charts={len(result.charts)}, crops={len(paths)} - 앞에서부터 순서 대응"
        )
    return lookup


def build_crop_lookups(results, images_dir) -> dict:
    """{screenshot_id -> {chart_region_id -> crop 경로}} 전체 자동 구성.

    images_dir: 캡처 페이지 폴더(그 아래 `_crops/<screenshot_id>/` 를 찾는다 —
    extract_screenshot 의 crop_dir 규약과 동일).
    """
    images_dir = Path(images_dir)
    lookups: dict = {}
    for result in results:
        crops_dir = images_dir / "_crops" / result.screenshot_id
        if not crops_dir.exists():
            continue
        lookup = map_chart_crops(result, crops_dir)
        if lookup:
            lookups[result.screenshot_id] = lookup
    if lookups:
        total = sum(len(v) for v in lookups.values())
        print(f"[INFO] crop 자동 대응: {len(lookups)}장 스크린샷, chart {total}건")
    return lookups


__all__ = ["build_crop_lookups", "chart_crop_paths", "map_chart_crops",
           "parse_crop_filename"]
