"""crop_map 스모크 테스트 (순수 + tmp 파일시스템, 서버/marp 불필요).

실행:
    uv run python -m side_projects.document_extraction.marp.test_crop_map_smoke
"""

import json
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.extraction.schemas import Chart, ExtractionResult
from side_projects.document_extraction.marp.crop_map import (
    build_crop_lookups,
    chart_crop_paths,
    map_chart_crops,
    parse_crop_filename,
)


def _result(screenshot_id: str, n_charts: int) -> ExtractionResult:
    result = ExtractionResult(source_image="x.webp", screenshot_id=screenshot_id)
    for i in range(n_charts):
        result.charts.append(Chart(region_id=f"c{i + 1:03d}", title=f"chart{i + 1}"))
    return result


def test_parse_crop_filename() -> None:
    assert parse_crop_filename("r002_chart.jpg") == ("r002", "chart")
    assert parse_crop_filename("r010_table.jpg") == ("r010", "table")
    assert parse_crop_filename("nounderscore.jpg") is None
    print("[PASS] test_parse_crop_filename")


def test_crops_json_preferred_over_filename_scan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp)
        # 파일명 스캔으로는 r001 이 먼저 오지만 crops.json 이 r005 만 chart 라고 명시
        (crops_dir / "r001_chart.jpg").write_bytes(b"x")
        (crops_dir / "r005_chart.jpg").write_bytes(b"x")
        metas = [
            {"region_id": "r005", "region_type": "chart",
             "crop_path": str(crops_dir / "r005_chart.jpg")},
            {"region_id": "r001", "region_type": "table",
             "crop_path": str(crops_dir / "r001_table.jpg")},
        ]
        (crops_dir / "crops.json").write_text(
            json.dumps(metas, ensure_ascii=False), encoding="utf-8")
        paths = chart_crop_paths(crops_dir)
        assert len(paths) == 1 and paths[0].endswith("r005_chart.jpg"), paths
    print("[PASS] test_crops_json_preferred_over_filename_scan")


def test_filename_scan_fallback_ordered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp)
        (crops_dir / "r010_chart.jpg").write_bytes(b"x")
        (crops_dir / "r002_chart.jpg").write_bytes(b"x")
        (crops_dir / "r003_table.jpg").write_bytes(b"x")  # chart 아님 -> 제외
        paths = chart_crop_paths(crops_dir)
        assert [Path(p).name for p in paths] == ["r002_chart.jpg", "r010_chart.jpg"]
    print("[PASS] test_filename_scan_fallback_ordered")


def test_order_mapping_charts_to_crops() -> None:
    """layout k번째 chart crop <-> evidence k번째 chart(cNNN) 순서 대응."""
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp)
        (crops_dir / "r002_chart.jpg").write_bytes(b"x")
        (crops_dir / "r007_chart.jpg").write_bytes(b"x")
        result = _result("doc_s001", n_charts=2)
        lookup = map_chart_crops(result, crops_dir)
        assert lookup["c001"].endswith("r002_chart.jpg")
        assert lookup["c002"].endswith("r007_chart.jpg")
    print("[PASS] test_order_mapping_charts_to_crops")


def test_count_mismatch_maps_prefix_only() -> None:
    """crop 1개 / chart 2개 -> 앞의 chart 만 대응(오대응 방지)."""
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp)
        (crops_dir / "r002_chart.jpg").write_bytes(b"x")
        result = _result("doc_s001", n_charts=2)
        lookup = map_chart_crops(result, crops_dir)
        assert set(lookup) == {"c001"}
    print("[PASS] test_count_mismatch_maps_prefix_only")


def test_build_crop_lookups_e2e() -> None:
    """images_dir/_crops/<screenshot_id>/ 규약으로 전체 lookup 을 구성."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = Path(tmp)
        crops_dir = images_dir / "_crops" / "doc_s001"
        crops_dir.mkdir(parents=True)
        (crops_dir / "r002_chart.jpg").write_bytes(b"x")
        r1 = _result("doc_s001", n_charts=1)
        r2 = _result("doc_s002", n_charts=1)  # crop 폴더 없음 -> lookup 제외
        lookups = build_crop_lookups([r1, r2], images_dir)
        assert "doc_s001" in lookups and "doc_s002" not in lookups
        assert lookups["doc_s001"]["c001"].endswith("r002_chart.jpg")
    print("[PASS] test_build_crop_lookups_e2e")


def main() -> int:
    test_parse_crop_filename()
    test_crops_json_preferred_over_filename_scan()
    test_filename_scan_fallback_ordered()
    test_order_mapping_charts_to_crops()
    test_count_mismatch_maps_prefix_only()
    test_build_crop_lookups_e2e()
    print("\n[INFO] 모든 crop_map 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
