"""Stage 7 검증(SSIM + 자동 강등) 스모크 테스트 (순수 함수, marp-cli/이미지 불필요).

실행:
    uv run python -m side_projects.document_extraction.marp.test_verify_smoke
"""

import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from side_projects.document_extraction.extraction.schemas import (
    Chart, ExtractionResult, Region)
from side_projects.document_extraction.marp.verify import (
    DEFAULT_SSIM_FLOOR,
    apply_downgrade_plans,
    plan_downgrade,
    flag_low_fidelity,
    slide_fidelity,
    ssim,
    verify_and_downgrade,
    whole_slide_marp,
)


def test_ssim_identical_is_one() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((64, 80)) * 255).astype(np.uint8)
    assert abs(ssim(img, img) - 1.0) < 1e-6
    print("[PASS] test_ssim_identical_is_one")


def test_ssim_drops_for_degraded() -> None:
    rng = np.random.default_rng(1)
    img = (rng.random((64, 80)) * 255).astype(np.uint8)
    noisy = np.clip(img.astype(np.int16) + 80, 0, 255).astype(np.uint8)
    s = ssim(img, noisy)
    assert s < 0.95, s
    # 완전 반전(가장 다른 구조)이면 더 낮게.
    assert ssim(img, 255 - img) < s
    print("[PASS] test_ssim_drops_for_degraded")


def test_slide_fidelity_resizes_mismatched_render() -> None:
    # 원본 100x120, 렌더가 절반 해상도(marp 출력 DPI 다름) -> 리사이즈 후 비교, 죽지 않음.
    # 구조 있는 이미지(슬라이드처럼 공간 일관성)는 다운샘플에도 충실도 높게 유지.
    yy, xx = np.mgrid[0:100, 0:120]
    grad = ((yy * 2 + xx) % 256).astype(np.uint8)
    orig = np.stack([grad, grad, grad], axis=-1)  # 그레이 그라디언트 RGB
    rendered = orig[::2, ::2, :].copy()  # 절반 해상도
    score = slide_fidelity(orig, rendered)
    assert 0.0 <= score <= 1.0
    assert score > 0.8, score
    print("[PASS] test_slide_fidelity_resizes_mismatched_render")


def test_flag_low_fidelity_threshold_boundary() -> None:
    # floor=0.90: 0.90 은 통과(>= floor), 0.899 는 flag.
    scores = [0.95, 0.90, 0.8999, 0.50]
    flagged = flag_low_fidelity(scores, threshold=0.90)
    assert flagged == [2, 3], flagged
    # 기본 floor 상수는 0.90 (design Q&A 확정).
    assert DEFAULT_SSIM_FLOOR == 0.90
    print("[PASS] test_flag_low_fidelity_threshold_boundary")


def _chart_result() -> ExtractionResult:
    r = ExtractionResult(source_image="cap/s1.webp", screenshot_id="doc1_s001",
                         screenshot_index=1)
    r.regions.append(Region(region_id="r001", type="title", text="Trend"))
    r.charts.append(Chart(region_id="c001", title="Yield", legend_labels=["A"],
                          visible_values=["1"]))
    return r


def test_plan_downgrade_demotes_chart_with_available_crop() -> None:
    # 차트가 네이티브 데이터표로 렌더됨(crop_lookup 비어 있음) + crop 가용 -> 그 영역 강등.
    r = _chart_result()
    plan = plan_downgrade(r, crop_lookup={}, available_crops={"c001": "crops/c001.jpg"},
                          capture_path="cap/s1.webp", slide_index=0)
    assert plan.demote_region_ids == ["c001"]
    assert plan.whole_slide is False
    print("[PASS] test_plan_downgrade_demotes_chart_with_available_crop")


def test_plan_downgrade_whole_slide_when_no_crop_helps() -> None:
    # 차트 crop 이 없거나 이미 다 raster -> 최후 안전망 = 슬라이드 전체 래스터.
    r = _chart_result()
    plan = plan_downgrade(r, crop_lookup={"c001": "crops/c001.jpg"}, available_crops={},
                          capture_path="cap/s1.webp", slide_index=2)
    assert plan.demote_region_ids == []
    assert plan.whole_slide is True
    assert plan.capture_path == "cap/s1.webp"
    assert plan.slide_index == 2
    print("[PASS] test_plan_downgrade_whole_slide_when_no_crop_helps")


def test_plan_downgrade_text_only_slide_escalates_whole_slide() -> None:
    # 차트가 아예 없는데 충실도 낮음(텍스트/레이아웃 mismatch) -> 전체 래스터로 강등.
    r = ExtractionResult(source_image="cap/s9.webp", screenshot_id="doc1_s009")
    r.regions.append(Region(region_id="r001", type="title", text="Summary"))
    plan = plan_downgrade(r, crop_lookup={}, available_crops={},
                          capture_path="cap/s9.webp", slide_index=8)
    assert plan.whole_slide is True
    assert plan.demote_region_ids == []
    print("[PASS] test_plan_downgrade_text_only_slide_escalates_whole_slide")


def test_whole_slide_marp_is_full_bleed_background() -> None:
    md = whole_slide_marp("cap/s1.png")
    # Marp 전체배경 이미지 directive(편집성 포기, 시각 충실 보증).
    assert "![bg" in md and "cap/s1.png" in md
    print("[PASS] test_whole_slide_marp_is_full_bleed_background")


def test_apply_downgrade_plans_demotes_chart_region() -> None:
    # 부분 강등: 차트 region 의 crop 을 deck 에 주입 -> 데이터표 대신 이미지로.
    r = _chart_result()
    plan = plan_downgrade(r, crop_lookup={}, available_crops={"c001": "crops/c001.jpg"},
                          capture_path="cap/s1.webp", slide_index=0)
    deck = apply_downgrade_plans([r], crop_lookups={}, plans=[plan],
                                 available_crops={"c001": "crops/c001.jpg"})
    assert "![w:600](crops/c001.jpg)" in deck   # 차트가 래스터로
    assert "| series | value |" not in deck      # 데이터표 대체 사라짐
    print("[PASS] test_apply_downgrade_plans_demotes_chart_region")


def test_apply_downgrade_plans_whole_slide_replaces_with_bg() -> None:
    r = _chart_result()
    plan = plan_downgrade(r, crop_lookup={"c001": "x"}, available_crops={},
                          capture_path="cap/s1.png", slide_index=0)
    deck = apply_downgrade_plans([r], crop_lookups={}, plans=[plan], available_crops={})
    assert "![bg" in deck and "cap/s1.png" in deck   # 전체 슬라이드 래스터
    assert "Yield" not in deck                        # 네이티브 내용 대체됨
    print("[PASS] test_apply_downgrade_plans_whole_slide_replaces_with_bg")


def test_verify_and_downgrade_graceful_when_marp_missing() -> None:
    # marp 부재(없는 base 명령) -> rendered=False, 예외 없음(graceful).
    r = _chart_result()
    with tempfile.TemporaryDirectory() as tmp:
        deck = Path(tmp) / "deck.md"
        deck.write_text("---\nmarp: true\n---\n# x\n", encoding="utf-8")
        report = verify_and_downgrade(
            [r], deck, ["cap/s1.png"], out_dir=Path(tmp) / "out",
            marp_cmd=["/definitely/not/here/marp-xyz"])
        assert report["rendered"] is False
        assert report["corrected_deck"] is None
    print("[PASS] test_verify_and_downgrade_graceful_when_marp_missing")


def main() -> int:
    test_ssim_identical_is_one()
    test_ssim_drops_for_degraded()
    test_slide_fidelity_resizes_mismatched_render()
    test_flag_low_fidelity_threshold_boundary()
    test_plan_downgrade_demotes_chart_with_available_crop()
    test_plan_downgrade_whole_slide_when_no_crop_helps()
    test_plan_downgrade_text_only_slide_escalates_whole_slide()
    test_whole_slide_marp_is_full_bleed_background()
    test_apply_downgrade_plans_demotes_chart_region()
    test_apply_downgrade_plans_whole_slide_replaces_with_bg()
    test_verify_and_downgrade_graceful_when_marp_missing()
    print("\n[INFO] 모든 verify 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
