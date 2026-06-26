"""verify_consensus_path._verify_one 스모크 — populated 경로가 'ok'/'insufficient_s' 를
실제로 구별하는지(빈 경로만 도는 게 아니라) 검증한다.

consensus_crops 의 IO seam(load_cond/load_gray/clean_image)만 패치하고 **블러 게이트는
실제 CV 로 돈다**(build_consensus_template). 동일 텍스처 crop N장을 disk events 트리로
깔아, N>=min_s 면 'ok', 미달이면 'insufficient_s' 가 나오는지 본다 — 즉 검증기가 staged
S 히스토리를 실제로 빌드까지 끌고 가는지를 증명한다(OM/SEM 분리도 함께: SEM 만 적재).

실행:
  uv run python poc/workflow_3/align/diagnostics/test_verify_consensus_path.py
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poc.workflow_3.align import consensus_crops as cc
from poc.workflow_3.align.diagnostics import verify_consensus_path as vc


def _checker(n=600, cell=8):
    """강한 edge 가 있는 결정적 그레이(블러 게이트 분모가 0 이 되지 않게)."""
    yy, xx = np.mgrid[0:n, 0:n]
    return (((xx // cell + yy // cell) % 2) * 255).astype(np.uint8)


class _FakeCond:
    # crosshair_xy 는 cursor ×10 (OVERSAMPLE=10) → 이미지 (300,300) = 600x600 중앙.
    crosshair_xy = (3000, 3000)
    raw = {"accelerating_voltage": ["1"]}   # msr_modality -> 'sem'.


class _FakeTpl:
    def __init__(self, w, h):
        self.raw_image = np.zeros((h, w), dtype=np.uint8)   # crop size = (w,h).


def _make_events(root: Path, eqp, cls, rcp, n):
    """root/eqp/cls/rcp/events/<id>/S1.jpeg 를 n개 만든다(내용은 load_gray 가 가로챔)."""
    events = root / eqp / cls / rcp / "events"
    for i in range(n):
        d = events / f"20260612_09{i:02d}_0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "S1.jpeg").write_bytes(b"x")


def _patch_io(monkey):
    """consensus_crops 의 IO + 검증기의 center-tpl 빌더만 stub(블러 게이트는 실제)."""
    gray = _checker()

    def restore_factory(name, obj, attr):
        orig = getattr(obj, attr)
        monkey.append((obj, attr, orig))
        setattr(obj, attr, name)

    restore_factory(lambda p: _FakeCond(), cc, "load_cond")
    restore_factory(lambda p: gray.copy(), cc, "load_gray")
    restore_factory(lambda g, cond: g, cc, "clean_image")            # crosshair 제거 생략.
    restore_factory(lambda assets: {"sem": (_FakeTpl(80, 60), (0, 0))},
                    vc, "build_center_tpls_for_sizing")


def _unpatch(monkey):
    for obj, attr, orig in monkey:
        setattr(obj, attr, orig)


def _run_one(n_events):
    monkey = []
    _patch_io(monkey)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_events(root, "E1", "c", "r", n_events)
            assets = SimpleNamespace(eqp_id="E1", class_name="c", recipe_name="r")
            return vc._verify_one(assets, min_s=4, max_events=8, cache_root=root)
    finally:
        _unpatch(monkey)


def test_enough_s_builds_consensus():
    """N=4 (>= floor3) → SEM 'ok', OM 'no_crops'(적재 없음 → modality 분리 확인)."""
    cache_key, n_events, n_images, rows = _run_one(4)
    assert cache_key == "c/r", cache_key
    assert n_images == 4, n_images
    by_mod = {mod: reason for (mod, _ns, _nc, reason) in rows}
    assert by_mod.get("sem") == "ok", rows
    assert by_mod.get("om") == "no_crops", rows
    print("[OK] test_enough_s_builds_consensus")


def test_insufficient_s_falls_back():
    """N=2 (< floor3) → SEM 'insufficient_s'(라이브에선 rcp 강등)."""
    cache_key, n_events, n_images, rows = _run_one(2)
    by_mod = {mod: reason for (mod, _ns, _nc, reason) in rows}
    assert by_mod.get("sem") == "insufficient_s", rows
    print("[OK] test_insufficient_s_falls_back")


def test_no_cache_reports_empty():
    """staged S 0장 → rows 비고 n_images=0(빈 경로는 라이브에서 rcp)."""
    cache_key, n_events, n_images, rows = _run_one(0)
    assert n_images == 0 and rows == [], (n_images, rows)
    print("[OK] test_no_cache_reports_empty")


if __name__ == "__main__":
    test_enough_s_builds_consensus()
    test_insufficient_s_falls_back()
    test_no_cache_reports_empty()
    print("\n=== verify_consensus_path: 3/3 통과 ===")
