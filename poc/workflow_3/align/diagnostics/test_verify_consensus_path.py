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
from poc.workflow_3.align.clean_align_image import OVERSAMPLE
from poc.workflow_3.align.cond_file import CondInfo
from poc.workflow_3.align.diagnostics import verify_consensus_path as vc
from poc.workflow_3.align.matching.engine import build_template


def _checker(n=600, cell=8):
    """강한 edge 가 있는 결정적 그레이(블러 게이트 분모가 0 이 되지 않게)."""
    yy, xx = np.mgrid[0:n, 0:n]
    return (((xx // cell + yy // cell) % 2) * 255).astype(np.uint8)


def _fake_cond(n=600):
    """`load_cond` 가 실제로 돌려주는 타입(CondInfo) 그대로 만든다.

    손으로 적은 stand-in 클래스를 쓰던 자리다. 그 fake 는 consumer 가 그때 읽던 두
    필드(crosshair_xy/raw)만 갖고 있었는데, 나중에 `cond_for_image` 가 생기면서
    `cond.pixel` 을 읽자 AttributeError 로 깨졌다 - 실제 CondInfo 는 dataclass 라
    그 필드를 늘 갖고 있으므로(기본 None) production 에는 없는 고장이었다.
    producer 의 타입을 그대로 쓰면 필드가 추가돼도 따라오고, 이름이 바뀌면
    조용히 통과하는 대신 여기서 터진다.

    pixel 을 로드 크기와 같게 둬서 `cond_for_image` 는 무보정 통과다(원래 픽스처가
    가정하던 좌표계 그대로): cursor 프레임은 Pixel x10 이라 crosshair 3000 -> 이미지
    (300,300) = 600x600 중앙.
    """
    return CondInfo(
        pixel=(n, n),
        crosshair_xy=(n * OVERSAMPLE // 2, n * OVERSAMPLE // 2),
        raw={"accelerating_voltage": ["1"]},   # msr_modality -> 'sem'.
    )


def _tpl(w, h):
    """실 producer(build_template)로 center template 을 만든다 - crop 크기 = (w,h).

    `_fake_cond` 와 같은 이유로 stand-in 클래스를 쓰지 않는다. 이 자리의 예전 fake 는
    raw_image 하나만 갖고 있었고, 옆 파일(test_consensus_crops)의 쌍둥이 fake 는
    consumer 가 읽기 시작한 align_offset_xy 를 나중에 얻었다 - 두 사본이 이미 서로
    어긋나 있었다는 뜻이다. producer 를 쓰면 그 어긋남 자체가 생기지 않는다.
    """
    return build_template(
        np.zeros((h, w), np.uint8), recipe_id="c/r", version="test", key_type="sem"
    )


def _make_events(root: Path, cls, rcp, n):
    """root/cls/rcp/events/<id>/S1.jpeg 를 n개 만든다(eqp 무관 pool — 경로에 eqp 없음).

    내용은 load_gray 가 가로채므로 빈 파일. consensus_gather._events_dir_for 가
    `<root>/<class>/<recipe>/events` 로 읽는 것과 동일 레이아웃.
    """
    events = root / cls / rcp / "events"
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

    restore_factory(lambda p: _fake_cond(), cc, "load_cond")
    restore_factory(lambda p: gray.copy(), cc, "load_gray")
    restore_factory(lambda g, cond: g, cc, "clean_image")            # crosshair 제거 생략.
    restore_factory(lambda assets: {"sem": (_tpl(80, 60), (0, 0))},
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
            _make_events(root, "c", "r", n_events)
            # eqp_id 는 일부러 다르게 줘도(여러 장비) 같은 class/recipe pool 을 읽어야 한다.
            assets = SimpleNamespace(eqp_id="ANY_EQP", class_name="c", recipe_name="r")
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
