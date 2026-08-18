# poc/workflow_3/align/test_consensus_crops.py
"""consensus_crops adapter 스모크 — modality resolve 폴백 + crop/coreg + drop 집계.

load_cond/load_gray 를 monkeypatch 해 cond.txt 형식 결합 없이 adapter 로직만 검증한다.
"""
import sys
import numpy as np

import poc.workflow_3.align.consensus_crops as cc
from poc.workflow_3.align.cond_file import CondInfo
from poc.workflow_3.align.matching.engine import build_template


def _cond(xy, pixel=None):
    """실 producer 타입(CondInfo)으로 픽스처를 만든다 — 손으로 빚은 fake 금지."""
    return CondInfo(pixel=pixel, crosshair_xy=xy)


def _tpl(w, h):
    """실 producer(build_template)로 center template 을 만든다 - 손으로 빚은 fake 금지.

    consumer 가 읽는 것은 raw_image 크기(=crop 크기)와 align_offset_xy 뿐이지만,
    그 두 개만 가진 stand-in 을 두면 consumer 가 다른 필드를 읽기 시작하는 순간
    조용히 깨진다(_FakeCond 가 cond.pixel 에서 그렇게 깨졌다). 평평한 zeros 라도
    build_template 이 edge_map/distance_transform 까지 채운 진짜 AlignKeyTemplate 이다.
    """
    return build_template(
        np.zeros((h, w), np.uint8), recipe_id="c/r", version="test", key_type="sem"
    )


def _patch(monkeypatch_map):
    """모듈 전역을 임시 교체하고 원복 함수 반환."""
    saved = {k: getattr(cc, k) for k in monkeypatch_map}
    for k, v in monkeypatch_map.items():
        setattr(cc, k, v)
    def restore():
        for k, v in saved.items():
            setattr(cc, k, v)
    return restore


def _events(tmp, n, prefix="20260612_0900"):
    """events/<id>/S1.jpeg 더미 파일 n개 생성(내용은 monkeypatch 가 가로채므로 빈 파일)."""
    ev = tmp / "c" / "r" / "events"
    ev.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = ev / f"{prefix}{i:02d}_r_lot"
        d.mkdir()
        (d / "S1.jpeg").write_bytes(b"x")
    return tmp


def test_resolve_mod_falls_back_to_recipe_mod():
    restore = _patch({"msr_modality": lambda cond: None})
    try:
        assert cc._resolve_mod(_cond((1, 1)), "sem") == "sem"
        assert cc._resolve_mod(_cond((1, 1)), None) is None
    finally:
        restore()


def test_load_coregistered_crops_groups_and_caps(tmp_path):
    gray = (np.random.RandomState(0).rand(200, 200) * 255).astype(np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _cond((1000, 1000)),        # 중앙 crosshair(×10 → 100,100)
        "clean_image": lambda g, cond: g,                  # 정제 no-op (테스트)
        "cursor_to_image": lambda xy, oversample=10: (xy[0] / 10.0, xy[1] / 10.0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 6)
        center = {"sem": (_tpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=4)
        assert set(out) == {"sem"}
        assert len(out["sem"]) == 4               # cap=max_events
        assert out["sem"][0].shape == (32, 40)    # center tpl 크기로 crop
    finally:
        restore()


def test_missing_crosshair_dropped(tmp_path):
    gray = np.zeros((200, 200), np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _cond(None),       # crosshair 없음 → drop
        "clean_image": lambda g, cond: g,
        "cursor_to_image": lambda xy, oversample=10: (0, 0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 4)
        center = {"sem": (_tpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=8)
        assert out.get("sem", []) == []           # 전부 drop
    finally:
        restore()


def test_pixel_mismatch_recenters_crop(tmp_path):
    # cond 는 100px 기준(crosshair cursor (500,500) → 100-기준 (50,50))인데 이미지는
    # 200 으로 로드 — 정규화 후 crop 중심은 (100,100)이어야 한다. 고정 /10 만 쓰면
    # (50,50) 중심의 엉뚱한 crop 이 나오고, 모든 S 가 같은 오차라 blur 게이트도 못 잡는다.
    gray = (np.random.RandomState(1).rand(200, 200) * 255).astype(np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _cond((500, 500), pixel=(100, 100)),
        "clean_image": lambda g, cond: g,          # 정제 no-op → 내용 비교 가능.
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 1)
        center = {"sem": (_tpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=4)
        assert len(out.get("sem", [])) == 1
        expected = gray[100 - 16:100 + 16, 100 - 20:100 + 20]   # (100,100) 중심 40x32.
        assert np.array_equal(out["sem"][0], expected), "crop 이 정규화된 crosshair 중심이 아님"
    finally:
        restore()


if __name__ == "__main__":
    import tempfile, pathlib, traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1; print(f"[FAIL] {name}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
