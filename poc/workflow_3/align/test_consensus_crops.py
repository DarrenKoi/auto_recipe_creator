# poc/workflow_3/align/test_consensus_crops.py
"""consensus_crops adapter 스모크 — modality resolve 폴백 + crop/coreg + drop 집계.

load_cond/load_gray 를 monkeypatch 해 cond.txt 형식 결합 없이 adapter 로직만 검증한다.
"""
import sys
import numpy as np

import poc.workflow_3.align.consensus_crops as cc


class _FakeCond:
    def __init__(self, xy):
        self.crosshair_xy = xy  # cursor frame ×10 좌표(없으면 None)


class _FakeTpl:
    def __init__(self, w, h):
        self.raw_image = np.zeros((h, w), np.uint8)
        self.align_offset_xy = (0, 0)


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
    ev = tmp / "E1" / "c" / "r" / "events"
    ev.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = ev / f"{prefix}{i:02d}_r_lot"
        d.mkdir()
        (d / "S1.jpeg").write_bytes(b"x")
    return tmp


def test_resolve_mod_falls_back_to_recipe_mod():
    restore = _patch({"msr_modality": lambda cond: None})
    try:
        assert cc._resolve_mod(_FakeCond((1, 1)), "sem") == "sem"
        assert cc._resolve_mod(_FakeCond((1, 1)), None) is None
    finally:
        restore()


def test_load_coregistered_crops_groups_and_caps(tmp_path):
    gray = (np.random.RandomState(0).rand(200, 200) * 255).astype(np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _FakeCond((1000, 1000)),    # 중앙 crosshair(×10 → 100,100)
        "clean_image": lambda g, cond: g,                  # 정제 no-op (테스트)
        "cursor_to_image": lambda xy, oversample=10: (xy[0] / 10.0, xy[1] / 10.0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 6)
        center = {"sem": (_FakeTpl(40, 32), (0, 0))}
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
        "load_cond": lambda p: _FakeCond(None),   # crosshair 없음 → drop
        "clean_image": lambda g, cond: g,
        "cursor_to_image": lambda xy, oversample=10: (0, 0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 4)
        center = {"sem": (_FakeTpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=8)
        assert out.get("sem", []) == []           # 전부 drop
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
