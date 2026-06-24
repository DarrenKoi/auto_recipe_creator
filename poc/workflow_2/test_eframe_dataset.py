"""E-frame 데이터셋 전용 루트 선택 + 데이터셋 health 사전점검 테스트.

Phase 2(E-frame confirmation) 보정 실행은 S-only golden 과 분리된 전용 루트
(ALIGN_EFRAME_ROOT)를 walk 한다. 이 파일은 그 루트 선택 로직과, recipe 별
rcp/S/E/cond 존재를 집계해 'E_CONFIRMED 도달 불가' recipe 를 사전에 경고하는
health 리포트를 Mac 합성 디렉터리로 검증한다(엔진/실데이터 불필요).
"""
import os

import poc.workflow_2.golden_reregister_report_cond as r


# --- Slice 1: 리포트 walk 루트 선택 (eframe 우선, golden 폴백) ---------------

def test_resolve_root_prefers_eframe_when_set(monkeypatch):
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", "/tmp/golden_x")
    monkeypatch.setenv("ALIGN_EFRAME_ROOT", "/tmp/eframe_y")
    root, src = r._resolve_report_root()
    assert src == "eframe"
    assert str(root).endswith("eframe_y")


def test_resolve_root_falls_back_to_golden(monkeypatch):
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", "/tmp/golden_x")
    monkeypatch.delenv("ALIGN_EFRAME_ROOT", raising=False)
    root, src = r._resolve_report_root()
    assert src == "golden"
    assert str(root).endswith("golden_x")


def test_resolve_root_blank_eframe_ignored(monkeypatch):
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", "/tmp/golden_x")
    monkeypatch.setenv("ALIGN_EFRAME_ROOT", "   ")   # 공백뿐이면 미설정 취급.
    root, src = r._resolve_report_root()
    assert src == "golden"


# --- 합성 데이터셋 빌더 (엔진/실이미지 불필요: 경로 존재만으로 assets 해석) ----

def _make_recipe(root, cls, recipe, *, rcp=True, n_s=1, n_e=1, cond=True):
    """root/EQP1/<cls>/<recipe>/{from_rcp,from_msr} 합성. 더미 1바이트 파일 + cond sidecar."""
    from poc.workflow_3.align.assets import (
        FROM_RCP_DIRNAME, FROM_MSR_DIRNAME, RCP_OM_STEM,
    )
    from poc.workflow_3.align.cond_file import cond_path_for

    rdir = root / "EQP1" / cls / recipe
    frcp = rdir / FROM_RCP_DIRNAME
    fmsr = rdir / FROM_MSR_DIRNAME
    frcp.mkdir(parents=True, exist_ok=True)
    fmsr.mkdir(parents=True, exist_ok=True)
    if rcp:
        (frcp / f"{RCP_OM_STEM}.jpg").write_bytes(b"x")
    for i in range(n_s):
        p = fmsr / f"S{i:04d}.jpg"
        p.write_bytes(b"x")
        if cond:
            cp = cond_path_for(p)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text("Magnification=104\n", encoding="utf-8")
    for i in range(n_e):
        p = fmsr / f"E{1000 + i:04d}.jpg"
        p.write_bytes(b"x")
        if cond:
            cp = cond_path_for(p)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text("Magnification=104\n", encoding="utf-8")
    return rdir


def _row_by_recipe(rows, recipe):
    return next(d for d in rows if d["recipe"].endswith(recipe))


# --- Slice 2: _dataset_health 집계 + missing 플래그 --------------------------

def test_dataset_health_counts_complete_recipe(tmp_path):
    _make_recipe(tmp_path, "C1", "good", rcp=True, n_s=2, n_e=3, cond=True)
    assets = r._walk_recipes(tmp_path)
    rows = r._dataset_health(assets)
    g = _row_by_recipe(rows, "good")
    assert g["rcp"] == 1 and g["n_s"] == 2 and g["n_e"] == 3
    assert g["n_cond"] == 5          # S(2)+E(3) 모두 cond sidecar 존재.
    assert g["missing"] == []        # rcp+S+E 모두 있으니 confirm-capable.


def test_dataset_health_flags_missing_e(tmp_path):
    _make_recipe(tmp_path, "C1", "no_e", rcp=True, n_s=2, n_e=0, cond=True)
    rows = r._dataset_health(r._walk_recipes(tmp_path))
    g = _row_by_recipe(rows, "no_e")
    assert g["n_e"] == 0
    assert g["missing"] == ["E"]     # E 없음 -> E_CONFIRMED 도달 불가.


def test_dataset_health_flags_missing_cond(tmp_path):
    _make_recipe(tmp_path, "C1", "no_cond", rcp=True, n_s=1, n_e=1, cond=False)
    g = _row_by_recipe(r._dataset_health(r._walk_recipes(tmp_path)), "no_cond")
    assert g["n_cond"] == 0          # sidecar 없음 -> 런타임 modality 라우팅서 skip 될 프레임.
    assert g["missing"] == []        # cond 는 hard-miss 아님(rcp/S/E 는 있음).


# --- Slice 3: _format_dataset_health ASCII 리포트 ----------------------------

def test_format_health_summary_and_incomplete(tmp_path):
    rows = [
        {"recipe": "C1/good", "rcp": 1, "n_s": 2, "n_e": 3, "n_cond": 5, "missing": []},
        {"recipe": "C1/no_e", "rcp": 1, "n_s": 2, "n_e": 0, "n_cond": 2, "missing": ["E"]},
    ]
    out = r._format_dataset_health(rows)
    assert out == out.encode("ascii", "replace").decode("ascii")   # ASCII only.
    assert "2 recipes" in out
    assert "confirm-capable 1" in out
    assert "E-bearing 1" in out
    assert "C1/no_e: missing E" in out


def test_format_health_all_complete_has_no_incomplete_list(tmp_path):
    rows = [{"recipe": "C1/good", "rcp": 1, "n_s": 1, "n_e": 1, "n_cond": 2, "missing": []}]
    out = r._format_dataset_health(rows)
    assert "all 1 recipes have rcp+S+E" in out
    assert "missing" not in out


def test_format_health_flags_cond_gap(tmp_path):
    rows = [{"recipe": "C1/g", "rcp": 1, "n_s": 2, "n_e": 2, "n_cond": 1, "missing": []}]
    out = r._format_dataset_health(rows)
    assert "cond sidecar gap" in out   # 4 frames vs 1 cond -> gap note.


# --- Slice 4: run() 이 eframe 루트를 walk 하고 no_data 경로를 보존 ----------

def test_run_walks_eframe_root_on_empty(monkeypatch, tmp_path, capsys):
    golden = tmp_path / "golden"; eframe = tmp_path / "eframe"
    golden.mkdir(); eframe.mkdir()                      # 둘 다 빈 루트.
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(golden))
    monkeypatch.setenv("ALIGN_EFRAME_ROOT", str(eframe))
    monkeypatch.setenv("REREGISTER_E_CONFIRM", "1")
    assert r.run() == "[WARNING] no_data"               # eframe(빈) walk -> no_data.
    out = capsys.readouterr().out
    assert "source=eframe" in out                       # golden 아닌 eframe 을 골랐다.
    assert str(eframe) in out


# --- Slice 5: golden_eval_config 의 EFRAME_ROOT -> ALIGN_EFRAME_ROOT 브리지 ---

def test_seed_env_bridges_eframe_root_when_set(monkeypatch):
    import poc.workflow_2.golden_eval_config_loader as cfg
    monkeypatch.delenv("ALIGN_EFRAME_ROOT", raising=False)
    monkeypatch.setattr(cfg, "EFRAME_ROOT", "/tmp/eframe_z", raising=False)
    cfg.seed_env()
    assert os.environ["ALIGN_EFRAME_ROOT"] == "/tmp/eframe_z"


def test_seed_env_skips_eframe_root_when_none(monkeypatch):
    import poc.workflow_2.golden_eval_config_loader as cfg
    monkeypatch.delenv("ALIGN_EFRAME_ROOT", raising=False)
    monkeypatch.setattr(cfg, "EFRAME_ROOT", None, raising=False)
    cfg.seed_env()
    assert "ALIGN_EFRAME_ROOT" not in os.environ   # None 이면 미브리지(env 미설정).
