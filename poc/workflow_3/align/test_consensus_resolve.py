# poc/workflow_3/align/test_consensus_resolve.py
"""resolve_templates 오케스트레이션 — 킬스위치/consensus 채택/insufficient 폴백/cold-sync."""
import sys
import poc.workflow_3.align.consensus_resolve as cr


class _Assets:
    eqp_id = "E1"; class_name = "c"; recipe_name = "r"


class _Tpl:
    def __init__(self, tag): self.tag = tag


def _patch(d):
    saved = {k: getattr(cr, k) for k in d}
    for k, v in d.items(): setattr(cr, k, v)
    return lambda: [setattr(cr, k, v) for k, v in saved.items()]


def test_killswitch_returns_rcp():
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"OM": _Tpl("rcp_om")},
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=False,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["OM"].tag == "rcp_om"
    finally:
        restore()


def test_consensus_adopted_when_enough():
    cons_tpl = _Tpl("cons_sem")
    class _Res:
        template = cons_tpl; reason = "ok"; modality = "sem"; n_crops = 5; edge_ratio = 1.0; lap_ratio = 1.0
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"SEM": _Tpl("rcp_sem")},
        "build_center_tpls_for_sizing": lambda assets: {"sem": (_Tpl("center"), (0, 0))},
        "load_coregistered_crops": lambda *a, **k: {"sem": [object()] * 5},
        "build_consensus_template": lambda crops, *, recipe_id, modality, policy: _Res(),
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=True,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["SEM"].tag == "cons_sem"
    finally:
        restore()


def test_insufficient_falls_back_to_rcp_without_sync_when_warm():
    class _Res:
        template = None; reason = "insufficient_s"; modality = "sem"; n_crops = 1; edge_ratio = None; lap_ratio = None
    calls = {"wait": 0}
    def _wait(*a, **k): calls["wait"] += 1; return False
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"SEM": _Tpl("rcp_sem")},
        "build_center_tpls_for_sizing": lambda assets: {"sem": (_Tpl("center"), (0, 0))},
        "load_coregistered_crops": lambda *a, **k: {"sem": [object()]},   # 1 crop < min_s, 그러나 warm(>=1)
        "build_consensus_template": lambda crops, *, recipe_id, modality, policy: _Res(),
        "wait_for_gather": _wait,
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=True,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["SEM"].tag == "rcp_sem"
        assert calls["wait"] == 0            # warm(>=1 crop) → cold-sync 미발동
    finally:
        restore()


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try: fn(); print(f"[PASS] {fn.__name__}")
        except Exception: failed += 1; print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass"); sys.exit(1 if failed else 0)
