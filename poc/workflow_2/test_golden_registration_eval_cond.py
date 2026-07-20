"""registration verifier A/B 드라이버(_RegAccum) 합성 테스트 — 실데이터 불필요.

핵심 불변:
  - _tally 는 arm/fuse 공통 집계 규약: raw(순위 효과)와 ref(순위+정밀화)를 분해하고,
    promote/regress 는 B0 대비 방향 전이만 센다.
  - fuse 의사-arm 은 fallback 아닌 arm 들의 순열 RRF 합의 — 전 arm 거부면 B0 강등.
  - hook(_process)은 harness ctx 계약({gc, cons_tpl, gray, xy, mod, recipe, path})만으로
    동작해야 한다(합성 frame 위 실제 ensemble proposer 재실행 포함).

실행: uv run pytest poc/workflow_2/test_golden_registration_eval_cond.py
"""

import io
import json
import math
from pathlib import Path

import numpy as np

from poc.workflow_2 import registration_lab as reg
import poc.workflow_2.golden_registration_eval_cond as gre
from poc.workflow_3.align.matching.engine import build_template


# --- _median -------------------------------------------------------------------------

def test_median_empty_and_odd_even():
    assert gre._median([]) is None
    assert gre._median([3.0]) == 3.0
    assert gre._median([1.0, 2.0, 10.0]) == 2.0
    assert gre._median([1.0, 2.0, 3.0, 4.0]) == 2.5


# --- _RegAccum 합성 end-to-end -------------------------------------------------------

def _make_ctx():
    """registration_lab 합성 scene 을 harness ctx 계약으로 포장."""
    frame, tpl_img, truth, _ = reg._make_scene()
    cons_tpl = build_template(tpl_img, recipe_id="synthetic/rec", version="t",
                              key_type="om", align_offset_xy=(0, 0))
    return {
        "recipe": "synthetic/rec", "mod": "om", "path": Path("S0001.jpeg"),
        "gray": frame, "xy": (float(truth[0]), float(truth[1])),
        "cons_tpl": cons_tpl, "rcp_tpl": None,
        "gc": {"cand_xys": None, "topk_rank": 1},
        "gr": None,
    }


def test_accum_process_populates_cells_and_fuse_row():
    fh = io.StringIO()
    acc = gre._RegAccum(fh, None, ("phase", "mind"), fuse=True)
    acc(_make_ctx())
    assert acc.n_hook_err == 0, "hook 예외 발생"
    assert acc.n_points == 1
    row = json.loads(fh.getvalue().strip())
    # 실제 arm + fuse 의사-arm 모두 row 에 존재.
    assert set(row["arms"]) == {"phase", "mind", "fuse"}
    assert row["bucket"] in ("rank1_ok", "rank_error", "proposer_miss")
    # cell 은 (arm, mod) 키 — fuse 포함.
    for name in ("phase", "mind", "fuse"):
        assert acc.cells[(name, "om")]["n"] == 1
    # fuse 는 실제 verifier 를 다시 돌리지 않으므로 runtime/n_cand 비집계.
    assert acc.cells[("fuse", "om")]["rt_ms"] == 0.0
    assert acc.cells[("fuse", "om")]["n_cand"] == 0
    # report_arms 로 summary 가 그대로 만들어져야 한다(err median 포함).
    for name in acc.report_arms:
        s = gre._arm_summary(acc, name)
        assert s["n"] == 1
        assert "err_ref_med_px" in s


def test_tally_promote_regress_and_raw_ref_split():
    acc = gre._RegAccum(io.StringIO(), None, ("phase",), fuse=False)
    base_pts = [(100.0, 100.0), (50.0, 50.0)]
    base_d = [40.0, 5.0]   # top-1 은 GT 에서 멀고(top=0 이면 miss), top-2 가 정답 근처.
    gt, tol = (55.0, 55.0), 10.0
    # 후보 재정렬로 top=1 을 앞세우고 refined 가 GT 를 더 파고들면 promote.
    e = acc._tally("phase", "om", "r", 1, False, (54.0, 54.0),
                   base_pts, base_d, gt, tol, b0_hit=False, bucket="rank_error")
    assert e["hit_raw"] and e["hit_ref"] and e["changed"]
    st = acc.cells[("phase", "om")]
    assert st["promote"] == 1 and st["regress"] == 0
    assert st["sub_n"] == 1 and st["sub_ref"] == 1   # rank_error 부분집합 집계.
    # 반대로 B0 hit 였는데 재정렬이 top 을 먼 후보로 바꾸면 regress.
    e2 = acc._tally("phase", "om", "r", 0, False, base_pts[0],
                    [(55.0, 55.0), (200.0, 200.0)], [2.0, 150.0], gt, tol,
                    b0_hit=True, bucket="rank1_ok")
    assert not e2["hit_ref"]
    assert acc.cells[("phase", "om")]["regress"] == 1


def test_fuse_falls_back_to_b0_when_all_arms_reject():
    """전 arm fallback 이면 fuse 도 top=0 + fallback=True (B0 강등 규약)."""
    fh = io.StringIO()
    acc = gre._RegAccum(fh, None, ("mind",), fuse=True)
    ctx = _make_ctx()
    ctx["gray"] = np.full_like(ctx["gray"], 128)   # 평탄 frame — 모든 후보 거부 유도.
    acc(ctx)
    if acc.n_points:   # proposer 가 평탄 frame 에서 후보를 냈을 때만 의미 있음.
        row = json.loads(fh.getvalue().strip())
        assert row["arms"]["fuse"]["fallback"] is True
        assert row["arms"]["fuse"]["top"] == 0


def test_default_arms_are_all_known():
    assert set(gre.ARMS) <= set(reg.REG_ARM_NAMES)
    assert gre.FUSE_ON in (True, False)
    assert gre.PROD_ARM in (True, False)


# --- prod / prod_mind 의사-arm ------------------------------------------------------

def test_prod_arm_replicates_production_selection():
    """prod = 운영 NCC rerank 재현, prod_mind = prod+mind RRF 결합 — 둘 다 집계에 존재."""
    fh = io.StringIO()
    acc = gre._RegAccum(fh, None, ("mind",), fuse=False, prod=True)
    acc(_make_ctx())
    assert acc.n_hook_err == 0, "prod arm hook 예외 발생"
    assert acc.n_points == 1
    assert acc.report_arms == ["mind", "prod", "prod_mind"]
    row = json.loads(fh.getvalue().strip())
    assert set(row["arms"]) == {"mind", "prod", "prod_mind"}
    # prod 는 절대 fallback 하지 않는다(항상 sel 점수를 낸다).
    assert row["arms"]["prod"]["fallback"] is False
    assert isinstance(row["arms"]["prod"]["sel"], float)
    assert "mind_fallback" in row["arms"]["prod_mind"]
    for name in ("prod", "prod_mind"):
        assert acc.cells[(name, "om")]["n"] == 1
        s = gre._arm_summary(acc, name)
        assert s["n"] == 1
    # digest 에도 pseudo-arm 이 실린다.
    stats = {a: gre._arm_summary(acc, a) for a in acc.report_arms}
    line = gre._digest_line(acc, stats, 1, 1)
    assert "prod r1=" in line and "prod_mind r1=" in line


def test_prod_arm_off_keeps_report_arms():
    acc = gre._RegAccum(io.StringIO(), None, ("mind",), fuse=False, prod=False)
    assert acc.report_arms == ["mind"]


# --- route (modality-aware ecc) 의사-arm --------------------------------------------

def test_route_arms_present_only_with_prod_ecc_mind():
    # ecc·mind·prod 모두 있을 때만 route3/route2 가 붙는다.
    acc = gre._RegAccum(io.StringIO(), None, ("ecc", "mind"), fuse=False,
                        prod=True, route=True)
    assert acc.report_arms[-3:] == ["route3", "route2", "route_sw"]
    # mind 없으면 route 성립 안 함.
    acc2 = gre._RegAccum(io.StringIO(), None, ("ecc",), fuse=False, prod=True, route=True)
    assert "route3" not in acc2.report_arms
    # prod 없으면 route 성립 안 함.
    acc3 = gre._RegAccum(io.StringIO(), None, ("ecc", "mind"), fuse=False,
                         prod=False, route=True)
    assert acc3.report_arms == ["ecc", "mind"]


def test_route_tallies_both_and_labels_om_as_prod_mind():
    """route3/route2 가 집계되고, OM 점에서는 둘 다 prod_mind 와 동일 top 을 고른다."""
    fh = io.StringIO()
    acc = gre._RegAccum(fh, None, ("ecc", "mind"), fuse=False, prod=True, route=True)
    acc(_make_ctx())                      # _make_ctx 는 mod="om".
    assert acc.n_hook_err == 0
    row = json.loads(fh.getvalue().strip())
    assert {"route3", "route2", "route_sw", "prod_mind"} <= set(row["arms"])
    # OM 이므로 route* 셋 다 top 이 prod_mind 와 같아야 한다(OM 은 모두 sel⊕mind).
    for name in ("route3", "route2", "route_sw"):
        assert row["arms"][name]["top"] == row["arms"]["prod_mind"]["top"]
        assert acc.cells[(name, "om")]["n"] == 1
        assert gre._arm_summary(acc, name)["n"] == 1


# --- overlay 표기 -------------------------------------------------------------------

# --- 커버리지 집계 (어떤 방법으로도 GT 못 잡은 점) ---------------------------------

def test_coverage_invariant_and_bucket_split():
    """oracle_hit + all_miss == n, 그리고 all_miss 는 bucket 별로 분해된다."""
    acc = gre._RegAccum(io.StringIO(), None, ("phase",), fuse=False)
    acc(_make_ctx())
    assert acc.coverage["om"]["n"] == 1
    s = gre._coverage_summary(acc.coverage["om"])
    assert s["oracle_hit"] + s["all_miss"] == s["n"]
    # all_miss 는 세 bucket miss 의 합이며 proposer_miss/rank_error 로 분해된다.
    assert s["all_miss"] == (s["miss_proposer_miss"] + s["miss_rank_error"]
                             + s["miss_rank1_ok"])
    # rank1_ok 버킷은 정의상 b0_hit 라 커버리지 miss 가 0 이어야 한다.
    assert s["miss_rank1_ok"] == 0


def test_coverage_all_miss_on_flat_proposer_miss():
    """평탄 frame: proposer 가 후보를 내도 GT 도달 불가 -> all_miss(대개 proposer_miss)."""
    acc = gre._RegAccum(io.StringIO(), None, ("mind",), fuse=False, prod=True)
    ctx = _make_ctx()
    ctx["gray"] = np.full_like(ctx["gray"], 128)
    ctx["xy"] = (5.0, 5.0)      # GT 를 구석에 둬서 어떤 후보도 tol 안에 못 오게.
    acc(ctx)
    if acc.n_points:            # proposer 가 평탄 frame 에서 후보를 냈을 때만 유의미.
        s = gre._coverage_summary(acc.coverage[ctx["mod"]])
        assert s["oracle_hit"] == 0 and s["all_miss"] == 1


def test_digest_has_oracle_and_allmiss():
    acc = gre._RegAccum(io.StringIO(), None, ("mind",), fuse=True, prod=True)
    acc(_make_ctx())
    stats = {a: gre._arm_summary(acc, a) for a in acc.report_arms}
    line = gre._digest_line(acc, stats, 1, 1)
    assert "oracle=" in line and "allmiss[pm=" in line and "/re=" in line


def test_overlay_groups_same_pick_and_adds_banner(tmp_path):
    """같은 점을 고른 arm 은 그룹(동심 링 + 라벨)으로, 상단엔 범례 배너가 붙는다."""
    import cv2
    acc = gre._RegAccum(io.StringIO(), tmp_path, ("ecc", "mind"), fuse=True, prod=True)
    gray = np.full((200, 300), 40, dtype=np.uint8)
    arm_tops = {"ecc": (80.0, 100.0), "mind": (220.0, 100.0), "fuse": (220.0, 100.0),
                "prod": (80.0, 100.0), "prod_mind": (220.0, 100.0)}
    acc._save_overlay(gray, (150.0, 100.0), (80.0, 100.0), arm_tops,
                      "cls/rec", "S0001.jpeg", bucket="rank_error")
    files = list(tmp_path.glob("*.jpg"))
    assert len(files) == 1 and acc.n_overlay == 1
    img = cv2.imread(str(files[0]))
    # 배너(26px)만큼 세로가 늘어난다.
    assert img.shape[0] == 200 + 26 and img.shape[1] == 300
    # 그룹 라벨/배너가 실제 그려졌는지: 순수 회색 프레임보다 유채색 픽셀이 존재.
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    assert (np.abs(b - g) + np.abs(g - r)).max() > 100, "컬러 마킹이 그려지지 않음"
