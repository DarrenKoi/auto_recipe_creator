# poc/workflow_2/golden_combined_eval_cond.py
"""golden_combined_eval_cond — production routed pipeline(consensus 우선 · rcp 폴백) 통합 평가.

왜 통합인가
-----------
production(`correct_align_fail_auto`, consensus_resolve.resolve_templates)은 recipe·modality 별로
consensus 가 신뢰 가능(같은 modality S >= min_s + blur 가드)하면 consensus, 아니면 rcp 로 *라우팅*한다.
그런데 기존 오프라인 벤치는 둘을 따로 본다:

  - golden_localization_eval_cond.py : rcp 단일 키 localization 만 (production 의 *폴백 arm*)
  - golden_consensus_eval_cond.py    : consensus 가 rcp 보다 나은가 A/B (comparative)

둘 다 production 이 실제로 내보내는 숫자가 아니다. 이 드라이버는 두 검증된 드라이버를 *그대로*
재사용(LOO consensus 수학은 align_similarity._consensus_template_ab 그대로 — bit-drift 0)해서
production 과 같은 라우팅을 적용한 **routed 정확도**를 낸다.

세 가지 리포트 축
-----------------
  (A) consensus scaling : consensus-eligible recipe 를 n_S(LOO 에 쓰인 S 장수)별로 층화 →
                          "consensus 많을수록 rank1/topk 가 오르나" 곡선. (S-희박이면 high bin 이 비어
                          신뢰 못 함 — n_recipes 를 같이 본다.)
  (B) rcp-only arm      : consensus 불가(S<min_s) recipe 의 rcp box localization. consensus 가 못 돕는
                          영역이라 여기가 matching-engine 레버(edge_ncc 등 ALIGN_ENSEMBLE_LAB_MODE)의 testbed.
  (C) routed overall    : eligible → consensus, rcp-only → rcp 로 라우팅한 frame-weighted 종합.

지표 정의(양 arm 동일): in_topk = 진실이 후보 pool 에 듦, rank1 = 진실이 top 후보(topk_rank==1).
양 arm 모두 in_topk/topk_rank 를 노출하므로 routed 종합이 내부적으로 일관된다.

주의: consensus arm 의 rcp counterfactual 은 center 템플릿(_consensus_template_ab 내부), rcp-only arm 은
box 템플릿(production 포팅 경로)이다. routed *pick* 은 일관되나(eligible=consensus, rcp-only=rcp box),
arm 간 rcp 정의가 달라 lift 비교는 arm 별로만 해석한다.

실행(인자 없음, No-CLI 규약):

    uv run python poc/workflow_2/golden_combined_eval_cond.py
    ALIGN_ENSEMBLE_LAB_MODE=edge_ncc uv run python poc/workflow_2/golden_combined_eval_cond.py  # (B) 레버 평가

env: ALIGN_GOLDEN_ROOT(데이터 루트), CONSENSUS_MIN_S(consensus 최소 S, 바닥 3), ALIGN_ENSEMBLE_LAB_MODE(rcp-only arm 채널).
Mac dev(golden 없음)에선 [WARNING] 후 no_data 로 깨끗이 빠진다.
"""

import os
import sys
import json
from collections import OrderedDict
from pathlib import Path

if __package__ in (None, ""):   # 직접 실행 시 repo 루트를 path 에 (다른 드라이버와 동일 관용).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_2 import golden_localization_eval_cond as glec
from poc.workflow_2 import golden_consensus_eval_cond as gce
from poc.workflow_3.util.time_utils import make_timestamp_tag


OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_combined_eval_cond"


# ======================================================================
# CONFIG — 여기 3줄만 고치고 `uv run python poc/workflow_2/golden_combined_eval_cond.py`.
# env·CLI 인자 불필요(No-CLI 규약 + env 입력 회피). env 가 설정돼 있으면 그게 우선(하위호환).
# ======================================================================
GOLDEN_ROOT = None     # 골든 데이터 루트(예: r"C:\data\align_images"). None = 기본 경로/ALIGN_GOLDEN_ROOT.
LAB_MODE = ""          # rcp-only arm 매처 채널. "" = production ensemble, "edge_ncc" = C4 레버 평가.
MIN_S = None           # consensus 최소 S(바닥 3). None = consensus 드라이버 기본값.
# ======================================================================


def _apply_config():
    """CONFIG 상수를 적용하고 골든 루트 Path 를 돌려준다. env 가 있으면 env 우선(하위호환).

    상수→내부 env/모듈attr 브리지는 구현 세부일 뿐, 사용자는 env 를 만질 필요가 없다.
    - LAB_MODE: gle 가 호출시점에 ALIGN_ENSEMBLE_LAB_MODE 를 읽으므로 os.environ 에 주입.
    - MIN_S: gce 가 import 시점에 CONSENSUS_MIN_S 를 읽으므로 모듈 attr 를 덮어쓴다(floor 3).
    """
    if LAB_MODE and not os.getenv("ALIGN_ENSEMBLE_LAB_MODE"):
        os.environ["ALIGN_ENSEMBLE_LAB_MODE"] = LAB_MODE
    if MIN_S is not None and not os.getenv("CONSENSUS_MIN_S"):
        gce.CONSENSUS_MIN_S = max(3, int(MIN_S))
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    if root_env:
        return Path(root_env)
    if GOLDEN_ROOT:
        return Path(GOLDEN_ROOT)
    return glec.GOLDEN_ROOT

# (A) consensus scaling 층화 bin — n_S_loo(= dominant modality LOO 점 수) 기준. floor 3.
S_COUNT_BINS = [
    (3, 3, "S=3"),
    (4, 4, "S=4"),
    (5, 6, "S=5-6"),
    (7, 9, "S=7-9"),
    (10, None, "S>=10"),
]

# rcp-only arm 에서 production-relevant 셀 우선순위(box=cond-box crop 포팅 경로 우선).
_CELL_PRIORITY = ("box__inpaint", "center__inpaint", "box__raw", "center__raw")


def _bin_by_s_count(per_recipe, edges=S_COUNT_BINS):
    """consensus per-recipe row 들을 n_S_loo 로 층화 → bin 별 frame-weighted rate.

    per_recipe row 는 rate(반올림)만 노출하므로 n_S_loo 로 가중평균한다(report metric).
    같은 recipe 의 LOO 점들은 n_S_loo 가 동일하므로 frame 가중 = recipe 내 점 수 가중.
    """
    bins = OrderedDict(
        (lbl, {"label": lbl, "lo": lo, "hi": hi, "n_recipes": 0, "n_frames": 0,
               "cons_rank1_sum": 0.0, "cons_topk_sum": 0.0,
               "rcp_rank1_sum": 0.0, "rcp_topk_sum": 0.0, "lift_sum": 0.0})
        for lo, hi, lbl in edges
    )
    for r in per_recipe:
        ns = int(r["n_S_loo"])
        for lo, hi, lbl in edges:
            if ns >= lo and (hi is None or ns <= hi):
                b = bins[lbl]
                b["n_recipes"] += 1
                b["n_frames"] += ns
                b["cons_rank1_sum"] += r["cons_rank1_rate"] * ns
                b["cons_topk_sum"] += r["cons_in_topk_rate"] * ns
                b["rcp_rank1_sum"] += r["rcp_rank1_rate"] * ns
                b["rcp_topk_sum"] += r["rcp_in_topk_rate"] * ns
                b["lift_sum"] += r["lift"] * ns
                break
    out = []
    for b in bins.values():
        nf = b["n_frames"]
        out.append({
            "label": b["label"], "n_recipes": b["n_recipes"], "n_frames": nf,
            "cons_rank1_rate": round(b["cons_rank1_sum"] / nf, 3) if nf else None,
            "cons_in_topk_rate": round(b["cons_topk_sum"] / nf, 3) if nf else None,
            "rcp_rank1_rate": round(b["rcp_rank1_sum"] / nf, 3) if nf else None,
            "rcp_in_topk_rate": round(b["rcp_topk_sum"] / nf, 3) if nf else None,
            "lift": round(b["lift_sum"] / nf, 3) if nf else None,
        })
    return out


def _pick_cell(cells):
    """row['cells'] 에서 production-relevant 셀 1개(box__inpaint 우선) 선택. 없으면 None."""
    if not cells:
        return None
    for k in _CELL_PRIORITY:
        c = cells.get(k)
        if c is not None:
            return c
    # 우선순위 밖이라도 아무 셀이나(방어).
    for c in cells.values():
        if c is not None:
            return c
    return None


def _arm_rates(cells_list):
    """선택된 셀 list → {n, rank1_rate(topk_rank==1), in_topk_rate}. 양 arm 동일 정의."""
    n = len(cells_list)
    if n == 0:
        return {"n": 0, "rank1_rate": None, "in_topk_rate": None}
    rank1 = sum(1 for c in cells_list if c.get("topk_rank") == 1)
    topk = sum(1 for c in cells_list if c.get("in_topk"))
    return {"n": n, "rank1_rate": round(rank1 / n, 3), "in_topk_rate": round(topk / n, 3)}


def _routed_overall(cons_frames, cons_rank1_rate, cons_topk_rate, rcp_stats):
    """eligible(consensus) + rcp-only(rcp box) frame-weighted 종합. routed pick 기준."""
    nc = int(cons_frames or 0)
    nr = int(rcp_stats.get("n", 0))
    tot = nc + nr
    if tot == 0:
        return {"n_frames": 0, "consensus_frames": nc, "rcp_only_frames": nr,
                "rank1_rate": None, "in_topk_rate": None}

    def _blend(cr, rr):
        cr = cr or 0.0
        rr = rr or 0.0
        return round((cr * nc + rr * nr) / tot, 3)

    return {
        "n_frames": tot, "consensus_frames": nc, "rcp_only_frames": nr,
        "rank1_rate": _blend(cons_rank1_rate, rcp_stats.get("rank1_rate")),
        "in_topk_rate": _blend(cons_topk_rate, rcp_stats.get("in_topk_rate")),
    }


def _score_rcp_only(rec_assets, center_tpls, box_tpls):
    """consensus 불가 recipe 의 모든 S 프레임을 rcp box localization 으로 채점 → 선택 셀 list."""
    from poc.workflow_3.align.assets import iter_msr_images
    cells = []
    msr_imgs = iter_msr_images(rec_assets)
    if gle.MAX_S_PER_RECIPE is not None:
        msr_imgs = msr_imgs[:gle.MAX_S_PER_RECIPE]
    for p in msr_imgs:
        row = glec._process_msr_cond(p, center_tpls, box_tpls, recipe_id=rec_assets.recipe_id)
        if row is None:
            continue
        cell = _pick_cell(row.get("cells"))
        if cell is not None:
            cells.append(cell)
    return cells


def run() -> str:
    """routed pipeline 통합 평가(인자 없음). 반환: success | no_data | no_eligible."""
    root = _apply_config()          # CONFIG 상수(또는 env) 적용 + 골든 루트 결정.
    glec._apply_matcher_default()   # rcp-only arm: ensemble 기본(ALIGN_USE_ENSEMBLE=0 으로 끌 수 있음).
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[WARNING] golden 데이터를 찾지 못했습니다: {root} "
              f"(env ALIGN_GOLDEN_ROOT 로 경로 지정). combined 판은 self-test 없음.")
        return "no_data"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher_mode = gle._matcher_for_eval().__name__
    print(f"[INFO] (combined routed) recipe {len(recipes)}개 → {out_dir}  [rcp matcher={matcher_mode}]")
    print(f"[INFO] consensus min_s={gce.CONSENSUS_MIN_S} (floor 3) · clean_frame={gce.CLEAN_FRAME}")
    if os.getenv("ALIGN_ENSEMBLE_LAB_MODE"):
        print(f"[INFO] rcp-only arm lab ensemble: mode={os.getenv('ALIGN_ENSEMBLE_LAB_MODE')} "
              f"channels={','.join(gle._lab_channels_for_eval())}")

    # 1) recipe 별 template + consensus 입력(entry) 빌드. rcp-only 채점용으로 template 보관.
    by_recipe = {}                    # rec_key -> entry (consensus A/B 입력; n_s>0 만)
    tpls_by_key = {}                  # rec_key -> (assets, center_tpls, box_tpls)  rcp-only 채점용
    for assets in recipes:
        if assets is None:
            continue
        try:
            center_tpls, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        if not any(v is not None for v in center_tpls.values()):
            continue
        rec_key = gce._recipe_key(assets)
        entry = gce._build_cond_by_recipe(assets, center_tpls)
        tpls_by_key[rec_key] = (assets, center_tpls, box_tpls)
        if len(entry["s_frames"]):
            by_recipe[rec_key] = entry

    # 2) consensus arm — 검증된 LOO A/B 그대로(min_s 미달 recipe 는 내부에서 skip).
    res = gce._consensus_template_ab(
        by_recipe, min_s=gce.CONSENSUS_MIN_S, out_dir=None,
        frame_loader=gce._cleaned_frame_loader if gce.CLEAN_FRAME else None)
    per_recipe = res["per_recipe"] if res else []
    eligible_keys = {r["recipe"] for r in per_recipe}
    cons_frames = res["n_S_loo"] if res else 0
    cons_rank1 = res["overall_cons_rank1_rate"] if res else None
    cons_topk = res["overall_cons_in_topk_rate"] if res else None

    # 3) rcp-only arm — eligible 이 아닌 모든 recipe 의 rcp box localization.
    rcp_only_cells = []
    n_rcp_only_recipes = 0
    for rec_key, (assets, center_tpls, box_tpls) in tpls_by_key.items():
        if rec_key in eligible_keys:
            continue
        cells = _score_rcp_only(assets, center_tpls, box_tpls)
        if cells:
            n_rcp_only_recipes += 1
            rcp_only_cells.extend(cells)
    rcp_only = _arm_rates(rcp_only_cells)

    # 4) 리포트 조립.
    scaling = _bin_by_s_count(per_recipe)
    routed = _routed_overall(cons_frames, cons_rank1, cons_topk, rcp_only)
    summary = {
        "matcher_rcp_only": matcher_mode,
        "lab_mode": os.getenv("ALIGN_ENSEMBLE_LAB_MODE", ""),
        "lab_channels": (list(gle._lab_channels_for_eval())
                         if os.getenv("ALIGN_ENSEMBLE_LAB_MODE") else []),
        "consensus_min_s": gce.CONSENSUS_MIN_S,
        "n_recipes_total": len(tpls_by_key),
        "n_recipes_consensus_eligible": len(eligible_keys),
        "n_recipes_rcp_only": n_rcp_only_recipes,
        "consensus_arm": {
            "n_frames": cons_frames,
            "rank1_rate": cons_rank1,
            "in_topk_rate": cons_topk,
            "rcp_counterfactual_rank1_rate": res["overall_rcp_rank1_rate"] if res else None,
            "rcp_counterfactual_in_topk_rate": res["overall_rcp_in_topk_rate"] if res else None,
            "overall_lift": res["overall_lift"] if res else None,
            "rank1_lift": res["rank1_lift"] if res else None,
        },
        "consensus_scaling_by_s_count": scaling,   # (A)
        "rcp_only_arm": rcp_only,                   # (B)
        "routed_overall": routed,                   # (C)
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_report(summary)
    digest = _digest_line(summary)
    (out_dir / "digest.txt").write_text(digest + "\n", encoding="utf-8")
    # 사용자가 콘솔 전체 대신 이 한 줄만 복사해 전달하도록 맨 끝에 크게 찍는다.
    print("\n" + "=" * 70)
    print("[DIGEST] " + digest)
    print("=" * 70)
    print(f"[INFO] (이 한 줄만 복사해서 주면 됨. 파일: {out_dir / 'digest.txt'})")
    return "success" if (cons_frames or rcp_only["n"]) else "no_eligible"


def _digest_line(summary):
    """콘솔 마지막에 찍는 한 줄 다이제스트 — 사용자가 이 줄만 복사해 전달.

    포맷(고정): lab/minS | routed r1/topk | cons r1/topk lift | rcp_only r1/topk | scaling[bin:cons_r1 ...].
    숫자만 보면 (A) scaling 추세 / (B) rcp-only 레버 효과 / (C) routed 정확도를 다 읽을 수 있다.
    """
    ca = summary["consensus_arm"]
    ro = summary["rcp_only_arm"]
    r = summary["routed_overall"]
    scaling = " ".join(
        f"{b['label']}:{b['cons_rank1_rate']}"
        for b in summary["consensus_scaling_by_s_count"] if b["n_frames"]
    ) or "-"
    lab = summary["lab_mode"] or "off"
    return (
        f"lab={lab} minS={summary['consensus_min_s']} | "
        f"routed r1/topk={r['rank1_rate']}/{r['in_topk_rate']} (n={r['n_frames']}) | "
        f"cons r1/topk={ca['rank1_rate']}/{ca['in_topk_rate']} lift={ca['overall_lift']} "
        f"(n={ca['n_frames']},rec={summary['n_recipes_consensus_eligible']}) | "
        f"rcp_only r1/topk={ro['rank1_rate']}/{ro['in_topk_rate']} "
        f"(n={ro['n']},rec={summary['n_recipes_rcp_only']}) | "
        f"scaling[{scaling}]"
    )


def _print_report(summary):
    """stdout 리포트 — 3축(scaling / rcp-only / routed) 한눈."""
    ca = summary["consensus_arm"]
    print(f"\n[INFO] === recipe 분할 === total={summary['n_recipes_total']} "
          f"consensus-eligible={summary['n_recipes_consensus_eligible']} "
          f"rcp-only={summary['n_recipes_rcp_only']}")

    print("\n[INFO] === (A) consensus scaling (n_S 별 — '많을수록 좋은가') ===")
    print(f"    {'bin':<8} {'recipes':>7} {'frames':>7} "
          f"{'cons r1/topk':>16} {'rcp r1/topk':>16} {'lift':>6}")
    for b in summary["consensus_scaling_by_s_count"]:
        if not b["n_frames"]:
            continue
        print(f"    {b['label']:<8} {b['n_recipes']:>7} {b['n_frames']:>7} "
              f"{str(b['cons_rank1_rate'])+'/'+str(b['cons_in_topk_rate']):>16} "
              f"{str(b['rcp_rank1_rate'])+'/'+str(b['rcp_in_topk_rate']):>16} {str(b['lift']):>6}")
    print("    * recipes 가 적은 high bin 은 신뢰 못 함(S-희박). cons r1/topk 가 bin 따라 오르면 = 많을수록 좋음.")

    print("\n[INFO] === (B) rcp-only arm (consensus 불가 — edge_ncc 등 레버 testbed) ===")
    ro = summary["rcp_only_arm"]
    print(f"    n_frames={ro['n']}  rank1={ro['rank1_rate']}  in_topk={ro['in_topk_rate']}"
          f"   [matcher={summary['matcher_rcp_only']}"
          + (f", lab={summary['lab_mode']}" if summary['lab_mode'] else "") + "]")
    print("    * ALIGN_ENSEMBLE_LAB_MODE=edge_ncc 로 재실행해 이 숫자가 오르는지 = 레버 효과.")

    print("\n[INFO] === (C) routed overall (production 라우팅: eligible→consensus, rest→rcp) ===")
    r = summary["routed_overall"]
    print(f"    n_frames={r['n_frames']} (consensus {r['consensus_frames']} + rcp-only {r['rcp_only_frames']})")
    print(f"    routed rank1={r['rank1_rate']}  in_topk={r['in_topk_rate']}")
    print(f"    consensus arm: rank1={ca['rank1_rate']} topk={ca['in_topk_rate']} "
          f"(vs rcp counterfactual r1={ca['rcp_counterfactual_rank1_rate']} "
          f"topk={ca['rcp_counterfactual_in_topk_rate']}, lift={ca['overall_lift']})")
    print("    * routed pick 은 일관(eligible=consensus, rest=rcp box). arm 간 rcp 정의(center vs box)는 lift 해석시 주의.")


if __name__ == "__main__":
    _status = run()
    raise SystemExit(0 if _status in ("success", "no_data") else 1)
