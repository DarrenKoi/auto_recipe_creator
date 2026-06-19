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

# golden_eval_config.py 상수 → env. *gce/glec import 전에* (gce 는 import 시점에 CONSENSUS_MIN_S 읽음).
from poc.workflow_2.golden_eval_config_loader import seed_env
seed_env()

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_2 import golden_localization_eval_cond as glec
from poc.workflow_2 import golden_consensus_eval_cond as gce
from poc.workflow_2.ensemble_lab import template_periodicity, PERIODICITY_TAU
from poc.workflow_3.util.time_utils import make_timestamp_tag


OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_combined_eval_cond"

# split 판정 임계 — golden_eval_config(실편집/gitignored)에서 읽고, 없으면 기본값(example 과 동일).
try:
    from poc.workflow_2.golden_eval_config import (
        SPLIT_MIN_FRAMES, SPLIT_MIN_RECIPES, SPLIT_RANK1_GAP, SPLIT_RANK1_FLOOR, SPLIT_DOMINANCE,
    )
except ImportError:   # 파일 부재/구버전 — 기본값 폴백.
    SPLIT_MIN_FRAMES, SPLIT_MIN_RECIPES = 30, 5
    SPLIT_RANK1_GAP, SPLIT_RANK1_FLOOR, SPLIT_DOMINANCE = 0.10, 0.70, 0.40

_SPLIT_CFG = {
    "SPLIT_MIN_FRAMES": SPLIT_MIN_FRAMES, "SPLIT_MIN_RECIPES": SPLIT_MIN_RECIPES,
    "SPLIT_RANK1_GAP": SPLIT_RANK1_GAP, "SPLIT_RANK1_FLOOR": SPLIT_RANK1_FLOOR,
    "SPLIT_DOMINANCE": SPLIT_DOMINANCE,
}


# 설정은 별도 파일 golden_eval_config.py 에서 편집(세 드라이버 공용; seed_env() 가 env 로 브리지).
# golden_eval_config.example.py 를 golden_eval_config.py 로 복사해 GOLDEN_ROOT/LAB_MODE/MIN_S 만 고친다.


def _resolve_golden_root():
    """골든 루트 Path. seed_env() 가 이미 config→env 브리지를 했으므로 env 만 보면 된다."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    return Path(root_env) if root_env else glec.GOLDEN_ROOT

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
        # 층화 축 = consensus 풀 크기(history=과거 S 장수, LOO=fm-1; 구버전 row 는 n_S_loo 폴백).
        # 가중치 = eval frame 수(n_S_loo) — 풀 크기와 분리해야 history(풀 크고 frame 적음)가 안 왜곡됨.
        pool = int(r.get("cons_pool_n", r["n_S_loo"]))
        w = int(r["n_S_loo"])
        for lo, hi, lbl in edges:
            if pool >= lo and (hi is None or pool <= hi):
                b = bins[lbl]
                b["n_recipes"] += 1
                b["n_frames"] += w
                b["cons_rank1_sum"] += r["cons_rank1_rate"] * w
                b["cons_topk_sum"] += r["cons_in_topk_rate"] * w
                b["rcp_rank1_sum"] += r["rcp_rank1_rate"] * w
                b["rcp_topk_sum"] += r["rcp_in_topk_rate"] * w
                b["lift_sum"] += r["lift"] * w
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


_FAILURE_TYPES = ("rank1_hit", "look_alike", "recall_miss", "periodic_look_alike")


def _classify_cell(cell):
    """per-frame 셀 → 실패유형. topk_rank==1=rank1_hit, in_topk(=rank 2..k)=look_alike, 그 외=recall_miss."""
    if cell.get("topk_rank") == 1:
        return "rank1_hit"
    if cell.get("in_topk"):
        return "look_alike"
    return "recall_miss"


def _with_shares(acc):
    """counts dict(키 _FAILURE_TYPES + 'n') → {n, <type>:{n, share}}. share = n_type/n."""
    n = int(acc.get("n", 0))
    out = {"n": n}
    for t in _FAILURE_TYPES:
        c = int(acc.get(t, 0))
        out[t] = {"n": c, "share": round(c / n, 3) if n else None}
    return out


def _failure_hist_by_modality(cells):
    """rcp_only per-frame 셀 list → {mod: _with_shares}. periodic_look_alike = look_alike & cell['periodic']."""
    by_mod = {}
    for c in cells:
        mod = (c.get("mod") or "unknown").lower()
        acc = by_mod.setdefault(mod, {t: 0 for t in _FAILURE_TYPES} | {"n": 0})
        t = _classify_cell(c)
        acc[t] += 1
        acc["n"] += 1
        if t == "look_alike" and c.get("periodic"):
            acc["periodic_look_alike"] += 1
    return {mod: _with_shares(acc) for mod, acc in by_mod.items()}


def _failure_hist_from_rates(per_recipe):
    """consensus per_recipe rows → {mod: _with_shares}. per-frame 없이 rate 에서 재구성.

    누적 클램프로 합이 정확히 n 을 보장:
      recall_miss = min(n,        round(n*(1-in_topk)))   (n 이하 보장)
      look_alike  = min(n-miss,   round(n*(in_topk-rank1))) (나머지 예산 이하 보장)
      rank1_hit   = n - recall_miss - look_alike          (항상 >= 0, share 합 <= 1.0)
    periodic recipe(row['periodic'])면 look_alike 전부 periodic.
    """
    by_mod = {}
    for r in per_recipe:
        mod = (r.get("modality") or "unknown").lower()
        n = int(r["n_S_loo"])
        topk = float(r["cons_in_topk_rate"])
        rank1 = float(r["cons_rank1_rate"])
        recall_miss = max(0, min(n, int(round(n * (1.0 - topk)))))
        look_alike = max(0, min(n - recall_miss, int(round(n * (topk - rank1)))))
        rank1_hit = n - recall_miss - look_alike
        acc = by_mod.setdefault(mod, {t: 0 for t in _FAILURE_TYPES} | {"n": 0})
        acc["rank1_hit"] += rank1_hit
        acc["look_alike"] += look_alike
        acc["recall_miss"] += recall_miss
        acc["n"] += n
        if r.get("periodic"):
            acc["periodic_look_alike"] += look_alike
    return {mod: _with_shares(acc) for mod, acc in by_mod.items()}


def _merge_failure_hists(*hists):
    """여러 _with_shares dict 합산 → 새 _with_shares (counts 합산 후 share 재계산)."""
    acc = {t: 0 for t in _FAILURE_TYPES} | {"n": 0}
    for h in hists:
        if not h:
            continue
        for t in _FAILURE_TYPES:
            acc[t] += int(h.get(t, {}).get("n", 0))
        acc["n"] += int(h.get("n", 0))
    return _with_shares(acc)


def _routed_failure_hist(cons_hist, rcp_hist):
    """modality 별 routed 실패히스토그램 = consensus arm + rcp_only arm 합산."""
    return {mod: _merge_failure_hists(cons_hist.get(mod), rcp_hist.get(mod))
            for mod in set(cons_hist) | set(rcp_hist)}


def _confusion_at(samples, thr):
    """[(score, hit_bool)] 를 임계 thr 에서 채점한 혼동행렬. predict pos = score>=thr, actual pos = hit.

    반환 {tp, fp, tn, fn, acc}. n==0 이면 acc=None.
    """
    tp = sum(1 for s, h in samples if h and s >= thr)
    fp = sum(1 for s, h in samples if (not h) and s >= thr)
    fn = sum(1 for s, h in samples if h and s < thr)
    tn = sum(1 for s, h in samples if (not h) and s < thr)
    n = tp + fp + fn + tn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": round((tp + tn) / n, 3) if n else None}


def _youden_threshold(samples):
    """[(score, hit_bool)] → Youden 최적 임계. hit=True 가 positive(임계 이상이면 맞다고 예측).

    임계 후보 = 등장한 score 들. J = TPR - FPR 최대점. 한 클래스라도 비면 None.
    J 동점 시 낮은 임계(sensitivity 우선)가 선택된다.
    """
    pos = [s for s, h in samples if h]
    neg = [s for s, h in samples if not h]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"thr": None, "J": None, "tpr": None, "fpr": None, "n_pos": n_pos, "n_neg": n_neg}
    best = None
    for thr in sorted({s for s, _ in samples}):
        tpr = sum(1 for s in pos if s >= thr) / n_pos
        fpr = sum(1 for s in neg if s >= thr) / n_neg
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, thr, tpr, fpr)
    j, thr, tpr, fpr = best
    return {"thr": round(thr, 4), "J": round(j, 4),
            "tpr": round(tpr, 3), "fpr": round(fpr, 3), "n_pos": n_pos, "n_neg": n_neg}


def _youden_by_modality(cells, prod_thr=None, prod_match_thr=None):
    """rcp_only 셀 → {mod: _youden_threshold}. score(None 아님)+hit 있는 셀만.

    prod_thr 가 주어지면 각 modality 에 prod_thr, delta_vs_prod(=최적 thr - prod_thr),
    confusion_prod(그 임계의 혼동행렬)를 덧붙인다 — L1(per-mod 임계 분리) 증거.

    *1차 prod 기준은 actuation 게이트* = ensemble_adjust_threshold(0.4727, 고-recall)다.
    align-point 보정의 실제 절대 score 컷(align_point_correction.py 의 reposition/low_match
    판정)이 이 값을 쓰므로, balanced 인 ensemble_match_threshold(0.6053)가 아니라 adjust 로
    비교해야 벤치가 실제 보정 결정경로를 반영한다. prod_match_thr 가 주어지면 balanced 임계의
    혼동행렬도 confusion_match 로 함께 낸다("두 임계 모두 리포트").
    """
    by_mod = {}
    for c in cells:
        if c.get("score") is None or c.get("hit") is None:
            continue
        by_mod.setdefault((c.get("mod") or "unknown").lower(), []).append(
            (float(c["score"]), bool(c["hit"])))
    out = {}
    for mod, s in by_mod.items():
        y = _youden_threshold(s)
        if prod_thr is not None and y["thr"] is not None:
            y = {**y, "prod_thr": round(float(prod_thr), 4),
                 "delta_vs_prod": round(y["thr"] - float(prod_thr), 4),
                 "confusion_prod": _confusion_at(s, float(prod_thr))}
            if prod_match_thr is not None:
                y["prod_match_thr"] = round(float(prod_match_thr), 4)
                y["confusion_match"] = _confusion_at(s, float(prod_match_thr))
        out[mod] = y
    return out


# split 판정 로직(아래) — run() 의 Step 2(modality 전략 분기 리포트)에서 사용.
# named lever 는 *modality 특정* 개입이다(spec §5): L2=OM lattice-NMS 주기억제, L3=SEM struct
# proposer. 그래서 (modality, 지배 bucket) 으로 키를 잡는다. 관측 패턴이 설계와 반대면
# (예: SEM 이 periodic 지배) named lever 로 단정하지 않고 중립 라벨 '{mod}:{bucket}' 을 내
# 다음 plan 이 lever 를 고르게 한다.
_LEVER_BY_MOD_BUCKET = {
    ("om", "periodic_look_alike"): "L2_om_periodicity",
    ("sem", "recall_miss"): "L3_sem_recall",
}
# modality 무관(공유) lever — 특정 modality 개입이 아닌 bucket.
_LEVER_BY_BUCKET_SHARED = {
    "other_look_alike": "shared_rerank",
}


def _lever_for(mod, bucket):
    """(modality, 지배 bucket) → lever 라벨. 설계 매핑이면 named lever, 공유 bucket 이면 공유 lever,
    그 외 설계 밖 조합은 중립 '{mod}:{bucket}'(named lever 오라벨 방지)."""
    if bucket is None:
        return None
    if (mod, bucket) in _LEVER_BY_MOD_BUCKET:
        return _LEVER_BY_MOD_BUCKET[(mod, bucket)]
    if bucket in _LEVER_BY_BUCKET_SHARED:
        return _LEVER_BY_BUCKET_SHARED[bucket]
    return f"{mod}:{bucket}"


def _dominant_failure(fail, dominance):
    """_with_shares dict -> 지배 실패 bucket 또는 None. 분모 = 총 실패(look_alike+recall_miss).

    bucket: periodic_look_alike / other_look_alike(= look_alike-periodic) / recall_miss.
    최대 bucket share >= dominance 면 그 bucket, 아니면 None(지배 없음).
    """
    la = int(fail.get("look_alike", {}).get("n", 0))
    pla = int(fail.get("periodic_look_alike", {}).get("n", 0))
    rm = int(fail.get("recall_miss", {}).get("n", 0))
    buckets = {"periodic_look_alike": pla, "other_look_alike": max(0, la - pla), "recall_miss": rm}
    total_fail = sum(buckets.values())
    if total_fail == 0:
        return None
    top = max(buckets, key=buckets.get)
    return top if buckets[top] / total_fail >= dominance else None


def _recipes_by_modality(per_recipe, rcp_only_cells):
    """modality 별 기여 recipe 수(중복 제거). consensus=row['recipe'], rcp_only=cell['rec_key']."""
    by_mod = {}
    for r in per_recipe:
        by_mod.setdefault((r.get("modality") or "unknown").lower(), set()).add(r.get("recipe"))
    for c in rcp_only_cells:
        by_mod.setdefault((c.get("mod") or "unknown").lower(), set()).add(c.get("rec_key"))
    return {mod: len(s) for mod, s in by_mod.items()}


def _split_verdict(fail_om, fail_sem, rate_om, rate_sem, cfg):
    """OM/SEM 실패히스토그램(routed) + rate -> verdict.

    게이트: n_frames>=MIN_FRAMES AND n_recipes>=MIN_RECIPES (미달 -> verdict=insufficient).
    격차: |r1차| >= RANK1_GAP OR min(r1) < RANK1_FLOOR.
    분기: 두 modality 지배 실패유형이 서로 다름.
    verdict: 격차&분기 -> SPLIT(+lever), 격차만 -> shared_tune, 그 외 -> no_split.
    rate_om/rate_sem: {n_frames, rank1_rate, n_recipes} (run() 가 routed rate + _recipes_by_modality 로 조립).
    한 modality 만 지배 실패유형이 있으면 divergent=False → shared_tune.
    """
    insufficient = [name for name, rate in (("om", rate_om), ("sem", rate_sem))
                    if rate.get("n_frames", 0) < cfg["SPLIT_MIN_FRAMES"]
                    or rate.get("n_recipes", 0) < cfg["SPLIT_MIN_RECIPES"]]
    if insufficient:
        return {"verdict": "insufficient", "insufficient_mods": insufficient, "gap_reason": None,
                "dominant_om": None, "dominant_sem": None, "suggested_levers": []}

    r_om = rate_om.get("rank1_rate") or 0.0
    r_sem = rate_sem.get("rank1_rate") or 0.0
    gap_abs = abs(r_om - r_sem) >= cfg["SPLIT_RANK1_GAP"]
    gap_floor = min(r_om, r_sem) < cfg["SPLIT_RANK1_FLOOR"]
    gap = gap_abs or gap_floor
    gap_reason = ("abs_diff" if gap_abs else "floor") if gap else None
    dom_om = _dominant_failure(fail_om, cfg["SPLIT_DOMINANCE"])
    dom_sem = _dominant_failure(fail_sem, cfg["SPLIT_DOMINANCE"])
    divergent = dom_om is not None and dom_sem is not None and dom_om != dom_sem

    if gap and divergent:
        verdict = "SPLIT"
        levers = sorted({_lever_for("om", dom_om), _lever_for("sem", dom_sem)} - {None})
    elif gap:
        verdict, levers = "shared_tune", []
    else:
        verdict, levers = "no_split", []
    return {"verdict": verdict, "insufficient_mods": [], "gap_reason": gap_reason,
            "dominant_om": dom_om, "dominant_sem": dom_sem, "suggested_levers": levers}


def _consensus_by_modality(per_recipe):
    """consensus per_recipe rows -> {modality: {n_frames, rank1_rate, in_topk_rate}} (frame-weighted).

    OM(저배율·반복패턴 多)과 SEM(box/직선) 이 같은 단일 CV 정책에서 다르게 동작하는지 측정용.
    """
    acc = {}
    for r in per_recipe:
        mod = (r.get("modality") or "unknown").lower()
        w = int(r["n_S_loo"])
        a = acc.setdefault(mod, {"n": 0, "r1": 0.0, "tk": 0.0})
        a["n"] += w
        a["r1"] += r["cons_rank1_rate"] * w
        a["tk"] += r["cons_in_topk_rate"] * w
    return {mod: {"n_frames": a["n"],
                  "rank1_rate": round(a["r1"] / a["n"], 3) if a["n"] else None,
                  "in_topk_rate": round(a["tk"] / a["n"], 3) if a["n"] else None}
            for mod, a in acc.items()}


def _arm_rates_by_modality(cells_list):
    """rcp-only 셀 list 를 cell['mod'] 로 쪼개 modality 별 _arm_rates."""
    by_mod = {}
    for c in cells_list:
        by_mod.setdefault((c.get("mod") or "unknown").lower(), []).append(c)
    return {mod: _arm_rates(cs) for mod, cs in by_mod.items()}


def _routed_by_modality(cons_by_mod, rcp_by_mod):
    """modality 별 routed 종합 (eligible→consensus, rest→rcp). frame-weighted."""
    out = {}
    for mod in set(cons_by_mod) | set(rcp_by_mod):
        c = cons_by_mod.get(mod, {})
        rp = rcp_by_mod.get(mod, {})
        out[mod] = _routed_overall(
            c.get("n_frames", 0), c.get("rank1_rate"), c.get("in_topk_rate"),
            {"n": rp.get("n", 0), "rank1_rate": rp.get("rank1_rate"),
             "in_topk_rate": rp.get("in_topk_rate")})
    return out


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
    root = _resolve_golden_root()   # seed_env() 가 config→env 브리지 완료. 여기선 env 만.
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
    periodic_by_key_mod = {}          # (rec_key, mod) -> bool (등록 key 가 주기/대칭 = template_periodicity > tau)
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
        # periodicity 는 modality 별로 따로 — dual OM/SEM recipe 에서 한쪽(OM) 주기성이
        # 다른쪽(SEM) look_alike 로 새면 OM/SEM split 증거가 오염된다. center_tpls 는 mod 키이므로
        # (rec_key, mod) 로 저장하고, 셀/row 는 각자의 modality 로 조회한다.
        for mod, val in center_tpls.items():
            tpl = val[0] if val is not None else None
            if tpl is None:
                continue
            try:
                is_periodic = template_periodicity(tpl.raw_image) > PERIODICITY_TAU
            except Exception as exc:
                print(f"[WARNING] periodicity 계산 실패 {assets.recipe_id}/{mod}: {exc}")
                is_periodic = False
            periodic_by_key_mod[(rec_key, mod)] = is_periodic
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
            for c in cells:                       # 실패유형 히스토그램용 태깅(순수 헬퍼는 이 필드만 읽음).
                # 셀의 실제 매칭 modality(cell['mod'])로 periodicity 조회 — 교차오염 차단.
                c["periodic"] = periodic_by_key_mod.get(
                    (rec_key, (c.get("mod") or "").lower()), False)
                c["rec_key"] = rec_key
            n_rcp_only_recipes += 1
            rcp_only_cells.extend(cells)
    rcp_only = _arm_rates(rcp_only_cells)

    # consensus per_recipe row 에 periodic(recipe·modality-level) join — _failure_hist_from_rates 가 읽음.
    # row 의 dominant modality(row['modality'])로 조회 — OM↔SEM 교차오염 차단.
    for r in per_recipe:
        r["periodic"] = periodic_by_key_mod.get(
            (r["recipe"], (r.get("modality") or "").lower()), False)

    # 4) 리포트 조립.
    scaling = _bin_by_s_count(per_recipe)
    routed = _routed_overall(cons_frames, cons_rank1, cons_topk, rcp_only)
    n_history = sum(1 for r in per_recipe if r.get("mode") == "history")
    # modality 층화(Step 1) — OM/SEM 가 같은 단일 CV 정책에서 다르게 동작하는지 측정.
    cons_by_mod = _consensus_by_modality(per_recipe)
    rcp_by_mod = _arm_rates_by_modality(rcp_only_cells)
    routed_by_mod = _routed_by_modality(cons_by_mod, rcp_by_mod)

    # Step 2: 실패유형 히스토그램(양 arm + routed) + per-mod Youden + split verdict.
    # consensus row 는 'modality', rcp 셀은 'mod' 키 — 각 헬퍼가 맞는 필드를 읽고 _routed_failure_hist 가 병합.
    cons_fail = _failure_hist_from_rates(per_recipe)
    rcp_fail = _failure_hist_by_modality(rcp_only_cells)
    routed_fail = _routed_failure_hist(cons_fail, rcp_fail)
    youden_by_mod = _youden_by_modality(
        rcp_only_cells,
        prod_thr=gle.STRUCTURE_POLICY.ensemble_adjust_threshold,        # actuation 게이트(1차).
        prod_match_thr=gle.STRUCTURE_POLICY.ensemble_match_threshold)   # balanced(참고용 2차).
    n_rec_by_mod = _recipes_by_modality(per_recipe, rcp_only_cells)

    def _mod_rate(mod):
        rt = routed_by_mod.get(mod, {})
        return {"n_frames": rt.get("n_frames", 0), "rank1_rate": rt.get("rank1_rate"),
                "n_recipes": n_rec_by_mod.get(mod, 0)}

    empty_hist = _with_shares({"n": 0})
    split_verdict = _split_verdict(
        routed_fail.get("om", empty_hist), routed_fail.get("sem", empty_hist),
        _mod_rate("om"), _mod_rate("sem"), _SPLIT_CFG)

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
            "n_recipes_history": n_history,                       # 별도 history root 로 consensus 빌드한 recipe.
            "n_recipes_loo": len(eligible_keys) - n_history,      # history 없어 from_msr LOO 로 빌드한 recipe.
            "rcp_counterfactual_rank1_rate": res["overall_rcp_rank1_rate"] if res else None,
            "rcp_counterfactual_in_topk_rate": res["overall_rcp_in_topk_rate"] if res else None,
            "overall_lift": res["overall_lift"] if res else None,
            "rank1_lift": res["rank1_lift"] if res else None,
        },
        "consensus_scaling_by_s_count": scaling,   # (A)
        "rcp_only_arm": rcp_only,                   # (B)
        "routed_overall": routed,                   # (C)
        "by_modality": {                            # Step 1: OM vs SEM 층화(split 여부 판단 근거).
            "consensus": cons_by_mod,
            "rcp_only": rcp_by_mod,
            "routed": routed_by_mod,
            "failure_modes": {                      # Step 2: 실패유형 분해(어떻게 깨지나).
                "consensus": cons_fail, "rcp_only": rcp_fail, "routed": routed_fail,
            },
            "youden": youden_by_mod,                # Step 2: per-mod 임계 분리(L1 증거; rcp_only arm).
        },
        "split_verdict": split_verdict,             # Step 2: SPLIT/shared_tune/no_split/insufficient.
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


def _mod_evidence_tok(summary):
    """digest 용 per-modality 증거 토큰: OM/SEM 의 실패share(la/pm/rm) + youden thr/J. 없으면 '-'."""
    bm = summary.get("by_modality") or {}
    fm = bm.get("failure_modes", {}).get("routed", {})
    yd = bm.get("youden", {})
    parts = []
    for m in ("om", "sem"):
        h = fm.get(m)
        if not h:
            continue
        la = h.get("look_alike", {}).get("share")
        pm = h.get("periodic_look_alike", {}).get("share")
        rm = h.get("recall_miss", {}).get("share")
        y = yd.get(m) or {}
        yt = f"{y.get('thr')}/{y.get('J')}" if y.get("thr") is not None else "-"
        parts.append(f"{m.upper()}:la/pm/rm={la}/{pm}/{rm} yd={yt}")
    return " ".join(parts) or "-"


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
    cons_mode = f"hist:{ca.get('n_recipes_history', 0)}/loo:{ca.get('n_recipes_loo', 0)}"
    routed_mod = (summary.get("by_modality") or {}).get("routed", {})

    def _mod(m):
        d = routed_mod.get(m)
        return f"{m.upper()} r1/topk={d['rank1_rate']}/{d['in_topk_rate']} (n={d['n_frames']})" if d else f"{m.upper()} -"

    sv = summary.get("split_verdict") or {}
    verdict_tok = (f"verdict={sv.get('verdict', '-')}"
                   + (f"(om={sv.get('dominant_om')},sem={sv.get('dominant_sem')})"
                      if sv.get('verdict') == 'SPLIT' else ""))
    evidence_tok = _mod_evidence_tok(summary)
    return (
        f"lab={lab} minS={summary['consensus_min_s']} consMode={cons_mode} | "
        f"routed r1/topk={r['rank1_rate']}/{r['in_topk_rate']} (n={r['n_frames']}) | "
        f"cons r1/topk={ca['rank1_rate']}/{ca['in_topk_rate']} lift={ca['overall_lift']} "
        f"(n={ca['n_frames']},rec={summary['n_recipes_consensus_eligible']}) | "
        f"rcp_only r1/topk={ro['rank1_rate']}/{ro['in_topk_rate']} "
        f"(n={ro['n']},rec={summary['n_recipes_rcp_only']}) | "
        f"mod[{_mod('om')} {_mod('sem')}] | "
        f"scaling[{scaling}] | "
        f"{verdict_tok} | "
        f"evid[{evidence_tok}]"
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

    bm = summary.get("by_modality") or {}
    print("\n[INFO] === (Step 1) modality 층화 — OM vs SEM (단일 CV 정책에서 다르게 동작하나) ===")
    print(f"    {'mod':<6} {'consensus r1/topk (n)':<26} {'rcp_only r1/topk (n)':<24} {'routed r1/topk (n)':<22}")
    for mod in ("om", "sem"):
        c = bm.get("consensus", {}).get(mod, {})
        rp = bm.get("rcp_only", {}).get(mod, {})
        rt = bm.get("routed", {}).get(mod, {})
        cs = f"{c.get('rank1_rate')}/{c.get('in_topk_rate')} ({c.get('n_frames', 0)})"
        rs = f"{rp.get('rank1_rate')}/{rp.get('in_topk_rate')} ({rp.get('n', 0)})"
        ts = f"{rt.get('rank1_rate')}/{rt.get('in_topk_rate')} ({rt.get('n_frames', 0)})"
        print(f"    {mod.upper():<6} {cs:<26} {rs:<24} {ts:<22}")
    print("    * OM(저배율·반복패턴) rank1 이 SEM 보다 확연히 낮고 실패유형이 다르면 → modality-specific CV 정책 검토.")

    fm = (summary.get("by_modality") or {}).get("failure_modes", {}).get("routed", {})
    yd = (summary.get("by_modality") or {}).get("youden", {})
    print("\n[INFO] === (Step 2) 실패유형 분해 (routed; 'rank1 이 왜 낮나') ===")
    print(f"    {'mod':<6} {'n':>5} {'rank1_hit':>10} {'look_alike':>11} "
          f"{'periodic_la':>12} {'recall_miss':>12}   youden(thr/J,n+/-)")
    _sh = lambda hist, t: hist.get(t, {}).get("share")   # 실패유형 share 안전 추출.
    for mod in ("om", "sem"):
        h = fm.get(mod, {})
        y = yd.get(mod, {})
        if not h:
            continue
        if y.get("thr") is not None:
            ys = f"{y.get('thr')}/{y.get('J')} ({y.get('n_pos')}+/{y.get('n_neg')}-)"
            if y.get("delta_vs_prod") is not None:
                conf = y.get("confusion_prod") or {}
                ys += f" d_prod={y.get('delta_vs_prod')} accP={conf.get('acc')}"
                confm = y.get("confusion_match")
                if confm is not None:
                    ys += f" accM={confm.get('acc')}"
        else:
            ys = "-"
        print(f"    {mod.upper():<6} {h.get('n', 0):>5} {str(_sh(h, 'rank1_hit')):>10} "
              f"{str(_sh(h, 'look_alike')):>11} {str(_sh(h, 'periodic_look_alike')):>12} "
              f"{str(_sh(h, 'recall_miss')):>12}   {ys}")
    print("    * periodic_la 지배=OM 주기억제(L2), recall_miss 지배=SEM recall proposer(L3)."
          " d_prod=최적임계-actuation게이트(adjust 0.4727) 차이, accP=adjust 게이트 정확도,"
          " accM=balanced(match 0.6053) 게이트 정확도(L1 증거).")

    sv = summary.get("split_verdict") or {}
    print(f"\n[INFO] === (Step 2) SPLIT 판정 === verdict={sv.get('verdict', '-')}  "
          f"gap_reason={sv.get('gap_reason')}  "
          f"OM 지배={sv.get('dominant_om')}  SEM 지배={sv.get('dominant_sem')}  "
          f"권장 lever={sv.get('suggested_levers') or '-'}"
          + (f"  (insufficient: {sv.get('insufficient_mods')})"
             if sv.get('verdict') == 'insufficient' else ""))


if __name__ == "__main__":
    _status = run()
    raise SystemExit(0 if _status in ("success", "no_data") else 1)
