"""reranker rule A/B — 전체 S 프레임에서 여러 rerank 규칙의 최종 align-point hit_rate 비교.

신호 프로브 결과 NCC 가 chamfer_miss 를 11/11 분리. 이 러너는 그게 *전체*에서도 통하는지(=NCC
reranking 이 0.698 proposer 이득을 최종 hit 으로 전환하면서 both_hit 을 안 깨는지)를 검증한다.

ensemble pool(chamfer/orb/err)을 프레임당 1회 만들고 후보별 NCC 를 더해, 같은 pool 에 여러 규칙을
적용한 최종 픽 hit_rate 를 한 run 으로 비교:
- baseline      : compute_align_key_score (C1 단일 proposer, 현 0.422).
- ens_orb       : chamfer_w·chamfer + orb_w·orb (현 production ensemble, 현 0.407).
- ens_ncc       : chamfer_w·chamfer + ncc_w·max(0,ncc)  — ORB 제거 + NCC reranker.
- ens_ncc_only  : argmax(ncc)                            — 순수 NCC reranker.
NCC 는 *reranker*(구조-제안된 소수 후보 중 판별)로만 쓴다 — primary matcher 로서의 NCC 금지
([[project_align_key_matching_constraint]])는 별개. 가중치는 env(RERANK_CHAMFER_W/RERANK_NCC_W).

또한 ens_ncc 픽의 결정 score(sel = chamfer_w·chamfer + ncc_w·max(0,ncc)) + GT hit 분포로
decision threshold(match/adjust)를 ROC 캘리브해 콘솔·summary 에 출력한다(decision/score 정비 §4).
오피스 1회 실행이면 ensemble_match/adjust_threshold 숫자가 바로 찍혀 jsonl 반입이 불필요하다.

입력: golden 데이터. 출력: rule_ab/<ts>/{summary.json(+calibration), rows.jsonl(+ens_ncc_sel)}.
실행(오피스): uv run python poc/workflow_2/reranker_rule_ab.py
"""
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import json
from pathlib import Path

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.assets import iter_msr_images, load_gray
from poc.workflow_3.align.matching.engine import STRUCTURE_POLICY, compute_align_key_score, preprocess_for_matching
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_3.align.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.align.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_3.align.diagnostics.align_point_correction import _tool_label
from poc.workflow_2.localization_ab_ensemble import _err_norm, _predicted_align_point
from poc.workflow_2.localization_regression_diag import _ensemble_pool
from poc.workflow_2.reranker_signal_probe import _candidate_signals
from poc.workflow_3.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "reranker_rule_ab"

# ensemble rerank 규칙(baseline 은 별도 함수). 표 순서.
ENS_RULES = ("ens_orb", "ens_ncc", "ens_ncc_only")
ALL_RULES = ("baseline",) + ENS_RULES


def _ens_ncc_sel(c, chamfer_w, ncc_w):
    """ens_ncc 결정 score = chamfer_w·chamfer + ncc_w·max(0,ncc) (음의 NCC 는 0 클램프).

    production 매처(align_key_matcher.compute_align_key_score_ensemble 의 selection sel)와
    동일한 식 — 이 한 정의를 _rule_picks(랭킹)와 캘리브(threshold 산출) 양쪽이 공유해야
    덤프한 sel 분포가 실제 production 결정 score 와 일치한다(공식 drift 방지).
    """
    return chamfer_w * c["chamfer"] + ncc_w * max(0.0, c["ncc"])


def _rule_picks(pool, *, chamfer_w, ncc_w):
    """NCC 기반 rerank 규칙별 picked_idx. pool[i] 는 chamfer·ncc 키를 가진다.

    ens_ncc      : argmax(chamfer_w·chamfer + ncc_w·max(0,ncc)) — 음의 상관은 0 으로 클램프.
    ens_ncc_only : argmax(ncc) (raw).
    """
    def _amax(key):
        return max(range(len(pool)), key=lambda i: key(pool[i]))
    return {
        "ens_ncc": _amax(lambda c: _ens_ncc_sel(c, chamfer_w, ncc_w)),
        "ens_ncc_only": _amax(lambda c: c["ncc"]),
    }


def _calibrate_thresholds(pairs, recall_target=0.95):
    """ens_ncc 의 (sel 점수, GT hit) 쌍에서 decision threshold 2개를 데이터로 산출.

    decision/score 정비(spec 2026-06-09-ensemble-decision-score)의 §4 캘리브:
    ensemble 결정 score = sel = rerank_chamfer_w·chamfer + rerank_ncc_w·max(0,ncc) 의
    분포에서 hit/miss 를 가르는 컷을 정한다. 가중치가 baseline(chamfer+ORB)과 달라
    기존 0.62/0.40 임계를 그대로 못 쓰므로 ROC 로 재캘리브.

    - match  = Youden J(=TPR−FPR) 최대점의 sel 컷(균형). 동률이면 더 낮은 컷 채택.
    - adjust = recall(TPR) ≥ ``recall_target`` 인 가장 큰 sel 컷(고-recall, hit 대부분 포착).
               항상 match 이하로 클램프(adjust 는 더 느슨한 게이트).
    sel 컷 후보는 *관측된 sel 값*만 쓴다(분포에 실재하는 경계). 한쪽 클래스만이면 None.

    pairs: list[(float sel, bool hit)]. 반환 dict 또는 None.
    """
    pos = [s for s, h in pairs if h]
    neg = [s for s, h in pairs if not h]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    cands = sorted({s for s, _ in pairs})   # 오름차순 — 관측 sel 경계.
    best_t, best_j, best_stats = cands[0], -1.0, None
    for t in cands:
        tp = sum(1 for s in pos if s >= t)
        fp = sum(1 for s in neg if s >= t)
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best_j:   # strict → 동률 시 더 낮은(=먼저 만난) 컷 유지.
            best_j, best_t = j, t
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            best_stats = {"tpr": round(tpr, 4), "fpr": round(fpr, 4),
                          "precision": round(prec, 4), "recall": round(tpr, 4)}
    # adjust: TPR 은 t 에 단조 비증가 → 조건 만족하는 마지막(가장 큰) 컷.
    adjust_t = cands[0]
    for t in cands:
        if sum(1 for s in pos if s >= t) / n_pos >= recall_target:
            adjust_t = t
    adjust_t = min(adjust_t, best_t)
    # threshold 는 관측 sel 값이라 production(score >= mt)이 캘리브 분할을 정확히 재현하도록
    # 6 자리까지 보존(4 자리 반올림은 컷을 관측값 아래로 내려 경계 프레임 분류를 뒤집을 수 있음).
    return {"match": round(best_t, 6), "adjust": round(adjust_t, 6),
            "youden_j": round(best_j, 4), "n_pos": n_pos, "n_neg": n_neg,
            "at_match": best_stats, "recall_target": recall_target}


def _confusion(hit_a, hit_b):
    """두 hit 불리언 리스트의 confusion (a 기준 gain/regress)."""
    conf = {"both_hit": 0, "only_a": 0, "only_b": 0, "both_miss": 0}
    for ha, hb in zip(hit_a, hit_b):
        if ha and hb:
            conf["both_hit"] += 1
        elif ha:
            conf["only_a"] += 1
        elif hb:
            conf["only_b"] += 1
        else:
            conf["both_miss"] += 1
    return conf


def run():
    """reranker rule A/B (인자 없음). 반환 success|no_data."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터 없음: {root}")
        return "no_data"
    chamfer_w = float(os.getenv("RERANK_CHAMFER_W", "0.5") or "0.5")
    ncc_w = float(os.getenv("RERANK_NCC_W", "0.5") or "0.5")
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = STRUCTURE_POLICY
    n_rec = len(recipes)
    print(f"[INFO] reranker rule A/B 시작 — recipe {n_rec}개 "
          f"(ens_ncc: chamfer_w={chamfer_w}, ncc_w={ncc_w})")

    hits = {r: [] for r in ALL_RULES}
    rows = []
    calib_pairs = []   # ens_ncc (sel 점수, GT hit) — decision threshold 캘리브용(§4).
    drop = {"no_box_tpl": 0, "non_S": 0, "routing_miss": 0, "no_crosshair": 0,
            "load_failed": 0, "no_pool": 0}
    n = 0
    for ri, assets in enumerate(recipes, 1):
        if assets is None:
            continue
        try:
            _center, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        available = {m for m, v in box_tpls.items() if v is not None}
        if not available:
            drop["no_box_tpl"] += 1
            continue
        for p in iter_msr_images(assets):
            if _tool_label(p.name) != "S":
                drop["non_S"] += 1
                continue
            cond = load_cond(p)
            routed = glec._route_modality(cond, available)
            if routed is None or box_tpls.get(routed) is None:
                drop["routing_miss"] += 1
                continue
            tpl, (dx, dy) = box_tpls[routed]
            if not (cond and cond.crosshair_xy is not None):
                drop["no_crosshair"] += 1
                continue
            gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
            gt_xy = (int(round(gx)), int(round(gy)))
            try:
                frame = clean_image(load_gray(p), cond)
            except Exception:
                drop["load_failed"] += 1
                continue
            _edges, frame_dt = preprocess_for_matching(frame)
            short = max(1, min(tpl.raw_image.shape[0], tpl.raw_image.shape[1]))

            pool, picked_idx = _ensemble_pool(
                tpl, frame, frame_dt, (dx, dy), gt_xy, short, policy)
            if not pool:
                drop["no_pool"] += 1
                continue
            for c in pool:
                sig = _candidate_signals(tpl.raw_image, frame, c)
                c["ncc"] = float(sig["ncc"]) if sig is not None else -1.0

            res_b = compute_align_key_score(tpl, frame, scales=COMPARE_SCALES, policy=policy)
            base_pred = _predicted_align_point(res_b, (dx, dy))
            base_hit = _err_norm(base_pred, gt_xy, short) <= GT_TOL_NORM

            picks = {"ens_orb": picked_idx}
            picks.update(_rule_picks(pool, chamfer_w=chamfer_w, ncc_w=ncc_w))
            rule_hit = {"baseline": base_hit}
            for r in ENS_RULES:
                rule_hit[r] = pool[picks[r]]["err"] <= GT_TOL_NORM
            for r in ALL_RULES:
                hits[r].append(rule_hit[r])
            # ens_ncc 픽의 결정 score(=sel) + hit → threshold 캘리브. ensemble production
            # 결정 score 가 될 값이라(decision/score 정비 §2) 이 분포로 컷을 정한다.
            ens_ncc_sel = _ens_ncc_sel(pool[picks["ens_ncc"]], chamfer_w, ncc_w)
            calib_pairs.append((ens_ncc_sel, bool(rule_hit["ens_ncc"])))
            rows.append({"recipe": assets.recipe_id, "msr": p.name,
                         "ens_ncc_sel": round(ens_ncc_sel, 4),
                         **{r: bool(rule_hit[r]) for r in ALL_RULES}})
            n += 1
            if n % 25 == 0:
                print(f"[INFO] 진행 {n} S frames (recipe {ri}/{n_rec})")

    if not n:
        print(f"[ERROR] 처리된 S 프레임 없음. (누락 {drop})")
        return "no_data"

    hit_rate = {r: round(sum(hits[r]) / n, 3) for r in ALL_RULES}
    # 최강 ensemble 규칙 vs baseline confusion(both_hit 깨짐/이득 확인).
    best_rule = max(ENS_RULES, key=lambda r: hit_rate[r])
    conf = _confusion(hits[best_rule], hits["baseline"])
    calib = _calibrate_thresholds(calib_pairs)   # ens_ncc sel 분포 → match/adjust threshold(§4).
    summary = {"n": n, "GT_TOL_NORM": GT_TOL_NORM, "chamfer_w": chamfer_w, "ncc_w": ncc_w,
               "drop": drop, "hit_rate": hit_rate, "best_ens_rule": best_rule,
               "best_vs_baseline": {"both_hit": conf["both_hit"], "only_best_gain": conf["only_a"],
                                    "only_baseline_regress": conf["only_b"], "both_miss": conf["both_miss"],
                                    "net": round((conf["only_a"] - conf["only_b"]) / n, 3)},
               "calibration": calib}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] S 채택 {n}장 | 누락 {drop}")
    print(f"\n[INFO] === reranker rule A/B (S {n}장, tol={GT_TOL_NORM}, "
          f"ens_ncc w={chamfer_w}/{ncc_w}) ===")
    print(f"  {'rule':<14} {'hit_rate':>9}")
    for r in ALL_RULES:
        print(f"  {r:<14} {hit_rate[r]:>9}")
    print(f"\n  >>> 최강 ensemble 규칙: {best_rule} (hit {hit_rate[best_rule]}) "
          f"vs baseline {hit_rate['baseline']} = net {summary['best_vs_baseline']['net']:+}")
    print(f"  {best_rule} vs baseline: both_hit={conf['both_hit']} "
          f"gain={conf['only_a']} regress={conf['only_b']} both_miss={conf['both_miss']}")
    print("  해석: best > baseline & regress 작음 → NCC reranker 가 이득 전환(production 후보).")
    if calib is not None:
        print(f"\n[INFO] === ens_ncc threshold 캘리브 (sel = {chamfer_w}·chamfer + {ncc_w}·max(0,ncc)) ===")
        print(f"  >>> ensemble_match_threshold  = {calib['match']}  (Youden J={calib['youden_j']}, "
              f"at_match: prec={calib['at_match']['precision']} recall={calib['at_match']['recall']} "
              f"fpr={calib['at_match']['fpr']})")
        print(f"  >>> ensemble_adjust_threshold = {calib['adjust']}  "
              f"(recall_target={calib['recall_target']})")
        print(f"  (n_pos={calib['n_pos']} n_neg={calib['n_neg']}) — 이 두 값을 MatchPolicy 에 하드코딩.")
        if calib["adjust"] >= calib["match"]:
            print("  [WARNING] adjust == match — 'adjust' gray-zone 소멸(clamp). 모든 결정이 "
                  "match/low 로만 갈림. 데이터 분리가 깨끗하면 발생 가능.")
    else:
        print("\n[WARNING] 캘리브 불가(한쪽 클래스만) — threshold 산출 실패.")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
