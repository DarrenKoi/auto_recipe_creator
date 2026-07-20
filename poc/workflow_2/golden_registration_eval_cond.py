"""top-K 후보 국소 registration verifier A/B 드라이버 (cond GT, recipe 샘플링).

`docs/study/cv/align_fail_cv_methods_research_ko.md` 의 P0/P1 verifier arm
(ECC / phase[raw·grad] / phase→ECC cascade / SIFT·AKAZE+RANSAC / MIND-like[P1-D])을
**production 무수정**으로 A/B 한다. arm 이 2개 이상이면 'fuse' 의사-arm(각 arm 재정렬의
RRF 합의 — `registration_lab.rrf_fuse_orders`)도 같은 표에 집계한다.

- B0(기준선) = 기존 consensus LOO harness(`align_similarity._consensus_template_ab`)
  그대로: proposer(3채널 RRF ensemble) top-K 의 1순위가 정답이면 hit.
- 각 arm = 같은 top-K 후보를 `registration_lab` verifier 로 재정렬/정밀화한 뒤의
  1순위 hit. 거부된 후보는 baseline 순서 유지(전부 거부면 B0 로 강등 = 안전 폴백).
- harness 는 `combined_renderer` per-point hook 으로만 붙는다 — 기존 드라이버/엔진
  파일은 한 줄도 수정하지 않는다. hook 안에서 proposer 를 같은 인자로 1회 재실행해
  후보의 scale 을 복원한다(gc 는 xy 만 보관; 결정적이라 gc 와 동일 후보 — mismatch
  카운터로 상시 검증). 비용은 point 당 proposer 2회 — 그래서 **샘플링이 기본**이다.

실패 분류(문서 §2)도 같은 실행에서 공짜로 집계한다:
  proposer_miss(정답이 top-K 에 없음) / rank_error(있는데 1순위 아님) / rank1_ok.
verifier 는 rank_error 버킷만 고칠 수 있다 — 버킷 크기가 이 실험의 기대 상한이다.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/golden_registration_eval_cond.py
설정 (우선순위: env > golden_eval_config.py > 기본값):
    ALIGN_REG_SAMPLE_N    / REG_SAMPLE_N    recipe 표본 수 (기본 0=전체; 1차 실측 후
                                            표본 40 → 전체로 승격 — CI 가 0 을 포함해
                                            표본 확대가 판정 조건)
    ALIGN_REG_SAMPLE_SEED / REG_SAMPLE_SEED 표본 시드 (기본 0; 같은 시드=같은 표본)
    ALIGN_REG_ARMS        / REG_ARMS        쉼표 arm 목록 (기본 = "ecc,mind";
                                            2026-07-20 오피스 1차 A/B 결과 반영 —
                                            phase/phase_ecc 는 regress 2배로 유해,
                                            sift/akaze 는 전 후보 거부(0/0), grad_phase
                                            는 churn. 전체 7종 재실험은 이 env 로 복원:
                                            ALIGN_REG_ARMS 에 REG_ARM_NAMES 전체 지정.
                                            'fuse' 는 arm 이 아니라 자동 의사-arm)
    ALIGN_REG_FUSE        / REG_FUSE        RRF 합의 의사-arm on/off (기본 1; arm>=2 필요)
    ALIGN_REG_PROD_ARM    / REG_PROD_ARM    production selection 재현 의사-arm on/off (기본 1).
                                            'prod' = 운영 NCC rerank(sel = rerank_chamfer_w
                                            ·chamfer + rerank_ncc_w·max(0,ncc), engine 그대로)
                                            로 후보를 고르는 기준 arm — B0(RRF 순서)가 아니라
                                            이것이 오늘 production 의 실제 성적이다.
                                            'prod_mind' = prod 순서와 mind 순서의 RRF 결합
                                            (mind 가 arm 목록에 있을 때만) — 포팅 후보안.
                                            판정: prod_mind > prod 여야 mind 포팅 가치가 있다
                                            (mind > B0 만으로는 NCC 와 이득이 겹칠 수 있음).
    ALIGN_REG_OVERLAY_MAX / REG_OVERLAY_MAX top-1 이 바뀐 행 overlay 저장 상한 (기본 60)
  골든 루트/MIN_S/CLEAN_FRAME 등은 consensus 드라이버와 동일 env 를 그대로 따른다.
출력: stdout 표 + [DIGEST] 한 줄 + DEBUG_IMAGE_DIR/golden_registration_eval_cond/<ts>/
      {rows.jsonl, summary.json, digest.txt, overlays/, consensus/}
(cond.txt 실데이터 전용 — 데이터 없으면 no_data 로 종료. 합성 검증은
 `uv run python poc/workflow_2/registration_lab.py` 의 self-test 가 담당.)
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
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from poc.workflow_2 import DEBUG_IMAGE_DIR

# golden_eval_config.py 상수(GOLDEN_ROOT/HISTORY_ROOT/MIN_S/LAB_MODE) → env.
# gce 의 import 시점 CONSENSUS_MIN_S 읽기 전에 와야 한다(다른 드라이버와 동일 순서).
from poc.workflow_2.golden_eval_config_loader import seed_env
seed_env()

from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_2 import golden_localization_eval_cond as glec
from poc.workflow_2 import golden_consensus_eval_cond as gce
from poc.workflow_2 import registration_lab as reg
from poc.workflow_2.align_similarity import (
    COMPARE_SCALES,
    GT_TOL_NORM,
    TOPK_CANDIDATES,
    USE_ENSEMBLE_PROPOSER,
    _consensus_template_ab,
    _propose_topk,
)
from poc.workflow_3.align.matching.engine import (
    DEFAULT_POLICY,
    _candidate_ncc,
    _rescore_positions_to_candidates,
    preprocess_for_matching,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag

# 선택 상수는 golden_eval_config.py 에 있으면 읽는다(없어도 동작 — env/기본값 폴백).
try:
    from poc.workflow_2 import golden_eval_config as _cfg
except Exception:
    _cfg = None


def _opt(env_name, cfg_name, default, cast=int):
    """설정 1개 해석 — 우선순위: 실제 env > golden_eval_config.py 상수 > 기본값."""
    raw = os.getenv(env_name)
    if raw not in (None, ""):
        return cast(raw)
    val = getattr(_cfg, cfg_name, None) if _cfg is not None else None
    if val in (None, ""):
        return default
    return cast(val)


SAMPLE_N = _opt("ALIGN_REG_SAMPLE_N", "REG_SAMPLE_N", 0)           # 0 = 전체 recipe.
SAMPLE_SEED = _opt("ALIGN_REG_SAMPLE_SEED", "REG_SAMPLE_SEED", 0)
# 기본 arm = 1차 실측 생존자만(ecc,mind) — fuse 가 유해 arm 에 희석되지 않게.
ARMS = tuple(a.strip() for a in
             _opt("ALIGN_REG_ARMS", "REG_ARMS", "ecc,mind", str).split(",")
             if a.strip())
FUSE_ON = bool(_opt("ALIGN_REG_FUSE", "REG_FUSE", 1)) and len(ARMS) >= 2
PROD_ARM = bool(_opt("ALIGN_REG_PROD_ARM", "REG_PROD_ARM", 1))
OVERLAY_MAX = _opt("ALIGN_REG_OVERLAY_MAX", "REG_OVERLAY_MAX", 60)

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_registration_eval_cond"

# overlay 색(BGR): GT=초록, B0 top-1=하늘, arm top-1 은 아래 표.
_CLR_GT = (0, 200, 0)
_CLR_B0 = (255, 200, 0)
_ARM_CLR = {"ecc": (0, 170, 255), "phase": (255, 0, 255),
            "phase_ecc": (255, 128, 0), "grad_phase": (128, 220, 128),
            "sift": (0, 255, 255), "akaze": (0, 0, 255),
            "mind": (255, 255, 0), "fuse": (255, 255, 255),
            "prod": (0, 215, 255), "prod_mind": (180, 105, 255)}


def _bucket(rank):
    """실패 분류(문서 §2): 정답의 baseline top-K 순위 → 버킷."""
    if rank is None:
        return "proposer_miss"
    return "rank1_ok" if rank == 1 else "rank_error"


def _new_cell():
    return {"n": 0, "b0": 0, "raw": 0, "ref": 0, "changed": 0, "fallback": 0,
            "promote": 0, "regress": 0, "sub_n": 0, "sub_b0": 0, "sub_ref": 0,
            "n_cand": 0, "n_ok": 0, "rt_ms": 0.0, "rej": Counter(),
            "err_b0": [], "err_ref": []}


class _RegAccum:
    """per-point hook 본체 — `_consensus_template_ab` 의 combined_renderer 로 꽂힌다.

    hook 예외는 harness 가 삼키지만(경고 후 계속) 그러면 그 점이 조용히 빠지므로,
    내부에서도 전부 잡아 n_hook_err 로 계수한다. 측정 자체(gc/gr)는 hook 과 무관하게
    harness 가 이미 끝냈으므로 hook 실패가 B0 수치를 오염시키지 않는다.
    """

    def __init__(self, rows_fh, overlay_dir, arms, fuse=False, prod=False):
        self.rows_fh = rows_fh
        self.overlay_dir = overlay_dir
        self.arms = arms
        self.fuse = fuse
        self.prod = prod
        self.report_arms = list(arms) + (["fuse"] if fuse else [])
        if prod:
            self.report_arms.append("prod")
            if "mind" in arms:
                self.report_arms.append("prod_mind")
        self.cells = defaultdict(_new_cell)          # {(arm, mod): cell}
        self.per_recipe = defaultdict(lambda: defaultdict(
            lambda: {"n": 0, "b0": 0, "arm": 0}))    # [arm][recipe]
        self.buckets = defaultdict(Counter)          # [mod][bucket]
        self.n_points = 0
        self.n_mismatch = 0
        self.n_overlay = 0
        self.n_hook_err = 0

    def __call__(self, ctx):
        try:
            self._process(ctx)
        except Exception as exc:
            self.n_hook_err += 1
            print(f"[WARNING] reg hook 실패 {ctx.get('recipe')}/{getattr(ctx.get('path'), 'name', '?')}: {exc}")

    def _process(self, ctx):
        gc = ctx.get("gc")
        cons_tpl = ctx.get("cons_tpl")
        if gc is None or cons_tpl is None:
            return
        gray, gt, mod, rec = ctx["gray"], ctx["xy"], ctx["mod"], ctx["recipe"]
        tpl_img = cons_tpl.raw_image
        th, tw = tpl_img.shape[:2]
        short = max(1, min(tw, th))
        tol_px = GT_TOL_NORM * short

        # proposer 재실행(gc 와 동일 인자·동일 함수 → 결정적 동일 후보; scale 복원 목적).
        frame_dt = None if USE_ENSEMBLE_PROPOSER else preprocess_for_matching(gray)[1]
        cands = _propose_topk(cons_tpl, gray, frame_dt, scales=COMPARE_SCALES,
                              topk=TOPK_CANDIDATES)
        if not cands:
            return
        gc_xys = gc.get("cand_xys") or []
        if gc_xys and [int(cands[0].xy[0]), int(cands[0].xy[1])] != list(gc_xys[0]):
            self.n_mismatch += 1

        ox, oy = getattr(cons_tpl, "align_offset_xy", (0, 0)) or (0, 0)
        scales = [float(getattr(c, "scale", 1.0) or 1.0) for c in cands]
        base_pts = [(c.xy[0] + ox * sc, c.xy[1] + oy * sc)
                    for c, sc in zip(cands, scales)]
        base_d = [math.hypot(p[0] - gt[0], p[1] - gt[1]) for p in base_pts]
        base_rank = next((i + 1 for i, d in enumerate(base_d) if d <= tol_px), None)
        bucket = _bucket(base_rank)
        b0_hit = base_rank == 1
        self.buckets[mod][bucket] += 1
        self.n_points += 1

        row = {"recipe": rec, "mod": mod, "msr": ctx["path"].name,
               "gt": [round(float(gt[0]), 1), round(float(gt[1]), 1)], "short": short,
               "n_cand": len(cands), "base_rank": base_rank,
               "gc_rank": gc.get("topk_rank"), "bucket": bucket, "arms": {}}
        changed_any = False
        arm_tops = {}
        arm_runs = {}    # {arm: (order, fallback, verdicts)} — fuse 재료.
        for arm in self.arms:
            verdicts = [reg.refine_candidate(arm, tpl_img, gray, tuple(c.xy), sc)
                        for c, sc in zip(cands, scales)]
            order, fallback = reg.rerank_by_verdicts(verdicts)
            arm_runs[arm] = (order, fallback, verdicts)
            top = order[0]
            v = verdicts[top]
            if v.ok and v.refined_xy is not None:
                sc = scales[top]
                ref_pt = (v.refined_xy[0] + ox * sc, v.refined_xy[1] + oy * sc)
            else:
                ref_pt = base_pts[top]
            entry = self._tally(arm, mod, rec, top, fallback, ref_pt,
                                base_pts, base_d, gt, tol_px, b0_hit, bucket)
            st = self.cells[(arm, mod)]
            st["n_cand"] += len(cands)
            st["n_ok"] += sum(1 for x in verdicts if x.ok)
            st["rt_ms"] += sum(x.runtime_ms for x in verdicts)
            st["rej"].update(x.reject_reason for x in verdicts if not x.ok)
            entry.update({
                "score": None if v.score is None else round(float(v.score), 4),
                "shift_px": v.shift_frame_px,
                "n_ok": sum(1 for x in verdicts if x.ok),
                "rejects": dict(Counter(x.reject_reason for x in verdicts if not x.ok)),
            })
            row["arms"][arm] = entry
            changed_any = changed_any or entry["changed"]
            arm_tops[arm] = ref_pt

        # fuse 의사-arm: fallback 이 아닌 arm 들의 재정렬 순열을 RRF 합의로 융합.
        if self.fuse:
            live = [order for order, fb, _ in arm_runs.values() if not fb]
            if live:
                forder = reg.rrf_fuse_orders(live, len(cands))
                top, fallback = forder[0], False
            else:
                top, fallback = 0, True     # 전 arm 거부 → B0 강등(단일 arm 과 동일 규약).
            # 융합 top 의 정밀 좌표 = 그 후보를 ok 로 판정한 shift arm 들의 refined 평균.
            pts = []
            for arm in self.arms:
                if arm in reg.SCORE_ONLY_ARMS:
                    continue
                v = arm_runs[arm][2][top]
                if v.ok and v.refined_xy is not None:
                    sc = scales[top]
                    pts.append((v.refined_xy[0] + ox * sc, v.refined_xy[1] + oy * sc))
            ref_pt = (sum(p[0] for p in pts) / len(pts),
                      sum(p[1] for p in pts) / len(pts)) if pts else base_pts[top]
            entry = self._tally("fuse", mod, rec, top, fallback, ref_pt,
                                base_pts, base_d, gt, tol_px, b0_hit, bucket)
            entry["n_arms_live"] = len(live)
            row["arms"]["fuse"] = entry
            changed_any = changed_any or entry["changed"]
            arm_tops["fuse"] = ref_pt

        # production selection 재현 의사-arm 'prod' — 운영 NCC rerank(engine 과 동일 식)로
        # 후보를 고른다. B0(RRF 순서)는 운영의 실제 선택이 아니므로, mind 의 이득이 NCC 와
        # 겹치는지(prod 만으로 충분) 직교인지(prod_mind > prod)는 이 arm 이 있어야 판별된다.
        if self.prod:
            fdt = frame_dt if frame_dt is not None else preprocess_for_matching(gray)[1]
            rescored = _rescore_positions_to_candidates(
                cons_tpl, fdt, [(tuple(c.xy), sc) for c, sc in zip(cands, scales)])
            sels = []
            for rc in rescored:
                ncc = _candidate_ncc(cons_tpl.raw_image, gray, rc.xy, rc.scale)
                ncc_pos = max(0.0, ncc) if ncc is not None else 0.0
                sels.append(DEFAULT_POLICY.rerank_chamfer_w * rc.chamfer_score
                            + DEFAULT_POLICY.rerank_ncc_w * ncc_pos)
            prod_order = sorted(range(len(cands)), key=lambda i: (-sels[i], i))
            top = prod_order[0]
            entry = self._tally("prod", mod, rec, top, False, base_pts[top],
                                base_pts, base_d, gt, tol_px, b0_hit, bucket)
            entry["sel"] = round(float(sels[top]), 4)
            row["arms"]["prod"] = entry
            changed_any = changed_any or entry["changed"]
            arm_tops["prod"] = base_pts[top]

            # 포팅 후보안 'prod_mind' — prod 순서와 mind 재정렬 순서의 RRF 결합(순위 기반,
            # sel/mind score 는 척도가 달라 직접 합산 불가). mind 가 전 후보 거부면 prod 단독.
            if "mind" in arm_runs:
                morder, mfb, _ = arm_runs["mind"]
                forder = (reg.rrf_fuse_orders([prod_order, morder], len(cands))
                          if not mfb else prod_order)
                top = forder[0]
                entry = self._tally("prod_mind", mod, rec, top, False, base_pts[top],
                                    base_pts, base_d, gt, tol_px, b0_hit, bucket)
                entry["mind_fallback"] = mfb
                row["arms"]["prod_mind"] = entry
                changed_any = changed_any or entry["changed"]
                arm_tops["prod_mind"] = base_pts[top]

        self.rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if changed_any and self.overlay_dir is not None and self.n_overlay < OVERLAY_MAX:
            self._save_overlay(gray, gt, base_pts[0], arm_tops, rec, ctx["path"].name)

    def _tally(self, name, mod, rec, top, fallback, ref_pt,
               base_pts, base_d, gt, tol_px, b0_hit, bucket):
        """arm(또는 fuse 의사-arm) 하나의 per-point 집계 — cell/per_recipe 갱신 + row entry."""
        raw_pt = base_pts[top]
        d_raw = math.hypot(raw_pt[0] - gt[0], raw_pt[1] - gt[1])
        d_ref = math.hypot(ref_pt[0] - gt[0], ref_pt[1] - gt[1])
        hit_raw = d_raw <= tol_px          # 순위 효과만(원좌표) — refine 효과와 분해.
        hit_ref = d_ref <= tol_px          # 순위 + 정밀화(최종 성능).
        st = self.cells[(name, mod)]
        st["n"] += 1
        st["b0"] += b0_hit
        st["raw"] += hit_raw
        st["ref"] += hit_ref
        st["changed"] += (top != 0)
        st["fallback"] += fallback
        if b0_hit and not hit_ref:
            st["regress"] += 1
        if (not b0_hit) and hit_ref:
            st["promote"] += 1
        if bucket != "proposer_miss":      # verifier 가 고칠 수 있는 유일한 영역.
            st["sub_n"] += 1
            st["sub_b0"] += b0_hit
            st["sub_ref"] += hit_ref
        if b0_hit:
            st["err_b0"].append(base_d[0])
        if hit_ref:
            st["err_ref"].append(d_ref)
        pr = self.per_recipe[name][rec]
        pr["n"] += 1
        pr["b0"] += b0_hit
        pr["arm"] += hit_ref
        return {"top": top, "changed": top != 0, "fallback": fallback,
                "hit_raw": hit_raw, "hit_ref": hit_ref, "err_ref_px": round(d_ref, 1)}

    def _save_overlay(self, gray, gt, b0_pt, arm_tops, rec, msr_name):
        """top-1 이 바뀐 행의 검증용 overlay(문서 §4.3 false-positive 추적)."""
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(canvas, (int(gt[0]), int(gt[1])), _CLR_GT,
                       cv2.MARKER_CROSS, 26, 2)
        cv2.circle(canvas, (int(b0_pt[0]), int(b0_pt[1])), 12, _CLR_B0, 2)
        for arm, pt in arm_tops.items():
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 6,
                       _ARM_CLR.get(arm, (200, 200, 200)), 2)
        legend = "GT=green  B0=skyblue  " + "  ".join(
            f"{a}={'/'.join(map(str, _ARM_CLR.get(a, ())))}" for a in arm_tops)
        cv2.putText(canvas, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = self.overlay_dir / f"{rec.replace('/', '__')}__{Path(msr_name).stem}.jpg"
        cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.n_overlay += 1


def _rate(num, den):
    return round(num / den, 3) if den else None


def _median(xs):
    """중앙값(px, 1자리) — 빈 리스트면 None."""
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    val = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
    return round(val, 1)


def _arm_summary(acc, arm):
    """arm 하나의 overall/모드별 집계 + per-recipe paired bootstrap CI."""
    mods = sorted({m for (a, m) in acc.cells if a == arm})
    tot = _new_cell()
    per_mod = {}
    for m in mods:
        c = acc.cells[(arm, m)]
        for k in ("n", "b0", "raw", "ref", "changed", "fallback", "promote",
                  "regress", "sub_n", "sub_b0", "sub_ref", "n_cand", "n_ok", "rt_ms"):
            tot[k] += c[k]
        tot["rej"].update(c["rej"])
        tot["err_b0"] += c["err_b0"]
        tot["err_ref"] += c["err_ref"]
        per_mod[m] = {
            "n": c["n"], "b0_r1": _rate(c["b0"], c["n"]),
            "arm_r1_raw": _rate(c["raw"], c["n"]), "arm_r1_ref": _rate(c["ref"], c["n"]),
            "sub_n": c["sub_n"], "sub_b0_r1": _rate(c["sub_b0"], c["sub_n"]),
            "sub_arm_r1": _rate(c["sub_ref"], c["sub_n"]),
            "promote": c["promote"], "regress": c["regress"],
            "changed": c["changed"], "fallback": c["fallback"],
            "reject_rate": _rate(c["n_cand"] - c["n_ok"], c["n_cand"]),
            "rejects": dict(c["rej"].most_common()),
            "ms_per_point": _rate(c["rt_ms"], c["n"]),
            # hit 행의 GT 거리 중앙값(px): b0 는 원좌표, arm 은 정밀화 후 — sub-pixel 효과 축.
            "err_b0_med_px": _median(c["err_b0"]),
            "err_ref_med_px": _median(c["err_ref"]),
        }
    diffs = [(p["arm"] - p["b0"]) / p["n"]
             for p in acc.per_recipe[arm].values() if p["n"]]
    ci = gce._bootstrap_ci(diffs)
    return {
        "n": tot["n"], "b0_r1": _rate(tot["b0"], tot["n"]),
        "arm_r1_raw": _rate(tot["raw"], tot["n"]),
        "arm_r1_ref": _rate(tot["ref"], tot["n"]),
        "delta_ref": (None if tot["n"] == 0
                      else round((tot["ref"] - tot["b0"]) / tot["n"], 3)),
        "delta_ci95": [None if math.isnan(x) else round(x, 3) for x in ci],
        "n_recipes": len(diffs),
        "sub_n": tot["sub_n"], "sub_b0_r1": _rate(tot["sub_b0"], tot["sub_n"]),
        "sub_arm_r1": _rate(tot["sub_ref"], tot["sub_n"]),
        "promote": tot["promote"], "regress": tot["regress"],
        "changed": tot["changed"], "fallback": tot["fallback"],
        "reject_rate": _rate(tot["n_cand"] - tot["n_ok"], tot["n_cand"]),
        "rejects": dict(tot["rej"].most_common()),
        "ms_per_point": _rate(tot["rt_ms"], tot["n"]),
        "err_b0_med_px": _median(tot["err_b0"]),
        "err_ref_med_px": _median(tot["err_ref"]),
        "per_modality": per_mod,
    }


def _print_arm_table(arm_stats):
    print("\n" + "=" * 76)
    print("[INFO] === registration verifier A/B (B0 = consensus proposer top-1) ===")
    print(f"    {'arm':<11}{'mod':<6}{'n':>5}  {'B0_r1':>6} {'raw':>6} {'ref':>6} "
          f"{'d_ref':>6}  {'promo/reg':>9} {'rej%':>5} {'ms/pt':>7} {'err_med':>9}")
    for arm, s in arm_stats.items():
        rows = [("all", s)] + sorted(s["per_modality"].items())
        for mod, r in rows:
            d = (r.get("delta_ref") if mod == "all"
                 else (None if r["arm_r1_ref"] is None or r["b0_r1"] is None
                       else round(r["arm_r1_ref"] - r["b0_r1"], 3)))
            err = f"{r['err_b0_med_px']}>{r['err_ref_med_px']}"
            print(f"    {arm:<11}{mod:<6}{r['n']:>5}  "
                  f"{str(r['b0_r1']):>6} {str(r['arm_r1_raw']):>6} {str(r['arm_r1_ref']):>6} "
                  f"{str(d):>6}  {r['promote']:>4}/{r['regress']:<4} "
                  f"{str(r['reject_rate']):>5} {str(r['ms_per_point']):>7} {err:>9}")
    print("=" * 76)
    print("  raw = 순위 효과만(원좌표) / ref = 순위+정밀화(최종). d_ref = ref - B0.")
    print("  promo = B0 miss -> arm hit / reg = B0 hit -> arm miss (reg 가 안전성 지표).")
    print("  err_med = hit 행 GT 거리 중앙값 px, B0>arm — sub-pixel 정밀화 효과 축.")


def _print_subset_table(arm_stats):
    print("\n[INFO] === gt_in_topk 부분집합 (verifier 가 고칠 수 있는 유일한 영역) ===")
    print(f"    {'arm':<7}{'sub_n':>6}  {'B0_r1':>6} {'arm_r1':>7}")
    for arm, s in arm_stats.items():
        print(f"    {arm:<7}{s['sub_n']:>6}  {str(s['sub_b0_r1']):>6} {str(s['sub_arm_r1']):>7}")
    print("  이 표가 오르지 않으면 개선은 우연(전체 평균의 버킷 구성 차이)이다 — 문서 §P0-A.")


def _digest_line(acc, arm_stats, n_sampled, n_total):
    om = sum(acc.buckets.get("om", Counter()).values())
    sem = sum(acc.buckets.get("sem", Counter()).values())
    bt = Counter()
    for c in acc.buckets.values():
        bt.update(c)
    n = max(1, acc.n_points)
    parts = [f"reg recipes={n_sampled}/{n_total} pts={acc.n_points} (om={om} sem={sem})",
             f"buckets miss={bt['proposer_miss'] / n:.2f} "
             f"rank_err={bt['rank_error'] / n:.2f} rank1={bt['rank1_ok'] / n:.2f}"]
    for arm, s in arm_stats.items():
        ci = s["delta_ci95"]
        ci_s = (f"ci[{ci[0]:+.3f},{ci[1]:+.3f}]"
                if ci and ci[0] is not None else "ci[nan]")
        # modality별 델타 — SEM/OM 어느 쪽에서 이득이 나는지 digest 만으로 판독.
        mod_parts = []
        for m, r in sorted(s["per_modality"].items()):
            if r["arm_r1_ref"] is not None and r["b0_r1"] is not None:
                mod_parts.append(f"{m}{r['arm_r1_ref'] - r['b0_r1']:+.3f}")
        mod_s = f" [{'/'.join(mod_parts)}]" if mod_parts else ""
        parts.append(f"{arm} r1={s['arm_r1_ref']} d={s['delta_ref']:+.3f}{mod_s} {ci_s} "
                     f"p/r={s['promote']}/{s['regress']}"
                     if s["delta_ref"] is not None else f"{arm} n=0")
    if acc.n_mismatch:
        parts.append(f"MISMATCH={acc.n_mismatch}")
    if acc.n_hook_err:
        parts.append(f"HOOK_ERR={acc.n_hook_err}")
    return "[DIGEST] " + " | ".join(parts)


def run() -> str:
    """registration verifier A/B 실행 (인자 없음). 반환: success | no_data | no_rows."""
    bad = [a for a in ARMS if a not in reg.REG_ARM_NAMES]
    if bad:
        print(f"[ERROR] 알 수 없는 arm: {bad} (가능: {reg.REG_ARM_NAMES})")
        return "no_data"

    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = [a for a in (gle._collect_recipes(root) if root.is_dir() else [])
               if a is not None]
    if not recipes:
        print(f"[WARNING] golden 데이터를 찾지 못했습니다: {root} "
              f"(env ALIGN_GOLDEN_ROOT 로 경로 지정). cond 실데이터 전용 드라이버 - "
              f"합성 검증은 registration_lab.py self-test 를 실행.")
        return "no_data"

    recipes.sort(key=gce._recipe_key)     # 파일시스템 순서 제거 → 표본이 머신 간 결정적.
    n_total = len(recipes)
    if SAMPLE_N and SAMPLE_N < n_total:
        recipes = random.Random(SAMPLE_SEED).sample(recipes, SAMPLE_N)
        recipes.sort(key=gce._recipe_key)

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] (registration A/B) recipe {len(recipes)}/{n_total}개 "
          f"(sample_n={SAMPLE_N or '전체'}, seed={SAMPLE_SEED}) → {out_dir}")
    print(f"[INFO] arms={','.join(ARMS)}{'+fuse' if FUSE_ON else ''}"
          f"{'+prod' if PROD_ARM else ''}"
          f"  topk={TOPK_CANDIDATES}  tol={GT_TOL_NORM}"
          f"  proposer={'ensemble' if USE_ENSEMBLE_PROPOSER else 'c1'}"
          f"  clean_frame={'ON' if gce.CLEAN_FRAME else 'OFF'}"
          f"  min_s={gce.CONSENSUS_MIN_S}")
    print("[INFO] 비용 주의: point 당 proposer 2회(측정 1 + scale 복원 1) + arm 별 후보 검증.")

    by_recipe = {}
    for assets in recipes:
        try:
            center_tpls, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        if not any(v is not None for v in center_tpls.values()):
            continue
        entry = gce._build_cond_by_recipe(assets, center_tpls, box_tpls)
        if entry["s_frames"]:
            by_recipe[gce._recipe_key(assets)] = entry
    print(f"[INFO] S(crosshair) 있는 recipe: {len(by_recipe)}개")
    if not by_recipe:
        print("[ERROR] 평가 가능한 recipe 가 없습니다.")
        return "no_rows"

    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        acc = _RegAccum(fh, overlay_dir, ARMS, fuse=FUSE_ON, prod=PROD_ARM)
        res = _consensus_template_ab(
            by_recipe, min_s=gce.CONSENSUS_MIN_S, out_dir=out_dir,
            combined_renderer=acc,
            frame_loader=gce._cleaned_frame_loader if gce.CLEAN_FRAME else None)
    if res is None or acc.n_points == 0:
        print("[ERROR] LOO 가능한 점이 없습니다 (min_s 미달 또는 hook 미호출).")
        return "no_rows"
    res.pop("consensus_points", None)     # raw 점 목록은 summary 에 불필요(용량).

    # === B0 anchor (harness 자체 집계 — 이 표본의 consensus/rcp 기준선) ===
    print(f"\n[INFO] === B0 anchor (harness, 표본 {res['n_recipes']} recipe / "
          f"{res['n_S_loo']} pts) ===")
    print(f"  cons in_topk={res['overall_cons_in_topk_rate']} "
          f"rank1={res['overall_cons_rank1_rate']}  |  "
          f"rcp in_topk={res['overall_rcp_in_topk_rate']} "
          f"rank1={res['overall_rcp_rank1_rate']}")

    # === 실패 분류(문서 §2 — verifier 기대 상한 산정) ===
    print("\n[INFO] === 실패 분류 (baseline top-K 기준) ===")
    for mod in sorted(acc.buckets):
        c = acc.buckets[mod]
        n = sum(c.values())
        print(f"    {mod:<5} n={n:<5} miss={c['proposer_miss']}({c['proposer_miss'] / n:.2f}) "
              f"rank_err={c['rank_error']}({c['rank_error'] / n:.2f}) "
              f"rank1={c['rank1_ok']}({c['rank1_ok'] / n:.2f})")

    arm_stats = {arm: _arm_summary(acc, arm) for arm in acc.report_arms}
    _print_arm_table(arm_stats)
    _print_subset_table(arm_stats)

    if acc.n_mismatch:
        print(f"\n[WARNING] proposer 재실행 top-1 불일치 {acc.n_mismatch}점 - "
              f"비결정성 의심(결과 해석 주의).")
    if acc.n_hook_err:
        print(f"[WARNING] hook 예외 {acc.n_hook_err}점 - rows.jsonl 에서 빠짐.")

    digest = _digest_line(acc, arm_stats, len(recipes), n_total)
    summary = {
        "config": {"sample_n": SAMPLE_N, "sample_seed": SAMPLE_SEED, "arms": list(ARMS),
                   "fuse": FUSE_ON, "prod_arm": PROD_ARM,
                   "topk": TOPK_CANDIDATES, "gt_tol_norm": GT_TOL_NORM,
                   "proposer": "ensemble" if USE_ENSEMBLE_PROPOSER else "c1",
                   "clean_frame": gce.CLEAN_FRAME, "min_s": gce.CONSENSUS_MIN_S,
                   "n_recipes_sampled": len(recipes), "n_recipes_total": n_total},
        "n_points": acc.n_points,
        "n_mismatch": acc.n_mismatch,
        "n_hook_err": acc.n_hook_err,
        "buckets": {m: dict(c) for m, c in acc.buckets.items()},
        "arms": arm_stats,
        "harness_b0": {k: res[k] for k in
                       ("n_recipes", "n_S_loo", "overall_cons_in_topk_rate",
                        "overall_cons_rank1_rate", "overall_rcp_in_topk_rate",
                        "overall_rcp_rank1_rate") if k in res},
        "digest": digest,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "digest.txt").write_text(digest + "\n", encoding="utf-8")
    print("\n" + digest)
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    _status = run()
    raise SystemExit(0 if _status in ("success", "no_data") else 1)
