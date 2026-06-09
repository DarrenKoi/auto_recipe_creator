"""회귀 프레임 진단 — localization e2e A/B 의 only_base(baseline hit·ensemble miss) 프레임을
ensemble 내부 pool 까지 까서 *왜 reranker 가 진실을 못 골랐나*를 분류한다.

주의: 이 진단은 *ORB-selection 시절* ensemble 회귀 분류용이다(orb_flip 카테고리). production 은
2026-06-09 부터 NCC reranker selection — _ensemble_pool 은 ORB-selection 을 재현하므로
(reranker_rule_ab 의 ens_orb 비교 기준과 동일) 현재 production 픽이 아니라 ORB-시절 픽을 분류한다.

배경: e2e A/B(localization_ab_ensemble)에서 ensemble proposer recall@8=0.698 인데 최종 hit 는
0.407(baseline 0.422 보다 낮음). 진실이 pool 에 70% 있는데 못 고름 → 병목이 reranker. 이 러너는
각 회귀(only_base) 프레임에서 ensemble 후보 pool 의 per-candidate (chamfer, orb, combined, GT거리)를
재현해 회귀를 3분류:
- proposer_miss : 진실이 pool 에 없음(0.698 의 잔여 미스).
- orb_flip      : chamfer-top 은 정답인데 ORB(combined)가 decoy 로 뒤집음 → ORB 가 유해.
- chamfer_miss  : 진실이 pool 엔 있으나 chamfer 가 decoy 를 1위로 → chamfer 직교 reranker 필요.
대조용으로 only_ens(ensemble 만 hit) 프레임도 같은 pool 을 기록한다.

입력: localization_ab_ensemble 의 최신 rows.jsonl(또는 ALIGN_DIAG_ROWS 로 경로 지정).
출력: diag/<ts>/{diag.jsonl, summary.json, overlays/*.jpg}.
실행(오피스): uv run python poc/workflow_2/localization_regression_diag.py
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

import cv2

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_3.vision.align_key_matcher import (
    STRUCTURE_POLICY,
    _crop_with_padding,
    _rescore_positions_to_candidates,
    compute_align_key_score,
    compute_orb_inlier_ratio,
    preprocess_for_matching,
)
# 소스에서 직접 — align_key_matcher.compute_ensemble_candidates 는 lazy placeholder(None)다.
from poc.workflow_3.vision.ensemble_proposer import compute_ensemble_candidates
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_3.vision.align_point_correction import _tool_label
from poc.workflow_2.localization_ab_ensemble import (
    OUTPUT_ROOT as AB_ROOT, _err_norm, _predicted_align_point,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "localization_regression_diag"


def _classify_regression(pool, *, picked_idx, tol):
    """ensemble pool(후보별 err·chamfer) + picked 후보로 회귀 원인 분류.

    proposer_miss : 진실(err<=tol)이 pool 에 없음.
    orb_flip      : chamfer-top 이 within tol 인데 picked≠chamfer_top → combined(ORB)가 뒤집음.
    chamfer_miss  : 진실은 pool 에 있으나 chamfer-top 이 within tol 아님 → chamfer 변별 실패.
    other         : chamfer-top within tol & picked==chamfer_top(=hit, 회귀 아님 — 방어적).
    """
    truth = [i for i, c in enumerate(pool) if c["err"] <= tol]
    if not truth:
        return "proposer_miss"
    chamfer_top = max(range(len(pool)), key=lambda i: pool[i]["chamfer"])
    if pool[chamfer_top]["err"] <= tol:
        return "orb_flip" if picked_idx != chamfer_top else "other"
    return "chamfer_miss"


def _latest_rows():
    """localization_ab_ensemble 최신 run 의 rows.jsonl 경로(없으면 None)."""
    env = os.getenv("ALIGN_DIAG_ROWS")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    cands = sorted(AB_ROOT.glob("*/rows.jsonl"))
    return cands[-1] if cands else None


def _load_flagged(rows_path):
    """rows.jsonl → (only_base, only_ens) 각각 {(recipe, msr)} 집합."""
    only_base, only_ens = set(), set()
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["recipe"], r["msr"])
            if r["hit_base"] and not r["hit_ens"]:
                only_base.add(key)
            elif r["hit_ens"] and not r["hit_base"]:
                only_ens.add(key)
    return only_base, only_ens


def _ensemble_pool(tpl, frame, frame_dt, offset, gt_xy, short, policy):
    """compute_align_key_score_ensemble 내부와 동일하게 pool 재현 → (pool, picked_idx).

    pool[i] = {xy, scale, chamfer, orb, combined, err(align-point 정규화 거리)}.
    """
    ens = compute_ensemble_candidates(
        tpl.raw_image, frame, scales=COMPARE_SCALES, top_n=policy.top_n)
    # production(compute_align_key_score_ensemble)과 정합: ens.fused 를 top_n 으로 먼저 cap.
    # pool 은 어차피 cands[:top_n] 이라 출력 동일(rescore 순서 보존), shadow rescore 비용만 절약.
    # 주: production 은 top_n 전부 chamfer 0 이면 no_candidates 로 종료하나 여기선 그대로 pool 을
    # 만든다 — calib 에선 그 프레임이 sel≈0.5·ncc(<match) 의 true-negative 로 들어가 threshold 에
    # 무해하므로 별도 guard 는 두지 않는다.
    cands = _rescore_positions_to_candidates(
        tpl, frame_dt, [(c.xy, c.scale) for c in ens.fused[:policy.top_n]])
    pool = []
    for c in cands:
        cx, cy = c.xy
        tw, th = c.template_size
        ch = c.chamfer_score
        orb = 0.0
        if ch > 0.0 and tw > 0 and th > 0:
            crop, _o = _crop_with_padding(frame, cx, cy, tw, th, pad=1.6)
            orb, _i, _m = compute_orb_inlier_ratio(tpl.raw_image, crop)
        combined = policy.chamfer_weight * ch + policy.orb_weight * orb
        err = _err_norm((cx + offset[0], cy + offset[1]), gt_xy, short)
        # chamfer 는 6 자리 — reranker_rule_ab 캘리브가 c["chamfer"] 로 sel 을 재산출하므로
        # production full-precision chamfer 에 가깝게 둬야 sel 분포가 일치(4 자리는 컷을 흔들 수 있음).
        pool.append({"xy": [int(cx), int(cy)], "scale": round(float(c.scale), 3),
                     "chamfer": round(float(ch), 6), "orb": round(float(orb), 4),
                     "combined": round(float(combined), 4), "err": round(float(err), 4)})
    picked_idx = max(range(len(pool)), key=lambda i: pool[i]["combined"]) if pool else -1
    return pool, picked_idx


def _save_overlay(frame, gt_xy, base_xy, ens_xy, pool, offset, out_path):
    """진단 오버레이 — GT(초록)·baseline(파랑)·ensemble(주황)·pool 후보 align point(회색)."""
    canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for c in pool:
        ax, ay = c["xy"][0] + offset[0], c["xy"][1] + offset[1]
        cv2.circle(canvas, (int(ax), int(ay)), 4, (160, 160, 160), 1)
    cv2.drawMarker(canvas, (int(gt_xy[0]), int(gt_xy[1])), (0, 200, 0),
                   cv2.MARKER_CROSS, 28, 2)
    cv2.circle(canvas, (int(base_xy[0]), int(base_xy[1])), 9, (220, 120, 0), 2)
    cv2.circle(canvas, (int(ens_xy[0]), int(ens_xy[1])), 6, (0, 140, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def run():
    """회귀 프레임 진단 (인자 없음). 반환 success|no_data."""
    rows_path = _latest_rows()
    if rows_path is None:
        print("[ERROR] rows.jsonl 없음 — 먼저 localization_ab_ensemble 실행 "
              "(또는 ALIGN_DIAG_ROWS 지정).")
        return "no_data"
    only_base, only_ens = _load_flagged(rows_path)
    flagged = {"only_base": only_base, "only_ens": only_ens}
    want = only_base | only_ens
    want_recipes = {r for r, _m in want}
    print(f"[INFO] rows: {rows_path}")
    print(f"[INFO] 진단 대상 — only_base(회귀) {len(only_base)} · only_ens(이득) {len(only_ens)} "
          f"= {len(want)} 프레임, {len(want_recipes)} recipe")
    if not want:
        print("[INFO] 불일치 프레임 없음 — 진단할 것 없음.")
        return "no_data"

    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터 없음: {root}")
        return "no_data"
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = STRUCTURE_POLICY
    diag_rows = []
    # 회귀(only_base) 분류 카운트 — 다음 fix 를 가리킴.
    cls = {"proposer_miss": 0, "orb_flip": 0, "chamfer_miss": 0, "other": 0}
    processed = 0
    for assets in recipes:
        if assets is None or assets.recipe_id not in want_recipes:
            continue
        try:
            _center, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        available = {m for m, v in box_tpls.items() if v is not None}
        if not available:
            continue
        for p in iter_msr_images(assets):
            key = (assets.recipe_id, p.name)
            kind = ("only_base" if key in only_base
                    else "only_ens" if key in only_ens else None)
            if kind is None:
                continue
            cond = load_cond(p)
            routed = glec._route_modality(cond, available)
            if routed is None or box_tpls.get(routed) is None:
                continue
            tpl, (dx, dy) = box_tpls[routed]
            if not (cond and cond.crosshair_xy is not None):
                continue
            gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
            gt_xy = (int(round(gx)), int(round(gy)))
            try:
                frame = clean_image(load_gray(p), cond)
            except Exception:
                continue
            _edges, frame_dt = preprocess_for_matching(frame)
            short = max(1, min(tpl.raw_image.shape[0], tpl.raw_image.shape[1]))

            res_b = compute_align_key_score(
                tpl, frame, scales=COMPARE_SCALES, policy=policy)
            base_pred = _predicted_align_point(res_b, (dx, dy))
            base_err = _err_norm(base_pred, gt_xy, short)
            pool, picked_idx = _ensemble_pool(
                tpl, frame, frame_dt, (dx, dy), gt_xy, short, policy)
            if not pool:
                continue
            ens_pick = pool[picked_idx]
            ens_pred = (ens_pick["xy"][0] + dx, ens_pick["xy"][1] + dy)

            category = ""
            if kind == "only_base":
                category = _classify_regression(pool, picked_idx=picked_idx, tol=GT_TOL_NORM)
                cls[category] += 1

            row = {"recipe": assets.recipe_id, "msr": p.name, "modality": routed,
                   "kind": kind, "category": category,
                   "gt": [gt_xy[0], gt_xy[1]],
                   "base": {"pred": [int(base_pred[0]), int(base_pred[1])],
                            "err": round(float(base_err), 4),
                            "chamfer": round(float(res_b.chamfer_score), 4),
                            "orb": round(float(res_b.orb_inlier_ratio), 4)},
                   "ens": {"pred": [int(ens_pred[0]), int(ens_pred[1])],
                           "picked_idx": picked_idx, "err": ens_pick["err"]},
                   "pool": pool}
            diag_rows.append(row)
            _save_overlay(frame, gt_xy, base_pred, ens_pred, pool, (dx, dy),
                          out_dir / "overlays" / f"{kind}_{assets.recipe_id}_{p.stem}.jpg")
            processed += 1
            if processed % 10 == 0:
                print(f"[INFO] 진행 {processed}/{len(want)}")

    if not diag_rows:
        print("[ERROR] 재해결된 진단 프레임 없음.")
        return "no_data"

    with (out_dir / "diag.jsonl").open("w", encoding="utf-8") as fh:
        for r in diag_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_base = sum(1 for r in diag_rows if r["kind"] == "only_base")
    summary = {"rows_src": str(rows_path), "processed": processed,
               "only_base": n_base, "only_ens": len(diag_rows) - n_base,
               "regression_class": cls, "GT_TOL_NORM": GT_TOL_NORM}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[INFO] === 회귀(only_base) 분류 (n={n_base}) ===")
    for k in ("proposer_miss", "orb_flip", "chamfer_miss", "other"):
        share = f"{cls[k] / n_base:.2f}" if n_base else "-"
        print(f"  {k:<14} {cls[k]:>3}  ({share})")
    print("  해석: orb_flip 多→ORB 유해(게이트/가중↓) / chamfer_miss 多→chamfer 직교 reranker "
          "/ proposer_miss 多→proposer 잔여")
    print(f"[INFO] 완료: {out_dir} (diag.jsonl, overlays/)")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
