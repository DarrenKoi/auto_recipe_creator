"""reranker 신호 프로브 — chamfer_miss 회귀 프레임에서 *어떤 chamfer-직교 신호가 진실 후보를
picked-decoy 위로 올리는가*를 정량화한다. "어떤 reranker 를 만들지"를 추측 아닌 데이터로 정한다.

배경: localization_regression_diag 결과 회귀의 41%가 chamfer_miss(진실이 pool 엔 있으나 chamfer 가
decoy 를 1위로). chamfer 는 포화 → 직교 신호 필요. 후보 신호(전부 CV, 설계규칙 준수):
- MI(mutual information): intensity 단조 재맵핑에 강건 — 공정 변화로 밝기 달라진 align-key 에 적합.
- box-crop NCC: 정규화 상호상관(메모리상 픽셀 동일성 가정이라 약함 — 대조군).
각 chamfer_miss 프레임에서 진실-대표 후보 vs picked-decoy 의 MI·NCC 를 계산해, 신호가 진실을 위로
올리는지(separates: signal(truth) > signal(decoy)) 집계. 분리율 높은 신호가 reranker 후보.

입력: localization_regression_diag 최신 diag.jsonl(또는 ALIGN_PROBE_DIAG 로 경로 지정).
출력: probe/<ts>/{probe.jsonl, summary.json}.
실행(오피스): uv run python poc/workflow_2/reranker_signal_probe.py
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

import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_3.vision.align_key_matcher import (
    STRUCTURE_POLICY, _frame_patch, _ncc, _resize_template, preprocess_for_matching,
)
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_2.localization_regression_diag import (
    OUTPUT_ROOT as DIAG_ROOT, _classify_regression, _ensemble_pool,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "reranker_signal_probe"

SIGNALS = ("mi", "ncc")


def _mi(a, b, bins=32):
    """두 동일 크기 패치의 상호정보량(nats, ≥0). 결합 히스토그램 기반. 한쪽 상수면 0."""
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    hist, _xe, _ye = np.histogram2d(av, bv, bins=bins)
    total = hist.sum()
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    px_py = px[:, None] * py[None, :]
    return float((pxy[nz] * np.log(pxy[nz] / px_py[nz])).sum())


def _candidate_signals(tpl_raw, frame, cand):
    """후보(xy·scale)에서 chamfer-직교 신호 — {mi, ncc}. 패치 추출 불가 시 None."""
    cx, cy = cand["xy"]
    tpl_r = _resize_template(tpl_raw, cand["scale"])
    th, tw = tpl_r.shape[:2]
    patch = _frame_patch(frame, cx, cy, tw, th)
    if patch is None or patch.shape != tpl_r.shape:
        return None
    return {"mi": _mi(tpl_r, patch), "ncc": _ncc(tpl_r, patch)}


def _latest_diag():
    """localization_regression_diag 최신 run 의 diag.jsonl 경로(없으면 None)."""
    env = os.getenv("ALIGN_PROBE_DIAG")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    cands = sorted(DIAG_ROOT.glob("*/diag.jsonl"))
    return cands[-1] if cands else None


def _chamfer_miss_keys(diag_path):
    """diag.jsonl → category=='chamfer_miss' 인 {(recipe, msr)} 집합."""
    keys = set()
    with diag_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("category") == "chamfer_miss":
                keys.add((r["recipe"], r["msr"]))
    return keys


def run():
    """chamfer_miss 프레임에서 직교 신호 분리율 프로브 (인자 없음). 반환 success|no_data."""
    diag_path = _latest_diag()
    if diag_path is None:
        print("[ERROR] diag.jsonl 없음 — 먼저 localization_regression_diag 실행 "
              "(또는 ALIGN_PROBE_DIAG 지정).")
        return "no_data"
    keys = _chamfer_miss_keys(diag_path)
    want_recipes = {r for r, _m in keys}
    print(f"[INFO] diag: {diag_path}")
    print(f"[INFO] chamfer_miss 대상 {len(keys)} 프레임 ({len(want_recipes)} recipe)")
    if not keys:
        print("[INFO] chamfer_miss 프레임 없음 — 프로브할 것 없음.")
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

    probe_rows = []
    # 신호별 분리 카운트 + margin(truth-decoy) 리스트.
    sep = {s: 0 for s in SIGNALS}
    margins = {s: [] for s in SIGNALS}
    n = 0
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
            if (assets.recipe_id, p.name) not in keys:
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
            pool, picked_idx = _ensemble_pool(
                tpl, frame, frame_dt, (dx, dy), gt_xy, short, policy)
            if not pool:
                continue
            # 재현된 pool 로 분류 확인 — chamfer_miss 가 아니면 스킵(경계 변동).
            if _classify_regression(pool, picked_idx=picked_idx, tol=GT_TOL_NORM) != "chamfer_miss":
                continue
            truth_idxs = [i for i, c in enumerate(pool) if c["err"] <= GT_TOL_NORM]
            if not truth_idxs:
                continue
            truth_idx = min(truth_idxs, key=lambda i: pool[i]["err"])
            decoy_idx = picked_idx
            sig_t = _candidate_signals(tpl.raw_image, frame, pool[truth_idx])
            sig_d = _candidate_signals(tpl.raw_image, frame, pool[decoy_idx])
            if sig_t is None or sig_d is None:
                continue

            row = {"recipe": assets.recipe_id, "msr": p.name, "modality": routed,
                   "truth_idx": truth_idx, "decoy_idx": decoy_idx,
                   "truth": {"err": pool[truth_idx]["err"], "chamfer": pool[truth_idx]["chamfer"],
                             **{s: round(sig_t[s], 4) for s in SIGNALS}},
                   "decoy": {"err": pool[decoy_idx]["err"], "chamfer": pool[decoy_idx]["chamfer"],
                             **{s: round(sig_d[s], 4) for s in SIGNALS}}}
            for s in SIGNALS:
                margin = sig_t[s] - sig_d[s]
                margins[s].append(margin)
                row[f"{s}_separates"] = bool(margin > 0)
                if margin > 0:
                    sep[s] += 1
            probe_rows.append(row)
            n += 1

    if not n:
        print("[ERROR] 프로브된 chamfer_miss 프레임 없음.")
        return "no_data"

    with (out_dir / "probe.jsonl").open("w", encoding="utf-8") as fh:
        for r in probe_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _median(v):
        return round(float(np.median(v)), 4) if v else None

    summary = {"diag_src": str(diag_path), "n": n, "GT_TOL_NORM": GT_TOL_NORM,
               "separates": {s: sep[s] for s in SIGNALS},
               "separate_rate": {s: round(sep[s] / n, 3) for s in SIGNALS},
               "median_margin": {s: _median(margins[s]) for s in SIGNALS}}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[INFO] === reranker 신호 프로브 (chamfer_miss n={n}, tol={GT_TOL_NORM}) ===")
    print(f"  {'signal':<8} {'separates(truth>decoy)':>22} {'median_margin':>14}")
    for s in SIGNALS:
        print(f"  {s:<8} {sep[s]:>4}/{n}  ({sep[s] / n:.2f}){'':>6} {str(_median(margins[s])):>14}")
    best = max(SIGNALS, key=lambda s: sep[s])
    print(f"  => 최강 분리 신호: {best} ({sep[best]}/{n}). 분리율 높으면 그 신호로 reranker 설계.")
    print("     낮으면(둘 다 ~0.5) intensity 신호로도 변별 불가 → 학습 descriptor/VLM tie-break 필요.")
    print(f"[INFO] 완료: {out_dir} (probe.jsonl)")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
