"""localization e2e A/B — compute_align_key_score(baseline) vs compute_align_key_score_ensemble.

proposer_recall_ab 가 proposer recall(후보 집합 membership)만 쟀다면, 이 러너는 두 *완성 함수*의
최종 픽 정확도를 잰다: 예측 align point(best_xy + align_offset)가 GT(cond crosshair) 허용오차
내인가(hit). proposer recall 0.698 이 실제 최종 좌표 정확도로 전환됐는지 보는 정답 지표.

두 arm 은 동일 입력·동일 config(scales=COMPARE_SCALES, policy=STRUCTURE_POLICY — 생산 fallback/
static-compare 경로와 동일). 차이는 오직 proposer 단계: baseline=C1 chamfer 단일, ensemble=3채널
RRF + chamfer rescore + reranker selection(2026-06-09 이후 production=NCC; 그 전엔 ORB).
modality 라우팅·box template·cond GT 는
golden_localization_eval_cond 재사용. 설계: docs/specs/2026-06-09-ensemble-proposer-
production-integration-design.md.

측정 주의(버그 아님, 해석용):
- 모집단은 S(tool self-reported success) 프레임만 — proposer_recall_ab(0.698)와 *동일 모집단*
  이라 apples-to-apples. 단 ensemble 이득은 drift/외형변화(주로 E)에서 더 클 수 있어 S-only 는
  보수적(이득 과소·회귀 과대 추정 경향). E 모집단 평가는 별도.
- err 정규화 기준(short)은 box template 단변 — proposer_recall_ab 와 동일 척도. recipe 별
  template 크기 차이로 절대 px 허용오차가 달라져 aggregate hit_rate 는 이질적(전체 척도).
- paired A/B: 한 arm 이라도 매칭 실패 시 양쪽 모두 프레임 제외(동일 프레임 집합 유지).
- 해상도 캡 없음(full-res) — proposer_recall_ab 의 PROPOSER_MAX_DIM 미적용.
- per-frame 은 rows.jsonl 저장 → recipe 별 회귀·부트스트랩 CI 등 사후 분석용.

실행(오피스, golden 데이터 필요): uv run python poc/workflow_2/localization_ab_ensemble.py
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
from pathlib import Path

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_3.vision.align_key_matcher import (
    STRUCTURE_POLICY,
    compute_align_key_score,
    compute_align_key_score_ensemble,
)
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_3.vision.align_point_correction import _tool_label
from poc.workflow_3.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "localization_ab_ensemble"


def _predicted_align_point(result, offset):
    """매칭 결과의 align point = best_xy(템플릿 중심) + align_offset."""
    return (result.best_xy[0] + offset[0], result.best_xy[1] + offset[1])


def _err_norm(pred_xy, gt_xy, short):
    """예측 align point 와 GT 의 거리(short-side 정규화) — GT_TOL_NORM 과 동일 척도."""
    return math.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]) / float(short)


def _percentile(values, q):
    """정렬된 값 리스트의 q 분위(0~1). 빈 리스트 → None."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 4)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 4)


def run():
    """baseline vs ensemble 최종 픽 정확도 A/B (인자 없음). 반환 success|no_data."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터 없음: {root}")
        return "no_data"
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rec = len(recipes)
    print(f"[INFO] localization e2e A/B 시작 — recipe {n_rec}개 "
          f"(scales=COMPARE_SCALES, policy=STRUCTURE_POLICY)")

    # 정규화 거리 오차(예측 align point ↔ GT). hit = err <= GT_TOL_NORM.
    base_err, ens_err = [], []
    # 프레임별 hit confusion — 순이득/회귀를 직접 본다.
    conf = {"both_hit": 0, "only_ens": 0, "only_base": 0, "both_miss": 0}
    # decision=match 빈도(참고).
    dec_match = {"base": 0, "ens": 0}
    # 누락 사유 — proposer_recall_ab 와 동일 회계 + arm별 매칭 실패 분리.
    drop = {"no_box_tpl": 0, "non_S": 0, "routing_miss": 0, "no_crosshair": 0,
            "load_failed": 0, "match_failed_base": 0, "match_failed_ens": 0}
    rows = []   # per-frame 기록 — recipe 별 회귀·부트스트랩 CI 등 사후 분석용.
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
                gray_raw = load_gray(p)
            except Exception:
                drop["load_failed"] += 1
                continue
            frame = clean_image(gray_raw, cond)        # crosshair 제거(box__inpaint 경로와 동일).
            short = max(1, min(tpl.raw_image.shape[0], tpl.raw_image.shape[1]))
            # 두 arm 동일 입력·동일 config — 차이는 함수 내부 proposer 뿐. paired A/B 라
            # 한쪽이라도 실패하면 양쪽 모두 프레임 제외(동일 프레임 집합) + 책임 arm 계측.
            try:
                res_b = compute_align_key_score(
                    tpl, frame, scales=COMPARE_SCALES, policy=STRUCTURE_POLICY)
            except Exception as exc:
                drop["match_failed_base"] += 1
                print(f"[WARNING] baseline 매칭 실패 {p.name}: {exc}")
                continue
            try:
                res_e = compute_align_key_score_ensemble(
                    tpl, frame, scales=COMPARE_SCALES, policy=STRUCTURE_POLICY)
            except Exception as exc:
                drop["match_failed_ens"] += 1
                print(f"[WARNING] ensemble 매칭 실패 {p.name}: {exc}")
                continue

            eb = _err_norm(_predicted_align_point(res_b, (dx, dy)), gt_xy, short)
            ee = _err_norm(_predicted_align_point(res_e, (dx, dy)), gt_xy, short)
            base_err.append(eb)
            ens_err.append(ee)
            hb, he = eb <= GT_TOL_NORM, ee <= GT_TOL_NORM
            if hb and he:
                conf["both_hit"] += 1
            elif he:
                conf["only_ens"] += 1
            elif hb:
                conf["only_base"] += 1
            else:
                conf["both_miss"] += 1
            if res_b.decision == "match":
                dec_match["base"] += 1
            if res_e.decision == "match":
                dec_match["ens"] += 1
            rows.append({"recipe": assets.recipe_id, "msr": p.name, "modality": routed,
                         "err_base": round(eb, 4), "err_ens": round(ee, 4),
                         "hit_base": bool(hb), "hit_ens": bool(he)})
            n += 1
            if n % 25 == 0:
                print(f"[INFO] 진행 {n} S frames (recipe {ri}/{n_rec})")

    if not n:
        print(f"[ERROR] 처리된 S 프레임 없음. (누락 {drop})")
        return "no_data"

    # net 은 원시 카운트에서 계산(반올림된 값끼리 빼면 ±0.001 손실).
    nb = sum(1 for e in base_err if e <= GT_TOL_NORM)
    ne = sum(1 for e in ens_err if e <= GT_TOL_NORM)
    hit_base = round(nb / n, 3)
    hit_ens = round(ne / n, 3)
    summary = {
        "n": n, "GT_TOL_NORM": GT_TOL_NORM, "drop": drop, "confusion": conf,
        "decision_match": dec_match,
        "hit_rate": {"baseline": hit_base, "ensemble": hit_ens,
                     "net": round((ne - nb) / n, 3)},
        "err_norm": {
            "baseline": {"median": _percentile(base_err, 0.5),
                         "p75": _percentile(base_err, 0.75)},
            "ensemble": {"median": _percentile(ens_err, 0.5),
                         "p75": _percentile(ens_err, 0.75)},
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] S 채택 {n}장 | 누락 {drop}")
    print(f"\n[INFO] === localization e2e A/B (S {n}장, tol={GT_TOL_NORM}) ===")
    print(f"  {'variant':<14} {'hit_rate':>9} {'err_median':>11} {'err_p75':>9}")
    print(f"  {'baseline(C1)':<14} {hit_base:>9} "
          f"{summary['err_norm']['baseline']['median']:>11} "
          f"{summary['err_norm']['baseline']['p75']:>9}")
    print(f"  {'ensemble':<14} {hit_ens:>9} "
          f"{summary['err_norm']['ensemble']['median']:>11} "
          f"{summary['err_norm']['ensemble']['p75']:>9}")
    print(f"\n  >>> hit_rate net = {summary['hit_rate']['net']:+}  "
          f"(baseline {hit_base} → ensemble {hit_ens})")
    print(f"  confusion: both_hit={conf['both_hit']} only_ens(gain)={conf['only_ens']} "
          f"only_base(regress)={conf['only_base']} both_miss={conf['both_miss']}")
    print(f"  decision=match: base={dec_match['base']} ens={dec_match['ens']}")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
