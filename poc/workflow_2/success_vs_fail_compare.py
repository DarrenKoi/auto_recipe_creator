"""success(golden) vs align-fail 의 rcp↔msr 차이 비교 — 엔지니어 재등록 가이드라인 산출.

목적:
  align 성공/실패 시 rcp(등록 key) ↔ msr(실측)의 차이가 어떻게 다른지를 **recipe-정규화된
  상대 ratio**(rcp_vs_consensus / S_internal 일관성)로 비교한다. golden(항상 성공) 분포로
  healthy 기준선·임계를 실측 calibration 하고, 그 임계를 fail 데이터에 적용해 "어느 recipe 의
  key 가 drift 했나"를 엔지니어에게 가이드한다.

원칙:
  - **절대 차이 점수는 recipe 가 다르면 비교 불가**(recipe 마다 key 생김새·엣지밀도가 제각각)
    → 항상 *상대 ratio* 로 비교. recipe 내부 S 들로 자기정규화하므로 recipe 간 비교가 가능.
  - golden 은 E 가 없어(전부 S) align_similarity 의 S/E·truth-forced·gt-topk 블록은 불필요.
    여기선 상대 staleness(`_reference_quality`)만 재사용한다.
  - 최종 재등록 판정은 **자동 단독 금지** — CV 는 후보를 거르고 플래그된 것만 엔지니어가 확인.
  - 관련: [[project_matcher_flat_chamfer_distinctiveness]], `docs/align_success_dataset_plan.md` §7.

설계상 align_similarity.py 의 헬퍼(`_build_templates`/`_process_msr`/`_reference_quality`)를 import
재사용하며 그 파일은 수정하지 않는다(중복 로직 방지). golden 은 E 가 없어 그 파일을 직접 실행하지
않고 본 전용 모듈이 healthy 기준선 + 비교만 담당한다.

레이아웃(사용자가 office 에서 직접 적재 — 수집 스크립트 없음):
  align_images_golden/<eqp>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr}   # S 만

실행:
    uv run python poc/workflow_2/success_vs_fail_compare.py   # golden+fail 있으면 비교, 없으면 self-test
출력: stdout 요약 + DEBUG_IMAGE_DIR/success_vs_fail/<ts>/{golden_rows.jsonl, fail_rows.jsonl,
      compare_summary.json, guideline.md(한국어)}
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_2.align_similarity import (
    RELATIVE_STALE_RATIO,
    _build_templates,
    _process_msr,
    _reference_quality,
)

# golden 루트 — fail 루트(align_images)의 형제. 사용자가 이 경로에 success 데이터를 적재.
ALIGN_GOLDEN_ROOT = ALIGN_IMAGES_ROOT.parent / "align_images_golden"

# ratio 가 의미있는(판정가능) status 만 비교/임계에 쓴다. 나머지(insufficient_S,
# S_inconsistent, low_texture_inconclusive, no_rcp)는 ratio 판정 보류 대상.
SCORABLE_STATUSES = ("ok", "stale_replace")
GOLDEN_STALE_PCTL = 10   # healthy ratio 분포 하위 percentile → calibrated stale 임계 후보.
GOLDEN_CV_PCTL = 90      # healthy CV 분포 상위 percentile → calibrated S_inconsistent 임계 후보.


# ── crop 수집 (align_similarity.analyze 와 동일 경로 재사용) ──────────────────────

def _gather_crops_by_recipe(root: Path, *, limit_per_recipe=None) -> dict:
    """``root`` 아래 recipe 들의 crops_by_recipe — `_reference_quality` 입력 형식.

    `_process_msr` 로 S-at-crosshair crop 을 모은다(align_similarity.analyze 와 동일).
    golden 은 전부 S 라 e_paths 는 보통 빈다.
    """
    from poc.workflow_3.vision.align_fail_assets import (
        iter_msr_images,
        iter_recipe_dirs,
        resolve_assets,
    )

    out: dict = {}
    leaves = iter_recipe_dirs(root)
    for i, leaf in enumerate(leaves, 1):
        assets = resolve_assets(leaf)
        if assets.recipe_om is None and assets.recipe_sem is None:
            continue
        try:
            center_tpls, box_tpls = _build_templates(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {leaf}: {exc}")
            continue
        tag = f"{assets.eqp_id}/{assets.class_name}/{assets.recipe_id}"
        entry = {
            "rcp": {m: t.raw_image for m, t in center_tpls.items() if t is not None},
            "s_frames": [],
            "e_paths": [],
        }
        msr_images = iter_msr_images(assets)
        if limit_per_recipe:
            msr_images = msr_images[:limit_per_recipe]
        print(f"[{i}/{len(leaves)}] {tag}  msr={len(msr_images)}")
        for msr_path in msr_images:
            try:
                row, xhair_crop, xhair_mod = _process_msr(
                    msr_path, center_tpls=center_tpls, box_tpls=box_tpls)
            except Exception as exc:
                print(f"[WARNING] {msr_path.name}: {type(exc).__name__}: {exc}")
                continue
            if row is None:
                continue
            if (row["label"] == "S" and xhair_crop is not None
                    and xhair_mod is not None and row.get("crosshair_xy") is not None):
                entry["s_frames"].append({
                    "path": msr_path, "xy": tuple(row["crosshair_xy"]),
                    "mod": xhair_mod, "crop": xhair_crop,
                })
            elif row["label"] == "E":
                entry["e_paths"].append(msr_path)
        out[tag] = entry
    return out


# ── calibration / 비교 (이 모듈의 핵심 신규 로직 — self-test 대상) ────────────────

def _pctl(vals: list, p: float):
    return round(float(np.percentile(vals, p)), 3) if vals else None


def _scorable(rows: list) -> list:
    """ratio 가 의미있는 recipe 행만(status ∈ SCORABLE_STATUSES)."""
    return [r for r in rows
            if r.get("status") in SCORABLE_STATUSES
            and isinstance(r.get("relative_ratio"), (int, float))]


def _status_counts(rows: list) -> dict:
    c: dict = {}
    for r in rows:
        s = r.get("status", "?")
        c[s] = c.get(s, 0) + 1
    return dict(sorted(c.items()))


def _ratio_stats(rows: list) -> dict | None:
    """scorable 행의 relative_ratio 분포(min/percentile/median/max)."""
    ratios = [r["relative_ratio"] for r in _scorable(rows)]
    if not ratios:
        return None
    return {
        "n": len(ratios),
        "min": round(min(ratios), 3),
        "p5": _pctl(ratios, 5),
        "p10": _pctl(ratios, 10),
        "p25": _pctl(ratios, 25),
        "median": round(statistics.median(ratios), 3),
        "max": round(max(ratios), 3),
    }


def _calibrate(golden_rows: list) -> dict | None:
    """golden healthy 분포 → 임계 후보 + 현재 임계의 거짓양성."""
    scor = _scorable(golden_rows)
    ratios = [r["relative_ratio"] for r in scor]
    # cvs 는 _scorable 가 아니라 *CV 가 계산된 모든 golden 행*(S_inconsistent 포함)에서 뽑는다.
    # 의도적 비대칭: S_INCONSISTENT_CV 임계 자체를 calibration 하려면 성공 recipe 의 전체 CV
    # 분포가 필요하다(scorable 로 거르면 이미 CV≤cold-start 게이트를 통과한 행만 남아 임계를
    # 더 풀어야 하는 경우를 발견 못 함). ratios 는 반대로 incoherent consensus 의 ratio 를
    # 못 믿으므로 _scorable 로 거른다.
    cvs = [r["s_internal_cv"] for r in golden_rows
           if isinstance(r.get("s_internal_cv"), (int, float))]
    if not ratios:
        return None
    fp = sum(1 for x in ratios if x < RELATIVE_STALE_RATIO)
    return {
        "n_scorable": len(ratios),
        "current_stale_ratio": RELATIVE_STALE_RATIO,
        # 현재(cold-start) 임계로 golden 이 stale 로 찍히는 수 — 0 이 이상적(plan §4.2).
        "false_positive_at_current": fp,
        "false_positive_rate_at_current": round(fp / len(ratios), 3),
        # 실측 calibration 제안: healthy 하위 10% ratio / 상위 90% CV.
        "suggested_stale_ratio": _pctl(ratios, GOLDEN_STALE_PCTL),
        "suggested_inconsistent_cv": _pctl(cvs, GOLDEN_CV_PCTL) if cvs else None,
    }


def _apply_threshold(rows: list, stale_ratio: float) -> dict:
    """rows 에 stale 임계 적용 → 플래그된 recipe(ratio 오름차순)."""
    scor = _scorable(rows)
    flagged = sorted((r for r in scor if r["relative_ratio"] < stale_ratio),
                     key=lambda r: r["relative_ratio"])
    return {
        "stale_ratio": stale_ratio,
        "n_scorable": len(scor),
        "n_flagged": len(flagged),
        "flagged_rate": round(len(flagged) / len(scor), 3) if scor else None,
        "flagged_recipes": [
            {"recipe": r.get("recipe"), "relative_ratio": r["relative_ratio"],
             "s_internal_cv": r.get("s_internal_cv"), "n_S": r.get("n_S")}
            for r in flagged
        ],
    }


def _build_comparison(golden_rows: list, fail_rows: list) -> dict:
    """golden/fail 분포 + calibration + fail 에 임계 적용 결과."""
    cal = _calibrate(golden_rows)
    out = {
        "golden": {
            "n_recipes": len(golden_rows),
            "status_counts": _status_counts(golden_rows),
            "ratio_dist": _ratio_stats(golden_rows),
        },
        "fail": {
            "n_recipes": len(fail_rows),
            "status_counts": _status_counts(fail_rows),
            "ratio_dist": _ratio_stats(fail_rows),
        },
        "calibration": cal,
    }
    if cal:
        out["fail_at_current_threshold"] = _apply_threshold(fail_rows, cal["current_stale_ratio"])
        if cal["suggested_stale_ratio"] is not None:
            out["fail_at_suggested_threshold"] = _apply_threshold(
                fail_rows, cal["suggested_stale_ratio"])
    return out


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_comparison(cmp: dict) -> None:
    g, f = cmp["golden"], cmp["fail"]
    print("\n[INFO] SUCCESS(golden) vs FAIL — rcp↔msr 상대 ratio 비교")
    print(f"  golden recipes={g['n_recipes']}  status={g['status_counts']}")
    print(f"  fail   recipes={f['n_recipes']}  status={f['status_counts']}")
    print(f"  golden ratio 분포: {g['ratio_dist']}")
    print(f"  fail   ratio 분포: {f['ratio_dist']}")
    cal = cmp.get("calibration")
    if not cal:
        print("  [WARNING] golden scorable recipe 없음 — 임계 calibration 불가.")
        return
    print(f"\n[INFO] CALIBRATION (golden healthy 기준):")
    print(f"  현재 임계 stale_ratio={cal['current_stale_ratio']} → golden 거짓양성="
          f"{cal['false_positive_at_current']}/{cal['n_scorable']} "
          f"({cal['false_positive_rate_at_current']})  (0 이상적)")
    print(f"  제안 임계: stale_ratio={cal['suggested_stale_ratio']}(healthy p{GOLDEN_STALE_PCTL}) "
          f"inconsistent_cv={cal['suggested_inconsistent_cv']}(healthy p{GOLDEN_CV_PCTL})")
    cur = cmp.get("fail_at_current_threshold")
    sug = cmp.get("fail_at_suggested_threshold")
    if cur:
        print(f"\n[INFO] FAIL 에 임계 적용 → drift(재등록 검토) recipe:")
        print(f"  @현재({cur['stale_ratio']}): {cur['n_flagged']}/{cur['n_scorable']} "
              f"({cur['flagged_rate']})")
        if sug:
            print(f"  @제안({sug['stale_ratio']}): {sug['n_flagged']}/{sug['n_scorable']} "
                  f"({sug['flagged_rate']})")
        shown = sug or cur
        print(f"  (목록 = @{'제안' if sug else '현재'}({shown['stale_ratio']}) 기준 플래그 recipe)")
        for r in shown["flagged_recipes"][:20]:
            print(f"    {str(r['recipe'])[:44]:<44} ratio={r['relative_ratio']} "
                  f"cv={r['s_internal_cv']} nS={r['n_S']}")
    print("  * golden 보다 fail 의 ratio 분포가 낮으면 → align fail 이 rcp staleness 와 연관.")
    print("  * 플래그 recipe = rcp 가 자기 성공 cluster 의 outlier → 엔지니어 재등록 검토 대상.")


def _build_guideline(cmp: dict) -> str:
    """엔지니어용 한국어 가이드라인 markdown."""
    g, f = cmp["golden"], cmp["fail"]
    cal = cmp.get("calibration")
    lines = [
        "# Align 재등록 가이드라인 (success vs fail rcp↔msr 비교)",
        "",
        "> 자동 생성 — `success_vs_fail_compare.py`. **최종 재등록 판정은 엔지니어 확인 후.**",
        "",
        "## 1. 분포 요약",
        f"- golden(성공) recipes={g['n_recipes']}, status={g['status_counts']}",
        f"- fail recipes={f['n_recipes']}, status={f['status_counts']}",
        f"- golden ratio 분포: {g['ratio_dist']}",
        f"- fail ratio 분포: {f['ratio_dist']}",
        "",
    ]
    if not cal:
        lines += ["## 2. 판정 불가", "golden scorable recipe 가 없어 임계 calibration 불가. "
                  "recipe당 S≥3 + 일관성 있는 golden 을 더 모아야 함(plan §3 규모 가이드)."]
        return "\n".join(lines) + "\n"
    lines += [
        "## 2. 임계 (golden healthy 기준)",
        f"- 현재(cold-start) stale_ratio={cal['current_stale_ratio']} → golden 거짓양성 "
        f"{cal['false_positive_at_current']}/{cal['n_scorable']} "
        f"({cal['false_positive_rate_at_current']}). 0 이어야 방법이 건강한 rcp 를 안 건드림.",
        f"- 실측 제안: stale_ratio={cal['suggested_stale_ratio']} (healthy 하위 {GOLDEN_STALE_PCTL}%), "
        f"inconsistent_cv={cal['suggested_inconsistent_cv']} (healthy 상위 {GOLDEN_CV_PCTL}%).",
        "  > recipe 가 ~30 미만이면 percentile 임계는 *잠정* — plan §3 규모 가이드 참고.",
        "",
        "## 3. 재등록 검토 대상 (fail 중 stale 플래그)",
    ]
    target = cmp.get("fail_at_suggested_threshold") or cmp.get("fail_at_current_threshold")
    if target and target["flagged_recipes"]:
        lines.append(f"임계 ratio<{target['stale_ratio']} 기준 {target['n_flagged']}개:")
        lines.append("")
        lines.append("| recipe | relative_ratio | s_internal_cv | n_S |")
        lines.append("|---|---|---|---|")
        for r in target["flagged_recipes"]:
            lines.append(f"| {r['recipe']} | {r['relative_ratio']} | "
                         f"{r['s_internal_cv']} | {r['n_S']} |")
    else:
        lines.append("플래그된 recipe 없음 (현 임계 기준 fail 의 rcp 가 모두 healthy 범위).")
    lines += [
        "",
        "## 4. 해석",
        "- golden 보다 fail 의 ratio 분포가 낮으면 → **align fail 이 rcp staleness 와 연관**.",
        "- 플래그 recipe = rcp 가 자기 성공 cluster 의 outlier → **재등록 검토 대상**.",
        "- ratio 가 healthy 범위인데 fail → staleness 외 원인(공정/측정) → 재등록으로 안 풀림.",
    ]
    return "\n".join(lines) + "\n"


def compare() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DEBUG_IMAGE_DIR / "success_vs_fail" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] golden(성공) crop 수집...")
    golden_rows = _reference_quality(_gather_crops_by_recipe(ALIGN_GOLDEN_ROOT))
    print("[INFO] fail crop 수집...")
    fail_rows = _reference_quality(_gather_crops_by_recipe(ALIGN_IMAGES_ROOT))

    cmp = _build_comparison(golden_rows, fail_rows)
    (out_dir / "golden_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in golden_rows), encoding="utf-8")
    (out_dir / "fail_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in fail_rows), encoding="utf-8")
    (out_dir / "compare_summary.json").write_text(
        json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "guideline.md").write_text(_build_guideline(cmp), encoding="utf-8")
    _print_comparison(cmp)
    print(f"\n[INFO] 저장: {out_dir}/compare_summary.json, guideline.md, *_rows.jsonl")
    return "success"


def _self_test() -> bool:
    """calibration/비교 로직 — 합성 per-recipe 행으로(이미지 트리 없이) 검증."""
    # 합성 golden: healthy(ratio 높고 CV 낮음).
    golden = [{"recipe": f"g{i}", "status": "ok", "relative_ratio": r,
               "s_internal_cv": 0.2, "n_S": 8}
              for i, r in enumerate(
                  [0.95, 0.90, 1.00, 0.88, 0.92, 0.85, 0.97, 1.05, 0.90, 0.93])]
    # 합성 fail: 일부 drift(낮은 ratio) + 일부 healthy.
    fail = [{"recipe": f"f{i}", "status": "ok", "relative_ratio": r,
             "s_internal_cv": 0.25, "n_S": 5}
            for i, r in enumerate([0.40, 0.50, 0.45, 0.80, 0.90, 0.30, 0.55, 0.70])]
    # 판정 보류 status 는 scorable 에서 빠져야 함.
    fail.append({"recipe": "f_incon", "status": "S_inconsistent",
                 "relative_ratio": 0.1, "s_internal_cv": 0.9, "n_S": 4})

    cmp = _build_comparison(golden, fail)
    g_med = cmp["golden"]["ratio_dist"]["median"]
    f_med = cmp["fail"]["ratio_dist"]["median"]
    assert g_med > f_med, f"golden median 이 fail 보다 높아야: {g_med} vs {f_med}"

    cal = cmp["calibration"]
    assert cal["false_positive_at_current"] == 0, f"golden FP@current 0 이어야: {cal}"
    assert cal["suggested_stale_ratio"] is not None

    cur = cmp["fail_at_current_threshold"]
    assert cur["n_scorable"] == 8, f"S_inconsistent 가 scorable 에 섞임: {cur['n_scorable']}"
    assert cur["n_flagged"] > 0, f"fail stale 플래그 >0 이어야: {cur}"
    flagged_recipes = {r["recipe"] for r in cur["flagged_recipes"]}
    assert "f_incon" not in flagged_recipes, "보류 status 가 플래그됨(scorable 누수)"

    print(f"[INFO] self-test: golden median={g_med} > fail median={f_med}; "
          f"suggested_stale_ratio={cal['suggested_stale_ratio']}; "
          f"fail flagged@current={cur['n_flagged']}/{cur['n_scorable']}")

    # 빈 입력 안전(데이터 0 일 때 crash 금지).
    empty = _build_comparison([], [])
    assert empty["calibration"] is None
    guideline = _build_guideline(empty)
    assert "판정 불가" in guideline
    assert "재등록 가이드라인" in _build_guideline(cmp)

    print("[INFO] self-test 통과.")
    return True


def run() -> str:
    try:
        from poc.workflow_3.vision.align_fail_assets import iter_recipe_dirs
        has_golden = bool(iter_recipe_dirs(ALIGN_GOLDEN_ROOT))
        has_fail = bool(iter_recipe_dirs(ALIGN_IMAGES_ROOT))
    except Exception:
        has_golden = has_fail = False
    if has_golden and has_fail:
        return compare()
    print(f"[WARNING] golden({has_golden})/fail({has_fail}) 데이터 부족 — 합성 self-test 로 대체.")
    print(f"          golden 경로: {ALIGN_GOLDEN_ROOT}\n")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
