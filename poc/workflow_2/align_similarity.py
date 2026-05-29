"""rcp align 영역(정중앙) ↔ msr(S/E) 유사도 진단 — calibration / 지표 선택용.

목적:
  - rcp 의 align point 는 *정중앙* 이다(흰 box 는 보조 단서). 이 중앙 영역이 msr 에서
    얼마나 유사하게 나타나는지를 **S/E 라벨별로** 재서, 유사도가 "key 있음/없음" 을
    가르는 신호인지 + 임계값을 정한다. S/E 라벨을 ground truth 로 쓴다.
  - E 의 두 유형(사용자 설명)을 우리 유사도로 재현되는지 검증:
      * E-type1: crosshair 가 *틀린 위치* → 장비가 거기서 낮은 점수 → fail.
                 ⇒ "crosshair 위치 유사도" 가 낮아야 한다.
      * E-type2: crosshair 를 아예 못 잡음 → no crosshair.

세 위치에서 유사도를 잰다 (rcp 중앙 template 기준):
  1. at_crosshair : msr 에서 검출된 crosshair 주변 ROI (v2 검출). S 높음 / E-type1 낮음.
  2. at_center    : msr 정중앙 주변 ROI. 성공 정렬이면 align point 가 중앙에 옴.
  3. free_best    : msr 전체 free 검색 best. "key 가 어디든 있나?" E-type1 에서 높으면 이동 복구 가능.

지표: matcher score(Chamfer+ORB, STRUCTURE_POLICY) 주지표 + MI + NCC(지표 변별력 비교).
비교용으로 rcp 흰 box template 의 free_best 도 같이 계산해 "center vs box 중 무엇이 더 잘
가르나" 를 답한다.

좌표 권한 원칙(VLM 아니라 CV)과 무관한 *순수 진단* — 실데이터(오피스)에서 한 번 돌려
S/E 분리 통계를 텍스트로 회신받아 임계/지표를 정한다([[feedback_no_office_data_to_mac]]).

실행:
    uv run python poc/workflow_2/align_similarity.py   # 실데이터면 분석, 없으면 self-test
출력: stdout 요약 + DEBUG_IMAGE_DIR/align_similarity/<ts>/{rows.jsonl, summary.json}
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
)

# rcp 중앙 align 영역 crop 의 면적 비율 (각 변 = sqrt). align_point_correction 의 fallback 과 동일 계열.
CENTER_AREA_RATIO = 0.15
# 정적 비교 scale band (rcp/msr 거의 같은 배율 가정) — align_point_correction.COMPARE_SCALES 와 동일.
COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)
# at_center / at_crosshair ROI 한 변 = template 변 × 이 배수 (검색 여유).
ROI_FACTOR = 1.8
# 한 recipe 당 처리할 msr 상한 (None=전부). 빠른 시험용.
LIMIT_PER_RECIPE = None
# 참조 staleness 판정 — S-at-crosshair(ground truth) 대비 rcp-center 의 median matcher score 가
# 이보다 낮으면 "rcp 참조가 현재 측정과 너무 달라 재등록 권장". STRUCTURE_POLICY.adjust_threshold(0.40)와 정렬.
REFERENCE_STALE_SCORE = 0.40


# ====================================================================
# 유사도 지표.
# ====================================================================


def _mi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """두 동일 크기 gray crop 의 mutual information (밝기/대비 drift 에 강건)."""
    a = a.ravel()
    b = b.ravel()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / max(hist.sum(), 1.0)
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    nz = pab > 0
    pa_pb = pa[:, None] * pb[None, :]
    return float((pab[nz] * np.log(pab[nz] / pa_pb[nz])).sum())


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """두 동일 크기 crop 의 정규화 상관 (pixel 동일성 baseline — 보통 drift 에 약함)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _matched_crop(frame: np.ndarray, center_xy, tw: int, th: int, scale: float) -> np.ndarray | None:
    """best 위치/스케일에서 msr crop 을 떼어 template 크기로 리사이즈 (MI/NCC 입력용)."""
    cw = max(1, int(round(tw * scale)))
    ch = max(1, int(round(th * scale)))
    cx, cy = center_xy
    x0 = max(0, int(cx - cw // 2))
    y0 = max(0, int(cy - ch // 2))
    x1 = min(frame.shape[1], x0 + cw)
    y1 = min(frame.shape[0], y0 + ch)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)


def _window_roi(frame_shape, center_xy, tw: int, th: int, factor: float = ROI_FACTOR):
    """center_xy 주변, template 의 factor 배 크기 ROI (x, y, w, h). 매칭을 국소화한다."""
    h, w = frame_shape[:2]
    rw = min(w, int(tw * factor))
    rh = min(h, int(th * factor))
    cx, cy = center_xy
    x0 = int(max(0, min(w - rw, cx - rw // 2)))
    y0 = int(max(0, min(h - rh, cy - rh // 2)))
    return (x0, y0, rw, rh)


def _score(template, frame, *, roi=None):
    """compute_align_key_score 래퍼 — (score, chamfer, orb, best_xy, best_scale)."""
    r = compute_align_key_score(
        template, frame, roi_hint=roi, scales=COMPARE_SCALES, policy=STRUCTURE_POLICY,
    )
    return r.score, r.chamfer_score, r.orb_inlier_ratio, r.best_xy, r.best_scale


def _race(templates: dict, frame, *, roi=None):
    """OM/SEM template 중 점수 높은 쪽 채택. 반환 (modality, score, chamfer, orb, best_xy, best_scale)."""
    best = None
    for mod, tpl in templates.items():
        if tpl is None:
            continue
        try:
            s, ch, orb, xy, sc = _score(tpl, frame, roi=roi)
        except Exception as exc:
            print(f"[WARNING] score 실패 ({mod}): {exc}")
            continue
        if best is None or s > best[1]:
            best = (mod, s, ch, orb, xy, sc)
    return best  # None 가능.


# ====================================================================
# 템플릿 빌드 (rcp 중앙 + 흰 box).
# ====================================================================


def _build_templates(assets):
    """recipe rcp 에서 center / box template 을 modality 별로 만든다.

    반환: (center_templates: dict[mod->tpl], box_templates: dict[mod->tpl|None])
    center 는 정중앙 면적 crop(= align 영역), box 는 흰 unique-area box 안쪽 crop.
    """
    from poc.workflow_2.align_fail_assets import load_gray
    from poc.workflow_2.align_point_correction import (
        _centered_area_crop_bbox,
        _detect_white_box,
        _inner_crop_for_box,
    )

    center: dict = {}
    box: dict = {}
    sources = [("om", assets.recipe_om, "rcp_om", "om"),
               ("sem", assets.recipe_sem, "rcp_sem", "sem")]
    for mod, path, version, key_type in sources:
        if path is None:
            continue
        gray = load_gray(path)
        cx, cy, cw, ch = _centered_area_crop_bbox(gray, CENTER_AREA_RATIO)
        center_crop = gray[cy:cy + ch, cx:cx + cw].copy()
        center[mod] = build_template(center_crop, recipe_id=assets.recipe_id,
                                     version=version + "_center", key_type=key_type)
        det = _detect_white_box(gray)
        if det is not None:
            inner, _bbox = _inner_crop_for_box(gray, det)
            box[mod] = build_template(inner, recipe_id=assets.recipe_id,
                                      version=version + "_box", key_type=key_type)
        else:
            box[mod] = None
    return center, box


# ====================================================================
# 한 msr 행.
# ====================================================================


def _process_msr(msr_path, *, center_tpls, box_tpls):
    from poc.workflow_2.align_fail_assets import load_gray
    from poc.workflow_2.align_point_correction import _tool_label
    from poc.workflow_2.crosshair_detect import detect_crosshair

    gray = load_gray(msr_path)
    h, w = gray.shape[:2]
    label = _tool_label(msr_path.name)
    center_xy = (w // 2, h // 2)

    # 대표 template 크기 (center, 첫 modality) — ROI 산정용.
    any_center = next((t for t in center_tpls.values() if t is not None), None)
    if any_center is None:
        return None
    th, tw = any_center.raw_image.shape[:2]

    # 1) free best (center template race).
    free = _race(center_tpls, gray)
    free_best = free[1] if free else None
    free_xy = free[4] if free else None
    free_scale = free[5] if free else 1.0
    free_mod = free[0] if free else None
    free_dist_center = float(np.hypot(free_xy[0] - center_xy[0], free_xy[1] - center_xy[1])) if free_xy else None

    # 2) at_center (center template, ROI = 중앙 window).
    center_roi = _window_roi(gray.shape, center_xy, tw, th)
    at_center = _race(center_tpls, gray, roi=center_roi)
    at_center_score = at_center[1] if at_center else None

    # 3) at_crosshair (v2 검출 → ROI = crosshair window). S 에서 이 위치가 ground-truth align 영역.
    #    rcp-center vs 여기 = 참조 staleness 측정 → score(기하)+MI(정보)+NCC(pixel) 모두 기록.
    ch_res = detect_crosshair(gray)
    at_crosshair_score = mi_xhair = ncc_xhair = None
    if ch_res.xy is not None:
        ch_roi = _window_roi(gray.shape, ch_res.xy, tw, th)
        at_ch = _race(center_tpls, gray, roi=ch_roi)
        if at_ch:
            at_crosshair_score = at_ch[1]
            ch_mod, _s, _c, _o, ch_xy, ch_scale = at_ch
            tpl = center_tpls[ch_mod]
            crop = _matched_crop(gray, ch_xy, tpl.raw_image.shape[1], tpl.raw_image.shape[0], ch_scale)
            if crop is not None:
                mi_xhair = _mi(tpl.raw_image, crop)
                ncc_xhair = _ncc(tpl.raw_image, crop)

    # MI / NCC — center template (winner modality) vs free-best crop.
    mi_free = ncc_free = None
    if free_mod is not None and free_xy is not None:
        tpl = center_tpls[free_mod]
        crop = _matched_crop(gray, free_xy, tpl.raw_image.shape[1], tpl.raw_image.shape[0], free_scale)
        if crop is not None:
            mi_free = _mi(tpl.raw_image, crop)
            ncc_free = _ncc(tpl.raw_image, crop)

    # 비교: box template free best.
    box_free = _race(box_tpls, gray) if any(v is not None for v in box_tpls.values()) else None
    free_best_box = box_free[1] if box_free else None

    return {
        "msr": msr_path.name,
        "label": label,
        "modality_used": free_mod,
        "free_best": free_best,
        "free_best_xy": list(free_xy) if free_xy else None,
        "free_best_dist_to_center": free_dist_center,
        "at_center": at_center_score,
        "crosshair_xy": list(ch_res.xy) if ch_res.xy else None,
        "crosshair_conf": ch_res.confidence,
        "crosshair_reason": ch_res.debug.get("reason"),
        "at_crosshair": at_crosshair_score,
        "mi_free": mi_free,
        "ncc_free": ncc_free,
        "mi_xhair": mi_xhair,
        "ncc_xhair": ncc_xhair,
        "free_best_box": free_best_box,
    }


# ====================================================================
# 분리도(separability) 통계.
# ====================================================================


def _separation(s_vals, e_vals) -> dict:
    """S(높아야)/E(낮아야) 두 분포의 분리도 — median + best threshold(balanced acc)."""
    s = [v for v in s_vals if isinstance(v, (int, float))]
    e = [v for v in e_vals if isinstance(v, (int, float))]
    out = {"n_s": len(s), "n_e": len(e)}
    if not s or not e:
        out["note"] = "insufficient samples"
        out["median_s"] = statistics.median(s) if s else None
        out["median_e"] = statistics.median(e) if e else None
        return out
    out["median_s"] = round(statistics.median(s), 4)
    out["median_e"] = round(statistics.median(e), 4)
    best_thr, best_bacc = None, -1.0
    for t in sorted(set(s + e)):
        tpr = sum(1 for v in s if v >= t) / len(s)     # S 를 맞춘 비율.
        tnr = sum(1 for v in e if v < t) / len(e)      # E 를 맞춘 비율.
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, t
    out["best_threshold"] = round(best_thr, 4)
    out["balanced_accuracy"] = round(best_bacc, 3)
    return out


def _summarize(rows: list[dict]) -> dict:
    by_label = {"S": [r for r in rows if r["label"] == "S"],
                "E": [r for r in rows if r["label"] == "E"]}
    metrics = ["free_best", "at_center", "at_crosshair", "mi_free", "ncc_free", "free_best_box"]
    sep = {}
    for m in metrics:
        sep[m] = _separation([r[m] for r in by_label["S"]], [r[m] for r in by_label["E"]])

    # E-type 추정: crosshair 있고 at_crosshair 낮은데 free_best 높음 = 이동 복구 가능(type1).
    e_with_ch = [r for r in by_label["E"] if r["crosshair_xy"] is not None and r["at_crosshair"] is not None]
    e_no_ch = [r for r in by_label["E"] if r["crosshair_xy"] is None]
    recoverable = [r for r in e_with_ch
                   if r["free_best"] is not None and r["at_crosshair"] is not None
                   and r["free_best"] - r["at_crosshair"] > 0.1]

    # 참조 staleness — recipe 별 rcp-center vs S-at-crosshair(ground truth). 낮으면 rcp 재등록 권장.
    by_recipe: dict[str, list] = {}
    for r in rows:
        by_recipe.setdefault(r["recipe"], []).append(r)
    ref_quality = []
    for rec, rs in by_recipe.items():
        s_ch = [x for x in rs if x["label"] == "S" and x["at_crosshair"] is not None]
        if not s_ch:
            continue
        med_score = statistics.median([x["at_crosshair"] for x in s_ch])
        mis = [x["mi_xhair"] for x in s_ch if isinstance(x["mi_xhair"], (int, float))]
        nccs = [x["ncc_xhair"] for x in s_ch if isinstance(x["ncc_xhair"], (int, float))]
        ref_quality.append({
            "recipe": rec,
            "n_S_with_crosshair": len(s_ch),
            "median_score_at_crosshair": round(med_score, 4),
            "median_mi": round(statistics.median(mis), 4) if mis else None,
            "median_ncc": round(statistics.median(nccs), 4) if nccs else None,
            "stale_recommend_replace": med_score < REFERENCE_STALE_SCORE,
        })
    ref_quality.sort(key=lambda d: d["median_score_at_crosshair"])  # worst-first.

    return {
        "counts": {"S": len(by_label["S"]), "E": len(by_label["E"]),
                   "other": len(rows) - len(by_label["S"]) - len(by_label["E"]), "total": len(rows)},
        "separation": sep,
        "E_breakdown": {
            "with_crosshair": len(e_with_ch),
            "no_crosshair": len(e_no_ch),
            "recoverable_by_move(free>>at_crosshair)": len(recoverable),
        },
        "reference_quality": ref_quality,
        "n_recipes_stale": sum(1 for d in ref_quality if d["stale_recommend_replace"]),
        "n_recipes_scored": len(ref_quality),
        "reference_stale_threshold": REFERENCE_STALE_SCORE,
    }


def _print_summary(summary: dict) -> None:
    c = summary["counts"]
    print(f"\n[INFO] images: S={c['S']} E={c['E']} other={c['other']} total={c['total']}")
    print("\n[INFO] S/E 분리도 (median_s 높고 median_e 낮을수록, balanced_accuracy 1.0 에 가까울수록 좋은 지표):")
    print(f"  {'metric':<14} {'med_S':>8} {'med_E':>8} {'thr':>8} {'bACC':>6}  n(S/E)")
    for m, s in summary["separation"].items():
        print(f"  {m:<14} {str(s.get('median_s')):>8} {str(s.get('median_e')):>8} "
              f"{str(s.get('best_threshold','-')):>8} {str(s.get('balanced_accuracy','-')):>6}  "
              f"{s.get('n_s')}/{s.get('n_e')}")
    eb = summary["E_breakdown"]
    print(f"\n[INFO] E 유형: with_crosshair={eb['with_crosshair']}  no_crosshair={eb['no_crosshair']}  "
          f"recoverable_by_move={eb['recoverable_by_move(free>>at_crosshair)']}")
    print("  * at_crosshair 의 med_E 가 med_S 보다 확실히 낮으면 → 우리 유사도가 장비 fail(낮은 점수)을 재현 = 지표 자격 검증")
    print("  * free_best 는 S/E 둘 다 높을 수 있음(key 가 E 에도 존재) → at_crosshair 가 진짜 변별자")

    # 참조 staleness — recipe 별 rcp-center vs S-at-crosshair. 낮은 recipe 는 rcp 재등록 권장.
    rq = summary.get("reference_quality", [])
    if rq:
        print(f"\n[INFO] 참조 품질 (rcp-center vs S-at-crosshair = ground truth). "
              f"score<{summary['reference_stale_threshold']} 이면 rcp 재등록 권장.")
        print(f"[INFO] stale recipes: {summary['n_recipes_stale']}/{summary['n_recipes_scored']} (worst-first, 상위 20개)")
        print(f"  {'recipe':<42} {'score':>6} {'MI':>6} {'NCC':>6} {'nS':>4}  replace?")
        for d in rq[:20]:
            print(f"  {d['recipe'][:42]:<42} {d['median_score_at_crosshair']:>6.3f} "
                  f"{str(d['median_mi']):>6} {str(d['median_ncc']):>6} {d['n_S_with_crosshair']:>4}  "
                  f"{'YES' if d['stale_recommend_replace'] else ''}")
        print("  * 이 목록이 곧 '교체(재등록)해야 할 rcp' 후보. MI/NCC 가 낮은데 score 만 높으면 외형 drift 큼(주의).")


# ====================================================================
# 엔트리.
# ====================================================================


def analyze(*, limit_per_recipe=LIMIT_PER_RECIPE) -> str:
    from poc.workflow_2.align_fail_assets import (
        iter_msr_images,
        iter_recipe_dirs,
        resolve_assets,
    )

    leaves = iter_recipe_dirs()
    if not leaves:
        print("[ERROR] align_images recipe 없음.")
        return "no_assets"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DEBUG_IMAGE_DIR / "align_similarity" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fp:
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
            msr_images = iter_msr_images(assets)
            if limit_per_recipe:
                msr_images = msr_images[:limit_per_recipe]
            print(f"[{i}/{len(leaves)}] {tag}  msr={len(msr_images)}")
            for msr_path in msr_images:
                try:
                    row = _process_msr(msr_path, center_tpls=center_tpls, box_tpls=box_tpls)
                except Exception as exc:
                    print(f"[WARNING] {msr_path.name}: {type(exc).__name__}: {exc}")
                    continue
                if row is None:
                    continue
                row["recipe"] = tag
                rows.append(row)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\n[INFO] 저장: {out_dir}/summary.json , rows.jsonl")
    return "success"


# ====================================================================
# 합성 self-test.
# ====================================================================


def _patterned(seed: int, size=(60, 60)) -> np.ndarray:
    """엣지가 뚜렷한 작은 패턴 (Chamfer 가 변별 가능하도록)."""
    rs = np.random.RandomState(seed)
    img = np.full(size, 40, dtype=np.uint8)
    for _ in range(6):
        p1 = (rs.randint(0, size[1]), rs.randint(0, size[0]))
        p2 = (rs.randint(0, size[1]), rs.randint(0, size[0]))
        cv2.line(img, p1, p2, 230, 2)
    cv2.rectangle(img, (10, 10), (size[1] - 10, size[0] - 10), 255, 2)
    return img


def _frame_with(pattern, *, at, canvas=(300, 400), bg=50) -> np.ndarray:
    rs = np.random.RandomState(7)
    img = np.full(canvas, bg, dtype=np.uint8)
    img = cv2.add(img, rs.randint(0, 20, canvas).astype(np.uint8))
    ph, pw = pattern.shape
    x, y = at
    img[y:y + ph, x:x + pw] = pattern
    return img


def _self_test() -> bool:
    """rcp 중앙 패턴이 S frame 중앙엔 있고 E frame 엔 없을 때, 유사도가 S>E 로 갈리는지."""
    from poc.workflow_2.align_key_matcher import build_template as _bt

    align = _patterned(1)
    other = _patterned(99)
    rcp_center_tpl = _bt(align, recipe_id="T", version="rcp_om_center", key_type="om")

    # S: 중앙에 align 패턴. E: 중앙에 other 패턴.
    cxy = (400 // 2 - 30, 300 // 2 - 30)
    s_frame = _frame_with(align, at=cxy)
    e_frame = _frame_with(other, at=cxy)

    s = _race({"om": rcp_center_tpl}, s_frame)
    e = _race({"om": rcp_center_tpl}, e_frame)
    assert s is not None and e is not None, "score 산출 실패"
    print(f"[INFO] self-test free_best: S={s[1]:.3f}  E={e[1]:.3f}")
    ok = s[1] > e[1]
    if not ok:
        print("[ERROR] S 가 E 보다 높지 않음 — 유사도 변별 실패(합성).")

    # separation 함수 sanity.
    sep = _separation([0.8, 0.75, 0.9], [0.3, 0.4, 0.2])
    assert sep["balanced_accuracy"] == 1.0, f"separation 오류: {sep}"
    assert sep["median_s"] > sep["median_e"]
    print(f"[INFO] self-test separation OK: {sep}")
    print("[INFO] self-test 통과." if ok else "[ERROR] self-test 실패.")
    return ok


def run() -> str:
    try:
        from poc.workflow_2.align_fail_assets import iter_recipe_dirs
        has_data = bool(iter_recipe_dirs())
    except Exception:
        has_data = False
    if has_data:
        return analyze()
    print("[WARNING] align_images 데이터 없음 — 합성 self-test 로 대체합니다.\n")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
