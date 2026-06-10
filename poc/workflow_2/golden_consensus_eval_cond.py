"""consensus 재등록 A/B (cond GT) — 단일 rcp template 대신 recipe 의 S 들로 만든
consensus template 이 proposer recall(gt_in_topk)을 올리나 검증한다.

왜 consensus 인가
----------------
PROPOSER_WALL: 진실(align point)이 matcher 후보(top-N)에 자주 없다([[project_matcher_flat_chamfer_distinctiveness]]).
단일 rcp 등록 key 는 공정 드리프트로 stale 해질 수 있다. recipe 의 *최근 성공(S)* 들을
평균낸 consensus 는 현재 외형을 추종 → 후보에 진실이 들어올 확률↑. 과거 실데이터(정제 전)
에서 in_topk +0.282 (0.436→0.718) 로 검증된 유일한 강한 레버.

정렬(registration) 규칙 — **crosshair 중심, 이미지 중심 절대 금지**
-----------------------------------------------------------------
rcp 는 이미지 중심 = align point(고정 anchor). 그러나 msr S 는 웨이퍼마다 달라 이미지
중심이 align point 가 아니다. 각 S 의 align point 는 **crosshair**(align point 를
신뢰성 있게 표시 — 사용자 확인 2026-06-08)에 있으므로, 모든 S crop 을 crosshair 로
정렬해 median 을 떠야 또렷한 align-key 가 된다.

cond 개선점
----------
과거 align_similarity 는 raw 프레임에서 crop → 중앙에 crosshair 잔존(distractor).
cond.txt 가 정확한 crosshair 좌표를 주므로 **crop 전에 crosshair 를 inpaint** 로
지운다 → 중앙 distractor 없는 더 깨끗한 consensus.

검증/측정은 검증된 `align_similarity._consensus_template_ab`(LOO + blur 가드 +
apples-to-apples baseline)를 그대로 재사용한다(로직 중복/표류 방지). baseline = center
template(offset 0) — consensus 도 crosshair 정렬이라 offset 0, 동일 척도 A/B.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/golden_consensus_eval_cond.py
  golden 루트: 기본 align_images_golden/, env ALIGN_GOLDEN_ROOT 로 override.
  매칭 프레임: 기본 CLEAN(cond 구동 crosshair 제거 — consensus 중앙 inpaint 잔상과
  프레임 GT crosshair 의 가짜 lock 차단). env CONSENSUS_CLEAN_FRAME=0 이면 raw 판
  (과거 측정과 동일) — 두 판의 lift 를 비교해 consensus lift 가 진짜인지 판정한다.
출력: stdout + DEBUG_IMAGE_DIR/golden_consensus_eval_cond/<ts>/{summary.json, consensus/*.png,
  combined/<recipe>/*_combined.jpg}
  combined 는 localization cond 판과 같은 "한 장 추적" 패널 — [좌: rcp center(baseline) +
  LOO consensus 템플릿 스택 | 우: msr 프레임 + GT(crosshair) + 양쪽 top-N 후보].
  consensus 가 align point 를 잘 잡는지(후보가 GT 에 모이는지)를 눈으로 확인한다.
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
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_2.align_similarity import (
    GT_TOL_NORM, USE_ENSEMBLE_PROPOSER, _consensus_template_ab, _matched_crop,
)
from poc.workflow_2.ensemble_lab import (
    PERIODICITY_TAU, miss_predictor_stats, template_periodicity,
)
from poc.workflow_3.vision.align_point_correction import _tool_label
from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import (
    MSR_OM_MAG_MAX, MSR_SEM_MAG_MIN, _to_int, load_cond,
    msr_modality as _msr_modality,
)
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_3.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_consensus_eval_cond"

# co-registration: integer-crosshair 정렬이 남긴 sub-pixel 잔차를 phase-correlation 으로
# 마저 맞춰 median blur(edge_ratio<0.70 경고)를 줄인다 → consensus 가 또렷해져 membership·
# rank1 둘 다 개선 기대. env CONSENSUS_COREGISTER=0 으로 끄면 A/B 비교 가능.
COREGISTER = os.getenv("CONSENSUS_COREGISTER", "1") != "0"

# LOO 매칭 프레임 정제: raw 프레임은 GT 위치에 진짜 crosshair 가 남아 있고, consensus 템플릿
# 중앙에는 inpaint 잔상 십자가 코히런트하게 쌓여 있어(모든 S crop 이 crosshair 중심 정렬 +
# median 은 공통 신호 보존) chamfer 가 crosshair↔crosshair 로 lock 하면 in_topk 가 *가짜로*
# 부풀 수 있다(rcp baseline 은 이 이득이 없어 A/B 비대칭). 2026-06-10 오피스 관찰: score
# 0.6~0.8 인데 실제 실패율 높음 — 직선 매칭이 점수만 올린 정황. 그래서 기본 ON 으로 매칭
# 프레임도 cond 구동 clean_image 로 지우고, env CONSENSUS_CLEAN_FRAME=0(raw)과 lift 를 비교:
# 정제 후에도 lift 유지 → 진짜 / 무너짐 → crosshair artifact.
CLEAN_FRAME = os.getenv("CONSENSUS_CLEAN_FRAME", "1") != "0"
COREG_ITERS = 2                 # ref median 을 다듬으며 원본 crop 을 재정렬(보간 누적 방지).
COREG_MAX_SHIFT_FRAC = 0.3      # 추정 shift 가 crop 변의 이 비율 초과면 spurious → 정렬 생략.

# LOO consensus 최소 S(같은 modality) 장수. align_similarity.AB_MIN_S 기본 4 지만, 이 golden
# set 은 S 가 희박하다(probe 2026-06-08: 298 recipe 중 ≥4 는 1개뿐, 135개가 정확히 3장).
# 4 면 recipe 1개만 통과해 A/B 가 무의미 → 3 으로 낮춰 ~136 recipe(~411 LOO)를 살린다.
# 단, S=3 recipe 는 LOO consensus 가 2장(others)으로 빌드돼 약하다 — blur 가드/lift 로 판정.
MIN_S_FLOOR = 3   # LOO 바닥: len(others)>=2 가드상 fm>=3 이어야 점이 하나라도 난다 → 2 이하는 무의미.
_MIN_S_ENV = int(os.getenv("CONSENSUS_MIN_S", "3"))


def _floor_min_s(value):
    """min_s 를 바닥 MIN_S_FLOOR(3)로 보정한다.

    align_similarity LOO 는 `len(others)>=2` 를 요구하므로 같은 modality fm>=3 이어야 LOO 점이
    하나라도 난다. 2 이하를 줘도 결과가 동일(=조용한 no-op)하므로 3 으로 올려 knob 을 정직하게.
    """
    return max(MIN_S_FLOOR, value)


CONSENSUS_MIN_S = _floor_min_s(_MIN_S_ENV)


def _align_to_ref(img, ref):
    """img 를 ref 에 sub-pixel 평행이동 정렬(phase correlation). 과도 shift 면 원본 반환."""
    h, w = ref.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(img.astype(np.float32), ref.astype(np.float32), win)
    if abs(dx) > COREG_MAX_SHIFT_FRAC * w or abs(dy) > COREG_MAX_SHIFT_FRAC * h:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def coregister_crops(crops):
    """crop 들을 공통 reference(다듬어진 median)에 sub-pixel 정렬해 median 을 또렷하게.

    매 iter 마다 ref=median(현재 정렬본) 으로 갱신하되, 정렬은 항상 *원본* 에서 한 번만
    적용해 보간 blur 누적을 막는다. crop 2장 미만이면 그대로 반환.
    """
    if len(crops) < 2:
        return crops
    aligned = list(crops)
    for _ in range(COREG_ITERS):
        ref = np.median(np.stack([a.astype(np.float32) for a in aligned]), 0).astype(np.uint8)
        aligned = [_align_to_ref(c, ref) for c in crops]
    return aligned


def _cleaned_frame_loader(f):
    """LOO 매칭 프레임 로더 — cond 구동 clean_image 로 crosshair(+box)를 지운 프레임.

    `_consensus_template_ab(frame_loader=...)` 주입용. s_frames 는 `_precrop_drop_reason`
    가드를 통과한 것만 오므로 cond 는 항상 있지만, 방어적으로 없으면 raw 를 돌려준다.
    """
    gray = load_gray(f["path"])
    cond = load_cond(f["path"])
    return clean_image(gray, cond) if cond is not None else gray


def _cond_crosshair_xy(cond):
    """cond.crosshair_xy(cursor frame, ×10) → 이미지 px (x, y). 없으면 None."""
    if cond is None or cond.crosshair_xy is None:
        return None
    gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
    return (int(round(gx)), int(round(gy)))


def _cond_consensus_crop(gray, cond, size_wh):
    """crosshair(=align point) 중심·고정 size 의 *정제된*(crosshair 제거) crop. 없으면 None.

    consensus 재료. 모든 S 를 crosshair 로 정렬해 median 이 또렷해지게 하고, crop 전에
    clean_image 로 crosshair(+box) 를 지워 중앙 distractor 를 없앤다(cond 개선점).
    """
    xy = _cond_crosshair_xy(cond)
    if xy is None:
        return None
    cleaned = clean_image(gray, cond)        # crosshair(+box) 제거 후 자른다.
    w, h = size_wh
    return _matched_crop(cleaned, xy, w, h, 1.0)


# msr modality 추론(_msr_modality)·MAG 상수는 cond_file 공유 모듈로 이동(단일 출처).
# 여기서는 alias import 로 기존 이름(_msr_modality)을 보존한다.


def _resolve_mod(cond, recipe_mod):
    """msr 프레임 routing modality. 우선순위: msr 키/배율 추론 → recipe rcp modality 폴백.

    msr cond 엔 Scope 가 없으므로(사용자 확인) Scope 는 보지 않는다 — _msr_modality(키/배율)로
    결정하고, 미상이면 recipe 의 단일 rcp modality 로 폴백, 그것도 없으면 None(skip).
    이 단계가 과거 missing_modality 대량 누락(dual-rcp recipe + Scope 부재)을 해소한다.
    """
    return _msr_modality(cond) or recipe_mod


def _recipe_key(assets):
    """by_recipe dict 의 *고유* 키. assets.recipe_id 는 recipe_name(leaf)만이라 eqp/class 가
    달라도 leaf 이름이 같으면 dict 에서 덮어써져 데이터가 유실된다(probe: 298 dir → 276 고유).
    eqp/class/recipe 조합으로 고유화한다(라벨로도 읽기 좋음 — '/' 는 파일명에서 '__' 치환됨).
    """
    return f"{assets.eqp_id}/{assets.class_name}/{assets.recipe_id}"


def _precrop_drop_reason(cond, xy, mod, has_tpl):
    """S 프레임이 crop 이전 단계에서 빠지는 사유(없으면 None=채택). coverage 손실 가시화.

    조용히 continue 하던 지점들(코드리뷰 [4])을 사유별로 분류해 집계·경고하기 위함.
    우선순위: 근본 원인부터(cond 부재 → crosshair 부재 → modality 미상 → template 없음).
    """
    if cond is None:
        return "missing_cond"
    if xy is None:
        return "missing_crosshair"
    if mod is None:
        return "missing_modality"
    if not has_tpl:
        return "no_template"
    return None


def _build_cond_by_recipe(assets, center_tpls):
    """한 recipe → `_consensus_template_ab` 입력 항목.

    baseline rcp_tpls = center template(offset 0). s_frames 의 crop 은 cond crosshair
    중심·crosshair 제거된 고정 size(해당 modality center tpl 크기). E 는 가드용 경로만.
    """
    entry = {
        "rcp_tpls": {m: t for m, (t, _off) in center_tpls.items() if t is not None},
        "s_frames": [],
        "e_paths": [],
        "mod_counts": Counter(),     # *해결된* modality 분포(om/sem/unresolved) — 키/배율 추론 결과.
        "drop_counts": Counter(),    # S 프레임 누락 사유(coverage 손실 가시화, code-review [4]/[5]).
    }
    # recipe 의 rcp modality (단일이면 그것, om/sem 둘 다면 모호 → None). msr modality 미상 시 폴백.
    rcp_mods = [m for m, v in center_tpls.items() if v is not None]
    recipe_mod = rcp_mods[0] if len(rcp_mods) == 1 else None
    for p in iter_msr_images(assets):
        label = _tool_label(p.name)
        if label == "E":
            entry["e_paths"].append(p)
            continue
        if label != "S":
            continue
        cond = load_cond(p)
        mod = _resolve_mod(cond, recipe_mod)                           # msr 키/배율 → recipe 폴백(Scope 없음).
        entry["mod_counts"][mod or "unresolved"] += 1                  # 해결된 modality 분포.
        xy = _cond_crosshair_xy(cond)
        # 같은 modality center tpl 로 sizing(없으면 첫 가용 tpl) — omdf 등 drop 방지.
        tpl_item = center_tpls.get(mod) or next(
            (t for t in center_tpls.values() if t is not None), None)
        reason = _precrop_drop_reason(cond, xy, mod, tpl_item is not None)
        if reason:
            entry["drop_counts"][reason] += 1     # 조용히 버리지 않고 사유별 집계.
            continue
        tpl = tpl_item[0]
        size_wh = (tpl.raw_image.shape[1], tpl.raw_image.shape[0])
        try:
            gray = load_gray(p)
        except Exception as exc:
            print(f"[WARNING] msr 로드 실패 {p.name}: {exc}")
            entry["drop_counts"]["load_failed"] += 1
            continue
        crop = _cond_consensus_crop(gray, cond, size_wh)
        if crop is None:                          # OOB/너무 작음 — code-review [5].
            entry["drop_counts"]["crop_failed"] += 1
            continue
        entry["s_frames"].append({"path": p, "xy": xy, "mod": mod, "crop": crop})

    # co-registration: modality 별로(외형이 달라 섞으면 안 됨) crop 들을 sub-pixel 정렬해
    # median blur 를 줄인다. 정렬은 crop 내용만 바꾸고 crosshair GT(xy)·full-frame 은 불변.
    if COREGISTER and entry["s_frames"]:
        by_mod = defaultdict(list)
        for f in entry["s_frames"]:
            by_mod[f["mod"]].append(f)
        for fs in by_mod.values():
            for f, aligned in zip(fs, coregister_crops([f["crop"] for f in fs])):
                f["crop"] = aligned
    return entry


# --- 결합 패널(combined) — localization cond 판과 같은 "한 장 추적" 시각화 ----------
# 좌: [RCP center(baseline) / LOO consensus] 템플릿 세로 스택(외형·blur 비교),
# 우: msr 프레임 + GT(crosshair) + 양쪽 top-N 후보(주황=consensus, 시안=rcp).
# 색은 localization cond 판과 동일 톤(BGR): GT=초록, consensus 예측=주황, rcp 예측=시안.
_CMB_GT = (0, 200, 0)
_CMB_CONS = (0, 170, 255)
_CMB_RCP = (220, 180, 0)


def _resize_to_width(canvas, target_w):
    """aspect 유지하며 target_w 로 리사이즈(세로 스택용 — 이미 같으면 그대로)."""
    h, w = canvas.shape[:2]
    if w == target_w:
        return canvas
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(canvas, (target_w, new_h), interpolation=interp)


def _tpl_stack(cons_tpl, rcp_tpl, *, mod):
    """좌측 패널: rcp center(baseline) 위 + LOO consensus 아래 세로 스택(BGR, 라벨 띠 포함).

    같은 크기 crop 이지만(consensus sizing 이 center tpl 기준) 방어적으로 너비를 맞춘다.
    consensus 가 baseline 보다 또렷한지(blur 가드의 시각판)를 한눈에 비교하는 용도.
    """
    panels = []
    if rcp_tpl is not None:
        panels.append(glec._with_header(
            cv2.cvtColor(rcp_tpl.raw_image, cv2.COLOR_GRAY2BGR),
            f"RCP center {mod.upper()} (baseline)"))
    panels.append(glec._with_header(
        cv2.cvtColor(cons_tpl.raw_image, cv2.COLOR_GRAY2BGR),
        f"CONSENSUS LOO {mod.upper()}"))
    target_w = max(p.shape[1] for p in panels)
    return cv2.vconcat([_resize_to_width(p, target_w) for p in panels])


def _render_msr_canvas(gray, xy, gc, gr, *, recipe, msr_name):
    """msr 프레임 위에 GT(crosshair) + consensus/rcp top-N 후보를 그린 BGR canvas.

    rank-1 후보(=matcher 가 실제로 찍을 점)는 큰 마커 + GT 연결선, 나머지 후보는 작은 원.
    legend 에 in_topk rank / truth↔최근접 후보 거리(best_cand_dist_norm)를 함께 적는다.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    cx, cy = int(xy[0]), int(xy[1])
    cv2.drawMarker(canvas, (cx, cy), _CMB_GT, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(canvas, (cx, cy), 10, _CMB_GT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{recipe}/{msr_name}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    legend = [("GT (crosshair)", _CMB_GT)]
    for name, res, col in (("cons", gc, _CMB_CONS), ("rcp", gr, _CMB_RCP)):
        if not res or not res.get("cand_xys"):
            continue
        xs = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
              for x, y in res["cand_xys"]]
        for px, py in xs[1:]:
            cv2.circle(canvas, (px, py), 4, col, 1, cv2.LINE_AA)
        ax, ay = xs[0]
        cv2.line(canvas, (cx, cy), (ax, ay), col, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (ax, ay), col, cv2.MARKER_TILTED_CROSS, 18, 2)
        rank = res.get("topk_rank")
        legend.append((f"{name} top{len(xs)} rank={rank if rank is not None else 'MISS'} "
                       f"d={res['best_cand_dist_norm']}", col))
    for i, (text, col) in enumerate(legend):
        cv2.putText(canvas, text, (6, 38 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return canvas


def _make_combined_renderer(combined_dir, *, frame_tag="raw"):
    """`_consensus_template_ab` 의 combined_renderer 훅 — LOO 한 점당 결합 패널 1장 저장.

    [좌: 템플릿 스택 | 우: msr 후보 overlay] 를 같은 높이로 붙여
    combined/<recipe('/'→'__')>/<msr>_<mod>_combined.jpg 로 쓴다(consensus/ PNG 와 동일 키).
    ctx["gray"] 는 측정에 실제로 쓴 프레임이므로 CLEAN_FRAME 판에선 정제 프레임이 그려진다
    — frame_tag(raw/clean)를 제목에 박아 어느 판인지 그림만 봐도 알게 한다.
    """
    def _render(ctx):
        msr = _render_msr_canvas(
            ctx["gray"], ctx["xy"], ctx["gc"], ctx["gr"],
            recipe=ctx["recipe"], msr_name=f"{ctx['path'].name} [{frame_tag}]")
        left = _tpl_stack(ctx["cons_tpl"], ctx["rcp_tpl"], mod=ctx["mod"])
        target_h = max(left.shape[0], msr.shape[0])
        left = glec._resize_to_height(left, target_h)
        msr = glec._resize_to_height(msr, target_h)
        sep = np.full((target_h, glec._PANEL_SEP_PX, 3), glec._PANEL_SEP_BGR, dtype=np.uint8)
        panel = cv2.hconcat([left, sep, msr])
        sub = combined_dir / ctx["recipe"].replace("/", "__")
        sub.mkdir(parents=True, exist_ok=True)
        out = sub / f"{ctx['path'].stem}_{ctx['mod']}_combined.jpg"
        cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return _render


def run() -> str:
    """consensus(cond) A/B 실행 (인자 없음). 반환: success | no_data | no_ab."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터를 찾지 못했습니다: {root} (env ALIGN_GOLDEN_ROOT).")
        return "no_data"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] (consensus cond A/B) recipe {len(recipes)}개 → {out_dir}")
    print(f"[INFO] co-registration: {'ON' if COREGISTER else 'OFF'} "
          f"(env CONSENSUS_COREGISTER=0 으로 끄고 A/B 비교 가능)")
    print(f"[INFO] 매칭 프레임: {'CLEAN(crosshair 제거)' if CLEAN_FRAME else 'RAW(crosshair 잔존)'} "
          f"(env CONSENSUS_CLEAN_FRAME=0 이면 raw — crosshair 가짜 lock A/B 용)")
    print(f"[INFO] proposer: {'ENSEMBLE (C1 canny + C2 scharr + C3 orient, RRF)' if USE_ENSEMBLE_PROPOSER else 'C1 (canny chamfer)'} "
          f"(env CONSENSUS_USE_ENSEMBLE=1 이면 ensemble — 기본 0=C1; in_topk A/B 용)")

    if _MIN_S_ENV < CONSENSUS_MIN_S:   # 2 이하는 LOO 가 못 나와 무의미 → 바닥 3 으로 보정됨.
        print(f"[WARNING] CONSENSUS_MIN_S={_MIN_S_ENV} 는 무의미(LOO 바닥 fm>=3) → "
              f"{CONSENSUS_MIN_S} 로 보정합니다.")

    by_recipe = {}
    mod_total = Counter()
    drop_total = Counter()
    periodicities = []   # [(rec_key, periodicity)] — Phase 1 재등록 후보 신호.
    for assets in recipes:
        if assets is None:
            continue
        try:
            center_tpls, _box = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        if not any(v is not None for v in center_tpls.values()):
            continue
        entry = _build_cond_by_recipe(assets, center_tpls)
        mod_total.update(entry["mod_counts"])
        drop_total.update(entry["drop_counts"])
        n_s = len(entry["s_frames"])
        n_drop = sum(entry["drop_counts"].values())
        rec_key = _recipe_key(assets)          # eqp/class/recipe 고유 키 — leaf 충돌 방지.
        # template 내재 모호성(Phase 1): 등록 key 가 주기/대칭이면 유일 위치가 없음 → 재등록 후보.
        # modality 중 max(가장 모호한 쪽)로 recipe 를 대표.
        rec_periodicity = max(
            (template_periodicity(t.raw_image) for t, _off in center_tpls.values()
             if t is not None), default=0.0)
        periodicities.append((rec_key, round(rec_periodicity, 3)))
        if n_s:
            by_recipe[rec_key] = entry
        print(f"[INFO] {rec_key}: S(crosshair) {n_s}장, E {len(entry['e_paths'])}장"
              + (f"  [누락 {n_drop}: {dict(entry['drop_counts'])}]" if n_drop else ""))

    # === template 내재 모호성(Phase 1): 재등록 후보 비율 ===
    n_periodic = sum(1 for _k, p in periodicities if p > PERIODICITY_TAU)
    n_tpl = len(periodicities)
    periodic_rate = round(n_periodic / n_tpl, 3) if n_tpl else 0.0
    worst = sorted(periodicities, key=lambda kp: kp[1], reverse=True)
    print(f"\n[INFO] === template 모호성(재등록 후보, tau={PERIODICITY_TAU}) === "
          f"periodic {n_periodic}/{n_tpl} (rate={periodic_rate}) — 상위: "
          + (", ".join(f"{k}={p}" for k, p in worst[:5]) or "(없음)"))

    # msr cond 엔 Scope 가 없어 키/배율로 modality 를 가른다(Scope 분포는 더 이상 추적 안 함).
    print(f"\n[INFO] === msr S 해결된 modality(키/배율) === "
          f"om={mod_total.get('om', 0)} sem={mod_total.get('sem', 0)} "
          f"unresolved={mod_total.get('unresolved', 0)} "
          f"(OM=!OM_Brightness/Mag<200, SEM=Accelerating_voltage/Mag>500)")

    # S 프레임 누락 집계(coverage 손실 가시화) — 조용한 skip 추방(code-review [4]/[5]).
    n_kept = sum(len(e["s_frames"]) for e in by_recipe.values())
    n_dropped = sum(drop_total.values())
    if n_dropped:
        print(f"[WARNING] === S 프레임 누락 {n_dropped}장 (채택 {n_kept}장) === "
              f"{dict(drop_total.most_common())}")
        print("    missing_cond=sidecar 없음 · missing_crosshair=cond 에 crosshair 없음 · "
              "missing_modality=키/배율·recipe 둘 다 미상 · no_template=center tpl 없음 · "
              "crop_failed=OOB/너무작음 · load_failed=이미지 로드 실패")
    else:
        print(f"[INFO] S 프레임 누락 없음 (채택 {n_kept}장).")

    # 결합 패널(rcp/consensus 템플릿 | msr 후보) — localization cond 판과 동일한 "한 장 추적".
    combined_dir = (out_dir / "combined") if glec.SAVE_OVERLAYS else None
    renderer = None
    if combined_dir is not None:
        combined_dir.mkdir(parents=True, exist_ok=True)
        renderer = _make_combined_renderer(
            combined_dir, frame_tag="clean" if CLEAN_FRAME else "raw")

    res = _consensus_template_ab(
        by_recipe, min_s=CONSENSUS_MIN_S, out_dir=out_dir,
        combined_renderer=renderer,
        frame_loader=_cleaned_frame_loader if CLEAN_FRAME else None)
    if res is None:
        print(f"[ERROR] consensus A/B 불가 — LOO 가능한(같은 modality ≥{CONSENSUS_MIN_S}) recipe 가 없음.")
        return "no_ab"

    res["coregister"] = COREGISTER      # min_s 는 _consensus_template_ab 가 이미 반환에 넣는다.
    res["clean_frame"] = CLEAN_FRAME    # 매칭 프레임 정제 여부 — raw 판과 lift 비교 키.
    res["proposer"] = "ensemble" if USE_ENSEMBLE_PROPOSER else "c1"   # C1 vs ensemble A/B 키.
    res["template_periodic_rate"] = periodic_rate              # Phase 1 재등록 후보 비율.
    res["template_periodicities"] = dict(periodicities)        # recipe 별 모호성(재등록 우선순위).
    res["modality_distribution"] = dict(mod_total)
    res["drop_distribution"] = dict(drop_total)
    # Phase1 보정: per-recipe periodicity ↔ per-point miss(in_topk=False) 결합 → 예측력(AUC)/Youden tau.
    # per_recipe 의 cons_in_topk_rate·n_S_loo 로 hit/miss 를 복원(같은 recipe 점들은 periodicity 동일).
    _per_period = dict(periodicities)
    _cal_scores, _cal_missed = [], []
    for pr in res.get("per_recipe", []):
        p = _per_period.get(pr["recipe"])
        if p is None:
            continue
        n = pr["n_S_loo"]
        n_miss = max(0, min(n, int(round(n * (1.0 - pr["cons_in_topk_rate"])))))
        _cal_scores.extend([p] * n)
        _cal_missed.extend([True] * n_miss + [False] * (n - n_miss))
    res["periodicity_miss_calibration"] = miss_predictor_stats(_cal_scores, _cal_missed)
    (out_dir / "summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    rcp = res["overall_rcp_in_topk_rate"]
    cons = res["overall_cons_in_topk_rate"]
    lift = res["overall_lift"]
    print("\n" + "=" * 64)
    print("[INFO] === consensus 재등록 A/B (cond, LOO; 이 블록만 읽어주면 됨) ===")
    print(f"  recipes={res['n_recipes']}  S_loo={res['n_S_loo']}  min_s={CONSENSUS_MIN_S}  "
          f"(baseline=center tpl, offset0, co-reg={'ON' if COREGISTER else 'OFF'}, "
          f"frame={'CLEAN' if CLEAN_FRAME else 'RAW'}, "
          f"proposer={'ENSEMBLE' if USE_ENSEMBLE_PROPOSER else 'C1'})")
    if CLEAN_FRAME:
        print("  * frame=CLEAN: crosshair 가짜 lock 차단판. raw(CONSENSUS_CLEAN_FRAME=0) 대비 "
              "lift 유지=진짜 / 급락=과거 lift 는 crosshair artifact.")
    if CONSENSUS_MIN_S < 4:
        print(f"  ⚠ min_s={CONSENSUS_MIN_S}: S={CONSENSUS_MIN_S} recipe 의 LOO consensus 는 "
              f"{CONSENSUS_MIN_S - 1}장으로 빌드돼 약함 — lift 가 양수여도 blur 가드 함께 확인 "
              f"(env CONSENSUS_MIN_S=4 로 강한판 비교 가능).")
    print(f"  in_topk:  rcp(center)={rcp}  →  consensus={cons}   lift={lift:+}")
    print(f"  rank1:    rcp(center)={res['overall_rcp_rank1_rate']}  →  "
          f"consensus={res['overall_cons_rank1_rate']}   lift={res['rank1_lift']:+}")
    print(f"  blur 가드(낮으면 median 흐림): edge_ratio_to_S="
          f"{res['cons_edge_density_ratio_to_S_median']}  lap_ratio_to_S="
          f"{res['cons_lap_var_ratio_to_S_median']}")
    # consensus 잔여 miss 분포 — "ensemble 을 consensus 에 얹으면 더 오르나" 의 사전 진단.
    md = res.get("cons_miss_dist_distribution")
    if md and md.get("n"):
        print(f"  consensus miss 거리(truth↔최근접 후보, short-side 상대; tol={GT_TOL_NORM}):")
        print(f"    n_miss={md['n']}  median={md['median']}  p25={md['p25']}  "
              f"p75={md['p75']}  max={md['max']}")
        print(f"    bins={md['bins']}")
        print("    * near 多 → ensemble 후보 recall 로 끌어들일 여지(시도 가치) / "
              "far·veryfar 多 → 구조적 모호성(ensemble 무력, 다른 축 필요)")
    verdict = ("CONSENSUS 채택 권장(lift≥+0.05)" if lift >= 0.05
               else "효과 미미/음수 — proposer ensemble 또는 VLM-region 로 전환 검토")
    print(f"  >>> 판정: {verdict}")
    print("=" * 64)
    # Phase1 보정 — periodicity 가 miss 를 예측하나 (재등록 신호 검증 + tau 보정).
    cal = res.get("periodicity_miss_calibration") or {}
    print("\n[INFO] === Phase1 보정: periodicity → miss 예측력 (per-point) ===")
    print(f"  n={cal.get('n')}  miss={cal.get('n_miss')}  hit={cal.get('n_hit')}")
    print(f"  mean periodicity: miss={cal.get('mean_miss')}  hit={cal.get('mean_hit')}  (miss>hit 여야 신호)")
    print(f"  AUC={cal.get('auc')}  (0.5=무신호, >=0.7 쓸만, >=0.8 강함)")
    print(f"  Youden tau*={cal.get('best_tau')}  (TPR={cal.get('tpr')} FPR={cal.get('fpr')})  vs 현재 tau={PERIODICITY_TAU}")
    if cal.get("auc") is not None:
        _v = ("쓸만함 → tau* 로 재등록 후보 보정 가능" if cal["auc"] >= 0.7
              else "약함/무신호 → periodicity metric 재설계 필요(제외반경/매칭영역 한정 등)")
        print(f"  >>> 판정: AUC {cal['auc']} → {_v}")
    print("=" * 64)
    print(f"\n[INFO] 완료: {out_dir}  (consensus 템플릿: {out_dir}/consensus/"
          + (f", 결합 패널: {combined_dir}/" if combined_dir is not None else "")
          + ")")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
