"""Step 3 — recipe 등록 SEM align 이미지와 현재 실패 SEM 이미지를 classical CV 로 비교.

Align fail 은 *대개 live key 가 등록 이미지와 다르게 보여서* 발생하므로, 픽셀
동일성이 아니라 edge 구조(Chamfer 위주, ``STRUCTURE_POLICY``)로 "등록된 key 가
현재 이미지의 어디에 있을 법한지" 를 점수화한다. 최종 판정은 hard match 가
아니라 best-candidate score + 사람이 확인할 overlay 다.

입력: `align_fail_assets.resolve_assets_auto()` 가 해석한 recipe 의
recipe_sem(from_rcp/IMAP0002) 와 current_sem(from_msr 최신 E*).
(없으면 합성 self-test 로 파이프라인만 점검).

실행:
    uv run python poc/workflow_2/compare_align_images.py
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import load_gray, resolve_assets_auto
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정.
# ====================================================================
# recipe 폴더 선택은 align_fail_assets 가 담당(환경변수 override 또는 최신 자동).

# 실제 다운로드 자산이 없을 때 합성 데이터로 파이프라인을 점검할지 여부.
RUN_SELFTEST_IF_NO_ASSETS = True

# 정적 비교용 scale band — 같은 배율 가정이므로 1.0 주변만 본다.
# (live 미니어처 탐색의 BROAD_SCALES 와는 목적이 다르다.)
COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)

# template 이 항상 search 이미지 안에 들어가도록 두를 replicate 여백 비율
# (template 한 변 대비). 0.3 이면 frame 을 template 의 약 1.3배로 키운다.
PAD_RATIO = 0.35


@dataclass
class CompareReport:
    """recipe ↔ current 비교 결과."""

    recipe_id: str
    recipe_sem_path: str
    current_sem_path: str
    score: float
    chamfer_score: float
    orb_inlier_ratio: float
    best_scale: float
    best_xy_in_current: tuple[int, int]
    decision: str
    verdict: str
    overlay_path: str
    side_by_side_path: str


def _pad_frame(frame: np.ndarray, template_shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """template 이 항상 frame 안에 들어가도록 replicate border 로 여백을 두른다.

    반환: (padded_frame, pad_x, pad_y). best_xy 를 원본 좌표로 되돌릴 때 pad 만큼 뺀다.
    """
    th, tw = template_shape[:2]
    pad_x = int(round(tw * PAD_RATIO))
    pad_y = int(round(th * PAD_RATIO))
    padded = cv2.copyMakeBorder(
        frame, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
    )
    return padded, pad_x, pad_y


# 판정별 헤더 색상 (BGR). cv2 는 ASCII 만 그리므로 한글 verdict 는 이미지에 넣지 않는다.
_DECISION_BGR = {
    "match": (60, 170, 60),    # green
    "adjust": (40, 170, 220),  # amber
    "low": (60, 60, 200),      # red
}


def _label_panel(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """패널 좌상단에 반투명 라벨 바를 얹는다 (어느 쪽이 recipe/current 인지 표시)."""
    out = img_bgr.copy()
    bar_h = 26
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def _fit_height(img_bgr: np.ndarray, target_h: int) -> np.ndarray:
    """종횡비를 유지하며 height 를 target_h 로 맞춘다 (side-by-side concat 전 정규화)."""
    ih, iw = img_bgr.shape[:2]
    scale = target_h / ih
    new_w = max(1, int(round(iw * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, target_h), interpolation=interp)


def _build_side_by_side(
    recipe_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    decision: str,
    score: float,
    chamfer: float,
    orb: float,
    best_scale: float,
) -> np.ndarray:
    """recipe(왼쪽) vs current(오른쪽) 를 같은 높이로 붙이고, 위에 점수 헤더를 얹는다."""
    left = cv2.cvtColor(recipe_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)

    target_h = max(left.shape[0], right.shape[0])
    left = _label_panel(_fit_height(left, target_h), "RECIPE (from_rcp)")
    right = _label_panel(_fit_height(right, target_h), "CURRENT (from_msr)")

    gap = np.full((target_h, 4, 3), (40, 40, 40), dtype=np.uint8)
    body = np.hstack([left, gap, right])

    # 점수 헤더 — 판정 색으로 채우고 ASCII 수치를 두 줄로.
    header_h = 64
    header_color = _DECISION_BGR.get(decision, (90, 90, 90))
    header = np.full((header_h, body.shape[1], 3), header_color, dtype=np.uint8)
    cv2.putText(
        header, f"decision={decision.upper()}  score={score:.3f}",
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        header, f"chamfer={chamfer:.3f}  orb={orb:.3f}  scale={best_scale:.2f}",
        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([header, body])


def _verdict_for(decision: str, score: float) -> str:
    """사람이 읽을 한 줄 판정 문구."""
    if decision == "match":
        return "등록 key 가 현재 이미지에서 강하게 일치 - align fail 원인은 위치/스테이지일 가능성"
    if decision == "adjust":
        return "구조는 유사하나 drift 존재 - 등록 위치 근처에 key 후보 있음(보정 권장)"
    return "현재 이미지에서 등록 key 와 닮은 구조를 찾지 못함 - live 탐색(step 4~7) 필요"


def compare_pair(
    recipe_sem: np.ndarray,
    current_sem: np.ndarray,
    *,
    recipe_id: str,
    recipe_sem_path: str,
    current_sem_path: str,
    out_dir: Path,
) -> CompareReport:
    """recipe SEM 이미지를 template 으로, current SEM 이미지를 search 대상으로 비교."""
    template = build_template(
        recipe_sem,
        recipe_id=recipe_id,
        version="downloaded",
        nm_per_pixel=None,
        key_type="sem",
    )

    padded, pad_x, pad_y = _pad_frame(current_sem, template.raw_image.shape)
    result = compute_align_key_score(
        template,
        padded,
        scales=COMPARE_SCALES,
        policy=STRUCTURE_POLICY,
    )

    # best_xy 를 원본 current 좌표계로 환원.
    bx, by = result.best_xy
    best_xy_current = (int(bx - pad_x), int(by - pad_y))

    overlay_path = out_dir / "compare_overlay.jpg"
    save_overlay_jpeg(result.debug_overlay, overlay_path)

    side_by_side = _build_side_by_side(
        recipe_sem,
        current_sem,
        decision=result.decision,
        score=result.score,
        chamfer=result.chamfer_score,
        orb=result.orb_inlier_ratio,
        best_scale=result.best_scale,
    )
    side_by_side_path = out_dir / "compare_side_by_side.jpg"
    cv2.imwrite(str(side_by_side_path), side_by_side)

    verdict = _verdict_for(result.decision, result.score)
    report = CompareReport(
        recipe_id=recipe_id,
        recipe_sem_path=recipe_sem_path,
        current_sem_path=current_sem_path,
        score=float(result.score),
        chamfer_score=float(result.chamfer_score),
        orb_inlier_ratio=float(result.orb_inlier_ratio),
        best_scale=float(result.best_scale),
        best_xy_in_current=best_xy_current,
        decision=result.decision,
        verdict=verdict,
        overlay_path=str(overlay_path),
        side_by_side_path=str(side_by_side_path),
    )
    return report


# ------------------------------------------------------------------
# 합성 self-test — 실제 자산이 없을 때 파이프라인만 점검.
# ------------------------------------------------------------------


def _make_selftest_pair() -> tuple[np.ndarray, np.ndarray]:
    """등록 key(template) 와, 그 key 가 drift+노이즈된 current 이미지를 만든다."""
    from poc.workflow_2.test_align_key_match import (  # 재사용: 합성 패턴/배경 생성기.
        embed_pattern,
        make_synthetic_template,
        make_wafer_background,
    )

    pattern = make_synthetic_template(key_type="box")
    # recipe 등록 이미지: key + 약간의 주변 context(작은 배경 테두리).
    recipe = cv2.copyMakeBorder(pattern, 24, 24, 24, 24, cv2.BORDER_REPLICATE)

    # current: 같은 key 를 drift(회전/콘트라스트/노이즈)시켜 배경에 박는다.
    bg = make_wafer_background()
    current, _gt, _w, _h = embed_pattern(
        bg, pattern,
        rotation_deg=3.0, scale=0.92, brightness=-12, contrast=0.8, rng_seed=99,
    )
    return recipe, current


def run() -> str:
    started = time.time()
    assets = resolve_assets_auto()
    recipe_id = assets.recipe_id if assets is not None else ""

    tag = time.strftime("%y%m%d_%H%M%S")
    out_dir = DEBUG_IMAGE_DIR / "compare_align_images" / f"{tag}_{recipe_id or 'selftest'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe_sem = current_sem = None
    recipe_sem_path = current_sem_path = ""

    if assets is not None and assets.recipe_sem is not None and assets.current_sem is not None:
        recipe_sem = load_gray(assets.recipe_sem)
        current_sem = load_gray(assets.current_sem)
        recipe_sem_path = str(assets.recipe_sem)
        current_sem_path = str(assets.current_sem)

    if recipe_sem is None or current_sem is None:
        if not RUN_SELFTEST_IF_NO_ASSETS:
            print("[ERROR] recipe_sem/current_sem 자산을 찾지 못했고 self-test 도 꺼져 있습니다.")
            return "no_assets"
        print("[WARNING] 실제 자산 없음 → 합성 self-test 로 파이프라인 점검")
        recipe_sem, current_sem = _make_selftest_pair()
        recipe_id = recipe_id or "SELFTEST"
        recipe_sem_path = "<selftest:recipe_sem>"
        current_sem_path = "<selftest:current_sem>"

    report = compare_pair(
        recipe_sem,
        current_sem,
        recipe_id=recipe_id,
        recipe_sem_path=recipe_sem_path,
        current_sem_path=current_sem_path,
        out_dir=out_dir,
    )

    (out_dir / "compare_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[INFO] compare: decision={report.decision} score={report.score:.3f} "
        f"(chamfer={report.chamfer_score:.3f} orb={report.orb_inlier_ratio:.3f}) "
        f"best_scale={report.best_scale:.2f} best_xy_current={report.best_xy_in_current}"
    )
    print(f"[INFO] verdict: {report.verdict}")
    print(f"[INFO] overlay: {report.overlay_path}")
    print(f"[INFO] side_by_side: {report.side_by_side_path}")
    print(f"[INFO] elapsed={time.time() - started:.2f}s, out_dir={out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
