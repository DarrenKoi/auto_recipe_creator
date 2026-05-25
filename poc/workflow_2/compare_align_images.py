"""Step 3 — recipe 등록 SEM align 이미지와 현재 실패 SEM 이미지를 classical CV 로 비교.

Align fail 은 *대개 live key 가 등록 이미지와 다르게 보여서* 발생하므로, 픽셀
동일성이 아니라 edge 구조(Chamfer 위주, ``STRUCTURE_POLICY``)로 "등록된 key 가
현재 이미지의 어디에 있을 법한지" 를 점수화한다. 최종 판정은 hard match 가
아니라 best-candidate score + 사람이 확인할 overlay 다.

입력: ``ALIGN_FAIL_DOWNLOAD_DIR/<recipe_id>/`` 의 recipe_sem.* 와 current_sem.*
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
from poc.workflow_2.align_fail_assets import latest_recipe_dir, load_gray, resolve_assets
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수로만 조정.
# ====================================================================
# 비우면 다운로드 루트에서 가장 최근 align fail 폴더를 자동 선택한다.
RECIPE_ID_OVERRIDE = r""

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
    recipe_id = (RECIPE_ID_OVERRIDE or "").strip() or (latest_recipe_dir() or "")

    tag = time.strftime("%y%m%d_%H%M%S")
    out_dir = DEBUG_IMAGE_DIR / "compare_align_images" / f"{tag}_{recipe_id or 'selftest'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe_sem = current_sem = None
    recipe_sem_path = current_sem_path = ""

    if recipe_id:
        assets = resolve_assets(recipe_id)
        if assets.recipe_sem is not None and assets.current_sem is not None:
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
    print(f"[INFO] elapsed={time.time() - started:.2f}s, out_dir={out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
