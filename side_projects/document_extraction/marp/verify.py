"""Stage 7: 재렌더 검증(SSIM) + 자동 강등 결정 (marp_roundtrip_design.md Stage 7).

원본 슬라이드 캡처 vs Marp 재렌더 이미지의 구조 유사도(SSIM)를 측정하고, 임계
미달 슬라이드를 어떻게 강등(시각 영역 -> 래스터 crop, 최후엔 슬라이드 전체 래스터)
할지 결정한다. **자동 강등이 충실도의 안전망**(design Stage 7).

설계 원칙: 순수 결정 로직(ssim/flag/plan)은 numpy 만으로 집에서 검증되고,
실제 render+score 루프(verify_and_downgrade)는 marp-cli/이미지가 필요해 office
에서 돈다(없으면 graceful degrade). skimage 비의존 — SSIM 은 numpy 로 직접.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from side_projects.document_extraction.marp import generate as _generate
from side_projects.document_extraction.marp.generate import evidence_to_marp
from side_projects.document_extraction.marp.render import render_deck

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:  # 리사이즈 폴백(아래 _resize_to)에서 안내.
    _PIL_AVAILABLE = False


# SSIM 안정화 상수(Wang et al. 2004, 데이터 레인지 L=255 기준).
_SSIM_C1 = (0.01 * 255.0) ** 2
_SSIM_C2 = (0.03 * 255.0) ** 2

# 슬라이드 충실도 floor(이 미만이면 강등 대상). design Q&A 확정값; office 보정 가능.
DEFAULT_SSIM_FLOOR = 0.90


def _to_gray_float(img) -> "np.ndarray":
    """이미지를 그레이스케일 float64 2D 로. RGB/RGBA 는 luminance 가중 평균."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        # alpha 채널 무시, BT.601 luminance.
        rgb = arr[:, :, :3]
        arr = rgb @ np.array([0.299, 0.587, 0.114])
    return arr


def ssim(a, b) -> float:
    """두 이미지의 전역 SSIM(단일 윈도우, 순수 numpy). 동일 -> 1.0, 무관 -> ~0.

    Wang et al. 2004 의 luminance·contrast·structure 곱. skimage 없이 전체
    이미지를 하나의 통계 윈도우로 계산한다(슬라이드 단위 충실도엔 전역으로 충분).
    크기가 다르면 호출측(slide_fidelity)에서 맞춘 뒤 넘긴다.
    """
    ga = _to_gray_float(a)
    gb = _to_gray_float(b)
    if ga.shape != gb.shape:
        raise ValueError(f"ssim: shape mismatch {ga.shape} vs {gb.shape}")
    mu_a = ga.mean()
    mu_b = gb.mean()
    va = ga.var()
    vb = gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    num = (2 * mu_a * mu_b + _SSIM_C1) * (2 * cov + _SSIM_C2)
    den = (mu_a**2 + mu_b**2 + _SSIM_C1) * (va + vb + _SSIM_C2)
    return float(num / den)


def _resize_to(img, shape_hw):
    """img 를 (height, width)=shape_hw 그레이스케일로 리사이즈한다. PIL 우선,
    없으면 numpy 최근접 이웃 폴백(의존성 추가 없이도 동작)."""
    gray = _to_gray_float(img)
    th, tw = shape_hw
    if gray.shape == (th, tw):
        return gray
    if _PIL_AVAILABLE:
        pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        pil = pil.resize((tw, th), Image.BILINEAR)
        return np.asarray(pil, dtype=np.float64)
    # 폴백: 최근접 이웃(테스트/오프라인 안전망).
    sh, sw = gray.shape
    ys = (np.arange(th) * sh / th).astype(int).clip(0, sh - 1)
    xs = (np.arange(tw) * sw / tw).astype(int).clip(0, sw - 1)
    return gray[np.ix_(ys, xs)]


def slide_fidelity(original, rendered) -> float:
    """원본 캡처 vs 재렌더 이미지의 충실도(SSIM). 렌더 해상도가 달라도 원본
    크기에 맞춰 리사이즈 후 비교하므로 marp 출력 DPI 에 무관하다. [0,1]."""
    orig_gray = _to_gray_float(original)
    rendered_gray = _resize_to(rendered, orig_gray.shape)
    return ssim(orig_gray, rendered_gray)


@dataclass
class DowngradePlan:
    """저충실도 슬라이드 1장의 강등 계획(순수 결정 결과).

    demote_region_ids: 네이티브->래스터로 강등할 차트 region_id(가용 crop 있음).
    whole_slide: 부분 강등으로 못 살릴 때 슬라이드 전체를 원본 캡처 래스터로 대체
                 (최후 안전망, design Q&A 확정).
    capture_path: whole_slide 시 사용할 원본 슬라이드 캡처 경로.
    slide_index: deck 내 0-기반 슬라이드 위치.
    """

    slide_index: int
    demote_region_ids: list = field(default_factory=list)
    whole_slide: bool = False
    capture_path: str = ""


def plan_downgrade(result, crop_lookup, *, available_crops, capture_path, slide_index):
    """저충실도 슬라이드를 어떻게 강등할지 결정한다(순수). design Stage 7 안전망.

    1) 네이티브 데이터표로 렌더된 차트(crop_lookup 에 없음) 중 가용 crop 이 있는
       region 을 래스터로 강등(부분 강등 우선 — 편집성 보존).
    2) 살릴 region 이 없으면(차트 없음/이미 다 raster/ crop 부재) 슬라이드 전체를
       원본 캡처 래스터로 강등(최후 안전망).
    텍스트/레이아웃 mismatch 의 OCR 재합성 분기는 office 후속(여기선 전체 래스터로 보증).
    """
    crop_lookup = crop_lookup or {}
    available_crops = available_crops or {}
    demote = [
        c.region_id
        for c in result.charts
        if c.region_id not in crop_lookup and c.region_id in available_crops
    ]
    if demote:
        return DowngradePlan(slide_index=slide_index, demote_region_ids=demote,
                             whole_slide=False, capture_path=capture_path)
    return DowngradePlan(slide_index=slide_index, demote_region_ids=[],
                         whole_slide=True, capture_path=capture_path)


def flag_low_fidelity(scores, *, threshold=DEFAULT_SSIM_FLOOR):
    """SSIM 점수 리스트에서 floor 미만인 슬라이드 인덱스들을 반환(순수). floor 동률은
    통과(>= floor). 반환 인덱스는 강등 루프의 대상."""
    return [i for i, s in enumerate(scores) if float(s) < threshold]


def whole_slide_marp(capture_path) -> str:
    """슬라이드 전체를 원본 캡처 래스터로 대체하는 Marp 슬라이드(순수). 편집성을
    포기하는 대신 시각 충실도를 보증한다 — 최후 안전망(design Q&A). Marp 전체배경
    directive `![bg fit]` 로 전면 표시."""
    return f"![bg fit]({capture_path})"


def apply_downgrade_plans(results, crop_lookups, plans, *, available_crops=None,
                          with_frontmatter=True, theme="default") -> str:
    """강등 계획을 반영해 deck(.md) 를 다시 만든다(순수). 부분 강등은 해당 차트
    region 의 crop 을 주입해 데이터표 대신 이미지로, 전체 강등은 슬라이드를 원본
    캡처 래스터로 대체. 계획 없는 슬라이드는 Stage 5 그대로.

    crop_lookups: {screenshot_id -> {region_id -> 경로}} (기존 deck 의 crop).
    available_crops: {region_id -> 경로} (강등 시 주입할 가용 crop, deck 전역 유일 가정).
    plans: DowngradePlan 리스트(slide_index 로 results 와 정렬).
    theme: 원본 deck 과 같은 테마 유지(보정 deck 의 프론트매터).
    """
    crop_lookups = crop_lookups or {}
    available_crops = available_crops or {}
    plans_by_idx = {p.slide_index: p for p in plans}
    slides = []
    for idx, result in enumerate(results):
        plan = plans_by_idx.get(idx)
        if plan is not None and plan.whole_slide:
            slides.append(whole_slide_marp(plan.capture_path))
            continue
        lookup = dict(crop_lookups.get(result.screenshot_id, {}))
        if plan is not None:
            for rid in plan.demote_region_ids:
                if rid in available_crops:
                    lookup[rid] = available_crops[rid]
        slide = evidence_to_marp(result, crop_lookup=lookup)
        if slide.strip():
            slides.append(slide)
    body = "\n\n---\n\n".join(slides)
    if not with_frontmatter:
        return body
    return _generate.frontmatter_for_theme(theme) + "\n" + body + "\n"


def _load_image(path):
    """이미지 파일을 numpy 배열로(I/O). PIL 필요; 없으면 None(graceful)."""
    if not _PIL_AVAILABLE:
        print("[WARNING] PIL 부재 - 이미지 로드 불가, SSIM 검증 건너뜀.")
        return None
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float64)
    except Exception as exc:   # noqa: BLE001 - 로드 실패는 graceful.
        print(f"[WARNING] 이미지 로드 실패 {path}: {exc}")
        return None


def verify_and_downgrade(results, deck_path, capture_paths, *, out_dir,
                         crop_lookups=None, available_crops=None,
                         threshold=DEFAULT_SSIM_FLOOR, marp_cmd=None,
                         theme="default", theme_css=None):
    """Stage 7 루프(I/O): deck 렌더 -> 슬라이드별 SSIM -> 저충실도 강등 -> 보정 deck
    재작성. design 의 '자동 강등이 충실도의 안전망' 을 실행한다.

    capture_paths: 슬라이드 인덱스 순서의 원본 캡처 경로 리스트(SSIM 기준).
    out_dir: 렌더 PNG + 보정 deck 출력 폴더.
    theme/theme_css: deck 생성에 쓴 커스텀 테마(렌더 --theme + 보정 deck 프론트매터).
    반환 report dict: {rendered, scores, flagged, plans, corrected_deck}.

    marp 부재/이미지 부재면 graceful degrade(rendered=False) — office 에서 marp
    설치 후 실행. 1-pass 강등(재렌더 검증 반복은 호출측에서 다시 호출).
    """
    out_dir = Path(out_dir)
    report = {"rendered": False, "scores": [], "flagged": [], "plans": [],
              "corrected_deck": None}
    render = render_deck(deck_path, out_dir, fmt="png", marp_cmd=marp_cmd,
                         theme_css=theme_css)
    if not render.ok:
        print("[WARNING] 렌더 미수행 - Stage 7 검증/강등 건너뜀(graceful).")
        return report
    pngs = [p for p in render.outputs if p.lower().endswith(".png")]
    report["rendered"] = True

    scores = []
    for idx, cap in enumerate(capture_paths):
        if idx >= len(pngs):
            scores.append(0.0)   # 렌더 슬라이드 수 부족 -> 최저(강등 유도).
            continue
        orig = _load_image(cap)
        rendered_img = _load_image(pngs[idx])
        if orig is None or rendered_img is None:
            scores.append(0.0)
            continue
        scores.append(slide_fidelity(orig, rendered_img))
    report["scores"] = scores

    flagged = flag_low_fidelity(scores, threshold=threshold)
    report["flagged"] = flagged
    if not flagged:
        print(f"[INFO] 모든 슬라이드 충실도 >= {threshold} - 강등 불필요.")
        return report

    plans = []
    for idx in flagged:
        cap = capture_paths[idx] if idx < len(capture_paths) else ""
        plans.append(plan_downgrade(
            results[idx], crop_lookup=(crop_lookups or {}).get(results[idx].screenshot_id, {}),
            available_crops=available_crops, capture_path=str(cap), slide_index=idx))
    report["plans"] = plans

    corrected = apply_downgrade_plans(results, crop_lookups or {}, plans,
                                      available_crops=available_crops, theme=theme)
    corrected_path = out_dir / "deck_corrected.md"
    corrected_path.write_text(corrected, encoding="utf-8")
    report["corrected_deck"] = str(corrected_path)
    print(f"[INFO] {len(flagged)} 슬라이드 강등 -> 보정 deck: {corrected_path}")
    return report


__all__ = [
    "DEFAULT_SSIM_FLOOR",
    "DowngradePlan",
    "apply_downgrade_plans",
    "flag_low_fidelity",
    "plan_downgrade",
    "slide_fidelity",
    "ssim",
    "verify_and_downgrade",
    "whole_slide_marp",
]
