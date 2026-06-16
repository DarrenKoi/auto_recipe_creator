"""Align fail 캡처 화면에 'align point 보정 가능 여부' 를 정적으로 판정·표시한다.

검증된 보정 엔진(`compute_align_key_score_ensemble` + `key_visibility_gate`)을 *실제
동작(reposition/OK 클릭) 없이* 정적 프레임(점검 모니터가 박제한 fail 스크린샷)에 적용해,
rcp 등록 align key 가 이 화면에서 보이는지 = 자동 보정이 가능한지 판정하고, 결과를 원본
스크린샷 위에 overlay 로 그린다. consensus cache(align_consensus_cache)에 이미 stage 된
최근 성공(S) event 수는 read-only 컨텍스트로 함께 표기한다(재등록 우선순위 자료).

설계 원칙(2026-06-11 결정):
  * 새 CV 알고리즘을 도입하지 않는다 — production 보정과 같은 엔진/임계를 그대로 쓴다.
    consensus 는 "표시(개수)" 만 연결하고, verdict 자체는 검증된 rcp 경로가 낸다.
  * 점검 모니터는 SEM panel ROI 를 따로 잡지 않으므로 *전체 캡처 창* 을 프레임으로 쓴다.
    PAUSED_SCALES(near-native)를 쓰므로 key 가 캡처 패널에서 거의 native 크기로 보인다는
    가정이며, 실데이터로 검증 후 scale band 는 workflow_2 에서 조정한다.

verdict:
  * "possible"     — GATE_ACT: key 가 또렷이 보임 → 자동 reposition+OK 가능.
  * "ambiguous"    — GATE_ENGINEER_REVIEW: 보이나 만성 모호(second_ratio>tau) → 엔지니어.
  * "not_visible"  — GATE_FALLBACK: key 가 이 화면에 안 보임 → pan/zoom 탐색 또는 엔지니어.
  * "no_assets"    — rcp 등록 OM/SEM 이미지를 못 찾음(판정 불가).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3.align.assets import load_gray, resolve_assets_auto
from poc.workflow_3.align.correction import (
    GATE_ACT,
    GATE_ENGINEER_REVIEW,
    GATE_FALLBACK,
    PAUSED_SCALES,
    key_visibility_gate,
)
from poc.workflow_3.align.templates import build_templates_from_assets
from poc.workflow_3.align.matching.engine import (
    STRUCTURE_POLICY,
    compute_align_key_score_ensemble,
)
from poc.workflow_3.align.consensus_gather import count_staged_events

LOG_COMPONENT = "align_feasibility"

# verdict -> (라벨, BGR 색).
_VERDICT_STYLE = {
    "possible": ("CORRECTION POSSIBLE", (0, 200, 0)),
    "ambiguous": ("AMBIGUOUS (engineer review)", (0, 165, 255)),
    "not_visible": ("NOT POSSIBLE (key not visible)", (0, 0, 230)),
    "no_assets": ("NO ASSETS (cannot judge)", (160, 160, 160)),
}

# route intent -> verdict.
_ROUTE_TO_VERDICT = {
    GATE_ACT: "possible",
    GATE_ENGINEER_REVIEW: "ambiguous",
    GATE_FALLBACK: "not_visible",
}


@dataclass
class FeasibilityResult:
    """정적 보정 가능성 판정 + 마킹 산출 경로."""

    verdict: str                       # possible | ambiguous | not_visible | no_assets
    decision: str                      # matcher decision (match/adjust/low) 또는 ""
    score: float
    second_ratio: float | None
    best_scale: float
    match_xy: tuple | None             # 프레임 내 best match 중심.
    align_xy: tuple | None             # offset 반영한 align target 점.
    modality: str                      # "OM" | "SEM" | ""
    consensus_events: int
    consensus_images: int
    marked_path: Path | None
    json_path: Path | None
    frame_wh: tuple | None             # 매칭에 쓴 프레임 크기 (w, h) — align_xy 와 동일 좌표계.


def _font_scale(width: int) -> float:
    """프레임 너비에 비례한 putText 폰트 스케일(대형 RCS 창 대비 가독성)."""
    return max(0.6, min(2.0, width / 1400.0))


def _draw_banner(canvas: np.ndarray, lines: list, color: tuple) -> None:
    """좌상단에 반투명 배경 + 텍스트 라인들을 그린다(첫 줄은 verdict, 색 강조)."""
    h, w = canvas.shape[:2]
    fs = _font_scale(w)
    pad = int(12 * fs)
    line_h = int(34 * fs)
    box_w = min(w - 2 * pad, int(max(len(s) for s in lines) * 16 * fs) + 2 * pad)
    box_h = pad * 2 + line_h * len(lines)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.rectangle(canvas, (pad, pad), (pad + box_w, pad + box_h), color, max(2, int(3 * fs)))

    y = pad + line_h
    for i, text in enumerate(lines):
        c = color if i == 0 else (255, 255, 255)
        thick = max(2, int(3 * fs)) if i == 0 else max(1, int(2 * fs))
        cv2.putText(canvas, text, (pad * 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs if i == 0 else fs * 0.72, c, thick, cv2.LINE_AA)
        y += line_h


def _draw_match_marks(canvas, match_xy, align_xy, tpl_wh, scale, color) -> None:
    """match bbox/중심 + align target 십자선을 그린다."""
    if match_xy is not None and tpl_wh is not None:
        bw = int(tpl_wh[0] * scale)
        bh = int(tpl_wh[1] * scale)
        x0, y0 = int(match_xy[0] - bw / 2), int(match_xy[1] - bh / 2)
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), color, 2)
        cv2.circle(canvas, (int(match_xy[0]), int(match_xy[1])), 4, color, -1)
    if align_xy is not None:
        ax, ay = int(align_xy[0]), int(align_xy[1])
        r = 18
        cv2.line(canvas, (ax - r, ay), (ax + r, ay), (255, 255, 0), 2)
        cv2.line(canvas, (ax, ay - r), (ax, ay + r), (255, 255, 0), 2)
        cv2.putText(canvas, "align", (ax + r + 4, ay + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def mark_align_feasibility(
    frame_path: Path,
    *,
    eqp_id: str,
    recipe_id: str,
    cond_box_crop: bool = True,
    reregister_ratio_threshold: float | None = None,
) -> FeasibilityResult:
    """캡처 스크린샷에 보정 가능성을 판정해 overlay(_marked.jpg)+json 으로 남긴다.

    rcp 등록 align key(OM/SEM)를 정적 프레임에 ensemble 매칭하고 key_visibility_gate
    로 verdict 를 정한다. modality 는 점검 모드에 read_mode 가 없어, 존재하는 template
    들을 모두 매칭해 *최고 점수* 를 채택한다. consensus cache 의 S event 수는 read-only
    로 읽어 표기만 한다(verdict 에 영향 없음). 예외/자산 부재도 캡처 위에 배너를 남겨
    엔지니어가 한눈에 보게 한다. `FeasibilityResult` 반환.
    """
    frame_path = Path(frame_path)
    out_marked = frame_path.with_name(frame_path.stem + "_marked.jpg")
    out_json = frame_path.with_name(frame_path.stem + "_feasibility.json")

    consensus_events, consensus_images = count_staged_events(eqp_id, recipe_id)

    # 원본을 color 로 읽어 그 위에 그린다(매칭은 gray).
    color = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if color is None:
        print(f"[WARNING] feasibility: 캡처 이미지를 읽지 못함 - 분석 생략: {frame_path}")
        return FeasibilityResult(
            "no_assets", "", 0.0, None, 0.0, None, None, "",
            consensus_events, consensus_images, None, None, None,
        )

    # align_xy(아래에서 계산)와 동일 좌표계 — load_gray 가 리사이즈 없이 같은 파일을 읽으므로
    # color(전체 캡처) 크기가 곧 매칭 프레임 크기다. 호출부의 image→screen 변환에 쓴다.
    frame_wh = (color.shape[1], color.shape[0])

    assets = resolve_assets_auto(eqp_id=eqp_id, recipe_name=recipe_id)
    templates = build_templates_from_assets(assets, cond_box_crop=cond_box_crop) if assets else {}

    verdict = "no_assets"
    decision = ""
    score = 0.0
    second_ratio = None
    best_scale = 0.0
    match_xy = None
    align_xy = None
    modality = ""
    color_bgr = _VERDICT_STYLE["no_assets"][1]

    if templates:
        gray = load_gray(frame_path)
        best = None  # (mode, template, result)
        for mode, template in templates.items():
            result = compute_align_key_score_ensemble(
                template, gray, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY
            )
            if best is None or result.score > best[2].score:
                best = (mode, template, result)

        modality, template, result = best
        decision = result.decision
        score = float(result.score)
        second_ratio = result.second_ratio
        best_scale = float(result.best_scale)
        match_xy = (int(result.best_xy[0]), int(result.best_xy[1]))

        route = key_visibility_gate(
            result, reregister_ratio_threshold=reregister_ratio_threshold
        )
        verdict = _ROUTE_TO_VERDICT.get(route, "not_visible")
        color_bgr = _VERDICT_STYLE[verdict][1]

        # align target = match 중심 + (rcp offset * best_scale). offset (0,0)이면 match 중심.
        ox, oy = template.align_offset_xy
        align_xy = (
            int(match_xy[0] + round(ox * best_scale)),
            int(match_xy[1] + round(oy * best_scale)),
        )
        th, tw = template.raw_image.shape[:2]
        _draw_match_marks(color, match_xy, align_xy, (tw, th), best_scale, color_bgr)

    # ---- 배너 텍스트 ----
    label = _VERDICT_STYLE[verdict][0]
    sr_txt = f"{second_ratio:.3f}" if second_ratio is not None else "-"
    lines = [label]
    if verdict != "no_assets":
        lines.append(
            f"decision={decision} score={score:.3f} 2nd/best={sr_txt} "
            f"scale={best_scale:.2f} mode={modality or '-'}"
        )
    else:
        lines.append(f"recipe={recipe_id or '-'} (rcp OM/SEM not found)")
    lines.append(f"consensus: {consensus_events} S events ({consensus_images} imgs)")
    _draw_banner(color, lines, color_bgr)

    try:
        cv2.imwrite(str(out_marked), color)
    except Exception as exc:
        print(f"[WARNING] feasibility marked 저장 실패: {exc}")
        out_marked = None

    payload = {
        "verdict": verdict,
        "decision": decision,
        "score": score,
        "second_ratio": second_ratio,
        "best_scale": best_scale,
        "match_xy": list(match_xy) if match_xy else None,
        "align_xy": list(align_xy) if align_xy else None,
        "modality": modality,
        "consensus_events": consensus_events,
        "consensus_images": consensus_images,
        "eqp_id": eqp_id,
        "recipe_id": recipe_id,
        "frame": str(frame_path),
    }
    try:
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] feasibility json 저장 실패: {exc}")
        out_json = None

    print(
        f"[INFO] feasibility: verdict={verdict} decision={decision or '-'} "
        f"score={score:.3f} 2nd/best={sr_txt} mode={modality or '-'} "
        f"consensus={consensus_events}ev/{consensus_images}img -> {out_marked}"
    )
    return FeasibilityResult(
        verdict, decision, score, second_ratio, best_scale,
        match_xy, align_xy, modality, consensus_events, consensus_images,
        out_marked, out_json, frame_wh,
    )


__all__ = ["FeasibilityResult", "mark_align_feasibility"]
