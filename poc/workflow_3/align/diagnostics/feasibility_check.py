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

마킹 정직성(2026-06-18): verdict 게이트(reregister tau=0.98)는 production 보정과 동일하게
유지하되, 매처의 *always-on* 변별 플래그(``result.distinctive``, engine max_second_ratio=0.94)
를 렌더링에 반영한다. 비변별 매칭(best peak 이 2nd 와 거의 동률)이면 score 가 임계를 넘어
``decision=match`` 가 떠도 배너에 ``[NON-DISTINCT]`` 를 붙이고 align 십자선을 확신 노랑 대신
verdict 색 + "align?" 로 그려, coin-flip 좌표를 확정 정답처럼 보이지 않게 한다. 게이트보다
엄격한 0.94 라 0.94~0.98 회색지대(게이트는 possible)도 시각적으로 경고된다.
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
from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box
from poc.workflow_3.util import env_float

LOG_COMPONENT = "align_feasibility"

# PM 모드를 기본 채택하되, *다른* modality template 이 PM 채택분보다 이 이상(sel score)
# 높으면 PM 오독으로 보고 점수 승자로 폴백한다. PM 한 글자 오독이 엉뚱한 OM/SEM key 를
# 확정하지 못하게 하는 안전 가드(콜드스타트값, 실데이터 보정 대상).
PM_OVERRIDE_MARGIN = env_float("ALIGN_FAIL_PM_OVERRIDE_MARGIN", 0.15)

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
    pm_text: str | None = None         # PM 박스에서 읽은 배율 텍스트(원문) 또는 None.
    pm_mode: str | None = None         # PM 텍스트 → "OM" | "SEM" | None(모호).
    sem_box_bbox: tuple | None = None  # 검출된 live SEM box (l,t,r,b) 풀프레임 px, 없으면 None.
    mode_source: str = ""              # modality 결정 근거: "PM(<text>)" | "score_override(...)" | "score".
    pm_text_source: str = ""           # PM 텍스트 출처: "inline_vlm" | "ocr_crop" | "".


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


def _draw_match_marks(canvas, match_xy, align_xy, tpl_wh, scale, color, *, ambiguous=False) -> None:
    """match bbox/중심 + align target 십자선을 그린다.

    ``ambiguous`` 면(매처가 best peak 을 2nd 대비 유일하다고 보지 못한 비변별 매칭)
    align 십자선을 확신 노랑 대신 verdict 색(``color``)으로 그리고 라벨도 "align?" 로
    바꿔, 좌표가 coin-flip 일 수 있음을 표시한다. 점을 확정 정답처럼 그리지 않는다.
    """
    if match_xy is not None and tpl_wh is not None:
        bw = int(tpl_wh[0] * scale)
        bh = int(tpl_wh[1] * scale)
        x0, y0 = int(match_xy[0] - bw / 2), int(match_xy[1] - bh / 2)
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), color, 2)
        cv2.circle(canvas, (int(match_xy[0]), int(match_xy[1])), 4, color, -1)
    if align_xy is not None:
        ax, ay = int(align_xy[0]), int(align_xy[1])
        r = 18
        cross_color = color if ambiguous else (255, 255, 0)
        label = "align? (ambiguous)" if ambiguous else "align"
        cv2.line(canvas, (ax - r, ay), (ax + r, ay), cross_color, 2)
        cv2.line(canvas, (ax, ay - r), (ax, ay + r), cross_color, 2)
        cv2.putText(canvas, label, (ax + r + 4, ay + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cross_color, 2, cv2.LINE_AA)


def _draw_sem_box(canvas, box) -> None:
    """검출된 live SEM box(l,t,r,b)를 초록 사각형 + 'SEM box' 라벨로 그린다.

    box 가 None 이면(검출 실패/미수행) 아무것도 그리지 않는다 — '검출 못함' 은
    배너 텍스트(sembox=...)로 알린다. align 십자선과 색이 겹치지 않게 초록을 쓴다.
    """
    if box is None:
        return
    l, t, r, b = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    cv2.rectangle(canvas, (l, t), (r, b), (0, 220, 0), 2)
    cv2.putText(canvas, "SEM box", (l + 4, max(0, t - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)


def _draw_pm_box(canvas, pm_box_px) -> None:
    """검출된 PM 박스(dict l/t/r/b 픽셀)를 청록 사각형 + 'PM' 라벨로 그린다(있을 때만).

    PM 위치를 overlay(_marked.jpg)에 남겨 어느 영역을 읽었는지 한눈에 검증하게 한다.
    SEM box(초록)·align(노랑)과 색이 겹치지 않게 청록을 쓴다.
    """
    if not pm_box_px:
        return
    l, t = int(pm_box_px["left"]), int(pm_box_px["top"])
    r, b = int(pm_box_px["right"]), int(pm_box_px["bottom"])
    cv2.rectangle(canvas, (l, t), (r, b), (255, 255, 0), 2)
    cv2.putText(canvas, "PM", (l, max(0, t - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def _maybe_detect_sem_box(frame_path: Path, vlm_client, *, ocr_client=None, pm_two_stage=False):
    """가능하면 live SEM box + PM 모드를 검출한다. client 없거나 실패하면 None.

    개발 PC(Flask VLM 부재)나 VLM 오류에서도 호출부가 전체 프레임 매칭으로 안전하게
    폴백하도록 예외를 삼키고 None 을 돌려준다. two_stage 면 PM crop 을 PaddleOCR 로
    재독한다(crop 은 메모리에서만 만들고 디버그 이미지는 더 이상 저장하지 않는다).
    """
    if vlm_client is None:
        return None
    try:
        from PIL import Image

        with Image.open(frame_path) as image:
            return detect_sem_box(
                image, vlm_client,
                ocr_client=ocr_client, two_stage=pm_two_stage,
            )
    except Exception as exc:
        print(f"[WARNING] SEM box 검출 실패(전체 프레임 매칭으로 폴백): {exc}")
        return None


def mark_align_feasibility(
    frame_path: Path,
    *,
    eqp_id: str,
    recipe_id: str,
    cond_box_crop: bool = True,
    reregister_ratio_threshold: float | None = None,
    vlm_client=None,
    ocr_client=None,
    pm_two_stage: bool = False,
) -> FeasibilityResult:
    """캡처 스크린샷에 보정 가능성을 판정해 overlay(_marked.jpg)+json 으로 남긴다.

    rcp 등록 align key(OM/SEM)를 정적 프레임에 ensemble 매칭하고 key_visibility_gate
    로 verdict 를 정한다. `vlm_client` 가 주어지면 먼저 live SEM box 를 검출해
    (1) PM 박스 텍스트로 OM/SEM modality 를 정하고(읽으면 그 template 만 매칭, 못 읽으면
    기존 '전체 매칭 후 최고점수' 폴백), (2) box 안쪽만 잘라 매칭한 뒤 box 원점만큼
    align point 를 풀프레임 좌표로 되돌리고, (3) box 사각형을 overlay 에 그린다. client 가
    없거나(개발 PC) 검출이 실패하면 전체 캡처 창 매칭으로 안전하게 폴백한다.
    consensus cache 의 S event 수는 read-only 로 읽어 표기만 한다(verdict 에 영향 없음).
    예외/자산 부재도 캡처 위에 배너를 남겨 엔지니어가 한눈에 보게 한다.
    `FeasibilityResult` 반환.
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
    distinctive = True   # 매처가 best peak 을 2nd 대비 유일하다고 봤는가(engine flag, 0.94).
    best_scale = 0.0
    match_xy = None
    align_xy = None
    modality = ""
    color_bgr = _VERDICT_STYLE["no_assets"][1]
    pm_text = None
    pm_mode = None
    pm_text_source = ""
    pm_box_px = None
    sem_box_bbox = None
    mode_source = ""
    sembox_state = "off"   # off(미수행) | located | not_located.

    if templates:
        gray = load_gray(frame_path)   # 전체 캡처 창 grayscale.

        # ---- live SEM box 검출(가능하면) → PM 모드 + box ROI. ----
        detect = _maybe_detect_sem_box(
            frame_path, vlm_client, ocr_client=ocr_client, pm_two_stage=pm_two_stage
        )
        box = None              # (l, t, r, b) 풀프레임 px.
        origin = (0, 0)         # 매칭 ROI 의 풀프레임 원점.
        match_gray = gray
        if detect is not None:
            pm_text = detect.pm_text
            pm_mode = detect.pm_mode
            pm_text_source = detect.pm_text_source
            pm_box_px = detect.pm_box_px
            if detect.detected and detect.bbox_px:
                b = detect.bbox_px
                l, t = int(b["left"]), int(b["top"])
                r, bm = int(b["right"]), int(b["bottom"])
                # box 안쪽만 잘라 매칭(UI 크롬 오검출 차단). 비정상 ROI 면 전체로 폴백.
                if r - l >= 2 and bm - t >= 2:
                    box = (l, t, r, bm)
                    sem_box_bbox = box
                    origin = (l, t)
                    match_gray = gray[t:bm, l:r]
                    sembox_state = "located"
                else:
                    sembox_state = "not_located"
            else:
                sembox_state = "not_located"

        # ---- modality 결정: 항상 모든 template 을 매칭(점수 비교 + PM 가드용)한 뒤 선택. ----
        # PM 모드가 읽히면 기본 채택하되, 다른 modality 가 PM_OVERRIDE_MARGIN 이상 높은
        # 점수면 PM 오독으로 보고 점수 승자로 폴백(misread 가 엉뚱한 key 를 확정 못하게).
        results = {
            mode: compute_align_key_score_ensemble(
                tpl, match_gray, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY
            )
            for mode, tpl in templates.items()
        }
        score_mode = max(results, key=lambda m: results[m].score)
        if pm_mode in results:
            if (
                score_mode != pm_mode
                and results[score_mode].score - results[pm_mode].score > PM_OVERRIDE_MARGIN
            ):
                modality = score_mode
                mode_source = f"score_override(PM={pm_text})"
            else:
                modality = pm_mode
                mode_source = f"PM({pm_text})"
        else:
            modality = score_mode
            mode_source = "score"
        template = templates[modality]
        result = results[modality]

        decision = result.decision
        score = float(result.score)
        second_ratio = result.second_ratio
        distinctive = bool(result.distinctive)
        best_scale = float(result.best_scale)
        # ROI-local best_xy → box 원점만큼 더해 풀프레임 좌표로 환산(클릭 좌표계 일치).
        match_xy = (
            int(result.best_xy[0]) + origin[0],
            int(result.best_xy[1]) + origin[1],
        )

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
        _draw_sem_box(color, box)   # 먼저 박스(초록) → 그 위에 match/align 마크.
        _draw_pm_box(color, pm_box_px)   # PM 박스(청록) — crop 검증용.
        _draw_match_marks(
            color, match_xy, align_xy, (tw, th), best_scale, color_bgr,
            ambiguous=not distinctive,
        )

    # ---- 배너 텍스트 ----
    label = _VERDICT_STYLE[verdict][0]
    sr_txt = f"{second_ratio:.3f}" if second_ratio is not None else "-"
    # 비변별 매칭이면 decision 옆에 [NON-DISTINCT] 를 붙여, score 가 임계를 넘어
    # decision=match 가 떠도 2nd peak 과 거의 동률(coin-flip)임을 숨기지 않는다.
    distinct_flag = "" if distinctive else " [NON-DISTINCT]"
    lines = [label]
    if verdict != "no_assets":
        lines.append(
            f"decision={decision}{distinct_flag} score={score:.3f} 2nd/best={sr_txt} "
            f"scale={best_scale:.2f} mode={modality or '-'}[{mode_source or '-'}]"
        )
        lines.append(
            f"sembox={sembox_state} pm={pm_text or '-'}->{pm_mode or '-'}"
            f"[{pm_text_source or '-'}]"
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
        "distinctive": distinctive,
        "best_scale": best_scale,
        "match_xy": list(match_xy) if match_xy else None,
        "align_xy": list(align_xy) if align_xy else None,
        "modality": modality,
        "mode_source": mode_source,
        "pm_text": pm_text,
        "pm_mode": pm_mode,
        "pm_text_source": pm_text_source,
        "pm_box_px": pm_box_px,
        "sem_box_state": sembox_state,
        "sem_box_bbox": list(sem_box_bbox) if sem_box_bbox else None,
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
        f"[INFO] feasibility: verdict={verdict} decision={decision or '-'}{distinct_flag} "
        f"score={score:.3f} 2nd/best={sr_txt} mode={modality or '-'}[{mode_source or '-'}] "
        f"sembox={sembox_state} pm={pm_text or '-'}->{pm_mode or '-'} "
        f"consensus={consensus_events}ev/{consensus_images}img -> {out_marked}"
    )
    return FeasibilityResult(
        verdict, decision, score, second_ratio, best_scale,
        match_xy, align_xy, modality, consensus_events, consensus_images,
        out_marked, out_json, frame_wh,
        pm_text=pm_text, pm_mode=pm_mode, sem_box_bbox=sem_box_bbox,
        mode_source=mode_source, pm_text_source=pm_text_source,
    )


__all__ = ["FeasibilityResult", "mark_align_feasibility"]
