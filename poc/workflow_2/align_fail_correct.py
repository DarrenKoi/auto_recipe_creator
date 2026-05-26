"""Align Fail 1차(primary) 보정 — paused 화면의 crosshair 를 recipe 점으로 옮기고 OK.

도메인(사용자 확정):

* ALID=9006 Align Fail 이 나면 SEM Monitor 는 **paused live 화면**에 가로/세로 십자선
  (crosshair)을 그려 *현재(=잘못된)* align 시도 위치를 표시하고 멈춘다.
* 레시피에 등록된 align key(``align_img_from_rcp``: IMAP0001=OM, IMAP0002=SEM)에는
  엔지니어가 **이미지에 박스를 그려** 유니크한 위치를 표시했고, 그 박스의 중심
  (보통 이미지 정중앙)이 정렬해야 할 *target point* 다. 박스 *안의 모양*이 live SEM
  에도 보여야 하며, live 는 contrast/brightness/shape 가 조금씩 달라 픽셀 동일성이
  아니라 edge 구조로 매칭한다(``STRUCTURE_POLICY`` + Chamfer, CLAHE 전처리).
* 보정 = crosshair 를 recipe-matched 점으로 옮긴 뒤 OK 를 눌러 진행.
  - 더블클릭 = 클릭점을 화면 중심으로 recenter → crosshair 가 그 점으로(=move_to_point).
  - OK 버튼은 SEM ROI 밖 dialog 컨트롤 → VLM 으로 위치를 찾아 screen 좌표로 single click.

흐름의 위치(중요): align key 는 **대개 잘못된 crosshair 근처에 이미 보인다**. 따라서
본 모듈(즉시 reposition+OK)이 **primary** 경로다. ``live_align_search`` 의 pan/zoom
two-phase 탐색은 *아무것도 안 보일 때만* 도는 **fallback** 이다. 둘을 가르는 단일
기준이 ``key_visibility_gate`` 다.

실행(Mac 데모): uv run python poc/workflow_2/align_fail_correct.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import AlignFailAssets, load_gray, resolve_assets_auto
from poc.workflow_2.align_key_matcher import (
    DEFAULT_SCALES,
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)
from poc.workflow_2.live_align_search import (
    MIN_CONFIRM_SCALE,
    LiveSearchConfig,
    LiveSearchOutcome,
    SEMMonitorController,
    clamp_to_fov,
    live_align_search,
    route_template,
)

# paused fail 화면은 *레시피 등록 배율* 에서 멈춘 것이므로, key 가 보인다면 거의
# native(~1.0) 크기로 보인다. 따라서 broad(miniature) band 가 아니라 near-native band
# 로 매칭한다. broad scale 을 쓰면 tiny-scale chamfer 과신으로 featureless 프레임도
# 거짓 match 가 나서 primary 경로로 잘못 진입한다(live_align_search 의 terminal 가드와
# 동일한 함정). pan/zoom 으로 줌아웃 탐색하는 fallback 만 broad band 를 쓴다.
PAUSED_SCALES = DEFAULT_SCALES

# OK 위치를 screen 프레임에서 찾아 (x, y) 또는 None 을 돌려주는 주입형 locator.
OkLocator = Callable[[np.ndarray], "tuple[int, int] | None"]


# ------------------------------------------------------------------
# 설정 / 결과.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionConfig:
    """primary correction 정책. 가시성 게이트 임계는 STRUCTURE_POLICY 를 따른다."""

    click_margin_ratio: float = 0.12  # best_xy 를 FOV 안쪽으로 clamp (live search 와 공유).
    require_ok_button: bool = True  # OK 위치 확인 실패 시 corrected 대신 escalate.
    settle_sec: float = 0.0  # 제스처 후 대기(실장비 안정화).
    crop_template_to_box: bool = False  # 등록 이미지의 엔지니어 박스 내부로 template crop(미보정, 기본 off).


@dataclass
class CorrectionOutcome:
    """보정 결과. 어느 경로로 끝났는지 + 좌표/decision 기록."""

    status: str  # "corrected" | "fallback_<status>" | "escalated" | "no_assets"
    path: str  # "primary" | "fallback"
    key_decision: str  # 가시성 게이트 판정에 쓰인 matcher decision.
    best_xy: tuple[int, int] | None  # reposition 한 FOV 좌표(clamp 후).
    ok_screen_xy: tuple[int, int] | None  # 클릭한 OK 버튼의 screen 좌표.
    fallback: LiveSearchOutcome | None  # fallback 으로 갔을 때의 live search 결과.
    history: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------
# 가시성 게이트 — primary vs fallback 분기의 단일 기준.
# ------------------------------------------------------------------


def key_visibility_gate(result: AlignKeyMatchResult) -> bool:
    """paused frame 에서 recipe key 가 '지금 여기' 인식되는가 — primary vs fallback 분기.

    "키가 이 전체 프레임에 있는가"(존재/부재) 판정이라, 이미 localize 된 상태의 drift
    허용을 위해 임계를 낮춘 ``STRUCTURE_POLICY`` 를 그대로 쓰면 featureless 배경도
    adjust 로 새어 들어온다(chamfer 단독 ~0.4~0.6, orb=0). 그래서:

    * ``best_scale`` 가 충분(>=MIN_CONFIRM_SCALE)해야 한다 — tiny-scale chamfer 과신 차단.
    * 강한 ``match`` 는 edge 구조만으로 인정.
    * 약한 ``adjust`` 는 **feature 보강(orb>0)** 이 있을 때만 가시로 인정 — 배경 거짓양성
      차단. align fail 의 drift 된 진짜 key 는 같은 fab 패턴이라 보통 일부 ORB 가 살아있다.

    True → primary reposition+OK, False → live_align_search fallback(아무것도 안 보임).
    임계/조건은 cold-start 이며 실데이터 calibration 대상.
    """
    if result.best_scale < MIN_CONFIRM_SCALE:
        return False
    if result.decision == "match":
        return True
    if result.decision == "adjust":
        return result.orb_inlier_ratio > 0.0
    return False


# ------------------------------------------------------------------
# 등록 이미지 → template (OM/SEM).
# ------------------------------------------------------------------


def extract_annotation_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """등록 이미지에 엔지니어가 그린 사각형 주석의 *내부* bbox 를 추정한다.

    박스는 어두운 배경 위 밝은 얇은 사각형으로 그려져 있다(burned-in). 가장 큰
    사각형 윤곽을 찾아 그 내부를 반환한다. 반환 (left, top, right, bottom) 또는 None.

    주의: **미보정(uncalibrated)** 추정이다. align key 자체가 box-in-box 모양일 수
    있어 주석 사각형과 혼동될 수 있으므로, 실제 오피스 파일로 검증하기 전까지는
    ``CorrectionConfig.crop_template_to_box`` 기본값(off)으로 두고 전체 이미지를 쓴다.
    """
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int] | None = None
    best_area = 0.0
    img_area = float(w * h)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        bx, by, bw, bh = cv2.boundingRect(approx)
        area = float(bw * bh)
        # 전체 이미지 테두리(>90%)나 너무 작은(<2%) 사각형은 제외.
        if area >= 0.90 * img_area or area <= 0.02 * img_area:
            continue
        if area > best_area:
            best_area = area
            # 선 두께만큼 안쪽으로 살짝 들여 *내부* 만 남긴다.
            pad = max(2, int(round(0.01 * (bw + bh) / 2)))
            best = (
                bx + pad,
                by + pad,
                min(w, bx + bw - pad),
                min(h, by + bh - pad),
            )
    return best


def _load_template(path: Path, *, recipe_id: str, key_type: str, crop_to_box: bool) -> AlignKeyTemplate:
    """경로의 등록 이미지를 (선택적으로 박스 crop 후) AlignKeyTemplate 으로 만든다."""
    gray = load_gray(path)
    if crop_to_box:
        box = extract_annotation_box(gray)
        if box is not None:
            left, top, right, bottom = box
            gray = gray[top:bottom, left:right]
            print(f"[INFO] {key_type} template 박스 내부로 crop: {box}")
        else:
            print(f"[WARNING] {key_type} 주석 박스를 찾지 못해 전체 이미지를 사용합니다.")
    return build_template(gray, recipe_id=recipe_id, version="v0", key_type=key_type)


def build_templates_from_assets(
    assets: AlignFailAssets, *, crop_to_box: bool = False
) -> dict[str, AlignKeyTemplate]:
    """recipe_om/recipe_sem → {'OM': ..., 'SEM': ...}. live_align_search 와 동일 계약.

    존재하는 등록 이미지만 담는다(부분 생성 허용). 둘 다 없으면 빈 dict.
    """
    templates: dict[str, AlignKeyTemplate] = {}
    if assets.recipe_om is not None:
        templates["OM"] = _load_template(
            assets.recipe_om, recipe_id=assets.recipe_id, key_type="om", crop_to_box=crop_to_box
        )
    if assets.recipe_sem is not None:
        templates["SEM"] = _load_template(
            assets.recipe_sem, recipe_id=assets.recipe_id, key_type="sem", crop_to_box=crop_to_box
        )
    return templates


# ------------------------------------------------------------------
# primary 보정 오케스트레이션.
# ------------------------------------------------------------------


def correct_align_fail(
    controller: SEMMonitorController,
    templates: dict[str, AlignKeyTemplate],
    *,
    vlm_client=None,
    ok_locator: OkLocator | None = None,
    config: CorrectionConfig = CorrectionConfig(),
    fallback_config: LiveSearchConfig = LiveSearchConfig(),
    dry_run: bool = True,
    debug_dir: Path | None = None,
) -> CorrectionOutcome:
    """paused Align Fail 화면에서 crosshair 를 recipe-matched 점으로 옮기고 OK.

    1) capture() 로 paused SEM ROI 프레임 캡처.
    2) read_mode() 로 OM/SEM template 라우팅(route_template).
    3) compute_align_key_score(scales=PAUSED_SCALES, policy=STRUCTURE_POLICY).
    4) key_visibility_gate:
       - True  → clamp_to_fov(best_xy) → move_to_point(=더블클릭 recenter) →
                 capture_screen() 에서 OK 버튼을 찾아 click_screen(OK).
       - False → live_align_search(...) 로 위임(아무것도 안 보일 때만 pan/zoom).

    dry_run=True 면 좌표를 계산·로그·overlay 만 하고 실제 actuation(move/click)은 하지
    않는다(Mac-safe, procedure §5 Phase 3). ok_locator 를 직접 주입하면 VLM 없이도
    테스트할 수 있고, 미주입 시 vlm_client 로 locate_ok_button 을 래핑한다.
    """
    if not templates:
        raise ValueError("templates 가 비어 있습니다 (OM/SEM 중 최소 하나 필요)")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []

    frame = controller.capture()
    fh, fw = frame.shape[:2]
    mode = (controller.read_mode() or "").upper()
    template = route_template(templates, mode)

    result = compute_align_key_score(
        template, frame, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY
    )
    history.append(
        {
            "stage": "paused_match",
            "mode": mode,
            "decision": result.decision,
            "score": float(result.score),
            "chamfer": float(result.chamfer_score),
            "orb": float(result.orb_inlier_ratio),
            "best_scale": float(result.best_scale),
            "best_xy": [int(result.best_xy[0]), int(result.best_xy[1])],
        }
    )
    print(
        f"[INFO] paused frame: mode={mode or '-'} decision={result.decision} "
        f"score={result.score:.3f} (ch={result.chamfer_score:.3f} orb={result.orb_inlier_ratio:.3f}) "
        f"scale={result.best_scale:.2f} best_xy={result.best_xy}"
    )
    if debug_dir is not None:
        save_overlay_jpeg(result.debug_overlay, debug_dir / "paused_match.jpg")

    # ---- 가시성 게이트: key 가 안 보이면 fallback 탐색으로 위임. ----
    if not key_visibility_gate(result):
        print("[INFO] key 가 paused 화면에 보이지 않음 → fallback(live_align_search) 위임")
        outcome = live_align_search(
            controller,
            templates,
            config=fallback_config,
            debug_dir=(debug_dir / "fallback") if debug_dir is not None else None,
        )
        return CorrectionOutcome(
            status=f"fallback_{outcome.status}",
            path="fallback",
            key_decision=result.decision,
            best_xy=None,
            ok_screen_xy=None,
            fallback=outcome,
            history=history,
        )

    # ---- PRIMARY: crosshair 를 best_xy 로 reposition. ----
    cx, cy = clamp_to_fov(result.best_xy[0], result.best_xy[1], fw, fh, config.click_margin_ratio)
    print(f"[INFO] reposition: 더블클릭 recenter → ({cx}, {cy}){' [dry-run]' if dry_run else ''}")
    if not dry_run:
        controller.move_to_point(cx, cy)
        if config.settle_sec:
            time.sleep(config.settle_sec)

    # ---- OK 버튼 위치 확인(screen 좌표) 후 single click. ----
    resolved_locator = ok_locator
    if resolved_locator is None and vlm_client is not None:
        from poc.workflow_2.vlm_ok_button_box import locate_ok_button

        resolved_locator = lambda f: locate_ok_button(frame_bgr=f, client=vlm_client)  # noqa: E731

    ok_xy: tuple[int, int] | None = None
    if resolved_locator is not None:
        screen = controller.capture_screen()
        try:
            ok_xy = resolved_locator(screen)
        except Exception as exc:  # VLM 실패는 치명적이지 않게 escalate 로 흡수.
            print(f"[ERROR] OK 버튼 탐지 실패: {exc}")
    else:
        print("[WARNING] OK locator 가 없습니다(vlm_client/ok_locator 미주입).")

    if ok_xy is None:
        if config.require_ok_button:
            print("[WARNING] OK 버튼을 찾지 못함 → 엔지니어 확인용 escalate")
            return CorrectionOutcome(
                status="escalated",
                path="primary",
                key_decision=result.decision,
                best_xy=(cx, cy),
                ok_screen_xy=None,
                fallback=None,
                history=history,
            )
        print("[INFO] OK 버튼 생략(require_ok_button=false), reposition 까지만 수행")
    else:
        print(f"[INFO] OK 클릭: screen=({ok_xy[0]}, {ok_xy[1]}){' [dry-run]' if dry_run else ''}")
        if not dry_run:
            controller.click_screen(ok_xy[0], ok_xy[1])

    return CorrectionOutcome(
        status="corrected",
        path="primary",
        key_decision=result.decision,
        best_xy=(cx, cy),
        ok_screen_xy=ok_xy,
        fallback=None,
        history=history,
    )


def correct_align_fail_auto(
    controller: SEMMonitorController,
    *,
    vlm_client=None,
    ok_locator: OkLocator | None = None,
    config: CorrectionConfig = CorrectionConfig(),
    dry_run: bool = True,
    debug_dir: Path | None = None,
) -> CorrectionOutcome:
    """자산을 자동 해석(resolve_assets_auto)해 template 을 만들고 correct_align_fail 실행."""
    assets = resolve_assets_auto()
    if assets is None:
        print("[ERROR] align fail recipe 폴더를 찾지 못했습니다.")
        return CorrectionOutcome("no_assets", "primary", "low", None, None, None, [])
    templates = build_templates_from_assets(assets, crop_to_box=config.crop_template_to_box)
    if not templates:
        print(f"[ERROR] 등록 OM/SEM 이미지가 없습니다: {assets.recipe_dir}")
        return CorrectionOutcome("no_assets", "primary", "low", None, None, None, [])
    return correct_align_fail(
        controller,
        templates,
        vlm_client=vlm_client,
        ok_locator=ok_locator,
        config=config,
        dry_run=dry_run,
        debug_dir=debug_dir,
    )


# ==================================================================
# Mac 데모 — key 가 paused 화면에 이미 보이는 primary 경로(가상 화면 + mock).
# ==================================================================


def _make_primary_demo():
    """key 가 화면 중앙 근처에 보이는 paused 프레임 + 그 key 의 등록 template 을 만든다."""
    from poc.workflow_2.live_align_search import _MockSEMMonitor
    from poc.workflow_2.test_align_key_match import make_synthetic_template, make_wafer_background

    pattern = make_synthetic_template(key_type="box")  # 128px native.
    recipe_img = cv2.copyMakeBorder(pattern, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    template = build_template(recipe_img, recipe_id="DEMO-SEM-001", version="v0", key_type="sem")

    # key 가 시작 FOV 안에 보이도록 시작 위치에 박아 둔다(=primary 경로가 바로 fire).
    wafer = make_wafer_background(frame_size=(2400, 3200))
    target_xy = (900, 1300)
    th, tw = pattern.shape[:2]
    wafer[target_xy[1] - th // 2 : target_xy[1] + th // 2, target_xy[0] - tw // 2 : target_xy[0] + tw // 2] = pattern

    # 가상 전체 화면: 검은 배경에 흰 'OK' 사각형(stub locator 가 그 중심을 돌려줌).
    screen = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(screen, (640, 540), (740, 580), 255, 2)

    monitor = _MockSEMMonitor(
        wafer,
        screen_size=(512, 768),
        start_xy=target_xy,  # native 배율 + key 가 중앙 → 즉시 보임.
        zoom_factors=(4.0, 3.0, 2.0, 1.4, 1.0),
        start_mag_index=4,  # native.
        mode="SEM",
        screen_image=screen,
    )
    return monitor, {"SEM": template}


def main() -> int:
    print("[INFO] align_fail_correct Mac 데모 시작 (primary reposition + OK, dry-run)")
    monitor, templates = _make_primary_demo()

    run_tag = time.strftime("%y%m%d_%H%M%S")
    debug_dir = DEBUG_IMAGE_DIR / "align_fail_correct_demo" / run_tag

    # VLM 없이 동작하도록 OK locator 를 stub(흰 사각형 중심)으로 주입.
    def _stub_ok_locator(_screen: np.ndarray) -> tuple[int, int]:
        return (690, 560)

    outcome = correct_align_fail(
        monitor,
        templates,
        ok_locator=_stub_ok_locator,
        dry_run=True,
        debug_dir=debug_dir,
    )

    print(
        f"[INFO] status={outcome.status} path={outcome.path} "
        f"decision={outcome.key_decision} best_xy={outcome.best_xy} ok_xy={outcome.ok_screen_xy}"
    )
    print(f"[INFO] debug overlays: {debug_dir}")
    ok = outcome.status == "corrected" and outcome.path == "primary"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
