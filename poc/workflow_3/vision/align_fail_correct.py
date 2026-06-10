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
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from poc.workflow_3.logger import log_work2_event
from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_fail_assets import AlignFailAssets, load_gray, resolve_assets_auto
from poc.workflow_3.vision.align_key_matcher import (
    DEFAULT_SCALES,
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score_ensemble,
    save_overlay_jpeg,
)
from poc.workflow_3.vision.live_align_search import (
    MIN_CONFIRM_SCALE,
    LiveSearchConfig,
    LiveSearchOutcome,
    NotifyFn,
    SEMMonitorController,
    clamp_to_fov,
    live_align_search,
    route_template,
)

LOG_COMPONENT = "align_fail_correct"

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

    # "corrected" | "fallback_<status>" | "escalated_no_ok" | "ok_detect_error" | "no_assets"
    status: str
    path: str  # "primary" | "fallback"
    key_decision: str  # 가시성 게이트 판정에 쓰인 matcher decision.
    best_xy: tuple[int, int] | None  # reposition 한 FOV 좌표(clamp 후).
    ok_screen_xy: tuple[int, int] | None  # 클릭한 OK 버튼의 screen 좌표.
    fallback: LiveSearchOutcome | None  # fallback 으로 갔을 때의 live search 결과.
    error: str | None = None  # OK 탐지 등에서 발생한 예외 요약(정상 not-found 와 구분).
    history: list[dict] = field(default_factory=list)
    # --- 매칭 모호도(read-only surface; notify 재등록 판정용, 보정 동작에는 영향 없음) ---
    second_ratio: float | None = None  # matcher 2nd/best chamfer(1.0 에 가까울수록 모호).
    score_gap: float | None = None  # best - 2nd chamfer.
    distinctive: bool = True  # best 가 2nd 대비 유일한가(데이터 결손 시 True → false-flag 방지).


def _with_key_ambiguity(
    outcome: CorrectionOutcome, result: AlignKeyMatchResult
) -> CorrectionOutcome:
    """matcher result 의 모호도 신호(second_ratio/score_gap/distinctive)를 outcome 에 stamp.

    이미 fail 시점에 계산되어 버려지던 값을 notify 경로로 끌어올리기 위한 read-only surface.
    보정 동작/visibility 게이트(max_second_ratio=0.94)와 무관하며 기존 필드는 보존한다.
    match 이전 반환(no_assets 등)은 호출하지 않아 기본값(None/None/True)을 유지한다.
    """
    return replace(
        outcome,
        second_ratio=result.second_ratio,
        score_gap=result.score_gap,
        distinctive=result.distinctive,
    )


# ------------------------------------------------------------------
# 가시성 게이트 — primary vs fallback 분기의 단일 기준.
# ------------------------------------------------------------------


def key_visibility_gate(result: AlignKeyMatchResult) -> bool:
    """paused frame 에서 recipe key 가 '지금 여기' 인식되는가 — primary vs fallback 분기.

    "키가 이 전체 프레임에 있는가"(존재/부재) 판정. ensemble 경로(decision/score 정비)에서
    decision 은 calibrated sel 임계(match 0.6053/adjust 0.4727)로 재판정되므로, featureless
    배경(chamfer~0.4~0.6·NCC 낮음 → sel~0.25)은 대개 decision="low" 로 1차 차단된다. 그래서:

    * ``best_scale`` 가 충분(>=MIN_CONFIRM_SCALE)해야 한다 — tiny-scale chamfer 과신 차단.
    * 강한 ``match`` 는 edge 구조만으로 인정.
    * 약한 ``adjust`` 는 **구조 유일성(distinctive)** 이 있을 때만 가시로 인정 — 배경 거짓양성
      2차 차단(과거 orb>0 의 대체; ORB 폐지). distinctive 는 chamfer-pool 의 best peak 이
      2nd 대비 유일한가 = "key 구조가 실제로 존재하는가" presence 신호. 불확실하면 fallback.

    True → primary reposition+OK, False → live_align_search fallback(아무것도 안 보임).
    임계/조건은 cold-start 이며 실데이터 calibration 대상.
    """
    if result.best_scale < MIN_CONFIRM_SCALE:
        return False
    if result.decision == "match":
        return True
    if result.decision == "adjust":
        return result.distinctive
    return False


# ------------------------------------------------------------------
# 등록 이미지 → template (OM/SEM).
# ------------------------------------------------------------------


def extract_annotation_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """등록 이미지에 엔지니어가 그린 흰색 unique-area 박스의 *inset 내부* bbox 를 추정한다.

    검출/inset 로직은 ``align_point_correction`` 의 canonical 구현
    (``_detect_white_box`` + ``_inner_crop_for_box``) 에 위임한다 — top-hat → Otsu →
    hollowness/edge-margin/aspect gate 로 거른 뒤, 박스 stroke (보통 1~3 px) 두께만큼
    안쪽으로 깎은 영역만 남긴다. 흰 박스 outline 픽셀이 template 에 새어들어 매칭이
    흰색을 좇으며 점수가 떨어지는 것을 막는다. ``align_similarity`` 도 같은 경로를 쓴다.

    이 모듈에 따로 있던 approxPolyDP 기반 복제 검출기를 제거하고 단일화한 것이다
    (얇은 outline 에서 4-코너 근사가 자주 실패하던 약점을 canonical gate 가 보완).

    반환: inset 적용된 내부 bbox ``(left, top, right, bottom)`` 또는 None.

    주의: align key 자체가 box-in-box 모양이면 주석 박스와 혼동될 수 있어 여전히
    **미보정(uncalibrated)** 이다. 실제 오피스 파일로 검증하기 전까지는
    ``CorrectionConfig.crop_template_to_box`` 기본값(off)으로 두고 전체 이미지를 쓴다.
    """
    # align_similarity 와 동일한 lazy import 패턴 — 모듈 로드 순서 의존 회피.
    from poc.workflow_3.vision.align_point_correction import _detect_white_box, _inner_crop_for_box

    box = _detect_white_box(gray)
    if box is None:
        return None
    _inner_gray, (ix, iy, iw, ih) = _inner_crop_for_box(gray, box)
    return (int(ix), int(iy), int(ix + iw), int(iy + ih))


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
    notify_fn: NotifyFn | None = None,
    dry_run: bool = True,
    debug_dir: Path | None = None,
) -> CorrectionOutcome:
    """paused Align Fail 화면에서 crosshair 를 recipe-matched 점으로 옮기고 OK.

    1) capture() 로 paused SEM ROI 프레임 캡처.
    2) read_mode() 로 OM/SEM template 라우팅(route_template).
    3) compute_align_key_score_ensemble(scales=PAUSED_SCALES, policy=STRUCTURE_POLICY).
    4) key_visibility_gate:
       - True  → clamp_to_fov(best_xy) → move_to_point(=더블클릭 recenter) →
                 capture_screen() 에서 OK 버튼을 찾아 click_screen(OK).
       - False → live_align_search(...) 로 위임(아무것도 안 보일 때만 pan/zoom).

    dry_run=True 면 좌표를 계산·로그·overlay 만 하고 실제 actuation(move/click)은 하지
    않는다(Mac-safe, procedure §5 Phase 3). ok_locator 를 직접 주입하면 VLM 없이도
    테스트할 수 있고, 미주입 시 vlm_client 로 locate_ok_button 을 래핑한다.

    notify_fn 은 fallback(live_align_search) escalation 알림 콜백으로 그대로 전달된다.
    모든 종료 분기는 console([INFO]/[ERROR]) + 파일 로그(log_work2_event)로 기록한다.
    OK 탐지 *예외*는 정상 'OK 안 보임'(escalated_no_ok)과 구분해 ok_detect_error 로 surface
    하며 error 필드에 예외 요약을 담는다(조용히 삼켜 escalate 로 위장하지 않는다).
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

    # ensemble 경로(decision/score 정비): decision 은 calibrated sel 임계 재판정, orb=0(폐지).
    # key_visibility_gate 의 adjust 분기는 orb>0 → distinctive 로 대체(위 게이트 참조).
    result = compute_align_key_score_ensemble(
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
            notify_fn=notify_fn,
            debug_dir=(debug_dir / "fallback") if debug_dir is not None else None,
        )
        result_outcome = CorrectionOutcome(
            status=f"fallback_{outcome.status}",
            path="fallback",
            key_decision=result.decision,
            best_xy=None,
            ok_screen_xy=None,
            fallback=outcome,
            history=history,
        )
        log_work2_event(
            component=LOG_COMPONENT,
            message="fallback_delegated",
            level="warning",
            status=result_outcome.status,
            key_decision=result.decision,
            pan_count=outcome.pan_count,
        )
        return _with_key_ambiguity(result_outcome, result)

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
        from poc.workflow_3.vision.vlm_ok_button_box import locate_ok_button

        resolved_locator = lambda f: locate_ok_button(frame_bgr=f, client=vlm_client)  # noqa: E731

    ok_xy: tuple[int, int] | None = None
    ok_error: str | None = None
    if resolved_locator is not None:
        screen = controller.capture_screen()
        try:
            ok_xy = resolved_locator(screen)
        except Exception as exc:
            # 예외는 '정상 not-found' 와 다르다. 조용히 삼키지 않고 정확히 로깅한 뒤
            # ok_detect_error 로 surface 한다(console + 파일 로그 + 전체 traceback).
            ok_error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] OK 버튼 탐지 중 예외: {ok_error}")
            traceback.print_exc()
            log_work2_event(
                component=LOG_COMPONENT,
                message="ok_detect_error",
                level="error",
                key_decision=result.decision,
                best_xy=f"({cx},{cy})",
                error=ok_error,
            )
    else:
        print("[WARNING] OK locator 가 없습니다(vlm_client/ok_locator 미주입).")
        log_work2_event(
            component=LOG_COMPONENT,
            message="ok_locator_missing",
            level="warning",
            key_decision=result.decision,
        )

    # OK 탐지 *예외* → 정상 escalate 와 구분해 surface(견고성: 실제 버그를 숨기지 않음).
    if ok_error is not None:
        return _with_key_ambiguity(
            CorrectionOutcome(
                status="ok_detect_error",
                path="primary",
                key_decision=result.decision,
                best_xy=(cx, cy),
                ok_screen_xy=None,
                fallback=None,
                error=ok_error,
                history=history,
            ),
            result,
        )

    if ok_xy is None:
        if config.require_ok_button:
            print("[WARNING] OK 버튼을 찾지 못함(정상 not-found) → 엔지니어 확인용 escalate")
            log_work2_event(
                component=LOG_COMPONENT,
                message="escalated_no_ok",
                level="warning",
                key_decision=result.decision,
                best_xy=f"({cx},{cy})",
            )
            return _with_key_ambiguity(
                CorrectionOutcome(
                    status="escalated_no_ok",
                    path="primary",
                    key_decision=result.decision,
                    best_xy=(cx, cy),
                    ok_screen_xy=None,
                    fallback=None,
                    history=history,
                ),
                result,
            )
        print("[INFO] OK 버튼 생략(require_ok_button=false), reposition 까지만 수행")
    else:
        print(f"[INFO] OK 클릭: screen=({ok_xy[0]}, {ok_xy[1]}){' [dry-run]' if dry_run else ''}")
        if not dry_run:
            controller.click_screen(ok_xy[0], ok_xy[1])

    log_work2_event(
        component=LOG_COMPONENT,
        message="corrected",
        level="info",
        key_decision=result.decision,
        best_xy=f"({cx},{cy})",
        ok_screen_xy=str(ok_xy),
        dry_run=dry_run,
    )
    return _with_key_ambiguity(
        CorrectionOutcome(
            status="corrected",
            path="primary",
            key_decision=result.decision,
            best_xy=(cx, cy),
            ok_screen_xy=ok_xy,
            fallback=None,
            history=history,
        ),
        result,
    )


def correct_align_fail_auto(
    controller: SEMMonitorController,
    *,
    vlm_client=None,
    ok_locator: OkLocator | None = None,
    config: CorrectionConfig = CorrectionConfig(),
    notify_fn: NotifyFn | None = None,
    dry_run: bool = True,
    debug_dir: Path | None = None,
    eqp_id: str = "",
    class_name: str = "",
    recipe_name: str = "",
) -> CorrectionOutcome:
    """자산을 자동 해석(resolve_assets_auto)해 template 을 만들고 correct_align_fail 실행.

    eqp_id/class_name/recipe_name 을 주면 resolve_assets_auto 의 override 로 전달된다
    (알람 RECIPE_ID 가 "<class>/<recipe>" 형태면 recipe_name 에 그대로 줘도 된다).
    미지정 시 기존처럼 최신 align fail 폴더를 자동 선택한다.
    """
    assets = resolve_assets_auto(
        eqp_id=eqp_id, class_name=class_name, recipe_name=recipe_name
    )
    if assets is None:
        print("[ERROR] align fail recipe 폴더를 찾지 못했습니다.")
        log_work2_event(component=LOG_COMPONENT, message="no_assets", level="error")
        return CorrectionOutcome("no_assets", "primary", "low", None, None, None)
    templates = build_templates_from_assets(assets, crop_to_box=config.crop_template_to_box)
    if not templates:
        print(f"[ERROR] 등록 OM/SEM 이미지가 없습니다: {assets.recipe_dir}")
        log_work2_event(
            component=LOG_COMPONENT, message="no_assets", level="error",
            recipe_dir=str(assets.recipe_dir),
        )
        return CorrectionOutcome("no_assets", "primary", "low", None, None, None)
    return correct_align_fail(
        controller,
        templates,
        vlm_client=vlm_client,
        ok_locator=ok_locator,
        config=config,
        notify_fn=notify_fn,
        dry_run=dry_run,
        debug_dir=debug_dir,
    )


# ==================================================================
# Mac 데모 — key 가 paused 화면에 이미 보이는 primary 경로(가상 화면 + mock).
# ==================================================================


def _make_primary_demo(key_in_view: bool = True):
    """가상 wafer + mock 컨트롤러 + 등록 template 을 만든다.

    key_in_view=True  → key 를 시작 FOV 중앙에 박아 PRIMARY 경로가 바로 fire.
    key_in_view=False → key 를 wafer 에 두지 않아 paused 프레임이 featureless →
      가시성 게이트 low → FALLBACK 위임. mock 은 stateful(pan 하면 capture 가 바뀜)이라
      live_align_search 의 실제 pan/zoom 전이를 exercise 한다.
    """
    from poc.workflow_3.vision.live_align_search import _MockSEMMonitor
    from poc.workflow_3.vision.test_align_key_match import make_synthetic_template, make_wafer_background

    pattern = make_synthetic_template(key_type="box")  # 128px native.
    recipe_img = cv2.copyMakeBorder(pattern, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    template = build_template(recipe_img, recipe_id="DEMO-SEM-001", version="v0", key_type="sem")

    wafer = make_wafer_background(frame_size=(2400, 3200))
    target_xy = (900, 1300)
    if key_in_view:
        th, tw = pattern.shape[:2]
        wafer[target_xy[1] - th // 2 : target_xy[1] + th // 2, target_xy[0] - tw // 2 : target_xy[0] + tw // 2] = pattern
        start_xy = target_xy  # native 배율 + key 가 중앙 → 즉시 보임.
    else:
        start_xy = (700, 1400)  # key 없음 → 어디를 봐도 featureless.

    # 가상 전체 화면: 검은 배경에 흰 'OK' 사각형(stub locator 가 그 중심을 돌려줌).
    screen = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(screen, (640, 540), (740, 580), 255, 2)

    monitor = _MockSEMMonitor(
        wafer,
        screen_size=(512, 768),
        start_xy=start_xy,
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
