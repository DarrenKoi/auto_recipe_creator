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

실행(Mac 데모): uv run python poc/workflow_3/align/correction.py
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
from poc.workflow_3.align.assets import resolve_assets_auto
from poc.workflow_3.align.consensus_resolve import resolve_templates
from poc.workflow_3.align.matching.engine import (
    DEFAULT_SCALES,
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score_ensemble,
    save_overlay_jpeg,
)
from poc.workflow_3.align.live_search import (
    MIN_CONFIRM_SCALE,
    LiveSearchConfig,
    LiveSearchOutcome,
    NotifyFn,
    SEMMonitorController,
    clamp_to_fov,
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
    # OK 버튼을 실제로 누를지. False 면 reposition 까지만 하고 OK 는 엔지니어에게 넘긴다
    # (status=awaiting_engineer_ok -> cube 알림 발송). 실전 투입 초기의 반자동 모드로,
    # 잘못된 좌표가 그대로 측정으로 확정되는 것을 사람이 막을 수 있게 한다.
    # 라이브러리 기본은 설계된 전체 동작(True); 운영 루프 기본값은 Workflow3Settings 참조.
    ok_click_enabled: bool = True
    # key 부재 시 live_align_search(pan/zoom)로 위임할지. False 면 actuation 없이
    # escalated_key_not_visible 로 끝낸다 - 라이브러리 기본은 설계된 전체 동작(True),
    # 운영 루프 기본값은 Workflow3Settings.fallback_search_enabled 를 따른다.
    fallback_search_enabled: bool = True
    settle_sec: float = 0.0  # 제스처 후 대기(실장비 안정화).
    cond_box_crop: bool = True  # cond.box_ltrb 기반 box-crop template(+decoupled offset). False -> whole-template(구 동작) 롤백.
    # 만성 모호 키 게이트(Tier 0.1). second_ratio 가 이 값을 넘으면 present 라도 auto-act 대신
    # engineer_review 로 보류한다. None(기본)이면 게이트는 과거 act/fallback 2분기만 — 동작 불변.
    # 운영 루프는 Workflow3Settings.reregister_second_ratio_threshold(기본 0.98)를 주입한다.
    reregister_ratio_threshold: float | None = None
    # consensus 라우팅 설정(resolve_templates 에 그대로 전달).
    consensus_enabled: bool = True             # consensus 라우팅 마스터 토글(off -> 순수 rcp).
    consensus_min_s: int = 4                   # modality 별 신뢰 최소 S 수(floor 3).
    consensus_max_events: int = 8              # 캐시에서 로드할 최대 이벤트 수.
    consensus_sync_timeout_sec: float = 8.0    # cold-cache bounded 대기(초).


@dataclass
class CorrectionOutcome:
    """보정 결과. 어느 경로로 끝났는지 + 좌표/decision 기록."""

    # "corrected" | "awaiting_engineer_ok" | "fallback_<status>" | "escalated_ambiguous_key"
    # | "escalated_key_not_visible" | "escalated_no_ok" | "ok_detect_error" | "no_assets"
    #
    # monitor 계층이 사이클 문맥을 반영해 **추가 status 로 치환**할 수 있다(2026-08-18):
    # "view_only_observation"(다른 엔지니어 점유 - 보정 자체를 건너뜀),
    # "corrected_unverified"(점유 미상 - 보정했으나 반영 여부 미확인). 정의는
    # `monitor/notify.py` 에 있다 - align 은 RCS 세션 공유를 알 수 없고 monitor 를
    # 임포트할 수도 없기 때문이다(단방향: monitor -> align).
    # 따라서 이 필드를 소비할 때 **정확 비교(== / !=)만 쓸 것.** 접두사/부분 일치로
    # 판정하면 치환된 status 가 조용히 다른 분기로 샌다.
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


# 가시성 게이트가 돌려주는 route intent (Tier 0.1: bool → router).
GATE_ACT = "act"                          # primary reposition + OK.
GATE_FALLBACK = "fallback_search"         # live_align_search pan/zoom (키가 안 보임).
GATE_ENGINEER_REVIEW = "engineer_review"  # 키는 보이나 만성 모호 → 자동보정 보류, 엔지니어 확인.
# 예약(Tier 3): "vlm_region" — VLM ROI 힌트 후 CV fine-coord. 아직 핸들러 없음이라 미발행.


def key_visibility_gate(
    result: AlignKeyMatchResult,
    *,
    reregister_ratio_threshold: float | None = None,
) -> str:
    """paused frame 의 route intent 결정 — act(primary) vs fallback_search vs engineer_review.

    2단계 판정:

    1) presence — "키가 이 전체 프레임에 있는가"(존재/부재). 과거 bool 게이트와 동일 기준.
       ensemble 경로에서 decision 은 calibrated sel 임계(match 0.6053/adjust 0.4727)로
       재판정되므로 featureless 배경(sel~0.25)은 대개 "low" 로 1차 차단된다.
         * ``best_scale`` 가 충분(>=MIN_CONFIRM_SCALE)해야 한다 — tiny-scale chamfer 과신 차단.
         * 강한 ``match`` 는 edge 구조만으로 present 인정.
         * 약한 ``adjust`` 는 **구조 유일성(distinctive)** 이 있을 때만 present — 배경 거짓양성
           2차 차단(과거 orb>0 의 대체; ORB 폐지).
       present 아니면 → "fallback_search"(아무것도 안 보임 → live_align_search pan/zoom).

    2) isolation (Tier 0.1, opt-in) — present 라도 ``second_ratio`` 가
       ``reregister_ratio_threshold`` 를 넘으면 chamfer best peak 이 2nd 대비 고립되지 않은
       "만성 모호" key (S-LOO golden tau*, AUC0.91 미스예측)다. 평평한 score surface 에서
       확신 reposition+OK 는 오정렬·오클릭 위험이 크므로 auto-act 대신 "engineer_review" 로
       보류한다. 임계 None(기본)이면 이 단계를 건너뛰어 과거 act/fallback 2분기를 그대로 보존한다.

    반환: "act" | "fallback_search" | "engineer_review".
    임계/조건은 cold-start 이며 실데이터 calibration 대상.
    """
    # --- 1) presence: 과거 bool False 조건 → fallback_search. ---
    if result.best_scale < MIN_CONFIRM_SCALE:
        return GATE_FALLBACK
    present = result.decision == "match" or (
        result.decision == "adjust" and result.distinctive
    )
    if not present:
        return GATE_FALLBACK
    # --- 2) isolation(opt-in): present 하나 만성 모호 → 자동보정 보류. ---
    if (
        reregister_ratio_threshold is not None
        and result.second_ratio is not None
        and result.second_ratio > reregister_ratio_threshold
    ):
        return GATE_ENGINEER_REVIEW
    return GATE_ACT


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
    grid_mag=None,
    grid_reg_mag: float | None = None,
    grid_config=None,
) -> CorrectionOutcome:
    """paused Align Fail 화면에서 crosshair 를 recipe-matched 점으로 옮기고 OK.

    1) capture() 로 paused SEM ROI 프레임 캡처.
    2) read_mode() 로 OM/SEM template 라우팅(route_template).
    3) compute_align_key_score_ensemble(scales=PAUSED_SCALES, policy=STRUCTURE_POLICY).
    4) key_visibility_gate → route:
       - "act"             → clamp_to_fov(best_xy) → move_to_point(=더블클릭 recenter) →
                             capture_screen() 에서 OK 버튼을 찾아 click_screen(OK).
       - "fallback_search" → live_align_search(...) 로 위임(아무것도 안 보일 때만 pan/zoom).
       - "engineer_review" → present 하나 만성 모호(second_ratio>tau) → actuation 없이
                             escalated_ambiguous_key 로 보류(config.reregister_ratio_threshold 주입 시).

    dry_run=True 면 좌표를 계산·로그·overlay 만 하고 실제 actuation(move/click)은 하지
    않는다(Mac-safe, procedure §5 Phase 3). ok_locator 를 직접 주입하면 VLM 없이도
    테스트할 수 있고, 미주입 시 vlm_client 로 locate_ok_button 을 래핑한다.

    notify_fn 은 fallback(live_align_search) escalation 알림 콜백으로 그대로 전달된다.
    fallback 탐색은 ``grid_search.search_around`` 하나로 간다: ``grid_mag``(MagnificationControl)
    + ``grid_reg_mag``(등록 배율) 이 있으면 절대 배율 zoom-out + FOV 격자, 아니면(또는 grid 가
    배율 판독 실패로 degrade 하면) 종전 legacy live_align_search.
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
    # 끝 밴드 고정 = scale band 가 live box/template 실제 비율을 못 덮는다는 신호.
    # 후보가 있었을 때만 의미가 있다(no_candidates 는 best_scale=1.0 기본값).
    scale_pinned = bool(
        result.reject_reason != "no_candidates"
        and result.best_scale in (min(PAUSED_SCALES), max(PAUSED_SCALES))
    )
    if scale_pinned:
        print(
            f"[WARNING] best_scale={result.best_scale:.2f} 가 scale band "
            f"({min(PAUSED_SCALES)}~{max(PAUSED_SCALES)}) 끝에 고정 - band 가 "
            f"live box/template 비율을 못 덮을 수 있음 (좌표 신뢰도 확인 필요)"
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
            "scale_pinned": scale_pinned,
        }
    )
    print(
        f"[INFO] paused frame: mode={mode or '-'} decision={result.decision} "
        f"score={result.score:.3f} (ch={result.chamfer_score:.3f} orb={result.orb_inlier_ratio:.3f}) "
        f"scale={result.best_scale:.2f} best_xy={result.best_xy}"
    )
    if debug_dir is not None:
        save_overlay_jpeg(result.debug_overlay, debug_dir / "paused_match.jpg")

    # ---- 가시성 게이트: route intent 에 따라 분기 (act / fallback_search / engineer_review). ----
    route = key_visibility_gate(
        result, reregister_ratio_threshold=config.reregister_ratio_threshold
    )
    if route == GATE_FALLBACK and not config.fallback_search_enabled:
        # pan/zoom 을 하지 않고 끝낸다. actuation 이 전혀 없으므로 stage 는 그대로다.
        # status 는 corrected 가 아니라서 notify 가 cube 로 엔지니어를 부른다.
        print("[WARNING] key 가 paused 화면에 보이지 않음 + fallback search 비활성 "
              "(ALIGN_FAIL_FALLBACK_SEARCH=0) -> pan 없이 엔지니어 확인으로 넘깁니다.")
        log_work2_event(
            component=LOG_COMPONENT,
            message="escalated_key_not_visible",
            level="warning",
            key_decision=result.decision,
            score=f"{result.score:.3f}",
            best_scale=f"{result.best_scale:.2f}",
        )
        return _with_key_ambiguity(
            CorrectionOutcome(
                status="escalated_key_not_visible",
                path="primary",
                key_decision=result.decision,
                best_xy=None,
                ok_screen_xy=None,
                fallback=None,
                history=history,
            ),
            result,
        )

    if route == GATE_FALLBACK:
        from poc.workflow_3.align.grid_search import search_around

        outcome = search_around(
            controller, templates,
            grid_mag=grid_mag, reg_mag=grid_reg_mag, grid_config=grid_config,
            legacy_config=fallback_config, notify_fn=notify_fn,
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

    if route == GATE_ENGINEER_REVIEW:
        # key 는 present 하나 second_ratio>tau(만성 모호) — 평평한 score surface 에서 확신
        # reposition+OK 는 오정렬·오클릭 위험이 크다. 자동 보정을 보류하고 엔지니어 확인으로
        # escalate 한다(actuation 없음). status!=corrected 라 notify 가 cube 로 알린다.
        sr_txt = f"{result.second_ratio:.3f}" if result.second_ratio is not None else "-"
        print(f"[WARNING] align key 가 보이나 만성 모호(second_ratio={sr_txt}) → 자동 보정 보류, 엔지니어 확인")
        log_work2_event(
            component=LOG_COMPONENT,
            message="escalated_ambiguous_key",
            level="warning",
            key_decision=result.decision,
            second_ratio=sr_txt,
        )
        return _with_key_ambiguity(
            CorrectionOutcome(
                status="escalated_ambiguous_key",
                path="primary",
                key_decision=result.decision,
                best_xy=None,
                ok_screen_xy=None,
                fallback=None,
                history=history,
            ),
            result,
        )

    # ---- PRIMARY: crosshair 를 align point 로 reposition. ----
    # template 이 들고 다니는 align offset(=rcp px image_center - box_center)을 best_scale 로
    # 환산해 match 중심에 더한다 -> frame 의 진짜 align point. offset (0,0)이면 best_xy 그대로.
    ox, oy = template.align_offset_xy
    align_x = result.best_xy[0] + round(ox * result.best_scale)
    align_y = result.best_xy[1] + round(oy * result.best_scale)
    cx, cy = clamp_to_fov(align_x, align_y, fw, fh, config.click_margin_ratio)
    print(f"[INFO] reposition: 더블클릭 recenter → ({cx}, {cy}){' [dry-run]' if dry_run else ''}")
    if not dry_run:
        controller.move_to_point(cx, cy)
        if config.settle_sec:
            time.sleep(config.settle_sec)

    # ---- OK 버튼 위치 확인(screen 좌표) 후 single click. ----
    resolved_locator = ok_locator
    if resolved_locator is None and vlm_client is not None:
        from poc.workflow_3.align.ok_button import locate_ok_button

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

    # 반자동 모드: reposition 은 했지만 OK 는 누르지 않는다. 좌표 확정 권한을 사람에게
    # 남기는 경로라 status 를 corrected 와 반드시 구분해야 한다 — notify 는 corrected 면
    # cube 를 생략하므로, 여기서 corrected 를 돌려주면 "OK 눌러달라"는 알림이 사라진다.
    # OK 탐지 결과(좌표/예외)는 나중에 자동 클릭을 켤 때의 근거로 그대로 실어 보낸다.
    if not config.ok_click_enabled:
        ok_txt = f"screen={ok_xy}" if ok_xy is not None else f"미검출({ok_error or 'not found'})"
        print(f"[INFO] OK 클릭 보류(ok_click_enabled=false) - 엔지니어 확인 필요: {ok_txt}")
        log_work2_event(
            component=LOG_COMPONENT,
            message="awaiting_engineer_ok",
            level="warning",
            key_decision=result.decision,
            best_xy=f"({cx},{cy})",
            ok_xy=str(ok_xy),
            error=ok_error or "",
        )
        return _with_key_ambiguity(
            CorrectionOutcome(
                status="awaiting_engineer_ok",
                path="primary",
                key_decision=result.decision,
                best_xy=(cx, cy),
                ok_screen_xy=ok_xy,
                fallback=None,
                error=ok_error,
                history=history,
            ),
            result,
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
    fallback_config: LiveSearchConfig | None = None,
    notify_fn: NotifyFn | None = None,
    dry_run: bool = True,
    debug_dir: Path | None = None,
    eqp_id: str = "",
    class_name: str = "",
    recipe_name: str = "",
    grid_mag=None,
    grid_config=None,
) -> CorrectionOutcome:
    """자산을 자동 해석(resolve_assets_auto)해 template 을 만들고 correct_align_fail 실행.

    ``grid_mag``(grid_search.MagnificationControl) 가 오면 현재 modality 의 등록 이미지
    cond.txt 에서 Magnification 을 읽어 함께 넘긴다 — 없으면 search_around 가 legacy 로 간다.

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
        # 두 no_assets 를 구분해 둔다 - 알림에서 "경로 설정 문제" 와 "등록 이미지 없음"
        # 은 엔지니어가 할 일이 다르다.
        return CorrectionOutcome(
            "no_assets", "primary", "low", None, None, None,
            error="recipe 폴더 미검출(ALIGN_IMAGES_DIR / EQP·RECIPE 경로 확인)",
        )
    templates = resolve_templates(
        assets,
        # 자동선택(eqp_id="") 경로에선 resolve_assets_auto 가 채운 assets.eqp_id 가 권위.
        # 빈 eqp_id 를 넘기면 consensus cache 경로가 어긋나 항상 cold 로 보인다(리뷰).
        eqp_id=assets.eqp_id or eqp_id,
        consensus_enabled=config.consensus_enabled,
        min_s=config.consensus_min_s,
        max_events=config.consensus_max_events,
        sync_timeout_sec=config.consensus_sync_timeout_sec,
        cond_box_crop=config.cond_box_crop,
    )
    if not templates:
        print(f"[ERROR] 등록 OM/SEM 이미지가 없습니다: {assets.recipe_dir}")
        log_work2_event(
            component=LOG_COMPONENT, message="no_assets", level="error",
            recipe_dir=str(assets.recipe_dir),
        )
        return CorrectionOutcome(
            "no_assets", "primary", "low", None, None, None,
            error=f"등록 OM/SEM 이미지 없음: {assets.recipe_dir}",
        )
    reg_mag = None
    if grid_mag is not None:
        from poc.workflow_3.align.cond_file import load_cond
        from poc.workflow_3.align.grid_search import registered_magnification

        mode = (controller.read_mode() or "").upper()
        rcp = assets.recipe_om if "OM" in mode else assets.recipe_sem
        reg_mag = registered_magnification(load_cond(rcp)) if rcp is not None else None
    return correct_align_fail(
        controller,
        templates,
        vlm_client=vlm_client,
        ok_locator=ok_locator,
        config=config,
        # None 이면 라이브러리 기본값(LiveSearchConfig()) - 인자를 안 넘기던 기존 동작 유지.
        fallback_config=fallback_config or LiveSearchConfig(),
        notify_fn=notify_fn,
        dry_run=dry_run,
        debug_dir=debug_dir,
        grid_mag=grid_mag,
        grid_reg_mag=reg_mag,
        grid_config=grid_config,
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
    from poc.workflow_3.align.live_search import _MockSEMMonitor
    from poc.workflow_3.align.matching.test_engine import make_synthetic_template, make_wafer_background

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
