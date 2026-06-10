"""Step 4~7 — live SEM monitor 를 조작하며 align key 위치를 찾는 two-phase search loop.

설계(이 세션의 grill 결과):

* Phase A (broad): 저배율로 zoom-out 한 상태에서 recipe template 을 크게 축소한
  ``BROAD_SCALES`` 로 "miniature" key 후보를 찾으며 사각 spiral 로 pan 한다.
  pan(=새 영역 탐색)만 10회 budget 에 카운트한다(step 7).
* Phase B (confirm): 강한 후보가 보이면 그 점을 더블클릭으로 recenter 한 뒤,
  discrete wheel 로 한 단계씩 zoom-in 하며 scale~1.0 부근에서 재매칭해 확정한다.
  recenter/zoom 은 budget 에 포함하지 않되 각각 별도 상한으로 묶는다.

물리 규약(실장비):
* 더블클릭 = 클릭 지점을 FOV 중심으로 recenter (배율 불변). 한 번에 최대 ~FOV 절반 pan.
* wheel = FOV 중심 기준 discrete 배율 단계.
* monitor mode(OM/SEM) 에 따라 recipe OM/SEM template 중 하나로 매칭(template routing).

판정(grill 결과): align fail 은 대개 live key 가 등록 이미지와 다르므로 hard match 를
강요하지 않는다. structure 위주(``STRUCTURE_POLICY``) 로 후보를 ranking 하고, budget
소진 시 *최고 점수 후보* 를 recenter/zoom 해 엔지니어 확인용으로 보고한다.

실장비 연결은 capture/move/zoom/read_mode 4개 주입점으로만 한다(office-only). 본 모듈은
Mac 에서 배율을 흉내내는 가상 wafer mock 으로 two-phase 흐름을 검증할 수 있다.

알려진 한계(중요 — 실데이터 calibration/probe 로 풀어야 함):
* Broad(miniature) 단계의 chamfer 단독 점수는 *변별력이 약하다*. template 을 크게
  축소(~0.15~0.3)하면 edge 픽셀 수가 적어, feature 없는 배경 위에서도 chamfer 가
  높게 나오기 쉽다. ORB 도 그 스케일에선 keypoint 를 못 잡는다. 따라서 broad 단계는
  "여기에 key 가 있을 수 있다" 는 *후보 제안* 수준이며, 진짜 align key 인지의 판정은
  반드시 confirm 단계(zoom-in 후 scale~1.0 + ORB 일치)로 미룬다. 본 loop 의 terminal
  match 가드(``best_scale>=MIN_CONFIRM_SCALE and orb>0``)가 broad 단계의 과신으로
  인한 거짓 종료를 막는다. 다만 "miniature 가 정말 align key 인가" 를 저배율에서
  직접 변별하는 일은 CV 만으로는 신뢰도가 낮으며, 이것이 step 1·2 의 VLM probe
  (``vlm_align_key_box.py``)로 별도 평가하려는 바로 그 능력이다.
* 따라서 align fail 처럼 live key 가 등록 이미지와 다른 hard 케이스에서 broad 단계가
  엉뚱한 후보를 쫓을 수 있다. budget(pan 10회) 과 escalation 으로 무한 루프만 막고,
  최종 책임 판정은 best-candidate 를 엔지니어에게 넘기는 것으로 둔다.

실행(Mac 데모):
    uv run python poc/workflow_3/vision/live_align_search.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.align_key_matcher import (
    BROAD_SCALES,
    STRUCTURE_POLICY,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)
from poc.workflow_3.vision.search_align_key import _square_spiral_step


# 두 phase 모두에서 쓰는 단일 wide scale band. zoom-in 으로 key 가 miniature(작은
# scale) → native(≈1.0) 로 커지는 동안 후보가 끊기지 않도록 broad~confirm 을 잇는다.
WIDE_SCALES = BROAD_SCALES + (0.65, 0.8, 1.0, 1.2)

# 확정(terminal) match 가드: 충분히 zoom-in 되어(>=0.6) feature 도 일치할 때만
# "match" 로 종료한다. tiny-scale chamfer 단독 고득점의 과신을 막는다.
MIN_CONFIRM_SCALE = 0.6


# ------------------------------------------------------------------
# 실장비 adapter 주입점 (office-only). Mac mock 은 동일 시그니처로 구현.
# ------------------------------------------------------------------


class SEMMonitorController(Protocol):
    """live SEM monitor 제어 인터페이스.

    구현 4종만 채우면 동일 loop 가 사무실/Mac 양쪽에서 돈다.
    """

    def capture(self) -> np.ndarray:
        """현재 FOV 를 grayscale numpy 로 반환."""

    def move_to_point(self, fov_x: int, fov_y: int) -> None:
        """FOV 픽셀 (fov_x, fov_y) 을 더블클릭해 그 점을 중심으로 recenter."""

    def zoom(self, direction: int) -> None:
        """wheel 한 단계. direction=+1 zoom-in, -1 zoom-out (FOV 중심 기준)."""

    def read_mode(self) -> str:
        """monitor mode label 을 반환 ('OM' | 'SEM' | 'unknown')."""

    def capture_screen(self) -> np.ndarray:
        """전체 화면(또는 RCS 창) 을 반환한다 — SEM ROI 가 아니라 dialog 포함.

        ``capture()`` 는 SEM Monitor ROI 만 자르지만, OK 같은 dialog 버튼은 ROI
        밖에 있어 별도의 전체 프레임이 필요하다. 반환 좌표가 곧 screen 좌표가 되어
        ``click_screen`` 으로 그대로 넘길 수 있다.
        """

    def click_screen(self, screen_x: int, screen_y: int) -> None:
        """SCREEN(절대) 픽셀 (screen_x, screen_y) 을 단일 클릭한다.

        OK 같은 dialog 버튼은 SEM ROI *밖* 에 있으므로 FOV-local 좌표를 쓰는
        move_to_point(=ROI 내부 더블클릭 recenter) 와 달리 화면 절대 좌표로 누른다.
        두 제스처의 좌표공간이 다르다는 점이 핵심. 실장비 구현은 SAFE_MODE 로
        실제 마우스 출력을 막는다.
        """


# ------------------------------------------------------------------
# 설정 / 상태 / 결과.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSearchConfig:
    """search policy 파라미터 (recipe/Tool 별 캘리브레이션 대상)."""

    pan_budget: int = 10  # step 7 — pan(새 영역 탐색)만 카운트하는 hard cap.
    initial_zoom_out_steps: int = 3  # 시작 시 broad 시야 확보용 zoom-out.
    max_zoom_in_steps: int = 4  # Phase B 에서 candidate 당 zoom-in 상한.
    low_streak_limit: int = 5  # 연속 low → 엔지니어 escalation.
    # pan 한 step 의 FOV 픽셀 이동량. 더블클릭은 최대 ~FOV 절반까지만 가능하므로
    # FOV 폭의 절반보다 작게 둔다(아래 clamp_to_fov 으로 한 번 더 보정).
    pan_step_px: int = 220
    # Phase A→B 전이 임계: broad score 가 이 값 이상이면 후보로 보고 추격.
    candidate_score: float = STRUCTURE_POLICY.adjust_threshold
    # 더블클릭 좌표가 FOV 가장자리에 너무 붙지 않도록 한 안쪽 여백 비율.
    click_margin_ratio: float = 0.12


@dataclass
class CandidateRecord:
    """지금까지 본 최고 후보(엔지니어 보고용)."""

    score: float
    fov_xy: tuple[int, int]  # 후보가 보였던 FOV 내 좌표.
    iter_idx: int
    phase: str
    decision: str


@dataclass
class LiveSearchState:
    phase: str = "broad"  # "broad" | "confirm"
    pan_count: int = 0
    zoom_in_count: int = 0
    low_streak: int = 0
    spiral_idx: int = 0
    best: CandidateRecord | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class LiveSearchOutcome:
    status: str  # "match" | "best_candidate" | "exhausted" | "escalated"
    final_decision: str
    best: CandidateRecord | None
    pan_count: int
    history: list[dict]


NotifyFn = Callable[[LiveSearchState, list[dict]], None]


# ------------------------------------------------------------------
# 좌표 헬퍼.
# ------------------------------------------------------------------


def clamp_to_fov(fov_x: int, fov_y: int, w: int, h: int, margin_ratio: float) -> tuple[int, int]:
    """클릭 좌표를 FOV 안쪽(여백 포함)으로 clamp.

    실장비 클릭은 FOV 밖을 누를 수 없고, 가장자리 클릭은 한 번에 최대치를 pan 한다.
    안전 여백을 둔다. live search 의 recenter 와 primary correction 의 reposition 이
    공유하는 헬퍼다.
    """
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    return (
        int(min(max(fov_x, mx), w - mx)),
        int(min(max(fov_y, my), h - my)),
    )


# ------------------------------------------------------------------
# 메인 two-phase loop.
# ------------------------------------------------------------------


def live_align_search(
    controller: SEMMonitorController,
    templates: dict[str, AlignKeyTemplate],
    *,
    config: LiveSearchConfig = LiveSearchConfig(),
    notify_fn: NotifyFn | None = None,
    debug_dir: Path | None = None,
    settle_sec: float = 0.0,
) -> LiveSearchOutcome:
    """live SEM monitor 에서 align key 를 찾는다.

    ``templates`` 는 {"OM": ..., "SEM": ...}. 매 iteration 마다 monitor mode 를 읽어
    해당 template 으로 라우팅한다(없으면 SEM, 그것도 없으면 임의 한 개).
    """
    if not templates:
        raise ValueError("templates 가 비어 있습니다 (OM/SEM 중 최소 하나 필요)")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    state = LiveSearchState()

    # 시작: broad 시야 확보를 위해 zoom-out. (budget 비포함)
    for _ in range(max(0, config.initial_zoom_out_steps)):
        controller.zoom(-1)
    if settle_sec:
        time.sleep(settle_sec)

    iter_idx = 0
    last_decision = "low"
    while True:
        frame = controller.capture()
        fh, fw = frame.shape[:2]
        mode = (controller.read_mode() or "").upper()
        template = route_template(templates, mode)

        # 단일 wide band 로 매칭한다. phase 는 scale 이 아니라 *행동*(pan vs zoom-in)만 좌우.
        result = compute_align_key_score(
            template, frame, scales=WIDE_SCALES, policy=STRUCTURE_POLICY
        )
        last_decision = result.decision

        _update_best(state, result, iter_idx)
        _log_iter(state, result, iter_idx, mode, fw, fh, debug_dir)

        # ---- 종료: 확정(terminal) match — 충분히 zoom-in 되고 feature 도 일치해야. ----
        confirmed = (
            result.decision == "match"
            and result.best_scale >= MIN_CONFIRM_SCALE
            and result.orb_inlier_ratio > 0.0
        )
        if confirmed:
            return LiveSearchOutcome(
                status="match",
                final_decision="match",
                best=state.best,
                pan_count=state.pan_count,
                history=state.history,
            )

        if state.phase == "broad":
            if result.score >= config.candidate_score:
                # 후보 발견 → recenter 하고 confirm phase 로. (pan budget 비포함)
                cx, cy = clamp_to_fov(
                    result.best_xy[0], result.best_xy[1], fw, fh, config.click_margin_ratio
                )
                controller.move_to_point(cx, cy)
                state.phase = "confirm"
                state.zoom_in_count = 0
                state.low_streak = 0
            else:
                # 아무 것도 없음 → spiral 로 pan(새 영역). budget 카운트.
                if state.pan_count >= config.pan_budget:
                    return _finish_with_best(state, "exhausted", last_decision)
                state.low_streak += 1
                if state.low_streak >= config.low_streak_limit:
                    if notify_fn is not None:
                        notify_fn(state, state.history[-config.low_streak_limit:])
                    return _finish_with_best(state, "escalated", last_decision)
                _do_pan(controller, state, config, fw, fh)
        else:  # confirm phase
            if result.score >= config.candidate_score and state.zoom_in_count < config.max_zoom_in_steps:
                # 후보 유지 → 중심으로 미세 recenter 후 한 단계 zoom-in. (budget 비포함)
                cx, cy = clamp_to_fov(
                    result.best_xy[0], result.best_xy[1], fw, fh, config.click_margin_ratio
                )
                controller.move_to_point(cx, cy)
                controller.zoom(+1)
                state.zoom_in_count += 1
            else:
                # 후보를 놓쳤거나 zoom-in 상한 → broad 로 복귀해 계속 pan.
                state.phase = "broad"
                # 너무 좁아진 시야를 원복.
                for _ in range(state.zoom_in_count):
                    controller.zoom(-1)
                state.zoom_in_count = 0
                if state.pan_count >= config.pan_budget:
                    return _finish_with_best(state, "best_candidate", last_decision)
                _do_pan(controller, state, config, fw, fh)

        if settle_sec:
            time.sleep(settle_sec)
        iter_idx += 1


def route_template(templates: dict[str, AlignKeyTemplate], mode: str) -> AlignKeyTemplate:
    """monitor mode 로 OM/SEM template 을 고른다. 없으면 합리적 fallback.

    live search 와 primary correction 이 동일 라우팅을 쓰도록 공유 헬퍼로 둔다.
    """
    if mode in templates:
        return templates[mode]
    if "OM" in mode and "OM" in templates:
        return templates["OM"]
    if "SEM" in templates:
        return templates["SEM"]
    return next(iter(templates.values()))


def _do_pan(
    controller: SEMMonitorController,
    state: LiveSearchState,
    config: LiveSearchConfig,
    fw: int,
    fh: int,
) -> None:
    """사각 spiral 의 다음 step 만큼 pan(더블클릭 recenter). pan_count 증가."""
    state.spiral_idx += 1
    dx, dy = _square_spiral_step(state.spiral_idx, config.pan_step_px)
    # recenter-on-point: 중심에서 (dx,dy) 떨어진 점을 클릭하면 그 점이 새 중심이 된다.
    target_x, target_y = clamp_to_fov(
        fw // 2 + dx, fh // 2 + dy, fw, fh, config.click_margin_ratio
    )
    controller.move_to_point(target_x, target_y)
    state.pan_count += 1


def _update_best(state: LiveSearchState, result: AlignKeyMatchResult, iter_idx: int) -> None:
    if state.best is None or result.score > state.best.score:
        state.best = CandidateRecord(
            score=float(result.score),
            fov_xy=(int(result.best_xy[0]), int(result.best_xy[1])),
            iter_idx=iter_idx,
            phase=state.phase,
            decision=result.decision,
        )


def _finish_with_best(state: LiveSearchState, status: str, decision: str) -> LiveSearchOutcome:
    return LiveSearchOutcome(
        status=status,
        final_decision=decision,
        best=state.best,
        pan_count=state.pan_count,
        history=state.history,
    )


def _log_iter(
    state: LiveSearchState,
    result: AlignKeyMatchResult,
    iter_idx: int,
    mode: str,
    fw: int,
    fh: int,
    debug_dir: Path | None,
) -> None:
    record = {
        "iter": iter_idx,
        "phase": state.phase,
        "mode": mode,
        "pan_count": state.pan_count,
        "zoom_in_count": state.zoom_in_count,
        "decision": result.decision,
        "score": float(result.score),
        "chamfer": float(result.chamfer_score),
        "orb": float(result.orb_inlier_ratio),
        "best_scale": float(result.best_scale),
        "best_xy": [int(result.best_xy[0]), int(result.best_xy[1])],
    }
    state.history.append(record)
    if debug_dir is not None:
        save_overlay_jpeg(
            result.debug_overlay,
            debug_dir / f"iter_{iter_idx:03d}_{state.phase}_{result.decision}.jpg",
        )
    print(
        f"[INFO] iter={iter_idx:02d} phase={state.phase:<7} mode={mode or '-':<4} "
        f"pan={state.pan_count}/{'?'} decision={result.decision:<6} "
        f"score={result.score:.3f} (ch={result.chamfer_score:.3f} orb={result.orb_inlier_ratio:.3f}) "
        f"scale={result.best_scale:.2f}"
    )


# ==================================================================
# Mac 데모 — 배율을 흉내내는 가상 wafer + mock controller.
# ==================================================================


class _MockSEMMonitor:
    """가상 wafer 위에서 capture/move/zoom/read_mode 를 흉내내는 Mac mock.

    magnification 은 mag_index 로 표현한다. zoom_factor(mag_index) 가 클수록 한
    screen 프레임이 더 넓은 wafer 영역을 담으므로(저배율), 박혀 있는 align key 는
    화면에서 더 작게(miniature) 보인다.
    """

    def __init__(
        self,
        wafer: np.ndarray,
        *,
        screen_size: tuple[int, int],  # (h, w)
        start_xy: tuple[int, int],
        zoom_factors: tuple[float, ...],
        start_mag_index: int,
        mode: str = "SEM",
        screen_image: np.ndarray | None = None,
    ) -> None:
        self.wafer = wafer
        self.sh, self.sw = screen_size
        self.pos = list(start_xy)  # wafer 좌표(중심).
        self.zoom_factors = zoom_factors
        self.mag = int(start_mag_index)
        self.mode = mode
        self.screen_image = screen_image  # capture_screen 용 가상 전체 화면(선택).
        self.screen_clicks: list[tuple[int, int]] = []  # click_screen 호출 기록(검증용).

    def _factor(self) -> float:
        return self.zoom_factors[max(0, min(len(self.zoom_factors) - 1, self.mag))]

    def capture(self) -> np.ndarray:
        f = self._factor()
        win_w = int(self.sw * f)
        win_h = int(self.sh * f)
        cx, cy = self.pos
        x0 = int(cx - win_w / 2)
        y0 = int(cy - win_h / 2)
        # wafer 경계 밖은 100(중간 밝기) 패딩.
        crop = np.full((win_h, win_w), 100, dtype=np.uint8)
        wx0, wy0 = max(0, x0), max(0, y0)
        wx1 = min(self.wafer.shape[1], x0 + win_w)
        wy1 = min(self.wafer.shape[0], y0 + win_h)
        if wx1 > wx0 and wy1 > wy0:
            crop[wy0 - y0 : wy1 - y0, wx0 - x0 : wx1 - x0] = self.wafer[wy0:wy1, wx0:wx1]
        return cv2.resize(crop, (self.sw, self.sh), interpolation=cv2.INTER_AREA)

    def move_to_point(self, fov_x: int, fov_y: int) -> None:
        f = self._factor()
        # screen 클릭 → wafer 변위(recenter-on-point).
        self.pos[0] += int((fov_x - self.sw / 2) * f)
        self.pos[1] += int((fov_y - self.sh / 2) * f)

    def zoom(self, direction: int) -> None:
        self.mag = max(0, min(len(self.zoom_factors) - 1, self.mag + direction))

    def read_mode(self) -> str:
        return self.mode

    def capture_screen(self) -> np.ndarray:
        # 가상 전체 화면. 미지정 시 ROI 캡처를 그대로 돌려준다(데모/테스트는 OK locator 를 stub).
        if self.screen_image is not None:
            return self.screen_image
        return self.capture()

    def click_screen(self, screen_x: int, screen_y: int) -> None:
        # mock 은 화면 절대 클릭을 기록만 한다(가상 wafer 에는 영향 없음).
        self.screen_clicks.append((int(screen_x), int(screen_y)))


def _make_demo_template_and_wafer() -> tuple[AlignKeyTemplate, np.ndarray, tuple[int, int]]:
    """등록 SEM template(고배율 native) 과, 그 key 가 한 곳에 박힌 큰 wafer 를 만든다."""
    from poc.workflow_3.vision.test_align_key_match import (
        make_synthetic_template,
        make_wafer_background,
    )

    pattern = make_synthetic_template(key_type="box")  # 128px native.
    # recipe template: key + 약간의 context.
    recipe_img = cv2.copyMakeBorder(pattern, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    template = build_template(
        recipe_img, recipe_id="DEMO-SEM-001", version="v0", nm_per_pixel=None, key_type="sem"
    )

    wafer = make_wafer_background(frame_size=(2400, 3200))
    target_xy = (2900, 1400)  # 시작 위치/초기 FOV 밖 → pan 이 필요하도록 배치.
    th, tw = pattern.shape[:2]
    x0 = target_xy[0] - tw // 2
    y0 = target_xy[1] - th // 2
    wafer[y0 : y0 + th, x0 : x0 + tw] = pattern
    return template, wafer, target_xy


def main() -> int:
    print("[INFO] live_align_search Mac 데모 시작 (가상 wafer, 배율 시뮬)")
    template, wafer, target_xy = _make_demo_template_and_wafer()

    # zoom_factors: index0=가장 zoom-out(4배 넓은 시야 → key 1/4 크기), 마지막=native 1.0.
    zoom_factors = (4.0, 3.0, 2.0, 1.4, 1.0)
    monitor = _MockSEMMonitor(
        wafer,
        screen_size=(512, 768),
        start_xy=(700, 1400),  # target 에서 서쪽으로 떨어진 시작점(초기 FOV 밖).
        zoom_factors=zoom_factors,
        start_mag_index=len(zoom_factors) - 1,  # native 에서 시작 → loop 이 zoom-out 함.
        mode="SEM",
    )

    run_tag = time.strftime("%y%m%d_%H%M%S")
    debug_dir = DEBUG_IMAGE_DIR / "live_search_demo" / run_tag
    config = LiveSearchConfig(pan_budget=10, initial_zoom_out_steps=4, pan_step_px=240)

    outcome = live_align_search(
        monitor,
        {"SEM": template},
        config=config,
        debug_dir=debug_dir,
    )

    print(f"[INFO] status={outcome.status} final_decision={outcome.final_decision} pan_count={outcome.pan_count}")
    if outcome.best is not None:
        print(
            f"[INFO] best candidate: score={outcome.best.score:.3f} "
            f"fov_xy={outcome.best.fov_xy} iter={outcome.best.iter_idx} phase={outcome.best.phase}"
        )
    print(f"[INFO] target_xy(wafer)={target_xy}, debug overlays: {debug_dir}")
    # 데모 성공 기준: match 로 종료했거나, 강한 best candidate 를 확보.
    ok = outcome.status == "match" or (
        outcome.best is not None and outcome.best.score >= STRUCTURE_POLICY.match_threshold
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
