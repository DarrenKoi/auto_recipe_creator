"""Tool 화면에서 시작해 SEM monitor 의 FOV 를 옮겨가며 align key 를 찾는 search loop.

설계 문서 ``docs/search_align_key.md`` §5 의 흐름을 그대로 코드로 옮겼다.

```
[FOV 캡처] ─→ [compute_align_key_score] ─→
   match  → 좌표 반환, 종료
   adjust → (선택) VLM 보조 조언 → stage 이동
   low    → low_count++, 한도 도달 시 엔지니어 호출 → stage 이동
```

본 모듈은 두 가지 *injection point* 를 둔다.

- ``capture_fn(state) -> np.ndarray`` — 현재 stage 위치에서 SEM monitor FOV
  를 grayscale numpy 로 반환. 사무실에서는 RCS 창 캡처, Mac 데모에서는
  가상 wafer 의 일부 윈도우.
- ``move_stage_fn(state, dx, dy) -> AlignKeySearchState`` — stage 를
  (dx, dy) 만큼 이동시킨 새 상태를 반환. 사무실에서는 pywinauto 클릭/드래그,
  Mac 데모에서는 단순 좌표 업데이트.

이 분리 덕분에 동일 search loop 를 사무실 (Windows + RCS) 과 Mac (합성
wafer) 양쪽에서 그대로 돌릴 수 있다.

Mac 데모 실행:
    uv run python poc/workflow_2/search_align_key.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_key_matcher import (
    AlignKeyMatchResult,
    AlignKeyTemplate,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)


# ------------------------------------------------------------------
# 설정 / 상태 / 결과 dataclass.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AlignKeySearchConfig:
    """search loop 의 정책 파라미터.

    실 운영에서는 recipe / Tool 별로 다를 수 있어 dict[recipe_id, config] 로
    보관할 예정 (§7.3 의 임계값 캘리브레이션 후).
    """

    max_iters: int = 20
    low_streak_limit: int = 5  # §7.3 의 N=5 hysteresis.
    coarse_step_px: int = 200  # 한 번의 stage 이동 거리 (FOV pixel 기준).
    spiral_growth: float = 1.0  # 사각 spiral 의 단계 증가율.


@dataclass
class AlignKeySearchState:
    """search loop 가 추적하는 가변 상태."""

    position: tuple[int, int]  # 현재 stage 좌표 (개념적, capture_fn 가 해석).
    iter_idx: int = 0
    low_streak: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class AlignKeySearchOutcome:
    """search 종료 결과."""

    status: str  # "match" | "exhausted" | "escalated"
    final_position: tuple[int, int] | None
    last_result: AlignKeyMatchResult | None
    iter_count: int
    history: list[dict]


# ------------------------------------------------------------------
# 다음 stage 이동 결정.
# ------------------------------------------------------------------


def _square_spiral_step(idx: int, step: int) -> tuple[int, int]:
    """사각 spiral 의 idx 번째 *step delta* — 직전 위치 대비 (dx, dy).

    Leg 길이는 1,1,2,2,3,3,4,4,... 이고 방향은 R,U,L,D 순환이다. 따라서
    누적 좌표는 ``(0,0) → (1,0) → (1,-1) → (0,-1) → (-1,-1) → (-1,0) → ...``.
    호출자는 매 iteration 마다 idx=1,2,3,... 으로 호출해 *현재 위치에 더할*
    delta 를 받는다 (origin 기준 누적값이 아님 — 그게 이전 버그였음).

    coarse search 의 default 전략. VLM 힌트가 있으면 override 가능.
    """
    if idx <= 0:
        return 0, 0
    # idx 가 어느 leg 에 속하는지 찾는다. Leg 1,2 는 길이 1, leg 3,4 는 길이 2,
    # 일반적으로 leg L 의 길이는 ⌈L/2⌉. 누적 길이가 idx 이상이 되면 그 leg.
    leg = 0
    cum = 0
    while cum < idx:
        leg += 1
        leg_length = (leg + 1) // 2
        cum += leg_length
    direction = (leg - 1) % 4  # 0=R, 1=U, 2=L, 3=D.
    if direction == 0:
        return step, 0
    if direction == 1:
        return 0, -step
    if direction == 2:
        return -step, 0
    return 0, step


# ------------------------------------------------------------------
# 메인 search loop.
# ------------------------------------------------------------------


CaptureFn = Callable[[AlignKeySearchState], np.ndarray]
MoveStageFn = Callable[[AlignKeySearchState, int, int], AlignKeySearchState]
VLMConsultFn = Callable[[AlignKeyMatchResult, AlignKeySearchState], tuple[int, int]]
NotifyFn = Callable[[AlignKeySearchState, list[AlignKeyMatchResult]], None]


def search_align_key(
    template: AlignKeyTemplate,
    initial_state: AlignKeySearchState,
    capture_fn: CaptureFn,
    move_stage_fn: MoveStageFn,
    *,
    config: AlignKeySearchConfig = AlignKeySearchConfig(),
    vlm_consult_fn: VLMConsultFn | None = None,
    notify_fn: NotifyFn | None = None,
    debug_dir: Path | None = None,
) -> AlignKeySearchOutcome:
    """Tool 화면에서 시작해 align key 의 stage / FOV 위치를 찾아낸다.

    종료 조건은 셋 중 하나.

    1. ``match`` 판정 → 위치 반환.
    2. ``low`` 가 ``config.low_streak_limit`` 회 연속 → 엔지니어 호출 후
       ``"escalated"`` 상태로 종료.
    3. ``config.max_iters`` 도달 → ``"exhausted"`` 상태로 종료.
    """
    state = initial_state
    last_result: AlignKeyMatchResult | None = None
    recent_lows: list[AlignKeyMatchResult] = []

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i in range(config.max_iters):
        state.iter_idx = i
        frame = capture_fn(state)

        result = compute_align_key_score(template, frame)
        last_result = result

        record = {
            "iter": i,
            "position": list(state.position),
            "decision": result.decision,
            "score": float(result.score),
            "chamfer": float(result.chamfer_score),
            "orb": float(result.orb_inlier_ratio),
            "best_xy": list(result.best_xy),
        }
        state.history.append(record)

        if debug_dir is not None:
            save_overlay_jpeg(
                result.debug_overlay,
                debug_dir / f"iter_{i:03d}_{result.decision}.jpg",
            )

        print(
            f"[INFO] iter={i:02d} pos={state.position} decision={result.decision} "
            f"score={result.score:.3f} (chamfer={result.chamfer_score:.3f} "
            f"orb={result.orb_inlier_ratio:.3f})"
        )

        if result.decision == "match":
            return AlignKeySearchOutcome(
                status="match",
                final_position=state.position,
                last_result=result,
                iter_count=i + 1,
                history=state.history,
            )

        # adjust / low 별 다음 행동 결정.
        if result.decision == "adjust":
            state.low_streak = 0
            if vlm_consult_fn is not None:
                dx, dy = vlm_consult_fn(result, state)
            else:
                # VLM 미장착 prototype: 작은 step 으로 nudge.
                dx, dy = _square_spiral_step(i + 1, config.coarse_step_px // 4)
        else:  # "low"
            state.low_streak += 1
            recent_lows.append(result)
            if state.low_streak >= config.low_streak_limit:
                if notify_fn is not None:
                    notify_fn(state, recent_lows[-config.low_streak_limit:])
                else:
                    print(
                        f"[WARNING] low_streak={state.low_streak} 도달 — 엔지니어 호출"
                        f" 핸들러 미장착, escalated 로 종료"
                    )
                return AlignKeySearchOutcome(
                    status="escalated",
                    final_position=state.position,
                    last_result=result,
                    iter_count=i + 1,
                    history=state.history,
                )
            dx, dy = _square_spiral_step(i + 1, config.coarse_step_px)

        state = move_stage_fn(state, dx, dy)

    return AlignKeySearchOutcome(
        status="exhausted",
        final_position=state.position,
        last_result=last_result,
        iter_count=config.max_iters,
        history=state.history,
    )


# ------------------------------------------------------------------
# Mac 데모 — 가상 wafer + mock capture / move.
# ------------------------------------------------------------------


def _build_virtual_wafer(
    template_pattern: np.ndarray,
    *,
    wafer_size: tuple[int, int] = (1600, 2400),
    target_xy: tuple[int, int] = (1100, 1700),
    seed: int = 7,
) -> np.ndarray:
    """저주파 노이즈 배경 + 한 위치에 합성 align key 를 박은 거대한 wafer."""
    rng = np.random.default_rng(seed)
    h, w = wafer_size
    coarse = rng.normal(0, 30, (h // 16, w // 16))
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    wafer = np.clip(130 + coarse, 60, 200).astype(np.uint8)

    th, tw = template_pattern.shape[:2]
    tx, ty = target_xy
    x0 = max(0, tx - tw // 2)
    y0 = max(0, ty - th // 2)
    wafer[y0 : y0 + th, x0 : x0 + tw] = template_pattern
    return wafer


def _make_capture_fn(wafer: np.ndarray, fov_size: tuple[int, int]):
    """state.position 을 중심으로 fov_size 윈도우를 잘라서 grayscale 반환."""
    fh, fw = fov_size

    def _capture(state: AlignKeySearchState) -> np.ndarray:
        cx, cy = state.position
        x0 = int(cx - fw // 2)
        y0 = int(cy - fh // 2)
        # 경계 보정 — FOV 가 wafer 밖으로 나가면 그 방향만 0 으로 패딩.
        # 주의: numpy 의 wafer[a:b] 는 b<0 일 때 "끝에서 |b|" 로 해석되므로,
        # 슬라이스 양 끝을 모두 [0, dim] 안으로 명시 clamp 한 뒤
        # x1c < x0c (또는 y1c < y0c) 인 경우 빈 view 처리.
        x0c = max(0, min(wafer.shape[1], x0))
        y0c = max(0, min(wafer.shape[0], y0))
        x1c = max(x0c, min(wafer.shape[1], x0 + fw))
        y1c = max(y0c, min(wafer.shape[0], y0 + fh))
        view = wafer[y0c:y1c, x0c:x1c]
        if view.shape != (fh, fw):
            padded = np.full((fh, fw), 100, dtype=np.uint8)
            ox = x0c - x0
            oy = y0c - y0
            if view.size > 0:
                padded[oy : oy + view.shape[0], ox : ox + view.shape[1]] = view
            return padded
        return view.copy()

    return _capture


def _move_stage_mock(state: AlignKeySearchState, dx: int, dy: int) -> AlignKeySearchState:
    """가상 stage 이동 — position 만 갱신."""
    cx, cy = state.position
    return AlignKeySearchState(
        position=(cx + dx, cy + dy),
        iter_idx=state.iter_idx,
        low_streak=state.low_streak,
        history=state.history,
    )


def _make_synthetic_template_for_demo() -> np.ndarray:
    """test_align_key_match 의 box 패턴과 동일 구성 (간단히 인라인)."""
    size = 128
    img = np.full((size, size), 120, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    dark = 40
    for r_ratio, t in ((0.42, 4), (0.28, 3), (0.14, 3)):
        r = int(size * r_ratio)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), dark, t)
    outer = int(size * 0.42)
    mid = int(size * 0.28)
    ax = cx - int((outer + mid) * 0.5)
    ay = cy - int((outer + mid) * 0.5)
    for dx, dy, r in ((0, 0, 4), (10, 0, 3), (0, 8, 3)):
        cv2.circle(img, (ax + dx, ay + dy), r, dark, -1)
    cv2.line(img, (ax + 14, ay - 2), (ax + 14, ay + 6), dark, 2)
    cv2.line(img, (ax + 14, ay + 6), (ax + 6, ay + 6), dark, 2)
    bx = cx + int((outer + mid) * 0.5)
    by = cy + int((outer + mid) * 0.5)
    cv2.circle(img, (bx, by), 4, dark, -1)
    cv2.line(img, (bx - 6, by - 6), (bx + 4, by - 6), dark, 2)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return img


def main() -> int:
    """Mac 데모 — 가상 wafer 위에서 search loop 가 align key 를 찾아내는지 확인."""
    print("[INFO] search_align_key Mac 데모 시작")

    template_pattern = _make_synthetic_template_for_demo()
    template = build_template(
        template_pattern,
        recipe_id="DEMO-BOX-001",
        version="v0",
        nm_per_pixel=None,
        key_type="box",
    )

    fov_size = (512, 768)  # height, width
    wafer_size = (1600, 2400)  # (h, w)
    # 정답을 spiral 의 두 번째 step 위치 근처에 배치 (R 방향 한 번 이동 후 진입).
    target_xy = (1500, 800)  # (x, y)
    wafer = _build_virtual_wafer(
        template_pattern,
        wafer_size=wafer_size,
        target_xy=target_xy,
        seed=7,
    )
    capture_fn = _make_capture_fn(wafer, fov_size)

    # 시작 위치는 target 으로부터 약 500px 서쪽 — iter 0 FOV 에는 안 들어옴.
    initial = AlignKeySearchState(position=(1000, 800))

    run_tag = time.strftime("%y%m%d_%H%M%S")
    debug_dir = Path(DEBUG_IMAGE_DIR) / "search_demo" / run_tag

    config = AlignKeySearchConfig(
        max_iters=25,
        low_streak_limit=5,
        coarse_step_px=400,  # FOV 폭의 약 절반.
    )

    outcome = search_align_key(
        template,
        initial,
        capture_fn,
        _move_stage_mock,
        config=config,
        debug_dir=debug_dir,
    )

    print(f"[INFO] outcome.status = {outcome.status}")
    print(f"[INFO] outcome.final_position = {outcome.final_position}")
    print(f"[INFO] outcome.iter_count = {outcome.iter_count}")
    print(f"[INFO] debug overlays: {debug_dir}")

    if outcome.status == "match" and outcome.last_result is not None:
        # final_position 은 FOV 중심. matcher 가 반환한 best_xy 는 FOV 내부 좌표.
        # FOV 중심 + (best_xy - FOV 중심) = wafer 좌표.
        cx, cy = outcome.final_position
        fov_h, fov_w = fov_size
        bx, by = outcome.last_result.best_xy
        wafer_xy = (cx - fov_w // 2 + bx, cy - fov_h // 2 + by)
        err = float(np.hypot(wafer_xy[0] - target_xy[0], wafer_xy[1] - target_xy[1]))
        print(f"[INFO] localized wafer_xy={wafer_xy} ground_truth={target_xy} err={err:.1f}px")
        return 0 if err < 30.0 else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
