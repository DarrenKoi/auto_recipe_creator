"""Search-around 재설계 — 절대 배율 zoom-out + FOV 단위 격자 sweep + odometry.

설계: ``docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md``.
기존 ``live_search``(휠 + 프레임 픽셀 spiral)는 착지점 근처(~1 FOV)를 못 벗어난다 —
step 이 픽셀이라 배율에 역비례로 줄고 zoom-out 은 배율을 거의 안 바꾸는 휠이기 때문.
여기서는 모든 거리를 **FOV 비율**, 모든 scale 을 **배율비**로 다룬다.

단위계: 매칭 scale = ``base(fw / source_wh[0]) x cur_mag / reg_mag`` 를 **원본 template** 에
건다 - primary(correction)와 글자 그대로 같은 호출이라 '첫 화면에서는 잡히는데 탐색에서는 안
잡히는' 갈림이 없다. 표시 px 환산은 크기 계산(zoom-out 단/보폭/중복 반경)에만 쓴다.
FOV_um = 135,000 / Mag (``docs/study/hitachi_mag_fov_pixel_260828.md``).

커버리지: 격자 보폭은 1 FOV 가 아니라 **검출 footprint**(프레임 - template)다(`footprint_stride`).
판정: 'key 가 있다' 는 ``accept_fn`` 하나 - correction 이 primary 와 같은 게이트를 넘긴다.

배율 변경은 컨트롤러 Protocol 에 넣지 않고 **주입 함수**로 받는다(``MagnificationControl``).
PM 드롭다운 + OCR 판독 코드는 office-only 라 ``monitor/cycle.py`` 에 남고, 이 모듈은 Mac 에서
mock 으로 전부 검증된다. 진입점은 ``search_around`` — grid 가 배율을 못 읽으면 legacy 로 넘긴다.
"""

import math
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

import poc.workflow_3.align.live_search as _live_search
from poc.workflow_3.align.live_search import (
    MIN_CONFIRM_SCALE,
    CandidateRecord,
    LiveSearchConfig,
    LiveSearchOutcome,
    clamp_to_fov,
    route_template,
)
from poc.workflow_3.align.matching.engine import (
    DEFAULT_SCALES,
    STRUCTURE_POLICY,
    AlignKeyTemplate,
    _resize_template,
    build_template,
    compute_align_key_score_ensemble,
    save_overlay_jpeg,
    template_frame_scale,
)
from poc.workflow_3.align.search_pattern import square_spiral_step
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted

# 기준 화면 폭(µm). FOV_um = FOV_UM_CONSTANT / Mag. 상수는 OFFICE-VERIFY(표본 1건 0.02% 차).
FOV_UM_CONSTANT = 135_000.0

# OM 은 SEM panel 위 휠 한 칸으로 두 단을 오간다: 위 = 210, 아래 = 104 (사용자 확인 2026-09-17).
OM_WHEEL_UP_MAG = 210.0
OM_WHEEL_DOWN_MAG = 104.0


# ------------------------------------------------------------------
# 순수 계산.
# ------------------------------------------------------------------


def fov_um(mag: float) -> float:
    """배율 -> 시료 위 FOV 폭(µm)."""
    return FOV_UM_CONSTANT / float(mag)


def key_px_at(mag: float, reg_mag: float, key_px: float) -> float:
    """등록 배율의 표시 key 크기 → 배율 mag에서 남는 크기."""
    return float(key_px) * float(mag) / float(reg_mag)


def choose_zoom_out_mag(options, reg_mag: float, key_px: float, min_key_px: int):
    """key 가 ``min_key_px`` 이상으로 남는 **가장 낮은** 드롭다운 배율. 없거나 등록 배율
    이상이면 None(zoom-out 하지 않는다).

    key_px는 FOV 폭이 아니라 표시 scale로 환산한 template crop의 짧은 변이다.
    """
    ok = sorted(m for m in options if key_px_at(m, reg_mag, key_px) >= min_key_px)
    if not ok or ok[0] >= reg_mag:
        return None
    return ok[0]


def spiral_cells(count: int) -> list[tuple[int, int]]:
    """착지 셀 (0,0) 을 제외한 사각 spiral 순서의 셀 오프셋 count 개."""
    cells: list[tuple[int, int]] = []
    x = y = 0
    for idx in range(1, max(0, count) + 1):
        dx, dy = square_spiral_step(idx, 1)
        x, y = x + dx, y + dy
        cells.append((x, y))
    return cells


# 격자 보폭 = 한 프레임의 **검출 footprint**. matcher 는 valid-mode 라 template 창이 프레임에
# 통째로 들어가는 자리만 본다 - key 중심이 놓일 수 있는 폭은 (프레임 - template)이지 1 FOV 가
# 아니다. 종전 1-FOV 보폭은 footprint 끼리 이어지지 않아 대각선으로 0.2~0.8 FOV 어긋난 key 를
# 어느 프레임에서도 온전히 못 봤다(2026-09-19 진단: key 옆을 지나가고도 sweep 을 계속 돈 원인).
STRIDE_OVERLAP = 0.9   # odometry/클릭 오차만큼 footprint 를 겹친다.
# 보폭 하한은 0 을 피하는 용도일 뿐이다(template >= 프레임이면 footprint <= 0). footprint 보다
# 크게 잡으면 그만큼 구멍이 다시 생긴다 - 셀 수는 pan_budget 이 묶으므로 하한을 키울 이유가 없다.
MIN_STRIDE_FOV = 0.1
SMALL_FOOTPRINT_FOV = 0.25  # 이보다 좁으면 예산 안에서 덮는 면적이 작다 - 경고만.


def footprint_stride(frame_px: float, tpl_px: float) -> float:
    """탐색 배율에서 template 표시 크기 -> 그 축의 격자 보폭(px)."""
    return float(min(frame_px, max(MIN_STRIDE_FOV * frame_px, (frame_px - tpl_px) * STRIDE_OVERLAP)))


def plan_grid(fov_um: float, radius_um: float, budget: int) -> list[tuple[int, int]]:
    """2R 박스를 덮는 홀수 n×n 격자의 셀 오프셋(셀 단위)을 spiral 순서로 낸다.

    ``fov_um`` 은 **셀 한 칸**의 시료 위 크기다(보폭이 footprint 라 1 FOV 보다 작다).
    n = ceil(2R / 셀) 를 홀수로 올려 착지 셀이 중심에 오게 한다. 예산을 넘으면 안쪽
    링부터 예산만큼만(바깥 링 일부 생략). 셀 하나가 이미 2R 을 덮으면 빈 목록.
    """
    n = math.ceil(2.0 * radius_um / fov_um)
    if n <= 1:
        return []
    if n % 2 == 0:
        n += 1
    return spiral_cells(min(n * n - 1, budget))


def _nearest_mag(options, value):
    """value 에 가장 가까운 배율. 동률이면 낮은 쪽. 빈 목록이면 None."""
    if not options:
        return None
    return min(options, key=lambda m: (abs(m - value), m))


def registered_magnification(cond):
    """cond.txt 의 Magnification(단위 없는 str, 예 '30000') -> float. 없으면 None."""
    return cond.magnification if cond is not None else None


def normalize_template(template: AlignKeyTemplate, fw: int) -> AlignKeyTemplate:
    """원본 FOV 폭 기준으로 template/offset을 표시 픽셀에 맞춘다.

    이후 매칭 scale 은 순수 배율비 ``cur_mag / reg_mag`` 가 된다. align_offset 도 같은 비율.
    """
    raw = template.raw_image
    if template.source_wh is None:
        raise ValueError("missing_source_geometry")
    sw, sh = template.source_wh
    base = template_frame_scale(template, (sh * fw / sw, fw))
    if abs(base - 1.0) < 1e-3:
        return template
    ox, oy = template.align_offset_xy
    return build_template(
        _resize_template(raw, base), recipe_id=template.recipe_id, version=template.version,
        nm_per_pixel=None, key_type=template.key_type,
        align_offset_xy=(int(round(ox * base)), int(round(oy * base))),
        source_wh=(fw, round(sh * base)),
        source_magnification=template.source_magnification,
    )


def phase_correlate_shift(prev: np.ndarray, cur: np.ndarray):
    """연속 프레임의 stage 이동량(px). 이미지가 (dx,dy) 움직였으면 stage 는 반대로 간 것.

    Hanning 창을 씌운다 - 주기 구조(SEM line/space)에서 경계 누설이 가짜 피크를 만든다.
    """
    try:
        a = prev.astype(np.float32)
        b = cur.astype(np.float32)
        h, w = a.shape[:2]
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        if not np.isfinite(dx) or not np.isfinite(dy) or resp <= 0.0:
            return None
        return (-float(dx), -float(dy))
    except Exception:
        return None


class Odometer:
    """클릭마다 (명령, 측정) 이동량을 받아 게이트를 지난 값만 누적한다.

    측정값은 연속 프레임 phase correlation 에서 온다. 주기 구조 위에서는 한 주기 어긋난
    값이 높은 cc 로 나올 수 있으므로 ``|측정 − 명령| > tol_fov × FOV`` 면 명령값으로 폴백하고
    ``drift_flags`` 를 올린다(중단이 아니라 기록). 측정이 없으면(None) 명령값.
    ``position`` 은 원점 기준 누적(px, 탐색 배율 프레임 기준) - 원점 복귀에 쓴다.
    """

    def __init__(self, fov_px: int, tol_fov: float) -> None:
        self.tol_px = float(tol_fov) * float(fov_px)
        self.position = (0.0, 0.0)
        self.drift_flags = 0
        self.log: list[dict] = []

    def record(self, commanded, measured):
        cx, cy = float(commanded[0]), float(commanded[1])
        used = (cx, cy)
        flagged = False
        if measured is not None:
            mx, my = float(measured[0]), float(measured[1])
            if abs(mx - cx) <= self.tol_px and abs(my - cy) <= self.tol_px:
                used = (mx, my)
            else:
                flagged = True
                self.drift_flags += 1
        self.position = (self.position[0] + used[0], self.position[1] + used[1])
        self.log.append({
            "commanded": [cx, cy],
            "measured": None if measured is None else [float(measured[0]), float(measured[1])],
            "used": list(used), "flagged": flagged,
        })
        return used


# ------------------------------------------------------------------
# 설정 / 주입점.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GridSearchConfig:
    """탐색 정책. 운영 루프는 cycle.py 가 Workflow3Settings.search_* 에서 조립한다."""

    radius_um: float = 30.0        # 탐색 반경 R(시험값, 2026-08-28). 박스 = 2R.
    min_key_px: int = 60           # zoom-out 후 key 가 이보다 작아지면 그 단은 안 쓴다(오피스 실측 상수).
    pan_budget: int = 10           # SEM sweep 셀 수 상한(= 1 FOV step 수).
    # OM spiral 셀 수 상한. OM 은 저배율이라 한 칸이 wafer 위에서 크게 움직인다 - SEM(고배율 한 칸은
    # 작다)과 따로 줄인다. 8 = 착지 셀 둘레 한 바퀴(3x3). 2026-09-17 사용자 결정.
    om_pan_budget: int = 8
    click_margin_ratio: float = 0.12  # recenter 클릭의 FOV 안쪽 여백 -> 1 클릭 최대 0.38 FOV.
    odom_tol_fov: float = 0.15     # |측정 - 명령| 허용(FOV 비율). 넘으면 명령값 폴백 + flag.
    # 추격 대상 최소 점수(sweep 은 zoom-out 단일 scale 매칭이라 key 가 보여도 점수가 낮다 -
    # 2026-09-15 오피스: OM key 위를 지나가고도 추격 0회). 종전 0.40 은 `decision != "low"`
    # 와 겹쳐 실효 임계가 ensemble_adjust 0.4727 이었다(겉보기 게이트). 이제 점수만 본다 -
    # 정확성은 등록 배율 confirm(decision == "match") 이 지킨다. env ALIGN_FAIL_SEARCH_CANDIDATE_SCORE.
    candidate_score: float = 0.30
    max_chase: int = 3             # 추격할 후보 수 상한(점수순). 추격마다 배율 왕복이 들어간다.


@dataclass(frozen=True)
class MagnificationControl:
    """배율 주입점 - PM 드롭다운 옵션 읽기와 절대 배율 선택기.

    ``options_fn() -> [배율, ...]`` 은 실장비에서 **드롭다운을 여는 일**이라 선택 직전에 한 번만
    불린다(연 김에 바로 행을 눌러야 한다). ``set_fn(target) -> 판독 배율 | None`` 의 판독은 PM
    box OCR 이며 None 이면 '모름' 이다 - 명령값을 믿지 않는다(계약 1). ``read_fn() -> 판독 배율 |
    None`` 은 클릭 없이 지금 PM box 만 읽는다(OM 휠 토글 전후 확인용). Mac 은 list/lambda.
    """

    options_fn: Callable[[], list]
    set_fn: Callable[[float], float | None]
    read_fn: Callable[[], float | None] | None = None


def _om_to_registered_step(controller, mag: MagnificationControl, reg_mag: float, meta: dict) -> float:
    """OM 을 등록 이미지와 같은 단(104/210 중 cond 배율에 가까운 쪽)으로 휠 토글하고, 매칭에 쓸
    현재 배율(등록 배율 단위)을 돌려준다.

    OM 은 단마다 화면이 달라 보여 다른 단에서 recipe/consensus 이미지를 scale 만 바꿔 대 보면
    놓친다 - 같은 단으로 맞추고 찾는다. cond 의 OM 배율은 '104 부근'(recipe 마다 다름)이라 판독값을
    그대로 쓰지 않고 단 비율(판독 / 목표 단)로 환산한다. 휠이 원격에 안 먹었으면 실제로 보이는 단의
    비율로 매칭한다. PM 판독이 없으면 종전처럼 등록 단에 있다고 보고 진행한다.
    """
    target = min((OM_WHEEL_DOWN_MAG, OM_WHEEL_UP_MAG), key=lambda m: abs(m - reg_mag))
    read = mag.read_fn() if mag.read_fn is not None else None
    meta["om_mag_before"] = read
    if read is None:
        meta["reason"] = "om_mag_unreadable"
        print("[WARNING] grid search: OM PM 배율 판독 실패 - 등록 단에 있다고 보고 진행")
        return reg_mag
    if abs(read - target) > 1.0:
        print(f"[INFO] grid search: OM {read:.0f} -> 등록 단 {target:.0f} 로 휠 "
              f"{'위' if target > read else '아래'} 한 칸")
        controller.zoom(1 if target > read else -1)
        after = mag.read_fn()
        meta["om_mag_after"] = after
        if after is None:
            meta["reason"] = "om_mag_unreadable"
            print("[WARNING] grid search: 휠 후 OM PM 배율 판독 실패 - 등록 단으로 바뀌었다고 보고 진행")
            return reg_mag
        if abs(after - target) > 1.0:
            meta["reason"] = "om_wheel_not_applied"
            print(f"[WARNING] grid search: 휠 후에도 OM {after:.0f} - 그 단 비율로 매칭")
        read = after
    return reg_mag * read / target


class _Stage:
    """sweep/추격/복귀가 공유하는 이동 원시 연산. 위치는 탐색 배율 프레임 px 로 누적.

    settle 은 하지 않는다 - 실장비 controller 가 move_to_point 뒤에 스스로 쉰다.
    """

    def __init__(self, controller, fw, fh, config: GridSearchConfig, shift_fn, odometer: Odometer):
        self.c, self.fw, self.fh, self.cfg = controller, fw, fh, config
        self.shift_fn = shift_fn
        self.odo = odometer
        # 축별 1 클릭 상한(0.38 FOV) - 프레임이 정사각이 아니면 y 는 fh 기준이다.
        self.max_click_x = (0.5 - config.click_margin_ratio) * fw
        self.max_click_y = (0.5 - config.click_margin_ratio) * fh
        self.frame = None  # 마지막 캡처(odometry 기준).

    def capture(self):
        self.frame = self.c.capture()
        return self.frame

    def _click(self, dx, dy):
        """중심에서 (dx,dy) 떨어진 점을 더블클릭 -> stage 가 (dx,dy) 만큼 간다."""
        x, y = clamp_to_fov(self.fw / 2 + dx, self.fh / 2 + dy, self.fw, self.fh,
                            self.cfg.click_margin_ratio)
        cmd = (x - self.fw / 2, y - self.fh / 2)
        prev = self.frame
        self.c.move_to_point(int(x), int(y))
        cur = self.capture()
        measured = self.shift_fn(prev, cur) if (self.shift_fn is not None and prev is not None) else None
        self.odo.record(cmd, measured)

    def move_px(self, dx, dy, stop_when=None) -> bool:
        """(dx,dy) px 만큼 stage 를 옮긴다 - 한 클릭 최대 0.38 FOV 로 쪼갠다. abort 면 False.

        이동량이 1px 미만이면 클릭하지 않는다. ``max(1, ...)`` 때문에 delta 0 도 FOV
        중심을 한 번 더블클릭했는데, 그건 이동이 아니라 잡음이다 - 격자가 0 셀일 때
        (한 FOV 가 이미 2R 을 덮는 경우) 복귀의 이 클릭 하나가 화면에서는 "탐색이
        한 번 움직이고 끝났다" 로 보인다.
        """
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            return True
        n = max(1, math.ceil(abs(dx) / self.max_click_x), math.ceil(abs(dy) / self.max_click_y))
        for _ in range(n):
            if is_aborted():
                return False
            self._click(dx / n, dy / n)
            # 다음 클릭 전에 현재 프레임을 판정한다. True면 그 위치에 그대로 둔다.
            if stop_when is not None and stop_when(self.frame, self.odo.position):
                break
        return True

    def move_to(self, tx, ty, stop_when=None) -> bool:
        px, py = self.odo.position
        return self.move_px(tx - px, ty - py, stop_when=stop_when)


# ------------------------------------------------------------------
# 오케스트레이션.
# ------------------------------------------------------------------


def grid_align_search(
    controller,
    templates: dict,
    mag: MagnificationControl,
    *,
    reg_mag: float,
    config: GridSearchConfig = GridSearchConfig(),
    match_fn: Callable[..., object] | None = None,
    shift_fn: Callable[[np.ndarray, np.ndarray], tuple | None] | None = phase_correlate_shift,
    notify_fn=None,
    debug_dir: Path | None = None,
    accept_fn: Callable[[object], bool] | None = None,
) -> LiveSearchOutcome:
    """절대 배율 zoom-out -> 매 클릭 판정 -> 후보 추격/confirm -> 복귀.

    status: "match" | "exhausted" | "aborted" | "degraded"(배율 판독 실패 - 호출부가 legacy
    경로로 넘긴다). meta 에 search_mag/cells_visited/odometry/final_position_px/restore_failed
    를 남긴다. ``notify_fn(state, history)`` 는 legacy 와 같은 escalation 콜백 - 못 찾고
    끝날 때 한 번 부른다(cycle 의 live_search_escalation 감사 로그가 grid 경로에서도 남게).

    ``accept_fn(match_result) -> bool`` 은 'key 가 있다' 의 판정이다. correction 은 primary 와
    **같은** ``key_visibility_gate`` 를 넘긴다 - 게이트가 둘이면 갈린다(종전 탐색은 match 만
    받아, primary 가 받는 adjust+distinctive key 를 중심에 두고도 지나쳤다). 미주입 = match-only.
    """
    match = match_fn or (lambda t, f, **kw: compute_align_key_score_ensemble(
        t, f, policy=STRUCTURE_POLICY, **kw))
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    reg_mag = float(reg_mag)
    meta: dict = {"search_mag": None, "cells_visited": 0, "drift_flags": 0,
                  "restore_mag": None, "restore_failed": False, "reason": None}
    history: list[dict] = []

    frame = controller.capture()
    fh, fw = frame.shape[:2]
    mode = (controller.read_mode() or "").upper()
    template = route_template(templates, mode)
    if template.source_wh is None:
        meta["reason"] = "missing_source_geometry"
        return _outcome("degraded", None, 0, history, meta)
    try:
        base = template_frame_scale(template, frame.shape)
    except ValueError as exc:
        meta["reason"] = "invalid_source_geometry"
        print(f"[WARNING] grid source geometry: {exc}")
        return _outcome("degraded", None, 0, history, meta)
    # 매칭은 primary 와 **같은 호출**이다: 원본 template 에 scale = base x 배율비. template 을
    # 미리 리샘플하면 edge 를 다시 뽑고 NCC 가 두 번 보간되어, 같은 key 가 primary 에서는 잡히고
    # 탐색에서는 안 잡힐 여지가 생긴다. 크기 계산(zoom-out 단/보폭/중복 반경)만 표시 px 로 한다.
    tw0, th0 = template.raw_image.shape[1] * base, template.raw_image.shape[0] * base
    accept = accept_fn or (lambda r: r.decision == "match"
                           and r.best_scale / base >= MIN_CONFIRM_SCALE)
    meta.update(source_wh=list(template.source_wh), frame_wh=[fw, fh], base_scale=base)
    odo = Odometer(fov_px=fw, tol_fov=config.odom_tol_fov)
    stage = _Stage(controller, fw, fh, config, shift_fn, odo)
    stage.frame = frame

    # ---- §1 zoom-out (SEM 만). ----
    cur_mag = reg_mag
    back_target = None  # SEM 배율 선택 시: 등록 배율 최근접 단(confirm/복귀용).
    if "OM" in mode:
        cur_mag = _om_to_registered_step(controller, mag, reg_mag, meta)
        stage.frame = controller.capture()
    else:
        options = mag.options_fn()
        if not options:
            # OCR 이 행을 하나도 못 읽었다. 내릴 단도, 열린 드롭다운을 닫을 행도 없다 -
            # selector 가 닫기를 시도했고, 여기서는 모르는 상태로 격자를 돌리지 않는다.
            meta["reason"] = "no_mag_options"
            print("[WARNING] grid search: PM 드롭다운 옵션 0개 -> legacy 경로로 degrade")
            return _outcome("degraded", None, 0, history, meta)
        target = choose_zoom_out_mag(options, reg_mag, min(tw0, th0), config.min_key_px)
        back_target = _nearest_mag(options, reg_mag)
        if target is None:
            # 등록 배율 최근접 단으로 닫아도 실제 배율은 다를 수 있어 반드시 판독한다.
            target = back_target
        read = mag.set_fn(target)
        if read is None:
            meta["reason"] = "mag_unreadable"
            print("[WARNING] grid search: 배율 선택 후 PM 배율 판독 실패 -> legacy 경로로 degrade")
            return _outcome("degraded", None, 0, history, meta)
        cur_mag = float(read)
        stage.frame = controller.capture()
    meta["search_mag"] = cur_mag
    scale = cur_mag / reg_mag
    # 셀 보폭 = 축별 footprint. n 은 짧은 쪽 보폭으로 잡아야 커버에 구멍이 안 난다.
    step_x, step_y = footprint_stride(fw, tw0 * scale), footprint_stride(fh, th0 * scale)
    if "OM" in mode:
        cells = spiral_cells(config.om_pan_budget)
    else:
        cells = plan_grid(fov_um=fov_um(cur_mag) * min(step_x, step_y) / fw,
                          radius_um=config.radius_um, budget=config.pan_budget)
    meta["stride_fov"] = [round(step_x / fw, 3), round(step_y / fh, 3)]
    print(f"[INFO] grid search: reg={reg_mag:.0f} search={cur_mag:.0f} scale={scale:.3f} "
          f"fw={fw} stride=({step_x / fw:.2f},{step_y / fh:.2f})FOV cells={len(cells)}")
    if min(fw - tw0 * scale, fh - th0 * scale) < SMALL_FOOTPRINT_FOV * min(fw, fh):
        print(f"[WARNING] grid search: template({tw0 * scale:.0f}x{th0 * scale:.0f}) 이 프레임"
              f"({fw}x{fh})을 거의 채운다 - 이 배율에서는 key 가 중심 근처일 때만 보인다"
              "(예산 안에서 덮는 면적이 작다). 더 낮은 배율 단이 있는지 PM 드롭다운/min_key_px 확인")

    # 탐색이 매긴 프레임마다 match overlay(찾은 박스 + score/decision) 한 장. history 의 image 가
    # 그 파일을 가리켜 "key 위를 지나갔는데 왜 안 잡혔나" 를 grid_search.json 과 함께 대조한다.
    frames_dir = debug_dir / "grid_frames" if debug_dir is not None else None
    if frames_dir is not None:
        try:
            save_overlay_jpeg(_resize_template(template.raw_image, base * scale), frames_dir / "template_search.jpg")
            meta["template_image"] = "grid_frames/template_search.jpg"
        except Exception as exc:
            print(f"[WARNING] grid template 이미지 저장 실패: {exc}")

    def _log_frame(rec, overlay):
        seq = len(history) - 1
        print(f"[INFO] grid #{seq:03d} {rec['phase']:<7} cell={tuple(rec['cell'])} "
              f"score={rec['score']:.3f} decision={rec['decision']} xy={tuple(rec['xy'])}")
        if frames_dir is None:
            return
        name = (f"{seq:03d}_{rec['phase']}_c{rec['cell'][0]}_{rec['cell'][1]}"
                f"_{rec['decision']}_{rec['score']:.2f}.jpg")
        try:
            save_overlay_jpeg(overlay, frames_dir / name)
            rec["image"] = f"grid_frames/{name}"
        except Exception as exc:
            print(f"[WARNING] grid frame 저장 실패({name}): {exc}")

    def _score(frame, pos, cell, phase, mag_ratio=None):
        """프레임 하나를 primary 와 같은 호출로 매긴다. accepted 는 confirm 가능한 배율에서만 참."""
        ratio = scale if mag_ratio is None else mag_ratio
        r = match(template, frame, scales=tuple(base * ratio * s for s in DEFAULT_SCALES))
        ox, oy = template.align_offset_xy
        align_xy = [int(r.best_xy[0] + round(ox * r.best_scale)), int(r.best_xy[1] + round(oy * r.best_scale))]
        # pos = 이 프레임 중심의 누적 위치(px), target = 후보 align point 의 누적 위치(추격 목적지).
        # scale 은 등록 표시 크기 대비(= best_scale / base) - MIN_CONFIRM_SCALE 과 같은 단위다.
        rec = {"cell": list(cell), "phase": phase, "pos": [float(pos[0]), float(pos[1])],
               "target": [float(pos[0]) + align_xy[0] - fw / 2, float(pos[1]) + align_xy[1] - fh / 2],
               "score": float(r.score), "xy": align_xy,
               "match_xy": list(r.best_xy), "scale": float(r.best_scale / base),
               "decision": r.decision, "orb": float(r.orb_inlier_ratio),
               "distinctive": bool(r.distinctive), "second_ratio": r.second_ratio,
               # 부호를 살린 NCC(기록 전용) - 큰 음수면 자리는 맞고 화면 극성이 반대다.
               "ncc": None if r.best_ncc is None else float(r.best_ncc),
               "accepted": bool(ratio >= MIN_CONFIRM_SCALE and accept(r))}
        history.append(rec)
        _log_frame(rec, r.debug_overlay)
        return rec

    # 같은 key 를 여러 프레임에서 본 후보는 한 번만 쫓는다 - 이동 중 프레임끼리 크게 겹쳐 한 key 가
    # 2~3번 잡히고, 그대로 두면 max_chase 를 한 자리에 다 쓴다. 반경은 탐색 배율의 key 크기다
    # (그 안의 다른 후보는 confirm 프레임이 어차피 함께 본다).
    key_px = max(tw0, th0) * scale
    chased: list[dict] = []
    best: CandidateRecord | None = None
    mag_unknown = False

    def _chasable(rec):
        return (len(chased) < max(0, config.max_chase)
                and not any(abs(rec["target"][0] - c["target"][0]) < key_px
                            and abs(rec["target"][1] - c["target"][1]) < key_px for c in chased))

    def _chase(rec) -> str:
        """후보를 중심으로 데려와 등록 배율(최근접 단)에서 판정. "found" | "miss" | "stop"."""
        nonlocal best, mag_unknown
        chased.append(rec)
        print(f"[INFO] grid search: 추격 {len(chased)}/{config.max_chase} "
              f"score={rec['score']:.3f} decision={rec['decision']} (본 프레임 #{history.index(rec):03d})")
        if not stage.move_to(*rec["target"]):
            return "stop"
        back_mag = cur_mag
        if back_target is not None and abs(back_target - cur_mag) > 1e-6:
            back = mag.set_fn(back_target)
            if back is None:
                # 배율을 바꾸려 했는데 판독이 없다 = 장비가 어느 배율인지 모른다. 모르는 scale 로
                # confirm 하지도, px 단위를 모른 채 stage 를 더 옮기지도 않는다(계약 1).
                meta["reason"] = "mag_unreadable_confirm"
                mag_unknown = True
                print("[WARNING] grid search: confirm 배율 판독 실패 -> 추격 중단")
                return "stop"
            back_mag = float(back)
        # confirm 게이트 = accept(primary 와 같은 판정) + 판독 배율비 >= 0.6. legacy 의 orb>0 는
        # 쓰지 않는다 - SEM junction key 는 ORB 특징점이 빈약해 진짜 match 도 orb=0 이다.
        c = _score(stage.capture(), odo.position, rec["cell"], "confirm", mag_ratio=back_mag / reg_mag)
        if c["accepted"]:
            best = CandidateRecord(score=c["score"], fov_xy=tuple(c["xy"]), iter_idx=len(history),
                                   phase="confirm", decision=c["decision"])
            meta.update(restore_mag=back_mag, confirm_distinctive=c["distinctive"],
                        confirm_second_ratio=c["second_ratio"])
            return "found"
        # 놓침 -> 탐색 배율로 돌아가 이어 간다. 복귀 판독이 없으면 이후 이동의 px 단위를 모른다.
        if abs(back_mag - cur_mag) > 1e-6:
            restored = mag.set_fn(cur_mag)
            if restored is None or abs(float(restored) - cur_mag) > 0.01 * cur_mag:
                # 판독이 없거나 다른 단이다 - 탐색 배율 px 로 계속 움직이면 전부 어긋난다.
                meta["reason"] = "mag_unreadable_restore"
                mag_unknown = True
                print(f"[WARNING] grid search: 탐색 배율 복귀 실패(판독={restored}, 목표={cur_mag:.0f}) "
                      "-> 탐색 중단")
                return "stop"
            stage.frame = controller.capture()
        return "miss"

    def _react(rec) -> str:
        """방금 매긴 프레임에 대한 조치. "" = 계속 sweep."""
        if rec["accepted"]:
            # confirm 가능한 배율에서 primary 게이트를 통과 - 그 자리에 둔다(상위가 reposition).
            nonlocal best
            best = CandidateRecord(score=rec["score"], fov_xy=tuple(rec["xy"]), iter_idx=0,
                                   phase=rec["phase"], decision=rec["decision"])
            meta.update(restore_mag=cur_mag, confirm_distinctive=rec["distinctive"],
                        confirm_second_ratio=rec["second_ratio"])
            return "found"
        # zoom-out 에서는 accept 가 구조적으로 불가(scale < 0.6)다. 강한 후보를 두고 남은 셀을 다
        # 돌면 수십 번의 이동 뒤에야 되돌아오고 그 사이 odometry 오차가 쌓인다 - 바로 쫓는다.
        # 조건은 match 다: 작은 scale 에서는 배경도 adjust(0.47~0.49)를 내서, adjust 로 쫓으면
        # 진짜 key 를 보기 전에 예산을 다 쓴다(test_chase_finds_key_one_search_fov_away 가 잡았다).
        # 나머지 후보는 종전대로 sweep 뒤 점수순 - 그 몫으로 추격 1회는 반드시 남긴다.
        if (scale < MIN_CONFIRM_SCALE and rec["decision"] == "match"
                and len(chased) < config.max_chase - 1 and _chasable(rec)):
            return _chase(rec)
        return ""

    # ---- §2 sweep: 매 클릭 직후 판정. accept 는 즉시 멈추고, 강한 후보는 즉시 쫓는다. ----
    records = [_score(stage.frame, odo.position, (0, 0), "sweep")]
    outcome = _react(records[-1])
    for cell in cells:
        while outcome in ("", "miss") and not is_aborted():
            hit: list[str] = []

            def _observe(shot, pos):
                records.append(_score(shot, pos, cell, "transit"))
                verdict = _react(records[-1])
                if verdict:
                    hit.append(verdict)
                return bool(hit) or is_aborted()

            if not stage.move_to(cell[0] * step_x, cell[1] * step_y, stop_when=_observe):
                break
            outcome = hit[0] if hit else ""
            if not hit:
                meta["cells_visited"] += 1
                break
            # 추격이 놓쳤으면(miss) 같은 셀로 가던 길을 잇는다.
        if outcome in ("found", "stop") or is_aborted():
            break
    pan_count = meta["cells_visited"]

    # ---- §4 남은 후보 추격: 점수순. ----
    if outcome in ("", "miss") and not is_aborted():
        pool = sorted((r for r in records if r["score"] >= config.candidate_score),
                      key=lambda r: -r["score"])
        print(f"[INFO] grid search: sweep 끝 - 후보 {len(pool)}/{len(records)} "
              f"(candidate_score>={config.candidate_score}), 추격 {len(chased)}/{config.max_chase} 사용")
        for rec in pool:
            if is_aborted() or len(chased) >= max(0, config.max_chase):
                break
            if _chasable(rec):
                outcome = _chase(rec)
                if outcome != "miss":
                    break
    aborted = is_aborted()
    status = "match" if outcome == "found" and not aborted else "exhausted"
    if best is None:
        top = max(records, key=lambda r: r["score"])
        best = CandidateRecord(score=top["score"], fov_xy=tuple(top["xy"]), iter_idx=0,
                               phase=top["phase"], decision=top["decision"])

    # ---- §5 복귀 (match 는 그 자리에 둔다). 실패는 restore_failed 로 남긴다(스펙 §5). ----
    if status != "match":
        if aborted:
            status = "aborted"
            print(f"[WARNING] 긴급 해제({abort_reason()}) - grid search 중단(복귀 생략).")
        else:
            # 배율을 모르면 px 단위도 모른다 - 원점 이동을 하지 않고 restore_failed 로 남긴다.
            moved = (not mag_unknown) and stage.move_to(0.0, 0.0)
            restored = True
            if back_target is not None and abs(back_target - cur_mag) > 1e-6:
                meta["restore_mag"] = mag.set_fn(back_target)
                restored = meta["restore_mag"] is not None
            meta["restore_failed"] = not (moved and restored)
            if meta["restore_failed"]:
                print("[WARNING] grid search: 원점/배율 복귀 실패(restore_failed) - 엔지니어 확인 필요")
            if notify_fn is not None:
                notify_fn(SimpleNamespace(low_streak=0, pan_count=pan_count), history)
    meta["drift_flags"] = odo.drift_flags
    meta["odometry"] = odo.log
    meta["final_position_px"] = list(odo.position)
    return _outcome(status, best, pan_count, history, meta)


def _outcome(status, best, pan_count, history, meta):
    return LiveSearchOutcome(
        status=status, final_decision=(best.decision if best else "low"),
        best=best, pan_count=pan_count, history=history, meta=meta,
    )


def search_around(
    controller,
    templates: dict,
    *,
    grid_mag: MagnificationControl | None,
    reg_mag: float | None,
    grid_config: GridSearchConfig | None = None,
    legacy_config: LiveSearchConfig = LiveSearchConfig(),
    notify_fn=None,
    debug_dir: Path | None = None,
    accept_fn=None,
) -> LiveSearchOutcome:
    """fallback 탐색의 단일 진입점: grid(배율 주입 + 등록 배율이 있을 때), 아니면 legacy.

    grid 가 ``degraded``(배율 판독 실패)로 끝나면 legacy 로 넘기고 그 outcome.meta 에
    ``degraded_from`` 을 남긴다. correction 은 이 함수 하나만 부른다.
    """
    degraded_from = None
    if grid_mag is not None and reg_mag is not None:
        print("[INFO] key 가 paused 화면에 보이지 않음 → fallback(grid_align_search) 위임")
        out = grid_align_search(controller, templates, grid_mag, reg_mag=reg_mag,
                                config=grid_config or GridSearchConfig(),
                                notify_fn=notify_fn, debug_dir=debug_dir, accept_fn=accept_fn)
        if debug_dir is not None:
            # 셀별 score/decision/scale 이 "key 위를 지나갔는데 왜 안 잡혔나" 의 유일한 근거다.
            # work2.log 를 grep 하지 않아도 되게 한 파일로 남긴다.
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "grid_search.json").write_text(json.dumps(
                    {"status": out.status, "meta": out.meta, "history": out.history},
                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            except Exception as exc:
                print(f"[WARNING] grid_search.json 저장 실패: {exc}")
        if out.status != "degraded":
            return out
        degraded_from = out.meta.get("reason")
        print(f"[WARNING] grid search degraded({degraded_from}) -> legacy live_align_search")
    elif grid_mag is not None:
        print("[WARNING] 등록 배율(cond Magnification) 없음 -> grid search 대신 legacy fallback")
    print("[INFO] key 가 paused 화면에 보이지 않음 → fallback(live_align_search) 위임")
    # 모듈 속성으로 부른다 - 테스트가 live_search.live_align_search 를 바꿔 끼울 수 있게.
    out = _live_search.live_align_search(controller, templates, config=legacy_config,
                                         notify_fn=notify_fn, debug_dir=debug_dir)
    if degraded_from is not None:
        out.meta["degraded_from"] = degraded_from
    return out
