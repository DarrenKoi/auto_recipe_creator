"""Search-around 재설계 — 절대 배율 zoom-out + FOV 단위 격자 sweep + odometry.

설계: ``docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md``.
기존 ``live_search``(휠 + 프레임 픽셀 spiral)는 착지점 근처(~1 FOV)를 못 벗어난다 —
step 이 픽셀이라 배율에 역비례로 줄고 zoom-out 은 배율을 거의 안 바꾸는 휠이기 때문.
여기서는 모든 거리를 **FOV 비율**, 모든 scale 을 **배율비**로 다룬다.

단위계: ``base = fw / source_wh[0]``로 template crop과 offset을 한 번 리샘플하면,
이후 매칭 scale은 순수 ``cur_mag / reg_mag``다. crop 폭을 원본 FOV 폭으로 쓰지 않는다.
FOV_um = 135,000 / Mag (``docs/study/hitachi_mag_fov_pixel_260828.md``).

배율 변경은 컨트롤러 Protocol 에 넣지 않고 **주입 함수**로 받는다(``MagnificationControl``).
PM 드롭다운 + OCR 판독 코드는 office-only 라 ``monitor/cycle.py`` 에 남고, 이 모듈은 Mac 에서
mock 으로 전부 검증된다. 진입점은 ``search_around`` — grid 가 배율을 못 읽으면 legacy 로 넘긴다.
"""

import math
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
    STRUCTURE_POLICY,
    AlignKeyTemplate,
    _resize_template,
    build_template,
    compute_align_key_score_ensemble,
    template_frame_scale,
)
from poc.workflow_3.align.search_pattern import square_spiral_step
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted

# 기준 화면 폭(µm). FOV_um = FOV_UM_CONSTANT / Mag. 상수는 OFFICE-VERIFY(표본 1건 0.02% 차).
FOV_UM_CONSTANT = 135_000.0


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


def plan_grid(fov_um: float, radius_um: float, budget: int) -> list[tuple[int, int]]:
    """2R 박스를 덮는 홀수 n×n 격자의 셀 오프셋(FOV 단위)을 spiral 순서로 낸다.

    n = ceil(2R / FOV) 를 홀수로 올려 착지 셀이 중심에 오게 한다. 예산을 넘으면 안쪽
    링부터 예산만큼만(바깥 링 일부 생략). FOV 가 이미 2R 을 덮으면 빈 목록.
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
    pan_budget: int = 10           # sweep 셀 수 상한(= 1 FOV step 수).
    click_margin_ratio: float = 0.12  # recenter 클릭의 FOV 안쪽 여백 -> 1 클릭 최대 0.38 FOV.
    odom_tol_fov: float = 0.15     # |측정 - 명령| 허용(FOV 비율). 넘으면 명령값 폴백 + flag.
    candidate_score: float = STRUCTURE_POLICY.adjust_threshold  # 추격 대상 최소 점수.
    max_chase: int = 3             # 추격할 후보 수 상한(점수순). 추격마다 배율 왕복이 들어간다.


@dataclass(frozen=True)
class MagnificationControl:
    """배율 주입점 - PM 드롭다운 옵션 읽기와 절대 배율 선택기.

    ``options_fn() -> [배율, ...]`` 은 실장비에서 **드롭다운을 여는 일**이라 선택 직전에 한 번만
    불린다(연 김에 바로 행을 눌러야 한다). ``set_fn(target) -> 판독 배율 | None`` 의 판독은 PM
    box OCR 이며 None 이면 '모름' 이다 - 명령값을 믿지 않는다(계약 1). Mac 은 list/lambda.
    """

    options_fn: Callable[[], list]
    set_fn: Callable[[float], float | None]


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

    def move_px(self, dx, dy) -> bool:
        """(dx,dy) px 만큼 stage 를 옮긴다 - 한 클릭 최대 0.38 FOV 로 쪼갠다. abort 면 False."""
        n = max(1, math.ceil(abs(dx) / self.max_click_x), math.ceil(abs(dy) / self.max_click_y))
        for _ in range(n):
            if is_aborted():
                return False
            self._click(dx / n, dy / n)
        return True

    def move_to(self, tx, ty) -> bool:
        px, py = self.odo.position
        return self.move_px(tx - px, ty - py)


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
) -> LiveSearchOutcome:
    """절대 배율 zoom-out -> 격자 sweep(collect) -> best-first 추격/confirm -> 복귀.

    status: "match" | "exhausted" | "aborted" | "degraded"(배율 판독 실패 - 호출부가 legacy
    경로로 넘긴다). meta 에 search_mag/cells_visited/odometry/final_position_px/restore_failed
    를 남긴다. ``notify_fn(state, history)`` 는 legacy 와 같은 escalation 콜백 - 못 찾고
    끝날 때 한 번 부른다(cycle 의 live_search_escalation 감사 로그가 grid 경로에서도 남게).
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
    source_template = route_template(templates, mode)
    if source_template.source_wh is None:
        meta["reason"] = "missing_source_geometry"
        return _outcome("degraded", None, 0, history, meta)
    try:
        base = template_frame_scale(source_template, frame.shape)
        template = normalize_template(source_template, fw)
    except ValueError as exc:
        meta["reason"] = "invalid_source_geometry"
        print(f"[WARNING] grid source geometry: {exc}")
        return _outcome("degraded", None, 0, history, meta)
    meta.update(source_wh=list(source_template.source_wh), frame_wh=[fw, fh], base_scale=base)
    odo = Odometer(fov_px=fw, tol_fov=config.odom_tol_fov)
    stage = _Stage(controller, fw, fh, config, shift_fn, odo)
    stage.frame = frame

    # ---- §1 zoom-out (SEM 만). ----
    cur_mag = reg_mag
    back_target = None  # SEM 배율 선택 시: 등록 배율 최근접 단(confirm/복귀용).
    if "OM" in mode:
        cells = spiral_cells(config.pan_budget)
    else:
        options = mag.options_fn()
        if not options:
            # OCR 이 행을 하나도 못 읽었다. 내릴 단도, 열린 드롭다운을 닫을 행도 없다 -
            # selector 가 닫기를 시도했고, 여기서는 모르는 상태로 격자를 돌리지 않는다.
            meta["reason"] = "no_mag_options"
            print("[WARNING] grid search: PM 드롭다운 옵션 0개 -> legacy 경로로 degrade")
            return _outcome("degraded", None, 0, history, meta)
        target = choose_zoom_out_mag(options, reg_mag, min(template.raw_image.shape[:2]), config.min_key_px)
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
        # 셀 step 은 x=fw, y=fh (FOV 는 폭 기준이라 세로는 fh/fw 배). n 은 짧은 변으로 잡아야
        # 세로 커버에 구멍이 안 난다.
        cells = plan_grid(fov_um=fov_um(cur_mag) * min(1.0, fh / fw),
                          radius_um=config.radius_um, budget=config.pan_budget)
    meta["search_mag"] = cur_mag
    scale = cur_mag / reg_mag
    print(f"[INFO] grid search: reg={reg_mag:.0f} search={cur_mag:.0f} scale={scale:.3f} "
          f"fw={fw} cells={len(cells)}")

    def _score(cell):
        r = match(template, stage.frame, scales=(scale,))
        ox, oy = template.align_offset_xy
        align_xy = [r.best_xy[0] + round(ox * r.best_scale), r.best_xy[1] + round(oy * r.best_scale)]
        rec = {"cell": list(cell), "score": float(r.score), "xy": align_xy,
               "match_xy": list(r.best_xy), "scale": float(r.best_scale),
               "decision": r.decision, "orb": float(r.orb_inlier_ratio)}
        history.append(rec)
        return rec

    # ---- §2 sweep: collect only. ----
    records = [_score((0, 0))]
    aborted = False
    for cell in cells:
        if not stage.move_to(cell[0] * fw, cell[1] * fh):
            aborted = True
            break
        meta["cells_visited"] += 1
        records.append(_score(cell))
    pan_count = meta["cells_visited"]

    best_rec = max(records, key=lambda r: r["score"])
    best = CandidateRecord(score=best_rec["score"], fov_xy=tuple(best_rec["xy"]), iter_idx=0,
                           phase="sweep", decision=best_rec["decision"])

    # ---- §4 chase: best-first, confirm at registered-nearest mag. ----
    status = "exhausted"
    if not aborted:
        chase = sorted(
            (r for r in records if r["score"] >= config.candidate_score and r["decision"] != "low"),
            key=lambda r: -r["score"],
        )[: max(0, config.max_chase)]
        for rec in chase:
            cx, cy = rec["cell"]
            if not stage.move_to(cx * fw, cy * fh):
                aborted = True
                break
            # 후보 점으로 recenter (odometry 포함).
            if not stage.move_px(rec["xy"][0] - fw / 2, rec["xy"][1] - fh / 2):
                aborted = True
                break
            back = mag.set_fn(back_target) if back_target is not None else None
            if back_target is not None and back is None:
                # 배율을 바꾸려 했는데 판독이 없다 = 장비가 어느 배율인지 모른다. 모르는
                # scale 로 confirm 하지 않는다(계약 1) - 추격을 멈추고 복귀로 간다.
                meta["reason"] = "mag_unreadable_confirm"
                print("[WARNING] grid search: confirm 배율 판독 실패 -> 추격 중단, 복귀")
                break
            back_mag = float(back) if back is not None else cur_mag
            frame_c = stage.capture()
            s2 = back_mag / reg_mag
            r = match(template, frame_c, scales=(s2,))
            ox, oy = template.align_offset_xy
            align_xy = (int(r.best_xy[0] + round(ox * r.best_scale)),
                        int(r.best_xy[1] + round(oy * r.best_scale)))
            history.append({"cell": rec["cell"], "phase": "confirm", "score": float(r.score),
                            "xy": list(align_xy), "match_xy": list(r.best_xy), "decision": r.decision,
                            "orb": float(r.orb_inlier_ratio), "scale": s2,
                            "distinctive": bool(r.distinctive), "second_ratio": r.second_ratio})
            # confirm 게이트: 단일 known scale(>= 0.6) 의 ensemble match. legacy 의 orb>0 는 쓰지
            # 않는다 - SEM junction key 는 ORB 특징점이 빈약해(aperture 문제) 진짜 match 도
            # orb=0 으로 나온다. distinctive 는 engine 규약대로 soft advisory 로만 기록한다
            # (chamfer-top 의 유일성이지 best_xy 의 유일성이 아니다 - hard gate 금지).
            if r.decision == "match" and s2 >= MIN_CONFIRM_SCALE:
                best = CandidateRecord(score=float(r.score), fov_xy=align_xy,
                                       iter_idx=len(history), phase="confirm", decision="match")
                status = "match"
                meta["restore_mag"] = back_mag
                meta["confirm_distinctive"] = bool(r.distinctive)
                meta["confirm_second_ratio"] = r.second_ratio
                break
            # 놓침 -> 탐색 배율로 돌아가 다음 후보.
            if abs(back_mag - cur_mag) > 1e-6:
                mag.set_fn(cur_mag)
                stage.frame = controller.capture()

    # ---- §5 복귀 (match 는 그 자리에 둔다). 실패는 restore_failed 로 남긴다(스펙 §5). ----
    if status != "match":
        if aborted:
            status = "aborted"
            print(f"[WARNING] 긴급 해제({abort_reason()}) - grid search 중단(복귀 생략).")
        else:
            moved = stage.move_to(0.0, 0.0)
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
                                notify_fn=notify_fn, debug_dir=debug_dir)
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
