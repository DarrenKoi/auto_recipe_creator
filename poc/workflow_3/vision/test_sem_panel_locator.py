"""``sem_panel_locator`` 의 합성 데이터 기반 self-test.

Mac 에서도 실행 가능. 실제 Tool 창 스크린샷(보안상 반출 금지)이 없어도,
``locate_panel`` 의 핵심 분기가 의도대로 도는지를 코드 레벨에서 박아둔다:

  1. 멀티-landmark argmax  — 여러 model landmark 중 가장 confident 한 것 선택
  2. offset 산술          — panel_roi = landmark_xy + (dx, dy, w, h)
  3. frame 경계 clamp     — offset 이 frame 밖으로 나가면 잘라낸다
  4. confidence floor 게이트 — landmark 가 없으면 None (false ROI 방지)

설계 의도(중요):
  합성 프레임의 **Live SEM 박스 내부는 매 케이스 다른 random 텍스처**로 채우고,
  landmark 는 **박스 바깥**에 둔다. 이는 실제 운영 전제 — "라이브 SEM 영상은
  매 프레임 변하지만, 박스 바깥 chrome landmark 는 고정이라 매칭이 유지된다" —
  를 그대로 모사한다. (landmark 를 박스 안에 두면 매칭이 깨진다는 것도 동시에
  보여준다.)

주의: 본 테스트가 PASS 한다고 "Live SEM 이 landmark 에서 고정 offset 이다"가
증명되는 건 아니다. 그 **가설** 은 office 실데이터(`test_match_on_captured_frames.py`)
로만 검증된다. 여기서 검증하는 건 `locate_panel` 의 **코드/산술** 뿐이다.

실행:
    uv run python poc/workflow_2/test_sem_panel_locator.py

산출물:
    poc/workflow_2/debug_images/sem_panel_locator/<YYMMDD_HHMMSS>/
        case_<id>_<label>_overlay.jpg   # 반환된 panel_roi 를 그린 프레임
        summary.json
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.vision.sem_panel_locator import (
    LANDMARK_CONF_MIN,
    load_landmarks,
    locate_panel,
)

# 재현 가능한 합성 데이터를 위해 고정 seed.
RNG = np.random.default_rng(7)

FRAME_SIZE = (900, 1400)  # (height, width) — 바깥 배치 Tool 창을 흉내.
# 위치 오차 허용치: matchTemplate max_loc 은 noise 에서 ±1~2px 흔들릴 수 있다.
LOC_TOL_PX = 3


# ------------------------------------------------------------------
# 합성 landmark / Tool 창 생성기.
# ------------------------------------------------------------------


def make_landmark(*, seed: int, size: tuple[int, int] = (70, 150)) -> np.ndarray:
    """고대비·텍스처 풍부한 grayscale landmark patch.

    실제 landmark(패널 타이틀바·코너 아이콘)처럼 TM_CCOEFF_NORMED 가
    뚜렷한 peak 를 잡을 수 있도록, 밝은 배경 위에 어두운 글자 같은 tick
    과 코너 마크를 model 별 seed 로 다르게 분포시킨다.
    """
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 230, dtype=np.uint8)  # 밝은 패널 chrome.
    dark = 30
    # 타이틀바 느낌의 가로 줄.
    cv2.rectangle(img, (2, 2), (w - 3, 12), dark, 1)
    # 글자를 흉내낸 세로 tick 들 (seed 로 위치/개수 변동 → model 별 구분).
    n_ticks = int(rng.integers(8, 14))
    for _ in range(n_ticks):
        x = int(rng.integers(6, w - 6))
        y0 = int(rng.integers(18, h - 18))
        cv2.line(img, (x, y0), (x, y0 + int(rng.integers(6, 16))), dark, 2)
    # 코너 마크 (L-shape) — distinctive corner.
    cv2.line(img, (4, h - 14), (4, h - 4), dark, 2)
    cv2.line(img, (4, h - 4), (16, h - 4), dark, 2)
    return img


def make_tool_window(
    *,
    landmark: np.ndarray,
    landmark_xy: tuple[int, int],
    sem_box: tuple[int, int, int, int],
    live_seed: int,
    noise_sigma: float = 0.0,
    place_landmark: bool = True,
) -> np.ndarray:
    """바깥 배치 Tool 창을 합성한다.

    배경은 균일한 중간 밝기(landmark/SEM 과 헷갈리지 않도록 저대비),
    SEM 박스 내부는 ``live_seed`` 로 만든 random 텍스처(=매 프레임 변하는
    라이브 영상 모사)로 채운다. landmark 는 박스 바깥에 붙인다.
    """
    h, w = FRAME_SIZE
    frame = np.full((h, w), 100, dtype=np.uint8)

    # --- Live SEM 박스: 매번 다른 random 텍스처 (라이브 영상 모사) ---
    sx, sy, sw, sh = sem_box
    live = np.random.default_rng(live_seed).integers(40, 210, size=(sh, sw), dtype=np.uint8)
    live = cv2.GaussianBlur(live, (0, 0), 1.2)
    frame[sy : sy + sh, sx : sx + sw] = live
    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), 20, 2)  # 박스 테두리.

    # --- landmark 를 박스 바깥(landmark_xy)에 붙인다 ---
    if place_landmark:
        lx, ly = landmark_xy
        lh, lw = landmark.shape[:2]
        frame[ly : ly + lh, lx : lx + lw] = landmark

    if noise_sigma > 0:
        noise = RNG.normal(0, noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame


def write_landmark_template(
    root: Path,
    *,
    model_id: str,
    landmark: np.ndarray,
    panel_offset: tuple[int, int, int, int],
    nm_per_pixel: float | None = None,
) -> None:
    """templates/sem_panel_landmarks/<model_id>/ 레이아웃으로 임시 템플릿 기록.

    load_landmarks 는 ``landmark.jpg`` 를 hardcode 로 읽으므로 동일 파일명 사용.
    (실제 landmark.jpg 도 JPEG 이므로, JPEG round-trip 까지 함께 검증된다.)
    """
    model_dir = root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(model_dir / "landmark.jpg"), landmark, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    meta = {"panel_offset": list(panel_offset), "nm_per_pixel": nm_per_pixel}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def offset_from(landmark_xy: tuple[int, int], sem_box: tuple[int, int, int, int]):
    """landmark top-left → SEM 박스 까지의 panel_offset (dx, dy, w, h)."""
    lx, ly = landmark_xy
    sx, sy, sw, sh = sem_box
    return (sx - lx, sy - ly, sw, sh)


# ------------------------------------------------------------------
# 케이스별 판정.
# ------------------------------------------------------------------


def roi_matches(roi, expected, *, tol: int = LOC_TOL_PX) -> tuple[bool, str]:
    """반환 ROI 가 기대 SEM 박스와 tol 픽셀 이내로 일치하는지."""
    if roi is None:
        return False, "panel_roi=None"
    ex, ey, ew, eh = expected
    dx = abs(roi[0] - ex)
    dy = abs(roi[1] - ey)
    dw = abs(roi[2] - ew)
    dh = abs(roi[3] - eh)
    if max(dx, dy, dw, dh) > tol:
        return False, f"roi={roi} expected={expected} (Δ=({dx},{dy},{dw},{dh}))"
    return True, f"ok roi={roi}"


# ------------------------------------------------------------------
# 메인.
# ------------------------------------------------------------------


def main() -> int:
    print("[INFO] sem_panel_locator synthetic self-test 시작")

    tmp_root = Path(tempfile.mkdtemp(prefix="sem_landmarks_"))
    run_tag = time.strftime("%y%m%d_%H%M%S")
    out_dir = Path(DEBUG_IMAGE_DIR) / "sem_panel_locator" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 임시 landmark 루트: {tmp_root}")
    print(f"[INFO] 출력 디렉토리: {out_dir}")

    results: list[dict] = []
    n_pass = 0

    try:
        # === 두 model landmark 등록 (멀티-landmark argmax 검증용) ===
        lm_a = make_landmark(seed=101)
        lm_b = make_landmark(seed=202)
        # model A: landmark 좌상단, SEM 박스는 그 오른쪽.
        a_xy = (60, 60)
        a_box = (340, 80, 760, 740)
        # model B: landmark 우상단, SEM 박스는 그 왼쪽 아래.
        b_xy = (1240, 50)
        b_box = (300, 120, 800, 680)
        write_landmark_template(tmp_root, model_id="MODEL_A",
                                landmark=lm_a, panel_offset=offset_from(a_xy, a_box))
        write_landmark_template(tmp_root, model_id="MODEL_B",
                                landmark=lm_b, panel_offset=offset_from(b_xy, b_box))

        landmarks = load_landmarks(tmp_root)
        if len(landmarks) != 2:
            print(f"[FAIL] load_landmarks 가 2개를 못 읽음: {len(landmarks)}")
            return 1

        cases = []

        # 1) identity — model A landmark 만 있는 깨끗한 프레임. ROI 정확 일치.
        f1 = make_tool_window(landmark=lm_a, landmark_xy=a_xy, sem_box=a_box, live_seed=1)
        cases.append(("identity_A", f1, "MODEL_A", a_box, True))

        # 2) live-texture + noise — SEM 내부 텍스처가 1)과 완전히 다르고 noise 추가.
        #    landmark 는 박스 바깥이라 그대로 매칭되어야 한다(운영 전제 모사).
        f2 = make_tool_window(landmark=lm_a, landmark_xy=a_xy, sem_box=a_box,
                              live_seed=999, noise_sigma=7.0)
        cases.append(("live_changed_noise_A", f2, "MODEL_A", a_box, True))

        # 3) 멀티-landmark argmax — frame 에 model B landmark 만 존재.
        f3 = make_tool_window(landmark=lm_b, landmark_xy=b_xy, sem_box=b_box, live_seed=5)
        cases.append(("select_B_among_two", f3, "MODEL_B", b_box, True))

        # 4) frame 경계 clamp — landmark 를 우하단 가까이 두어 ROI 가 frame 밖으로
        #    나가게 만든다. clamp 로 잘린 ROI 가 frame 안에 들어와야 한다.
        edge_xy = (FRAME_SIZE[1] - lm_a.shape[1] - 5, FRAME_SIZE[0] - lm_a.shape[0] - 5)
        edge_box = (edge_xy[0] + 40, edge_xy[1] + 40, 600, 500)  # 일부러 밖으로 삐져나감.
        # 임시 model 추가 (큰 offset).
        write_landmark_template(tmp_root, model_id="MODEL_EDGE",
                                landmark=make_landmark(seed=303),
                                panel_offset=offset_from(edge_xy, edge_box))
        landmarks_edge = load_landmarks(tmp_root)
        lm_edge = make_landmark(seed=303)
        f4 = make_tool_window(landmark=lm_edge, landmark_xy=edge_xy,
                             sem_box=(edge_xy[0], edge_xy[1], 40, 40), live_seed=8)
        m4 = locate_panel(f4, landmarks_edge)
        clamp_ok = (
            m4 is not None
            and m4.panel_roi[0] >= 0 and m4.panel_roi[1] >= 0
            and m4.panel_roi[0] + m4.panel_roi[2] <= FRAME_SIZE[1]
            and m4.panel_roi[1] + m4.panel_roi[3] <= FRAME_SIZE[0]
            and m4.panel_roi[2] > 0 and m4.panel_roi[3] > 0
        )
        note4 = f"roi={m4.panel_roi if m4 else None} (frame={FRAME_SIZE[1]}x{FRAME_SIZE[0]})"
        print(f"[{'PASS' if clamp_ok else 'FAIL'}] case=04 {'clamp_in_bounds':>22}  {note4}")
        n_pass += int(clamp_ok)
        results.append({"case": "clamp_in_bounds", "passed": clamp_ok, "note": note4,
                        "roi": list(m4.panel_roi) if m4 else None})

        # 5) confidence floor — landmark 가 아예 없는 프레임 → None 이어야 한다.
        f5 = make_tool_window(landmark=lm_a, landmark_xy=a_xy, sem_box=a_box,
                              live_seed=3, place_landmark=False)
        # MODEL_EDGE 까지 들어간 landmarks_edge 로 검사(가장 빡센 조건).
        m5 = locate_panel(f5, landmarks_edge)
        neg_ok = m5 is None
        print(f"[{'PASS' if neg_ok else 'FAIL'}] case=05 {'no_landmark_returns_None':>22}  "
              f"got={m5.confidence if m5 else None} (floor={LANDMARK_CONF_MIN})")
        n_pass += int(neg_ok)
        results.append({"case": "no_landmark_returns_None", "passed": neg_ok,
                        "confidence": (m5.confidence if m5 else None)})

        # === 1~3 케이스 실행 (positive: model 선택 + ROI 정확도 + confidence) ===
        for idx, (label, frame, exp_model, exp_box, _expect_match) in enumerate(cases, start=1):
            m = locate_panel(frame, landmarks)
            model_ok = (m is not None and m.model_id == exp_model)
            roi_ok, roi_note = roi_matches(m.panel_roi if m else None, exp_box)
            conf_ok = (m is not None and m.confidence >= LANDMARK_CONF_MIN)
            ok = model_ok and roi_ok and conf_ok
            n_pass += int(ok)

            conf = f"{m.confidence:.3f}" if m else "-"
            got_model = m.model_id if m else "None"
            print(f"[{'PASS' if ok else 'FAIL'}] case={idx:02d} {label:>22}  "
                  f"model={got_model}(exp {exp_model})  conf={conf}  {roi_note}")

            # overlay 산출물.
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if m is not None:
                x, y, ww, hh = m.panel_roi
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), (255, 0, 255), 3)
                lx, ly = m.landmark_xy
                cv2.circle(vis, (lx, ly), 6, (0, 255, 0), -1)
            cv2.imwrite(str(out_dir / f"case_{idx:02d}_{label}_overlay.jpg"), vis,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            results.append({"case": label, "passed": ok, "model": got_model,
                            "expected_model": exp_model, "confidence": (m.confidence if m else None),
                            "roi": list(m.panel_roi) if m else None, "expected_roi": list(exp_box),
                            "note": roi_note})

        n_total = 5
        summary = {"run_tag": run_tag, "n_total": n_total, "n_pass": n_pass, "cases": results}
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[INFO] self-test 완료: {n_pass}/{n_total} cases passed")
        print(f"[INFO] summary: {out_dir / 'summary.json'}")
        return 0 if n_pass == n_total else 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
