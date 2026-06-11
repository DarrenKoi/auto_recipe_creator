"""``align_key_matcher`` 의 합성 데이터 기반 smoke test.

Mac 에서도 실행 가능. 실제 SEM 데이터 없이도 매칭 알고리즘이 정상 동작하는지,
positive/negative 점수가 깔끔히 분리되는지를 확인한다.

실행:
    uv run python poc/workflow_3/align/matching/test_engine.py

산출물:
    poc/workflow_3/debug_images/align_search/<YYMMDD_HHMMSS>/
        case_<id>_<label>_overlay.jpg
        case_<id>_<label>_frame.jpg
        case_<id>_<label>_template.jpg
        case_<id>_<label>_result.json
        summary.json

실패 시 print 로 어떤 케이스가 통과하지 못했는지 표시한다.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.matching.engine import (
    AlignKeyMatchResult,
    build_template,
    compute_align_key_score,
    save_overlay_jpeg,
)

# 재현 가능한 합성 데이터를 위해 고정 seed.
RNG = np.random.default_rng(42)

FRAME_SIZE = (512, 768)  # (height, width)
TEMPLATE_SIZE = 128


# ------------------------------------------------------------------
# 합성 fiducial 패턴 생성기.
# ------------------------------------------------------------------


def make_synthetic_template(size: int = TEMPLATE_SIZE, key_type: str = "cross") -> np.ndarray:
    """SEM 의 align fiducial 을 흉내낸 grayscale 패턴.

    배경은 중간 밝기 (~120), 패턴 자체는 어두운 (~40) 두꺼운 stroke 으로
    그린다. 가장자리 부드럽게 만들고 mild Gaussian noise 를 주어, ORB 가
    keypoint 를 잡을 수 있는 표면 텍스처를 만든다.
    """
    img = np.full((size, size), 120, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    dark = 40
    if key_type == "cross":
        thickness = max(8, size // 12)
        cv2.line(img, (cx, 8), (cx, size - 8), dark, thickness)
        cv2.line(img, (8, cy), (size - 8, cy), dark, thickness)
    elif key_type == "box":
        # 3-box-in-box: 사용자 도메인 노트 "큰 박스 3~4개" 와 동일 형태.
        for r_ratio, t in ((0.42, 4), (0.28, 3), (0.14, 3)):
            r = int(size * r_ratio)
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), dark, t)
        # 회전 대칭 깨기 위한 비대칭 orientation cluster — 실제 align key 에도 흔히 있다.
        # ORB 가 distinctive keypoint 를 다수 잡을 수 있도록, outer 와 mid 사이
        # 좌상단 quadrant 에 작은 dot 들과 코너 마크를 분산 배치.
        outer = int(size * 0.42)
        mid = int(size * 0.28)
        ax = cx - int((outer + mid) * 0.5)
        ay = cy - int((outer + mid) * 0.5)
        # 3개 점이 비대칭으로 배치된 클러스터.
        for dx, dy, r in ((0, 0, 4), (10, 0, 3), (0, 8, 3)):
            cv2.circle(img, (ax + dx, ay + dy), r, dark, -1)
        # 코너 tick (L-shape).
        cv2.line(img, (ax + 14, ay - 2), (ax + 14, ay + 6), dark, 2)
        cv2.line(img, (ax + 14, ay + 6), (ax + 6, ay + 6), dark, 2)
        # 우하단에도 하나 — 두 개 anchor 가 있어야 RANSAC 이 안정.
        bx = cx + int((outer + mid) * 0.5)
        by = cy + int((outer + mid) * 0.5)
        cv2.circle(img, (bx, by), 4, dark, -1)
        cv2.line(img, (bx - 6, by - 6), (bx + 4, by - 6), dark, 2)
    elif key_type == "checker":
        cell = size // 4
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0:
                    cv2.rectangle(
                        img,
                        (c * cell, r * cell),
                        ((c + 1) * cell, (r + 1) * cell),
                        dark,
                        -1,
                    )
    else:
        raise ValueError(f"unknown key_type: {key_type}")

    # 가장자리 약간 부드럽게 + 표면 노이즈.
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    noise = RNG.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_synthetic_cluster(
    size: int = TEMPLATE_SIZE, *, spread: float = 1.0, with_inner_cross: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    """공간적으로 분산된 3~4 박스 클러스터 + 꼭지점(align point) 좌표를 만든다.

    실제 CD-SEM align key 형태에 더 충실하다: 박스 3~4개가 서로 **떨어져** 배치되고
    (동심 nested 아님), 그 중 **좌하단(3사분면) 박스**를 align point 꼭지점으로 지정한다.
    꼭지점은 클러스터 무게중심과 의도적으로 떨어져 있어, ``align_offset``(=이미지중심−
    템플릿중심) 보정 경로를 실질적으로 검증할 수 있게 한다(꼭지점=중심인 합성은 offset
    경로를 못 건드림).

    인자:
      spread: 박스 분산 정도. 클수록 꼭지점-무게중심 거리↑ → align_offset↑.
      with_inner_cross: True 면 일부 박스 안에 작은 십자(박스 내부에 갇힌 짧은 선,
        full-span 아님)를 그린다 — detect_crosshair 가 이를 측정 십자로 오인하지 않는지
        검증하는 용도.
    반환: (image, vertex_xy). vertex_xy = 좌하단 박스 중심(패턴 좌표계) = align point.
    """
    img = np.full((size, size), 120, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    dark = 40
    # (dx_ratio, dy_ratio, half_ratio) — 캔버스 중심 기준, y 는 아래 방향.
    # [0] 좌하단 = align point 꼭지점(나머지보다 크고 nested 로 distinctive).
    boxes = [
        (-0.22, 0.26, 0.12),   # 좌하단(3사분면) = 꼭지점.
        (-0.22, -0.18, 0.09),  # 좌상단.
        (0.24, -0.18, 0.08),   # 우상단.
        (0.24, 0.10, 0.06),    # 우중(비대칭 4번째).
    ]
    centers = []
    for i, (dxr, dyr, hr) in enumerate(boxes):
        bx = int(round(cx + spread * dxr * size))
        by = int(round(cy + spread * dyr * size))
        h = max(4, int(round(hr * size)))
        cv2.rectangle(img, (bx - h, by - h), (bx + h, by + h), dark, 3 if i == 0 else 2)
        if i == 0:  # 꼭지점 박스는 이중 outline 으로 더 distinctive.
            h2 = max(2, h // 2)
            cv2.rectangle(img, (bx - h2, by - h2), (bx + h2, by + h2), dark, 2)
        centers.append((bx, by, h))
        if with_inner_cross and i in (0, 1):
            # 박스 안에만 갇힌 짧은 십자(span « frame) — 측정 crosshair 와 구분되어야.
            a = max(2, h - 3)
            cv2.line(img, (bx - a, by), (bx + a, by), dark, 1)
            cv2.line(img, (bx, by - a), (bx, by + a), dark, 1)
    # 회전 대칭 깨기용 비대칭 dot — 꼭지점 박스 옆(ORB anchor).
    vbx, vby, vh = centers[0]
    for ddx, ddy in ((vh + 6, 0), (vh + 6, 5)):
        cv2.circle(img, (vbx + ddx, vby + ddy), 2, dark, -1)

    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    noise = RNG.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, (centers[0][0], centers[0][1])


# ------------------------------------------------------------------
# 합성 SEM 프레임 생성기.
# ------------------------------------------------------------------


def make_wafer_background(
    frame_size: tuple[int, int] = FRAME_SIZE,
    *,
    base: int = 130,
) -> np.ndarray:
    """저주파 밝기 변화가 있는 feature-sparse 웨이퍼 배경."""
    h, w = frame_size
    coarse = RNG.normal(0, 30, (h // 16, w // 16))
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(base + coarse, 60, 200).astype(np.uint8)
    return img


def add_charging_gradient(image: np.ndarray, *, max_delta: int = 40) -> np.ndarray:
    """SEM 의 charging 효과 — 한쪽 방향으로 밝기가 점진 변화."""
    h, w = image.shape[:2]
    grad_x = np.linspace(-max_delta, max_delta, w, dtype=np.float32)
    grad_y = np.linspace(-max_delta * 0.4, max_delta * 0.4, h, dtype=np.float32)
    grad = grad_y[:, None] + grad_x[None, :]
    return np.clip(image.astype(np.float32) + grad, 0, 255).astype(np.uint8)


def add_random_blobs(image: np.ndarray, *, count: int = 12) -> np.ndarray:
    """ORB 가 false positive 로 잡을 수 있는 random gaussian blob 추가."""
    h, w = image.shape[:2]
    canvas = image.astype(np.float32).copy()
    for _ in range(count):
        x = int(RNG.integers(0, w))
        y = int(RNG.integers(0, h))
        sigma = float(RNG.uniform(8, 20))
        amp = float(RNG.uniform(-50, 50))
        blob_size = int(sigma * 6)
        x0 = max(0, x - blob_size)
        y0 = max(0, y - blob_size)
        x1 = min(w, x + blob_size)
        y1 = min(h, y + blob_size)
        if x1 <= x0 or y1 <= y0:
            continue
        ys = np.arange(y0, y1)[:, None] - y
        xs = np.arange(x0, x1)[None, :] - x
        g = amp * np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
        canvas[y0:y1, x0:x1] += g
    return np.clip(canvas, 0, 255).astype(np.uint8)


def embed_pattern(
    background: np.ndarray,
    pattern: np.ndarray,
    *,
    rotation_deg: float,
    scale: float,
    brightness: int,
    contrast: float,
    rng_seed: int,
) -> tuple[np.ndarray, tuple[int, int], int, int]:
    """패턴을 회전/스케일/콘트라스트 변환 후 배경에 합성.

    반환: (composite, (cx, cy), placed_w, placed_h).
    """
    rng = np.random.default_rng(rng_seed)

    h0, w0 = pattern.shape[:2]

    # affine: 회전 + 스케일.
    new_w = max(8, int(round(w0 * scale)))
    new_h = max(8, int(round(h0 * scale)))
    scaled = cv2.resize(pattern, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        scaled,
        M,
        (new_w, new_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # contrast/brightness 변환 — 평균 주변에서 contrast 적용 후 brightness 더함.
    mean = float(rotated.mean())
    transformed = (rotated.astype(np.float32) - mean) * contrast + mean + brightness
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)

    bh, bw = background.shape[:2]
    margin = 20
    if bw <= new_w + 2 * margin or bh <= new_h + 2 * margin:
        raise ValueError("background too small for pattern")

    cx = int(rng.integers(margin + new_w // 2, bw - margin - new_w // 2))
    cy = int(rng.integers(margin + new_h // 2, bh - margin - new_h // 2))
    x0 = cx - new_w // 2
    y0 = cy - new_h // 2

    canvas = background.copy()

    # 패턴이 있는 영역에서만 합성 — 배경 밝기가 보이지 않도록 그대로 덮어쓴다.
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = transformed
    return canvas, (cx, cy), new_w, new_h


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = RNG.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# 테스트 케이스 정의.
# ------------------------------------------------------------------


def build_test_cases(template_pattern: np.ndarray):
    """5 positive + 5 negative 케이스를 (id, label, frame, gt_xy_or_None) 리스트로."""
    cases = []

    # ---- Positives (template 가 frame 안에 있음) ----
    bg = make_wafer_background()
    frame, gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=0.0, scale=1.0, brightness=0, contrast=1.0, rng_seed=11,
    )
    cases.append((1, "pos_identity", add_gaussian_noise(frame, 4), gt))

    bg = make_wafer_background()
    frame, gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=1.0, scale=1.0, brightness=10, contrast=0.95, rng_seed=12,
    )
    cases.append((2, "pos_mild", add_gaussian_noise(frame, 8), gt))

    bg = make_wafer_background()
    frame, gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=0.0, scale=0.85, brightness=-5, contrast=1.0, rng_seed=13,
    )
    cases.append((3, "pos_scale_down", add_gaussian_noise(frame, 8), gt))

    bg = make_wafer_background()
    frame, gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=0.0, scale=1.2, brightness=5, contrast=1.05, rng_seed=14,
    )
    cases.append((4, "pos_scale_up", add_gaussian_noise(frame, 8), gt))

    bg = make_wafer_background()
    frame, gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=2.0, scale=0.9, brightness=-10, contrast=0.85, rng_seed=15,
    )
    frame = add_charging_gradient(frame, max_delta=25)
    cases.append((5, "pos_hard", add_gaussian_noise(frame, 12), gt))

    # ---- Negatives (template 가 frame 에 없음) ----
    cases.append((6, "neg_plain_wafer",
                  add_gaussian_noise(make_wafer_background(), 6), None))

    blobs = add_random_blobs(make_wafer_background(), count=18)
    cases.append((7, "neg_random_blobs", add_gaussian_noise(blobs, 6), None))

    other = make_synthetic_template(key_type="checker")
    bg = make_wafer_background()
    frame, _gt, _w, _h = embed_pattern(
        bg, other,
        rotation_deg=0.0, scale=1.0, brightness=0, contrast=1.0, rng_seed=18,
    )
    cases.append((8, "neg_wrong_pattern", add_gaussian_noise(frame, 8), None))

    cases.append((9, "neg_strong_charging",
                  add_gaussian_noise(
                      add_charging_gradient(make_wafer_background(), max_delta=60),
                      6,
                  ),
                  None))

    bg = make_wafer_background()
    frame, _gt, _w, _h = embed_pattern(
        bg, template_pattern,
        rotation_deg=0.0, scale=2.0, brightness=0, contrast=1.0, rng_seed=20,
    )
    cases.append((10, "neg_out_of_scale", add_gaussian_noise(frame, 8), None))

    return cases


# ------------------------------------------------------------------
# 결과 직렬화.
# ------------------------------------------------------------------


def result_to_json(result: AlignKeyMatchResult) -> dict:
    payload = asdict(result)
    payload.pop("debug_overlay", None)
    payload["best_xy"] = list(result.best_xy)
    return payload


def passes_positive(result: AlignKeyMatchResult, gt_xy: tuple[int, int]) -> tuple[bool, str]:
    if result.decision not in ("match", "adjust"):
        return False, f"decision={result.decision}"
    dx = result.best_xy[0] - gt_xy[0]
    dy = result.best_xy[1] - gt_xy[1]
    err = float(np.hypot(dx, dy))
    if err > 20.0:
        return False, f"location_error={err:.1f}px"
    return True, f"ok (err={err:.1f}px)"


def passes_negative(result: AlignKeyMatchResult) -> tuple[bool, str]:
    if result.decision == "low":
        return True, "ok (low)"
    return False, f"decision={result.decision}"


# ------------------------------------------------------------------
# 메인.
# ------------------------------------------------------------------


def main() -> int:
    print("[INFO] align_key_matcher synthetic-data smoke test 시작")

    template_pattern = make_synthetic_template(key_type="box")
    template = build_template(
        template_pattern,
        recipe_id="SYN-BOX-001",
        version="v0",
        nm_per_pixel=None,
        key_type="box",
    )
    print(f"[INFO] 템플릿 생성 완료: shape={template.raw_image.shape}, edges={int(template.edge_map.astype(bool).sum())}px")

    cases = build_test_cases(template_pattern)

    run_tag = time.strftime("%y%m%d_%H%M%S")
    out_dir = Path(DEBUG_IMAGE_DIR) / "align_search" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 출력 디렉토리: {out_dir}")

    summary = []
    n_pass = 0

    for case_id, label, frame, gt in cases:
        result = compute_align_key_score(template, frame)
        is_pos = label.startswith("pos_")

        if is_pos:
            assert gt is not None
            ok, note = passes_positive(result, gt)
        else:
            ok, note = passes_negative(result)

        status_tag = "PASS" if ok else "FAIL"
        print(
            f"[{status_tag}] case={case_id:02d} {label:>22}  "
            f"score={result.score:.3f}  chamfer={result.chamfer_score:.3f}  "
            f"orb={result.orb_inlier_ratio:.3f}  decision={result.decision:<7} {note}"
        )

        if ok:
            n_pass += 1

        # 디버그 산출물.
        cv2.imwrite(
            str(out_dir / f"case_{case_id:02d}_{label}_frame.jpg"),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        cv2.imwrite(
            str(out_dir / f"case_{case_id:02d}_{label}_template.jpg"),
            template.raw_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        save_overlay_jpeg(
            result.debug_overlay,
            out_dir / f"case_{case_id:02d}_{label}_overlay.jpg",
        )
        result_payload = result_to_json(result)
        result_payload.update(
            {
                "case_id": case_id,
                "label": label,
                "is_positive": is_pos,
                "ground_truth_xy": list(gt) if gt is not None else None,
                "passed": ok,
                "note": note,
            }
        )
        (out_dir / f"case_{case_id:02d}_{label}_result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False)
        )
        summary.append(result_payload)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "n_total": len(cases),
                "n_pass": n_pass,
                "cases": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"[INFO] Prototype validated: {n_pass}/{len(cases)} cases passed")
    print(f"[INFO] summary: {summary_path}")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
