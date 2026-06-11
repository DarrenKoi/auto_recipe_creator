"""Tool 창 캡처 프레임 안에서 SEM monitor 패널 영역을 찾아주는 detector.

`poc/workflow_1/capture_window_frames_tool.py` 가 저장하는 프레임은
Remote Monitoring System 창 *전체* 다. SEM monitor 자체는 그 안의 일부
영역이고, tool model 마다 layout 이 다르다. 따라서 *model 별 landmark
crop* 들을 미리 받아두고, 매 프레임마다 cv2.matchTemplate (TM_CCOEFF_NORMED)
로 가장 자신 있는 landmark 를 골라 panel ROI 를 도출한다.

레이아웃:

```
poc/workflow_3/templates/sem_panel_landmarks/
└── <model_id>/
    ├── landmark.jpg     # UI 일부 (panel title bar, corner icon 등)
    └── meta.json        # {"panel_offset": [dx, dy, w, h], "nm_per_pixel": <float|null>}
```

(landmark crop 은 오피스에서 tool model 별로 캘리브레이션해 채운다.
legacy 샘플은 poc/workflow_2/templates/sem_panel_landmarks/ 에 남아 있을 수 있다.)

`panel_offset` 은 landmark 의 *top-left* 기준 상대 좌표/크기.
실제 frame 위 절대 좌표는 ``landmark_xy + (dx, dy)`` 다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------
# 상수.
# ------------------------------------------------------------------

# TM_CCOEFF_NORMED 결과는 -1..1. SEM panel 처럼 chrome 이 잘 보이는 UI 에서는
# 진짜 일치 시 0.85 이상이 나오지만, 보수적으로 0.70 floor 부터 신뢰.
LANDMARK_CONF_MIN = 0.70


# ------------------------------------------------------------------
# 데이터 클래스.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SEMPanelLandmark:
    """디스크에서 읽은 model 별 UI landmark + panel offset."""

    model_id: str
    landmark_gray: np.ndarray  # grayscale uint8.
    panel_offset: tuple[int, int, int, int]  # (dx, dy, w, h) — landmark top-left 기준.
    nm_per_pixel: float | None


@dataclass(frozen=True)
class SEMPanelMatch:
    """매 프레임마다 ``locate_panel`` 이 반환하는 결과."""

    model_id: str
    panel_roi: tuple[int, int, int, int]  # (x, y, w, h) — 프레임 절대 좌표.
    landmark_xy: tuple[int, int]  # landmark top-left 절대 좌표.
    confidence: float
    nm_per_pixel: float | None


# ------------------------------------------------------------------
# 내부 헬퍼.
# ------------------------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """다양한 dtype/channel 의 입력을 grayscale uint8 로 정규화."""
    if image is None:
        raise ValueError("image is None")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def _parse_panel_offset(raw: object, *, source: Path) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(
            f"{source}: panel_offset must be a 4-element list [dx, dy, w, h], "
            f"got {raw!r}"
        )
    dx, dy, w, h = (int(v) for v in raw)
    if w <= 0 or h <= 0:
        raise ValueError(
            f"{source}: panel_offset width/height must be positive, got w={w}, h={h}"
        )
    return dx, dy, w, h


# ------------------------------------------------------------------
# 로딩.
# ------------------------------------------------------------------


def load_landmarks(landmarks_dir: Path) -> list[SEMPanelLandmark]:
    """``landmarks_dir`` 아래의 모든 model 서브폴더를 읽어 landmark 목록을 만든다.

    각 서브폴더는 ``landmark.jpg`` 와 ``meta.json`` 을 가져야 한다.
    누락된 항목은 ``[WARNING]`` 으로 표시하고 skip 한다.
    """
    if not landmarks_dir.exists():
        print(f"[WARNING] landmark 디렉터리가 없습니다: {landmarks_dir}")
        return []

    landmarks: list[SEMPanelLandmark] = []
    for model_dir in sorted(p for p in landmarks_dir.iterdir() if p.is_dir()):
        landmark_path = model_dir / "landmark.jpg"
        meta_path = model_dir / "meta.json"
        if not landmark_path.exists():
            print(f"[WARNING] landmark.jpg 누락: {model_dir}")
            continue
        if not meta_path.exists():
            print(f"[WARNING] meta.json 누락: {model_dir}")
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARNING] meta.json 파싱 실패: {meta_path} ({exc})")
            continue

        if "panel_offset" not in meta:
            raise ValueError(f"{meta_path}: panel_offset 필드가 필요합니다")

        panel_offset = _parse_panel_offset(meta["panel_offset"], source=meta_path)
        nm_per_pixel = meta.get("nm_per_pixel")
        if nm_per_pixel is not None:
            nm_per_pixel = float(nm_per_pixel)

        image = cv2.imread(str(landmark_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARNING] landmark 이미지를 읽지 못했습니다: {landmark_path}")
            continue

        landmarks.append(
            SEMPanelLandmark(
                model_id=model_dir.name,
                landmark_gray=image,
                panel_offset=panel_offset,
                nm_per_pixel=nm_per_pixel,
            )
        )
        print(
            f"[INFO] landmark 로드: model={model_dir.name} "
            f"size={image.shape[1]}x{image.shape[0]} "
            f"panel_offset={panel_offset} nm_per_pixel={nm_per_pixel}"
        )

    if not landmarks:
        print(f"[WARNING] landmark 가 한 개도 로드되지 않았습니다: {landmarks_dir}")
    return landmarks


# ------------------------------------------------------------------
# 메인 detector.
# ------------------------------------------------------------------


def locate_panel(
    frame: np.ndarray,
    landmarks: list[SEMPanelLandmark],
    *,
    min_confidence: float = LANDMARK_CONF_MIN,
) -> SEMPanelMatch | None:
    """주어진 프레임에서 가장 자신 있는 landmark 를 골라 SEM panel ROI 를 반환.

    매 landmark 마다 cv2.matchTemplate(frame, landmark, TM_CCOEFF_NORMED) 를
    실행하여 최대 점수와 위치를 구한 뒤, 모든 후보 중 최고점이
    ``min_confidence`` 이상이면 그 landmark 로 SEM panel 좌표를 만든다.

    panel ROI 는 ``(landmark_x + dx, landmark_y + dy, w, h)`` 이고,
    frame 경계를 벗어나지 않도록 clamp 한다.
    """
    if not landmarks:
        return None

    frame_gray = _to_grayscale(frame)
    fh, fw = frame_gray.shape[:2]

    best: tuple[float, SEMPanelLandmark, tuple[int, int]] | None = None

    for lm in landmarks:
        lh, lw = lm.landmark_gray.shape[:2]
        if lh >= fh or lw >= fw:
            print(
                f"[WARNING] landmark 가 프레임보다 큼: model={lm.model_id} "
                f"landmark={lw}x{lh} frame={fw}x{fh}"
            )
            continue
        result = cv2.matchTemplate(frame_gray, lm.landmark_gray, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (float(max_val), lm, (int(max_loc[0]), int(max_loc[1])))

    if best is None:
        return None

    confidence, lm, landmark_xy = best
    if confidence < min_confidence:
        print(
            f"[WARNING] landmark 신뢰도 미달: best_model={lm.model_id} "
            f"confidence={confidence:.3f} < min={min_confidence:.3f}"
        )
        return None

    dx, dy, pw, ph = lm.panel_offset
    px = max(0, min(fw, landmark_xy[0] + dx))
    py = max(0, min(fh, landmark_xy[1] + dy))
    px2 = max(0, min(fw, landmark_xy[0] + dx + pw))
    py2 = max(0, min(fh, landmark_xy[1] + dy + ph))
    clamped_w = max(0, px2 - px)
    clamped_h = max(0, py2 - py)

    if clamped_w <= 0 or clamped_h <= 0:
        print(
            f"[WARNING] panel ROI 가 frame 밖으로 완전히 벗어남: "
            f"model={lm.model_id} landmark_xy={landmark_xy} offset={lm.panel_offset} "
            f"frame={fw}x{fh}"
        )
        return None

    return SEMPanelMatch(
        model_id=lm.model_id,
        panel_roi=(px, py, clamped_w, clamped_h),
        landmark_xy=landmark_xy,
        confidence=confidence,
        nm_per_pixel=lm.nm_per_pixel,
    )
