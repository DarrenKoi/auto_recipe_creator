"""RCS tool 창 기반 실장비 SEMMonitorController adapter (골격, 캘리브레이션 전).

`vision.live_align_search.SEMMonitorController` Protocol 의 실장비 구현.
Mac mock(`_MockSEMMonitor`)과 동일 시그니처라 `correct_align_fail` /
`live_align_search` 가 그대로 돈다.

좌표공간 계약(Protocol docstring 과 동일):
  * capture()        — SEM panel ROI 만 잘라 grayscale 로 반환 (FOV-local 좌표계 원점)
  * move_to_point()  — FOV-local 픽셀을 받아 화면 절대 좌표로 변환해 **더블클릭** recenter
  * capture_screen() — tool 창 전체 프레임(grayscale). 이 프레임의 픽셀 좌표가 곧
                       click_screen 이 받는 좌표다 (창 이미지 좌표 = "screen" 계약)
  * click_screen()   — capture_screen 프레임 좌표를 화면 절대 좌표로 변환해 single click
  * zoom()           — panel 중심에서 wheel 1단계 (FOV-centered zoom)
  * read_mode()      — v0: env ALIGN_SEM_MODE_OVERRIDE > 생성자 mode_default.
                       OCR(모드 라벨 crop + paddleocr)/픽셀 휴리스틱(OM 밝기 분포)은
                       오피스 캘리브레이션 단계의 후속 작업.

미캘리브레이션 항목(오피스 검증 전):
  * wheel 1단계 ↔ 배율 비율 (zoom_scroll_dy)
  * 더블클릭 recenter 의 실제 이동량/settle 시간
  * panel ROI landmark (templates/sem_panel_landmarks/<model_id>/)

모든 actuation 은 util.mouse_utils 의 action_enabled dry-run 게이트를 그대로
통과하므로, SAFE_MODE/dry-run 에서는 좌표 로그만 남고 실제 마우스는 움직이지
않는다.
"""

import os
import time

import cv2
import numpy as np

from poc.workflow_3 import TEMPLATES_DIR
# util/__init__ 는 pynput/pywinauto 부재 시 None 을 바인딩한다(import-안전).
# 실제 호출은 오피스(Windows+의존성 설치) 환경에서만 일어난다.
from poc.workflow_3.util import (
    capture_window,
    click_at_screen,
    foreground_window,
    image_point_to_screen,
    scroll_at_screen,
    window_rect_size,
)
from poc.workflow_3.vision.sem_panel_locator import (
    SEMPanelMatch,
    load_landmarks,
    locate_panel,
)

LOG_COMPONENT = "rcs_sem_controller"

DEFAULT_LANDMARKS_DIR = TEMPLATES_DIR / "sem_panel_landmarks"

# 캡처~제스처 사이 창 크기 드리프트 허용 오차(논리 px). 초과 시 좌표 무효로 중단.
RECT_DRIFT_TOL_PX = 2


def _to_gray(image) -> np.ndarray:
    """PIL Image / numpy 입력을 grayscale uint8 numpy 로 정규화한다."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        # PIL 캡처는 RGB(또는 RGBA) — BGR 가 아니라 RGB2GRAY 를 쓴다.
        code = cv2.COLOR_RGBA2GRAY if array.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array, code)
    raise ValueError(f"지원하지 않는 이미지 shape: {array.shape}")


class RCSSEMMonitor:
    """RCS tool 창 위에서 동작하는 실장비 SEMMonitorController 구현(골격)."""

    def __init__(
        self,
        tool_window,
        panel: SEMPanelMatch,
        *,
        action_enabled: bool = False,
        settle_sec: float = 0.5,
        zoom_scroll_dy: int = 1,
        mode_default: str = "SEM",
    ):
        self.tool_window = tool_window
        self.panel = panel
        self.action_enabled = action_enabled
        self.settle_sec = settle_sec
        self.zoom_scroll_dy = zoom_scroll_dy
        self.mode_default = mode_default
        # image_point_to_screen 의 DPI 보정에 쓰는 캡처 프레임 크기 (w, h).
        self._last_frame_size: tuple[int, int] | None = None
        # 캡처 시점의 창 rect 크기(논리 px) — 제스처 직전 리사이즈 드리프트 감지용.
        self._last_rect_size: tuple[int, int] | None = None
        self._mode_warned = False
        if action_enabled:
            print(
                "[WARNING] RCSSEMMonitor: 좌표/배율 캘리브레이션 미완료 상태에서 "
                "action_enabled=True, 단일 장비 pilot 외 사용 비권장(dry-run 권장)."
            )

    # ---- 캡처 ----

    def _capture_full_gray(self) -> np.ndarray:
        """tool 창 전체를 캡처해 grayscale 로 반환하고 프레임 크기를 캐시한다."""
        image = capture_window(self.tool_window)
        gray = _to_gray(image)
        h, w = gray.shape[:2]
        self._last_frame_size = (w, h)
        if callable(window_rect_size):
            self._last_rect_size = window_rect_size(self.tool_window)
        return gray

    def capture(self) -> np.ndarray:
        """현재 FOV(SEM panel ROI) 를 grayscale 로 반환한다."""
        frame = self._capture_full_gray()
        x, y, w, h = self.panel.panel_roi
        fh, fw = frame.shape[:2]
        x2, y2 = min(x + w, fw), min(y + h, fh)
        roi = frame[max(0, y):y2, max(0, x):x2]
        if roi.size == 0:
            raise RuntimeError(f"SEM panel ROI 가 프레임 밖입니다: roi={self.panel.panel_roi}, frame={fw}x{fh}")
        return roi

    def capture_screen(self) -> np.ndarray:
        """tool 창 전체 프레임(grayscale) — OK 같은 dialog 탐지용."""
        return self._capture_full_gray()

    # ---- 좌표 변환 ----

    def _frame_point_to_screen(self, frame_x: int, frame_y: int) -> dict[str, int] | None:
        """창 이미지(캡처 프레임) 좌표 → 화면 절대 좌표 (DPI 보정 포함)."""
        if self._last_frame_size is None:
            self._capture_full_gray()
        return image_point_to_screen(
            self.tool_window,
            {"x": int(frame_x), "y": int(frame_y)},
            image_size=self._last_frame_size,
        )

    # ---- 제스처 ----

    def _ensure_actionable(self, gesture: str) -> None:
        """클릭/스크롤 직전 게이트: foreground 재확보 + 창 크기 드리프트 검사.

        action_enabled 일 때만 강제한다(dry-run 은 좌표 로그만 남기므로 무해).
        foreground 를 못 잡으면 클릭이 사용자 창에 떨어질 수 있고, 캡처 후 창이
        리사이즈됐으면 내용 reflow 로 프레임 좌표가 무효다. 둘 다 잘못 클릭하느니
        RuntimeError 로 크게 실패한다(기존 변환 실패와 동일한 에러 모델 — 보정
        실패 경로가 알림으로 이어진다). 위치 이동은 변환이 live rect 로 흡수.
        """
        if not self.action_enabled:
            return
        if callable(foreground_window) and not foreground_window(
            self.tool_window, debug_label=f"sem_{gesture}"
        ):
            raise RuntimeError(f"{gesture}: tool 창 foreground 재확보 실패 (사용자 조작 중?)")
        if callable(window_rect_size) and self._last_rect_size is not None:
            current = window_rect_size(self.tool_window)
            if current is not None and (
                abs(current[0] - self._last_rect_size[0]) > RECT_DRIFT_TOL_PX
                or abs(current[1] - self._last_rect_size[1]) > RECT_DRIFT_TOL_PX
            ):
                raise RuntimeError(
                    f"{gesture}: 캡처 후 tool 창 크기 변경 감지 "
                    f"{self._last_rect_size}->{current} (좌표 무효, 재캡처 필요)"
                )

    def move_to_point(self, fov_x: int, fov_y: int) -> None:
        """FOV-local 픽셀을 더블클릭해 그 점을 중심으로 recenter 한다."""
        self._ensure_actionable("move_to_point")
        px, py = self.panel.panel_roi[0] + int(fov_x), self.panel.panel_roi[1] + int(fov_y)
        screen_point = self._frame_point_to_screen(px, py)
        if screen_point is None:
            raise RuntimeError("move_to_point: 창 좌표→스크린 변환 실패")
        click_at_screen(
            screen_point, "sem_recenter", 2, action_enabled=self.action_enabled
        )
        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

    def click_screen(self, screen_x: int, screen_y: int) -> None:
        """capture_screen 프레임 좌표를 single click 한다 (OK 버튼 등)."""
        self._ensure_actionable("click_screen")
        screen_point = self._frame_point_to_screen(screen_x, screen_y)
        if screen_point is None:
            raise RuntimeError("click_screen: 창 좌표→스크린 변환 실패")
        click_at_screen(
            screen_point, "sem_dialog_click", 1, action_enabled=self.action_enabled
        )
        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

    def zoom(self, direction: int) -> None:
        """SEM panel 중심에서 wheel 1단계 (direction=+1 zoom-in, -1 zoom-out).

        TODO(캘리브레이션): wheel 1단계당 배율 변화율을 오피스에서 측정해
        zoom_scroll_dy 와 live search 의 zoom step 모델을 맞춘다.
        """
        self._ensure_actionable("zoom")
        x, y, w, h = self.panel.panel_roi
        screen_point = self._frame_point_to_screen(x + w // 2, y + h // 2)
        if screen_point is None:
            raise RuntimeError("zoom: 창 좌표→스크린 변환 실패")
        dy = int(direction) * self.zoom_scroll_dy
        scroll_at_screen(screen_point, dy, "sem_zoom", 0, action_enabled=self.action_enabled)
        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

    # ---- 상태 ----

    def read_mode(self) -> str:
        """monitor mode label ('OM' | 'SEM' | 'unknown').

        v0: env ALIGN_SEM_MODE_OVERRIDE > mode_default. 실제 판독(모드 라벨 OCR
        또는 픽셀 휴리스틱)은 오피스 캘리브레이션 단계에서 채운다.
        """
        override = os.environ.get("ALIGN_SEM_MODE_OVERRIDE", "").strip().upper()
        if override:
            return override
        if not self._mode_warned:
            print(f"[WARNING] read_mode 미구현 - 기본값 {self.mode_default!r} 사용 (v0)")
            self._mode_warned = True
        return self.mode_default


def build_rcs_sem_monitor(
    tool_window,
    *,
    landmarks_dir=DEFAULT_LANDMARKS_DIR,
    action_enabled: bool = False,
    settle_sec: float = 0.5,
    zoom_scroll_dy: int = 1,
    mode_default: str = "SEM",
) -> RCSSEMMonitor | None:
    """tool 창에서 SEM panel 을 찾아 RCSSEMMonitor 를 만든다. 실패 시 None.

    landmark 가 없거나(`templates/sem_panel_landmarks/` 미캘리브레이션) 신뢰도가
    낮으면 None 을 반환해 호출부가 panel_not_found 로 처리하게 한다.
    """
    landmarks = load_landmarks(landmarks_dir)
    if not landmarks:
        print(f"[WARNING] SEM panel landmark 없음(미캘리브레이션): {landmarks_dir}")
        return None
    frame = _to_gray(capture_window(tool_window))
    panel = locate_panel(frame, landmarks)
    if panel is None:
        print("[WARNING] SEM panel 을 찾지 못함 (landmark 신뢰도 부족)")
        return None
    print(
        f"[INFO] SEM panel 확보: model={panel.model_id}, roi={panel.panel_roi}, "
        f"conf={panel.confidence:.3f}"
    )
    return RCSSEMMonitor(
        tool_window,
        panel,
        action_enabled=action_enabled,
        settle_sec=settle_sec,
        zoom_scroll_dy=zoom_scroll_dy,
        mode_default=mode_default,
    )


__all__ = ["RCSSEMMonitor", "build_rcs_sem_monitor", "DEFAULT_LANDMARKS_DIR"]
