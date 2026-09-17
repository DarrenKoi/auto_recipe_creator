"""RCS tool 창 기반 실장비 SEMMonitorController adapter (골격, 캘리브레이션 전).

`align.live_search.SEMMonitorController` Protocol 의 실장비 구현.
Mac mock(`_MockSEMMonitor`)과 동일 시그니처라 `correct_align_fail` /
`live_align_search` 가 그대로 돈다.

좌표공간 계약(Protocol docstring 과 동일):
  * capture()        — SEM panel ROI 만 잘라 grayscale 로 반환 (FOV-local 좌표계 원점)
  * move_to_point()  — FOV-local 픽셀을 받아 화면 절대 좌표로 변환해 **더블클릭** recenter
  * capture_screen() — tool 창 전체 프레임(grayscale). 이 프레임의 픽셀 좌표가 곧
                       click_screen 이 받는 좌표다 (창 이미지 좌표 = "screen" 계약)
  * click_screen()   — capture_screen 프레임 좌표를 화면 절대 좌표로 변환해 single click
  * zoom()           — panel 중심에서 wheel 1단계 (FOV-centered zoom)
  * read_mode()      — env ALIGN_SEM_MODE_OVERRIDE > mode_hint(PM 박스 판독) >
                       mode_default(경고). mode_hint 는 build_rcs_sem_monitor 가
                       detect_sem_box 의 pm_mode 를 그대로 주입한다.

panel ROI 확보 경로는 2 단이다(build_rcs_sem_monitor):
  1. **VLM live SEM box** — detect_sem_box(mai-ui). check-only 모니터에서 오피스
     검증된 경로이며, 장비별 사전 캘리브레이션이 필요 없다(기본).
  2. **landmark 템플릿 매칭** — templates/sem_panel_landmarks/<model_id>/.
     VLM 을 못 쓰거나 검출이 실패했을 때의 폴백(현재 대부분 미캘리브레이션).

미캘리브레이션 항목(오피스 검증 전):
  * wheel 1단계 ↔ 배율 비율 (zoom_scroll_dy)
  * 더블클릭 recenter 의 실제 이동량/settle 시간

모든 actuation 은 util.mouse_utils 의 action_enabled dry-run 게이트를 그대로
통과하므로, SAFE_MODE/dry-run 에서는 좌표 로그만 남고 실제 마우스는 움직이지
않는다.
"""

import os
import time

import cv2
import numpy as np

from poc.workflow_3 import TEMPLATES_DIR
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted
from poc.workflow_3.util.env_utils import env_float, env_int
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
from poc.workflow_3.sem_monitor.panel_locator import (
    SEMPanelMatch,
    load_landmarks,
    locate_panel,
)

LOG_COMPONENT = "rcs_sem_controller"

DEFAULT_LANDMARKS_DIR = TEMPLATES_DIR / "sem_panel_landmarks"

# 캡처~제스처 사이 창 크기 드리프트 허용 오차(논리 px). 초과 시 좌표 무효로 중단.
RECT_DRIFT_TOL_PX = 2

# 원격 뷰 클릭 성사 조건 - 시연 경로(demonstration_rcs_control)가 오피스 실측 3회로
# 얻은 값과 같다. 라이브 SEM box 의 recenter 더블클릭이 "커서는 정확히 가는데 화면이
# 안 움직이는" 증상(2026-09-15)을 보였고 원인도 같다: 도착 직후의 즉시 press/release
# 쌍은 원격의 입력 샘플링 사이로 빠진다. 줄이면 클릭이 안 먹는다(시연 문서와 동일).
REMOTE_PRE_CLICK_SETTLE_SEC = env_float("ALIGN_SEM_PRE_CLICK_SETTLE_SEC", 0.6)  # 도착 -> 누름
REMOTE_CLICK_HOLD_SEC = env_float("ALIGN_SEM_CLICK_HOLD_SEC", 0.15)             # 누름 유지

# recenter 클릭 수는 tool 의 SEM box 아이콘 모드에 따른다(사용자 보고 2026-09-15):
# crosshair 아이콘 선택 = 더블클릭, L자 아이콘 선택 = 싱글클릭. 코드는 화면에서 모드를
# 읽지 못하므로 오피스에서 맞춘다. L자 모드에 더블클릭을 보내면 두 번째 클릭이 이미
# 움직인 화면의 같은 지점을 다시 찍어 ~2배로 이동하고, crosshair 모드에 싱글은 무이동.
RECENTER_CLICKS = env_int("ALIGN_SEM_RECENTER_CLICKS", 2)

# tool 창 가림 복구. 다른 엔지니어가 접속을 시도하면 RCS 가 "Information"(Connection
# Request) 팝업을 **우리 화면에** 띄우고, 응답하지 않아도 3초 뒤 사라지면서 상대가
# 들어온다(사용자 보고 2026-09-16). 문제는 허용 여부가 아니라 그 3초다 - capture_window
# 는 창 핸들이 아니라 **창 rect 의 화면 그랩**이라 가려진 동안 찍으면 팝업 픽셀이
# 프레임에 들어오고, 매칭은 그 프레임에서 좌표를 뽑는다. 조용히 엉뚱한 점을 클릭하는
# 경로이므로 캡처 직전에 가림을 보고 걷힐 때까지 기다린다.
# 0 이하 = 감시 끔(롤백 스위치).
OCCLUSION_WAIT_SEC = env_float("ALIGN_SEM_OCCLUSION_WAIT_SEC", 6.0)  # 3s 팝업보다 넉넉히
OCCLUSION_POLL_SEC = env_float("ALIGN_SEM_OCCLUSION_POLL_SEC", 0.3)
# 판정 heartbeat(초). 가림이 없을 때도 이 간격으로 판정값을 찍는다 - 오피스에서
# "아무 줄도 안 나왔다" 가 두 가지(가림이 없었다 / 판정기가 못 본다)를 뜻해 버리는
# 것을 막는 유일한 장치다. 팝업이 원격 뷰 안에 그려지면 WindowFromPoint 는 계속
# 우리 창을 짚어 "none" 만 나오는데, heartbeat 가 있으면 그 사실이 화면에 보인다.
# 0 이하 = 상태가 바뀔 때만 출력.
OCCLUSION_LOG_SEC = env_float("ALIGN_SEM_OCCLUSION_LOG_SEC", 10.0)


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


def wait_unoccluded(occlusion_fn) -> str:
    """가림이 걷힐 때까지 폴링하고 마지막 판정을 돌려준다("none"/"unknown"/"partial"/"full").

    첫 판정의 "unknown" 은 막지 않는다 - 조회 실패이지 가림이 아니며, 이는
    `frame_meta.classify_occlusion` 이 세운 규약 그대로다(Mac 에서는 항상 unknown 이라
    이 게이트가 통째로 no-op 이 된다).

    그러나 **가림을 한 번 본 뒤의 "unknown" 은 해소가 아니다**(codex 재리뷰
    2026-09-16 FINDING 1). 판정기를 잃은 것과 화면이 깨끗해진 것은 다른 사건인데,
    종전 코드는 full -> unknown 을 '해소'로 읽고 즉시 캡처를 허용했다 - 가림을 이미
    관측한 상태에서 증거를 잃었을 때 통과시키는 것이 정확히 이 게이트가 막으려던
    silent-wrong 이다. 한 번 막히면 명시적 "none" 만 해소로 인정한다.

    예외는 던지지 않는다 - '기다렸는데 안 걷혔다' 를 어떻게 처리할지는 호출부가
    정한다(제스처 경로는 실패, panel 탐색은 panel_not_found).
    """
    if occlusion_fn is None or OCCLUSION_WAIT_SEC <= 0:
        return "unknown"

    def _read() -> str:
        try:
            return occlusion_fn()
        except Exception as exc:
            print(f"[WARNING] 가림 판정 실패: {exc}")
            return "unknown"

    state = _read()
    if state not in ("partial", "full"):
        return state
    blocked = state
    print(
        f"[WARNING] tool 창 가림 감지({blocked}) - 캡처 보류, "
        f"최대 {OCCLUSION_WAIT_SEC:.1f}s 대기 (접속 요청 팝업 추정)"
    )
    deadline = time.time() + OCCLUSION_WAIT_SEC
    while time.time() < deadline:
        if is_aborted():
            print(f"[WARNING] 긴급 해제({abort_reason()}) - 가림 대기 중단")
            return blocked
        time.sleep(max(0.05, OCCLUSION_POLL_SEC))
        state = _read()
        if state == "none":
            print("[INFO] 가림 해소(none) - 캡처 재개")
            return state
        if state in ("partial", "full"):
            blocked = state
        # "unknown" = 판정기를 잃음. 해소로 치지 않고 계속 기다린다.
    print(f"[WARNING] 가림이 {OCCLUSION_WAIT_SEC:.1f}s 안에 걷히지 않음({blocked})")
    return blocked


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
        mode_hint: str | None = None,
        occlusion_fn=None,
    ):
        self.tool_window = tool_window
        self.panel = panel
        self.action_enabled = action_enabled
        self.settle_sec = settle_sec
        self.zoom_scroll_dy = zoom_scroll_dy
        self.mode_default = mode_default
        # 화면에서 읽은 modality("OM"|"SEM"). None 이면 판독 실패 -> mode_default 경고 경로.
        self.mode_hint = (mode_hint or "").strip().upper() or None
        # 가림 판정자(주입). sem_monitor 는 monitor 아래 계층이라 frame_meta 를
        # 직접 import 할 수 없다(4-layer DAG) - cycle.py 가 채운다.
        # None 이면 게이트가 통째로 no-op 이라 mock/테스트/다른 호출부는 무영향.
        self.occlusion_fn = occlusion_fn
        # heartbeat 출력용 - 직전 판정과 마지막 출력 시각.
        self._occlusion_state = ""
        self._occlusion_logged_at = 0.0
        # image_point_to_screen 의 DPI 보정에 쓰는 캡처 프레임 크기 (w, h).
        self._last_frame_size: tuple[int, int] | None = None
        # 캡처 시점의 창 rect 크기(논리 px) — 제스처 직전 리사이즈 드리프트 감지용.
        self._last_rect_size: tuple[int, int] | None = None
        # panel_roi 가 유효한 프레임 크기 - 첫 캡처에서 고정한다. _last_* 는 캡처마다
        # 갱신되므로 '캡처~제스처' 드리프트만 잡고, 사이클 도중 리사이즈로 panel_roi
        # 자체가 낡은 것은 못 잡는다(codex 리뷰 2026-09-17).
        self._panel_frame_size: tuple[int, int] | None = None
        self._mode_warned = False
        if action_enabled:
            # 캘리브레이션 미완료 경고는 2026-07-07 오피스 캘리브레이션으로 사실이 아니게
            # 됐다. 실운전 중에 "dry-run 권장"을 찍으면 진짜 실패 원인을 가리므로, 무장
            # 사실만 남긴다.
            print("[INFO] RCSSEMMonitor: action_enabled=True - 실제 클릭/휠이 장비로 나갑니다.")

    # ---- 캡처 ----

    def _log_occlusion(self, state: str) -> None:
        """가림이 없을 때의 판정값을 상태 변화 시 + heartbeat 간격으로 찍는다.

        가림을 만났을 때만 찍으면 오피스에서 침묵이 두 가지를 뜻한다 - "아무도
        접속을 시도하지 않았다" 와 "팝업이 원격 뷰 안에 그려져 판정기가 못 본다".
        후자면 이 게이트는 켜져 있어도 무력이므로 구분이 되어야 한다.
        """
        now = time.time()
        changed = state != self._occlusion_state
        due = OCCLUSION_LOG_SEC > 0 and (now - self._occlusion_logged_at) >= OCCLUSION_LOG_SEC
        if not (changed or due):
            return
        print(f"[INFO] 가림 판정={state} (감시 동작 중 - 가림 감지 시 캡처를 보류합니다)")
        self._occlusion_state = state
        self._occlusion_logged_at = now

    def _wait_unoccluded(self) -> str:
        """캡처 직전 가림 대기. 안 걷히면 **실행 중일 때만** RuntimeError.

        예산을 넘겼다는 것은 "지금 화면을 못 본다"는 확정이다. 그대로 캡처하면
        오염된 프레임이 좌표 근거가 되는데, 매칭 점수는 backstop 이 아니다 -
        `key_visibility_gate` 는 낮은 점수를 `fallback_search` 로 보내고 그 경로는
        live_align_search 가 **스테이지를 실제로 움직인다**. 즉 "못 보면 가만히
        있는다" 가 아니라 "못 보면 장비를 움직인다" 가 된다(codex 리뷰 2026-09-16).

        그래서 `_ensure_actionable` 의 foreground 실패와 같은 모델로 크게 실패한다 -
        보정 실패는 failure_class 가 되어 cube 로 나가고 엔지니어가 받는다.
        dry-run/SAFE_MODE 는 좌표 로그만 남기므로 종전대로 통과시킨다.
        """
        state = self._probe_until_clear()
        if state in ("partial", "full") and self.action_enabled:
            raise RuntimeError(
                f"tool 창 가림({state})이 {OCCLUSION_WAIT_SEC:.1f}s 안에 걷히지 않음 - "
                f"화면을 못 보는 상태에서 좌표를 뽑지 않습니다"
            )
        return state

    def _probe_until_clear(self) -> str:
        """가림이 걷힐 때까지 기다리고 마지막 판정을 돌려준다(예외 없음)."""
        if self.occlusion_fn is None or OCCLUSION_WAIT_SEC <= 0:
            return "unknown"
        state = wait_unoccluded(self.occlusion_fn)
        if state not in ("partial", "full"):
            self._log_occlusion(state)
        return state

    def _capture_full_gray(self) -> np.ndarray:
        """tool 창 전체를 캡처해 grayscale 로 반환하고 프레임 크기를 캐시한다."""
        self._wait_unoccluded()
        image = capture_window(self.tool_window)
        gray = _to_gray(image)
        h, w = gray.shape[:2]
        if self._panel_frame_size is None:
            self._panel_frame_size = (w, h)
        elif (
            abs(w - self._panel_frame_size[0]) > RECT_DRIFT_TOL_PX
            or abs(h - self._panel_frame_size[1]) > RECT_DRIFT_TOL_PX
        ):
            # ponytail: 재검출(VLM) 대신 크게 실패 - 보정 실패 -> cube 로 엔지니어가 받는다.
            # 사이클 중 리사이즈가 잦다고 확인되면 panel 재검출을 붙인다.
            raise RuntimeError(
                f"tool 창 프레임 크기 변경 {self._panel_frame_size}->{(w, h)} - "
                f"SEM panel ROI 가 낡음(재검출 필요)"
            )
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
        """FOV-local 픽셀을 클릭해 그 점을 중심으로 recenter 한다(클릭 수 = RECENTER_CLICKS)."""
        self._ensure_actionable("move_to_point")
        px, py = self.panel.panel_roi[0] + int(fov_x), self.panel.panel_roi[1] + int(fov_y)
        screen_point = self._frame_point_to_screen(px, py)
        if screen_point is None:
            raise RuntimeError("move_to_point: 창 좌표→스크린 변환 실패")
        click_at_screen(
            screen_point, "sem_recenter", RECENTER_CLICKS, action_enabled=self.action_enabled,
            hold_sec=REMOTE_CLICK_HOLD_SEC, pre_click_settle_sec=REMOTE_PRE_CLICK_SETTLE_SEC,
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
            screen_point, "sem_dialog_click", 1, action_enabled=self.action_enabled,
            hold_sec=REMOTE_CLICK_HOLD_SEC, pre_click_settle_sec=REMOTE_PRE_CLICK_SETTLE_SEC,
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
        """monitor mode label ('OM' | 'SEM').

        우선순위: env ALIGN_SEM_MODE_OVERRIDE > mode_hint(PM 박스 판독) > mode_default.
        mode_hint 는 생성 시점의 PM 판독값이다 — align fail 은 장비가 멈춘 정지 화면이라
        (project memory: SEM monitor static at align fail) 사이클 도중 modality 가 바뀌지
        않으므로 매 호출 재판독하지 않는다.
        """
        override = os.environ.get("ALIGN_SEM_MODE_OVERRIDE", "").strip().upper()
        if override:
            return override
        if self.mode_hint:
            return self.mode_hint
        if not self._mode_warned:
            print(
                f"[WARNING] modality 판독값 없음(PM 미검출) - 기본값 {self.mode_default!r} 사용. "
                f"OM step 실패였다면 잘못된 template 로 매칭될 수 있음."
            )
            self._mode_warned = True
        return self.mode_default


def _panel_from_vlm_box(
    tool_window, vlm_client, *, ocr_client=None, two_stage: bool = False,
    reasons: list | None = None, fail_frame_path=None,
) -> tuple[SEMPanelMatch, str | None] | None:
    """detect_sem_box 로 live SEM box 를 잡아 (SEMPanelMatch, pm_mode) 로 변환한다.

    check-only 모니터가 쓰는 것과 같은 검출기라 장비별 사전 캘리브레이션이 필요 없다.
    VLM 부재/검출 실패/예외는 모두 None 으로 돌려 호출부가 landmark 폴백을 타게 한다
    (개발 PC 에서 import 조차 실패할 수 있어 함수 안에서 import 한다).
    """
    def _fail(reason: str, frame=None):
        """실패 사유를 남기고(선택) 그때 본 화면을 저장한다.

        사유를 구분하지 않으면 호출부가 정적 문자열 하나로 뭉개 저널에 적고,
        나중에 "클라이언트가 없었다" 와 "보았는데 없었다" 를 가릴 수 없게 된다.
        """
        if reasons is not None:
            reasons.append(reason)
        if frame is not None and fail_frame_path is not None:
            try:
                save_debug_jpeg(frame, Path(fail_frame_path))
                print(f"[INFO] SEM box 검출 실패 화면 저장: {fail_frame_path}")
            except Exception as exc:
                print(f"[WARNING] 실패 화면 저장 실패(무시): {exc}")
        return None

    if vlm_client is None:
        # 유일하게 콘솔에도 안 남던 경로. sem_box VLM 클라이언트 생성이 깨졌다는 뜻이다.
        print("[WARNING] SEM box VLM 클라이언트 없음 - live box 검출을 건너뛴다")
        return _fail("vlm_client_missing")
    try:
        frame_image = capture_window(tool_window)
    except Exception as exc:
        print(f"[WARNING] tool 창 캡처 실패(landmark 폴백 시도): {exc}")
        return _fail(f"capture_error:{type(exc).__name__}")
    try:
        from poc.workflow_3.sem_monitor.sem_box_detect import detect_sem_box

        detection = detect_sem_box(
            frame_image, vlm_client, ocr_client=ocr_client, two_stage=two_stage,
        )
    except Exception as exc:
        print(f"[WARNING] live SEM box 검출 실패(landmark 폴백 시도): {exc}")
        return _fail(f"sem_box_detect_error:{type(exc).__name__}:{exc}", frame_image)

    bbox = getattr(detection, "bbox_px", None)
    if not detection.detected or not bbox:
        # 점유 view-only 화면처럼 SEM Monitor 가 아예 안 떠 있으면 여기로 온다.
        # 오검출이 아니라 정당한 거부이므로, 화면을 남겨 눈으로 가리게 한다.
        pm = getattr(detection, "pm_text", None)
        print(f"[WARNING] live SEM box 미검출(landmark 폴백 시도) pm_text={pm!r}")
        return _fail(f"sem_box_not_detected(pm_text={pm!r})", frame_image)
    left, top = int(bbox["left"]), int(bbox["top"])
    width, height = int(bbox["right"]) - left, int(bbox["bottom"]) - top
    if width <= 0 or height <= 0:
        print(f"[WARNING] live SEM box 크기 이상(landmark 폴백 시도): {bbox}")
        return _fail(f"sem_box_degenerate:{bbox}", frame_image)

    panel = SEMPanelMatch(
        model_id="vlm_live_box",
        panel_roi=(left, top, width, height),
        landmark_xy=(left, top),  # landmark 없음 - ROI 원점을 그대로 둔다.
        confidence=float(detection.confidence or 0.0),
        nm_per_pixel=None,
    )
    return panel, detection.pm_mode


def build_rcs_sem_monitor(
    tool_window,
    *,
    vlm_client=None,
    ocr_client=None,
    pm_two_stage: bool = False,
    landmarks_dir=DEFAULT_LANDMARKS_DIR,
    reason_sink: list | None = None,
    fail_frame_path=None,
    action_enabled: bool = False,
    settle_sec: float = 0.5,
    zoom_scroll_dy: int = 1,
    mode_default: str = "SEM",
    occlusion_fn=None,
) -> RCSSEMMonitor | None:
    """tool 창에서 SEM panel 을 찾아 RCSSEMMonitor 를 만든다. 실패 시 None.

    panel ROI 는 (1) vlm_client 가 있으면 detect_sem_box 의 live SEM box, 실패 시
    (2) landmark 템플릿 매칭 순으로 확보한다. 둘 다 실패하면 None 을 반환해 호출부가
    panel_not_found 로 처리하게 한다. VLM 경로일 때는 같은 검출의 PM 판독값(pm_mode)을
    mode_hint 로 주입하므로 read_mode 가 OM/SEM 을 화면에서 읽은 값으로 답한다.
    """
    mode_hint: str | None = None
    reasons: list = []
    # panel 탐색도 같은 게이트를 지난다. 여기서 찍은 프레임은 panel_roi 와 mode_hint 로
    # **캐시되어 사이클 내내 재사용**되므로, 가려진 화면으로 한 번 잘못 잡으면 이후의
    # 깨끗한 캡처가 그것을 고쳐 주지 않는다(codex 리뷰 2026-09-16 FINDING 3).
    # _capture_full_gray 의 게이트는 이 시점 뒤에야 붙는다 - 그 전에 여기서 막는다.
    panel_occlusion = wait_unoccluded(occlusion_fn)
    if panel_occlusion in ("partial", "full"):
        print(f"[WARNING] SEM panel 탐색 보류 - tool 창 가림({panel_occlusion})")
        if reason_sink is not None:
            reason_sink.append(f"occluded_{panel_occlusion}")
        return None
    resolved = _panel_from_vlm_box(
        tool_window, vlm_client, ocr_client=ocr_client, two_stage=pm_two_stage,
        reasons=reasons, fail_frame_path=fail_frame_path,
    )

    def _give_up(landmark_reason: str):
        """VLM 사유 + landmark 사유를 합쳐 한 줄로 남긴다 - 저널이 이걸 그대로 적는다."""
        if reason_sink is not None:
            reason_sink.append(" + ".join(reasons + [landmark_reason]))
        return None

    if resolved is not None:
        panel, mode_hint = resolved
    else:
        landmarks = load_landmarks(landmarks_dir)
        if not landmarks:
            print(f"[WARNING] SEM panel landmark 없음(미캘리브레이션): {landmarks_dir}")
            return _give_up("landmark_missing")
        # VLM 왕복(수 초) 사이에 팝업이 뜰 수 있으므로 여기서 다시 본다 - 위의
        # 1회 검사는 _panel_from_vlm_box 직전 상태일 뿐이다(codex 재리뷰 FINDING 2).
        landmark_occlusion = wait_unoccluded(occlusion_fn)
        if landmark_occlusion in ("partial", "full"):
            print(f"[WARNING] landmark 폴백 보류 - tool 창 가림({landmark_occlusion})")
            return _give_up(f"occluded_{landmark_occlusion}")
        frame = _to_gray(capture_window(tool_window))
        panel = locate_panel(frame, landmarks)
        if panel is None:
            print("[WARNING] SEM panel 을 찾지 못함 (landmark 신뢰도 부족)")
            return _give_up("landmark_low_confidence")
    print(
        f"[INFO] SEM panel 확보: model={panel.model_id}, roi={panel.panel_roi}, "
        f"conf={panel.confidence:.3f}, mode={mode_hint or '-'}"
    )
    return RCSSEMMonitor(
        tool_window,
        panel,
        action_enabled=action_enabled,
        settle_sec=settle_sec,
        zoom_scroll_dy=zoom_scroll_dy,
        mode_default=mode_default,
        mode_hint=mode_hint,
        occlusion_fn=occlusion_fn,
    )


__all__ = ["RCSSEMMonitor", "build_rcs_sem_monitor", "DEFAULT_LANDMARKS_DIR"]
