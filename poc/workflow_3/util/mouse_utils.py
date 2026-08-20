"""마우스 클릭/스크롤 유틸리티.

RCS 는 로컬 PC 의 마우스 '움직임'을 원격 tool 화면으로 전달한다. pynput 의
``mouse.position = (x, y)`` 처럼 좌표를 한 번에 순간이동하면 OS 이벤트가 1개만 발생해
RCS 가 그 이동을 놓치거나 원격 커서가 목표(live SEM box) 안으로 실제로 들어가지 않는다.
그래서 커서 이동은 '한 번에 점프'가 아니라 '작은 단계로 미끄러뜨려'(glide) 연속 move
이벤트를 흘려보내고, 도착 후 박스 안에서 미세하게 흔들어(jiggle) RCS 가 커서를 그 위치에
확실히 등록하게 한 뒤에 wheel/click 을 건다.
"""

import time

from poc.workflow_3.util.abort_switch import abort_reason, is_aborted
from poc.workflow_3.util.env_utils import env_float, env_int

try:
    from pynput.mouse import Button, Controller as MouseController

    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False

# glide/jiggle 튜닝(환경변수로 오피스 RCS 반응성에 맞춰 보정).
_GLIDE_STEPS = env_int("ALIGN_FAIL_CURSOR_GLIDE_STEPS", 24)      # 시작→목표 분할 이동 횟수
_GLIDE_DELAY = env_float("ALIGN_FAIL_CURSOR_GLIDE_DELAY", 0.012)  # 각 단계 사이 sleep(초)
_JIGGLE_PX = env_int("ALIGN_FAIL_CURSOR_JIGGLE_PX", 3)            # 도착 후 흔들 픽셀(0=off)


def _glide_to(mouse, target_x: int, target_y: int) -> None:
    """현재 위치에서 목표 좌표까지 작은 단계로 미끄러져 이동한다(연속 move 이벤트 생성)."""
    try:
        start = mouse.position
        sx0, sy0 = int(start[0]), int(start[1])
    except Exception:
        sx0, sy0 = target_x, target_y
    steps = max(1, _GLIDE_STEPS)
    for i in range(1, steps + 1):
        if is_aborted():
            # 진행 중인 이동도 그 자리에서 끊는다. 사용자가 단축키를 눌렀는데
            # 커서가 300ms 더 끌려가면 "해제됐다"고 느껴지지 않는다.
            return
        ix = int(round(sx0 + (target_x - sx0) * i / steps))
        iy = int(round(sy0 + (target_y - sy0) * i / steps))
        mouse.position = (ix, iy)
        if _GLIDE_DELAY > 0:
            time.sleep(_GLIDE_DELAY)


def _jiggle(mouse, x: int, y: int) -> None:
    """목표 지점에서 미세하게 흔들어 RCS 가 커서를 그 위치에 등록하게 한다."""
    px = _JIGGLE_PX
    if px <= 0 or is_aborted():
        return
    for dx, dy in ((px, 0), (-px, 0), (0, px), (0, -px), (0, 0)):
        if is_aborted():
            return
        mouse.position = (x + dx, y + dy)
        if _GLIDE_DELAY > 0:
            time.sleep(_GLIDE_DELAY)


def click_at_screen(
    screen_point: dict[str, int],
    target_key: str,
    click_count: int = 1,
    *,
    action_enabled: bool = True,
    hold_sec: float = 0.0,
) -> bool:
    """스크린 좌표에서 마우스 클릭을 수행한다.

    `hold_sec` > 0 이면 `click()` 대신 press -> 유지 -> release 로 누른다. RCS 원격
    뷰처럼 입력을 주기적으로 샘플링해 장비로 넘기는 화면에서는, pynput 의 즉시
    press/release 쌍이 두 샘플 사이에 통째로 들어가 **눌린 적 없는 것으로** 넘어갈 수
    있다(2026-08-19 오피스: 커서는 버튼 위로 가는데 클릭만 안 먹음). 기본값 0.0 은
    종전 동작 그대로라 기존 호출부에 영향이 없다.
    """
    sx, sy = screen_point["x"], screen_point["y"]

    if is_aborted():
        # 긴급 해제(전역 단축키). 사용자가 마우스를 되찾은 상태이므로 어떤 출력도 내지 않는다.
        print(f"[WARNING] 긴급 해제 상태 - 마우스 출력 생략: reason={abort_reason()}")
        return False
    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] 클릭 생략: target={target_key}, screen=({sx}, {sy}), "
            f"click_count={click_count}, hold_sec={hold_sec}, "
            f"action_enabled={action_enabled}, pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    # 순간이동 대신 glide+jiggle — RCS 가 커서 이동을 따라와 클릭이 원격에서 목표 위(PM 버튼/
    # 드롭다운 행)에 걸리게 한다(teleport 면 RCS 가 이동을 놓쳐 엉뚱한 위치를 클릭).
    _glide_to(mouse, sx, sy)
    _jiggle(mouse, sx, sy)
    time.sleep(0.01)
    if hold_sec > 0:
        for _ in range(max(1, click_count)):
            mouse.press(Button.left)
            time.sleep(hold_sec)
            mouse.release(Button.left)
            time.sleep(0.02)
    else:
        mouse.click(Button.left, click_count)
    print(
        f"[INFO] 클릭 완료(glide+jiggle): target={target_key}, screen=({sx}, {sy}), "
        f"click_count={click_count}, hold_sec={hold_sec}"
    )
    return True


def move_cursor_to_screen(
    screen_point: dict[str, int],
    target_key: str,
    *,
    action_enabled: bool = True,
) -> bool:
    """스크린 좌표로 마우스 커서만 이동한다(클릭 없음).

    align point 매핑을 눈으로 검증하는 용도 — 장비에 부작용(클릭/recenter)을 주지 않고
    커서만 옮긴다. action_enabled=False 또는 pynput 부재면 DRY-RUN 로그만 남긴다.
    """
    sx, sy = screen_point["x"], screen_point["y"]

    if is_aborted():
        # 긴급 해제(전역 단축키). 사용자가 마우스를 되찾은 상태이므로 어떤 출력도 내지 않는다.
        print(f"[WARNING] 긴급 해제 상태 - 마우스 출력 생략: reason={abort_reason()}")
        return False
    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] 커서 이동 생략: target={target_key}, screen=({sx}, {sy}), "
            f"action_enabled={action_enabled}, pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    _glide_to(mouse, sx, sy)
    _jiggle(mouse, sx, sy)
    print(
        f"[INFO] 커서 이동 완료(glide+jiggle): target={target_key}, screen=({sx}, {sy}), "
        f"steps={_GLIDE_STEPS}, jiggle_px={_JIGGLE_PX}"
    )
    return True


def scroll_at_screen(
    screen_point: dict[str, int],
    dy: int,
    phase: str,
    step_index: int,
    *,
    action_enabled: bool = True,
) -> bool:
    """스크린 좌표에서 mouse wheel scroll 을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if is_aborted():
        # 긴급 해제(전역 단축키). 사용자가 마우스를 되찾은 상태이므로 어떤 출력도 내지 않는다.
        print(f"[WARNING] 긴급 해제 상태 - 마우스 출력 생략: reason={abort_reason()}")
        return False
    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] scroll 생략: phase={phase}, step={step_index}, "
            f"screen=({sx}, {sy}), dy={dy}, action_enabled={action_enabled}, "
            f"pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    # 순간이동 대신 glide — RCS 가 커서 이동을 따라와 wheel 이 SEM box 위에서 걸리게 한다.
    _glide_to(mouse, sx, sy)
    time.sleep(0.01)
    mouse.scroll(0, dy)
    print(f"[INFO] scroll 완료: phase={phase}, step={step_index}, screen=({sx}, {sy}), dy={dy}")
    return True
