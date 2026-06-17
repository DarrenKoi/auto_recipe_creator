"""마우스 클릭/스크롤 유틸리티.

RCS 는 로컬 PC 의 마우스 '움직임'을 원격 tool 화면으로 전달한다. pynput 의
``mouse.position = (x, y)`` 처럼 좌표를 한 번에 순간이동하면 OS 이벤트가 1개만 발생해
RCS 가 그 이동을 놓치거나 원격 커서가 목표(live SEM box) 안으로 실제로 들어가지 않는다.
그래서 커서 이동은 '한 번에 점프'가 아니라 '작은 단계로 미끄러뜨려'(glide) 연속 move
이벤트를 흘려보내고, 도착 후 박스 안에서 미세하게 흔들어(jiggle) RCS 가 커서를 그 위치에
확실히 등록하게 한 뒤에 wheel/click 을 건다.
"""

import time

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
        ix = int(round(sx0 + (target_x - sx0) * i / steps))
        iy = int(round(sy0 + (target_y - sy0) * i / steps))
        mouse.position = (ix, iy)
        if _GLIDE_DELAY > 0:
            time.sleep(_GLIDE_DELAY)


def _jiggle(mouse, x: int, y: int) -> None:
    """목표 지점에서 미세하게 흔들어 RCS 가 커서를 그 위치에 등록하게 한다."""
    px = _JIGGLE_PX
    if px <= 0:
        return
    for dx, dy in ((px, 0), (-px, 0), (0, px), (0, -px), (0, 0)):
        mouse.position = (x + dx, y + dy)
        if _GLIDE_DELAY > 0:
            time.sleep(_GLIDE_DELAY)


def click_at_screen(
    screen_point: dict[str, int],
    target_key: str,
    click_count: int = 1,
    *,
    action_enabled: bool = True,
) -> bool:
    """스크린 좌표에서 마우스 클릭을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] 클릭 생략: target={target_key}, screen=({sx}, {sy}), "
            f"click_count={click_count}, action_enabled={action_enabled}, "
            f"pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.click(Button.left, click_count)
    print(
        f"[INFO] 클릭 완료: target={target_key}, screen=({sx}, {sy}), "
        f"click_count={click_count}"
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
