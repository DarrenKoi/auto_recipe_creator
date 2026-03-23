"""마우스 클릭/스크롤 유틸리티."""

import time

try:
    from pynput.mouse import Button, Controller as MouseController

    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False


def click_at_screen(
    screen_point: dict[str, int],
    target_key: str,
    click_count: int = 1,
    *,
    action_enabled: bool = True,
) -> bool:
    """스크린 좌표에서 마우스 클릭을 수행한다.

    Returns:
        항상 True (호출자에서 성공 플래그로 사용).
    """
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


def scroll_at_screen(
    screen_point: dict[str, int],
    dy: int,
    phase: str,
    step_index: int,
    *,
    action_enabled: bool = True,
) -> bool:
    """스크린 좌표에서 mouse wheel scroll 을 수행한다.

    Returns:
        항상 True (호출자에서 성공 플래그로 사용).
    """
    sx, sy = screen_point["x"], screen_point["y"]

    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] scroll 생략: phase={phase}, step={step_index}, "
            f"screen=({sx}, {sy}), dy={dy}, action_enabled={action_enabled}, "
            f"pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.scroll(0, dy)
    print(f"[INFO] scroll 완료: phase={phase}, step={step_index}, screen=({sx}, {sy}), dy={dy}")
    return True
