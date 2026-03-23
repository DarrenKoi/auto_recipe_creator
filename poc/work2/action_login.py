"""RCS 로그인 창 자동 클릭 액션 스크립트.

VLM 파이프라인(ui-venus + mai-ui)으로 로그인 창의 UI 요소를 탐지한 뒤,
pynput 으로 실제 클릭을 수행한다.

실행 순서:
  1. 로그인 창 탐색
  2. 타겟 요소 VLM 탐지 (userid_input, password_input, login_button)
  3. 탐지된 요소를 순서대로 클릭

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/action_login.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_rcs_common import find_login_window, wait_for_rcs_main_window
from poc.work2.login_rcs_ui_venus_mai import (
    EXIT_SUCCESS,
    PREDEFINED_TARGETS,
    TargetResult,
    analyze_login_target,
)
from poc.work2.logger import log_work2_event
from poc.work2.util import (
    activate_window,
    capture_window,
    foreground_window,
    format_elapsed_ms,
)

try:
    from pynput.mouse import Button, Controller as MouseController
    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False
    print("[WARNING] pynput.mouse 미설치 — 클릭 동작은 로그만 출력됩니다.")

try:
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_KEYBOARD_AVAILABLE = True
except ImportError:
    PYNPUT_KEYBOARD_AVAILABLE = False
    print("[WARNING] pynput.keyboard 미설치 — 타이핑 동작은 로그만 출력됩니다.")

load_dotenv()

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

# ---------------------------------------------------------------------------
# 타이핑 기능 토글 — True 면 ID/PW 입력 후 로그인 버튼 클릭
# ---------------------------------------------------------------------------
TYPING_ENABLED = True

# 타이핑할 자격증명
CREDENTIAL_USER_ID = "2067928"
CREDENTIAL_PASSWORD = "1"

# 타이핑 대상 → 입력값 매핑
TYPING_TARGETS: dict[str, str] = {
    "userid_input": CREDENTIAL_USER_ID,
    "password_input": CREDENTIAL_PASSWORD,
}

# 클릭할 타겟 순서 (TYPING_ENABLED 시 타이핑 타겟이 앞에 추가됨)
_CLICK_ONLY_TARGETS = ["login_button"]
ACTION_TARGETS = (
    list(TYPING_TARGETS.keys()) + _CLICK_ONLY_TARGETS
    if TYPING_ENABLED
    else _CLICK_ONLY_TARGETS
)

# 클릭 전후 대기 시간 (초)
PRE_CLICK_SETTLE_SEC = 0.2
POST_CLICK_SETTLE_SEC = 0.3

# 타이핑 관련 대기 시간 (초)
CHAR_TYPE_DELAY_SEC = 0.03
POST_TYPE_SETTLE_SEC = 0.3

def _image_point_to_screen(window, image_point: dict) -> dict[str, int] | None:
    """이미지 픽셀 좌표를 스크린 절대 좌표로 변환한다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[ERROR] 창 rectangle 조회 실패: {exc}")
        return None

    return {
        "x": rect.left + image_point["x"],
        "y": rect.top + image_point["y"],
    }


# ---------------------------------------------------------------------------
# 클릭 / 타이핑 실행
# ---------------------------------------------------------------------------


def _click_at_screen(screen_point: dict, target_key: str, click_count: int = 1) -> bool:
    """스크린 좌표에서 마우스 좌클릭을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not PYNPUT_MOUSE_AVAILABLE:
        print(f"[INFO] [DRY-RUN] 클릭 생략 (pynput 없음): target={target_key}, screen=({sx}, {sy})")
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.click(Button.left, click_count)
    print(f"[INFO] 클릭 완료 (x{click_count}): target={target_key}, screen=({sx}, {sy})")
    return True


def _clear_and_type(text: str, target_key: str) -> bool:
    """기존 텍스트를 Ctrl+A → Delete 로 지운 뒤, 새 텍스트를 타이핑한다."""
    if not PYNPUT_KEYBOARD_AVAILABLE:
        print(f"[INFO] [DRY-RUN] 타이핑 생략 (pynput 없음): target={target_key}, text={text!r}")
        return True

    kb = KeyboardController()

    # Ctrl+A 로 전체 선택
    kb.press(Key.ctrl)
    kb.press("a")
    kb.release("a")
    kb.release(Key.ctrl)
    time.sleep(0.05)

    # Delete 로 선택 영역 삭제
    kb.press(Key.delete)
    kb.release(Key.delete)
    time.sleep(0.05)

    # 한 글자씩 타이핑
    for ch in text:
        kb.type(ch)
        time.sleep(CHAR_TYPE_DELAY_SEC)

    print(f"[INFO] 타이핑 완료: target={target_key}, text={text!r}")
    return True


def main() -> str:
    """로그인 창 타겟 탐지 후 클릭 액션을 수행한다."""
    script_started_at = time.time()

    log_work2_event(
        component=COMPONENT_NAME,
        message="action_started",
        log_name=LOG_NAME,
        targets=ACTION_TARGETS,
    )

    # 1. 로그인 창 탐색
    login_window, window_title, backend = find_login_window()
    if login_window is None:
        print(
            "[ERROR] 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        return "login_window_not_found"

    print(f"[INFO] 로그인 창 발견: title={window_title!r}, backend={backend}")

    # 2. 스크린샷 1회 캡처 → 모든 타겟 탐지에 재사용
    if not activate_window(login_window, debug_label="action_login_activate"):
        print("[ERROR] 로그인 창 활성화 실패")
        return "window_activate_failed"

    if not foreground_window(login_window, debug_label="action_login_capture"):
        print("[ERROR] 로그인 창 foreground 실패")
        return "window_activate_failed"

    try:
        shared_image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        return "capture_failed"

    print(f"[INFO] 스크린샷 캡처 완료: {shared_image.size[0]}x{shared_image.size[1]}")

    detected: list[tuple[str, TargetResult]] = []

    for target_key in ACTION_TARGETS:
        target_config = PREDEFINED_TARGETS.get(target_key)
        if target_config is None:
            print(f"[WARNING] 미정의 타겟 건너뜀: {target_key}")
            continue

        print(f"\n[INFO] === 타겟 탐지 시작: {target_key} ===")
        result = analyze_login_target(
            login_window, window_title, backend, target_config,
            image=shared_image,
        )

        if result.exit_code == EXIT_SUCCESS and result.point is not None:
            print(
                f"[INFO] 타겟 탐지 성공: {target_key}, "
                f"image_point=({result.point['x']}, {result.point['y']})"
            )
            detected.append((target_key, result))
        else:
            print(f"[WARNING] 타겟 탐지 실패: {target_key}, exit_code={result.exit_code}")

    if not detected:
        print("[ERROR] 탐지된 타겟이 없습니다. 클릭 생략.")
        return "no_targets_detected"

    # 3. 탐지된 타겟 순서대로 클릭
    print(f"\n[INFO] === 클릭 액션 시작 ({len(detected)}개 타겟) ===")
    click_results: list[dict] = []

    for target_key, result in detected:
        # 클릭 직전에 창을 다시 foreground 로 올린다
        print(f"[INFO] 창 재활성화: target={target_key}")
        foreground_window(
            login_window,
            debug_label=f"pre_click_{target_key}",
        )
        time.sleep(PRE_CLICK_SETTLE_SEC)

        screen_point = _image_point_to_screen(login_window, result.point)
        if screen_point is None:
            print(f"[ERROR] 스크린 좌표 변환 실패: target={target_key}")
            click_results.append({"target": target_key, "clicked": False})
            continue

        # 타이핑 대상인 경우: 클릭 → 기존 텍스트 지우고 타이핑
        type_value = TYPING_TARGETS.get(target_key) if TYPING_ENABLED else None
        if type_value is not None:
            print(
                f"[INFO] 타이핑 타겟 클릭: target={target_key}, "
                f"screen=({screen_point['x']}, {screen_point['y']})"
            )
            clicked = _click_at_screen(screen_point, target_key)
            if clicked:
                time.sleep(0.1)
                typed = _clear_and_type(type_value, target_key)
                click_results.append({"target": target_key, "clicked": clicked, "typed": typed})
            else:
                click_results.append({"target": target_key, "clicked": False, "typed": False})
            time.sleep(POST_TYPE_SETTLE_SEC)
        else:
            print(
                f"[INFO] 클릭 실행: target={target_key}, "
                f"image=({result.point['x']}, {result.point['y']}), "
                f"screen=({screen_point['x']}, {screen_point['y']})"
            )
            clicked = _click_at_screen(screen_point, target_key)
            click_results.append({"target": target_key, "clicked": clicked})
            time.sleep(POST_CLICK_SETTLE_SEC)

    # 클릭 결과 요약
    success_count = sum(1 for r in click_results if r["clicked"])
    print(
        f"\n[INFO] 클릭 완료: {success_count}/{len(click_results)} 성공, "
        f"소요={format_elapsed_ms(script_started_at)}"
    )

    if success_count == 0:
        log_work2_event(
            component=COMPONENT_NAME,
            message="action_finished",
            log_name=LOG_NAME,
            click_results=click_results,
            login_verified=False,
            elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
        )
        return "click_failed"

    # 4. 로그인 성공 확인 — 메인 RCS 창 대기
    print(f"\n[INFO] === 로그인 성공 확인 ===")
    rcs_window, rcs_title, _rcs_backend = wait_for_rcs_main_window()

    login_verified = rcs_window is not None
    log_work2_event(
        component=COMPONENT_NAME,
        message="action_finished",
        log_name=LOG_NAME,
        click_results=click_results,
        login_verified=login_verified,
        rcs_main_title=rcs_title,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )

    if login_verified:
        print(
            f"[INFO] 로그인 성공 확인! title={rcs_title!r}, "
            f"총 소요={format_elapsed_ms(script_started_at)}"
        )
        return "success"

    print(
        f"[WARNING] 로그인 버튼 클릭 후 메인 창 미확인. "
        f"총 소요={format_elapsed_ms(script_started_at)}"
    )
    return "login_not_verified"


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
