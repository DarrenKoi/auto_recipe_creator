"""RCS에서 열린 툴 화면을 타이틀바 우클릭 컨텍스트 메뉴로 닫는다 (Windows 전용).

동작 개요:
1) 툴 창을 탐색/활성화
2) 타이틀바 우클릭 → 시스템 컨텍스트 메뉴 열기
3) "X Close" 항목 클릭
4) 창이 닫혔는지 재확인

환경 변수:
    RCS_TOOL_NAME                  대상 툴명 (기본: MCD018)
    RCS_TOOL_SCREEN_TIMEOUT        창 탐색 대기 시간(초, 기본: 15)
    RCS_TOOL_SCREEN_SETTLE_SEC     폴링 간격(초, 기본: 0.5)
    RCS_TOOL_SCREEN_BACKENDS       pywinauto 백엔드 (기본: uia,win32)
    RCS_TOOL_CLOSE_ACTIVATE        감지 후 창 활성화 시도 여부 (기본: true)
    RCS_TOOL_CLOSE_VERIFY_TIMEOUT  클릭 후 닫힘 확인 시간(초, 기본: 5)
    RCS_TOOL_CLOSE_DEBUG           디버그 모드 (기본: false)
    RCS_TOOL_CLOSE_SAFE_MODE       true면 실제 클릭하지 않음
    SAFE_MODE                      RCS_TOOL_CLOSE_SAFE_MODE 미지정 시 기본값 소스
"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pywinauto import Desktop, mouse

from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env

DEFAULT_TOOL_NAME = "MCD018"
DEFAULT_TOOL_SCREEN_SETTLE_SEC = 0.5
DEFAULT_CLOSE_VERIFY_TIMEOUT = 5.0
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"


@dataclass(frozen=True)
class ToolCloseSettings:
    tool_name: str
    timeout: float
    check_interval: float
    backends: tuple[str, ...]
    activate_on_detect: bool
    close_verify_timeout: float
    debug: bool
    safe_mode: bool


def _resolve_safe_mode() -> bool:
    explicit = os.environ.get("RCS_TOOL_CLOSE_SAFE_MODE", "").strip()
    if explicit:
        return explicit.lower() in {"1", "true", "yes", "on", "y"}
    return env_flag("SAFE_MODE", True)


def load_settings() -> ToolCloseSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME
    raw_backends = [
        b.strip().lower()
        for b in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "uia,win32").split(",")
        if b.strip().lower() in {"win32", "uia"}
    ]
    backends = tuple(raw_backends) if raw_backends else ("uia", "win32")

    return ToolCloseSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=backends,
        activate_on_detect=env_flag("RCS_TOOL_CLOSE_ACTIVATE", True),
        close_verify_timeout=env_float("RCS_TOOL_CLOSE_VERIFY_TIMEOUT", DEFAULT_CLOSE_VERIFY_TIMEOUT),
        debug=env_flag("RCS_TOOL_CLOSE_DEBUG", False),
        safe_mode=_resolve_safe_mode(),
    )


def _title_matches(title: str, tool_name: str) -> bool:
    return tool_name.lower() in (title or "").lower()


def _try_activate(window, debug: bool) -> bool:
    try:
        if hasattr(window, "is_minimized") and window.is_minimized():
            window.restore()
            time.sleep(0.1)
    except Exception:
        pass

    try:
        window.set_focus()
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] set_focus 실패: {exc}")

    try:
        window.click_input(coords=(100, 18), button="left")
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] click_input 폴백 실패: {exc}")
        return False


def _scan_once(settings: ToolCloseSettings, log_on_match: bool = True):
    for backend in settings.backends:
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True,
                visible_only=True,
            )
        except Exception as exc:
            if settings.debug:
                print(f"[DEBUG] backend={backend} 창 조회 실패: {exc}")
            continue

        for win in windows:
            try:
                title = win.window_text() or ""
            except Exception:
                continue

            if settings.debug:
                print(f"[DEBUG] backend={backend} title={title!r}")

            if _title_matches(title, settings.tool_name):
                if log_on_match:
                    print(f"[INFO] 감지됨: backend={backend}, title={title!r}")
                return win
    return None


def _save_close_click_debug_image(
    window,
    title_click: tuple[int, int],
    close_click: tuple[int, int],
) -> None:
    """닫기 클릭 좌표를 창 스크린샷에 표시해 JPEG로 저장한다."""
    try:
        import mss
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"[WARNING] 디버그 이미지 의존성 로드 실패: {exc}")
        return

    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[WARNING] 디버그 이미지용 창 영역 조회 실패: {exc}")
        return

    region_w = rect.right - rect.left
    region_h = rect.bottom - rect.top
    if region_w <= 0 or region_h <= 0:
        print("[WARNING] 디버그 이미지 저장 건너뜀: 창 크기가 유효하지 않음")
        return

    try:
        with mss.mss() as sct:
            shot = sct.grab(
                {
                    "left": rect.left,
                    "top": rect.top,
                    "width": region_w,
                    "height": region_h,
                }
            )
            image = Image.frombytes("RGB", shot.size, shot.rgb)
    except Exception as exc:
        print(f"[WARNING] 디버그 스크린샷 캡처 실패: {exc}")
        return

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    def _mark(point: tuple[int, int], color: str, label: str) -> None:
        px = max(0, min(point[0] - rect.left, region_w - 1))
        py = max(0, min(point[1] - rect.top, region_h - 1))
        radius = 10
        draw.line([(px - radius, py), (px + radius, py)], fill=color, width=2)
        draw.line([(px, py - radius), (px, py + radius)], fill=color, width=2)
        draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)], outline=color, width=2)
        draw.text((px + radius + 4, py - radius - 2), f"{label} ({px},{py})", fill=color, font=font)

    _mark(title_click, "cyan", "titlebar_right_click")
    _mark(close_click, "orange", "close_menu_click")

    DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DEBUG_IMAGE_DIR / f"tool_close_{ts}_click_points.jpg"
    try:
        image.save(out_path, format="JPEG", quality=85)
        print(f"[INFO] 닫기 클릭 좌표 디버그 이미지 저장: {out_path}")
    except Exception as exc:
        print(f"[WARNING] 디버그 이미지 저장 실패: {exc}")


def _close_via_context_menu(window, settings: ToolCloseSettings) -> bool:
    """타이틀바 우클릭 → 시스템 컨텍스트 메뉴 → 'X Close' 클릭으로 창을 닫는다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[ERROR] 창 영역 조회 실패: {exc}")
        return False

    # 타이틀바 중앙 지점 계산 (상단에서 ~15px 아래)
    title_x = (rect.left + rect.right) // 2
    title_y = rect.top + 15
    print(f"[INFO] 타이틀바 우클릭 좌표: ({title_x}, {title_y})")

    # 컨텍스트 메뉴의 "X Close" 항목 예상 좌표
    # 우클릭 지점에서 약간 오른쪽 아래로 이동하면 "X Close" 항목 위치
    close_x = title_x + 30
    close_y = title_y + 10
    print(f"[INFO] 'X Close' 메뉴 클릭 좌표: ({close_x}, {close_y})")
    _save_close_click_debug_image(
        window,
        title_click=(title_x, title_y),
        close_click=(close_x, close_y),
    )

    if settings.safe_mode:
        print("[INFO] SAFE MODE 활성화: 실제 클릭은 수행하지 않음")
        return True

    # 타이틀바 우클릭으로 시스템 메뉴 열기
    try:
        mouse.right_click(coords=(title_x, title_y))
    except Exception as exc:
        print(f"[ERROR] 타이틀바 우클릭 실패: {exc}")
        return False

    time.sleep(0.3)

    try:
        mouse.click(coords=(close_x, close_y), button="left")
        print("[INFO] 'X Close' 클릭 완료")
        return True
    except Exception as exc:
        print(f"[ERROR] 'X Close' 클릭 실패: {exc}")
        return False


def _window_handle(window) -> int | None:
    try:
        handle = window.handle
        if isinstance(handle, int):
            return handle
    except Exception:
        return None
    return None


def _verify_closed(initial_handle: int | None, settings: ToolCloseSettings) -> bool:
    if settings.safe_mode:
        print("[INFO] SAFE MODE에서는 닫힘 검증을 건너뜀")
        return True

    deadline = time.time() + max(0.5, settings.close_verify_timeout)
    while time.time() < deadline:
        current = _scan_once(settings, log_on_match=False)
        if current is None:
            return True

        if initial_handle is not None:
            current_handle = _window_handle(current)
            if current_handle is not None and current_handle != initial_handle:
                return True

        time.sleep(max(0.1, settings.check_interval))

    return False


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    print(f"[INFO] 감지 대상 툴: {settings.tool_name}")
    print(f"[INFO] 탐색 백엔드: {settings.backends}")
    print(f"[INFO] SAFE MODE: {settings.safe_mode}")

    deadline = time.time() + max(0.5, settings.timeout)
    window = None
    while time.time() < deadline:
        window = _scan_once(settings)
        if window is not None:
            break
        time.sleep(max(0.1, settings.check_interval))

    if window is None:
        print(f"[ERROR] {settings.timeout:.1f}초 내에 툴 화면을 감지하지 못했습니다.")
        return 2

    if settings.activate_on_detect:
        if _try_activate(window, settings.debug):
            print("[INFO] 툴 화면 활성화 완료")
        else:
            print("[WARNING] 툴 화면 활성화 실패")

    initial_handle = _window_handle(window)

    closed = _close_via_context_menu(window, settings)
    if not closed:
        print("[ERROR] 컨텍스트 메뉴를 통한 닫기 수행 실패")
        return 5

    if _verify_closed(initial_handle, settings):
        print("[INFO] 툴 화면 닫힘 확인 완료")
        return 0

    print("[WARNING] 클릭 후에도 툴 화면이 남아있습니다.")
    return 6


if __name__ == "__main__":
    sys.exit(main())
