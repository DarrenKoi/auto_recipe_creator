"""RCS에서 툴 화면 창이 열렸는지 확인하고 VLM으로 UI 요소를 분석한다 (Windows 전용).

환경 변수:
    RCS_TOOL_NAME                대상 툴명 (기본: MCD018)
    RCS_TOOL_SCREEN_TIMEOUT      창 탐색 대기 시간(초, 기본: 15)
    RCS_TOOL_SCREEN_SETTLE_SEC   폴링 간격(초, 기본: 0.5)
    RCS_TOOL_SCREEN_BACKENDS     pywinauto 백엔드 (기본: uia,win32)
    RCS_TOOL_SCREEN_ACTIVATE     감지 후 창 활성화 시도 여부 (기본: true)
    RCS_TOOL_SCREEN_VLM_ANALYZE  감지 후 VLM 화면 분석 여부 (기본: true)
    RCS_TOOL_SCREEN_DEBUG        디버그 모드 (기본: false)
"""

import os
import sys
import time
from dataclasses import dataclass

from pywinauto import Desktop

from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env

try:
    from poc.work.screen_capture import ScreenCapture
    from poc.work.vlm_screen_analysis import VLMScreenAnalyzer
    from poc.work.config import PocConfig
    VLM_ANALYSIS_AVAILABLE = True
except ImportError:
    VLM_ANALYSIS_AVAILABLE = False

DEFAULT_TOOL_NAME = "MCD018"
DEFAULT_TOOL_SCREEN_SETTLE_SEC = 0.5


@dataclass(frozen=True)
class ToolScreenSettings:
    tool_name: str
    timeout: float
    debug: bool
    check_interval: float
    backends: tuple[str, ...]
    activate_on_detect: bool
    vlm_analyze: bool


def load_settings() -> ToolScreenSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME
    raw_backends = [
        b.strip().lower()
        for b in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "uia,win32").split(",")
        if b.strip().lower() in {"win32", "uia"}
    ]
    backends = tuple(raw_backends) if raw_backends else ("uia", "win32")

    return ToolScreenSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        debug=env_flag("RCS_TOOL_SCREEN_DEBUG", False),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=backends,
        activate_on_detect=env_flag("RCS_TOOL_SCREEN_ACTIVATE", True),
        vlm_analyze=env_flag("RCS_TOOL_SCREEN_VLM_ANALYZE", True),
    )


def _title_matches(title: str, tool_name: str) -> bool:
    """창 제목에 툴명이 포함되어 있으면 매칭 (대소문자 무시)."""
    return tool_name.lower() in (title or "").lower()


def _try_activate(window, debug: bool) -> bool:
    """창을 포커스한다. 실패 시 False 반환."""
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


def _scan_once(settings: ToolScreenSettings) -> object | None:
    """모든 백엔드에서 visible 창을 스캔하여 툴명이 포함된 첫 번째 창을 반환한다."""
    for backend in settings.backends:
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True,
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
                print(
                    f"[INFO] 감지됨: backend={backend}, title={title!r}"
                )
                return win

    return None


def _capture_and_analyze_screen(window, settings: ToolScreenSettings) -> None:
    """툴 화면을 캡처하고 VLM으로 UI 요소를 분석한다."""
    if not VLM_ANALYSIS_AVAILABLE:
        print("[INFO] VLM 분석 모듈 미설치 — 화면 분석 건너뜀")
        return

    # 창 영역 가져오기
    try:
        rect = window.rectangle()
        left, top = rect.left, rect.top
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        print(f"[INFO] 창 영역: ({left}, {top}) {width}x{height}")
    except Exception as exc:
        print(f"[WARNING] 창 영역 가져오기 실패: {exc}")
        return

    if width <= 0 or height <= 0:
        print("[WARNING] 창 크기가 유효하지 않음 — 분석 건너뜀")
        return

    # 화면 캡처
    try:
        capture = ScreenCapture(output_dir=".")
        image_data = capture.capture_region(
            x=left, y=top, width=width, height=height, save=False,
        )
        capture.close()
    except Exception as exc:
        print(f"[WARNING] 화면 캡처 실패: {exc}")
        return

    if not image_data:
        print("[WARNING] 캡처 데이터 없음 — 분석 건너뜀")
        return

    # 디버그 이미지 저장 (JPEG — PNG 대비 파일 크기 절감)
    debug_path = "debug_tool_screen.jpg"
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_data))
        img.save(debug_path, format="JPEG", quality=85)
        print(f"[INFO] 디버그 스크린샷 저장: {debug_path}")
    except Exception as exc:
        if settings.debug:
            print(f"[DEBUG] 디버그 이미지 저장 실패: {exc}")

    # VLM 전송용 WebP 변환
    vlm_image_data = image_data
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_data))
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=90)
        vlm_image_data = buf.getvalue()
        print(f"[INFO] VLM 전송 이미지: PNG {len(image_data):,}B → WebP {len(vlm_image_data):,}B")
    except Exception as exc:
        print(f"[WARNING] WebP 변환 실패, 원본 PNG로 전송: {exc}")

    # VLM 분석
    try:
        config = PocConfig.load()
        if not config.vlm.api_url:
            print("[INFO] VLM API URL 미설정 — 화면 분석 건너뜀")
            return

        analyzer = VLMScreenAnalyzer(
            api_key=config.vlm.api_key,
            api_base_url=config.vlm.api_url,
            model_name=config.vlm.model_name,
        )

        print(f"[INFO] VLM 화면 분석 시작 (모델: {config.vlm.model_name})")
        result = analyzer.analyze_screen(vlm_image_data, task="state_recognition")

        if result is None:
            print("[WARNING] VLM 분석 결과 없음")
            return

        print(f"[INFO] 화면 상태: {result.state_name} (확신도: {result.confidence:.2f})")
        print(f"[INFO] 설명: {result.description}")

        if result.ui_elements:
            print(f"[INFO] 감지된 UI 요소 ({len(result.ui_elements)}개):")
            for elem in result.ui_elements:
                name = elem.get("name", "?")
                elem_type = elem.get("type", "?")
                location = elem.get("location", "?")
                print(f"  - {name} ({elem_type}) @ {location}")
        else:
            print("[INFO] 감지된 UI 요소 없음")

        if result.suggested_actions:
            print(f"[INFO] 제안 액션: {', '.join(result.suggested_actions)}")

        print(f"[INFO] VLM 분석 소요시간: {result.processing_time_ms:.0f}ms")

    except Exception as exc:
        print(f"[WARNING] VLM 분석 실패: {exc}")


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    print(f"[INFO] 감지 대상 툴: {settings.tool_name}")
    print(f"[INFO] 탐색 백엔드: {settings.backends}")
    print(f"[INFO] 타임아웃: {settings.timeout}초")

    deadline = time.time() + max(0.5, settings.timeout)
    logged_debug_once = False

    while time.time() < deadline:
        if settings.debug and not logged_debug_once:
            print(f"[DEBUG] 스캔 시작 — 툴명={settings.tool_name!r}")
            logged_debug_once = True

        window = _scan_once(settings)

        if window is not None:
            if settings.activate_on_detect:
                activated = _try_activate(window, settings.debug)
                if activated:
                    print("[INFO] 툴 화면 활성화 완료")
                else:
                    print("[WARNING] 툴 화면 활성화 실패")
            print("[INFO] 툴 화면이 열렸습니다.")

            if settings.vlm_analyze:
                _capture_and_analyze_screen(window, settings)

            return 0

        time.sleep(max(0.1, settings.check_interval))

    print(f"[ERROR] {settings.timeout:.1f}초 내에 툴 화면을 감지하지 못했습니다.")
    print(f"[INFO] 감지 대상 툴명: {settings.tool_name!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
