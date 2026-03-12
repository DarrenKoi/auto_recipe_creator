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
from datetime import datetime
import io
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pywinauto import Desktop

from poc.work.config import PocConfig
from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env
from poc.work.screen_capture import ScreenCapture
from poc.work2 import debug_image_path
from poc.work2.flask_vlm import apply_pipeline_env_defaults
from poc.work2.vlm_screen_analysis import VLMScreenAnalyzer

DEFAULT_TOOL_NAME = "MCD018"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
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
    apply_pipeline_env_defaults()
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


def _to_int_coordinate(value, axis_size: int) -> int | None:
    """숫자/문자/정규화 좌표를 픽셀 정수로 변환한다."""
    if axis_size <= 0 or value is None or isinstance(value, bool):
        return None

    numeric: float | None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("%"):
                numeric = (float(text[:-1]) / 100.0) * (axis_size - 1)
            else:
                numeric = float(text)
        except ValueError:
            return None

    if 0.0 < numeric < 1.0:
        numeric = numeric * (axis_size - 1)

    coord = int(round(numeric))
    return max(0, min(coord, axis_size - 1))


def _extract_click_point(element: dict, image_w: int, image_h: int) -> tuple[int, int] | None:
    """UI 요소 JSON에서 클릭 좌표(x,y)를 추출한다."""
    if not isinstance(element, dict):
        return None

    x_raw = element.get("x")
    y_raw = element.get("y")
    if isinstance(element.get("click_point"), dict):
        click_point = element["click_point"]
        x_raw = click_point.get("x", x_raw)
        y_raw = click_point.get("y", y_raw)

    x = _to_int_coordinate(x_raw, image_w)
    y = _to_int_coordinate(y_raw, image_h)
    if x is None or y is None:
        return None
    return x, y


def _save_click_point_overlay(
    image_data: bytes,
    ui_elements: list[dict],
    image_w: int,
    image_h: int,
    out_path: Path,
) -> int:
    """VLM이 반환한 좌표를 스크린샷에 오버레이하여 JPEG로 저장한다."""
    image = Image.open(BytesIO(image_data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("lime", "cyan", "yellow", "orange", "red", "magenta")

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    marked_count = 0
    radius = 10
    for idx, elem in enumerate(ui_elements, start=1):
        if not isinstance(elem, dict):
            continue

        point = _extract_click_point(elem, image_w, image_h)
        if point is None:
            continue
        x, y = point

        color = colors[(idx - 1) % len(colors)]
        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)
        name = str(elem.get("name", f"elem_{idx}")).strip() or f"elem_{idx}"
        label = f"{idx}. {name} ({x},{y})"
        draw.text((x + radius + 4, y - radius - 2), label, fill=color, font=font)
        marked_count += 1

    if marked_count == 0:
        return 0

    image.save(out_path, format="JPEG", quality=85)
    return marked_count


def _capture_and_analyze_screen(window, settings: ToolScreenSettings) -> None:
    """툴 화면을 캡처하고 VLM으로 UI 요소를 분석한다."""
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

    pipeline_config = apply_pipeline_env_defaults()
    debug_model_name = str(
        pipeline_config.get("primary_model_name")
        or os.environ.get("VLM_MODEL_NAME")
        or ""
    ).strip()

    # 디버그 이미지 저장 (JPEG) + VLM 전송용 WebP 변환
    jpeg_size = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"tool_screen_{ts}.jpg",
        model_name=debug_model_name,
    )
    try:
        img = Image.open(io.BytesIO(image_data))
        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format="JPEG", quality=85)
        jpeg_size = jpeg_buf.tell()
        with open(debug_path, "wb") as f:
            f.write(jpeg_buf.getvalue())
        print(f"[INFO] 디버그 스크린샷 저장: {debug_path} ({jpeg_size:,}B)")
    except Exception as exc:
        if settings.debug:
            print(f"[DEBUG] 디버그 이미지 저장 실패: {exc}")

    vlm_image_data = image_data
    vlm_image_w = width
    vlm_image_h = height
    MAX_VLM_BYTES = 1_000_000  # 1MB 상한
    try:
        img = Image.open(io.BytesIO(image_data))
        vlm_image_w, vlm_image_h = img.size
        quality = 90
        while quality >= 10:
            webp_buf = io.BytesIO()
            img.save(webp_buf, format="WEBP", quality=quality)
            vlm_image_data = webp_buf.getvalue()
            webp_size = len(vlm_image_data)
            vlm_image_w, vlm_image_h = img.size
            if webp_size <= MAX_VLM_BYTES:
                break
            print(f"[INFO] WebP {webp_size:,}B > 1MB, quality {quality} → {quality - 10}")
            quality -= 10
        print(f"[INFO] VLM 전송 이미지: JPEG {jpeg_size:,}B → WebP {webp_size:,}B (q={quality})")
        if webp_size > MAX_VLM_BYTES:
            print(f"[WARNING] WebP가 여전히 {webp_size:,}B > 1MB — 리사이즈 적용")
            scale = (MAX_VLM_BYTES / webp_size) ** 0.5
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            webp_buf = io.BytesIO()
            img.save(webp_buf, format="WEBP", quality=70)
            vlm_image_data = webp_buf.getvalue()
            vlm_image_w, vlm_image_h = img.size
            print(f"[INFO] 리사이즈 후 WebP: {len(vlm_image_data):,}B ({new_w}x{new_h})")
    except Exception as exc:
        print(f"[WARNING] WebP 변환 실패, 원본 PNG로 전송: {exc}")

    # VLM 분석 (500 에러 시 축소 이미지로 재시도)
    try:
        config = PocConfig.load()
        if not config.vlm.api_url:
            print("[INFO] VLM API URL 미설정 — 화면 분석 건너뜀")
            return

        print(
            f"[INFO] pipeline: primary={pipeline_config['primary_service']} "
            f"({config.vlm.model_name}) -> ocr={pipeline_config['ocr_service']} "
            f"({pipeline_config['ocr_model_name']})"
        )
        print(
            f"[INFO] primary endpoint={config.vlm.api_url}, "
            f"ocr endpoint={pipeline_config['ocr_api_url']}"
        )

        analyzer = VLMScreenAnalyzer(
            api_key=config.vlm.api_key,
            api_base_url=config.vlm.api_url,
            model_name=config.vlm.model_name,
            pipeline_config=pipeline_config,
        )

        print(f"[INFO] VLM 화면 분석 시작 (모델: {config.vlm.model_name})")
        result = None
        try:
            result = analyzer.analyze_screen(
                vlm_image_data,
                task="state_recognition",
                image_width=vlm_image_w,
                image_height=vlm_image_h,
                ocr_context_label="RCS tool screen",
                ocr_focus_words=(settings.tool_name,),
            )
        except Exception as api_exc:
            exc_text = str(api_exc)
            if "500" in exc_text or "server" in exc_text.lower():
                print(f"[WARNING] VLM 500 에러, 축소 이미지로 재시도: {api_exc}")
                try:
                    img = Image.open(io.BytesIO(image_data))
                    new_w = img.width // 2
                    new_h = img.height // 2
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    retry_buf = io.BytesIO()
                    img.save(retry_buf, format="WEBP", quality=60)
                    retry_data = retry_buf.getvalue()
                    print(f"[INFO] 재시도 이미지: {len(retry_data):,}B ({new_w}x{new_h})")
                    result = analyzer.analyze_screen(
                        retry_data,
                        task="state_recognition",
                        image_width=new_w,
                        image_height=new_h,
                        ocr_context_label="RCS tool screen",
                        ocr_focus_words=(settings.tool_name,),
                    )
                    vlm_image_data = retry_data
                    vlm_image_w, vlm_image_h = new_w, new_h
                except Exception as retry_exc:
                    print(f"[ERROR] 재시도도 실패: {retry_exc}")
            else:
                print(f"[WARNING] VLM 호출 실패: {api_exc}")

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
                point = _extract_click_point(elem, vlm_image_w, vlm_image_h)
                if point is None:
                    print(f"  - {name} ({elem_type}) @ {location}")
                else:
                    print(f"  - {name} ({elem_type}) @ {location} -> click=({point[0]}, {point[1]})")

            overlay_path = debug_image_path(
                DEBUG_IMAGE_DIR,
                f"tool_screen_{ts}_vlm_points.jpg",
                model_name=config.vlm.model_name,
            )
            marked = _save_click_point_overlay(
                image_data=vlm_image_data,
                ui_elements=result.ui_elements,
                image_w=vlm_image_w,
                image_h=vlm_image_h,
                out_path=overlay_path,
            )
            if marked > 0:
                print(f"[INFO] VLM 좌표 오버레이 저장: {overlay_path} ({marked} points)")
            else:
                print("[INFO] 좌표(x,y)가 포함된 UI 요소가 없어 오버레이를 건너뜀")
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
