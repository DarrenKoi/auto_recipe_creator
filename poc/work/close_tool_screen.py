"""RCS에서 열린 툴 화면을 VLM으로 닫기 시도한다 (Windows 전용).

동작 개요:
1) 툴 창을 탐색/활성화
2) 창 스크린샷을 VLM에 전달해 좌상단 닫기(X) 좌표 추출
3) 단축키 없이 마우스로 X 클릭
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
    RCS_TOOL_CLOSE_USE_WEBP        VLM 전송 이미지를 WebP로 변환 (기본: true)
    RCS_TOOL_CLOSE_WEBP_QUALITY    WebP 품질 (기본: 90)
    SAFE_MODE                      RCS_TOOL_CLOSE_SAFE_MODE 미지정 시 기본값 소스
    VLM_API_URL/VLM_API_BASE_URL, VLM_API_KEY, VLM_MODEL_NAME, VLM_TEMPERATURE
"""

import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

from pywinauto import Desktop, mouse

from poc.work.config import PocConfig
from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env
from poc.work.screen_capture import ScreenCapture
from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

DEFAULT_TOOL_NAME = "MCD018"
DEFAULT_TOOL_SCREEN_SETTLE_SEC = 0.5
DEFAULT_CLOSE_VERIFY_TIMEOUT = 5.0
DEFAULT_VLM_MODEL = "Kimi-K2.5"
DEFAULT_VLM_TEMPERATURE = 0.0
DEFAULT_USE_WEBP = True
DEFAULT_WEBP_QUALITY = 90
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
    vlm_api_url: str
    vlm_api_key: str
    vlm_model: str
    vlm_temperature: float
    use_webp: bool
    webp_quality: int


def _resolve_safe_mode() -> bool:
    explicit = os.environ.get("RCS_TOOL_CLOSE_SAFE_MODE", "").strip()
    if explicit:
        return explicit.lower() in {"1", "true", "yes", "on", "y"}
    return env_flag("SAFE_MODE", True)


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARNING] 잘못된 {name} 값 '{value}', 기본값 {default} 사용")
        return default


def load_settings() -> ToolCloseSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME
    raw_backends = [
        b.strip().lower()
        for b in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "uia,win32").split(",")
        if b.strip().lower() in {"win32", "uia"}
    ]
    backends = tuple(raw_backends) if raw_backends else ("uia", "win32")

    config = PocConfig.load()
    vlm_api_url = (
        os.environ.get("VLM_API_URL", "").strip()
        or os.environ.get("VLM_API_BASE_URL", "").strip()
        or (config.vlm.api_url or "").strip()
    )
    vlm_api_key = os.environ.get("VLM_API_KEY", "").strip() or (config.vlm.api_key or "").strip()
    vlm_model = os.environ.get("VLM_MODEL_NAME", "").strip() or DEFAULT_VLM_MODEL

    return ToolCloseSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=backends,
        activate_on_detect=env_flag("RCS_TOOL_CLOSE_ACTIVATE", True),
        close_verify_timeout=env_float("RCS_TOOL_CLOSE_VERIFY_TIMEOUT", DEFAULT_CLOSE_VERIFY_TIMEOUT),
        debug=env_flag("RCS_TOOL_CLOSE_DEBUG", False),
        safe_mode=_resolve_safe_mode(),
        vlm_api_url=vlm_api_url,
        vlm_api_key=vlm_api_key,
        vlm_model=vlm_model,
        vlm_temperature=env_float("VLM_TEMPERATURE", DEFAULT_VLM_TEMPERATURE),
        use_webp=env_flag("RCS_TOOL_CLOSE_USE_WEBP", DEFAULT_USE_WEBP),
        webp_quality=_parse_int_env("RCS_TOOL_CLOSE_WEBP_QUALITY", DEFAULT_WEBP_QUALITY),
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


def _capture_window_png(window) -> bytes | None:
    try:
        rect = window.rectangle()
        left, top = rect.left, rect.top
        width = rect.right - rect.left
        height = rect.bottom - rect.top
    except Exception as exc:
        print(f"[WARNING] 창 영역 조회 실패: {exc}")
        return None

    if width <= 0 or height <= 0:
        print("[WARNING] 창 크기가 유효하지 않음")
        return None

    capture = ScreenCapture(output_dir=".")
    try:
        image_data = capture.capture_region(
            x=left,
            y=top,
            width=width,
            height=height,
            save=False,
        )
        return image_data
    except Exception as exc:
        print(f"[WARNING] 창 캡처 실패: {exc}")
        return None
    finally:
        capture.close()


def _build_close_prompt(width: int, height: int) -> tuple[str, str]:
    system_message = (
        "You detect a window close button in a GUI screenshot. "
        f"Image size is {width}x{height}. Respond ONLY in valid JSON."
    )
    prompt = "\n".join(
        [
            "Find the CLOSE button 'X' for this active tool window.",
            "Important: the close X is near the TOP-LEFT corner in this UI.",
            "Return ONE click point at the center of that X mark.",
            "x and y must be integer pixel coordinates inside the image.",
            "",
            "Return JSON only:",
            "{",
            '  "found": true,',
            '  "x": 0,',
            '  "y": 0,',
            '  "target_name": "close_x",',
            '  "confidence": 0.0',
            "}",
            "",
            "If not found, return:",
            "{",
            '  "found": false,',
            '  "target_name": "close_x"',
            "}",
        ]
    )
    return system_message, prompt


def _extract_json(text: str) -> dict:
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return json.loads(text[start:end].strip())
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return json.loads(text[start : end + 1])
    return json.loads(text)


def _to_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _debug_image_path(prefix: str = "close_point") -> Path:
    DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEBUG_IMAGE_DIR / f"{prefix}_{ts}.jpg"


def _save_close_point_debug_image(image_data: bytes, click_info: dict) -> None:
    """VLM 예측 클릭 좌표를 스크린샷에 마킹해 저장한다."""
    if not PIL_AVAILABLE:
        print("[WARNING] Pillow 미설치로 디버그 이미지 저장 불가")
        return

    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        print(f"[WARNING] 디버그 이미지 열기 실패: {exc}")
        return

    x = int(click_info["x"])
    y = int(click_info["y"])
    target_name = str(click_info.get("target_name", "close_x"))
    confidence = click_info.get("confidence_text", "N/A")
    width, height = image.size

    # 좌표를 이미지 경계 안으로 보정해 표시한다.
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    radius = 12
    color = "red"
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        outline=color,
        width=2,
    )
    label = f"{target_name} ({x},{y}) conf={confidence}"
    draw.text((x + radius + 4, max(0, y - 16)), label, fill=color, font=font)

    out_path = _debug_image_path("close_point")
    try:
        image.save(out_path, format="JPEG", quality=85)
        print(f"[INFO] 닫기 클릭포인트 디버그 이미지 저장: {out_path}")
    except Exception as exc:
        print(f"[WARNING] 디버그 이미지 저장 실패: {exc}")


MAX_VLM_BYTES = 1_000_000  # 1MB 상한


def _encode_vlm_image(image_data: bytes, settings: ToolCloseSettings) -> tuple[str, str]:
    """VLM 전송용 이미지 인코딩(WebP 우선, 1MB 상한, 불가 시 PNG)."""
    if settings.use_webp and PIL_AVAILABLE:
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            quality = max(1, min(100, settings.webp_quality))
            while quality >= 10:
                buf = BytesIO()
                image.save(buf, format="WEBP", quality=quality, method=4)
                payload = buf.getvalue()
                if len(payload) <= MAX_VLM_BYTES:
                    break
                print(f"[INFO] WebP {len(payload):,}B > 1MB, quality {quality} → {quality - 10}")
                quality -= 10
            else:
                payload = buf.getvalue()
            print(f"[INFO] VLM 이미지 포맷: webp (q={quality}), {len(payload):,}B")
            if len(payload) > MAX_VLM_BYTES:
                print(f"[WARNING] WebP가 여전히 {len(payload):,}B > 1MB — 리사이즈 적용")
                scale = (MAX_VLM_BYTES / len(payload)) ** 0.5
                new_w = int(image.width * scale)
                new_h = int(image.height * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)
                buf = BytesIO()
                image.save(buf, format="WEBP", quality=70, method=4)
                payload = buf.getvalue()
                print(f"[INFO] 리사이즈 후 WebP: {len(payload):,}B ({new_w}x{new_h})")
            return base64.b64encode(payload).decode("utf-8"), "image/webp"
        except Exception as exc:
            print(f"[WARNING] WebP 변환 실패, PNG 폴백: {exc}")
    elif settings.use_webp and not PIL_AVAILABLE:
        print("[WARNING] Pillow 미설치로 WebP 변환 불가, PNG 폴백")

    print(f"[INFO] VLM 이미지 포맷: png, {len(image_data):,}B")
    return base64.b64encode(image_data).decode("utf-8"), "image/png"


def _ask_close_point(image_data: bytes, width: int, height: int, settings: ToolCloseSettings) -> dict | None:
    if not settings.vlm_api_url:
        print("[ERROR] VLM API URL 미설정: VLM_API_URL 또는 VLM_API_BASE_URL 필요")
        return None

    image_b64, image_mime = _encode_vlm_image(image_data, settings)
    system_msg, prompt = _build_close_prompt(width, height)
    client = LangChainOpenAICompatibleVLMClient(
        base_url=settings.vlm_api_url,
        api_key=settings.vlm_api_key,
        timeout_sec=120.0,
    )
    request = ChatImageRequest(
        model=settings.vlm_model,
        system_message=system_msg,
        user_text=prompt,
        image_b64=image_b64,
        image_mime=image_mime,
        temperature=settings.vlm_temperature,
    )

    try:
        print(f"[INFO] VLM 호출: model={settings.vlm_model}, endpoint={client.endpoint}")
        started = time.time()
        raw = client.chat_with_image(request)
        elapsed_ms = (time.time() - started) * 1000
        print(f"[INFO] VLM 응답 수신 ({elapsed_ms:.0f}ms)")
        if settings.debug:
            print(f"[DEBUG] VLM raw:\n{raw}")
        data = _extract_json(raw)
    except Exception as exc:
        exc_text = str(exc)
        if "500" in exc_text or "server" in exc_text.lower():
            print(f"[WARNING] VLM 500 에러, 축소 이미지로 재시도: {exc}")
            try:
                retry_image = Image.open(BytesIO(image_data))
                if retry_image.mode != "RGB":
                    retry_image = retry_image.convert("RGB")
                new_w = retry_image.width // 2
                new_h = retry_image.height // 2
                retry_image = retry_image.resize((new_w, new_h), Image.LANCZOS)
                retry_buf = BytesIO()
                retry_image.save(retry_buf, format="WEBP", quality=60)
                retry_payload = retry_buf.getvalue()
                print(f"[INFO] 재시도 이미지: {len(retry_payload):,}B ({new_w}x{new_h})")
                retry_b64 = base64.b64encode(retry_payload).decode("utf-8")
                retry_system, retry_prompt = _build_close_prompt(new_w, new_h)
                retry_request = ChatImageRequest(
                    model=settings.vlm_model,
                    system_message=retry_system,
                    user_text=retry_prompt,
                    image_b64=retry_b64,
                    image_mime="image/webp",
                    temperature=settings.vlm_temperature,
                )
                started = time.time()
                raw = client.chat_with_image(retry_request)
                elapsed_ms = (time.time() - started) * 1000
                print(f"[INFO] 재시도 VLM 응답 수신 ({elapsed_ms:.0f}ms)")
                data = _extract_json(raw)
                # 좌표를 원본 크기로 스케일링
                rx = _to_int(data.get("x"))
                ry = _to_int(data.get("y"))
                if rx is not None and ry is not None:
                    data["x"] = int(rx * (width / new_w))
                    data["y"] = int(ry * (height / new_h))
                    print(f"[INFO] 좌표 스케일업: ({rx},{ry}) → ({data['x']},{data['y']})")
            except Exception as retry_exc:
                print(f"[ERROR] 재시도도 실패: {retry_exc}")
                return None
        else:
            print(f"[ERROR] VLM 호출/파싱 실패: {exc}")
            return None

    found = data.get("found")
    if isinstance(found, str):
        found_flag = found.strip().lower() in {"1", "true", "yes", "on"}
    else:
        found_flag = bool(found)
    if not found_flag:
        print("[WARNING] VLM이 close X를 찾지 못했다고 응답함")
        return None

    x = _to_int(data.get("x"))
    y = _to_int(data.get("y"))
    if x is None or y is None:
        print(f"[WARNING] VLM 좌표 파싱 실패: x={data.get('x')!r}, y={data.get('y')!r}")
        return None

    if not (0 <= x < width and 0 <= y < height):
        print(f"[WARNING] VLM 좌표가 창 범위를 벗어남: ({x}, {y}) not in {width}x{height}")
        return None

    target_name = str(data.get("target_name", "close_x")).strip() or "close_x"
    confidence = data.get("confidence")
    try:
        confidence_text = f"{float(confidence):.2f}"
    except Exception:
        confidence_text = "N/A"
    print(f"[INFO] VLM close 좌표: ({x}, {y}), target={target_name}, confidence={confidence_text}")
    return {"x": x, "y": y, "target_name": target_name, "confidence_text": confidence_text}


def _click_close(window, rel_x: int, rel_y: int, settings: ToolCloseSettings) -> bool:
    rect = window.rectangle()
    abs_x = max(rect.left, min(rect.left + rel_x, rect.right - 1))
    abs_y = max(rect.top, min(rect.top + rel_y, rect.bottom - 1))
    rel_x = abs_x - rect.left
    rel_y = abs_y - rect.top
    print(f"[INFO] 닫기 X 클릭 좌표: rel=({rel_x}, {rel_y}), abs=({abs_x}, {abs_y})")

    if settings.safe_mode:
        print("[INFO] SAFE MODE 활성화: 실제 클릭은 수행하지 않음")
        return True

    for attempt in range(1, 3):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] click_input 실패 (attempt={attempt}): {exc}")

        try:
            mouse.move(coords=(abs_x, abs_y))
            time.sleep(0.05)
            mouse.press(button="left", coords=(abs_x, abs_y))
            time.sleep(0.03)
            mouse.release(button="left", coords=(abs_x, abs_y))
            print(f"[INFO] mouse press/release 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] mouse press/release 실패 (attempt={attempt}): {exc}")

        time.sleep(0.2)

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
    print(f"[INFO] VLM 모델: {settings.vlm_model}")
    print(f"[INFO] VLM 이미지 WebP 사용: {settings.use_webp}")

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
    image_data = _capture_window_png(window)
    if not image_data:
        print("[ERROR] 툴 화면 캡처 실패")
        return 3

    rect = window.rectangle()
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    close_point = _ask_close_point(image_data, width, height, settings)
    if not close_point:
        print("[ERROR] VLM이 닫기 좌표를 제공하지 못했습니다.")
        return 4

    _save_close_point_debug_image(image_data, close_point)

    clicked = _click_close(window, close_point["x"], close_point["y"], settings)
    if not clicked:
        print("[ERROR] 닫기 클릭 수행 실패")
        return 5

    if _verify_closed(initial_handle, settings):
        print("[INFO] 툴 화면 닫힘 확인 완료")
        return 0

    print("[WARNING] 클릭 후에도 툴 화면이 남아있습니다.")
    return 6


if __name__ == "__main__":
    sys.exit(main())
