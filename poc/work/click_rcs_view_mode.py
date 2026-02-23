"""RCS 메인 화면에서 View/List 탭을 VLM으로 검출하고 View 탭을 클릭하는 스크립트 (Windows 전용).

이미 로그인된 RCS 메인 창을 데스크톱에서 찾아 스크린샷을 캡처한 뒤,
VLM에 View/List 탭 좌표를 요청하고, 디버그 이미지를 저장한 다음 View 탭을 클릭한다.
"""

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import mss
import mss.tools
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pywinauto import mouse
from pywinauto import Desktop

try:
    from .vlm_openai_client import ChatImageRequest, LangChainOpenAIVLMClient
    from .prompts import build_rcs_main_tab_locator_prompt
except ImportError:
    from vlm_openai_client import ChatImageRequest, LangChainOpenAIVLMClient
    from prompts import build_rcs_main_tab_locator_prompt

load_dotenv()

# ─────────────────────────── 설정 ───────────────────────────

VLM_API_URL = (
    os.environ.get("VLM_API_URL", "").strip()
    or os.environ.get("VLM_API_BASE_URL", "").strip()
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "win32"
MAIN_WINDOW_TITLE_REGEX = (
    os.environ.get("RCS_MAIN_WINDOW_REGEX", r"\brcs\b.*\[server\s*:[^\]]+\]").strip()
    or r"\brcs\b.*\[server\s*:[^\]]+\]"
)
DEBUG_MAIN_WINDOW_TITLES = (
    os.environ.get("RCS_DEBUG_MAIN_WINDOW_TITLES", "0").strip().lower()
    not in {"0", "false", "no", "off"}
)
_desktop_backends_raw = [
    item.strip().lower()
    for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "win32,uia").split(",")
    if item.strip()
]
_desktop_backends = _desktop_backends_raw + [PYWINAUTO_BACKEND]
DESKTOP_SCAN_BACKENDS = tuple(
    dict.fromkeys(b for b in _desktop_backends if b in {"uia", "win32"})
) or ("uia", "win32")

VLM_MODEL = os.environ.get("VLM_MODEL_NAME", "Kimi-K2.5").strip() or "Kimi-K2.5"

TARGET_ELEMENTS = ["view_tab", "list_tab"]
TARGET_CLICK_KEY = "view_tab"
ELEMENT_COLORS = {
    "view_tab": "orange",
    "list_tab": "cyan",
}

VLM_CLIENT = LangChainOpenAIVLMClient(
    base_url=VLM_API_URL,
    api_key=VLM_API_KEY,
    timeout_sec=120.0,
)

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0

try:
    CLICK_RETRY_COUNT = int(os.getenv("RCS_CLICK_RETRY_COUNT", "2"))
except ValueError:
    CLICK_RETRY_COUNT = 2

try:
    CLICK_RETRY_DELAY_SEC = float(os.getenv("RCS_CLICK_RETRY_DELAY_SEC", "0.25"))
except ValueError:
    CLICK_RETRY_DELAY_SEC = 0.25


# ─────────────────────────── 창 탐색 ───────────────────────────


def _is_main_window_title(title: str) -> bool:
    """메인 RCS 창 제목인지 정규식으로 판별한다."""
    try:
        return re.search(MAIN_WINDOW_TITLE_REGEX, title, flags=re.IGNORECASE) is not None
    except re.error:
        t = title.lower()
        return "rcs" in t and "[server" in t


def _scan_window_list(windows, source_name: str, debug_rows: list[str]):
    """창 목록을 순회하며 메인 RCS 창을 찾는다."""
    for idx, win in enumerate(windows, start=1):
        try:
            title = win.window_text() or ""
        except Exception as exc:
            debug_rows.append(f"{source_name}[{idx}] title-read-error={exc}")
            continue

        matched = _is_main_window_title(title)
        debug_rows.append(f"{source_name}[{idx}] matched={matched} title={title!r}")
        if matched:
            return win, title

    return None, ""


def _find_existing_main_window():
    """이미 떠 있는 메인 RCS 창(기 로그인 상태)을 데스크톱 전체에서 탐색한다."""
    debug_rows = []
    for backend in DESKTOP_SCAN_BACKENDS:
        try:
            desktop_windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True
            )
        except Exception as exc:
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        main_window, main_title = _scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows
        )
        if main_window is not None:
            return main_window, main_title, debug_rows

    return None, "", debug_rows


# ─────────────────────────── 스크린샷 ───────────────────────────


def _capture_window(window) -> Image.Image:
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
    rect = window.rectangle()
    region = {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_data))
    print(f"[INFO] 창 캡처 완료: {image.size[0]}x{image.size[1]} px")
    return image


# ─────────────────────────── VLM 호출 ───────────────────────────


def _encode_image(image: Image.Image) -> tuple[str, int, int]:
    """PIL Image를 base64 PNG로 인코딩한다."""
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    w, h = image.size
    print(f"[INFO] 이미지 인코딩: {w}x{h}, {len(buf.getvalue()) / 1024:.1f}KB")
    return b64, w, h


def _extract_json(text: str) -> dict:
    """VLM 응답 텍스트에서 JSON을 추출한다."""
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        if e != -1:
            return json.loads(text[s:e].strip())
    if "{" in text:
        s = text.find("{")
        e = text.rfind("}")
        if e > s:
            return json.loads(text[s : e + 1])
    return json.loads(text)


def _parse_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """VLM 응답 좌표를 정수로 변환하고 로그를 출력한다."""
    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"  [MISS] {key:20s} — VLM 응답에 없음")
            continue
        raw_x, raw_y = pt.get("x", 0), pt.get("y", 0)
        x, y = int(raw_x), int(raw_y)
        data[key] = {"x": x, "y": y}
        out = ""
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            out = " ← OUT OF BOUNDS"
        print(f"  [RAW ] {key:20s} — raw=({raw_x}, {raw_y}) → px=({x}, {y}){out}")
    return data


# ─────────────────────────── 클릭 ───────────────────────────


def _click_at(element_key: str, window, elements: dict) -> bool:
    """VLM 좌표를 스크린 좌표로 변환해 요소를 클릭한다."""
    pt = elements.get(element_key)
    if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
        print(f"[ERROR] 클릭 대상 '{element_key}' 좌표가 없습니다.")
        return False

    rect = window.rectangle()
    x = int(pt["x"]) + rect.left
    y = int(pt["y"]) + rect.top
    x = max(rect.left, min(x, rect.right - 1))
    y = max(rect.top, min(y, rect.bottom - 1))
    rel_x = max(0, min(int(pt["x"]), rect.right - rect.left - 1))
    rel_y = max(0, min(int(pt["y"]), rect.bottom - rect.top - 1))

    print(f"[INFO] '{element_key}' 클릭: screen=({x}, {y})")
    for attempt in range(1, max(1, CLICK_RETRY_COUNT) + 1):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] click_input 실패 (attempt={attempt}): {exc}")

        try:
            mouse.move(coords=(x, y))
            time.sleep(0.08)
            mouse.press(button="left", coords=(x, y))
            time.sleep(0.05)
            mouse.release(button="left", coords=(x, y))
            print(f"[INFO] mouse press/release 실행 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] mouse press/release 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, CLICK_RETRY_DELAY_SEC))

    return False


# ─────────────────────────── 디버그 이미지 ───────────────────────────


def _save_marked_image(
    image: Image.Image, elements: dict, colors: dict, filename: str
) -> None:
    """좌표를 원본 스크린샷 위에 십자선+원으로 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    r = 12
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        x, y = int(pt["x"]), int(pt["y"])
        color = colors.get(name, "white")
        draw.line([(x - r, y), (x + r, y)], fill=color, width=2)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=2)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=2)
        label = f"{name} ({x},{y})"
        draw.text((x + r + 3, y - 16), label, fill=color, font=font)

    out_path = Path(__file__).parent / filename
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


# ─────────────────────────── 메인 ───────────────────────────


def main() -> int:
    """메인 RCS 창을 찾아 View 탭을 VLM 좌표로 클릭한다."""
    print("[INFO] RCS 메인 창에서 View/List 탭 검출 시작")

    main_window, main_title, debug_rows = _find_existing_main_window()
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")
        if not debug_rows:
            print("[DEBUG] no visible top-level windows")
        else:
            for row in debug_rows:
                print(f"[DEBUG] {row}")

    if main_window is None:
        print("[ERROR] 로그인된 RCS 메인 창을 찾을 수 없습니다.")
        return 1

    print(f"[INFO] RCS 메인 창 발견: '{main_title}'")

    # 스크린샷 캡처
    image = _capture_window(main_window)
    img_b64, w, h = _encode_image(image)

    rect = main_window.rectangle()
    print(
        f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
        f"size={rect.right - rect.left}x{rect.bottom - rect.top}"
    )

    # VLM 호출
    system_msg, prompt = build_rcs_main_tab_locator_prompt(
        width=w,
        height=h,
        target_keys=TARGET_ELEMENTS,
    )

    try:
        request = ChatImageRequest(
            model=VLM_MODEL,
            system_message=system_msg,
            user_text=prompt,
            image_b64=img_b64,
            temperature=VLM_TEMPERATURE,
        )
        print(f"[INFO] VLM 호출: model={VLM_MODEL}, endpoint={VLM_CLIENT.endpoint}")
        start = time.time()
        raw = VLM_CLIENT.chat_with_image(request)
        elapsed = (time.time() - start) * 1000
        print(f"[INFO] 응답 수신 ({elapsed:.0f}ms)")
        print(f"[INFO] 원문 응답:\n{raw}\n")
    except Exception as exc:
        print(f"[ERROR] VLM 호출 실패: {exc}")
        return 2

    # 좌표 파싱
    try:
        data = _extract_json(raw)
        print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
        data = _parse_coords(data, TARGET_ELEMENTS, w, h)
    except Exception as exc:
        print(f"[ERROR] JSON 파싱 실패: {exc}")
        return 3

    detected = sum(1 for k in TARGET_ELEMENTS if k in data and isinstance(data[k], dict))
    print(f"[INFO] 검출률: {detected}/{len(TARGET_ELEMENTS)}")

    # 디버그 이미지 저장
    _save_marked_image(image, data, ELEMENT_COLORS, "debug_view_mode.png")

    # View 탭 클릭
    if not _click_at(TARGET_CLICK_KEY, main_window, data):
        print(f"[ERROR] '{TARGET_CLICK_KEY}' 클릭 실패")
        return 4

    print(f"[INFO] '{TARGET_CLICK_KEY}' 클릭 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
