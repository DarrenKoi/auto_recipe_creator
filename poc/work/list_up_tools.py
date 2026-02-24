"""RCS List 탭에서 특정 툴(MCD018) 좌표를 찾아 클릭한다 (Windows 전용).

기본 동작:
1) RCS 메인 창을 foreground/focus로 준비
2) List 화면에서 대상 툴 이름 좌표를 추출
3) 좌표를 기준으로 클릭하여 툴 화면으로 진입
"""

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

import mss
import mss.tools
from PIL import Image, ImageDraw, ImageFont
from pywinauto import Desktop, mouse

from poc.work.prompts import (
    build_rcs_select_tool_prompt,
)
from poc.work.rcs_common import (
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_TITLE_REGEX,
    TOOL_CONTAINER_ORDER,
    _is_visible,
    env_flag,
    env_float,
    load_env,
)
from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient

DEFAULT_MAIN_WINDOW_REGEX = r"\brcs\b.*\[server\s*:[^\]]+\]"
DEFAULT_LIST_SETTLE_SEC = 0.60
DEFAULT_CLICK_RETRY_COUNT = 2
DEFAULT_CLICK_RETRY_DELAY_SEC = 0.25
DEFAULT_VLM_MODEL = "Kimi-K2.5"
DEFAULT_VLM_TEMPERATURE = 0.0
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
TARGET_TOOL_NAME = "MCD018"


@dataclass(frozen=True)
class ListToolsSettings:
    window_title_regex: str
    timeout: float
    debug_main_window_titles: bool
    debug_tree: bool
    list_settle_sec: float
    click_retry_count: int
    click_retry_delay_sec: float
    vlm_api_url: str
    vlm_api_key: str
    vlm_model: str
    vlm_temperature: float


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARNING] 잘못된 {name} 값 '{value}', 기본값 {default} 사용")
        return default


def _debug_image_path(filename: str) -> Path:
    DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_IMAGE_DIR / filename


def load_settings() -> ListToolsSettings:
    """환경 변수 기반 설정을 로드한다."""
    load_env()

    main_regex = (
        os.environ.get("RCS_MAIN_WINDOW_REGEX", "").strip()
        or os.environ.get("RCS_WINDOW_TITLE", "").strip()
        or DEFAULT_MAIN_WINDOW_REGEX
        or DEFAULT_WINDOW_TITLE_REGEX
    )
    vlm_api_url = (
        os.environ.get("VLM_API_URL", "").strip()
        or os.environ.get("VLM_API_BASE_URL", "").strip()
    )
    return ListToolsSettings(
        window_title_regex=main_regex,
        timeout=env_float("RCS_TIMEOUT", DEFAULT_TIMEOUT),
        debug_main_window_titles=env_flag("RCS_DEBUG_MAIN_WINDOW_TITLES", False),
        debug_tree=env_flag("RCS_LIST_DEBUG", False),
        list_settle_sec=env_float("RCS_LIST_SETTLE_SEC", DEFAULT_LIST_SETTLE_SEC),
        click_retry_count=_parse_int_env("RCS_CLICK_RETRY_COUNT", DEFAULT_CLICK_RETRY_COUNT),
        click_retry_delay_sec=env_float(
            "RCS_CLICK_RETRY_DELAY_SEC",
            DEFAULT_CLICK_RETRY_DELAY_SEC,
        ),
        vlm_api_url=vlm_api_url,
        vlm_api_key=os.environ.get("VLM_API_KEY", "").strip(),
        vlm_model=os.environ.get("VLM_MODEL_NAME", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL,
        vlm_temperature=env_float("VLM_TEMPERATURE", DEFAULT_VLM_TEMPERATURE),
    )


def _desktop_scan_backends() -> tuple[str, ...]:
    pywinauto_backend = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "win32"
    raw = [
        item.strip().lower()
        for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "win32,uia").split(",")
        if item.strip()
    ]
    backends = raw + [pywinauto_backend]
    return tuple(dict.fromkeys(b for b in backends if b in {"uia", "win32"})) or ("uia", "win32")


def _is_main_window_title(title: str, regex_text: str) -> bool:
    try:
        return re.search(regex_text, title, flags=re.IGNORECASE) is not None
    except re.error:
        t = title.lower()
        return "rcs" in t and "[server" in t


def _scan_window_list(windows, source_name: str, regex_text: str, debug_rows: list[str]):
    for idx, win in enumerate(windows, start=1):
        try:
            title = win.window_text() or ""
        except Exception as exc:
            debug_rows.append(f"{source_name}[{idx}] title-read-error={exc}")
            continue

        matched = _is_main_window_title(title, regex_text)
        debug_rows.append(f"{source_name}[{idx}] matched={matched} title={title!r}")
        if matched:
            return win, title
    return None, ""


def _find_existing_main_window(settings: ListToolsSettings):
    """이미 로그인된 메인 RCS 창을 찾아 반환한다."""
    deadline = time.time() + max(1.0, settings.timeout)
    debug_rows: list[str] = []
    backends = _desktop_scan_backends()

    while time.time() < deadline:
        for backend in backends:
            try:
                windows = Desktop(backend=backend).windows(top_level_only=True, visible_only=True)
            except Exception as exc:
                debug_rows.append(f"desktop[{backend}] windows-error={exc}")
                continue

            main_window, main_title = _scan_window_list(
                windows, f"desktop[{backend}]", settings.window_title_regex, debug_rows
            )
            if main_window is not None:
                return main_window, main_title, debug_rows
        time.sleep(0.4)

    return None, "", debug_rows


def _capture_window(window) -> "Image.Image":
    """pywinauto 창 영역을 캡처하여 PIL Image로 반환한다."""
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


def _encode_image(image: "Image.Image") -> tuple[str, int, int]:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    payload = buf.getvalue()
    b64 = base64.b64encode(payload).decode("utf-8")
    w, h = image.size
    print(f"[INFO] 이미지 인코딩: {w}x{h}, {len(payload) / 1024:.1f}KB")
    return b64, w, h


def _extract_json(text: str) -> dict:
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


def _click_at(element_key: str, window, elements: dict, settings: ListToolsSettings) -> bool:
    point = elements.get(element_key)
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        print(f"[ERROR] 클릭 대상 '{element_key}' 좌표가 없습니다.")
        return False

    rect = window.rectangle()
    x = int(point["x"]) + rect.left
    y = int(point["y"]) + rect.top
    x = max(rect.left, min(x, rect.right - 1))
    y = max(rect.top, min(y, rect.bottom - 1))
    rel_x = max(0, min(int(point["x"]), rect.right - rect.left - 1))
    rel_y = max(0, min(int(point["y"]), rect.bottom - rect.top - 1))

    print(f"[INFO] '{element_key}' 클릭: screen=({x}, {y})")
    attempts = max(1, settings.click_retry_count)
    for attempt in range(1, attempts + 1):
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
            mouse.move(coords=(x, y))
            time.sleep(0.08)
            mouse.press(button="left", coords=(x, y))
            time.sleep(0.05)
            mouse.release(button="left", coords=(x, y))
            print(f"[INFO] mouse press/release 실행 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] mouse press/release 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, settings.click_retry_delay_sec))

    return False


def _save_target_marked_image(image: "Image.Image", point: dict, filename: str, label: str) -> None:
    """툴 클릭 좌표를 스크린샷에 마킹해서 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    radius = 12
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        return

    x = int(point["x"])
    y = int(point["y"])
    color = "lime"
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)
    draw.text((x + radius + 3, y - 16), f"{label} ({x},{y})", fill=color, font=font)

    out_path = _debug_image_path(filename)
    debug_img.save(out_path)
    print(f"[INFO] 툴 클릭 좌표 디버그 이미지 저장: {out_path}")


def _save_raw_image(image: "Image.Image", filename: str) -> None:
    out_path = _debug_image_path(filename)
    image.save(out_path)
    print(f"[INFO] 스냅샷 저장: {out_path}")


def _focus_main_window(window) -> None:
    try:
        if hasattr(window, "is_minimized") and window.is_minimized():
            window.restore()
            time.sleep(0.15)
    except Exception:
        pass

    try:
        window.set_focus()
    except Exception as exc:
        print(f"[WARNING] 메인 창 포커스 실패: {exc}")

    try:
        rect = window.rectangle()
        print(
            f"[INFO] 캡처 대상 창 영역: left={rect.left}, top={rect.top}, "
            f"size={rect.right - rect.left}x{rect.bottom - rect.top}"
        )
    except Exception as exc:
        print(f"[WARNING] 창 영역 조회 실패: {exc}")


def _find_target_tool(data: dict, target_tool_name: str) -> dict | None:
    if not isinstance(data, dict):
        print("[WARNING] VLM 응답이 JSON 객체가 아닙니다.")
        return None

    found = data.get("found")
    if isinstance(found, str):
        found_flag = found.strip().lower() in {"true", "1", "yes", "on"}
    else:
        found_flag = bool(found)

    if found_flag:
        name = str(data.get("matched_name", "")).strip() or target_tool_name
        x = _to_int(data.get("x"))
        y = _to_int(data.get("y"))
        if x is None or y is None:
            print("[WARNING] 대상 툴 좌표 파싱 실패")
            return None
        return {"name": name, "x": x, "y": y, "anchor": str(data.get("coord_anchor", "name_color_box_center"))}

    rows = data.get("tools")
    if not isinstance(rows, list):
        rows = data.get("rows")
    if not isinstance(rows, list):
        return None

    target = target_tool_name.strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name or name.strip().lower() != target:
            continue
        x = _to_int(row.get("x"))
        y = _to_int(row.get("y"))
        if (x is None or y is None) and isinstance(row.get("coord"), dict):
            x = _to_int(row["coord"].get("x"))
            y = _to_int(row["coord"].get("y"))
        if x is None or y is None:
            print(f"[WARNING] 대상 툴 '{name}' 좌표 파싱 실패")
            return None
        return {"name": name, "x": x, "y": y, "anchor": str(row.get("coord_anchor", row.get("anchor", "")))}

    return None


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


def _request_vlm(
    client: LangChainOpenAICompatibleVLMClient,
    settings: ListToolsSettings,
    system_message: str,
    prompt: str,
    image_b64: str,
) -> str:
    request = ChatImageRequest(
        model=settings.vlm_model,
        system_message=system_message,
        user_text=prompt,
        image_b64=image_b64,
        temperature=settings.vlm_temperature,
    )
    print(f"[INFO] VLM 호출: model={settings.vlm_model}, endpoint={client.endpoint}")
    start = time.time()
    raw = client.chat_with_image(request)
    elapsed = (time.time() - start) * 1000
    print(f"[INFO] 응답 수신 ({elapsed:.0f}ms)")
    print(f"[INFO] 원문 응답:\n{raw}\n")
    return raw


# ---------------------------------------------------------------------------
# 기존 UIA 기반 툴 목록 조회 (공개 함수 — select_tool.py 에서 임포트)
# ---------------------------------------------------------------------------

def get_tool_list(rcs_window) -> List[str]:
    """List 탭 영역에서 툴 이름 목록을 반환한다.

    UIA 컨트롤 탐색 순서:
        1. ListView (ListItem)
        2. TreeView (TreeItem)
        3. DataGrid / Table (DataItem)
    """

    def _collect(container_type: str, child_type: str) -> List[str]:
        containers = [
            c for c in rcs_window.descendants(control_type=container_type) if _is_visible(c)
        ]
        for container in containers:
            names = []
            for child in container.children(control_type=child_type):
                try:
                    text = (child.window_text() or "").strip()
                except Exception:
                    text = ""
                if text:
                    names.append(text)
            if names:
                print(f"[INFO] {container_type}/{child_type} 에서 {len(names)}개 항목 발견")
                return names
        return []

    for container_type, child_type in TOOL_CONTAINER_ORDER:
        tools = _collect(container_type, child_type)
        if tools:
            return tools

    print("[WARNING] 알려진 컨트롤 타입에서 툴을 찾지 못했습니다.")
    print("[WARNING] RCS_LIST_DEBUG=1로 컨트롤 트리를 확인하고 실제 타입을 파악하세요.")
    return []


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    if not settings.vlm_api_url:
        print("[ERROR] VLM_API_URL 또는 VLM_API_BASE_URL 환경변수가 필요합니다.")
        return 3

    client = LangChainOpenAICompatibleVLMClient(
        base_url=settings.vlm_api_url,
        api_key=settings.vlm_api_key,
        timeout_sec=120.0,
    )

    print("[INFO] RCS 메인 창 탐색 시작")
    main_window, main_title, debug_rows = _find_existing_main_window(settings)
    if settings.debug_main_window_titles:
        print(f"[DEBUG] 메인 창 regex: {settings.window_title_regex!r}")
        if not debug_rows:
            print("[DEBUG] no visible top-level windows")
        else:
            for row in debug_rows:
                print(f"[DEBUG] {row}")

    if main_window is None:
        print("[ERROR] 로그인된 RCS 메인 창을 찾을 수 없습니다.")
        return 4

    print(f"[INFO] RCS 메인 창 발견: '{main_title}'")
    if settings.debug_tree:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=5):")
        try:
            main_window.print_control_identifiers(depth=5)
        except Exception as exc:
            print(f"[WARNING] 컨트롤 트리 덤프 실패: {exc}")

    print("[INFO] List 탭 화면 기준으로 대상 툴 좌표 추출을 진행합니다.")
    time.sleep(max(0.0, settings.list_settle_sec))

    try:
        # 1) List 화면에서 대상 툴 좌표 추출
        _focus_main_window(main_window)
        list_image = _capture_window(main_window)
        _save_raw_image(list_image, "debug_list_panel.png")
        list_b64, list_w, list_h = _encode_image(list_image)
        list_system, list_prompt = build_rcs_select_tool_prompt(
            width=list_w,
            height=list_h,
            target_tool_name=TARGET_TOOL_NAME,
        )
        list_raw = _request_vlm(client, settings, list_system, list_prompt, list_b64)
        list_data = _extract_json(list_raw)
        print(f"[INFO] VLM 응답 JSON:\n{json.dumps(list_data, indent=2, ensure_ascii=False)}\n")
        target_tool = _find_target_tool(list_data, TARGET_TOOL_NAME)
        if not target_tool:
            print(f"[ERROR] 대상 툴 '{TARGET_TOOL_NAME}'을(를) 찾지 못했습니다.")
            return 6
        _save_target_marked_image(
            list_image,
            target_tool,
            filename="debug_target_tool_coords.png",
            label=target_tool.get("name", TARGET_TOOL_NAME),
        )
    except Exception as exc:
        print(f"[ERROR] 툴 좌표 추출 단계 실패: {exc}")
        return 5

    print(
        f"[INFO] 대상 툴 좌표: {target_tool['name']} "
        f"@ ({target_tool['x']}, {target_tool['y']})"
    )

    if not _click_at(
        target_tool["name"],
        main_window,
        {target_tool["name"]: {"x": target_tool["x"], "y": target_tool["y"]}},
        settings,
    ):
        print(f"[ERROR] 대상 툴 '{TARGET_TOOL_NAME}' 클릭 실패")
        return 7

    print(f"[INFO] 대상 툴 '{TARGET_TOOL_NAME}' 클릭 완료")

    return 0


if __name__ == "__main__":
    sys.exit(main())
