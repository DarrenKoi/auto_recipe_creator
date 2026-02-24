"""RCS 메인 화면의 List 탭에서 툴 목록/상태를 읽는다 (Windows 전용).

기본 동작:
1) 현재 화면이 이미 List 탭이라고 가정
2) List 영역의 툴 이름 + 상태등(녹색=on, 검정=off) + 좌표 추출

참고:
- `get_tool_list()` 함수는 기존 UIA 기반 조회 방식으로 유지되어
  `select_tool.py`에서 그대로 임포트해 사용할 수 있다.
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

try:
    import mss
    import mss.tools

    MSS_AVAILABLE = True
except ImportError:
    mss = None  # type: ignore[assignment]
    MSS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    PIL_AVAILABLE = False

try:
    from pywinauto import Desktop, mouse

    PYWINAUTO_AVAILABLE = True
except ImportError:
    Desktop = mouse = None  # type: ignore[assignment]
    PYWINAUTO_AVAILABLE = False

from poc.work.prompts import build_rcs_tool_list_reader_prompt
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
DEFAULT_TAB_SETTLE_SEC = 0.35
DEFAULT_LIST_SETTLE_SEC = 0.60
DEFAULT_CLICK_RETRY_COUNT = 2
DEFAULT_CLICK_RETRY_DELAY_SEC = 0.25
DEFAULT_VLM_MODEL = "Kimi-K2.5"
DEFAULT_VLM_TEMPERATURE = 0.0
TARGET_TAB_KEYS = ["view_tab", "list_tab"]
TAB_DEBUG_COLORS = {
    "view_tab": "orange",
    "list_tab": "cyan",
}
TAB_EXTRA_INSTRUCTIONS = (
    "Focus on the top-left tab strip only.",
    "Use the first letter anchors: 'V' in View, 'L' in List.",
    "View and List tabs are adjacent near the top-left corner.",
)
TOOL_LIST_EXTRA_INSTRUCTIONS = (
    "Assume this screenshot is already on the List tab.",
    "Do not search for or reason about View/List tab switching.",
    "Read only visible rows in the current list panel.",
    "Tool name is on the left, status light is on the right side of that row.",
    "If a tool name starts with numbers, keep those numbers as part of the name.",
    "Do not drop leading numeric prefixes even if they look like row indices.",
    "Green light means status=on, black light means status=off.",
    "Return x,y for each tool row in image pixel coordinates.",
    "Use x,y on the first letter of each tool name.",
    "Do not use status-light position for x,y.",
    "Set coord_anchor to first_letter for each row.",
    "Do not infer rows that are not visible.",
)


@dataclass(frozen=True)
class ListToolsSettings:
    window_title_regex: str
    timeout: float
    debug_main_window_titles: bool
    debug_tree: bool
    tab_settle_sec: float
    list_settle_sec: float
    click_retry_count: int
    click_retry_delay_sec: float
    vlm_api_url: str
    vlm_api_key: str
    vlm_model: str
    vlm_temperature: float
    target_tool_name: str
    target_tool_double_click: bool


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARNING] 잘못된 {name} 값 '{value}', 기본값 {default} 사용")
        return default


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
        tab_settle_sec=env_float("RCS_TAB_SETTLE_SEC", DEFAULT_TAB_SETTLE_SEC),
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
        target_tool_name=os.environ.get("RCS_TARGET_TOOL_NAME", "MCD018").strip(),
        target_tool_double_click=env_flag("RCS_TARGET_TOOL_DOUBLE_CLICK", True),
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


def _parse_tab_coords(data: dict, img_w: int, img_h: int) -> dict:
    for key in TARGET_TAB_KEYS:
        point = data.get(key)
        if not isinstance(point, dict):
            print(f"  [MISS] {key:20s} — VLM 응답에 없음")
            continue
        raw_x, raw_y = point.get("x", 0), point.get("y", 0)
        x, y = int(raw_x), int(raw_y)
        data[key] = {"x": x, "y": y}
        out = ""
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            out = " ← OUT OF BOUNDS"
        print(f"  [RAW ] {key:20s} — raw=({raw_x}, {raw_y}) → px=({x}, {y}){out}")
    return data


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


def _save_marked_image(image: "Image.Image", elements: dict, filename: str) -> None:
    """좌표를 스크린샷 위에 마킹해서 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    radius = 12
    for name, point in elements.items():
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            continue
        x, y = int(point["x"]), int(point["y"])
        color = TAB_DEBUG_COLORS.get(name, "white")
        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=color,
            width=2,
        )
        draw.text((x + radius + 3, y - 16), f"{name} ({x},{y})", fill=color, font=font)

    out_path = Path(__file__).parent / filename
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


def _save_raw_image(image: "Image.Image", filename: str) -> None:
    out_path = Path(__file__).parent / filename
    image.save(out_path)
    print(f"[INFO] 스냅샷 저장: {out_path}")


def _save_tool_rows_marked_image(
    image: "Image.Image",
    parsed_tools: list[dict],
    filename: str,
    target_tool_name: str = "",
) -> None:
    """툴 row 좌표를 스크린샷 위에 마킹해서 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    img_w, img_h = debug_img.size

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    target_norm = target_tool_name.strip().lower()
    radius = 10

    for idx, tool in enumerate(parsed_tools, start=1):
        x = _to_int(tool.get("x"))
        y = _to_int(tool.get("y"))
        if x is None or y is None:
            continue

        draw_x = max(0, min(x, img_w - 1))
        draw_y = max(0, min(y, img_h - 1))
        if draw_x != x or draw_y != y:
            print(
                f"[WARNING] row#{idx} 좌표가 이미지 범위를 벗어났습니다: "
                f"raw=({x}, {y}), clamped=({draw_x}, {draw_y})"
            )

        name = str(tool.get("name", "")).strip() or f"row#{idx}"
        status = str(tool.get("status", "")).strip().lower()
        if target_norm and name.lower() == target_norm:
            color = "yellow"
        elif status == "on":
            color = "lime"
        else:
            color = "red"

        draw.line([(draw_x - radius, draw_y), (draw_x + radius, draw_y)], fill=color, width=2)
        draw.line([(draw_x, draw_y - radius), (draw_x, draw_y + radius)], fill=color, width=2)
        draw.ellipse(
            [(draw_x - radius, draw_y - radius), (draw_x + radius, draw_y + radius)],
            outline=color,
            width=2,
        )
        label = f"{idx:02d}:{name} ({x},{y})"
        draw.text((draw_x + radius + 3, draw_y - 16), label, fill=color, font=font)

    out_path = Path(__file__).parent / filename
    debug_img.save(out_path)
    print(f"[INFO] 툴 좌표 디버그 이미지 저장: {out_path}")


def _normalize_tool_status(status_text: str, indicator_color: str) -> tuple[str, str]:
    status = (status_text or "").strip().lower()
    color = (indicator_color or "").strip().lower()

    if "green" in color:
        return "on", "green"
    if "black" in color:
        return "off", "black"

    if any(token in status for token in ("on", "running", "run", "active", "green")):
        return "on", "green"
    if any(token in status for token in ("off", "stop", "inactive", "black")):
        return "off", "black"

    return "off", "black"


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


def _parse_tool_rows(data: dict) -> list[dict]:
    raw_rows = data.get("tools")
    if not isinstance(raw_rows, list):
        for alt_key in ("rows", "tool_list", "items"):
            alt_rows = data.get(alt_key)
            if isinstance(alt_rows, list):
                raw_rows = alt_rows
                break

    if not isinstance(raw_rows, list):
        return []

    parsed: list[dict] = []
    for idx, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            print(f"[WARNING] row#{idx} 형식 오류(dict 아님): {row!r}")
            continue

        name = str(row.get("name", "")).strip()
        if not name:
            print(f"[WARNING] row#{idx} 이름 누락: {row!r}")
            continue

        status, color = _normalize_tool_status(
            str(row.get("status", "")),
            str(row.get("indicator_color", row.get("light", ""))),
        )

        raw_x = row.get("x")
        raw_y = row.get("y")
        if (raw_x is None or raw_y is None) and isinstance(row.get("coord"), dict):
            raw_x = row["coord"].get("x")
            raw_y = row["coord"].get("y")
        x = _to_int(raw_x)
        y = _to_int(raw_y)

        coord_anchor = str(row.get("coord_anchor", row.get("anchor", ""))).strip().lower()
        if coord_anchor in {"name", "light"}:
            print(
                f"[WARNING] row#{idx} coord_anchor={coord_anchor!r} 은 사용하지 않습니다: "
                f"name={name!r}"
            )
            coord_anchor = "invalid"
        elif coord_anchor not in {"first_letter", ""}:
            coord_anchor = ""

        parsed_row = {"name": name, "status": status, "indicator_color": color}
        if coord_anchor == "invalid":
            print(f"[WARNING] row#{idx} 좌표 폐기: first_letter 기준 좌표만 허용됩니다.")
        elif x is not None and y is not None:
            parsed_row["x"] = x
            parsed_row["y"] = y
            if coord_anchor == "first_letter":
                parsed_row["coord_anchor"] = coord_anchor
        else:
            print(
                f"[WARNING] row#{idx} 좌표 누락/파싱 실패: "
                f"name={name!r}, raw_x={raw_x!r}, raw_y={raw_y!r}"
            )

        parsed.append(parsed_row)

    return parsed


def _find_target_tool(parsed_tools: list[dict], target_name: str) -> dict | None:
    target_norm = target_name.strip().lower()
    if not target_norm:
        return None

    # 1) 완전 일치 우선
    for tool in parsed_tools:
        name = str(tool.get("name", "")).strip()
        if name.lower() == target_norm:
            return tool

    # 2) 부분 일치 보조
    for tool in parsed_tools:
        name = str(tool.get("name", "")).strip()
        if target_norm in name.lower():
            print(f"[WARNING] 타겟 툴 부분 일치 사용: target={target_name!r}, matched={name!r}")
            return tool

    return None


def _click_tool_row(window, tool: dict, settings: ListToolsSettings) -> bool:
    coord_anchor = str(tool.get("coord_anchor", "")).strip().lower()
    if coord_anchor != "first_letter":
        print(
            f"[ERROR] coord_anchor={coord_anchor or '(missing)'} 입니다. "
            f"first_letter만 허용됩니다: {tool.get('name', '(unknown)')}"
        )
        return False

    x = _to_int(tool.get("x"))
    y = _to_int(tool.get("y"))
    if x is None or y is None:
        print(f"[ERROR] 타겟 툴 좌표가 없습니다: {tool.get('name', '(unknown)')}")
        return False

    rect = window.rectangle()
    screen_x = max(rect.left, min(rect.left + x, rect.right - 1))
    screen_y = max(rect.top, min(rect.top + y, rect.bottom - 1))
    rel_x = max(0, min(x, rect.right - rect.left - 1))
    rel_y = max(0, min(y, rect.bottom - rect.top - 1))

    action = "double-click" if settings.target_tool_double_click else "single-click"
    coord_anchor = tool.get("coord_anchor", "?")
    print(
        f"[INFO] 타겟 툴 {action}: "
        f"name={tool.get('name')!r}, anchor={coord_anchor}, screen=({screen_x}, {screen_y})"
    )

    attempts = max(1, settings.click_retry_count)
    for attempt in range(1, attempts + 1):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            if settings.target_tool_double_click:
                window.double_click_input(coords=(rel_x, rel_y), button="left")
            else:
                window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] click_input 실패 (attempt={attempt}): {exc}")

        try:
            if settings.target_tool_double_click:
                mouse.double_click(button="left", coords=(screen_x, screen_y))
            else:
                mouse.click(button="left", coords=(screen_x, screen_y))
            print(f"[INFO] mouse fallback 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] mouse fallback 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, settings.click_retry_delay_sec))

    return False


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

    print("[INFO] 현재 화면이 이미 List 탭이라고 가정하고 툴 목록 추출을 진행합니다.")
    time.sleep(max(0.0, settings.list_settle_sec))

    try:
        # 1) List 화면에서 툴 목록/상태 추출
        list_image = _capture_window(main_window)
        _save_raw_image(list_image, "debug_list_panel.png")
        list_b64, list_w, list_h = _encode_image(list_image)
        list_system, list_prompt = build_rcs_tool_list_reader_prompt(
            width=list_w,
            height=list_h,
            extra_instructions=TOOL_LIST_EXTRA_INSTRUCTIONS,
        )
        list_raw = _request_vlm(client, settings, list_system, list_prompt, list_b64)
        list_data = _extract_json(list_raw)
        print(f"[INFO] 툴 목록 JSON:\n{json.dumps(list_data, indent=2, ensure_ascii=False)}\n")
        parsed_tools = _parse_tool_rows(list_data)
        _save_tool_rows_marked_image(
            list_image,
            parsed_tools,
            filename="debug_list_tools_coords.png",
            target_tool_name=settings.target_tool_name,
        )
    except Exception as exc:
        print(f"[ERROR] 툴 목록 추출 단계 실패: {exc}")
        return 5

    if not parsed_tools:
        print("[ERROR] VLM에서 유효한 툴 목록을 읽지 못했습니다.")
        return 6

    print(f"[INFO] 발견된 툴 목록 ({len(parsed_tools)}개):")
    for idx, tool in enumerate(parsed_tools, start=1):
        status_label = "ON " if tool["status"] == "on" else "OFF"
        color = tool["indicator_color"]
        if "x" in tool and "y" in tool:
            coord_anchor = tool.get("coord_anchor", "?")
            print(
                f"  {idx:3}. [{status_label}] {tool['name']} ({color}) "
                f"@ ({tool['x']}, {tool['y']}) [{coord_anchor}]"
            )
        else:
            print(f"  {idx:3}. [{status_label}] {tool['name']} ({color})")

    if settings.target_tool_name:
        target_tool = _find_target_tool(parsed_tools, settings.target_tool_name)
        if target_tool is None:
            print(f"[ERROR] 타겟 툴을 찾지 못했습니다: {settings.target_tool_name!r}")
            return 7

        if not _click_tool_row(main_window, target_tool, settings):
            print(f"[ERROR] 타겟 툴 클릭 실패: {settings.target_tool_name!r}")
            return 8

        print(f"[INFO] 타겟 툴 클릭 완료: {target_tool.get('name', settings.target_tool_name)!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
