"""RCS List 탭에서 특정 툴을 선택(클릭/더블클릭)한다 (Windows 전용).

우선 VLM으로 타겟 툴 좌표를 찾고 클릭하며, 필요하면 기존 UIA 방식으로 폴백한다.

환경 변수:
    RCS_MAIN_WINDOW_REGEX     연결할 RCS 창 제목 정규식 (우선순위 1)
    RCS_WINDOW_TITLE          연결할 RCS 창 제목 정규식 (우선순위 2)
    RCS_TOOL_NAME             선택할 툴 이름 (기본: MCD018)
    RCS_TIMEOUT               창 탐색 대기 제한 시간(초, 기본: 15)
    RCS_SELECT_NO_SWITCH      1/true/yes/on 이면 List 탭 자동 전환 생략
    RCS_SELECT_DOUBLE_CLICK   1/true/yes/on 이면 더블클릭
    RCS_SELECT_LIST_FIRST     1/true/yes/on 이면 선택 전에 전체 목록 출력
    RCS_SELECT_DEBUG          1/true/yes/on 이면 컨트롤 트리 덤프
    RCS_SELECT_LIST_SETTLE_SEC List 탭 대기 시간(초, 기본: 0.6)
    RCS_CLICK_RETRY_COUNT     클릭 재시도 횟수(기본: 2)
    RCS_CLICK_RETRY_DELAY_SEC 클릭 재시도 간격(초, 기본: 0.25)
    RCS_SELECT_UIA_FALLBACK   VLM 실패 시 UIA 폴백 허용(기본: true)
    VLM_API_URL/VLM_API_BASE_URL, VLM_API_KEY, VLM_MODEL_NAME, VLM_TEMPERATURE
"""

import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import mss
import mss.tools
from PIL import Image
from pywinauto import mouse

from poc.work.list_up_tools import get_tool_list
from poc.work.prompts import build_rcs_select_tool_prompt
from poc.work.rcs_common import (
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_TITLE_REGEX,
    TOOL_CONTAINER_ORDER,
    _is_visible,
    connect_rcs_window,
    env_flag,
    env_float,
    load_env,
    switch_tab,
)
from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient

DEFAULT_LIST_SETTLE_SEC = 0.60
DEFAULT_CLICK_RETRY_COUNT = 2
DEFAULT_CLICK_RETRY_DELAY_SEC = 0.25
DEFAULT_VLM_MODEL = "Kimi-K2.5"
DEFAULT_VLM_TEMPERATURE = 0.0
DEFAULT_TOOL_NAME = "MCD018"


@dataclass(frozen=True)
class SelectToolSettings:
    window_title: str
    tool_name: str
    timeout: float
    no_switch: bool
    double_click: bool
    show_list_first: bool
    debug: bool
    list_settle_sec: float
    click_retry_count: int
    click_retry_delay_sec: float
    uia_fallback: bool
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


def load_settings() -> SelectToolSettings:
    load_env()
    vlm_api_url = (
        os.environ.get("VLM_API_URL", "").strip()
        or os.environ.get("VLM_API_BASE_URL", "").strip()
    )
    window_title = (
        os.environ.get("RCS_MAIN_WINDOW_REGEX", "").strip()
        or os.environ.get("RCS_WINDOW_TITLE", "").strip()
        or DEFAULT_WINDOW_TITLE_REGEX
    )
    return SelectToolSettings(
        window_title=window_title,
        tool_name=os.environ.get("RCS_TOOL_NAME", DEFAULT_TOOL_NAME).strip() or DEFAULT_TOOL_NAME,
        timeout=env_float("RCS_TIMEOUT", DEFAULT_TIMEOUT),
        no_switch=env_flag("RCS_SELECT_NO_SWITCH", False),
        double_click=env_flag("RCS_SELECT_DOUBLE_CLICK", False),
        show_list_first=env_flag("RCS_SELECT_LIST_FIRST", False),
        debug=env_flag("RCS_SELECT_DEBUG", False),
        list_settle_sec=env_float("RCS_SELECT_LIST_SETTLE_SEC", DEFAULT_LIST_SETTLE_SEC),
        click_retry_count=_parse_int_env("RCS_CLICK_RETRY_COUNT", DEFAULT_CLICK_RETRY_COUNT),
        click_retry_delay_sec=env_float("RCS_CLICK_RETRY_DELAY_SEC", DEFAULT_CLICK_RETRY_DELAY_SEC),
        uia_fallback=env_flag("RCS_SELECT_UIA_FALLBACK", True),
        vlm_api_url=vlm_api_url,
        vlm_api_key=os.environ.get("VLM_API_KEY", "").strip(),
        vlm_model=os.environ.get("VLM_MODEL_NAME", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL,
        vlm_temperature=env_float("VLM_TEMPERATURE", DEFAULT_VLM_TEMPERATURE),
    )


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


def _capture_window(window) -> tuple["Image.Image", str, int, int]:
    rect = window.rectangle()
    region = {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }

    with mss.mss() as sct:
        shot = sct.grab(region)
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_bytes))
    image.load()
    width, height = image.size
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    print(f"[INFO] 창 캡처 완료: {width}x{height} px")
    print(f"[INFO] 이미지 인코딩: {width}x{height}, {len(png_bytes) / 1024:.1f}KB")
    return image, image_b64, width, height


def _save_snapshot(image: "Image.Image", filename: str) -> None:
    out_path = Path(__file__).parent / filename
    image.save(out_path)
    print(f"[INFO] 스냅샷 저장: {out_path}")


def _request_vlm(
    client: LangChainOpenAICompatibleVLMClient,
    settings: SelectToolSettings,
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


def _parse_vlm_target(data: dict) -> dict | None:
    row = data
    if isinstance(data.get("tool"), dict):
        row = data["tool"]
    elif isinstance(data.get("target"), dict):
        row = data["target"]

    found_raw = row.get("found", False)
    if isinstance(found_raw, bool):
        found = found_raw
    else:
        found = str(found_raw).strip().lower() in {"1", "true", "yes", "on", "y"}

    matched_name = str(row.get("matched_name", row.get("name", ""))).strip()
    match_type = str(row.get("match_type", "none")).strip().lower() or "none"
    coord_anchor = str(row.get("coord_anchor", row.get("anchor", ""))).strip().lower()
    x = _to_int(row.get("x"))
    y = _to_int(row.get("y"))

    if not found:
        return {
            "found": False,
            "matched_name": matched_name,
            "match_type": "none",
            "coord_anchor": coord_anchor,
        }

    if x is None or y is None:
        print("[WARNING] VLM 응답에 좌표가 없어서 타겟을 사용할 수 없습니다.")
        return None

    if match_type not in {"exact", "partial"}:
        match_type = "exact" if matched_name else "partial"

    return {
        "found": True,
        "matched_name": matched_name,
        "match_type": match_type,
        "x": x,
        "y": y,
        "coord_anchor": coord_anchor,
    }


def _click_tool_at_point(
    window,
    x: int,
    y: int,
    double_click: bool,
    settings: SelectToolSettings,
) -> bool:
    rect = window.rectangle()
    screen_x = max(rect.left, min(rect.left + x, rect.right - 1))
    screen_y = max(rect.top, min(rect.top + y, rect.bottom - 1))
    rel_x = max(0, min(x, rect.right - rect.left - 1))
    rel_y = max(0, min(y, rect.bottom - rect.top - 1))

    action = "double-click" if double_click else "single-click"
    print(f"[INFO] 타겟 툴 {action}: screen=({screen_x}, {screen_y})")

    attempts = max(1, settings.click_retry_count)
    for attempt in range(1, attempts + 1):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            if double_click:
                window.double_click_input(coords=(rel_x, rel_y), button="left")
            else:
                window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] click_input 실패 (attempt={attempt}): {exc}")

        try:
            if double_click:
                mouse.double_click(button="left", coords=(screen_x, screen_y))
            else:
                mouse.click(button="left", coords=(screen_x, screen_y))
            print(f"[INFO] mouse fallback 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARNING] mouse fallback 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, settings.click_retry_delay_sec))

    return False


def _find_tool_control(rcs_window, tool_name: str):
    """툴 이름과 부분 일치하는 UIA 컨트롤을 반환한다."""
    target = tool_name.strip().lower()

    for container_type, child_type in TOOL_CONTAINER_ORDER:
        containers = [
            c for c in rcs_window.descendants(control_type=container_type)
            if _is_visible(c)
        ]
        for container in containers:
            for child in container.children(control_type=child_type):
                try:
                    text = (child.window_text() or "").strip()
                except Exception:
                    text = ""
                if target in text.lower():
                    print(f"[INFO] 툴 발견 [{container_type}/{child_type}]: '{text}'")
                    return child

    return None


def _select_tool_uia(rcs_window, tool_name: str, double_click: bool = False) -> bool:
    ctrl = _find_tool_control(rcs_window, tool_name)
    if ctrl is None:
        print(f"[ERROR] '{tool_name}' 에 해당하는 툴을 찾지 못했습니다.")
        return False

    try:
        if double_click:
            ctrl.double_click_input()
            print(f"[INFO] (UIA) 툴 더블클릭 완료: '{tool_name}'")
        else:
            ctrl.click_input()
            print(f"[INFO] (UIA) 툴 클릭 완료: '{tool_name}'")
        return True
    except Exception as exc:
        print(f"[ERROR] (UIA) 툴 클릭 중 오류: {exc}")
        return False


def _select_tool_vlm(rcs_window, settings: SelectToolSettings) -> bool:
    if not settings.vlm_api_url:
        print("[WARNING] VLM_API_URL/VLM_API_BASE_URL 미설정 — VLM 선택 건너뜀")
        return False

    try:
        list_image, image_b64, width, height = _capture_window(rcs_window)
    except Exception as exc:
        print(f"[ERROR] 화면 캡처 실패: {exc}")
        return False

    _save_snapshot(list_image, "debug_select_tool_input.png")
    system_message, prompt = build_rcs_select_tool_prompt(width, height, settings.tool_name)

    client = LangChainOpenAICompatibleVLMClient(
        base_url=settings.vlm_api_url,
        api_key=settings.vlm_api_key,
        timeout_sec=120.0,
    )

    try:
        raw = _request_vlm(client, settings, system_message, prompt, image_b64)
        data = _extract_json(raw)
    except Exception as exc:
        print(f"[ERROR] VLM 응답 처리 실패: {exc}")
        return False

    target = _parse_vlm_target(data)
    if target is None:
        return False

    if not target["found"]:
        print(f"[ERROR] VLM이 타겟 툴을 찾지 못했습니다: {settings.tool_name!r}")
        return False

    coord_anchor = str(target.get("coord_anchor", "")).lower()
    if coord_anchor and coord_anchor != "first_letter":
        print(
            f"[ERROR] coord_anchor={coord_anchor!r} 은 허용되지 않습니다. "
            "first_letter만 사용합니다."
        )
        return False

    matched_name = target.get("matched_name", settings.tool_name)
    match_type = target.get("match_type", "unknown")
    print(f"[INFO] VLM 타겟 매칭: requested={settings.tool_name!r}, matched={matched_name!r}, type={match_type}")

    ok = _click_tool_at_point(
        rcs_window,
        int(target["x"]),
        int(target["y"]),
        double_click=settings.double_click,
        settings=settings,
    )
    if ok:
        print(f"[INFO] VLM 기반 툴 선택 완료: {matched_name!r}")
    return ok


def select_tool(rcs_window, settings: SelectToolSettings) -> bool:
    """List 탭에서 지정한 툴을 선택한다. (VLM 우선, 필요 시 UIA 폴백)"""
    time.sleep(max(0.0, settings.list_settle_sec))

    if _select_tool_vlm(rcs_window, settings):
        return True

    if settings.uia_fallback:
        print("[INFO] UIA 폴백으로 툴 선택을 재시도합니다.")
        return _select_tool_uia(
            rcs_window,
            settings.tool_name,
            double_click=settings.double_click,
        )

    print("[ERROR] VLM 기반 선택 실패, UIA 폴백 비활성화")
    return False


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()

    try:
        rcs_win = connect_rcs_window(settings.window_title, settings.timeout)
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    if not settings.no_switch:
        ok = switch_tab(rcs_win, "List")
        if not ok:
            print("[WARNING] List 탭 전환 실패 — 현재 탭에서 계속 진행합니다.")

    if settings.debug:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=5):")
        rcs_win.print_control_identifiers(depth=5)

    if settings.show_list_first:
        tools = get_tool_list(rcs_win)
        if tools:
            print(f"\n[INFO] 전체 툴 목록 ({len(tools)}개):")
            for i, name in enumerate(tools, 1):
                print(f"  {i:3}. {name}")
        print()

    ok = select_tool(rcs_win, settings)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
