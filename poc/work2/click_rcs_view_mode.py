"""RCS 메인 화면에서 View/List 탭을 VLM으로 검출하고 탭을 순차 클릭하는 스크립트 (Windows 전용).

이미 로그인된 RCS 메인 창을 데스크톱에서 찾아 스크린샷을 캡처한 뒤,
UI-Venus primary + PaddleOCR assist pipeline 으로 View/List 탭 좌표를 요청하고, 디버그 이미지를 저장한 다음
View 탭과 List 탭을 순서대로 클릭한다.
"""

import json
import os
import sys
import time
from pathlib import Path

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient
from poc.work2.flask_vlm import apply_work2_pipeline_env_defaults, load_work2_env
from poc.work2.pipeline_ocr import build_ocr_extra_instructions, collect_ocr_hint_result
from poc.work2.prompts import build_rcs_main_tab_locator_prompt
from poc.work2.rcs_utils import (
    capture_window,
    click_at,
    debug_image_path,
    encode_image_webp,
    extract_json,
    find_existing_main_window,
    is_main_window_title,
    parse_coords,
    save_marked_image,
)

load_work2_env()
WORK2_PIPELINE = apply_work2_pipeline_env_defaults()

# ─────────────────────────── 설정 ───────────────────────────

VLM_API_URL = str(WORK2_PIPELINE["primary_api_url"] or "")
VLM_API_KEY = str(WORK2_PIPELINE["primary_api_key"] or "")
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

VLM_MODEL = str(WORK2_PIPELINE["primary_model_name"] or "ui-venus-1.5-8b")

TARGET_ELEMENTS = ["view_tab", "list_tab"]
TAB_CLICK_SEQUENCE = ("view_tab", "list_tab")
TAB_CLICK_INTERVAL_SEC = 2.0
TAB_EXTRA_INSTRUCTIONS = (
    "Focus on the top-left tab strip only.",
    "Use the first letter of each tab as the primary anchor: 'V' in View, 'L' in List.",
    "View and List tabs are adjacent near the top-left corner.",
)
TAB_OCR_FOCUS_WORDS = ("View", "List")
LIST_TAB_X_OFFSET_FROM_VIEW = 50  # view_tab.x + this offset → list_tab.x

ELEMENT_COLORS = {
    "view_tab": "orange",
    "list_tab": "cyan",
}

DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"

VLM_CLIENT = LangChainOpenAICompatibleVLMClient(
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


def _main_title_matcher(title: str) -> bool:
    """모듈 설정 MAIN_WINDOW_TITLE_REGEX 기준 매칭 함수."""
    return is_main_window_title(title, MAIN_WINDOW_TITLE_REGEX)


# ─────────────────────────── 메인 ───────────────────────────


def main() -> int:
    """메인 RCS 창을 찾아 View/List 탭을 VLM 좌표로 순차 클릭한다."""
    print("[INFO] RCS 메인 창에서 View/List 탭 검출 시작")
    if VLM_API_URL:
        print(
            f"[INFO] work2 pipeline: primary={WORK2_PIPELINE['primary_service']} "
            f"({VLM_MODEL}) -> ocr={WORK2_PIPELINE['ocr_service']} "
            f"({WORK2_PIPELINE['ocr_model_name']})"
        )
        print(
            f"[INFO] work2 primary endpoint={VLM_API_URL}, "
            f"ocr endpoint={WORK2_PIPELINE['ocr_api_url']}"
        )
    else:
        print("[WARNING] work2 VLM API URL이 비어 있습니다. WORK2_FLASK_API_BASE_URL 또는 WORK2_VLM_API_URL을 확인하세요.")

    main_window, main_title, debug_rows = find_existing_main_window(
        DESKTOP_SCAN_BACKENDS, _main_title_matcher
    )
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
    image = capture_window(main_window)
    img_b64, w, h = encode_image_webp(image)

    rect = main_window.rectangle()
    print(
        f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
        f"size={rect.right - rect.left}x{rect.bottom - rect.top}"
    )

    ocr_result = collect_ocr_hint_result(
        image_b64=img_b64,
        image_width=w,
        image_height=h,
        image_mime="image/webp",
        pipeline_config=WORK2_PIPELINE,
        context_label="RCS main window tab strip",
        focus_words=TAB_OCR_FOCUS_WORDS,
    )

    # VLM 호출
    system_msg, prompt = build_rcs_main_tab_locator_prompt(
        width=w,
        height=h,
        target_keys=TARGET_ELEMENTS,
        extra_instructions=TAB_EXTRA_INSTRUCTIONS + build_ocr_extra_instructions(ocr_result),
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
        data = extract_json(raw)
        print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
        data = parse_coords(data, TARGET_ELEMENTS, w, h)
    except Exception as exc:
        print(f"[ERROR] JSON 파싱 실패: {exc}")
        return 3

    # list_tab 좌표를 view_tab 기준으로 오프셋 보정
    view_pt = data.get("view_tab")
    if isinstance(view_pt, dict) and "x" in view_pt and "y" in view_pt:
        derived_x = view_pt["x"] + LIST_TAB_X_OFFSET_FROM_VIEW
        derived_y = view_pt["y"]
        data["list_tab"] = {"x": derived_x, "y": derived_y}
        print(
            f"[INFO] list_tab 좌표를 view_tab 기준 오프셋으로 보정: "
            f"view({view_pt['x']},{view_pt['y']}) + offset({LIST_TAB_X_OFFSET_FROM_VIEW},0) "
            f"→ list({derived_x},{derived_y})"
        )

    detected = sum(1 for k in TARGET_ELEMENTS if k in data and isinstance(data[k], dict))
    print(f"[INFO] 검출률: {detected}/{len(TARGET_ELEMENTS)}")

    # 디버그 이미지 저장
    out_path = debug_image_path(DEBUG_IMAGE_DIR, "debug_view_mode.png")
    save_marked_image(image, data, ELEMENT_COLORS, out_path)

    # View/List 탭 순차 클릭
    for idx, click_key in enumerate(TAB_CLICK_SEQUENCE, start=1):
        if not click_at(
            click_key,
            main_window,
            data,
            retry_count=CLICK_RETRY_COUNT,
            retry_delay_sec=CLICK_RETRY_DELAY_SEC,
        ):
            print(f"[ERROR] '{click_key}' 클릭 실패")
            return 4
        print(f"[INFO] '{click_key}' 클릭 완료")
        if idx < len(TAB_CLICK_SEQUENCE):
            print(f"[INFO] 다음 탭 클릭까지 {TAB_CLICK_INTERVAL_SEC:.1f}초 대기")
            time.sleep(max(0.0, TAB_CLICK_INTERVAL_SEC))

    return 0


if __name__ == "__main__":
    sys.exit(main())
