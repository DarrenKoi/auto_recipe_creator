"""점유 'select' 팝업 검출 — 다른 엔지니어가 tool 을 쓰는 중이면 접속을 포기한다.

align fail 알람으로 tool monitor 에 접속하려 할 때, 이미 다른 사용자가 점유 중이면 RCS 가
작은 'select' 팝업(제어 공유 / 화면 공유 / 기존 사용자 강제 종료)을 띄운다. 이 팝업이 떠
있으면 tool 창이 안 열려 window 탐색이 헛돈다(과거: 10회 시도 후 포기).

검출 전략(hybrid):
  1. 빠른 창 제목 열거로 'select' 제목 창을 1차 탐지(저비용·결정적). 없으면 즉시 False.
  2. 제목이 있으면 그 창을 캡처해 VLM 으로 세 옵션(공유/종료)이 보이는지 확인(오검출↓).
     VLM 이 부재/실패하면 제목만으로 점유 판단(보수적 — 접속 중 'select' 창은 강한 신호).

세 옵션 중 어느 것도 클릭하지 않는다(사람 판단 영역). 검출되면 호출부가 즉시 접속을
포기하고 다음 알람을 기다린다. 모든 예외는 삼켜 False 로 — 검출 실패가 접속을 막지 않는다.
"""

from poc.workflow_3.util import (
    capture_window,
    collect_window_rows,
    find_window_by_title_prefix,
)
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import extract_json
from poc.workflow_3.vlm.prompts.prompt_select_popup import build_select_popup_prompt

LOG_COMPONENT = "occupied_popup"

# 점유 팝업으로 볼 창 제목(소문자·strip 후 비교). 'select' 단독 또는 prefix 허용.
SELECT_TITLE = "select"


def _is_select_title(title: str) -> bool:
    """창 제목이 점유 'select' 팝업으로 볼 만한지(소문자 strip 후 'select' 시작)."""
    if not title:
        return False
    return title.strip().lower().startswith(SELECT_TITLE)


def find_select_popup_window():
    """점유 'select' 팝업 **창 객체**만 돌려준다(없으면 None).

    `find_window_by_title_prefix` 는 `(window, title, backend)` 3-tuple 을 돌려주는데,
    호출부들이 그걸 창 하나로 받아 쓰고 있었다. 튜플은 절대 None 이 아니라서
    `if popup is None` 가드가 전부 죽고, capture/close 에 튜플이 그대로 들어가
    팝업 닫기와 화면 공유 요청이 **오피스에서 한 번도 동작하지 않았다**. 창을 꺼내는
    지점을 여기 하나로 모아 같은 실수가 되풀이되지 않게 한다.
    """
    if not callable(find_window_by_title_prefix):
        return None
    try:
        window, _title, _backend = find_window_by_title_prefix(SELECT_TITLE)
    except Exception as exc:
        print(f"[WARNING] select 팝업 창 탐색 실패: {exc}")
        return None
    return window


def _vlm_confirm_select(vlm_client) -> "bool | None":
    """'select' 제목 창을 캡처해 세 옵션이 보이는지 VLM 으로 확인한다.

    True=점유 팝업 확실, False=다른 창(오검출), None=확인 불가(캡처/호출 실패 → 호출부가
    제목만으로 폴백). 좌표가 아니라 yes/no 만 묻는다.

    False 는 모델이 **is_select_popup 을 불리언 false 로 명시**했을 때만 낸다. 키가 없거나
    타입이 다르면(모델 교체로 스키마가 흔들리는 경우) '아님'이 아니라 '모름'(None)이다 -
    '아님'으로 읽으면 점유 가드가 조용히 꺼져 점유된 tool 에 그대로 진입한다.
    """
    if capture_window is None:
        return None
    try:
        window = find_select_popup_window()
        if window is None:
            return None
        image = capture_window(window)
        image_b64, _w, _h = encode_image_webp(image.convert("RGB"), quality=90)
        system_message, user_message = build_select_popup_prompt()
        response = vlm_client.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_message,
            image_mime="image/webp",
            temperature=0.0,
        )
        parsed = extract_json(response.text)
        verdict = parsed.get("is_select_popup")
        if not isinstance(verdict, bool):
            print(
                "[WARNING] select 팝업 VLM 응답에 is_select_popup 불리언 없음"
                f"(제목만으로 판단): {parsed!r}"
            )
            return None
        return verdict
    except Exception as exc:
        print(f"[WARNING] select 팝업 VLM 확인 실패(제목만으로 판단): {exc}")
        return None


def detect_select_popup(vlm_client=None) -> bool:
    """점유 'select' 팝업이 떠 있으면 True. 검출 실패/비Windows 면 False(접속 진행).

    제목 열거 1차 → (있으면) VLM 확인. VLM 이 'select 아님'이라고 하면 오검출로 보고 False,
    VLM 부재/확인불가면 제목만으로 True(보수적). 모든 예외는 False 로 흡수한다.
    """
    if not callable(collect_window_rows):
        return False
    try:
        rows = collect_window_rows()
    except Exception as exc:
        print(f"[WARNING] 창 열거 실패(점유 검출 생략): {exc}")
        return False

    if not any(_is_select_title(getattr(r, "title", "")) for r in rows):
        return False

    if vlm_client is None:
        print("[INFO] 점유 'select' 팝업(제목) 감지 - VLM 없음, 제목만으로 점유 판단")
        return True

    confirmed = _vlm_confirm_select(vlm_client)
    if confirmed is None:
        print("[INFO] 점유 'select' 팝업(제목) 감지 - VLM 확인 불가, 제목만으로 점유 판단")
        return True
    if confirmed:
        print("[INFO] 점유 'select' 팝업 VLM 확인됨 - 접속 포기")
        return True
    print("[INFO] 'select' 제목 창 있으나 VLM 이 점유 팝업 아님으로 판단 - 접속 계속")
    return False


__all__ = ["SELECT_TITLE", "detect_select_popup", "find_select_popup_window"]
