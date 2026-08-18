"""점유 'select' 팝업 actuator - 화면 공유를 요청한다.

`occupied_popup.py` 는 팝업을 **검출만** 하는 fail-open detector 다(예외를 False 로
흡수해 검출 실패가 접속을 막지 않게 한다). 이 모듈은 반대로 **클릭하는** fail-closed
actuator 다 - 확신이 없으면 누르지 않는다. 두 정책이 정반대라 파일을 나눴다.

세 라디오(제어 공유 / 화면 공유 / 기존 사용자 강제 종료)가 세로로 붙어 있어 fine 단계가
한 칸 어긋나는 것이 가장 현실적인 실패이고, 그 어긋남의 최악이 강제 종료다. 그래서 좌표를
찍은 뒤 그 자리의 라벨을 좁은 crop 으로 OCR 해 확인하고, 확인되지 않으면 클릭하지 않는다.
확인 실패는 진단 산출물(crop, OCR 원문, 판정)을 남긴다 - Mac 에서는 팝업을 볼 수 없어
첫 오피스 실행이 실제 문구를 알려주는 유일한 경로이기 때문이다.
"""

import time
from dataclasses import dataclass

from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_LENIENT as CONFIRM_LENIENT,
)
from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_OFF as CONFIRM_OFF,
)
from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_STRICT as CONFIRM_STRICT,
)
from poc.workflow_3.vlm.label_verify import normalize_label

# 확인 게이트 토큰. 대소문자 무시 부분 일치로 비교하며 국문 표기도 함께 받는다.
# required 는 **언어별 묶음의 튜플**이다 - 한 묶음을 전부 만족해야 확인으로 친다.
# 절반만 읽힌 것은 확인이 아니다('share' 만으로는 제어 공유와 구분되지 않는다).
# 오피스 실제 문구는 첫 실행의 진단 산출물로 확인한 뒤 조정한다.
SHARE_SCREEN_REQUIRED = (("share", "screen"), ("공유", "화면"))
SHARE_SCREEN_FORBIDDEN = ("control", "terminat", "제어", "종료")
REQUEST_BTN_REQUIRED = (("request",), ("요청",))
REQUEST_BTN_FORBIDDEN = ("cancel", "취소")

VERDICT_CONFIRMED = "confirmed"
VERDICT_FORBIDDEN = "forbidden"
VERDICT_UNREADABLE = "unreadable"

ACCEPTED = "accepted"
DENIED_OR_TIMEOUT = "denied_or_timeout"

STATUS_REQUESTED = "requested"
STATUS_CONFIRM_FAILED = "confirm_failed"
STATUS_NOT_FOUND = "not_found"
STATUS_BLOCKED_SAFE_MODE = "blocked_safe_mode"
STATUS_ERROR = "error"

RADIO_KEY = "share_screen_radio"
BUTTON_KEY = "request_button"


def _any_token_contains(tokens, needles) -> bool:
    """토큰 중 하나라도 needles 의 어떤 문자열을 포함하는지.

    정규화는 `label_verify.normalize_label`(영숫자만 남기고 소문자)을 쓴다. 화이트리스트
    방식이라 OCR 이 붙여 오는 어떤 구두점에도 영향받지 않는다.
    """
    normalized = [normalize_label(token) for token in (tokens or [])]
    for needle in needles:
        target = normalize_label(needle)
        if target and any(target in token for token in normalized if token):
            return True
    return False


def classify_label(tokens, required_groups, forbidden) -> str:
    """읽은 토큰이 기대 라벨인지 판정한다.

    "confirmed"  : required 묶음 중 하나를 통째로 만족하고 forbidden 이 없다.
    "forbidden"  : forbidden 토큰이 읽혔다 (required 여부와 무관하게 우선).
    "unreadable" : 어느 쪽도 확정할 수 없다.

    forbidden 을 required 보다 **먼저** 본다. 'share screen' 과 'terminate' 가 함께
    읽히는 상황은 crop 이 옆 라디오까지 삼켰다는 뜻이라 클릭해서는 안 된다.

    `required_groups` 는 언어별 묶음의 튜플이며 한 묶음을 **전부** 만족해야 확인이다.
    묶음을 데이터로 선언하는 이유는 언어 판별이 상수의 성질이지 실행 시 추론할 것이
    아니기 때문이다(문자 종류로 갈라내면 혼합 표기 needle 이 조용히 오분류된다).
    """
    if _any_token_contains(tokens, forbidden):
        return VERDICT_FORBIDDEN

    for group in required_groups:
        if group and all(_any_token_contains(tokens, (needle,)) for needle in group):
            return VERDICT_CONFIRMED
    return VERDICT_UNREADABLE


def accepts_label(status: str, policy: str) -> bool:
    """확인 판정과 정책으로 클릭 허용 여부를 정한다.

    forbidden 은 **어떤 정책에서도** 통과하지 않는다. off 는 좌표 진단용이지 강제 종료
    오클릭까지 허용하라는 뜻이 아니다. 알 수 없는 정책 문자열은 strict 로 폴백한다
    (오타가 게이트를 조용히 여는 것을 막는다).
    """
    if status == VERDICT_FORBIDDEN:
        return False
    if policy in (CONFIRM_LENIENT, CONFIRM_OFF):
        return True
    return status == VERDICT_CONFIRMED


def wait_share_response(
    eqp_id: str,
    wait_sec: float,
    *,
    find_window_fn,
    sleep_fn=time.sleep,
    now_fn=time.monotonic,
    poll_sec: float = 1.0,
) -> tuple:
    """공유 요청 후 상대의 승낙을 기다린다. `(status, 찾은 창)` 을 돌려준다.

    승낙 신호는 '제목에 eqp_id 를 가진 Remote Monitoring 창의 등장' 하나뿐이다.
    거절과 무응답은 하나로 합친다 - 거절 시 RCS 화면이 확정되지 않았고, 어느 쪽이든
    결론은 '그 엔지니어가 점유하는 동안 접근 불가' 로 같아 동작이 갈리지 않는다.

    찾은 창을 **그대로 돌려주는** 이유는 호출부가 같은 창을 다시 찾지 않게 하기
    위해서다. 창 탐색은 매번 전체 창 열거 + 포커스 활성화를 동반하므로, 버리고 다시
    찾으면 이미 전면에 있는 창을 한 번 더 낚아채는 셈이 된다.

    이 대기는 **블로킹**이며 단일 RCS 커서를 모든 tool 의 알람이 직렬로 공유하므로,
    wait_sec 은 짧게 둔다(기본 10초). 그 안에 못 받아도 알람이 유지되는 한 cooldown 후
    다시 요청하므로 기회는 한 번뿐이 아니다.

    시계와 창 탐색은 주입받는다 - 실장비 없이 테스트하기 위해서다.
    """
    deadline = now_fn() + max(0.0, wait_sec)
    while now_fn() < deadline:
        try:
            found = find_window_fn(eqp_id)
            if found is not None:
                print(f"[INFO] 화면 공유 승낙됨 - tool 창 등장: EQP_ID={eqp_id}")
                return ACCEPTED, found
        except Exception as exc:
            # 탐색 1회 실패로 대기를 끝내지 않는다 - 창은 다음 poll 에 뜰 수 있다.
            print(f"[WARNING] 공유 대기 중 창 탐색 실패(계속 대기): {exc}")
        sleep_fn(poll_sec)
    print(f"[INFO] 화면 공유 무응답/거절: EQP_ID={eqp_id} ({wait_sec:.0f}s 경과)")
    return DENIED_OR_TIMEOUT, None


@dataclass
class ShareRequestResult:
    """공유 요청 시도 결과. 판정을 함께 실어 오피스 진단에 쓴다."""

    status: str
    radio_verdict: str = ""
    button_verdict: str = ""
    error: str = ""


def request_screen_share(
    settings,
    *,
    locate_fn,
    read_tokens_fn,
    click_fn,
    capture_fn,
    find_popup_fn,
) -> ShareRequestResult:
    """'select' 팝업에서 화면 공유를 선택하고 Request 를 누른다.

    **두 라벨을 모두 확인한 뒤에야 클릭을 시작한다.** 라디오를 먼저 누르고 버튼을
    확인하면, 버튼 확인이 실패했을 때 이미 남의 팝업 상태를 바꿔 놓은 뒤가 된다.

    협력자는 전부 주입받는다 - 실장비/VLM 없이 판정 로직을 시험하기 위해서다.
      locate_fn(image, target)             -> point dict | None  (**이미지 픽셀 좌표**)
      read_tokens_fn(image, point, key)    -> list[str]
      click_fn(window, image, point, key)  -> None
      capture_fn(window)                   -> image
      find_popup_fn()                      -> window | None

    `click_fn` 이 창과 이미지를 함께 받는 이유는 **좌표 변환 때문**이다. 로케이터가 주는
    점은 이미지 픽셀이고 클릭은 스크린 좌표라, 창 rect 와 이미지 크기의 배율 보정을
    거쳐야 한다(오피스 125/150% 배율). 변환을 빼면 확인 게이트가 무의미해진다 - 점 A 의
    라벨을 확인하고 점 B 를 누르는 셈이라, 하필 강제 종료 라디오에 떨어질 수 있다.
    변환은 창을 아는 orchestrator 가 하고, 이 모듈은 판정만 한다.
    """
    from poc.workflow_3.vlm.prompts.prompt_share_options import (
        REQUEST_BUTTON_TARGET,
        SHARE_SCREEN_TARGET,
    )

    policy = getattr(settings, "share_confirm_policy", CONFIRM_STRICT)
    verdicts: dict = {}

    def _result(status: str, error: str = "") -> ShareRequestResult:
        return ShareRequestResult(
            status=status,
            radio_verdict=verdicts.get(RADIO_KEY, ""),
            button_verdict=verdicts.get(BUTTON_KEY, ""),
            error=error,
        )

    try:
        popup = find_popup_fn()
        if popup is None:
            return _result(STATUS_NOT_FOUND)

        if not getattr(settings, "action_enabled", False):
            print("[INFO] SAFE_MODE - 공유 요청 클릭 차단(요청 발송 안 함)")
            return _result(STATUS_BLOCKED_SAFE_MODE)

        image = capture_fn(popup)

        plan = []
        for target, required, forbidden in (
            (SHARE_SCREEN_TARGET, SHARE_SCREEN_REQUIRED, SHARE_SCREEN_FORBIDDEN),
            (REQUEST_BUTTON_TARGET, REQUEST_BTN_REQUIRED, REQUEST_BTN_FORBIDDEN),
        ):
            point = locate_fn(image, target)
            if point is None:
                verdicts[target.key] = VERDICT_UNREADABLE
                print(f"[WARNING] 공유 팝업 요소 좌표 미검출 - 클릭 안 함: {target.key}")
                return _result(STATUS_CONFIRM_FAILED)

            tokens = read_tokens_fn(image, point, target.key)
            verdict = classify_label(tokens, required, forbidden)
            verdicts[target.key] = verdict
            if not accepts_label(verdict, policy):
                print(
                    f"[WARNING] 공유 팝업 라벨 확인 실패 - 클릭 안 함: {target.key} "
                    f"verdict={verdict} policy={policy} tokens={tokens!r}"
                )
                return _result(STATUS_CONFIRM_FAILED)
            plan.append((point, target.key))

        for point, key in plan:
            click_fn(popup, image, point, key)
        print("[INFO] 화면 공유 요청 발송 완료 - 상대 승낙 대기")
        return _result(STATUS_REQUESTED)
    except Exception as exc:
        # actuator 는 예외를 삼켜 성공으로 만들지 않는다. 조용한 성공이 최악이다.
        print(f"[ERROR] 공유 요청 중 예외: {exc}")
        return _result(STATUS_ERROR, error=str(exc))


__all__ = [
    "ACCEPTED",
    "BUTTON_KEY",
    "CONFIRM_LENIENT",
    "CONFIRM_OFF",
    "CONFIRM_STRICT",
    "DENIED_OR_TIMEOUT",
    "RADIO_KEY",
    "REQUEST_BTN_FORBIDDEN",
    "REQUEST_BTN_REQUIRED",
    "SHARE_SCREEN_FORBIDDEN",
    "SHARE_SCREEN_REQUIRED",
    "STATUS_BLOCKED_SAFE_MODE",
    "STATUS_CONFIRM_FAILED",
    "STATUS_ERROR",
    "STATUS_NOT_FOUND",
    "STATUS_REQUESTED",
    "ShareRequestResult",
    "VERDICT_CONFIRMED",
    "VERDICT_FORBIDDEN",
    "VERDICT_UNREADABLE",
    "accepts_label",
    "classify_label",
    "request_screen_share",
    "wait_share_response",
]
