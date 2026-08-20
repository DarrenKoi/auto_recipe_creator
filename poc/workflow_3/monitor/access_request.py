"""점유 중 들어온 접근 요청을 허용하는 fail-closed actuator.

우리가 tool 을 점유한 동안 다른 엔지니어가 접속을 시도하면 RCS 가 **우리 화면에**
허용/거부 확인 팝업을 띄운다. 방치하면 상대가 강제 종료로 우리 세션을 끊을 수 있으므로
(2026-08-20 오피스 확인), 사람이 개입하지 않아도 자동으로 허용해 준다.

`occupied_popup` 이 fail-**open** detector 인 것과 달리 이 모듈은 fail-**closed**
actuator 다 - `share_request` 와 같은 규약이며, 같은 확인 게이트를 그대로 쓴다:
좌표는 VLM 이 찍고 그 자리 라벨을 OCR 로 읽어 허용 계열로 확인될 때만 클릭한다.
거부/취소/종료가 읽히면 정책과 무관하게 클릭하지 않는다.

첫 오피스 실행에서는 팝업의 실제 문구를 모른다. 그래서 기본값이 **관찰 전용**
(`ALIGN_FAIL_ACCESS_GRANT=0`)이며, 읽은 토큰을 콘솔에 남기고 클릭은 하지 않는다.
문구가 확인되면 required 를 다듬고 1 로 올린다. 관찰 전용이어도 손해는 '상대가 우리를
끊는다'로 한정되고, 그건 창 닫힘(window_gone)으로 teardown 이 정상 처리한다.

협력자는 전부 주입받는다 - 실장비/VLM 없이 판정 로직을 Mac 에서 시험하기 위해서다.
"""

from dataclasses import dataclass, field

from poc.workflow_3.monitor.share_request import (
    CONFIRM_STRICT,
    VERDICT_UNREADABLE,
    accepts_label,
    classify_label,
)

LOG_COMPONENT = "access_request"

# 허용 계열 - 언어별 묶음 중 하나를 통째로 만족하면 확인이다(단어 1개짜리 묶음들).
GRANT_REQUIRED = (("allow",), ("accept",), ("grant",), ("허용",), ("승인",))
# 거부/취소/종료는 어떤 정책에서도 클릭하지 않는다.
GRANT_FORBIDDEN = ("deny", "reject", "cancel", "terminat", "거부", "취소", "종료")

STATUS_GRANTED = "granted"
STATUS_OBSERVED = "observed"            # 관찰 전용 - 문구만 남기고 클릭 안 함.
STATUS_CONFIRM_FAILED = "confirm_failed"
STATUS_NOT_FOUND = "not_found"
STATUS_BLOCKED_SAFE_MODE = "blocked_safe_mode"
STATUS_ERROR = "error"

GRANT_KEY = "access_grant_button"


@dataclass
class AccessRequestResult:
    status: str
    verdict: str = ""
    tokens: list = field(default_factory=list)
    error: str = ""


def grant_access_request(
    settings,
    *,
    locate_fn,
    read_tokens_fn,
    click_fn,
    capture_fn,
    find_popup_fn,
) -> AccessRequestResult:
    """접근 요청 팝업이 떠 있으면 허용 버튼을 확인 후 클릭한다.

      locate_fn(image, target)            -> point dict | None  (**이미지 픽셀 좌표**)
      read_tokens_fn(image, point, key)   -> list[str]
      click_fn(window, image, point, key) -> None
      capture_fn(window)                  -> image
      find_popup_fn()                     -> window | None

    `click_fn` 이 창과 이미지를 함께 받는 이유는 좌표 변환 때문이다(share_request 와
    동일 - 로케이터는 이미지 픽셀을 주고 클릭은 스크린 좌표라 배율 보정이 필요하다).
    """
    from poc.workflow_3.vlm.prompts.prompt_access_request import GRANT_BUTTON_TARGET

    policy = getattr(settings, "access_confirm_policy", CONFIRM_STRICT)

    try:
        popup = find_popup_fn()
        if popup is None:
            return AccessRequestResult(status=STATUS_NOT_FOUND)

        image = capture_fn(popup)
        point = locate_fn(image, GRANT_BUTTON_TARGET)
        if point is None:
            print("[WARNING] 접근 요청 팝업 허용 버튼 좌표 미검출 - 클릭 안 함")
            return AccessRequestResult(
                status=STATUS_CONFIRM_FAILED, verdict=VERDICT_UNREADABLE
            )

        tokens = read_tokens_fn(image, point, GRANT_KEY)
        verdict = classify_label(tokens, GRANT_REQUIRED, GRANT_FORBIDDEN)

        # 관찰 전용: 문구를 알아내는 것이 목적이므로 판정과 무관하게 클릭하지 않는다.
        if not getattr(settings, "access_grant_enabled", False):
            print(
                f"[INFO] 접근 요청 팝업 감지(관찰 전용, 클릭 안 함): "
                f"verdict={verdict} tokens={tokens!r}"
            )
            return AccessRequestResult(
                status=STATUS_OBSERVED, verdict=verdict, tokens=list(tokens or [])
            )

        if not getattr(settings, "action_enabled", False):
            print("[INFO] SAFE_MODE - 접근 요청 허용 클릭 차단")
            return AccessRequestResult(
                status=STATUS_BLOCKED_SAFE_MODE, verdict=verdict, tokens=list(tokens or [])
            )

        if not accepts_label(verdict, policy):
            print(
                f"[WARNING] 접근 요청 팝업 라벨 확인 실패 - 클릭 안 함: "
                f"verdict={verdict} policy={policy} tokens={tokens!r}"
            )
            return AccessRequestResult(
                status=STATUS_CONFIRM_FAILED, verdict=verdict, tokens=list(tokens or [])
            )

        click_fn(popup, image, point, GRANT_KEY)
        print(f"[INFO] 다른 엔지니어 접근 요청 허용 완료: tokens={tokens!r}")
        return AccessRequestResult(
            status=STATUS_GRANTED, verdict=verdict, tokens=list(tokens or [])
        )
    except Exception as exc:
        # actuator 는 예외를 삼켜 성공으로 만들지 않는다.
        print(f"[ERROR] 접근 요청 처리 중 예외: {exc}")
        return AccessRequestResult(status=STATUS_ERROR, error=str(exc))


__all__ = [
    "AccessRequestResult",
    "GRANT_FORBIDDEN",
    "GRANT_KEY",
    "GRANT_REQUIRED",
    "STATUS_BLOCKED_SAFE_MODE",
    "STATUS_CONFIRM_FAILED",
    "STATUS_ERROR",
    "STATUS_GRANTED",
    "STATUS_NOT_FOUND",
    "STATUS_OBSERVED",
    "grant_access_request",
]
