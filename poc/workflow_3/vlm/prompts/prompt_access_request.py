"""점유 중 들어온 '접근 요청' 확인 팝업의 허용 버튼 로케이트용 타겟 정의.

우리가 tool 을 점유한 동안 다른 엔지니어가 접속을 시도하면 RCS 가 우리 화면에
허용/거부 확인 팝업을 띄운다. 그 팝업의 **허용** 버튼만 고르게 한다.

설명에 거부/취소를 명시적으로 배제하는 이유는 share_request 와 같다 - 두 버튼이
나란히 붙어 있어 목표만 서술하면 인접 버튼과 구분이 약하다.
"""

from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

GRANT_BUTTON_TARGET = TargetConfig(
    key="access_grant_button",
    description=(
        "the button that ALLOWS / ACCEPTS the incoming access request from another "
        "user in the confirmation dialog (e.g. 'Allow', 'Accept', 'OK', '허용'). "
        "Do NOT pick the button that denies, rejects or cancels the request."
    ),
    vertical_pad_ratio=0.8,
    vertical_pad_min_px=12,
    min_crop_height=56,
)

__all__ = ["GRANT_BUTTON_TARGET"]
