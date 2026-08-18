"""점유 'select' 팝업의 라디오/버튼 로케이트용 타겟 정의.

2단계 로케이터(`vlm/ui_venus_mai_locator.py`)에 넘길 TargetConfig 다. 세 라디오가 세로로
촘촘히 붙어 있으므로 vertical_pad 하한을 낮춰 crop 이 위아래 항목을 삼키지 않게 한다 -
tool List 행에서 배운 것과 같은 이유이며, 비율만 낮추고 하한을 두면 하한이 지배해서
소용이 없다(`SELECT_TOOL_ROW_*` 교훈).

설명 문자열에 "고르지 말아야 할 것"을 명시하는 이유는, 세 옵션이 문구를 상당 부분 공유해
("Request to share the ...") 목표만 서술하면 인접 항목과 구분이 약해지기 때문이다.
"""

from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

SHARE_SCREEN_TARGET = TargetConfig(
    key="share_screen_radio",
    description=(
        "the radio button for requesting to SHARE THE SCREEN (view/observe only) "
        "in the occupied-tool selection dialog. "
        "Do NOT pick the option about sharing CONTROL of the tool, and do NOT pick "
        "the option about TERMINATING the existing user's session."
    ),
    vertical_pad_ratio=0.6,
    vertical_pad_min_px=10,
    min_crop_height=56,
)

REQUEST_BUTTON_TARGET = TargetConfig(
    key="request_button",
    description=(
        "the button labelled 'Request' that submits the selected option in the "
        "occupied-tool selection dialog. It is NOT the 'Cancel' button."
    ),
    vertical_pad_ratio=0.8,
    vertical_pad_min_px=12,
    min_crop_height=56,
)

__all__ = ["REQUEST_BUTTON_TARGET", "SHARE_SCREEN_TARGET"]
