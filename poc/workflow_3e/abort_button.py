"""측정 abort(Stop/중지) 버튼을 VLM 으로 찾는 locator.

측정 abort 잡(`workflow_3e/abort_cycle.run_abort_cycle`)의 마지막 단계에서 쓴다: MES 가
'N회 연속 측정 실패' 임계 알람을 쏜 tool 에 접속한 뒤, 진행 중인 측정을 멈추는 Stop/Abort
버튼을 눌러야 한다. 버튼은 RCS tool 창 위의 컨트롤이라 좌표가 **screen 절대 픽셀**이다.

설계 경계(align/ok_button.py 와 동일): VLM 은 버튼 *영역*만 식별한다. abort 여부는 MES 가
이미 결정했다(임계 알람). 여기서는 어떤 UI 버튼을 누를지 위치만 찾는다.
"""

import cv2
import numpy as np
from PIL import Image

from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import bbox_center, bbox_to_pixels, extract_json
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient


def _abort_button_system_prompt() -> str:
    """Abort/Stop 버튼 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a CD-SEM / VeritySEM metrology tool that is RUNNING "
        "a measurement recipe. The operator needs to STOP / ABORT the in-progress "
        "measurement run because too many points have failed.\n"
        "Locate the button that STOPS or ABORTS the running measurement. It is a "
        "clickable control, usually labelled 'Stop', 'Abort', '중지', or '정지'. Do NOT "
        "return a 'Pause', a 'Cancel' on an unrelated dialog, the window close (X) button, "
        "menu items, or the SEM image itself.\n"
        "Return strict JSON only. If no such Stop/Abort button is clearly visible, say so "
        "rather than guessing."
    )


def _abort_button_user_prompt() -> str:
    """Abort 버튼 탐지 사용자 프롬프트(엄격한 JSON 스키마)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "abort_button_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "abort_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "abort_button_bbox must tightly enclose the Stop/Abort button only. "
        "If none is clearly visible, set abort_button_visible=false, abort_button_bbox=null."
    )


def _confirm_system_prompt() -> str:
    """abort 확인 다이얼로그의 Yes/확인 탐지 시스템 프롬프트."""
    return (
        "A confirmation dialog has appeared asking the operator to confirm STOPPING / "
        "ABORTING the measurement run (e.g. 'Abort this run?', '측정을 중지하시겠습니까?').\n"
        "Locate the button that CONFIRMS the abort - usually 'Yes', 'OK', or '확인'. Do NOT "
        "return 'No', 'Cancel', or '취소'.\n"
        "Return strict JSON only. If no such confirm button is clearly visible, say so."
    )


def _confirm_user_prompt() -> str:
    """abort 확인 버튼 사용자 프롬프트(같은 스키마 재사용)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "abort_button_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "abort_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "abort_button_bbox must tightly enclose the Yes/OK/확인 confirm button only. "
        "If none is clearly visible, set abort_button_visible=false, abort_button_bbox=null."
    )


def _frame_to_webp_b64(frame_bgr: np.ndarray) -> tuple[str, int, int]:
    """grayscale/BGR/BGRA numpy 프레임을 WebP base64 로 인코딩. 반환 (b64, w, h).

    실장비/캡처 환경마다 채널 수가 다르다(mss 는 BGRA 4채널 등). 2/3/4 채널을 모두 RGB 로
    정규화해 3채널만 가정하다 cv2.error 로 터지지 않게 한다(ok_button 과 동일).
    """
    if frame_bgr.ndim == 2:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        rgb = cv2.cvtColor(frame_bgr[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"지원하지 않는 프레임 shape: {frame_bgr.shape}")
    image = Image.fromarray(rgb)
    return encode_image_webp(image, quality=90)


def _locate(frame_bgr, client, system_prompt, user_prompt):
    """공통 locate: 프레임 -> VLM -> bbox -> screen 중심 좌표. 없으면 None.

    스키마 키는 두 프롬프트가 동일하게 abort_button_visible / abort_button_bbox 를 쓴다.
    """
    image_b64, w, h = _frame_to_webp_b64(frame_bgr)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=system_prompt,
        user_text=user_prompt,
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("abort_button_visible") is not True:
        return None
    bbox_px = bbox_to_pixels(parsed.get("abort_button_bbox"), w, h, parsed.get("coord_system"))
    if bbox_px is None:
        return None
    center = bbox_center(bbox_px)
    return int(center["x"]), int(center["y"])


def locate_abort_button(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """전체 화면 프레임에서 Stop/Abort 버튼 중심의 SCREEN 픽셀 좌표를 반환(없으면 None)."""
    return _locate(frame_bgr, client, _abort_button_system_prompt(), _abort_button_user_prompt())


def locate_abort_confirm(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """abort 확인 다이얼로그의 Yes/확인 버튼 중심 SCREEN 좌표를 반환(없으면 None)."""
    return _locate(frame_bgr, client, _confirm_system_prompt(), _confirm_user_prompt())


__all__ = ["locate_abort_button", "locate_abort_confirm"]
