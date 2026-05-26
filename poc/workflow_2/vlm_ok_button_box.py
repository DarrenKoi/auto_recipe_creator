"""Align Fail 다이얼로그의 OK(확인) 버튼을 VLM 으로 찾는 locator.

primary correction(``align_fail_correct.py``) 흐름의 마지막 단계에서 쓴다:
crosshair 를 recipe-matched 점으로 옮긴(reposition) 뒤, 진행을 확정하는 OK 버튼을
눌러야 한다. OK 버튼은 SEM Monitor ROI 가 아니라 *전체 화면* 위 dialog 컨트롤이므로
좌표가 **screen 절대 픽셀**이다(=`SEMMonitorController.click_screen` 에 그대로 전달).

설계 경계(workflow_2 doc §8): VLM 은 버튼 *영역*만 식별한다. align key 좌표를
결정하는 일은 CV(``align_key_matcher``)가 하고, 여기서는 UI 버튼 위치만 찾는다.
``vlm_align_key_box.py`` 의 프롬프트/파싱 패턴과 동일한 헬퍼를 재사용한다.
"""

import cv2
import numpy as np
from PIL import Image

from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.util.json_utils import (
    bbox_center,
    bbox_to_pixels,
    extract_json,
)
from poc.workflow_1.vlm_client import Workflow1VLMClient


def _ok_button_system_prompt() -> str:
    """OK 버튼 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a CD-SEM / VeritySEM metrology tool that has "
        "paused on a wafer-alignment confirmation step. A small dialog is asking the "
        "operator to confirm the alignment after the crosshair has been placed.\n"
        "Locate the OK (확인) button that COMMITS / proceeds with the alignment. It is a "
        "clickable button, usually labelled 'OK', '확인', 'Apply', or 'Accept'. Do NOT "
        "return the Cancel / 취소 / Close / 닫기 button, and do NOT return menu items or "
        "the SEM image itself.\n"
        "Return strict JSON only. If no such OK button is clearly visible, say so rather "
        "than guessing."
    )


def _ok_button_user_prompt() -> str:
    """OK 버튼 탐지 사용자 프롬프트(엄격한 JSON 스키마)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "ok_button_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "ok_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "ok_button_bbox must tightly enclose the OK/확인 button only. "
        "If no OK button is clearly visible, set ok_button_visible=false, ok_button_bbox=null."
    )


def _frame_to_webp_b64(frame_bgr: np.ndarray) -> tuple[str, int, int]:
    """grayscale/BGR numpy 프레임을 WebP base64 로 인코딩한다. 반환 (b64, w, h)."""
    if frame_bgr.ndim == 2:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return encode_image_webp(image, quality=90)


def locate_ok_button(
    *,
    frame_bgr: np.ndarray,
    client: Workflow1VLMClient,
) -> tuple[int, int] | None:
    """전체 화면 프레임에서 OK 버튼 중심의 SCREEN 픽셀 좌표를 반환(없으면 None).

    relative_1000 bbox → ``bbox_1000_to_pixels`` (프레임 w·h 기준) → ``bbox_center``.
    여기서 프레임은 SEM ROI crop 이 아니라 *전체 화면* 이어야 하며, 그래서 반환 좌표가
    그대로 screen 절대 좌표가 된다(`click_screen` 에 전달).
    """
    image_b64, w, h = _frame_to_webp_b64(frame_bgr)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_ok_button_system_prompt(),
        user_text=_ok_button_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("ok_button_visible") is not True:
        return None
    bbox_1000 = normalize_bbox_1000(parsed.get("ok_button_bbox"))
    if bbox_1000 is None:
        return None
    bbox_px = bbox_1000_to_pixels(bbox_1000, w, h)
    center = bbox_center(bbox_px)
    return int(center["x"]), int(center["y"])
