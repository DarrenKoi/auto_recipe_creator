"""측정 abort(Stop/중지) 버튼을 VLM 으로 찾는 locator.

측정 abort 잡(`workflow_3e/abort_cycle.run_abort_cycle`)의 마지막 단계에서 쓴다: MES 가
'N회 연속 측정 실패' 임계 알람을 쏜 tool 에 접속한 뒤, 진행 중인 측정을 멈추는 Stop/Abort
버튼을 눌러야 한다. 버튼은 RCS tool 창 위의 컨트롤이라 좌표가 **screen 절대 픽셀**이다.

설계 경계(align/ok_button.py 와 동일): VLM 은 버튼 *영역*만 식별한다. abort 여부는 MES 가
이미 결정했다(임계 알람). 여기서는 어떤 UI 버튼을 누를지 위치만 찾는다.

**라벨 확인 게이트**: 좌표를 얻은 뒤 그 점 주변을 좁게 잘라 OCR 로 읽어 정말 Stop/Abort
버튼인지 확인한다(`rcs/tool_row_verify.py` 가 tool 행에 쓰는 것과 같은 `vlm/label_verify`
도구). 진행 중인 측정을 멈추는 클릭은 되돌릴 수 없어, 기본 정책이 **strict** 다 - 확인되지
않으면 누르지 않고 엔지니어에게 넘긴다("미확인 시 클릭 금지").

정책 상수 이름/값은 tool_row_verify 와 일부러 맞췄다(같은 어휘). 다만 workflow_3e 가
workflow_3/rcs 내부에 묶이지 않도록 여기서 따로 정의한다 - 공유하는 것은 어휘지 로직이
아니다(분류 규칙은 tool ID 냐 버튼 라벨이냐에 따라 다르다).
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import bbox_center, bbox_to_pixels, extract_json
from poc.workflow_3.vlm.label_verify import (
    crop_box_around_point,
    label_matches,
    read_text_near_point,
)
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

# --- 라벨 확인 정책 ---
#   off     : 게이트 비활성(검증 전 롤백 경로)
#   lenient : 명시적 mismatch 일 때만 거부 (unreadable 은 통과)
#   strict  : confirmed 일 때만 통과 (unreadable 도 거부) -- 기본값
ABORT_LABEL_POLICY_OFF = "off"
ABORT_LABEL_POLICY_LENIENT = "lenient"
ABORT_LABEL_POLICY_STRICT = "strict"
_VALID_LABEL_POLICIES = {
    ABORT_LABEL_POLICY_OFF,
    ABORT_LABEL_POLICY_LENIENT,
    ABORT_LABEL_POLICY_STRICT,
}

# 확인에 인정할 버튼 라벨. 프롬프트가 찾으라고 지시한 것과 같은 집합을 쓴다.
ABORT_BUTTON_LABELS = ("Stop", "Abort", "중지", "정지")
CONFIRM_BUTTON_LABELS = ("Yes", "OK", "확인")

# 버튼 크기 crop(점 주변). tool 행 strip 과 달리 가로로 길 필요가 없다.
LABEL_CROP_LEFT_RATIO = 0.04
LABEL_CROP_RIGHT_RATIO = 0.04
LABEL_CROP_HALF_HEIGHT_RATIO = 0.02


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


@dataclass
class ButtonLabelVerdict:
    """버튼 라벨 확인 결과. status = confirmed | mismatch | unreadable."""

    status: str
    matched_label: str = ""
    tokens: list = field(default_factory=list)
    raw_text: str = ""
    crop_image_path: str = ""


def load_abort_label_policy(default: str = ABORT_LABEL_POLICY_STRICT) -> str:
    """`MEAS_FAIL_ABORT_LABEL_CONFIRM` 에서 정책을 읽는다. 미지정/오타면 default."""
    raw = (os.environ.get("MEAS_FAIL_ABORT_LABEL_CONFIRM") or "").strip().lower()
    if raw in _VALID_LABEL_POLICIES:
        return raw
    if raw:
        print(f"[WARNING] 알 수 없는 abort 라벨 확인 정책 '{raw}' - '{default}' 사용")
    return default


def classify_button_tokens(tokens, expected_labels) -> ButtonLabelVerdict:
    """OCR 토큰이 기대 라벨 중 하나인지 판정한다.

    토큰이 아예 없으면 unreadable(작은 버튼 crop 은 OCR 이 비는 일이 흔하다). 무언가는
    읽혔는데 기대 라벨이 없으면 mismatch - 다른 버튼을 가리키고 있다는 뜻이라 가장 위험한
    신호다.
    """
    cleaned = [t for t in (tokens or []) if str(t).strip()]
    if not cleaned:
        return ButtonLabelVerdict(status="unreadable", tokens=list(cleaned))
    for label in expected_labels:
        if label_matches(cleaned, label):
            return ButtonLabelVerdict(
                status="confirmed", matched_label=label, tokens=list(cleaned)
            )
    return ButtonLabelVerdict(status="mismatch", tokens=list(cleaned))


def accepts_label(verdict: "ButtonLabelVerdict | None", policy: str) -> bool:
    """정책에 따라 이 좌표를 클릭해도 되는지. verdict None = 확인 자체가 불가."""
    if policy == ABORT_LABEL_POLICY_OFF:
        return True
    if verdict is None:
        # OCR 을 못 돌린 경우. strict 는 금지, lenient 는 '아님'이라는 증거가 없으니 통과.
        return policy == ABORT_LABEL_POLICY_LENIENT
    if policy == ABORT_LABEL_POLICY_STRICT:
        return verdict.status == "confirmed"
    return verdict.status != "mismatch"


def verify_button_label_at_point(
    image,
    point_xy: tuple,
    expected_labels,
    *,
    debug_image_dir,
    timestamp_tag: str,
    artifact_label: str,
    client=None,
) -> "ButtonLabelVerdict | None":
    """VLM 이 고른 점 주변을 OCR 로 읽어 라벨을 확인한다. 확인 불가면 None.

    좌표를 만들지 않는다 - 이미 정해진 점을 검증만 한다(프로젝트 역할 분담).
    """
    width, height = image.size
    box = crop_box_around_point(
        {"x": int(point_xy[0]), "y": int(point_xy[1])},
        width,
        height,
        left_ratio=LABEL_CROP_LEFT_RATIO,
        right_ratio=LABEL_CROP_RIGHT_RATIO,
        half_height_ratio=LABEL_CROP_HALF_HEIGHT_RATIO,
    )
    read = read_text_near_point(
        image,
        box,
        debug_image_dir=debug_image_dir,
        timestamp_tag=timestamp_tag,
        artifact_label=artifact_label,
        log_name="abort_label_verify",
        client=client,
    )
    if not read.ok:
        return None
    verdict = classify_button_tokens(read.tokens, expected_labels)
    verdict.raw_text = read.raw_text
    verdict.crop_image_path = read.crop_image_path
    return verdict


def locate_abort_button(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """전체 화면 프레임에서 Stop/Abort 버튼 중심의 SCREEN 픽셀 좌표를 반환(없으면 None)."""
    return _locate(frame_bgr, client, _abort_button_system_prompt(), _abort_button_user_prompt())


def locate_abort_confirm(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """abort 확인 다이얼로그의 Yes/확인 버튼 중심 SCREEN 좌표를 반환(없으면 None)."""
    return _locate(frame_bgr, client, _confirm_system_prompt(), _confirm_user_prompt())


__all__ = [
    "ABORT_BUTTON_LABELS",
    "ABORT_LABEL_POLICY_LENIENT",
    "ABORT_LABEL_POLICY_OFF",
    "ABORT_LABEL_POLICY_STRICT",
    "ButtonLabelVerdict",
    "CONFIRM_BUTTON_LABELS",
    "accepts_label",
    "classify_button_tokens",
    "load_abort_label_policy",
    "locate_abort_button",
    "locate_abort_confirm",
    "verify_button_label_at_point",
]
