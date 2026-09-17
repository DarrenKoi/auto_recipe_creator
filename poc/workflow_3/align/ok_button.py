"""Align Fail 다이얼로그의 OK(확인) 버튼을 **그 다이얼로그 안에서만** 찾는 locator.

primary correction(``align/correction.py``) 흐름의 마지막 단계에서 쓴다:
crosshair 를 recipe-matched 점으로 옮긴(reposition) 뒤, 진행을 확정하는 OK 버튼을
눌러야 한다. OK 버튼은 SEM Monitor ROI 가 아니라 *전체 화면* 위 dialog 컨트롤이므로
좌표가 **screen 절대 픽셀**이다(=`SEMMonitorController.click_screen` 에 그대로 전달).

**OK 는 흔한 버튼이다** (2026-09-15 오피스: `OK_CLICK=1` 첫 실행에서 다른 창의 OK 를
눌렀다). 그래서 전체 프레임에 "OK 버튼" 을 묻지 않고 두 단계로 좁힌다 -
시연 경로의 계약 ②("첫 요소의 라벨이 '창이 떴다'는 유일한 증거") 와 같은 구조다:

  1. **다이얼로그 먼저** - alignment 확인 다이얼로그의 bbox 를 VLM 에 한 요소로 묻고,
     그 crop 을 PaddleOCR 로 읽어 `OK_DIALOG_REQUIRED` 토큰(기본 `align`)이 있을 때만
     진행한다. 읽혔는데 토큰이 없으면 **다른 창**이므로 정책과 무관하게 거부한다.
  2. **OK 는 그 crop 안에서만** 찾고, 그 지점의 라벨을 OCR 로 읽어 `cancel/close/취소/
     닫기` 류(`OK_BUTTON_FORBIDDEN`)가 읽히면 어떤 정책에서도 누르지 않는다.

정책은 `ALIGN_OK_CONFIRM` (lenient 기본 | strict | off) - 시연/공유 요청 게이트와 같은
의미다: lenient 는 '못 읽음' 만 통과시키고, 읽혔는데 다른 문구면 거부한다.

설계 경계(workflow_2 doc §8): VLM 은 버튼 *영역*만 식별한다. align key 좌표를
결정하는 일은 CV(``align.matching.engine``)가 한다.
"""

import os
import time

import cv2
import numpy as np
from PIL import Image

from poc.workflow_3.util.event_dir import debug_root
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.rcs.tool_row_verify import (
    CONFIRM_POLICY_LENIENT,
    CONFIRM_POLICY_OFF,
    CONFIRM_POLICY_STRICT,
)
from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_center,
    bbox_to_pixels,
    extract_json,
)
from poc.workflow_3.vlm.label_verify import read_text_near_point
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

# 다이얼로그 확인 토큰(대소문자 무시 부분 일치). 오피스 실제 문구를 모르므로 env 로
# 바꿀 수 있고, 매 실행 OCR 원문을 콘솔에 남겨 대조한다: ALIGN_OK_DIALOG_TOKENS="align,얼라인"
OK_DIALOG_REQUIRED = tuple(
    t.strip().lower() for t in os.environ.get("ALIGN_OK_DIALOG_TOKENS", "align").split(",") if t.strip()
)
OK_BUTTON_REQUIRED = ("ok", "확인")
OK_BUTTON_FORBIDDEN = ("cancel", "close", "abort", "exit", "취소", "닫기", "중단")
# OK 라벨 crop 크기(다이얼로그 crop 기준 비율). 버튼 하나만 담는다.
OK_LABEL_HALF_W_RATIO = 0.12
OK_LABEL_HALF_H_RATIO = 0.06

VERDICT_CONFIRMED = "confirmed"
VERDICT_MISMATCH = "mismatch"
VERDICT_UNREADABLE = "unreadable"


def load_ok_confirm_policy() -> str:
    """ALIGN_OK_CONFIRM 을 읽는다. 오타는 strict 로 폴백해 게이트가 조용히 열리지 않게 한다."""
    raw = os.environ.get("ALIGN_OK_CONFIRM", CONFIRM_POLICY_LENIENT).strip().lower()
    return raw if raw in (CONFIRM_POLICY_LENIENT, CONFIRM_POLICY_STRICT, CONFIRM_POLICY_OFF) else CONFIRM_POLICY_STRICT


def classify_text(read_ok: bool, tokens, required, forbidden=()) -> str:
    """OCR 토큰을 confirmed / mismatch / unreadable 로 가른다.

    forbidden 은 required 보다 먼저 보고 mismatch 다. 읽혔는데(토큰 있음) required 가
    없으면 **다른 창/버튼**이므로 mismatch - unreadable 과 갈라야 lenient 가 오클릭을
    통과시키지 않는다(점유 게이트의 `mc_id_mismatch` 와 같은 구분).
    """
    words = [t.lower() for t in (tokens or []) if not t.startswith("[")]  # OCR 레이아웃 태그 제외
    if not read_ok or not words:
        return VERDICT_UNREADABLE
    if any(f in w for w in words for f in forbidden):
        return VERDICT_MISMATCH
    if any(r in w for w in words for r in required):
        return VERDICT_CONFIRMED
    return VERDICT_MISMATCH


def accepts(verdict: str, policy: str) -> bool:
    """mismatch 는 어떤 정책에서도 거부. lenient/off 는 unreadable 만 통과."""
    if verdict == VERDICT_MISMATCH:
        return False
    if policy in (CONFIRM_POLICY_LENIENT, CONFIRM_POLICY_OFF):
        return True
    return verdict == VERDICT_CONFIRMED


def _dialog_system_prompt() -> str:
    """단계 1: 다이얼로그 창 bbox."""
    return (
        "You analyse a screenshot of a CD-SEM / VeritySEM metrology tool that has "
        "paused on a wafer-alignment confirmation step. A small dialog window (message "
        "box) is asking the operator to confirm the alignment after the crosshair has "
        "been placed.\n"
        "Locate that dialog WINDOW as a whole: its bbox must enclose the dialog's title "
        "bar, message text and its buttons. Do NOT return the main tool window, the SEM "
        "image, or any other dialog. Return strict JSON only. If no such alignment dialog "
        "is clearly visible, say so rather than guessing."
    )


def _dialog_user_prompt() -> str:
    return (
        "Return JSON with this exact schema:\n"
        '{"dialog_visible": true, "coord_system": "relative_1000", '
        '"dialog_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0}, "confidence": 0.0}\n'
        "If no alignment dialog is clearly visible, set dialog_visible=false, dialog_bbox=null."
    )


def _ok_button_system_prompt() -> str:
    """단계 2: 다이얼로그 crop 안의 OK 버튼."""
    return (
        "This image is a cropped dialog from a CD-SEM / VeritySEM tool asking the "
        "operator to confirm a wafer alignment.\n"
        "Locate the OK (확인) button that COMMITS / proceeds with the alignment. It is a "
        "clickable button, usually labelled 'OK', '확인', 'Apply', or 'Accept'. Do NOT "
        "return the Cancel / 취소 / Close / 닫기 button.\n"
        "Return strict JSON only. If no such OK button is clearly visible, say so rather "
        "than guessing."
    )


def _ok_button_user_prompt() -> str:
    return (
        "Return JSON with this exact schema:\n"
        '{"ok_button_visible": true, "coord_system": "relative_1000", '
        '"ok_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0}, "confidence": 0.0}\n'
        "ok_button_bbox must tightly enclose the OK/확인 button only. "
        "If no OK button is clearly visible, set ok_button_visible=false, ok_button_bbox=null."
    )


def _frame_to_rgb_image(frame_bgr: np.ndarray) -> Image.Image:
    """grayscale/BGR/BGRA numpy 프레임을 PIL RGB 로 정규화한다.

    실장비 capture_screen() 은 채널 수가 환경마다 다르다(mss 는 BGRA 4채널 등).
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
    return Image.fromarray(rgb)


def _frame_to_webp_b64(frame_bgr: np.ndarray) -> tuple[str, int, int]:
    """numpy 프레임 -> WebP base64. 반환 (b64, w, h)."""
    return encode_image_webp(_frame_to_rgb_image(frame_bgr), quality=90)


def _locate_bbox(client, image: Image.Image, system_message: str, user_text: str,
                 visible_key: str, bbox_key: str) -> dict | None:
    """VLM 에 요소 하나의 bbox 를 묻고 픽셀 bbox 로 돌려준다(없으면 None)."""
    image_b64, w, h = encode_image_webp(image, quality=90)
    response = client.chat_with_image_b64(
        image_b64=image_b64, system_message=system_message, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get(visible_key) is not True:
        return None
    return bbox_to_pixels(parsed.get(bbox_key), w, h, parsed.get("coord_system"))


# probe_align_dialog 판정. 보정 사이클이 클릭 전에 "align fail 이 아직 살아 있나" 를
# 묻는 데도 쓴다(monitor/cycle.py) - OK 를 찾을 때와 **같은 판정**이어야 두 곳이 다른
# 답을 내지 않는다.
DIALOG_PRESENT = "present"   # alignment 다이얼로그로 확인(lenient 면 '못 읽음' 포함).
DIALOG_ABSENT = "absent"     # VLM 이 다이얼로그를 못 봄.
DIALOG_OTHER = "other"       # 창은 있는데 문구가 alignment 가 아님 - 다른 창.


def probe_align_dialog(
    frame_bgr: np.ndarray,
    client: Workflow1VLMClient,
    *,
    ocr_client=None,
    confirm_policy: str | None = None,
    debug_image_dir=None,
) -> tuple[str, dict | None]:
    """전체 화면에 alignment 다이얼로그가 떠 있는가 -> (DIALOG_*, 다이얼로그 픽셀 bbox).

    다이얼로그 bbox(VLM) -> 그 crop 의 문구 OCR 확인. 읽혔는데 `OK_DIALOG_REQUIRED` 가
    없으면 다른 창이다(DIALOG_OTHER). 예외는 삼키지 않는다 - 호출부가 '못 봄' 과 '판독
    실패' 를 갈라야 한다.
    """
    policy = confirm_policy or load_ok_confirm_policy()
    image = _frame_to_rgb_image(frame_bgr)
    artifact_dir = debug_image_dir or (debug_root() / "ok_button" / str(time.time_ns()))

    dialog = _locate_bbox(client, image, _dialog_system_prompt(), _dialog_user_prompt(),
                          "dialog_visible", "dialog_bbox")
    if dialog is None:
        # 못 봤을 때가 VLM 누락인지 오피스에서 대조할 근거는 이 프레임뿐이다.
        save_debug_jpeg(image, artifact_dir / "screen_no_dialog.jpg", quality=85)
        print("[INFO] align 다이얼로그: 보이지 않음")
        return DIALOG_ABSENT, None
    dialog_crop = image.crop((dialog["left"], dialog["top"], dialog["right"], dialog["bottom"]))
    save_debug_jpeg(dialog_crop, artifact_dir / "dialog.jpg")

    dialog_read = read_text_near_point(
        image, dialog, debug_image_dir=artifact_dir, timestamp_tag="dialog",
        artifact_label="ok_dialog", log_name="ok_button", client=ocr_client,
    )
    verdict = classify_text(dialog_read.ok, dialog_read.tokens or dialog_read.raw_text.split(),
                            OK_DIALOG_REQUIRED)
    print(f"[INFO] align 다이얼로그: OCR={dialog_read.raw_text!r} -> {verdict} "
          f"(required={OK_DIALOG_REQUIRED}, policy={policy})")
    if not accepts(verdict, policy):
        print("[WARNING] align 다이얼로그: 떠 있는 창이 alignment 다이얼로그로 확인되지 않음")
        return DIALOG_OTHER, dialog
    return DIALOG_PRESENT, dialog


def locate_ok_button(
    *,
    frame_bgr: np.ndarray,
    client: Workflow1VLMClient,
    ocr_client=None,
    confirm_policy: str | None = None,
    debug_image_dir=None,
) -> tuple[int, int] | None:
    """전체 화면 프레임에서 Align 다이얼로그의 OK 버튼 중심 SCREEN 픽셀 좌표(없으면 None).

    다이얼로그 확인(`probe_align_dialog`) -> crop 안 OK bbox -> OK 라벨 OCR 확인.
    어느 게이트에서든 거부되면 None 이며, 호출부는 정상 not-found(escalate)로 다룬다.
    프레임은 *전체 화면* 이어야 반환 좌표가 그대로 screen 절대 좌표가 된다(`click_screen`).
    """
    policy = confirm_policy or load_ok_confirm_policy()
    image = _frame_to_rgb_image(frame_bgr)
    artifact_dir = debug_image_dir or (debug_root() / "ok_button" / str(time.time_ns()))

    # 게이트 1: 이 창이 정말 alignment 다이얼로그인가.
    state, dialog = probe_align_dialog(
        frame_bgr, client, ocr_client=ocr_client, confirm_policy=policy,
        debug_image_dir=artifact_dir,
    )
    if state != DIALOG_PRESENT:
        print("[INFO] OK 탐지: alignment 다이얼로그가 확인되지 않아 클릭하지 않음")
        return None
    dialog_crop = image.crop((dialog["left"], dialog["top"], dialog["right"], dialog["bottom"]))

    # 단계 2: OK 는 다이얼로그 crop 안에서만.
    ok_bbox = _locate_bbox(client, dialog_crop, _ok_button_system_prompt(), _ok_button_user_prompt(),
                           "ok_button_visible", "ok_button_bbox")
    if ok_bbox is None:
        print("[INFO] OK 탐지: 다이얼로그 안에 OK 버튼이 보이지 않음")
        return None
    center = bbox_center(ok_bbox)
    x, y = dialog["left"] + int(center["x"]), dialog["top"] + int(center["y"])

    # 게이트 2: 그 지점의 라벨이 OK 인가(취소/닫기면 어떤 정책에서도 거부).
    half_w = max(12, int(dialog_crop.width * OK_LABEL_HALF_W_RATIO))
    half_h = max(8, int(dialog_crop.height * OK_LABEL_HALF_H_RATIO))
    label_box = {"left": max(0, x - half_w), "top": max(0, y - half_h),
                 "right": min(image.width, x + half_w), "bottom": min(image.height, y + half_h)}
    label_read = read_text_near_point(
        image, label_box, debug_image_dir=artifact_dir, timestamp_tag="ok",
        artifact_label="ok_label", log_name="ok_button", client=ocr_client,
    )
    verdict = classify_text(label_read.ok, label_read.tokens or label_read.raw_text.split(),
                            OK_BUTTON_REQUIRED, OK_BUTTON_FORBIDDEN)
    print(f"[INFO] OK 탐지: 버튼 라벨 OCR={label_read.raw_text!r} -> {verdict} at screen=({x}, {y})")
    if not accepts(verdict, policy):
        print("[WARNING] OK 탐지: 버튼 라벨이 OK 로 확인되지 않아 클릭하지 않음")
        return None
    return x, y
