"""클릭 직전 tool row 확인 게이트 - 한 줄짜리 좁은 strip 만 OCR 해서 ID 를 확인한다.

기존 확인 게이트는 좌측 list **전체**를 PaddleOCR `Spotting:` 으로 읽었는데, 이 화면
에서는 응답이 garbage 라 정상 검출까지 막아 결국 비활성화됐다(workflow_select_tool
의 `_locate_tool_via_vlm` docstring 참조). PaddleOCR-VL 은 문서 파서라 layout -> crop
-> recognize 가 정상 경로이고, 전체 UI 스크린샷 한 장을 통째로 주면 환각이 잦다.

그래서 여기서는 VLM(coarse->fine)이 고른 **클릭점 주변 한 줄**만 잘라 읽는다. 입력이
문서 한 줄 모양이라 PaddleOCR 이 잘하는 형태이고, 읽을 텍스트도 몇 글자뿐이라
환각 여지가 작다.

역할 분리는 그대로 지킨다: 좌표는 VLM 이 정하고, OCR 은 "그 지점의 텍스트가 목표
ID 가 맞는가" 만 판정한다. OCR 은 클릭 좌표를 만들지 않는다.

판정(RowVerdict.status):
  confirmed  - strip 에서 목표 ID 와 canonical 일치하는 토큰을 읽음
  mismatch   - 목표와 같은 길이의 **다른** ID 토큰을 읽음 (= 옆 행을 가리킴)
  unreadable - 아무것도 못 읽었거나 ID 같은 토큰이 없음 (판정 보류)
  error      - OCR 호출 자체가 실패
"""

import os
import time
from dataclasses import dataclass, field

from poc.workflow_3.debug_artifacts import (
    debug_image_path,
    save_debug_json,
    save_debug_text,
    save_debug_webp,
)
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.rcs.tool_name_match import canonicalize
from poc.workflow_3.util import crop_image
from poc.workflow_3.vlm.prompts import build_ocr_assist_prompt
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

OCR_SERVICE_SLUG = "paddleocr-vl-1.5"
OCR_MAX_TOKENS = 512

# 확인 정책 - config.py 가 읽는 값이 아니라 이 모듈 전용 env 다.
#   off     : 확인하지 않음 (게이트 이전 동작)
#   lenient : 명시적 mismatch 일 때만 거부 (unreadable 은 통과) -- 기본값
#   strict  : confirmed 일 때만 통과 (unreadable 도 거부)
CONFIRM_POLICY_OFF = "off"
CONFIRM_POLICY_LENIENT = "lenient"
CONFIRM_POLICY_STRICT = "strict"
_VALID_POLICIES = {CONFIRM_POLICY_OFF, CONFIRM_POLICY_LENIENT, CONFIRM_POLICY_STRICT}


def _env_float(name: str, default: float) -> float:
    """env 실수값 로드 (파싱 실패 시 기본값)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARNING] {name}={raw!r} 파싱 실패 -> 기본값 {default} 사용")
        return default


# strip 크기 - 모두 이미지 크기 대비 비율. 한 행만 담되 ID 텍스트 전체는 들어가게.
STRIP_HALF_HEIGHT_RATIO = _env_float("SELECT_TOOL_ROW_STRIP_HALF_HEIGHT_RATIO", 0.011)
STRIP_LEFT_RATIO = _env_float("SELECT_TOOL_ROW_STRIP_LEFT_RATIO", 0.055)
STRIP_RIGHT_RATIO = _env_float("SELECT_TOOL_ROW_STRIP_RIGHT_RATIO", 0.055)
# 작은 strip 을 그대로 주면 OCR 이 못 읽으므로 최소 높이까지 확대한다.
STRIP_MIN_HEIGHT_PX = int(_env_float("SELECT_TOOL_ROW_STRIP_MIN_HEIGHT", 72))
STRIP_MAX_UPSCALE = _env_float("SELECT_TOOL_ROW_STRIP_MAX_UPSCALE", 6.0)

# 장비 ID 길이 범위 - 실제 ID 는 길이가 섞여 있다(MCD427 6자 ~ 4DCDB807 8자).
# 이 범위는 '옆 행 ID 인가'를 판정할 때만 쓴다(목표 일치 판정에는 안 쓴다).
ID_MIN_LEN = int(_env_float("SELECT_TOOL_ROW_ID_MIN_LEN", 5))
ID_MAX_LEN = int(_env_float("SELECT_TOOL_ROW_ID_MAX_LEN", 12))


@dataclass
class RowVerdict:
    """클릭점 행 확인 결과."""

    status: str
    target_tool_name: str
    read_text: str = ""
    read_tokens: list[str] = field(default_factory=list)
    mismatch_token: str = ""
    strip_box: dict | None = None
    strip_image_path: str = ""
    response_path: str = ""
    elapsed_sec: float = 0.0

    @property
    def confirmed(self) -> bool:
        """목표 ID 를 실제로 읽었는지."""
        return self.status == "confirmed"


def load_confirm_policy(default: str = CONFIRM_POLICY_LENIENT) -> str:
    """SELECT_TOOL_ROW_CONFIRM 에서 확인 정책을 읽는다."""
    raw = os.getenv("SELECT_TOOL_ROW_CONFIRM", "").strip().lower()
    if not raw:
        return default
    if raw in _VALID_POLICIES:
        return raw
    print(
        f"[WARNING] SELECT_TOOL_ROW_CONFIRM={raw!r} 는 알 수 없는 값 "
        f"({sorted(_VALID_POLICIES)}) -> 기본값 {default!r} 사용"
    )
    return default


def accepts(verdict: "RowVerdict | None", policy: str) -> bool:
    """정책에 따라 이 판정으로 클릭해도 되는지 결정한다."""
    if policy == CONFIRM_POLICY_OFF or verdict is None:
        return True
    if policy == CONFIRM_POLICY_STRICT:
        return verdict.status == "confirmed"
    # lenient: 명시적으로 다른 ID 를 읽은 경우에만 거부한다.
    return verdict.status != "mismatch"


def build_strip_box(point: dict, image_width: int, image_height: int) -> dict:
    """클릭점 주변의 한 줄짜리 strip crop box 를 만든다."""
    half_h = max(6, int(image_height * STRIP_HALF_HEIGHT_RATIO))
    left_pad = max(10, int(image_width * STRIP_LEFT_RATIO))
    right_pad = max(10, int(image_width * STRIP_RIGHT_RATIO))
    return {
        "left": max(0, int(point["x"]) - left_pad),
        "top": max(0, int(point["y"]) - half_h),
        "right": min(image_width, int(point["x"]) + right_pad),
        "bottom": min(image_height, int(point["y"]) + half_h),
    }


def _upscale_strip(strip_image):
    """OCR 가독성을 위해 strip 을 최소 높이까지 확대한다."""
    width, height = strip_image.size
    if height <= 0 or height >= STRIP_MIN_HEIGHT_PX:
        return strip_image, 1.0
    scale = min(STRIP_MAX_UPSCALE, STRIP_MIN_HEIGHT_PX / float(height))
    if scale <= 1.0:
        return strip_image, 1.0
    resized = strip_image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
    )
    return resized, scale


def _tokens_from_text(raw_text: str) -> list[str]:
    """OCR 평문에서 공백/개행 기준 토큰을 뽑는다."""
    tokens: list[str] = []
    for line in (raw_text or "").splitlines():
        for token in line.split():
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _looks_like_tool_id(token: str) -> bool:
    """원문 토큰이 장비 ID 모양인지.

    조건: 영숫자만 + 글자와 숫자를 **모두** 포함 + 길이가 ID 범위 안.
    실제 ID 는 길이가 섞여 있다 (MCD427/CCDM21 6자, 4DCDB807/RKHV3101 8자). 그래서
    '목표와 같은 길이'로 거르면 안 된다 - 6자 목표 옆에 8자 ID 가 있으면 그 오클릭을
    놓친다. 대신 'ID 모양이면서 목표와 다르면' mismatch 로 본다.

    옆 컬럼 텍스트가 strip 에 걸려도 대부분 여기서 탈락한다: IP(10.1.2.3)는 점 때문에
    isalnum 실패, 'Status'/'Model' 은 숫자 없음, 순수 카운트 숫자는 글자 없음.
    """
    cleaned = (token or "").strip()
    if not cleaned or not cleaned.isalnum():
        return False
    if not (ID_MIN_LEN <= len(cleaned) <= ID_MAX_LEN):
        return False
    return any(ch.isdigit() for ch in cleaned) and any(ch.isalpha() for ch in cleaned)


def classify_tokens(tokens: list[str], tool_name: str) -> tuple[str, str]:
    """읽은 토큰들을 목표 ID 와 대조해 (status, mismatch_token) 을 만든다.

    canonicalize 는 OCR 혼동 문자(O/0, I/1 등)를 대표 문자로 눌러주므로, 여기서의
    일치/불일치는 글꼴 혼동이 아니라 실제로 다른 ID 인지를 본다.

    목표 ID 가 하나라도 있으면 confirmed 다 (mismatch 판정보다 우선). strip 에 목표
    행과 옆 행이 함께 걸렸을 때 정상 클릭을 거부하지 않기 위한 순서다.
    """
    canonical_target = canonicalize(tool_name)
    if not canonical_target:
        return "unreadable", ""

    if canonical_target in [canonicalize(token) for token in tokens]:
        return "confirmed", ""

    # 목표와 같은 길이이면서 장비 ID 모양인 토큰 = 이 컬럼의 다른 장비 ID 로 본다.
    # ID 모양 판정은 **canonicalize 이전 원문**으로 한다: canonicalize 는 S->5, I->1
    # 처럼 글자를 숫자로 눌러버려서, 'Status' 같은 순수 단어가 '5TATU5' 가 되어 숫자를
    # 가진 ID 처럼 보이게 된다(오탐 -> 정상 클릭 거부).
    for token in tokens:
        if not _looks_like_tool_id(token):
            continue
        return "mismatch", canonicalize(token)
    return "unreadable", ""


def verify_tool_row_at_point(
    image,
    point: dict,
    tool_name: str,
    *,
    debug_image_dir,
    timestamp_tag: str,
    log_name: str = "tool_row_verify",
    component_name: str = "tool_row_verify",
    artifact_label: str = "tool_row_verify",
    client: Workflow1VLMClient | None = None,
) -> RowVerdict:
    """클릭점이 실제로 목표 tool ID 행 위에 있는지 좁은 strip OCR 로 확인한다."""
    started_at = time.time()
    normalized = (tool_name or "").strip()
    width, height = image.size
    strip_box = build_strip_box(point, width, height)
    strip_image = crop_image(image, strip_box)
    strip_image, upscale = _upscale_strip(strip_image)

    strip_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_strip.webp",
        model_name=OCR_SERVICE_SLUG,
        timestamp_tag=timestamp_tag,
    )
    save_debug_webp(strip_image, strip_path, quality=90)

    ocr_client = client
    if ocr_client is None:
        try:
            ocr_client = Workflow1VLMClient(
                service_slug=OCR_SERVICE_SLUG,
                timeout_sec=30.0,
                log_name=log_name,
            )
        except Exception as exc:
            print(f"[WARNING] row 확인 OCR client 생성 실패(확인 보류): {exc}")
            return RowVerdict(
                status="error",
                target_tool_name=normalized,
                strip_box=strip_box,
                strip_image_path=str(strip_path),
                elapsed_sec=time.time() - started_at,
            )

    system_message, user_text = build_ocr_assist_prompt(*strip_image.size)
    try:
        response = ocr_client.chat_with_image_path(
            image_path=strip_path,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[WARNING] row 확인 OCR 호출 실패(확인 보류): {exc}")
        log_work2_event(
            component=component_name,
            message="row_verify_ocr_failed",
            level="warning",
            log_name=log_name,
            target_tool_name=normalized,
            error=exc,
        )
        return RowVerdict(
            status="error",
            target_tool_name=normalized,
            strip_box=strip_box,
            strip_image_path=str(strip_path),
            elapsed_sec=time.time() - started_at,
        )

    raw_text = (response.text or "").strip()
    tokens = _tokens_from_text(raw_text)
    status, mismatch_token = classify_tokens(tokens, normalized)

    response_path = debug_image_path(
        debug_image_dir,
        f"{artifact_label}_strip_ocr.txt",
        model_name=OCR_SERVICE_SLUG,
        timestamp_tag=timestamp_tag,
    )
    save_debug_text(response_path, raw_text)
    save_debug_json(
        debug_image_path(
            debug_image_dir,
            f"{artifact_label}_strip_ocr.json",
            model_name=OCR_SERVICE_SLUG,
            timestamp_tag=timestamp_tag,
        ),
        {
            "target_tool_name": normalized,
            "point": point,
            "strip_box": strip_box,
            "strip_upscale": upscale,
            "raw_text": raw_text,
            "tokens": tokens,
            "status": status,
            "mismatch_token": mismatch_token,
        },
    )

    if status == "confirmed":
        print(f"[INFO] row 확인 OK: point 위 텍스트가 {normalized!r} 와 일치")
    elif status == "mismatch":
        print(
            f"[WARNING] row 확인 실패: 목표 {normalized!r} 가 아니라 "
            f"{mismatch_token!r} 를 읽음 -> 옆 행 클릭 위험, 거부"
        )
        log_work2_event(
            component=component_name,
            message="row_verify_mismatch",
            level="warning",
            log_name=log_name,
            target_tool_name=normalized,
            read_token=mismatch_token,
            raw_text=raw_text[:200],
        )
    else:
        print(f"[INFO] row 확인 보류(strip 판독 불가): raw={raw_text[:60]!r}")

    return RowVerdict(
        status=status,
        target_tool_name=normalized,
        read_text=raw_text,
        read_tokens=tokens,
        mismatch_token=mismatch_token,
        strip_box=strip_box,
        strip_image_path=str(strip_path),
        response_path=str(response_path),
        elapsed_sec=time.time() - started_at,
    )


__all__ = [
    "CONFIRM_POLICY_LENIENT",
    "CONFIRM_POLICY_OFF",
    "CONFIRM_POLICY_STRICT",
    "RowVerdict",
    "accepts",
    "build_strip_box",
    "classify_tokens",
    "load_confirm_policy",
    "verify_tool_row_at_point",
]
