"""'Rot..' 회전 대화상자 정찰 — 라벨/좌표만 읽는다. **클릭도 커서 이동도 하지 않는다.**

왜 필요한가: SEM align key 는 OM 과 다른 회전으로 등록되는 경우가 있다(사용자 2026-09-16).
그 회전이 어긋난 채로 template 을 매칭하면 점수가 떨어지므로 보정 전에 장비 회전을 recipe
값(cond.txt `Image_rotation`)으로 맞춰야 하고, 그 조작이 Optics.. 바로 아래의 'Rot..' 버튼이다.

그런데 그 대화상자가 **어떻게 생겼는지 이 저장소는 모른다**. Mac 에서는 그 화면을 볼 수 없고
오피스 이미지는 반출할 수 없으므로, 확인 게이트(`required=` 토큰)를 블라인드로 적어 넣으면
'있지도 않은 문구를 기다리다 멈추는' 액추에이터가 된다. 실제 문구를 알아내는 유일한 경로가
이 정찰이다 — `demonstration_rcs_control` 이 'Work Sheet' 버튼 문구를 같은 방식으로 알아냈다.

전량 화면 OCR 은 쓰지 않는다: PaddleOCR-VL 은 UI 전체 스크린샷에서 환각을 낸다
([[project_paddleocr_vl_screenshot_hallucination]]). 그래서 이 저장소 규약대로
**VLM 이 좌표, OCR 은 그 자리 crop 만 판독**한다.

안전:
  * 캡처 + VLM/OCR 호출뿐이다. 마우스는 건드리지 않으므로 장비가 돌고 있어도 안전하고
    `SAFE_MODE` 와 무관하다(액추에이터가 없으니 게이트할 것도 없다).
  * 따라서 대화상자는 **엔지니어가 손으로 열어 둔다**. 여는 것까지 자동화하는 것은
    이 정찰 결과로 문구가 확정된 다음이다.

절차 (오피스):
  1. tool 창(Remote Monitoring)을 열고, 아래 DIALOG_OPEN 을 False 로 둔 채 실행 →
     'Optics...' 와 'Rot..' 버튼이 어떤 문구로 읽히는지, 어디에 있는지를 받는다.
  2. 엔지니어가 'Rot..' 를 눌러 대화상자를 띄운다.
  3. DIALOG_OPEN = True 로 바꿔 다시 실행 → 대화상자의 제목/값 필드/버튼 문구를 받는다.
  4. 콘솔의 [DIGEST] 와 [TOKENS] 줄을 그대로 붙여 주면 액추에이터의 확인 게이트를
     그 문구로 박는다.

실행: uv run python poc/workflow_3/monitor/probe_rotation_dialog.py
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3 import DEBUG_IMAGE_DIR  # noqa: E402
from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
from poc.workflow_3.util import make_timestamp_tag
from poc.workflow_3.util.image_utils import capture_window
from poc.workflow_3.vlm.label_verify import (
    crop_box_around_point,
    read_text_near_point,
    tokens_from_text,
)
from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig, analyze_window_target

# ===========================================================================
# 실행 인자 - 여기만 고쳐 쓴다 (이 프로젝트는 CLI 인자를 쓰지 않는다).
# 셸 env 가 같은 이름으로 이기며, 어느 쪽이 쓰였는지는 시작 시 콘솔에 찍힌다.
# ===========================================================================

TOOL_ID = ""          # 대상 tool 창의 장비 ID. 비우면 열려 있는 아무 Remote Monitoring 창.
DIALOG_OPEN = False   # True = 'Rot..' 대화상자가 **이미 열려 있다**(엔지니어가 손으로 열었다).

LOG_COMPONENT = "probe_rotation_dialog"

# 버튼 문구는 crop 좌우 30% 를 담아 읽는다(demo flow 와 같은 비율). 값 필드는 숫자가
# 짧아 좌우를 더 넓게 본다 - 라벨('Rotation')과 값('0.0')이 떨어져 있을 수 있다.
_BUTTON_CROP = dict(left_ratio=0.30, right_ratio=0.30, half_height_ratio=0.05)
_FIELD_CROP = dict(left_ratio=0.45, right_ratio=0.45, half_height_ratio=0.06)


def _closed_targets() -> list[tuple[TargetConfig, dict]]:
    """대화상자가 닫힌 상태에서 볼 것 - 버튼 열의 Optics.. / Rot.. 두 개."""
    return [
        (TargetConfig(
            key="optics_button",
            description=(
                "the 'Optics...' button in the Remote Monitoring window's button "
                "area, located directly above the 'PM' button. Use the first letter "
                "'O' as the anchor, then return the center of the Optics button."
            ),
        ), _BUTTON_CROP),
        (TargetConfig(
            key="rot_button",
            description=(
                "the 'Rot..' button in the Remote Monitoring window's button area, "
                "located directly BELOW the 'Optics...' button in the same vertical "
                "column of buttons. Its label is short and starts with 'R'. Use that "
                "first letter 'R' as the anchor, then return the center of the button."
            ),
        ), _BUTTON_CROP),
    ]


def _open_targets() -> list[tuple[TargetConfig, dict]]:
    """대화상자가 열린 상태에서 볼 것 - 제목 / 값 필드 / 확정·닫기 버튼.

    설명문은 '무엇이 있어야 하는가' 가 아니라 '어디를 보라' 로 적는다. 실제 라벨을
    모르는 상태라 문구를 단정하면 VLM 이 없는 것을 찾아 엉뚱한 점을 준다.
    """
    return [
        (TargetConfig(
            key="rot_dialog_title",
            description=(
                "the title bar text of the small dialog window that is on top of the "
                "Remote Monitoring window (the rotation dialog). Return the center of "
                "the title text."
            ),
        ), _FIELD_CROP),
        (TargetConfig(
            key="rot_value_field",
            description=(
                "inside the small rotation dialog, the editable text box that holds a "
                "numeric angle value (for example '0.0' or '90'). Return the center of "
                "that number, not the label next to it."
            ),
        ), _FIELD_CROP),
        (TargetConfig(
            key="rot_confirm_button",
            description=(
                "inside the small rotation dialog, the button at the bottom that "
                "applies the value (a short label such as 'OK', 'Set' or 'Apply'). "
                "Return the center of that button."
            ),
        ), _BUTTON_CROP),
        (TargetConfig(
            key="rot_close_button",
            description=(
                "inside the small rotation dialog, the button that dismisses it "
                "without applying (a short label such as 'Close' or 'Cancel'). "
                "Return the center of that button."
            ),
        ), _BUTTON_CROP),
    ]


def _probe_one(image, target, crop_kwargs, *, debug_dir) -> dict:
    """요소 하나: VLM 으로 좌표 → 그 자리 crop 을 OCR → 읽힌 토큰을 그대로 찍는다.

    **판정하지 않는다.** 이 단계의 목적은 문구를 알아내는 것이므로 required/forbidden
    게이트를 걸지 않고 읽힌 것을 전부 보고한다(게이트는 문구가 확정된 뒤 액추에이터에).
    """
    result = analyze_window_target(
        None, "Remote Monitoring System", "uia", target,
        debug_image_dir=debug_dir,
        log_name=LOG_COMPONENT,
        component_name=LOG_COMPONENT,
        artifact_prefix=target.key,
        image=image,
    )
    point = getattr(result, "point", None)
    if point is None:
        print(f"  [{target.key}] 좌표 미검출 - 화면에 없거나 다른 창이 덮었을 수 있다")
        return {"key": target.key, "point": None, "tokens": []}

    box = crop_box_around_point(point, image.width, image.height, **crop_kwargs)
    read = read_text_near_point(
        image, box,
        debug_image_dir=debug_dir,
        timestamp_tag=make_timestamp_tag(time.time()),
        artifact_label=target.key,
        log_name=LOG_COMPONENT,
    )
    raw = (read.raw_text or "").strip() if read.ok else ""
    tokens = tokens_from_text(raw) if read.ok else []
    print(f"  [{target.key}] px={point} ocr_ok={read.ok}")
    print(f"      [TOKENS] {tokens}")
    if raw:
        print(f"      [RAW] {raw[:200]!r}")
    return {"key": target.key, "point": point, "tokens": tokens}


def main() -> int:
    tool_id = os.environ.get("TOOL_ID", "").strip() or TOOL_ID
    dialog_open = os.environ.get("DIALOG_OPEN", "").strip() or ("1" if DIALOG_OPEN else "0")
    dialog_open = dialog_open.lower() in {"1", "true", "yes", "on", "y"}

    print("=" * 70)
    print("[INFO] Rot.. 대화상자 정찰 (읽기 전용 - 클릭/커서 이동 없음)")
    print(f"[INFO] TOOL_ID={tool_id or '(아무 창)'} DIALOG_OPEN={dialog_open}")
    print("=" * 70)

    window, title, _backend = find_remote_monitoring_window(tool_id)
    if window is None:
        print("[ERROR] Remote Monitoring 창을 찾지 못했습니다 - tool 을 먼저 열어 두세요.")
        return 1
    print(f"[INFO] 대상 창: {title!r}")

    try:
        image = capture_window(window)
    except Exception as exc:
        print(f"[ERROR] 창 캡처 실패: {exc}")
        return 1
    print(f"[INFO] 캡처: {image.size} (모든 요소가 이 한 프레임을 공유한다)")

    tag = make_timestamp_tag()
    debug_dir = DEBUG_IMAGE_DIR / "rot_probe" / tag
    targets = _open_targets() if dialog_open else _closed_targets()
    print(f"\n[INFO] === {'대화상자 열림' if dialog_open else '버튼 열'} 정찰 "
          f"({len(targets)}개 요소) ===")
    rows = [_probe_one(image, t, ck, debug_dir=debug_dir) for t, ck in targets]

    found = [r["key"] for r in rows if r["point"] is not None]
    print(f"\n[INFO] crop/OCR 산출물: {debug_dir}")
    print(f"[DIGEST] rot_probe dialog_open={int(dialog_open)} "
          f"found={len(found)}/{len(rows)} " +
          " ".join(f"{r['key']}={'|'.join(r['tokens']) or '-'}" for r in rows))
    if not dialog_open:
        print("[INFO] 다음: 엔지니어가 'Rot..' 를 눌러 대화상자를 띄운 뒤 "
              "DIALOG_OPEN=1 로 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
