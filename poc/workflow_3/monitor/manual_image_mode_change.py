"""열린 tool 창에서 **live 이미지 모드**(OM / OM-D / SEM)를 드롭다운으로 바꾼다.

왜 필요한가: OM-D 는 OM 의 명암이 **반전된** 화면이다. recipe 에 등록된 align key 와 live
화면의 모드가 어긋나면 같은 자리를 보고도 반전된 상을 매칭하게 되어 점수가 나지 않는다.
보정/탐색 전에 화면 모드를 recipe 쪽에 맞추는 조작이 필요하고, 이 진입점은 그 조작 하나만
손으로 시험한다(알람과 무관, 엔지니어가 먼저 연 tool 창에 붙는다).

  1. 제목에 EQP_ID 가 든 Remote Monitoring 창에 붙는다(접속하지 않는다 - `manual_click_button`
     과 같은 규약).
  2. VLM 이 image mode 콤보의 **드롭다운 화살표**를 찍고, 그 **왼쪽**을 OCR 로 읽어 현재 모드를
     확인한다. 아는 모드가 안 읽히면 그 콤보가 아니라고 보고 누르지 않는다.
  3. 이미 목표 모드면 열지 않는다(exit 6).
  4. 화살표를 누르고, 열린 목록을 **화살표 기준 영역 crop** 안에서만 찾아 목표 항목을 확인 후
     누른다. 목록은 tool 본 화면이 아니므로 탐색 범위를 이렇게 묶는 것이 유일한 안전장치다.
  5. 콤보 값이 목표 모드로 읽힐 때까지 반복 확인한다. 실패해도 **다시 누르지 않는다**.

`ButtonSpec`/`_confirm_point` 를 쓰지 않는 이유는 하나다: 그 확인은 토큰 **부분 일치**인데
여기 라벨은 `OM` 이 `OM-D` 의 접두라, 부분 일치면 OM-D 화면을 OM 으로 읽고 OM-D 행 대신 OM 행을
눌러도 통과한다. 이 파일은 모드를 **단어 전체**로만 인정한다(`read_mode`). 좌표(VLM)와 클릭
성사 조건(`build_click_kit` -> `perform_remote_click`)은 그대로 공유한다 - 포크 금지 규약.

실행: uv run python poc/workflow_3/monitor/manual_image_mode_change.py
리허설(클릭 차단, 현재 모드 판독까지만): SAFE_MODE=1 uv run python ...
종료 코드: 0=바꿈(또는 리허설), 2=사전조건 실패, 3=화살표/항목 못 찾음,
          4=모드 판독 실패/항목 라벨 불일치, 5=눌렀지만 바뀐 것을 확인 못 함, 6=이미 그 모드
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 판정 로직만 모듈 수준에서 import 한다(Mac 에서 시험할 수 있게). RCS/pywinauto/pynput 을
# 끌고 오는 것은 전부 `main()` 안에서 - 이 파일의 테스트는 Windows 없이 돈다.
from poc.workflow_3.monitor.button_registry import poll_until  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 셸 env(괄호 안 이름)가 있으면 env 가 이긴다.
# ===========================================================================

EQP_ID = "MCD513"            # (MANUAL_IMAGE_MODE_EQP_ID) 제목에 이 ID 가 든 tool 창
TARGET_MODE = "OM-D"         # (MANUAL_IMAGE_MODE_TARGET) IMAGE_MODES 의 key

SETTLE_SEC = 1.0             # (MANUAL_IMAGE_MODE_SETTLE_SEC) 화살표 클릭 -> 목록이 그려질 대기
VERIFY_TIMEOUT_SEC = 10.0    # (MANUAL_IMAGE_MODE_VERIFY_TIMEOUT_SEC) 바뀐 값 확인 상한
VERIFY_POLL_SEC = 0.5        # (MANUAL_IMAGE_MODE_VERIFY_POLL_SEC)
# 선택하지 못하고 끝날 때 열린 목록을 Esc 로 닫는다 - 엔지니어 화면에 목록을 남기지 않는다.
# **원격이 Esc 를 중계하는지는 오피스 미검증**이다. 부작용이 보이면 0 으로 둔다(대신 콘솔이
# "Esc 로 닫으세요" 를 남긴다).
CLOSE_WITH_ESCAPE = True

# 콤보의 현재 값은 화살표 **왼쪽**에 있다 - 좌우 비대칭 crop 이다(오른쪽을 넓히면 옆 위젯을
# 함께 읽는다). 값 한 줄만 담도록 세로는 좁게.
VALUE_LEFT_RATIO = 0.060
VALUE_RIGHT_RATIO = 0.004
VALUE_HALF_HEIGHT_RATIO = 0.012

# 열린 목록이 그려질 영역(화살표 기준). 목록은 콤보 폭이라 화살표에서 **왼쪽**으로 퍼지고,
# 보통 아래로 열리지만 화면 아래쪽에서는 위로 열린다 - 위쪽도 조금 연다.
LIST_LEFT_RATIO = 0.10
LIST_RIGHT_RATIO = 0.03
LIST_UP_RATIO = 0.06
LIST_DOWN_RATIO = 0.30

# 항목 라벨 확인 crop - **행 한 줄** 크기여야 한다. 넓으면 OM 행과 OM-D 행을 함께 읽어
# `read_mode` 가 모호로 보고 거부한다(거부는 옳은 방향이지만 그때는 이 값을 줄일 것).
ITEM_HALF_WIDTH_RATIO = 0.035
ITEM_HALF_HEIGHT_RATIO = 0.009
# 2단 로케이터의 fine crop 세로 하한. 촘촘한 목록 행은 기본 28px 하한이 위아래 행을 삼킨다.
ITEM_VERTICAL_PAD_MIN_PX = 10

# ===========================================================================
# 모드 표 - needle 은 **정규화된 단어 전체**다(영숫자만 남기고 소문자). 'om' 이 'om-d' 의
# 접두라 부분 일치를 쓸 수 없다. OCR 이 'OM-D' 를 'OM','D' 로 쪼개면 어느 모드도 확정되지
# 않고(fail-closed) 읽힌 토큰이 콘솔에 남는다 - 그 값을 보고 여기에 형태를 추가한다.
# 드롭다운 항목은 OM / OM-D / SEM 셋이다(2026-09-22 사용자 확인). SEM 에는 dark 모드가 없다
# - 즉 극성 반전은 OM <-> OM-D 사이에서만 일어난다.
# ===========================================================================

IMAGE_MODES = {
    "OM": frozenset({"om"}),
    "OM-D": frozenset({"omd"}),
    "SEM": frozenset({"sem"}),
}

ARROW_DESCRIPTION = (
    "the small downward-pointing arrow of the image mode combo box. That combo box shows "
    "the current image mode as text (for example 'OM', 'OM-D' or 'SEM') and the arrow is at "
    "the RIGHT end of it. The combo box sits directly ABOVE the buttons labeled 'Optics...' "
    "and 'OM ABC', next to the live image. Point at the center of that arrow, not at the "
    "text on its left and not at the buttons below it."
)

RESULT_CHANGED = "changed"
RESULT_ALREADY = "already"
RESULT_REHEARSAL = "rehearsal"
RESULT_ARROW_NOT_LOCATED = "arrow_not_located"
RESULT_MODE_UNREADABLE = "mode_unreadable"
RESULT_ITEM_NOT_LOCATED = "item_not_located"
RESULT_ITEM_NOT_CONFIRMED = "item_not_confirmed"
RESULT_UNVERIFIED = "unverified"

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2
EXIT_NOT_FOUND = 3
EXIT_NOT_CONFIRMED = 4
EXIT_NOT_VERIFIED = 5
EXIT_ALREADY = 6

EXIT_BY_RESULT = {
    RESULT_CHANGED: EXIT_OK,
    RESULT_REHEARSAL: EXIT_OK,
    RESULT_ALREADY: EXIT_ALREADY,
    RESULT_ARROW_NOT_LOCATED: EXIT_NOT_FOUND,
    RESULT_ITEM_NOT_LOCATED: EXIT_NOT_FOUND,
    RESULT_MODE_UNREADABLE: EXIT_NOT_CONFIRMED,
    RESULT_ITEM_NOT_CONFIRMED: EXIT_NOT_CONFIRMED,
    RESULT_UNVERIFIED: EXIT_NOT_VERIFIED,
}


def _word(token: str) -> str:
    """OCR 토큰을 비교용 단어로 정규화한다(영숫자만, 소문자). 'OM-D' -> 'omd'."""
    return "".join(ch for ch in str(token).lower() if ch.isalnum())


def read_mode(tokens, modes=IMAGE_MODES):
    """읽은 토큰에서 image mode 하나를 고른다. 확정 못 하면 None.

    **단어 전체** 비교다. 부분 일치를 쓰면 'om' 이 'om-d' 안에 들어 있어 OM-D 화면을 OM 으로
    읽고, 목록에서도 OM-D 행 대신 OM 행을 눌러도 통과한다.

    모드가 **둘 이상** 읽히면 None 이다 - crop 이 여러 행을 삼켰다는 뜻이라 확인이 아니다
    (넓은 crop 이 위 줄만 읽던 2026-09-18 실패와 같은 종류).
    """
    words = {_word(token) for token in tokens}
    hits = [key for key, forms in modes.items() if words & set(forms)]
    return hits[0] if len(hits) == 1 else None


def item_description(mode: str, modes=IMAGE_MODES) -> str:
    """목록 항목 프롬프트. 형제 이름은 **프롬프트에만** 넣는다.

    OCR `forbidden` 에 넣으면 안 된다 - 형제 항목은 확인 crop 에 함께 읽히기 쉬워 제 클릭을
    막는다(시연 Work Sheet / 메뉴 형제 계약).
    """
    others = ", ".join(f"'{key}'" for key in modes if key != mode)
    return (
        f"the row whose text is exactly '{mode}' in the dropdown list that is currently open "
        f"below or above the image mode combo box. The list also contains other rows ({others}); "
        f"'{mode}' is a different row from those and must not be confused with them. "
        f"Point at the center of the '{mode}' row."
    )


def list_box(point, width: int, height: int, *, left_ratio=LIST_LEFT_RATIO,
             right_ratio=LIST_RIGHT_RATIO, up_ratio=LIST_UP_RATIO,
             down_ratio=LIST_DOWN_RATIO) -> dict:
    """열린 목록을 찾을 영역(화살표 기준, 경계 clamp). 탐색을 이 안으로 묶는 것이 계약이다."""
    return {
        "left": max(0, int(point["x"] - width * left_ratio)),
        "top": max(0, int(point["y"] - height * up_ratio)),
        "right": min(width, int(point["x"] + width * right_ratio)),
        "bottom": min(height, int(point["y"] + height * down_ratio)),
    }


def change_image_mode(
    window,
    target_mode: str,
    *,
    capture_fn,
    locate_fn,
    read_fn,
    click_fn,
    sleep_fn,
    escape_fn=None,
    clock=time.monotonic,
    action_enabled: bool = True,
    settle_sec: float = SETTLE_SEC,
    verify_timeout_sec: float = VERIFY_TIMEOUT_SEC,
    verify_poll_sec: float = VERIFY_POLL_SEC,
) -> dict:
    """화살표 확인 -> 클릭 -> 항목 확인 -> 클릭 -> 값 재판독. 협력자는 전부 주입이다.

    협력자(다른 진입점과 같은 모양):
      capture_fn(window)            -> image
      locate_fn(image, target)      -> {"x","y"} | None (**이미지 픽셀 좌표**)
      read_fn(image, box, label)    -> list[str] (읽기 실패는 빈 목록)
      click_fn(window, image, point, key) -> None
      escape_fn()                   -> None (선택 못 하고 끝날 때 목록 닫기)

    확인되지 않으면 누르지 않고, 확인이 실패해도 다시 누르지 않는다(재클릭 금지 - 목록은
    토글이라 두 번째 클릭이 무엇을 할지 모른다).
    """
    from poc.workflow_3.vlm.label_verify import crop_box_around_point
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    out = {"target": target_mode, "current": None, "result": "", "point": None,
           "frame_wh": None, "value_tokens": [], "item_tokens": [],
           "first_seen_sec": None, "checks": 0}

    image = capture_fn(window)
    out["frame_wh"] = [image.width, image.height]
    arrow = locate_fn(image, TargetConfig(key="image_mode_arrow", description=ARROW_DESCRIPTION))
    if arrow is None:
        print("[WARNING] image mode 드롭다운 화살표를 찾지 못했습니다 - 클릭 안 함")
        out["result"] = RESULT_ARROW_NOT_LOCATED
        return out
    arrow = {"x": int(arrow["x"]), "y": int(arrow["y"])}
    out["point"] = dict(arrow)

    def _read_value(shot, label):
        box = crop_box_around_point(
            arrow, shot.width, shot.height,
            left_ratio=VALUE_LEFT_RATIO, right_ratio=VALUE_RIGHT_RATIO,
            half_height_ratio=VALUE_HALF_HEIGHT_RATIO,
        )
        return read_fn(shot, box, label)

    tokens = _read_value(image, "image_mode_value")
    out["value_tokens"] = list(tokens)
    current = read_mode(tokens)
    print(f"[INFO] 화살표 px={arrow} 왼쪽 읽힘={tokens[:8]!r} -> 현재 모드={current}")
    if current is None:
        # 아는 모드가 0개 또는 2개 이상 읽혔다 = 여기가 image mode 콤보라는 증거가 없다.
        print("[WARNING] 화살표 옆에서 image mode 를 확정하지 못했습니다 - 클릭 안 함")
        out["result"] = RESULT_MODE_UNREADABLE
        return out
    out["current"] = current
    if current == target_mode:
        print(f"[INFO] 이미 {target_mode} 입니다 - 드롭다운을 열지 않습니다")
        out["result"] = RESULT_ALREADY
        return out
    if not action_enabled:
        print(f"[INFO] 리허설(SAFE_MODE=1): {current} -> {target_mode} 로 바꿀 자리까지 확인했습니다")
        out["result"] = RESULT_REHEARSAL
        return out

    click_fn(window, image, arrow, "image_mode_arrow")
    sleep_fn(settle_sec)

    image = capture_fn(window)
    box = list_box(arrow, image.width, image.height)
    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    found = locate_fn(crop, TargetConfig(
        key="image_mode_item", description=item_description(target_mode),
        vertical_pad_min_px=ITEM_VERTICAL_PAD_MIN_PX,
    ))
    if found is None:
        print(f"[WARNING] 열린 목록에서 '{target_mode}' 항목을 찾지 못했습니다 box={box}")
        out["result"] = RESULT_ITEM_NOT_LOCATED
        _close_dropdown(escape_fn)
        return out
    item = {"x": int(found["x"]) + box["left"], "y": int(found["y"]) + box["top"]}

    item_tokens = read_fn(image, crop_box_around_point(
        item, image.width, image.height,
        left_ratio=ITEM_HALF_WIDTH_RATIO, right_ratio=ITEM_HALF_WIDTH_RATIO,
        half_height_ratio=ITEM_HALF_HEIGHT_RATIO,
    ), "image_mode_item")
    out["item_tokens"] = list(item_tokens)
    if read_mode(item_tokens) != target_mode:
        print(f"[WARNING] 항목 라벨 확인 실패 - 클릭 안 함: px={item} 읽힘={item_tokens[:8]!r} "
              f"기대={target_mode}")
        out["result"] = RESULT_ITEM_NOT_CONFIRMED
        _close_dropdown(escape_fn)
        return out

    print(f"[INFO] '{target_mode}' 항목 확인: px={item}")
    click_fn(window, image, item, "image_mode_item")

    def _reads_target():
        shot = capture_fn(window)
        return read_mode(_read_value(shot, "image_mode_verify")) == target_mode

    changed, first_seen, checks = poll_until(
        _reads_target, timeout_sec=verify_timeout_sec, interval_sec=verify_poll_sec,
        clock=clock, sleep=sleep_fn,
    )
    out["first_seen_sec"], out["checks"] = first_seen, checks
    out["result"] = RESULT_CHANGED if changed else RESULT_UNVERIFIED
    if not changed:
        # 다시 누르지 않는다 - 바뀌었는데 판독만 놓쳤을 수 있고, 목록이 닫힌 뒤의 재클릭은
        # 다시 목록을 여는 것이라 화면만 흐트러진다.
        print(f"[WARNING] 선택 뒤에도 콤보가 {target_mode} 로 읽히지 않습니다 - 다시 누르지 않습니다")
    return out


def _close_dropdown(escape_fn) -> None:
    """선택하지 못하고 끝날 때 열린 목록을 닫는다(엔지니어 화면에 남기지 않기)."""
    if escape_fn is None:
        print("[WARNING] 드롭다운이 열린 채 남았을 수 있습니다 - 화면에서 Esc 로 닫으세요")
        return
    try:
        escape_fn()
    except Exception as exc:
        print(f"[WARNING] Esc 전송 실패: {type(exc).__name__}: {exc}")


def _escape_fn(action_enabled: bool):
    """열린 목록을 닫는 Esc 전송기. 원격이 Esc 를 중계하는지는 오피스 미검증이다."""

    def _press():
        if not action_enabled:
            print("[INFO] Esc 전송 생략(SAFE_MODE)")
            return
        from pynput.keyboard import Key

        from poc.workflow_3.monitor.demonstration_rcs_control import _shared_keyboard

        board = _shared_keyboard()
        board.press(Key.esc)
        board.release(Key.esc)
        print("[INFO] Esc 전송(드롭다운 닫기) - 실제로 닫혔는지는 화면으로 확인")

    return _press


def main() -> int:
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.demonstration_rcs_control import (
        ALT_SETTLE_SEC,
        CLICK_HOLD_SEC,
        PRE_CLICK_SETTLE_SEC,
        _env_float,
        build_click_kit,
    )
    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
    from poc.workflow_3.util import make_timestamp_tag
    from poc.workflow_3.util.abort_switch import is_aborted, start_abort_hotkey
    from poc.workflow_3.util.event_dir import debug_root
    from poc.workflow_3.util.window_utils import print_elevation_status

    os.environ.setdefault("SAFE_MODE", "0")
    settings = load_workflow3_settings()
    eqp_id = os.environ.get("MANUAL_IMAGE_MODE_EQP_ID", "").strip() or EQP_ID
    target = (os.environ.get("MANUAL_IMAGE_MODE_TARGET", "").strip() or TARGET_MODE).upper()
    settle = _env_float("MANUAL_IMAGE_MODE_SETTLE_SEC", SETTLE_SEC)
    verify_timeout = _env_float("MANUAL_IMAGE_MODE_VERIFY_TIMEOUT_SEC", VERIFY_TIMEOUT_SEC)
    verify_poll = _env_float("MANUAL_IMAGE_MODE_VERIFY_POLL_SEC", VERIFY_POLL_SEC)

    if target not in IMAGE_MODES:
        print(f"[ERROR] 모르는 모드 '{target}'. 등록된 모드: {', '.join(IMAGE_MODES)}")
        return EXIT_PREFLIGHT_FAILED

    mode_text = "실클릭" if settings.action_enabled else "리허설(SAFE_MODE=1, 클릭 차단)"
    print(f"[INFO] image mode 변경: EQP_ID={eqp_id}, target={target}, {mode_text}")

    print_elevation_status()
    if not start_abort_hotkey(settings.abort_hotkey):
        print("[WARNING] 긴급 해제 단축키 미등록 - 중단하려면 터미널에서 프로세스를 종료하세요.")

    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] tool 창이 없습니다: EQP_ID={eqp_id}. 먼저 직접 접속하세요.")
        return EXIT_PREFLIGHT_FAILED
    print(f"[INFO] 열린 tool 창에 붙습니다: title={title!r}")
    if is_aborted():
        return EXIT_PREFLIGHT_FAILED

    from poc.workflow_3.vlm.label_verify import read_text_near_point, tokens_from_text

    debug_dir = debug_root() / "manual_image_mode" / make_timestamp_tag()
    # 가림 해제(Alt+click)는 쓰지 않는다 - 오피스에서 확인된 File Manager 에만 허용된 폴백이라
    # 새 요소로 일반화하지 않는다. reveal_* 비율은 kit 이 요구하는 값이라 중앙으로 둔다.
    kit = build_click_kit(
        settings,
        debug_dir=debug_dir,
        log_component="manual_image_mode",
        settle_sec=settle,
        pre_click_settle_sec=PRE_CLICK_SETTLE_SEC,
        click_hold_sec=CLICK_HOLD_SEC,
        alt_settle_sec=ALT_SETTLE_SEC,
        reveal_x_ratio=0.5,
        reveal_y_ratio=0.5,
    )

    def _read(image, box, label):
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(),
            artifact_label=label,
            log_name="manual_image_mode",
        )
        if not read.ok:
            print(f"[WARNING] OCR 실패({label}): {read.error}")
            return []
        return tokens_from_text(read.raw_text)

    out = change_image_mode(
        window, target,
        capture_fn=kit.capture, locate_fn=kit.locate, read_fn=_read, click_fn=kit.click,
        sleep_fn=time.sleep,
        escape_fn=_escape_fn(settings.action_enabled) if CLOSE_WITH_ESCAPE else None,
        action_enabled=settings.action_enabled,
        settle_sec=settle, verify_timeout_sec=verify_timeout, verify_poll_sec=verify_poll,
    )

    point, wh = out.get("point"), out.get("frame_wh")
    where = ""
    if point is not None and wh:
        # px 와 **창 비율**을 함께 남긴다 - 비율이라야 다음 실행/다른 장비에서 그대로 쓸 수 있다
        # (button_registry 의 center 와 같은 단위).
        where = (f" arrow_px=({point['x']}, {point['y']})"
                 f" arrow_center=({point['x'] / wh[0]:.3f}, {point['y'] / wh[1]:.3f})")
    print(f"[DIGEST] image_mode target={target} current={out['current']} "
          f"result={out['result']} checks={out['checks']} "
          f"first_seen_sec={out['first_seen_sec']}{where}")
    return EXIT_BY_RESULT.get(out["result"], EXIT_NOT_CONFIRMED)


if __name__ == "__main__":
    raise SystemExit(main())
