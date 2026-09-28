"""열린 tool 창에서 File Manager 를 열고 Class -> Recipe 를 골라 recipe 하나를 연다.

  1. `manual_click_button.main()` 으로 File Manager 를 연다(이미 열려 있으면 그대로 쓴다).
     가림 해제/열림 확인까지 그 진입점이 오피스 검증된 그대로 한다 - 포크하지 않는다.
  2. RECIPE_ID(`<class>/<recipe>`)의 class 를 Class 목록(1번째 열)에서 찾아 클릭한다.
  3. recipe 를 Recipe 목록(4번째 열)에서 찾아 클릭한다.

행 찾기(`find_row_in_column`): 열 머리글을 VLM 으로 찍고 OCR 로 확인 -> 그 아래 목록을 휠로
맨 위까지 올린 뒤 -> 한 화면씩 VLM 이 행을 찍고 **한 줄 strip** OCR 이 이름을 **전체 일치**로
확인할 때까지 휠을 내린다. 휠 뒤 목록 영역이 더 안 바뀌면 목록 끝이다.

타이핑 검색은 쓰지 않는다: 'RJ1BXXX_CG6300' 의 '_' 는 Shift 기호인데 이 원격은 쥐는 수정자를
넘기지 않는다(`type_multiline_text` 오피스 실측) - '_' 가 '-' 로 들어가 엉뚱한 행으로 뛴다.
전체 일치인 이유: 부분 일치면 'RJ1BXXX_CG6300A' 같은 이웃 이름도 통과한다
(`manual_image_mode_change` 의 OM/OM-D 와 같은 이유).

**오피스 미검증** - 열 머리글 아래 휠 위치, 행 높이, 휠 한 칸의 줄 수는 추정치다. 첫 실행은
`SAFE_MODE=1` 로 머리글/첫 화면 판독까지 보고, 콘솔의 `읽힘=` 토큰으로 strip 크기를 맞춘다.

실행: uv run python poc/workflow_3/monitor/manual_open_recipe.py
리허설(클릭/휠 차단): SAFE_MODE=1 uv run python ...
종료 코드: 0=recipe 클릭, 2=사전조건/File Manager 실패, 3=머리글/행 못 찾음
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 셸 env(괄호 안 이름)가 있으면 env 가 이긴다.
# ===========================================================================

EQP_ID = "MCD513"                  # (MANUAL_OPEN_RECIPE_EQP_ID) 제목에 이 ID 가 든 tool 창
# MES 알람의 RECIPE_ID 형태 그대로 '<class>/<recipe>' - 앞이 Class 열, 뒤가 Recipe 열.
RECIPE_ID = "RJ1BXXX_CG6300/RJ1B_ISOCO_EUVSPT"  # (MANUAL_OPEN_RECIPE_ID)

SETTLE_SEC = 1.0                   # 행 클릭 -> 다음 목록이 채워질 대기
SCROLL_SETTLE_SEC = 0.8            # 휠 -> 원격 화면이 다시 그려질 대기. 짧으면 '안 바뀜' 오판 = 목록 끝 오판
PAGE_NOTCHES = 3                   # 한 쪽 내릴 때 휠 칸 수 - 한 화면 행 수보다 적게(행을 건너뛰지 않게)
TOP_NOTCHES = 10                   # 맨 위로 올릴 때 휠 칸 수
MAX_PAGES = 60                     # 한 목록에서 볼 최대 쪽 수(쪽마다 VLM+OCR 한 쌍)
MAX_TOP_SCROLLS = 30

# 좌표는 전부 창 비율. 휠/변화 감지 지점 = 머리글에서 이만큼 아래(목록 안쪽이어야 한다).
SCROLL_OFFSET_RATIO = 0.15
# 목록 변화 감지 영역(휠 지점 기준 반폭, 위/아래).
CHANGE_HALF_WIDTH_RATIO = 0.06
CHANGE_UP_RATIO = 0.10
CHANGE_DOWN_RATIO = 0.15
CHANGE_TOL = 1.0                   # 회색조 평균 절대차 - 이보다 작으면 안 바뀐 것
# 행이 그 열에 속한다고 볼 머리글과의 가로 거리 상한. 열 너비 절반쯤.
COLUMN_HALF_WIDTH_RATIO = 0.12
# 행 확인 strip - **한 줄**만 담는다. 위아래 행이 섞이면 이웃 행의 같은 이름이 확인을 통과한다.
ROW_HALF_WIDTH_RATIO = 0.07
ROW_HALF_HEIGHT_RATIO = 0.011
ROW_VERTICAL_PAD_MIN_PX = 10       # 2단 로케이터 fine crop 세로 하한(촘촘한 목록)

RESULT_FOUND = "found"
RESULT_HEADER_NOT_FOUND = "header_not_found"
RESULT_HEADER_NOT_CONFIRMED = "header_not_confirmed"
RESULT_ROW_NOT_FOUND = "row_not_found"
RESULT_ABORTED = "aborted"

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2
EXIT_NOT_FOUND = 3


def _word(token) -> str:
    """비교용 정규화(영숫자만, 소문자). 'RJ1BXXX_CG6300' -> 'rj1bxxxcg6300'."""
    return "".join(ch for ch in str(token).lower() if ch.isalnum())


def name_matches(tokens, name: str) -> bool:
    """이름 **전체 일치**. OCR 이 '_' 에서 쪼갠 경우는 토큰을 이어 붙인 것도 본다.

    이어 붙인 쪽은 strip 에 다른 글자가 섞이면 실패한다(fail-closed) - 그때는 콘솔의
    `읽힘=` 을 보고 ROW_HALF_WIDTH_RATIO 를 줄인다.
    """
    want = _word(name)
    words = [w for w in (_word(t) for t in tokens) if w]
    return bool(want) and (want in words or "".join(words) == want)


def header_description(column: str) -> str:
    return (
        f"the column header labeled exactly '{column}' of the File Manager window. That "
        "window shows four side-by-side lists whose headers are 'Class', 'IDW', 'IDP' and "
        f"'Recipe'. Point at the '{column}' header text above its list, NOT at the window "
        "title bar 'File Manager( Class, IDW, IDP, Recipe )'."
    )


def row_description(column: str, name: str) -> str:
    return (
        f"the row whose text is exactly '{name}' in the '{column}' list of the File Manager "
        "window (four side-by-side lists: Class, IDW, IDP, Recipe). Look only inside the "
        f"'{column}' list. Other rows may share the beginning of the text but differ at the "
        "end; choose only the exact match. Point at the center of that row's text."
    )


def _strip(point, image) -> dict:
    from poc.workflow_3.vlm.label_verify import crop_box_around_point

    return crop_box_around_point(
        point, image.width, image.height,
        left_ratio=ROW_HALF_WIDTH_RATIO, right_ratio=ROW_HALF_WIDTH_RATIO,
        half_height_ratio=ROW_HALF_HEIGHT_RATIO,
    )


def change_box(anchor, width: int, height: int) -> dict:
    return {
        "left": max(0, int(anchor["x"] - width * CHANGE_HALF_WIDTH_RATIO)),
        "top": max(0, int(anchor["y"] - height * CHANGE_UP_RATIO)),
        "right": min(width, int(anchor["x"] + width * CHANGE_HALF_WIDTH_RATIO)),
        "bottom": min(height, int(anchor["y"] + height * CHANGE_DOWN_RATIO)),
    }


def view_unchanged(before, after, box, tol: float = CHANGE_TOL) -> bool:
    """휠 전후 목록 영역이 같은가 = 더 굴러가지 않았다(목록 끝)."""
    import numpy as np

    crop = (box["left"], box["top"], box["right"], box["bottom"])
    a = np.asarray(before.crop(crop).convert("L"), dtype=np.int16)
    b = np.asarray(after.crop(crop).convert("L"), dtype=np.int16)
    return float(np.abs(a - b).mean()) < tol


def find_row_in_column(
    window, column: str, name: str,
    *, capture_fn, locate_fn, read_fn, scroll_fn, sleep_fn,
    settle_sec: float = SCROLL_SETTLE_SEC, max_pages: int = MAX_PAGES,
    max_top_scrolls: int = MAX_TOP_SCROLLS,
):
    """열 `column` 에서 `name` 행을 찾아 `(image, point, result)` 를 돌려준다. 협력자는 주입이다.

      capture_fn(window) -> image
      locate_fn(image, target) -> {"x","y"} | None (이미지 픽셀)
      read_fn(image, box, label) -> list[str]
      scroll_fn(window, image, point, dy) -> bool (dy>0 = 위로; False = 긴급 해제 등으로 못 굴림)

    VLM 은 화면에 없는 이름에도 아무 행이나 찍을 수 있다 - 그래서 확인 실패는 '다음 쪽' 이지
    멈춤이 아니다. 누르는 것은 호출부다(확인된 점만 받는다).
    """
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    key = column.lower()
    image = capture_fn(window)
    header = locate_fn(image, TargetConfig(key=f"{key}_header",
                                           description=header_description(column)))
    if header is None:
        print(f"[WARNING] '{column}' 머리글을 찾지 못했습니다")
        return image, None, RESULT_HEADER_NOT_FOUND
    header = {"x": int(header["x"]), "y": int(header["y"])}
    words = {_word(t) for t in read_fn(image, _strip(header, image), f"{key}_header")}
    # 창 제목 'File Manager( Class, ... )' 에도 열 이름이 있다 - 'manager' 가 읽히면 제목이다.
    if _word(column) not in words or any("manager" in w for w in words):
        print(f"[WARNING] '{column}' 머리글 확인 실패 px={header} 읽힘={sorted(words)[:8]!r}")
        return image, None, RESULT_HEADER_NOT_CONFIRMED

    anchor = {"x": header["x"], "y": header["y"] + int(image.height * SCROLL_OFFSET_RATIO)}
    box = change_box(anchor, image.width, image.height)

    def _roll(image_, dy):
        """휠 한 번 -> `(새 화면, 굴러갔나)`. 못 굴렸으면 (image_, None)."""
        if not scroll_fn(window, image_, anchor, dy):
            return image_, None
        sleep_fn(settle_sec)
        after = capture_fn(window)
        return after, not view_unchanged(image_, after, box)

    for _ in range(max_top_scrolls):
        image, moved = _roll(image, TOP_NOTCHES)
        if moved is None:
            return image, None, RESULT_ABORTED
        if not moved:
            break

    row_target = TargetConfig(key=f"{key}_row", description=row_description(column, name),
                              vertical_pad_min_px=ROW_VERTICAL_PAD_MIN_PX)
    for page in range(1, max_pages + 1):
        point = locate_fn(image, row_target)
        if point is None:
            print(f"[INFO] [{column}] {page}쪽: 행 미검출 - 다음 쪽")
        else:
            point = {"x": int(point["x"]), "y": int(point["y"])}
            dx = abs(point["x"] - header["x"]) / image.width
            if point["y"] <= header["y"] or dx > COLUMN_HALF_WIDTH_RATIO:
                print(f"[INFO] [{column}] {page}쪽: 짚은 점이 목록 밖 px={point} dx={dx:.3f} - 다음 쪽")
            else:
                tokens = read_fn(image, _strip(point, image), f"{key}_row")
                if name_matches(tokens, name):
                    print(f"[INFO] [{column}] {page}쪽에서 '{name}' 확인: px={point}")
                    return image, point, RESULT_FOUND
                print(f"[INFO] [{column}] {page}쪽: 읽힘={tokens[:8]!r} != '{name}' - 다음 쪽")
        image, moved = _roll(image, -PAGE_NOTCHES)
        if moved is None:
            return image, None, RESULT_ABORTED
        if not moved:
            print(f"[WARNING] [{column}] 목록 끝까지 '{name}' 가 없습니다({page}쪽)")
            return image, None, RESULT_ROW_NOT_FOUND
    print(f"[WARNING] [{column}] {max_pages}쪽 안에 '{name}' 가 없습니다")
    return image, None, RESULT_ROW_NOT_FOUND


def main() -> int:
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import manual_click_button
    from poc.workflow_3.monitor.demonstration_rcs_control import (
        ALT_SETTLE_SEC,
        CLICK_HOLD_SEC,
        PRE_CLICK_SETTLE_SEC,
        build_click_kit,
    )
    from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
    from poc.workflow_3.util import make_timestamp_tag
    from poc.workflow_3.util.event_dir import debug_root
    from poc.workflow_3.util.mouse_utils import scroll_at_screen
    from poc.workflow_3.util.window_utils import image_point_to_screen
    from poc.workflow_3.vlm.label_verify import read_text_near_point, tokens_from_text

    os.environ.setdefault("SAFE_MODE", "0")
    eqp_id = os.environ.get("MANUAL_OPEN_RECIPE_EQP_ID", "").strip() or EQP_ID
    recipe_id = os.environ.get("MANUAL_OPEN_RECIPE_ID", "").strip() or RECIPE_ID
    class_name, _, recipe_name = (part.strip() for part in recipe_id.partition("/"))
    if not class_name or not recipe_name or "/" in recipe_name:
        print(f"[ERROR] RECIPE_ID 는 '<class>/<recipe>' 형태여야 합니다: {recipe_id!r}")
        return EXIT_PREFLIGHT_FAILED
    print(f"[INFO] recipe 열기: EQP_ID={eqp_id}, class={class_name}, recipe={recipe_name}")

    # 1. File Manager - 이 진입점 하나로 여는 것이 계약이라 대상은 덮어쓴다(셸 env 가 달라도).
    os.environ["MANUAL_CLICK_EQP_ID"] = eqp_id
    os.environ["MANUAL_CLICK_TARGET"] = "file_manager"
    code = manual_click_button.main()
    if code not in (manual_click_button.EXIT_OK, manual_click_button.EXIT_ALREADY_OPEN):
        print(f"[ERROR] File Manager 를 열지 못했습니다(exit={code}) - 멈춥니다")
        return EXIT_PREFLIGHT_FAILED

    settings = load_workflow3_settings()
    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] tool 창이 없습니다: EQP_ID={eqp_id}")
        return EXIT_PREFLIGHT_FAILED

    debug_dir = debug_root() / "manual_open_recipe" / make_timestamp_tag()
    kit = build_click_kit(
        settings,
        debug_dir=debug_dir,
        log_component="manual_open_recipe",
        settle_sec=SETTLE_SEC,
        pre_click_settle_sec=PRE_CLICK_SETTLE_SEC,
        click_hold_sec=CLICK_HOLD_SEC,
        alt_settle_sec=ALT_SETTLE_SEC,
        reveal_x_ratio=0.5,
        reveal_y_ratio=0.5,
    )

    def _read(image, box, label):
        read = read_text_near_point(
            image, box, debug_image_dir=debug_dir, timestamp_tag=make_timestamp_tag(),
            artifact_label=label, log_name="manual_open_recipe",
        )
        if not read.ok:
            print(f"[WARNING] OCR 실패({label}): {read.error}")
            return []
        return tokens_from_text(read.raw_text)

    def _scroll(window_, image, point, dy):
        screen = image_point_to_screen(window_, point, image_size=image.size)
        if screen is None:
            print(f"[WARNING] 휠 좌표 변환 실패 px={point}")
            return False
        return scroll_at_screen(screen, dy, "open_recipe", 0,
                                action_enabled=settings.action_enabled)

    for column, name in (("Class", class_name), ("Recipe", recipe_name)):
        image, point, result = find_row_in_column(
            window, column, name,
            capture_fn=kit.capture, locate_fn=kit.locate, read_fn=_read,
            scroll_fn=_scroll, sleep_fn=time.sleep,
        )
        if point is None:
            print(f"[DIGEST] open_recipe column={column} name={name} result={result}")
            return EXIT_NOT_FOUND
        kit.click(window, image, point, f"{column.lower()}_row")
        time.sleep(SETTLE_SEC)

    print(f"[DIGEST] open_recipe class={class_name} recipe={recipe_name} result=clicked "
          f"{'' if settings.action_enabled else '[dry-run]'}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
