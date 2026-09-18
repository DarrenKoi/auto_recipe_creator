"""열린 tool 창에서 버튼 하나를 확인 후 누른다. 있어야 할 버튼이 안 보이면 폴백으로
가린 창을 Alt+click 해 뒤로 밀어내고 다시 찾는다.

기본 대상은 **File Manager** 버튼(화면 아래쪽 버튼 그룹의 오른쪽). 이 버튼은 'SECS Terminal',
'Terminal Service' 같은 창에 자주 가려지는데, 엔지니어는 그 자리를 Alt+click 해서 덮은
창을 뒤로 보내고 버튼을 되살린다. 이 진입점이 같은 일을 한다:

  1. 제목에 EQP_ID 가 든 Remote Monitoring 창에 붙는다(접속은 하지 않는다 - 엔지니어가
     먼저 연다, `manual_align_correction.py` 와 같은 규약).
  2. VLM 이 버튼 좌표를 찍고 PaddleOCR 이 그 자리 라벨을 확인한다.
  3. **폴백 - 버튼이 안 보일 때만** VLM 으로 덮은 창의 제목줄을 찾아 Alt+click 해 뒤로
     보내고 다시 찾는다(최대 REVEAL_ATTEMPTS 번 - 두 장이면 라운드마다 앞 창). '안 보임' 은 VLM 미검출,
     또는 라벨 불일치이면서 그 예상 영역을 확대 OCR 해도 라벨이 없을 때다. 예상 영역에
     라벨이 읽히면 VLM 이 잘못 짚었을 뿐이라 Alt+click 하지 않고 멈춘다(exit 4).
  4. 확인되면 클릭한다. 확인이 안 되면 누르지 않는다.
  5. 열린 창의 제목줄('File Manager( Class, IDW, IDP, Recipe )')을 VLM+OCR 로 확인한다.

다른 가려진 버튼에 쓰려면 TARGET_* 상수만 바꾼다. 클릭/가림 해제 배선은
`demonstration_rcs_control.build_click_kit` 를 그대로 쓴다(원격 클릭 성사 조건이 오피스
실측값이라 포크하지 않는다).

실행: uv run python poc/workflow_3/monitor/manual_click_button.py
리허설(클릭/Alt 차단): SAFE_MODE=1 uv run python ...
종료 코드: 0=클릭+창 확인, 2=사전조건 실패, 3=가림 해제 후에도 못 찾음, 4=라벨 불일치/미검출,
          5=클릭했지만 'File Manager( Class, IDW, IDP, Recipe )' 창 미확인
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.config import load_workflow3_settings  # noqa: E402
from poc.workflow_3.monitor.demonstration_rcs_control import (  # noqa: E402
    ALT_SETTLE_SEC,
    CLICK_HOLD_SEC,
    CONFIRM_LABEL_REJECTED,
    CONFIRM_NOT_LOCATED,
    CONFIRM_NOT_VISIBLE,
    covering_window_point,
    FlowStep,
    PRE_CLICK_SETTLE_SEC,
    _env_float,
    _env_int,
    build_click_kit,
    locate_with_reveal,
)
from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window  # noqa: E402
from poc.workflow_3.util import make_timestamp_tag  # noqa: E402
from poc.workflow_3.util.abort_switch import is_aborted, start_abort_hotkey  # noqa: E402
from poc.workflow_3.util.event_dir import debug_root  # noqa: E402
from poc.workflow_3.util.window_utils import print_elevation_status  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 셸 env(괄호 안 이름)가 있으면 env 가 이긴다.
# ===========================================================================

EQP_ID = "MCD513"                  # (MANUAL_CLICK_EQP_ID) 제목에 이 ID 가 든 tool 창

TARGET_KEY = "file_manager_button"
# 첫 글자 anchor('F')를 쓰지 않는다 - 이 창에는 F 로 시작하는 버튼/텍스트가 많아 엉뚱한
# 것을 짚었다(2026-09-18 오피스). 전체 문구 두 단어 + "아래쪽 버튼 그룹" 위치로 고정한다
# (라이브 SEM box 기준 설명은 VLM 이 그 box 를 모를 수 있어 뺐다).
TARGET_DESCRIPTION = (
    "the button labeled with the two words 'File Manager' in the Remote Monitoring "
    "window. It is in the group of buttons along the BOTTOM of the screen, on the "
    "RIGHT side of that group. Look only in that bottom button group and find the "
    "button whose label reads exactly "
    "'File Manager' - ignore any other button or text that merely starts with 'F'. "
    "Click the center of that button."
)
# OCR 확인: 묶음 하나를 통째로 만족해야 한다. 'FileManager' 로 붙여 읽혀도 통과한다.
# needle 은 'manag' - 좁은 crop 에서 끝 글자가 잘려 'File', 'Manage' 로 읽혔다(2026-09-18
# 오피스). 부분 일치라 Manager/Manage/FileManager 모두 통과한다.
TARGET_REQUIRED = (("file", "manag"),)
# 라벨 확인 OCR crop: 클릭 지점 기준 창 폭의 좌우/창 높이의 위아래 비율. **버튼 한 개
# 크기**로 좁힌다. 기본(0.30/0.05)은 아래쪽 버튼 그룹에서 위/아래 줄까지 담아 OCR 이
# 위 줄 버튼만 읽고 'File Manager' 를 놓쳤다(2026-09-18 오피스, 가림 해제 뒤 unreadable).
# 좁은 crop 은 확대되어 글자도 또렷해진다. 점이 한 줄 위를 짚었다면 그 버튼 이름만
# 읽혀 strict 가 여전히 거부한다.
CONFIRM_HALF_WIDTH_RATIO = 0.10
CONFIRM_HALF_HEIGHT_RATIO = 0.015
# forbidden 은 비운다. 확인 OCR crop 이 클릭 지점 좌우 30% 를 담아 **아래쪽 버튼 그룹의
# 이웃 버튼(Exit/Close 등)이 반드시 함께 읽히고**, classify_label 은 forbidden 을 required
# 보다 먼저 봐서 맞는 점을 거부했다(2026-09-18 오피스: 점은 맞는데 클릭 안 됨). 데모의
# Work Sheet/File 이 같은 이유로 비웠다. 확인은 strict + required 두 단어가 맡는다.
TARGET_FORBIDDEN = ()
# strict: 'File Manager' 가 읽혀야만 누른다. lenient 는 다른 라벨(예: 'SECS Terminal')도
# '못 읽음' 으로 통과시키는데, 버튼이 가려졌을 때 VLM 이 찍는 곳이 바로 덮은 창이다.
CONFIRM_POLICY = "strict"

# 버튼이 있어야 할 자리(창 크기 대비 비율). '가려졌나' 를 보는 예상 영역 OCR 의 중심이다.
# Alt+click 지점은 이것이 아니라 VLM 이 찾은 **덮은 창의 제목줄**이다 - 창은 버튼 그룹
# 주변 어디에나 놓일 수 있어 고정 지점은 창을 빗나갔다(2026-09-18 오피스: 창보다 약간
# 위를 눌러 Alt+click 효과 없음).
REVEAL_X_RATIO = 0.80              # (MANUAL_CLICK_REVEAL_X_RATIO)
REVEAL_Y_RATIO = 0.90              # (MANUAL_CLICK_REVEAL_Y_RATIO)
COVER_KEY = "covering_window_title"
COVER_DESCRIPTION = (
    "a separate small window or dialog (for example 'SECS Terminal' or 'Terminal "
    "Service') that is floating ON TOP of the group of buttons along the bottom of "
    "the Remote Monitoring screen and hides some of those buttons. If several such "
    "windows overlap, choose the one in front. Point at the middle of that window's "
    "TITLE BAR (the bar at its top showing the window name)."
)
REVEAL_ATTEMPTS = 3                # (MANUAL_CLICK_REVEAL_ATTEMPTS) 창이 여러 장 겹칠 수 있다
SETTLE_SEC = 1.0                   # Alt+click 뒤 창이 다시 그려질 대기
# 라벨 불일치 때 '가려졌나 / 잘못 짚었나' 를 가르는 OCR 영역: 가림해제 지점을 중심으로
# 창 폭의 좌우 이 비율, 창 높이의 위아래 이 비율. 버튼 행 전체가 들어올 만큼 넓게.
EXPECTED_AREA_HALF_WIDTH_RATIO = 0.25
EXPECTED_AREA_HALF_HEIGHT_RATIO = 0.08

# 클릭 후 열린 창 확인. File Manager 창은 원격 뷰 안에 그려지므로(로컬 top-level 창이
# 아니다) 창 제목 조회로는 못 찾는다 - VLM 이 제목줄 좌표, OCR 이 제목 확인.
# 제목: "File Manager( Class, IDW, IDP, Recipe )". 'File Manager' 만으로는 방금 누른
# **버튼**도 통과하므로 제목에만 있는 IDW/Recipe 를 함께 요구한다(묶음 중 하나).
OPENED_KEY = "file_manager_window_title"
OPENED_DESCRIPTION = (
    "the title bar text of the large 'File Manager' window that just opened, which "
    "reads 'File Manager( Class, IDW, IDP, Recipe )'. Point at the middle of that "
    "title text, not at the 'File Manager' button in the bottom button group."
)
OPENED_REQUIRED = (("manag", "idw"), ("manag", "recipe"))
OPEN_WAIT_SEC = 2.0                # 클릭 -> 창이 원격 뷰에 그려질 대기

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2
EXIT_NOT_VISIBLE = 3
EXIT_NOT_CONFIRMED = 4
EXIT_NOT_OPENED = 5                # 클릭은 했는데 File Manager 창을 확인 못 함


def label_in_tokens(tokens, required) -> bool:
    """required 묶음 하나의 needle 이 전부 읽혔는가('FileManager' 로 붙어도 통과)."""
    text = " ".join(tokens).lower()
    return any(all(needle in text for needle in group) for group in required)


def main() -> int:
    os.environ.setdefault("SAFE_MODE", "0")
    settings = load_workflow3_settings()
    eqp_id = os.environ.get("MANUAL_CLICK_EQP_ID", "").strip() or EQP_ID
    reveal_attempts = max(0, _env_int("MANUAL_CLICK_REVEAL_ATTEMPTS", REVEAL_ATTEMPTS))
    x_ratio = _env_float("MANUAL_CLICK_REVEAL_X_RATIO", REVEAL_X_RATIO)
    y_ratio = _env_float("MANUAL_CLICK_REVEAL_Y_RATIO", REVEAL_Y_RATIO)

    mode = "실클릭" if settings.action_enabled else "리허설(SAFE_MODE=1, 클릭/Alt 차단)"
    print(f"[INFO] 버튼 클릭: EQP_ID={eqp_id}, target={TARGET_KEY}, {mode}, "
          f"가림해제 Alt+click 최대 {reveal_attempts}회 @ x={x_ratio:.2f}/y={y_ratio:.2f}")

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

    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    debug_dir = debug_root() / "manual_click_button" / make_timestamp_tag()
    kit = build_click_kit(
        settings,
        debug_dir=debug_dir,
        log_component="manual_click_button",
        settle_sec=SETTLE_SEC,
        pre_click_settle_sec=PRE_CLICK_SETTLE_SEC,
        click_hold_sec=CLICK_HOLD_SEC,
        alt_settle_sec=ALT_SETTLE_SEC,
        reveal_x_ratio=x_ratio,
        reveal_y_ratio=y_ratio,
        confirm_half_width_ratio=CONFIRM_HALF_WIDTH_RATIO,
        confirm_half_height_ratio=CONFIRM_HALF_HEIGHT_RATIO,
    )
    step = FlowStep(
        TargetConfig(key=TARGET_KEY, description=TARGET_DESCRIPTION),
        required=TARGET_REQUIRED,
        forbidden=TARGET_FORBIDDEN,
    )
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )

    label_seen = {}  # 예상 영역에서 라벨이 읽혔으면 그 box - 잘못 짚음 재시도에 쓴다

    def _should_reveal(image, reason):
        """가려졌을 때만 Alt+click 한다. VLM 이 잘못 짚었을 뿐이면 밀어내지 않는다.

        미검출이든 라벨 불일치든, 버튼이 **있어야 할 자리**를 넓게 잘라 확대 OCR 로 다시
        읽는다. 거기서 라벨이 읽히면 '보이는데 VLM 이 잘못 짚음/놓침' 이라 멈춘다 - 그때
        Alt+click 하면 멀쩡한 창만 뒤로 간다. OCR 은 여기서도 판독만 하고 좌표는 만들지
        않는다(클릭 좌표는 VLM 몫).
        """
        if reason not in (CONFIRM_NOT_LOCATED, CONFIRM_LABEL_REJECTED):
            return False
        center = covering_window_point(
            image.width, image.height, x_ratio=x_ratio, y_ratio=y_ratio,
        )
        box = crop_box_around_point(
            center, image.width, image.height,
            left_ratio=EXPECTED_AREA_HALF_WIDTH_RATIO,
            right_ratio=EXPECTED_AREA_HALF_WIDTH_RATIO,
            half_height_ratio=EXPECTED_AREA_HALF_HEIGHT_RATIO,
        )
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(),
            artifact_label=f"{TARGET_KEY}_expected_area",
            log_name="manual_click_button",
        )
        tokens = tokens_from_text(read.raw_text) if read.ok else []
        if not read.ok:
            print(f"[WARNING] 예상 영역 OCR 실패 - 가려졌는지 몰라 Alt+click 안 함: {read.error}")
            return False
        if label_in_tokens(tokens, TARGET_REQUIRED):
            print(f"[INFO] 예상 영역에 {TARGET_KEY} 라벨이 보입니다 - 가려진 게 아니라 "
                  f"VLM 이 잘못 짚었습니다. Alt+click 안 함, 예상 영역 crop 으로 다시 찾습니다. "
                  f"box={box}")
            label_seen["box"] = box
            return False
        print(f"[INFO] 예상 영역에 라벨 없음(읽힘={tokens[:12]!r}) - 가려진 것으로 봅니다")
        return True

    cover_target = TargetConfig(key=COVER_KEY, description=COVER_DESCRIPTION)

    def _reveal(window, image, round_index):
        """덮은 창을 VLM 으로 찾아 그 제목줄을 Alt+click 한다. 두 장이면 라운드마다 앞 창.

        짚은 곳에서 대상 라벨이 읽히면 누르지 않는다 - 그건 덮은 창이 아니라 tool 화면
        자체이고, 거기를 Alt+click 하면 tool 창이 뒤로 간다. 못 찾으면 고정 지점으로
        대신 누르지 않는다(그 지점이 빗나간 것이 이 경로를 만든 이유다).
        """
        point = kit.locate(image, cover_target)
        if point is None:
            print("[WARNING] 덮은 창을 찾지 못함 - Alt+click 안 함")
            return False
        tokens = kit.read_tokens(image, point, COVER_KEY)
        print(f"[INFO] 덮은 창 제목줄 후보: px={point} 읽힘={tokens[:12]!r}")
        if label_in_tokens(tokens, TARGET_REQUIRED):
            print(f"[WARNING] 짚은 곳에 {TARGET_KEY} 라벨이 있습니다 - 덮은 창이 아니라 "
                  "tool 화면이라 Alt+click 안 함")
            return False
        return kit.alt_click_at(window, image, point, round_index, note="덮은 창 제목줄")

    image, point, reason, reveals = locate_with_reveal(
        window, step,
        capture_fn=kit.capture, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
        policy=CONFIRM_POLICY, reveal_fn=_reveal, max_reveals=reveal_attempts,
        label=TARGET_KEY,
        should_reveal_fn=_should_reveal,
    )
    if point is None and label_seen:
        # 전체 화면에서는 비슷한 버튼이 많아 VLM 이 이웃을 짚었다(2026-09-18 오피스). 라벨이
        # 읽힌 예상 영역만 잘라 다시 찾게 하면 후보가 줄고 확대된다. 좌표는 여전히 VLM 이
        # 정하고(crop 좌표 + box 원점 = 전체 좌표), OCR 이 전체 이미지에서 다시 확인한다.
        box = label_seen["box"]
        crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))

        def _locate_in_crop(_full_image, target):
            found = kit.locate(crop, target)
            if found is None:
                return None
            return {"x": int(found["x"]) + box["left"], "y": int(found["y"]) + box["top"]}

        full_image = image
        _, point, reason, _ = locate_with_reveal(
            window, step,
            capture_fn=lambda _w: full_image, locate_fn=_locate_in_crop,
            read_tokens_fn=kit.read_tokens, policy=CONFIRM_POLICY,
            label=f"{TARGET_KEY}_crop",
        )
        print(f"[INFO] 예상 영역 crop 재탐색: {'확인됨' if point else '실패'} "
              f"point={point} reason={reason}")

    if point is None:
        print(f"[DIGEST] manual_click target={TARGET_KEY} result={reason} reveals={reveals}")
        return EXIT_NOT_VISIBLE if reason == CONFIRM_NOT_VISIBLE else EXIT_NOT_CONFIRMED

    kit.click(window, image, point, TARGET_KEY)
    time.sleep(OPEN_WAIT_SEC)

    # 다시 누르지 않는다 - 창이 떴는데 확인만 실패한 경우 두 번째 클릭이 무엇을 할지
    # 모른다. 확인 실패는 exit 5 로 남기고 debug crop 으로 대조한다.
    opened_step = FlowStep(
        TargetConfig(key=OPENED_KEY, description=OPENED_DESCRIPTION),
        required=OPENED_REQUIRED,
    )
    _, title_point, _, _ = locate_with_reveal(
        window, opened_step,
        capture_fn=kit.capture, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
        policy="strict", label=OPENED_KEY,
    )
    opened = title_point is not None
    print(f"[DIGEST] manual_click target={TARGET_KEY} result=clicked reveals={reveals} "
          f"opened={'yes' if opened else 'unconfirmed'}")
    return EXIT_OK if opened else EXIT_NOT_OPENED


if __name__ == "__main__":
    raise SystemExit(main())
