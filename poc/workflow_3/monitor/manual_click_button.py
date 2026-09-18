"""열린 tool 창에서 **레지스트리에 등록된 버튼 하나**를 확인 후 누른다.

누를 버튼은 `TARGET` 이름 하나로 고른다(`button_registry.BUTTONS`). 버튼별 차이(라벨,
위치 서술, OCR needle, 확인 crop, 클릭 뒤 열릴 창)는 레지스트리에 살고, VLM 프롬프트는
`button_registry.describe` 공통 템플릿 하나다 - 버튼마다 프롬프트를 고치지 않는다.

  1. 제목에 EQP_ID 가 든 Remote Monitoring 창에 붙는다(접속은 하지 않는다 - 엔지니어가
     먼저 연다, `manual_align_correction.py` 와 같은 규약).
  2. 클릭 뒤 열릴 창이 **이미** 보이면 누르지 않는다(already_open - 그 상태에서 누른 결과는
     효과 증거가 아니다).
  3. `find_button`: 등록 위치가 있으면 그 영역 crop 에서, 실패하면 전체 화면에서 VLM 이
     좌표를 찍고 PaddleOCR 이 라벨을 strict 확인한다. 확인이 안 되면 누르지 않는다.
  4. **폴백 - spec.reveal 인 버튼만**: 예상 영역에 라벨이 안 읽히면 덮은 창의 제목줄을
     VLM 으로 찾아 Alt+click 해 뒤로 보내고 다시 찾는다(최대 REVEAL_ATTEMPTS 번).
  5. 클릭 후 열릴 창 제목을 반복 확인해 '처음 확인된 시각' 을 [DIGEST] 에 남긴다(원격
     지연 관찰용). 확인이 안 돼도 다시 누르지 않는다.

성공하면 클릭 점을 창 비율 `center=(x, y)` 로 출력한다 - 위치 미등록 버튼은 그 값을
레지스트리에 옮기면 다음부터 등록 영역 crop 에서 먼저 찾는다.

실행: uv run python poc/workflow_3/monitor/manual_click_button.py
리허설(클릭/Alt 차단): SAFE_MODE=1 uv run python ...
종료 코드: 0=클릭(+창 확인 또는 확인 대상 없음), 2=사전조건 실패, 3=가림 해제 후에도 못 찾음,
          4=라벨 불일치/미검출, 5=클릭했지만 열릴 창 미확인, 6=이미 열려 있어 누르지 않음
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.config import load_workflow3_settings  # noqa: E402
from poc.workflow_3.monitor.button_registry import (  # noqa: E402
    BUTTONS,
    TOOL_WINDOW,
    find_button,
    label_in_tokens,
    poll_until,
    ratio_box,
    resolve,
)
from poc.workflow_3.monitor.demonstration_rcs_control import (  # noqa: E402
    ALT_SETTLE_SEC,
    CLICK_HOLD_SEC,
    CONFIRM_LABEL_REJECTED,
    CONFIRM_NOT_LOCATED,
    CONFIRM_NOT_VISIBLE,
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
TARGET = "file_manager"            # (MANUAL_CLICK_TARGET) key | label | 'window/label'
# strict: 라벨이 읽혀야만 누른다. lenient 는 다른 라벨(예: 'SECS Terminal')도 '못 읽음' 으로
# 통과시키는데, 버튼이 가려졌을 때 VLM 이 찍는 곳이 바로 덮은 창이다.
CONFIRM_POLICY = "strict"

REVEAL_ATTEMPTS = 3                # (MANUAL_CLICK_REVEAL_ATTEMPTS) 창이 여러 장 겹칠 수 있다
SETTLE_SEC = 1.0                   # Alt+click 뒤 창이 다시 그려질 대기
# Alt+click 지점은 VLM 이 찾은 **덮은 창의 제목줄**이다 - 고정 지점은 창을 빗나갔다
# (2026-09-18 오피스: 창보다 약간 위를 눌러 Alt+click 효과 없음).
COVER_KEY = "covering_window_title"
COVER_DESCRIPTION = (
    "a separate small window or dialog (for example 'SECS Terminal' or 'Terminal "
    "Service') that is floating ON TOP of the group of buttons along the bottom of "
    "the Remote Monitoring screen and hides some of those buttons. If several such "
    "windows overlap, choose the one in front. Point at the middle of that window's "
    "TITLE BAR (the bar at its top showing the window name)."
)

# 클릭 후 열린 창 확인(원격 뷰 안에 그려져 로컬 창 조회로는 못 찾는다 - VLM+OCR).
# 확인 한 번이 VLM+OCR 한 쌍(수 초)이라 '처음 확인된 시각' 의 해상도도 그만큼이다.
OPEN_TIMEOUT_SEC = 15.0            # (MANUAL_CLICK_OPEN_TIMEOUT_SEC)
OPEN_POLL_INTERVAL_SEC = 0.5       # (MANUAL_CLICK_OPEN_POLL_SEC)

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2
EXIT_NOT_VISIBLE = 3
EXIT_NOT_CONFIRMED = 4
EXIT_NOT_OPENED = 5                # 클릭은 했는데 열릴 창을 확인 못 함
EXIT_ALREADY_OPEN = 6              # 누르기 전부터 열릴 창이 보였다 - 누르지 않음


def main() -> int:
    os.environ.setdefault("SAFE_MODE", "0")
    settings = load_workflow3_settings()
    eqp_id = os.environ.get("MANUAL_CLICK_EQP_ID", "").strip() or EQP_ID
    target_name = os.environ.get("MANUAL_CLICK_TARGET", "").strip() or TARGET
    reveal_attempts = max(0, _env_int("MANUAL_CLICK_REVEAL_ATTEMPTS", REVEAL_ATTEMPTS))
    open_timeout = _env_float("MANUAL_CLICK_OPEN_TIMEOUT_SEC", OPEN_TIMEOUT_SEC)
    open_poll = _env_float("MANUAL_CLICK_OPEN_POLL_SEC", OPEN_POLL_INTERVAL_SEC)

    try:
        spec = resolve(target_name, BUTTONS)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return EXIT_PREFLIGHT_FAILED
    if spec.window != TOOL_WINDOW:
        # 팝업 안 버튼은 탐색을 그 팝업 내부로 제한해야 다른 창의 같은 라벨을 안 누른다 -
        # 그 영역을 검증할 수단이 생기기 전에는 누르지 않는다.
        print(f"[ERROR] {spec.key}: '{spec.window}' 창 안 버튼은 아직 지원하지 않습니다.")
        return EXIT_PREFLIGHT_FAILED
    if not spec.reveal:
        reveal_attempts = 0

    mode = "실클릭" if settings.action_enabled else "리허설(SAFE_MODE=1, 클릭/Alt 차단)"
    print(f"[INFO] 버튼 클릭: EQP_ID={eqp_id}, target={spec.key}('{spec.label}'), {mode}, "
          f"등록 위치={spec.center}, 가림해제 Alt+click 최대 {reveal_attempts}회")

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
        reveal_x_ratio=spec.center[0] if spec.center else 0.5,
        reveal_y_ratio=spec.center[1] if spec.center else 0.5,
        confirm_half_width_ratio=spec.confirm_half[0],
        confirm_half_height_ratio=spec.confirm_half[1],
    )

    opened_step = None
    if spec.opens_title:
        opened_step = FlowStep(
            TargetConfig(key=f"{spec.key}_opened", description=spec.opens_description),
            required=spec.opens_title,
        )

    def _opened_now() -> bool:
        _, title_point, _, _ = locate_with_reveal(
            window, opened_step,
            capture_fn=kit.capture, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
            policy="strict", label=opened_step.target.key,
        )
        return title_point is not None

    if opened_step is not None and _opened_now():
        print(f"[WARNING] {spec.key}: 누르기 전부터 열릴 창이 보입니다 - 누르지 않습니다")
        print(f"[DIGEST] manual_click target={spec.key} result=already_open")
        return EXIT_ALREADY_OPEN

    def _should_reveal(image, reason):
        """가려졌을 때만 Alt+click 한다. VLM 이 잘못 짚었을 뿐이면 밀어내지 않는다.

        버튼이 **있어야 할 자리**(등록 탐색 영역)를 확대 OCR 로 다시 읽어 라벨이 읽히면
        '보이는데 못 짚음' 이라 멈춘다 - 그때 Alt+click 하면 멀쩡한 창만 뒤로 간다.
        """
        if reason not in (CONFIRM_NOT_LOCATED, CONFIRM_LABEL_REJECTED) or not spec.reveal:
            return False
        box = ratio_box(spec, image.width, image.height, spec.search_half)
        if box is None:
            return False  # 있어야 할 자리를 모르면 가렸는지도 모른다
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(),
            artifact_label=f"{spec.key}_expected_area",
            log_name="manual_click_button",
        )
        if not read.ok:
            print(f"[WARNING] 예상 영역 OCR 실패 - 가려졌는지 몰라 Alt+click 안 함: {read.error}")
            return False
        tokens = tokens_from_text(read.raw_text)
        if label_in_tokens(tokens, spec.required):
            print(f"[INFO] 예상 영역에 {spec.key} 라벨이 보입니다 - 가려진 게 아니라 VLM 이 "
                  f"잘못 짚었습니다. Alt+click 안 함. box={box}")
            return False
        print(f"[INFO] 예상 영역에 라벨 없음(읽힘={tokens[:12]!r}) - 가려진 것으로 봅니다")
        return True

    cover_target = TargetConfig(key=COVER_KEY, description=COVER_DESCRIPTION)

    def _reveal(window, image, round_index):
        """덮은 창을 VLM 으로 찾아 그 제목줄을 Alt+click 한다. 두 장이면 라운드마다 앞 창.

        짚은 곳에서 대상 라벨이 읽히면 누르지 않는다 - 그건 tool 화면 자체이고, 거기를
        Alt+click 하면 tool 창이 뒤로 간다. 못 찾으면 고정 지점으로 대신 누르지 않는다.
        """
        point = kit.locate(image, cover_target)
        if point is None:
            print("[WARNING] 덮은 창을 찾지 못함 - Alt+click 안 함")
            return False
        tokens = kit.read_tokens(image, point, COVER_KEY)
        print(f"[INFO] 덮은 창 제목줄 후보: px={point} 읽힘={tokens[:12]!r}")
        if label_in_tokens(tokens, spec.required):
            print(f"[WARNING] 짚은 곳에 {spec.key} 라벨이 있습니다 - 덮은 창이 아니라 "
                  "tool 화면이라 Alt+click 안 함")
            return False
        return kit.alt_click_at(window, image, point, round_index, note="덮은 창 제목줄")

    source = {}

    def _confirm(image):
        point, reason, where = find_button(
            image, spec, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
            policy=CONFIRM_POLICY,
        )
        source["where"] = where
        return point, reason

    t_start = time.monotonic()
    image, point, reason, reveals = locate_with_reveal(
        window, FlowStep(TargetConfig(key=spec.key, description=spec.label),
                         required=spec.required),
        capture_fn=kit.capture, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
        policy=CONFIRM_POLICY, reveal_fn=_reveal, max_reveals=reveal_attempts,
        label=spec.key, should_reveal_fn=_should_reveal, confirm_fn=_confirm,
    )
    locate_sec = time.monotonic() - t_start

    if point is None:
        print(f"[DIGEST] manual_click target={spec.key} result={reason} reveals={reveals} "
              f"locate_sec={locate_sec:.1f}")
        return EXIT_NOT_VISIBLE if reason == CONFIRM_NOT_VISIBLE else EXIT_NOT_CONFIRMED

    print(f"[INFO] {spec.key} 확인({source.get('where')}): px={point} -> 레지스트리용 "
          f"center=({point['x'] / image.width:.3f}, {point['y'] / image.height:.3f})")
    t_click = time.monotonic()
    kit.click(window, image, point, spec.key)
    click_sec = time.monotonic() - t_click

    common = (f"target={spec.key} source={source.get('where')} reveals={reveals} "
              f"locate_sec={locate_sec:.1f} click_sec={click_sec:.1f}")
    if opened_step is None:
        # 열릴 창을 모르는 버튼 - 누른 것만 기록하고 효과는 화면으로 대조한다.
        print(f"[DIGEST] manual_click {common} result=clicked effect=unverified")
        return EXIT_OK

    # 다시 누르지 않는다 - 창이 떴는데 확인만 실패한 경우 두 번째 클릭이 무엇을 할지 모른다.
    opened, first_seen, checks = poll_until(
        _opened_now, timeout_sec=open_timeout, interval_sec=open_poll,
        clock=time.monotonic, sleep=time.sleep,
    )
    print(f"[DIGEST] manual_click {common} result=clicked "
          f"effect={'opened' if opened else 'unconfirmed'} "
          f"first_seen_after_click_sec={first_seen:.1f} checks={checks}")
    return EXIT_OK if opened else EXIT_NOT_OPENED


if __name__ == "__main__":
    raise SystemExit(main())
