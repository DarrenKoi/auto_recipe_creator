"""열린 tool 창에서 버튼 하나를 확인 후 누른다. 있어야 할 버튼이 안 보이면 폴백으로
가린 창을 Alt+click 해 뒤로 밀어내고 다시 찾는다.

기본 대상은 **File Manager** 버튼(라이브 SEM box 아래). 이 버튼은 'SECS Terminal',
'Terminal Service' 같은 창에 자주 가려지는데, 엔지니어는 그 자리를 Alt+click 해서 덮은
창을 뒤로 보내고 버튼을 되살린다. 이 진입점이 같은 일을 한다:

  1. 제목에 EQP_ID 가 든 Remote Monitoring 창에 붙는다(접속은 하지 않는다 - 엔지니어가
     먼저 연다, `manual_align_correction.py` 와 같은 규약).
  2. VLM 이 버튼 좌표를 찍고 PaddleOCR 이 그 자리 라벨을 확인한다.
  3. **좌표가 안 나오거나 그 자리 라벨이 File Manager 가 아니면**(가려짐 - VLM 이 덮은
     창을 짚는다) 버튼이 있어야 할 자리(REVEAL_X/Y_RATIO)를 Alt+click 하고 다시 찾는다.
     최대 REVEAL_ATTEMPTS 번.
  4. 확인되면 클릭한다. 확인이 안 되면 누르지 않는다.

다른 가려진 버튼에 쓰려면 TARGET_* 상수만 바꾼다. 클릭/가림 해제 배선은
`demonstration_rcs_control.build_click_kit` 를 그대로 쓴다(원격 클릭 성사 조건이 오피스
실측값이라 포크하지 않는다).

실행: uv run python poc/workflow_3/monitor/manual_click_button.py
리허설(클릭/Alt 차단): SAFE_MODE=1 uv run python ...
종료 코드: 0=클릭함, 2=사전조건 실패, 3=가림 해제 후에도 못 찾음, 4=라벨 불일치/미검출
"""

import os
import sys
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
TARGET_DESCRIPTION = (
    "the 'File Manager' button in the Remote Monitoring window. It is located "
    "directly BELOW the large live SEM image box. Use the first letter 'F' as the "
    "anchor, then click safely inside the File Manager button area."
)
# OCR 확인: 묶음 하나를 통째로 만족해야 한다. 'FileManager' 로 붙여 읽혀도 통과한다.
TARGET_REQUIRED = (("file", "manager"),)
TARGET_FORBIDDEN = ("cancel", "exit", "terminat", "close", "취소", "종료", "닫기")
# strict: 'File Manager' 가 읽혀야만 누른다. lenient 는 다른 라벨(예: 'SECS Terminal')도
# '못 읽음' 으로 통과시키는데, 버튼이 가려졌을 때 VLM 이 찍는 곳이 바로 덮은 창이다.
CONFIRM_POLICY = "strict"

# 가림 해제: 버튼이 **있어야 할 자리**를 Alt+click 한다(그 위를 덮은 창이 뒤로 간다).
# 창 크기 대비 비율. ponytail: 추정값 - 첫 오피스 실행에서 콘솔의 px/screen 으로 맞출 것.
REVEAL_X_RATIO = 0.30              # (MANUAL_CLICK_REVEAL_X_RATIO)
REVEAL_Y_RATIO = 0.85              # (MANUAL_CLICK_REVEAL_Y_RATIO)
REVEAL_ATTEMPTS = 3                # (MANUAL_CLICK_REVEAL_ATTEMPTS) 창이 여러 장 겹칠 수 있다
SETTLE_SEC = 1.0                   # Alt+click 뒤 창이 다시 그려질 대기

EXIT_OK = 0
EXIT_PREFLIGHT_FAILED = 2
EXIT_NOT_VISIBLE = 3
EXIT_NOT_CONFIRMED = 4


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

    kit = build_click_kit(
        settings,
        debug_dir=debug_root() / "manual_click_button" / make_timestamp_tag(),
        log_component="manual_click_button",
        settle_sec=SETTLE_SEC,
        pre_click_settle_sec=PRE_CLICK_SETTLE_SEC,
        click_hold_sec=CLICK_HOLD_SEC,
        alt_settle_sec=ALT_SETTLE_SEC,
        reveal_x_ratio=x_ratio,
        reveal_y_ratio=y_ratio,
    )
    step = FlowStep(
        TargetConfig(key=TARGET_KEY, description=TARGET_DESCRIPTION),
        required=TARGET_REQUIRED,
        forbidden=TARGET_FORBIDDEN,
    )
    image, point, reason, reveals = locate_with_reveal(
        window, step,
        capture_fn=kit.capture, locate_fn=kit.locate, read_tokens_fn=kit.read_tokens,
        policy=CONFIRM_POLICY, reveal_fn=kit.reveal, max_reveals=reveal_attempts,
        label=TARGET_KEY,
        # 라벨 불일치도 '덮은 창을 짚었다' 로 보고 밀어낸다(가려지는 게 이 버튼의 일상).
        reveal_on=(CONFIRM_NOT_LOCATED, CONFIRM_LABEL_REJECTED),
    )
    if point is None:
        print(f"[DIGEST] manual_click target={TARGET_KEY} result={reason} reveals={reveals}")
        return EXIT_NOT_VISIBLE if reason == CONFIRM_NOT_VISIBLE else EXIT_NOT_CONFIRMED

    kit.click(window, image, point, TARGET_KEY)
    print(f"[DIGEST] manual_click target={TARGET_KEY} result=clicked reveals={reveals}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
