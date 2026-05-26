"""RCS tool 창을 1장 캡처해 align_images 아래 captured_img_from_rcs 에 적재하고 닫는다.

`align_fail_alarm.py` 가 fail 이벤트 시 호출하는 재사용 코어이자, alarm 없이
독립적으로 (접속 → 스크린샷 → 창 닫기) 를 검증하는 standalone 테스트 진입점이다.

잦은 align fail 에 대응하기 위해 별도 realtime 캡처 대신 fail 시점에 1장만 박제하고
창을 닫아 다음 fail 과의 경합을 없앤다. 장비는 fail 시 정지하므로 SEM 모니터 화면이
정적이라 단일 스크린샷으로 충분하다.

저장 경로:
    align_images/<eqp_id>/<class>/<recipe>/captured_img_from_rcs/<tag>_rcs.jpg
    (recipe_id 가 "<class>/<recipe>" 형태라 eqp_id 와 합치면 3단계가 된다.)

독립 실행 (office, RCS 로그인 상태 가정):
    ALIGN_CAPTURE_EQP_ID=MCD916 \
    ALIGN_CAPTURE_RECIPE_ID="RJ1BXXX/Z_RJ1B_CBLHM2_FULL" \
      uv run python poc/workflow_1/rcs_screenshot.py
"""

import os
import time
from pathlib import Path

from poc.workflow_1 import ALIGN_IMAGES_DIR
from poc.workflow_1.debug_artifacts import save_debug_jpeg
from poc.workflow_1.login_rcs_common import wait_for_remote_monitoring_window # pyright: ignore[reportMissingImports]
from poc.workflow_1.util import (
    WINDOW_UTILS_AVAILABLE,
    capture_window,
    close_window,
    env_int,
    make_timestamp_tag,
)

# 접속(tool 더블클릭)은 선택 의존성 — standalone 실행에서만 필요하고, align_fail_alarm
# 은 자체적으로 접속한 뒤 capture 코어만 호출하므로 여기서 import 실패해도 무방하다.
try:
    from poc.workflow_1.workflow_select_tool import connect_to_tool # pyright: ignore[reportMissingImports]

    CONNECT_TOOL_AVAILABLE = True
except Exception as _connect_tool_import_exc:
    connect_to_tool = None
    CONNECT_TOOL_AVAILABLE = False
    print(f"[WARNING] workflow_select_tool 로드 실패 - 독립 실행 접속 비활성화: {_connect_tool_import_exc}")

# 캡처 이미지를 적재할 sibling 폴더명 (align_img_from_rcp / align_img_from_msr 와 나란히).
CAPTURED_RCS_DIRNAME = "captured_img_from_rcs"
# 접속 직후 tool 창이 뜰 때까지 대기 타임아웃(초).
CAPTURE_RCS_WINDOW_TIMEOUT_SEC = env_int("ALIGN_FAIL_CAPTURE_WINDOW_TIMEOUT_SEC", 15)
# 창이 뜬 뒤 SEM 영상이 렌더될 시간을 주는 settle 대기(초).
CAPTURE_RCS_SETTLE_SEC = float(env_int("ALIGN_FAIL_CAPTURE_SETTLE_SEC", 2))
# 독립 실행 시 접속 단계의 메인 창 탐색 타임아웃(초).
CONNECT_WINDOW_TIMEOUT_SEC = env_int("ALIGN_FAIL_CONNECT_WINDOW_TIMEOUT_SEC", 3)


def captured_dir_for(eqp_id: str, recipe_id: str) -> Path:
    """captured_img_from_rcs 저장 폴더 경로를 만든다.

    recipe_id 는 실제로 ``<class>/<recipe>`` 형태라, eqp_id 와 합치면 그대로
    ``align_images/<eqp>/<class>/<recipe>`` 3단계가 된다 (슬래시가 단계 구분).
    """
    recipe_rel = recipe_id.replace("\\", "/").strip("/")
    recipe_parts = [part for part in recipe_rel.split("/") if part]
    return ALIGN_IMAGES_DIR.joinpath(eqp_id, *recipe_parts, CAPTURED_RCS_DIRNAME)


def capture_and_close_rcs_window(
    eqp_id: str,
    recipe_id: str,
    *,
    window_timeout_sec: float = CAPTURE_RCS_WINDOW_TIMEOUT_SEC,
    settle_sec: float = CAPTURE_RCS_SETTLE_SEC,
) -> Path | None:
    """이미 열린 RCS tool 창을 찾아 1장 캡처·저장하고 창을 닫는다.

    접속(더블클릭)은 호출부 책임이다. 캡처 성공 시 저장 경로를, 실패/생략 시
    None 을 반환한다. 예외는 삼켜 호출 루프가 죽지 않게 한다.
    """
    if not WINDOW_UTILS_AVAILABLE:
        print(
            f"[INFO] window_utils 비활성 — RCS 캡처 생략 (os={os.name}, "
            f"WINDOW_UTILS_AVAILABLE={WINDOW_UTILS_AVAILABLE})"
        )
        return None
    try:
        tool_window, window_title, backend = wait_for_remote_monitoring_window(
            eqp_id,
            timeout_sec=window_timeout_sec,
        )
        if tool_window is None:
            print(f"[WARNING] RCS tool 창을 찾지 못해 캡처 생략: EQP_ID={eqp_id}")
            return None

        if settle_sec > 0:
            time.sleep(settle_sec)

        image = capture_window(tool_window)
        captured_dir = captured_dir_for(eqp_id, recipe_id)
        captured_dir.mkdir(parents=True, exist_ok=True)
        out_path = captured_dir / f"{make_timestamp_tag()}_rcs.jpg"
        save_debug_jpeg(image, out_path)
        print(f"[INFO] RCS 캡처 저장: EQP_ID={eqp_id}, path={out_path}")

        close_window(
            tool_window,
            debug_label=f"rcs_tool title={window_title!r} backend={backend}",
        )
        return out_path
    except Exception as exc:
        print(f"[WARNING] RCS 캡처 예외: EQP_ID={eqp_id}, error={exc}")
        return None


def connect_capture_close(
    eqp_id: str,
    recipe_id: str,
    *,
    action_enabled: bool = True,
) -> Path | None:
    """독립 실행용: 접속(더블클릭) → 캡처 → 닫기 를 한 번에 수행한다."""
    if not CONNECT_TOOL_AVAILABLE:
        print("[ERROR] connect_to_tool 비활성 — 독립 실행 불가 (office Windows 에서 실행하세요).")
        return None
    if not eqp_id:
        print("[ERROR] EQP_ID 가 비어 있습니다 (ALIGN_CAPTURE_EQP_ID).")
        return None

    print(f"[INFO] RCS 접속 시도: EQP_ID={eqp_id}")
    result = connect_to_tool(
        eqp_id,
        action_enabled=action_enabled,
        main_window_timeout_sec=CONNECT_WINDOW_TIMEOUT_SEC,
    )
    if result is None:
        print(f"[ERROR] 접속 실패 — 캡처 생략: EQP_ID={eqp_id}")
        return None
    if not getattr(result, "double_clicked", False):
        print(
            f"[WARNING] tool 더블클릭이 수행되지 않았습니다 "
            f"(action_enabled={action_enabled}) — 창이 안 열렸을 수 있음: EQP_ID={eqp_id}"
        )

    return capture_and_close_rcs_window(eqp_id, recipe_id)


def main() -> str:
    """ALIGN_CAPTURE_EQP_ID / ALIGN_CAPTURE_RECIPE_ID 로 접속→캡처→닫기 단독 실행."""
    eqp_id = os.getenv("ALIGN_CAPTURE_EQP_ID", "").strip()
    recipe_id = os.getenv("ALIGN_CAPTURE_RECIPE_ID", "").strip()
    out_path = connect_capture_close(eqp_id, recipe_id)
    return "success" if out_path is not None else "failed"


if __name__ == "__main__":
    raise SystemExit(0 if main() == "success" else 1)
