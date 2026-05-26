"""Align Fail 발생 장비로 RCS 접속(tool 더블클릭) 오케스트레이션 — workflow_2.

RCS 는 이미 로그인되어 List 탭이 보이는 상태라고 가정한다. align_fail_alarm 이
감지한 EQP_ID 를 받아 좌측 tool list 에서 해당 row 를 찾아 더블클릭한다.

무거운 머신(VLM client, locator 엔진, 윈도우 자동화)은 workflow_1 을 공유
라이브러리로 그대로 import 한다 (다른 workflow_2 모듈과 동일). workflow_1 의
``workflow_select_tool.main()`` 과 달리 접속 대상 tool 이름을 고정 default 로
두지 않고, 호출 시 인자(EQP_ID)로 받는다.

사용법(standalone 테스트):
  ALIGN_FAIL_CONNECT_TARGET=MCD630 uv run python poc/workflow_2/workflow_select_tool.py
"""

import os
import sys
import time
from pathlib import Path

from poc.workflow_1.login_rcs_common import wait_for_rcs_main_window # pyright: ignore[reportMissingImports]
from poc.workflow_1.util import format_elapsed_ms # pyright: ignore[reportMissingImports]
from poc.workflow_1.workflow_select_tool import ( # pyright: ignore[reportMissingImports]
    EXIT_MAIN_WINDOW_NOT_FOUND,
    EXIT_SUCCESS,
    ToolSelectionResult,
    select_tool_from_main_window,
)

WORKFLOW_2_DIR = Path(__file__).resolve().parent
DEBUG_ARTIFACT_DIR = WORKFLOW_2_DIR / "debug_images" / "workflow_select_tool"

EXIT_INVALID_TARGET = "invalid_connect_target"


def connect_to_tool(
    tool_name: str,
    *,
    action_enabled: bool = True,
    debug_image_dir=None,
    main_window_timeout_sec: float = 15.0,
) -> ToolSelectionResult | None:
    """지정 tool(EQP_ID)로 RCS 접속 — List 탭에서 찾아 더블클릭한다.

    RCS 가 로그인되어 메인 창이 떠 있다고 가정한다. ``tool_name`` 은
    align_fail_alarm 의 EQP_ID 를 그대로 넘긴다 (고정 default 없음).

    action_enabled=False 면 클릭만 생략하고 인식/디버그 저장은 그대로 수행한다.
    메인 창을 못 찾으면 None 을 반환한다.
    """
    normalized = (tool_name or "").strip()
    if not normalized:
        print("[WARNING] connect_to_tool: tool_name 이 비어 있어 접속을 건너뜁니다.")
        return None

    started_at = time.time()
    main_window, window_title, backend = wait_for_rcs_main_window(
        timeout_sec=main_window_timeout_sec,
    )
    if main_window is None:
        print(
            f"[ERROR] connect_to_tool: 메인 RCS 창을 찾지 못해 접속 실패 "
            f"(tool={normalized!r}). RCS 로그인 상태인지 확인하세요."
        )
        return None

    result = select_tool_from_main_window(
        main_window,
        window_title,
        backend,
        normalized,
        action_enabled=action_enabled,
        debug_image_dir=debug_image_dir or DEBUG_ARTIFACT_DIR,
    )
    print(
        f"[INFO] connect_to_tool 완료: tool={normalized!r}, "
        f"result={result.exit_code}, double_clicked={result.double_clicked}, "
        f"소요={format_elapsed_ms(started_at)}"
    )
    return result


def main() -> str:
    """standalone 테스트용 — ALIGN_FAIL_CONNECT_TARGET 으로 tool 이름을 받는다."""
    target = os.getenv("ALIGN_FAIL_CONNECT_TARGET", "").strip()
    if not target:
        print(
            "[ERROR] ALIGN_FAIL_CONNECT_TARGET 환경변수로 접속할 tool 이름을 지정하세요. "
            "(예: ALIGN_FAIL_CONNECT_TARGET=MCD630)"
        )
        return EXIT_INVALID_TARGET

    result = connect_to_tool(target)
    if result is None:
        return EXIT_MAIN_WINDOW_NOT_FOUND
    return result.exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
