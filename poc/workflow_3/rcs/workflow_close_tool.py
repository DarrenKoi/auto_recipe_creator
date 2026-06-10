"""열려 있는 RCS tool 창(Remote Monitoring System)을 찾아 닫는다.

`workflow_select_tool.py` 가 List 탭에서 tool 을 더블클릭하면 "Remote Monitoring
System - ...<tool id>..." 창이 뜬다. 이 모듈은 그 창을 **제목의 tool id 로** 찾아
닫는다. 화면 캡처/VLM 은 필요 없다(제목 매칭 + 창 닫기뿐).

닫기는 파괴적 동작이라, 찾은 창 제목에 tool id 가 실제로 들어있는지 한 번 더
확인한 뒤에만 닫는다(엉뚱한 창 닫기 방지).

tool 이름 입력 (connect_tool.py 와 동일한 env 사용 — 접속/닫기를 짝으로 쓸 수 있음):
  1. 환경변수 ACTION_TARGET_TOOL_NAME / ACTION_SELECT_TOOL_NAME / SELECT_TOOL_TARGET_ID
     가 있으면 그 값으로 1회 닫기(비대화 — 스크립트/배치용).
  2. 없으면 대화형 프롬프트로 입력받아 닫고, 반복한다. 빈 줄/'q' 입력 시 종료.

dry-run: CLOSE_TOOL_DRY_RUN=on 이면 창을 찾아 보고만 하고 실제로 닫지는 않는다
(어떤 창을 닫을지 안전하게 먼저 확인할 때 사용).

사용법:
  uv run python poc/workflow_3/rcs/workflow_close_tool.py                        # 대화형
  ACTION_TARGET_TOOL_NAME=MCD630 uv run python poc/workflow_3/rcs/workflow_close_tool.py   # 비대화 1회
  CLOSE_TOOL_DRY_RUN=on ACTION_TARGET_TOOL_NAME=MCD630 uv run python poc/workflow_3/rcs/workflow_close_tool.py
"""

import os
import sys
import time
from dataclasses import dataclass

from poc.workflow_3.rcs.login_rcs_common import (
    REMOTE_MONITORING_WINDOW_TITLE_PREFIX,
    wait_for_remote_monitoring_window,
)
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.util import close_window, format_elapsed_ms
from poc.workflow_3.rcs.workflow_select_tool import load_target_tool_name

LOG_NAME = "workflow_close_tool"
COMPONENT_NAME = LOG_NAME

EXIT_SUCCESS = "success"
EXIT_INVALID_TOOL_NAME = "invalid_tool_name"
EXIT_TOOL_WINDOW_NOT_FOUND = "tool_window_not_found"
EXIT_CLOSE_FAILED = "close_failed"


@dataclass
class ToolCloseResult:
    """tool 창 닫기 결과."""

    exit_code: str
    target_tool_name: str
    window_title: str = ""
    closed: bool = False


def close_tool(
    tool_name: str,
    *,
    action_enabled: bool = True,
    window_timeout_sec: float = 6.0,
    poll_interval_sec: float = 0.5,
) -> ToolCloseResult:
    """지정 tool(EQP_ID)의 Remote Monitoring 창을 찾아 닫는다.

    창을 못 찾으면 EXIT_TOOL_WINDOW_NOT_FOUND, 닫기 실패 시 EXIT_CLOSE_FAILED 를
    반환한다. `action_enabled=False`(dry-run)이면 창만 찾아 보고하고 닫지 않는다.
    """
    normalized = (tool_name or "").strip()
    if not normalized:
        print("[WARNING] close_tool: tool_name 이 비어 있어 닫기를 건너뜁니다.")
        return ToolCloseResult(EXIT_INVALID_TOOL_NAME, tool_name or "")

    started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="close_started",
        log_name=LOG_NAME,
        target_tool_name=normalized,
        action_enabled=action_enabled,
    )

    tool_window, window_title, backend = wait_for_remote_monitoring_window(
        normalized,
        timeout_sec=window_timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )
    if tool_window is None:
        print(
            f"[ERROR] close_tool: tool 창을 찾지 못했습니다 "
            f"(tool={normalized!r}, prefix={REMOTE_MONITORING_WINDOW_TITLE_PREFIX!r}). "
            f"창이 떠 있는지/제목에 tool id 가 있는지 확인하세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="tool_window_not_found",
            level="warning",
            log_name=LOG_NAME,
            target_tool_name=normalized,
        )
        return ToolCloseResult(EXIT_TOOL_WINDOW_NOT_FOUND, normalized)

    # 파괴적 동작 — 닫기 전에 제목에 tool id 가 실제로 들어있는지 한 번 더 확인한다.
    if normalized.lower() not in (window_title or "").lower():
        print(
            f"[ERROR] close_tool: 찾은 창 제목에 tool id 가 없어 닫지 않습니다(안전 차단): "
            f"title={window_title!r}, tool={normalized!r}"
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="tool_id_not_in_title",
            level="warning",
            log_name=LOG_NAME,
            target_tool_name=normalized,
            window_title=window_title,
        )
        return ToolCloseResult(EXIT_TOOL_WINDOW_NOT_FOUND, normalized, window_title)

    if not action_enabled:
        print(
            f"[INFO] [DRY-RUN] tool 창 발견(닫기 생략): tool={normalized!r}, "
            f"title={window_title!r}, backend={backend}"
        )
        return ToolCloseResult(EXIT_SUCCESS, normalized, window_title, closed=False)

    if close_window is None:
        print("[ERROR] close_tool: close_window 유틸을 쓸 수 없습니다(window_utils 미가용).")
        return ToolCloseResult(EXIT_CLOSE_FAILED, normalized, window_title)

    closed = close_window(
        tool_window,
        debug_label=f"close_tool {normalized} backend={backend} title={window_title!r}",
    )
    print(
        f"[INFO] close_tool 완료: tool={normalized!r}, closed={closed}, "
        f"title={window_title!r}, 소요={format_elapsed_ms(started_at)}"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="close_finished",
        log_name=LOG_NAME,
        target_tool_name=normalized,
        window_title=window_title,
        closed=closed,
    )
    return ToolCloseResult(
        EXIT_SUCCESS if closed else EXIT_CLOSE_FAILED,
        normalized,
        window_title,
        closed=closed,
    )


def _action_enabled() -> bool:
    """CLOSE_TOOL_DRY_RUN 이 켜져 있으면 실제 닫기를 생략한다(기본은 실제 닫기)."""
    return os.getenv("CLOSE_TOOL_DRY_RUN", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


def _close_once(tool_name: str, action_enabled: bool) -> str:
    """tool_name 의 창을 1회 닫고 exit_code 를 반환한다."""
    return close_tool(tool_name, action_enabled=action_enabled).exit_code


def main() -> str:
    """env 가 있으면 1회, 없으면 대화형 반복 닫기."""
    action_enabled = _action_enabled()
    if not action_enabled:
        print("[INFO] CLOSE_TOOL_DRY_RUN=on — 실제로 닫지 않고 대상 창만 확인합니다.")

    env_tool = load_target_tool_name()
    if env_tool:
        return _close_once(env_tool, action_enabled)

    print("[INFO] 닫을 tool 이름을 입력하세요 (빈 줄 또는 'q' 입력 시 종료).")
    while True:
        try:
            tool_name = input("close> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not tool_name or tool_name.lower() == "q":
            break
        exit_code = _close_once(tool_name, action_enabled)
        print(f"[INFO] {tool_name!r} → {exit_code}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
