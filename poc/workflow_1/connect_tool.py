"""필요할 때 특정 장비(tool)로 RCS 접속하는 수동 헬퍼.

RCS 가 이미 로그인되어 메인 창(List 탭)이 떠 있다고 가정한다. align fail 알람과
무관하게, 엔지니어가 임의 시점에 원하는 tool 로 접속할 때 쓴다. 접속 엔진은
`workflow_select_tool.connect_to_tool` 을 그대로 재사용한다(로직 중복 없음).

tool 이름 입력 (소스 상수를 매번 고칠 필요 없음):
  1. 환경변수 ACTION_TARGET_TOOL_NAME / ACTION_SELECT_TOOL_NAME / SELECT_TOOL_TARGET_ID
     가 있으면 그 값으로 1회 접속(비대화 — 스크립트/배치용).
  2. 없으면 대화형 프롬프트로 입력받아 접속하고, 다시 입력받기를 반복한다.
     빈 줄 또는 'q' 입력 시 종료(한 세션에서 여러 장비를 연속 접속 가능).

dry-run: CONNECT_TOOL_DRY_RUN=on 이면 인식/디버그 저장만 하고 실제 더블클릭은 생략.

사용법:
  uv run python poc/workflow_1/connect_tool.py                    # 대화형
  ACTION_TARGET_TOOL_NAME=MCD630 uv run python poc/workflow_1/connect_tool.py   # 비대화 1회
"""

import os
import sys

from poc.workflow_3.rcs.workflow_select_tool import (
    EXIT_MAIN_WINDOW_NOT_FOUND,
    EXIT_SUCCESS,
    connect_to_tool,
    load_target_tool_name,
)


def _action_enabled() -> bool:
    """CONNECT_TOOL_DRY_RUN 이 켜져 있으면 클릭을 생략한다(기본은 실제 클릭)."""
    return os.getenv("CONNECT_TOOL_DRY_RUN", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


def _connect_once(tool_name: str, action_enabled: bool) -> str:
    """tool_name 으로 1회 접속하고 exit_code 를 반환한다."""
    result = connect_to_tool(tool_name, action_enabled=action_enabled)
    if result is None:
        return EXIT_MAIN_WINDOW_NOT_FOUND
    return result.exit_code


def main() -> str:
    """env 가 있으면 1회, 없으면 대화형 반복 접속."""
    action_enabled = _action_enabled()
    if not action_enabled:
        print("[INFO] CONNECT_TOOL_DRY_RUN=on — 실제 더블클릭 없이 인식만 수행합니다.")

    env_tool = load_target_tool_name()
    if env_tool:
        return _connect_once(env_tool, action_enabled)

    print("[INFO] 접속할 tool 이름을 입력하세요 (빈 줄 또는 'q' 입력 시 종료).")
    while True:
        try:
            tool_name = input("tool> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not tool_name or tool_name.lower() == "q":
            break
        exit_code = _connect_once(tool_name, action_enabled)
        print(f"[INFO] {tool_name!r} → {exit_code}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
