"""
Workflow 4 package — 상태 머신 워크플로 프레임워크 레이어.

observe -> decide -> act -> verify 개념 위에서, 워크플로가 어디에 있는지(current node),
어디로 갈 수 있는지(transitions), 실패 시 어떻게 retry / fallback 되는지, 그리고 live
graph(현재 노드 강조)를 어떻게 시각화하는지 추적하는 상태 머신 프레임워크다.

workflow_4 는 poc.workflow_1 / poc.workflow_2 / poc.workflow_3 을 import 하지 않는다
(self-contained framework layer — workflow_3 의 철학을 따름). 외부 workflow 라이브러리
의존 없이 hand-rolled FSM 이며, 시각화는 mermaid snapshot 문자열 생성뿐이다(0 deps).

예외는 단 한 곳: `poc.workflow_4.adapters/` — 외부 시스템(wf3)과의 경계
서브패키지로, workflow_4 가 poc.workflow_3 을 import 할 수 있는 유일한 지점이다
(현재 cycle3 mirror 는 저널 JSON 을 직접 읽어 wf3 를 import 하지 않음).
"""

from pathlib import Path

WORKFLOW4_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW4_DIR / "debug_images"
LOG_DIR = WORKFLOW4_DIR / "logs"

# 런타임 산출물 디렉터리를 패키지 import 시점에 확보한다 (.gitignore 로 관리).
DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "DEBUG_IMAGE_DIR",
    "LOG_DIR",
    "WORKFLOW4_DIR",
]