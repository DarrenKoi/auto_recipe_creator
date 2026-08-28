"""workflow_4 adapters — 외부 시스템(workflow_3 등)과의 경계 서브패키지.

이 서브패키지는 **유일하게** workflow_4 가 `poc.workflow_3` 을 import 할 수 있는
경계다. 현재 cycle3 mirror 는 저널 JSON 을 직접 읽어 poc.workflow_3 를 import 하지
않지만(오프라인/테스트 안전), wf3 의존이 필요해지면 이 경계 안에서만 허용한다 —
프레임워크 코어(`framework/`)는 0 deps 를 유지한다.
"""

from .workflow3_cycle import CycleGraphMirror, StepSpec, build_step_chain_graph

__all__ = ["CycleGraphMirror", "StepSpec", "build_step_chain_graph"]
