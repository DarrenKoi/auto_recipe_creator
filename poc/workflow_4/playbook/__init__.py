"""Recovery Playbook 도메인 계층 - GUI 도, workflow_3 도 모르는 순수 데이터 계층.

이 패키지는 **workflow_3 를 import 하지 않는다.** 관측값은 plain dict/list 로 들어오고,
판정은 순수 함수가 한다. 그래야 같은 evaluator 를 candidate 생성 / offline replay /
shadow rule 선택이 함께 쓰면서 서로 어긋나지 않는다.

첫 조각은 Recovery Outcome 파생(`outcome.py`)이다. Verification 우선순위와 Outcome
파생의 **유일한 소유자**이며, 이후 티켓이 Guard 평가와 rule 선택으로 확장한다.
"""

from poc.workflow_4.playbook.outcome import (
    ABORTED,
    ESCALATED,
    OUTCOMES,
    RECOVERED,
    UNKNOWN,
    OutcomeResult,
    derive_outcome,
    format_episode_digest,
)

__all__ = [
    "ABORTED",
    "ESCALATED",
    "OUTCOMES",
    "RECOVERED",
    "UNKNOWN",
    "OutcomeResult",
    "derive_outcome",
    "format_episode_digest",
]
