"""poc 테스트 공통 설정.

cube must-deliver 알림은 보내기 전에 `logs/cube_outbox/` 에 파일로 남고, pump 스레드가
남은 파일을 다시 보낸다. 테스트가 실제 outbox 에 파일을 남기면 **오피스 PC 에서 그
테스트 메시지가 진짜 cube 로 나간다** - 그래서 테스트마다 outbox 를 tmp 로 돌리고
pump 는 띄우지 않는다(재발송은 `retry_outbox_once()` 를 직접 불러 시험한다).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_cube_outbox(monkeypatch, tmp_path):
    from poc.workflow_3.monitor import notify

    monkeypatch.setattr(notify, "_OUTBOX_DIR", tmp_path / "cube_outbox")
    monkeypatch.setattr(notify, "_OUTBOX_IN_FLIGHT", set())
    monkeypatch.setattr(notify, "_OUTBOX_PUMP_ENABLED", False)
