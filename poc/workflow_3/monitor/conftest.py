"""monitor 테스트 공통 설정.

사이클/모니터를 도는 테스트는 **실제** EVENTS_DIR 에 이벤트 폴더를 만들고, finally 의
보관 상한 정리(`prune_events`)로 개발 PC 의 이벤트 폴더를 지운다(2026-09-17 구 정리
로직이 실제로 debug_images 의 .gitkeep 까지 지웠다). 그래서 루트는 테스트마다 tmp 로
돌리고 정리는 끈다. 정리 로직 자체는 util/test_event_dir.py 가 tmp_path 에서 시험한다.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_event_folders(monkeypatch, tmp_path):
    from poc.workflow_3.monitor import cycle, recovery_episode

    monkeypatch.setattr(cycle, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(recovery_episode, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(cycle, "prune_events", lambda *a, **k: 0)
