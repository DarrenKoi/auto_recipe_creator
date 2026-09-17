"""monitor 테스트 공통 설정.

`run_alarm_cycle` 의 finally 는 보관 상한 정리(`prune_recordings`/`prune_debug_images`)를
**실제** ALIGN_IMAGES_DIR / DEBUG_IMAGE_DIR 에 건다. 사이클을 도는 테스트가 그대로 두면
개발 PC 의 debug_images 를 지운다(2026-09-17 실제로 .gitkeep 까지 지워졌다). 정리 로직
자체는 tmp_path 에서 직접 시험한다(test_cycle_images / test_prelude_recording).
"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_retention_prune(monkeypatch):
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "prune_recordings", lambda *a, **k: [])
    monkeypatch.setattr(cycle, "prune_debug_images", lambda *a, **k: 0)
