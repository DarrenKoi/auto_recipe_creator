"""workflow_3e — workflow_3 위에 얹는 알람 잡 확장 패키지.

workflow_3(실시간 align fail 루프)를 **단방향으로 import** 해 재사용하고, 새 MES 알람
기반 잡을 core 를 건드리지 않고 추가한다. 첫 확장은 '측정 실패(연속 N회) abort' 잡이다.

설계 원칙:
  * workflow_3e -> workflow_3 단방향 (확장이 core 를 import, 역방향 금지).
  * 단일 RCS 커서를 공유하므로 통합 슈퍼바이저(`monitor.py`) 한 프로세스에서 두 잡을
    직렬(blocking)로 돌린다. abort 는 큐잉 가능(즉시성 불필요)이라 별도 lock 불필요.
  * abort 클릭은 보정과 동일한 이중 게이트(SAFE_MODE off + MEAS_FAIL_ABORT_DRY_RUN=0).
    기본은 notify-only(검출+증거 캡처+엔지니어 cube, 클릭은 dry-run).

진입점:
    uv run python poc/workflow_3e/monitor.py   # align + 측정실패 abort 통합 루프
"""
