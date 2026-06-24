"""workflow_3e 설정 — workflow_3 설정 + 측정 실패 abort 잡 전용 MEAS_FAIL_* 필드.

Workflow3Settings 를 확장해 abort 잡 토글/게이트/ALID/locator service 를 더한다.
workflow_3/config.py 는 손대지 않는다(확장이 core 를 단방향 import). 로더는
load_workflow3_settings() 를 감싸 base 필드를 그대로 복사하고 4개 필드만 추가한다.

abort 클릭 이중 게이트(보정과 동일):
  * SAFE_MODE=1                  → action_enabled=False (모든 입력 차단, 상속 동작)
  * MEAS_FAIL_ABORT_DRY_RUN (기본 1) → abort 클릭만 추가 차단. 실제 abort 는
    SAFE_MODE off **이고** 이 값이 0일 때만 나간다.
"""

import dataclasses
import os

from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.util import env_flag


def _env_str(name: str, default: str) -> str:
    """공백 제거한 문자열 env (비어 있으면 default)."""
    value = os.environ.get(name, "").strip()
    return value or default


@dataclasses.dataclass(frozen=True)
class Workflow3eSettings(Workflow3Settings):
    """workflow_3 설정 + 측정 실패 abort 잡 필드."""

    # --- 측정 실패 abort 잡 (MES 임계 알람 기반) ---
    meas_fail_abort_enabled: bool = True       # 잡 마스터 토글(검출+알림).
    meas_fail_alid: str = ""                   # 임계 알람 ALID(오피스 확인 필요, 빈값=비활성).
    abort_action_dry_run: bool = True          # 클릭 게이트. 실제 abort 는 SAFE_MODE off + 0 일 때만.
    abort_button_vlm_service: str = "ui-venus" # Abort 버튼 locator route_slug(모델명 아님).


def load_workflow3e_settings() -> Workflow3eSettings:
    """env 오버라이드를 적용해 Workflow3eSettings 를 만든다(workflow_3 로더 위에 4필드 추가)."""
    base = load_workflow3_settings()

    # abort 클릭 이중 게이트: SAFE_MODE 켜지면 env 무관 dry-run(보정과 동일 패턴).
    abort_dry_run_requested = env_flag("MEAS_FAIL_ABORT_DRY_RUN", default=True)
    abort_action_dry_run = abort_dry_run_requested or base.safe_mode

    base_fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return Workflow3eSettings(
        **base_fields,
        meas_fail_abort_enabled=env_flag("MEAS_FAIL_ABORT_ENABLED", default=True),
        meas_fail_alid=_env_str("MEAS_FAIL_ALID", ""),
        abort_action_dry_run=abort_action_dry_run,
        abort_button_vlm_service=_env_str("MEAS_FAIL_ABORT_BUTTON_SERVICE", "ui-venus"),
    )


__all__ = ["Workflow3eSettings", "load_workflow3e_settings"]
