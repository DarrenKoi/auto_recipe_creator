"""workflow_extract 실행 파라미터 - env 주도 dataclass (CLI 인자 없음)."""

from dataclasses import dataclass

from poc.workflow_3.util import env_flag, env_float, env_int


@dataclass
class WorkflowExtractSettings:
    """그룹핑 규칙 임계값. 전부 첫 실측 후 조정 대상이다."""

    recenter_window_sec: float = 1.5    # R1: 클릭 직후 이 시간 안의 변화를 본다.
    recenter_min_ratio: float = 0.40    # R1: live_box 대비 변화 면적 비율 임계.
    dropdown_max_sec: float = 5.0       # R2: 열기 -> 고르기 최대 간격.
    focus_max_sec: float = 2.0          # R3: 포커스 클릭으로 흡수할 최대 간격.
    repeat_window_sec: float = 6.0      # R4: 반복 클릭으로 묶을 시간 창.
    repeat_min_count: int = 3           # R4: 반복으로 인정할 최소 횟수.
    same_target_px: int = 24            # R4: 라벨이 없을 때 동일 대상 판정 거리.
    thumbnails_enabled: bool = True     # step 별 표시 프레임 저장 on/off.


def load_workflow_extract_settings() -> WorkflowExtractSettings:
    """env override 를 적용한 설정을 만든다."""
    return WorkflowExtractSettings(
        recenter_window_sec=env_float("WORKFLOW_EXTRACT_RECENTER_WINDOW_SEC", 1.5),
        recenter_min_ratio=env_float("WORKFLOW_EXTRACT_RECENTER_MIN_RATIO", 0.40),
        dropdown_max_sec=env_float("WORKFLOW_EXTRACT_DROPDOWN_MAX_SEC", 5.0),
        focus_max_sec=env_float("WORKFLOW_EXTRACT_FOCUS_MAX_SEC", 2.0),
        repeat_window_sec=env_float("WORKFLOW_EXTRACT_REPEAT_WINDOW_SEC", 6.0),
        repeat_min_count=env_int("WORKFLOW_EXTRACT_REPEAT_MIN_COUNT", 3),
        same_target_px=env_int("WORKFLOW_EXTRACT_SAME_TARGET_PX", 24),
        thumbnails_enabled=env_flag("WORKFLOW_EXTRACT_THUMBNAILS", True),
    )
