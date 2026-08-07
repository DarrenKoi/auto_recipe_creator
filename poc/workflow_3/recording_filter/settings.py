"""recording_filter 실행 파라미터 — env 주도 dataclass (CLI 인자 없음)."""

from dataclasses import dataclass, field

from poc.workflow_3.util import env_float, env_int
from poc.workflow_3.vlm.flask_vlm import UI_VENUS_MODEL_NAME

# 기본 VLM service slug.
_DEFAULT_SERVICE = "mai-ui"


@dataclass
class RecordingFilterSettings:
    """필터 파이프라인 튜닝 파라미터 (bench 상수 이식)."""

    # ---- Stage 1: cv2 프레임 축소 ----
    diff_threshold: int = 25            # absdiff 이진화 임계
    resize_width: int = 1280            # diff 계산용 다운스케일 폭
    min_change_area_px: int = 5000      # 가장 큰 변화 blob 면적 임계(생존 조건)
    # ---- Stage 2a: 커서-기하 클릭 ----
    cursor_click_window_px: int = 200   # 커서 중심 정사각 ROI 한 변
    click_min_changed_px: int = 1500    # ROI 안 변화 픽셀 임계(클릭 조건)
    click_diff_threshold: int = 25      # native diff 마스크 임계
    # ---- VLM ----
    vlm_service: str = _DEFAULT_SERVICE
    vlm_model: str = field(default_factory=lambda: UI_VENUS_MODEL_NAME)
    vlm_request_delay_sec: float = 1.0  # 프록시 과부하 방지 간격
    max_vlm_calls: int = 0              # 0 = 생존 전체 처리(샘플링 없음)


def load_recording_filter_settings() -> RecordingFilterSettings:
    """env override 를 적용한 설정을 만든다."""
    return RecordingFilterSettings(
        diff_threshold=env_int("RECORDING_FILTER_DIFF_THRESHOLD", 25),
        resize_width=env_int("RECORDING_FILTER_RESIZE_WIDTH", 1280),
        min_change_area_px=env_int("RECORDING_FILTER_MIN_CHANGE_AREA_PX", 5000),
        cursor_click_window_px=env_int("RECORDING_FILTER_CLICK_WINDOW_PX", 200),
        click_min_changed_px=env_int("RECORDING_FILTER_CLICK_MIN_CHANGED_PX", 1500),
        click_diff_threshold=env_int("RECORDING_FILTER_CLICK_DIFF_THRESHOLD", 25),
        vlm_request_delay_sec=env_float("RECORDING_FILTER_VLM_REQUEST_DELAY_SEC", 1.0),
        max_vlm_calls=env_int("RECORDING_FILTER_MAX_VLM_CALLS", 0),
    )
