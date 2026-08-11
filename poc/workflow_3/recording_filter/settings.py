"""recording_filter 실행 파라미터 — env 주도 dataclass (CLI 인자 없음)."""

from dataclasses import dataclass

from poc.workflow_3.util import env_flag, env_float, env_int

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
    # service 는 route slug 다. 모델명은 서비스 엔트리가 들고 있으므로 여기서
    # 따로 두지 않는다 - 예전 vlm_model 필드는 mai-ui 라우트에 ui-venus 모델명을
    # 실어 보내는 불일치를 만들었다(2026-08-10 최종 리뷰 FINDING 4).
    vlm_service: str = _DEFAULT_SERVICE
    vlm_request_delay_sec: float = 1.0  # 프록시 과부하 방지 간격
    max_vlm_calls: int = 0              # 0 = 생존 전체 처리(샘플링 없음)
    # ---- Stage 1.5: 영역 게이트 ----
    region_gate_enabled: bool = True     # 0 이면 게이트 없이 전부 candidate.
    # ---- Stage 2c: 요소 라벨링 ----
    element_crop_px: int = 260           # 클릭 지점 주변 crop 한 변.
    element_ocr_service: str = "paddleocr-vl-1.5"
    element_vlm_service: str = "mai-ui"
    element_label_enabled: bool = True
    # ---- Stage 2b: 타이핑 구간 ----
    typing_detect_enabled: bool = True
    typing_cursor_still_px: int = 8       # 이 이상 움직이면 구간이 끊긴다.
    typing_min_burst_events: int = 3      # 구간으로 인정할 최소 change event 수.
    typing_burst_idle_sec: float = 1.5    # 변화가 이 시간 없으면 구간 종료.
    typing_focus_max_sec: float = 2.0     # 구간 직전 이 시간 안의 클릭을 필드로 본다.
    typing_ocr_service: str = "paddleocr-vl-1.5"
    # 국소성(locality) 가드 - 2026-08-11 리뷰 C2. 변화가 "필드 근처"에서 일어났음을
    # 요구하지 않으면, 커서를 세워둔 채 화면 아무 데나 반복해 바뀌는 것(예: 진행률
    # 패널 리페인트)이 전부 타이핑 구간이 되고 OCR 이 그 패널의 숫자를 값으로
    # 복원해 낸다 - value_source="ocr", confidence=1.0 인 완전한 허구 step 이다.
    typing_roi_max_px: int = 200           # 필드 기준점에서 change_bbox 중심까지 최대 거리.
    typing_roi_max_area_px: int = 40000    # 구간 ROI(합집합) 최대 면적(200x200 상당).


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
        region_gate_enabled=env_flag("RECORDING_FILTER_REGION_GATE", True),
        element_crop_px=env_int("RECORDING_FILTER_ELEMENT_CROP_PX", 260),
        element_label_enabled=env_flag("RECORDING_FILTER_ELEMENT_LABEL", True),
        typing_detect_enabled=env_flag("RECORDING_FILTER_TYPING_DETECT", True),
        typing_cursor_still_px=env_int("RECORDING_FILTER_TYPING_CURSOR_STILL_PX", 8),
        typing_min_burst_events=env_int("RECORDING_FILTER_TYPING_MIN_BURST_EVENTS", 3),
        typing_burst_idle_sec=env_float("RECORDING_FILTER_TYPING_BURST_IDLE_SEC", 1.5),
        typing_focus_max_sec=env_float("RECORDING_FILTER_TYPING_FOCUS_MAX_SEC", 2.0),
        typing_roi_max_px=env_int("RECORDING_FILTER_TYPING_ROI_MAX_PX", 200),
        typing_roi_max_area_px=env_int("RECORDING_FILTER_TYPING_ROI_MAX_AREA_PX", 40000),
    )
