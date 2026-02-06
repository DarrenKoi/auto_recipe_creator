"""
Error Pattern Detector

에러 패턴 감지 (빠른 색상 기반 + VLM 폴백)
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np
import cv2

from .models import ErrorPattern


class ErrorDetector:
    """
    에러 패턴 감지기

    Fast Path: RGB 임계값 기반 빠른 감지 (~1ms)
    VLM Path: 의미적 분석 (~1000ms, 옵션)
    """

    def __init__(
        self,
        use_color_detection: bool = True,
        use_vlm_fallback: bool = False,
        red_threshold: int = 150,
        color_variance_threshold: float = 50.0
    ):
        """
        Args:
            use_color_detection: 색상 기반 감지 사용 여부
            use_vlm_fallback: VLM 폴백 사용 여부
            red_threshold: 빨간색 임계값 (0-255)
            color_variance_threshold: 색상 분산 임계값
        """
        self.use_color_detection = use_color_detection
        self.use_vlm_fallback = use_vlm_fallback
        self.red_threshold = red_threshold
        self.color_variance_threshold = color_variance_threshold

        print(f"[INFO] ErrorDetector initialized")
        print(f"[INFO] Color detection: {use_color_detection}")
        print(f"[INFO] VLM fallback: {use_vlm_fallback}")

    def detect_errors(
        self,
        image: np.ndarray,
        frame_id: str
    ) -> List[ErrorPattern]:
        """
        에러 패턴 감지

        Args:
            image: 입력 이미지 (numpy array, RGB or BGR)
            frame_id: 프레임 ID

        Returns:
            ErrorPattern 리스트
        """
        errors = []

        # Fast Path: 색상 기반 감지
        if self.use_color_detection:
            color_errors = self._detect_by_color(image, frame_id)
            errors.extend(color_errors)

        # VLM Fallback: 의미적 분석
        if self.use_vlm_fallback and len(errors) == 0:
            vlm_errors = self._detect_by_vlm(image, frame_id)
            errors.extend(vlm_errors)

        return errors

    def _detect_by_color(
        self,
        image: np.ndarray,
        frame_id: str
    ) -> List[ErrorPattern]:
        """
        색상 기반 에러 감지 (빠른 경로)

        Args:
            image: 입력 이미지
            frame_id: 프레임 ID

        Returns:
            ErrorPattern 리스트
        """
        errors = []

        # BGR to RGB 변환 (필요시)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV는 BGR 형식 사용
            if image.mean() > 128:  # 간단한 휴리스틱
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
        else:
            return errors

        # 빨간색 영역 감지
        red_mask = self._detect_red_regions(rgb_image)

        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            red_mask, connectivity=8
        )

        # 각 영역 분석
        for i in range(1, num_labels):  # 0은 배경
            area = stats[i, cv2.CC_STAT_AREA]

            # 너무 작은 영역 무시 (노이즈)
            if area < 100:
                continue

            # 바운딩 박스
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = [x, y, x + w, y + h]

            # ErrorPattern 생성
            error = ErrorPattern(
                error_id=self._generate_error_id(frame_id),
                frame_id=frame_id,
                error_type="ui_error",  # 기본값
                error_message="Red region detected (possible error message)",
                severity="medium",
                recovery_action="Check error message and retry",
                bbox=bbox,
                detected_method="color",
                detected_at=datetime.now(),
                confidence=0.7  # 색상 기반은 중간 신뢰도
            )

            errors.append(error)

        if errors:
            print(f"[INFO] Detected {len(errors)} error regions by color in frame {frame_id}")

        return errors

    def _detect_red_regions(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        빨간색 영역 감지

        Args:
            rgb_image: RGB 이미지

        Returns:
            이진 마스크 (빨간색 영역 = 255)
        """
        # HSV 색공간으로 변환
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # 빨간색 범위 (HSV)
        # 빨간색은 HSV에서 두 범위에 걸쳐 있음
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        # 두 마스크 결합
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거 (morphology)
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        return red_mask

    def _detect_by_vlm(
        self,
        image: np.ndarray,
        frame_id: str
    ) -> List[ErrorPattern]:
        """
        VLM 기반 에러 감지 (느린 경로, 정확)

        Args:
            image: 입력 이미지
            frame_id: 프레임 ID

        Returns:
            ErrorPattern 리스트
        """
        # TODO: VLM API 통합
        # - VLMScreenAnalyzer 사용
        # - 프롬프트: "화면에 에러 메시지가 있습니까? 있다면 위치와 내용을 알려주세요"
        # - 응답 파싱하여 ErrorPattern 생성

        print(f"[INFO] VLM-based error detection not yet implemented for frame {frame_id}")
        return []

    def classify_error_type(self, error_message: str) -> str:
        """
        에러 메시지로부터 에러 타입 분류

        Args:
            error_message: 에러 메시지

        Returns:
            에러 타입 ("connection", "authentication", "timeout", etc.)
        """
        message_lower = error_message.lower()

        # 키워드 기반 분류
        if any(kw in message_lower for kw in ["connect", "connection", "네트워크", "연결"]):
            return "connection"
        elif any(kw in message_lower for kw in ["auth", "login", "password", "인증", "로그인"]):
            return "authentication"
        elif any(kw in message_lower for kw in ["timeout", "시간 초과", "응답 없음"]):
            return "timeout"
        elif any(kw in message_lower for kw in ["permission", "access", "권한", "접근"]):
            return "permission"
        elif any(kw in message_lower for kw in ["not found", "찾을 수 없음", "없음"]):
            return "not_found"
        else:
            return "ui_error"

    def classify_severity(self, error_type: str, area: int) -> str:
        """
        에러 심각도 분류

        Args:
            error_type: 에러 타입
            area: 에러 영역 크기 (픽셀)

        Returns:
            심각도 ("low", "medium", "high", "critical")
        """
        # 타입별 기본 심각도
        severity_map = {
            "connection": "high",
            "authentication": "high",
            "timeout": "medium",
            "permission": "medium",
            "not_found": "low",
            "ui_error": "low"
        }

        base_severity = severity_map.get(error_type, "low")

        # 영역 크기로 조정
        if area > 10000:  # 큰 에러 메시지
            if base_severity == "low":
                return "medium"
            elif base_severity == "medium":
                return "high"

        return base_severity

    def suggest_recovery_action(self, error_type: str) -> str:
        """
        에러 타입에 따른 복구 작업 제안

        Args:
            error_type: 에러 타입

        Returns:
            복구 작업 설명
        """
        recovery_map = {
            "connection": "네트워크 연결 확인 및 서버 주소 검증 후 재시도",
            "authentication": "사용자 인증 정보 확인 및 재입력",
            "timeout": "서버 응답 대기 후 재시도 또는 타임아웃 설정 증가",
            "permission": "관리자 권한으로 실행 또는 권한 설정 확인",
            "not_found": "대상 리소스 존재 여부 확인",
            "ui_error": "에러 메시지 확인 및 적절한 조치 수행"
        }

        return recovery_map.get(error_type, "에러 메시지를 확인하고 적절한 조치를 취하세요")

    def _generate_error_id(self, frame_id: str) -> str:
        """에러 고유 ID 생성"""
        return f"{frame_id}_error_{uuid.uuid4().hex[:8]}"


def create_error_detector(
    use_color: bool = True,
    use_vlm: bool = False
) -> ErrorDetector:
    """
    ErrorDetector 생성 헬퍼 함수

    Args:
        use_color: 색상 기반 감지 사용
        use_vlm: VLM 폴백 사용

    Returns:
        ErrorDetector 인스턴스
    """
    return ErrorDetector(
        use_color_detection=use_color,
        use_vlm_fallback=use_vlm
    )


if __name__ == "__main__":
    # 사용 예시
    import sys

    if len(sys.argv) < 2:
        print("Usage: python error_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame_id = "test_frame_001"

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    # 에러 감지
    detector = create_error_detector(use_color=True, use_vlm=False)
    errors = detector.detect_errors(image, frame_id)

    # 결과 출력
    print(f"\n[INFO] Found {len(errors)} error patterns:")
    for i, error in enumerate(errors, 1):
        print(f"\n{i}. Type: {error.error_type}")
        print(f"   Message: {error.error_message}")
        print(f"   Severity: {error.severity}")
        print(f"   BBox: {error.bbox}")
        print(f"   Recovery: {error.recovery_action}")
