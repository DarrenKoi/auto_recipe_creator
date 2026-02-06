"""
UI Element Detector

VLM 기반 UI 요소 감지 (버튼, 입력 필드, 라벨 등)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import numpy as np

from .models import UIElement


class UIElementDetector:
    """
    VLM 기반 UI 요소 감지기

    버튼, 입력 필드, 라벨, 드롭다운 등의 UI 요소를
    감지하고 위치를 추출
    """

    def __init__(
        self,
        detection_method: str = "vlm",
        vlm_provider: Optional[Any] = None,
        min_confidence: float = 0.5
    ):
        """
        Args:
            detection_method: 감지 방법 ("vlm", "yolo", "traditional_cv")
            vlm_provider: VLM 프로바이더 (VLMScreenAnalyzer 인스턴스)
            min_confidence: 최소 신뢰도 임계값
        """
        self.detection_method = detection_method
        self.vlm_provider = vlm_provider
        self.min_confidence = min_confidence

        print(f"[INFO] UIElementDetector initialized with method: {detection_method}")

        if detection_method == "vlm" and vlm_provider is None:
            print("[WARNING] VLM provider not set. VLM-based detection will fail.")

    def detect_elements(
        self,
        image: np.ndarray,
        frame_id: str,
        element_types: Optional[List[str]] = None
    ) -> List[UIElement]:
        """
        UI 요소 감지

        Args:
            image: 입력 이미지 (numpy array, RGB or BGR)
            frame_id: 프레임 ID
            element_types: 감지할 요소 타입 (None이면 모든 타입)
                ["button", "input", "label", "dropdown", "checkbox", etc.]

        Returns:
            UIElement 리스트
        """
        if element_types is None:
            element_types = ["button", "input", "label", "dropdown", "checkbox"]

        if self.detection_method == "vlm":
            return self._detect_by_vlm(image, frame_id, element_types)
        elif self.detection_method == "yolo":
            return self._detect_by_yolo(image, frame_id, element_types)
        elif self.detection_method == "traditional_cv":
            return self._detect_by_cv(image, frame_id, element_types)
        else:
            print(f"[ERROR] Unknown detection method: {self.detection_method}")
            return []

    def _detect_by_vlm(
        self,
        image: np.ndarray,
        frame_id: str,
        element_types: List[str]
    ) -> List[UIElement]:
        """
        VLM 기반 UI 요소 감지

        Args:
            image: 입력 이미지
            frame_id: 프레임 ID
            element_types: 감지할 요소 타입

        Returns:
            UIElement 리스트
        """
        if self.vlm_provider is None:
            print("[ERROR] VLM provider not set")
            return []

        # TODO: VLM API 통합
        # 프롬프트 예시:
        # "화면에서 다음 UI 요소들을 찾아주세요: 버튼, 입력 필드, 라벨
        #  각 요소에 대해 다음 정보를 JSON 형식으로 제공해주세요:
        #  - type: 요소 타입
        #  - label: 요소의 텍스트/라벨
        #  - bbox: 바운딩 박스 [x1, y1, x2, y2]
        #  - state: 요소 상태 (enabled/disabled/focused)"

        print(f"[INFO] VLM-based UI detection not yet implemented for frame {frame_id}")
        return []

    def _detect_by_yolo(
        self,
        image: np.ndarray,
        frame_id: str,
        element_types: List[str]
    ) -> List[UIElement]:
        """
        YOLO 기반 UI 요소 감지

        Args:
            image: 입력 이미지
            frame_id: 프레임 ID
            element_types: 감지할 요소 타입

        Returns:
            UIElement 리스트
        """
        # TODO: YOLO 모델 통합
        # - 사전 학습된 UI 요소 감지 모델 로드
        # - 추론 실행
        # - 결과를 UIElement로 변환

        print(f"[INFO] YOLO-based UI detection not yet implemented for frame {frame_id}")
        return []

    def _detect_by_cv(
        self,
        image: np.ndarray,
        frame_id: str,
        element_types: List[str]
    ) -> List[UIElement]:
        """
        전통적 컴퓨터 비전 기반 UI 요소 감지

        Args:
            image: 입력 이미지
            frame_id: 프레임 ID
            element_types: 감지할 요소 타입

        Returns:
            UIElement 리스트
        """
        import cv2

        elements = []

        # 간단한 휴리스틱 기반 감지
        # TODO: 더 정교한 알고리즘 구현

        # 회색조 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 엣지 감지
        edges = cv2.Canny(gray, 50, 150)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 직사각형 영역 필터링 (버튼/입력 필드 후보)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 크기 필터링
            if w < 20 or h < 10 or w > 500 or h > 100:
                continue

            # 종횡비 필터링 (버튼/입력 필드는 가로가 세로보다 김)
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 10:
                continue

            # UIElement 생성 (타입은 추론)
            element_type = self._infer_element_type(w, h, aspect_ratio)

            if element_type not in element_types:
                continue

            element = UIElement(
                element_id=self._generate_element_id(frame_id),
                frame_id=frame_id,
                element_type=element_type,
                label="",  # CV 방식으로는 라벨 추출 어려움
                bbox=[x, y, x + w, y + h],
                confidence=0.5,  # 낮은 신뢰도
                detection_method="traditional_cv",
                detected_at=datetime.now()
            )

            elements.append(element)

        print(f"[INFO] Detected {len(elements)} UI elements by CV in frame {frame_id}")
        return elements

    def _infer_element_type(
        self,
        width: int,
        height: int,
        aspect_ratio: float
    ) -> str:
        """
        크기와 종횡비로 요소 타입 추론

        Args:
            width: 너비
            height: 높이
            aspect_ratio: 종횡비

        Returns:
            요소 타입
        """
        # 간단한 휴리스틱
        if aspect_ratio > 5:
            return "input"  # 긴 입력 필드
        elif aspect_ratio > 2:
            return "button"  # 버튼
        else:
            return "label"  # 라벨 또는 기타

    def filter_by_type(
        self,
        elements: List[UIElement],
        element_type: str
    ) -> List[UIElement]:
        """
        타입으로 필터링

        Args:
            elements: UIElement 리스트
            element_type: 요소 타입

        Returns:
            필터링된 UIElement 리스트
        """
        return [e for e in elements if e.element_type == element_type]

    def filter_by_label(
        self,
        elements: List[UIElement],
        keyword: str,
        case_sensitive: bool = False
    ) -> List[UIElement]:
        """
        라벨 키워드로 필터링

        Args:
            elements: UIElement 리스트
            keyword: 검색 키워드
            case_sensitive: 대소문자 구분

        Returns:
            필터링된 UIElement 리스트
        """
        if not case_sensitive:
            keyword = keyword.lower()

        matching = []
        for element in elements:
            label = element.label if case_sensitive else element.label.lower()
            if keyword in label:
                matching.append(element)

        return matching

    def find_element_at_position(
        self,
        elements: List[UIElement],
        x: int,
        y: int
    ) -> Optional[UIElement]:
        """
        특정 위치의 UI 요소 찾기

        Args:
            elements: UIElement 리스트
            x: X 좌표
            y: Y 좌표

        Returns:
            해당 위치의 UIElement (없으면 None)
        """
        for element in elements:
            x1, y1, x2, y2 = element.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                return element

        return None

    def _generate_element_id(self, frame_id: str) -> str:
        """UI 요소 고유 ID 생성"""
        return f"{frame_id}_ui_{uuid.uuid4().hex[:8]}"


def create_ui_element_detector(
    method: str = "vlm",
    vlm_provider: Optional[Any] = None
) -> UIElementDetector:
    """
    UIElementDetector 생성 헬퍼 함수

    Args:
        method: 감지 방법
        vlm_provider: VLM 프로바이더

    Returns:
        UIElementDetector 인스턴스
    """
    return UIElementDetector(
        detection_method=method,
        vlm_provider=vlm_provider
    )


if __name__ == "__main__":
    # 사용 예시
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python ui_element_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame_id = "test_frame_001"

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    # UI 요소 감지 (CV 방식)
    detector = create_ui_element_detector(method="traditional_cv")
    elements = detector.detect_elements(image, frame_id)

    # 결과 출력
    print(f"\n[INFO] Found {len(elements)} UI elements:")
    for i, element in enumerate(elements, 1):
        print(f"\n{i}. Type: {element.element_type}")
        print(f"   Label: {element.label}")
        print(f"   BBox: {element.bbox}")
        print(f"   Confidence: {element.confidence:.4f}")

    # 결과 시각화
    output_image = image.copy()
    for element in elements:
        x1, y1, x2, y2 = element.bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output_image, element.element_type,
            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )

    output_path = "ui_detection_result.png"
    cv2.imwrite(output_path, output_image)
    print(f"\n[INFO] Visualization saved to {output_path}")
