"""
OCR Annotator for Frame Text Extraction

EasyOCR 기반 프레임 텍스트 추출 및 어노테이션
"""

from typing import List, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARNING] easyocr not available. Install with: pip install easyocr")

from .models import OCRResult


class OCRAnnotator:
    """
    EasyOCR 기반 OCR 텍스트 추출기

    한국어와 영어를 지원하며, 바운딩 박스와 신뢰도 점수 제공
    """

    def __init__(
        self,
        languages: List[str] = ["ko", "en"],
        gpu: bool = False,
        min_confidence: float = 0.3
    ):
        """
        Args:
            languages: 지원 언어 목록 (["ko", "en"])
            gpu: GPU 사용 여부
            min_confidence: 최소 신뢰도 임계값 (이하 결과 필터링)
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "easyocr is required. Install with: pip install easyocr"
            )

        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence

        print(f"[INFO] Initializing EasyOCR with languages: {languages}")
        print(f"[INFO] GPU enabled: {gpu}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("[INFO] EasyOCR initialized successfully")

    def extract_text(
        self,
        image: np.ndarray,
        frame_id: str,
        detail: int = 1
    ) -> List[OCRResult]:
        """
        이미지에서 텍스트 추출

        Args:
            image: 입력 이미지 (numpy array, RGB or BGR)
            frame_id: 프레임 ID
            detail: OCR 상세도 (0=fast, 1=balanced, 2=accurate)

        Returns:
            OCRResult 리스트
        """
        # EasyOCR 실행
        results = self.reader.readtext(image, detail=detail)

        ocr_results = []
        for detection in results:
            bbox_coords, text, confidence = detection

            # 신뢰도 필터링
            if confidence < self.min_confidence:
                continue

            # 바운딩 박스 변환 (polygon → rectangle)
            bbox = self._polygon_to_rectangle(bbox_coords)

            # 언어 감지
            language = self._detect_language(text)

            # OCRResult 생성
            ocr_result = OCRResult(
                ocr_id=self._generate_ocr_id(frame_id),
                frame_id=frame_id,
                text=text,
                language=language,
                confidence=float(confidence),
                bbox=bbox,
                extracted_at=datetime.now(),
                ocr_model="easyocr"
            )

            ocr_results.append(ocr_result)

        print(f"[INFO] Extracted {len(ocr_results)} text regions from frame {frame_id}")
        return ocr_results

    def extract_text_from_file(
        self,
        image_path: str,
        frame_id: str,
        detail: int = 1
    ) -> List[OCRResult]:
        """
        이미지 파일에서 텍스트 추출

        Args:
            image_path: 이미지 파일 경로
            frame_id: 프레임 ID
            detail: OCR 상세도

        Returns:
            OCRResult 리스트
        """
        image = np.array(Image.open(image_path))
        return self.extract_text(image, frame_id, detail=detail)

    def extract_text_batch(
        self,
        images: List[np.ndarray],
        frame_ids: List[str],
        detail: int = 1
    ) -> List[List[OCRResult]]:
        """
        배치 이미지에서 텍스트 추출

        Args:
            images: 이미지 리스트
            frame_ids: 프레임 ID 리스트
            detail: OCR 상세도

        Returns:
            각 이미지의 OCRResult 리스트의 리스트
        """
        if len(images) != len(frame_ids):
            raise ValueError("images and frame_ids must have the same length")

        all_results = []
        for image, frame_id in zip(images, frame_ids):
            results = self.extract_text(image, frame_id, detail=detail)
            all_results.append(results)

        return all_results

    def _polygon_to_rectangle(self, polygon: List[List[int]]) -> List[int]:
        """
        폴리곤 좌표를 직사각형 좌표로 변환

        Args:
            polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns:
            [x1, y1, x2, y2] (top-left, bottom-right)
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))

        return [x1, y1, x2, y2]

    def _detect_language(self, text: str) -> str:
        """
        텍스트의 언어 감지

        Args:
            text: 입력 텍스트

        Returns:
            "ko", "en", 또는 "mixed"
        """
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in text)
        has_english = any(char.isalpha() and ord(char) < 128 for char in text)

        if has_korean and has_english:
            return "mixed"
        elif has_korean:
            return "ko"
        elif has_english:
            return "en"
        else:
            return "unknown"

    def _generate_ocr_id(self, frame_id: str) -> str:
        """OCR 결과 고유 ID 생성"""
        return f"{frame_id}_ocr_{uuid.uuid4().hex[:8]}"

    def get_full_text(self, ocr_results: List[OCRResult]) -> str:
        """
        OCR 결과들을 하나의 텍스트로 결합

        Args:
            ocr_results: OCRResult 리스트

        Returns:
            결합된 텍스트
        """
        # y 좌표 기준으로 정렬 (위에서 아래로)
        sorted_results = sorted(ocr_results, key=lambda r: r.bbox[1])
        return " ".join([r.text for r in sorted_results])

    def filter_by_confidence(
        self,
        ocr_results: List[OCRResult],
        min_confidence: float
    ) -> List[OCRResult]:
        """
        신뢰도 기준으로 필터링

        Args:
            ocr_results: OCRResult 리스트
            min_confidence: 최소 신뢰도

        Returns:
            필터링된 OCRResult 리스트
        """
        return [r for r in ocr_results if r.confidence >= min_confidence]

    def filter_by_language(
        self,
        ocr_results: List[OCRResult],
        language: str
    ) -> List[OCRResult]:
        """
        언어 기준으로 필터링

        Args:
            ocr_results: OCRResult 리스트
            language: 언어 코드 ("ko", "en", "mixed")

        Returns:
            필터링된 OCRResult 리스트
        """
        return [r for r in ocr_results if r.language == language]

    def find_text_containing(
        self,
        ocr_results: List[OCRResult],
        keyword: str,
        case_sensitive: bool = False
    ) -> List[OCRResult]:
        """
        특정 키워드를 포함하는 텍스트 찾기

        Args:
            ocr_results: OCRResult 리스트
            keyword: 검색 키워드
            case_sensitive: 대소문자 구분 여부

        Returns:
            키워드를 포함하는 OCRResult 리스트
        """
        if not case_sensitive:
            keyword = keyword.lower()

        matching_results = []
        for result in ocr_results:
            text = result.text if case_sensitive else result.text.lower()
            if keyword in text:
                matching_results.append(result)

        return matching_results


def create_ocr_annotator(
    languages: List[str] = ["ko", "en"],
    use_gpu: bool = False,
    min_confidence: float = 0.3
) -> OCRAnnotator:
    """
    OCRAnnotator 생성 헬퍼 함수

    Args:
        languages: 지원 언어 목록
        use_gpu: GPU 사용 여부
        min_confidence: 최소 신뢰도

    Returns:
        OCRAnnotator 인스턴스
    """
    return OCRAnnotator(
        languages=languages,
        gpu=use_gpu,
        min_confidence=min_confidence
    )


if __name__ == "__main__":
    # 사용 예시
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_annotator.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame_id = "test_frame_001"

    # OCR 실행
    annotator = create_ocr_annotator(use_gpu=False)
    results = annotator.extract_text_from_file(image_path, frame_id)

    # 결과 출력
    print(f"\n[INFO] Found {len(results)} text regions:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: {result.text}")
        print(f"   Language: {result.language}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   BBox: {result.bbox}")

    # 전체 텍스트 추출
    full_text = annotator.get_full_text(results)
    print(f"\n[INFO] Full text: {full_text}")
