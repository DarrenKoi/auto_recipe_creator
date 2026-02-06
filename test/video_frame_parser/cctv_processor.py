"""
CCTV Processor

CCTV 영상 처리 파이프라인 (프레임 추출 + OCR + 작업 추출 + 에러 감지 + UI 감지)
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from .parser import VideoFrameParser
from .ocr_annotator import OCRAnnotator, create_ocr_annotator
from .action_extractor import ActionExtractor, create_action_extractor
from .error_detector import ErrorDetector, create_error_detector
from .ui_element_detector import UIElementDetector, create_ui_element_detector
from .text_embedder import TextEmbedder, create_text_embedder
from .db_handler import DatabaseHandler
from .models import FrameData
import numpy as np


class CCTVProcessor:
    """
    CCTV 영상 처리 파이프라인

    1. 비디오 프레임 추출
    2. OCR 텍스트 추출
    3. 작업 시퀀스 추출
    4. 에러 패턴 감지
    5. UI 요소 감지
    6. 텍스트 임베딩 생성 및 저장
    """

    def __init__(
        self,
        db_handler: DatabaseHandler,
        video_parser: Optional[VideoFrameParser] = None,
        ocr_annotator: Optional[OCRAnnotator] = None,
        action_extractor: Optional[ActionExtractor] = None,
        error_detector: Optional[ErrorDetector] = None,
        ui_detector: Optional[UIElementDetector] = None,
        text_embedder: Optional[TextEmbedder] = None
    ):
        """
        Args:
            db_handler: DatabaseHandler 인스턴스
            video_parser: VideoFrameParser 인스턴스 (옵션)
            ocr_annotator: OCRAnnotator 인스턴스 (옵션)
            action_extractor: ActionExtractor 인스턴스 (옵션)
            error_detector: ErrorDetector 인스턴스 (옵션)
            ui_detector: UIElementDetector 인스턴스 (옵션)
            text_embedder: TextEmbedder 인스턴스 (옵션)
        """
        self.db = db_handler
        self.video_parser = video_parser
        self.ocr_annotator = ocr_annotator
        self.action_extractor = action_extractor
        self.error_detector = error_detector
        self.ui_detector = ui_detector
        self.text_embedder = text_embedder

        print("[INFO] CCTVProcessor 초기화")

    def process_cctv_video(
        self,
        video_path: str,
        extract_ocr: bool = True,
        extract_actions: bool = True,
        detect_errors: bool = True,
        detect_ui_elements: bool = False,
        generate_text_embeddings: bool = True,
        manual_annotations_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        CCTV 비디오 처리

        Args:
            video_path: 비디오 파일 경로
            extract_ocr: OCR 추출 여부
            extract_actions: 작업 시퀀스 추출 여부
            detect_errors: 에러 감지 여부
            detect_ui_elements: UI 요소 감지 여부
            generate_text_embeddings: 텍스트 임베딩 생성 여부
            manual_annotations_path: 수동 어노테이션 파일 경로 (옵션)

        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        print(f"\n[INFO] CCTV 비디오 처리 시작: {video_path}")

        if not Path(video_path).exists():
            print(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {video_path}")
            return {"success": False, "error": "Video file not found"}

        results = {
            "video_path": video_path,
            "frames_extracted": 0,
            "ocr_results": 0,
            "action_sequences": 0,
            "error_patterns": 0,
            "ui_elements": 0,
            "text_embeddings": 0,
            "processing_time_seconds": 0.0
        }

        # 1. 프레임 추출
        print("\n[STEP 1] 프레임 추출 중...")
        frames = self._extract_frames(video_path)
        if not frames:
            print("[ERROR] 프레임 추출 실패")
            return {"success": False, "error": "Frame extraction failed"}

        results["frames_extracted"] = len(frames)
        print(f"[INFO] {len(frames)}개 프레임 추출 완료")

        # 2. OCR 추출
        if extract_ocr and self.ocr_annotator:
            print("\n[STEP 2] OCR 텍스트 추출 중...")
            ocr_count = self._extract_ocr(frames)
            results["ocr_results"] = ocr_count
            print(f"[INFO] {ocr_count}개 OCR 결과 추출 완료")

        # 3. 작업 시퀀스 추출
        if extract_actions and self.action_extractor:
            print("\n[STEP 3] 작업 시퀀스 추출 중...")
            action_count = self._extract_actions(
                frames, manual_annotations_path
            )
            results["action_sequences"] = action_count
            print(f"[INFO] {action_count}개 작업 시퀀스 추출 완료")

        # 4. 에러 감지
        if detect_errors and self.error_detector:
            print("\n[STEP 4] 에러 패턴 감지 중...")
            error_count = self._detect_errors(frames)
            results["error_patterns"] = error_count
            print(f"[INFO] {error_count}개 에러 패턴 감지 완료")

        # 5. UI 요소 감지
        if detect_ui_elements and self.ui_detector:
            print("\n[STEP 5] UI 요소 감지 중...")
            ui_count = self._detect_ui_elements(frames)
            results["ui_elements"] = ui_count
            print(f"[INFO] {ui_count}개 UI 요소 감지 완료")

        # 6. 텍스트 임베딩 생성
        if generate_text_embeddings and self.text_embedder:
            print("\n[STEP 6] 텍스트 임베딩 생성 중...")
            embedding_count = self._generate_text_embeddings(frames)
            results["text_embeddings"] = embedding_count
            print(f"[INFO] {embedding_count}개 텍스트 임베딩 생성 완료")

        # 처리 시간
        elapsed = time.time() - start_time
        results["processing_time_seconds"] = elapsed
        results["success"] = True

        print(f"\n[INFO] CCTV 비디오 처리 완료 (소요 시간: {elapsed:.1f}초)")
        return results

    def _extract_frames(self, video_path: str) -> List[FrameData]:
        """프레임 추출"""
        if not self.video_parser:
            print("[WARNING] VideoFrameParser가 설정되지 않았습니다")
            return []

        try:
            # VideoFrameParser 사용
            video_id = Path(video_path).stem
            frames = self.video_parser.parse(video_path)
            return frames
        except Exception as e:
            print(f"[ERROR] 프레임 추출 실패: {e}")
            return []

    def _extract_ocr(self, frames: List[FrameData]) -> int:
        """OCR 추출"""
        if not self.ocr_annotator:
            return 0

        total_ocr = 0

        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"[INFO] OCR 처리 중: {i+1}/{len(frames)}")

            # 이미지 로드
            if frame.image_path:
                ocr_results = self.ocr_annotator.extract_text_from_file(
                    frame.image_path, frame.frame_id
                )
            elif frame.image_data is not None:
                ocr_results = self.ocr_annotator.extract_text(
                    frame.image_data, frame.frame_id
                )
            else:
                continue

            # DB 저장
            if ocr_results:
                self.db.save_ocr_results_batch(ocr_results)
                total_ocr += len(ocr_results)

        return total_ocr

    def _extract_actions(
        self,
        frames: List[FrameData],
        annotations_path: Optional[str]
    ) -> int:
        """작업 시퀀스 추출"""
        if not self.action_extractor:
            return 0

        video_id = frames[0].video_id if frames else "unknown"

        # 수동 어노테이션 로드 (있는 경우)
        manual_annotations = None
        if annotations_path:
            try:
                manual_annotations = self.action_extractor.load_annotations(
                    annotations_path
                )
                print(f"[INFO] {len(manual_annotations)}개 수동 어노테이션 로드됨")
            except Exception as e:
                print(f"[WARNING] 어노테이션 로드 실패: {e}")

        # 작업 시퀀스 추출
        action_sequences = self.action_extractor.extract_from_frames(
            frames, video_id, manual_annotations
        )

        # DB 저장
        for action in action_sequences:
            self.db.save_action_sequence(action)

        return len(action_sequences)

    def _detect_errors(self, frames: List[FrameData]) -> int:
        """에러 감지"""
        if not self.error_detector:
            return 0

        total_errors = 0

        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"[INFO] 에러 감지 중: {i+1}/{len(frames)}")

            # 이미지 로드
            if frame.image_path:
                import cv2
                image = cv2.imread(frame.image_path)
            elif frame.image_data is not None:
                image = frame.image_data
            else:
                continue

            # 에러 감지
            errors = self.error_detector.detect_errors(image, frame.frame_id)

            # DB 저장
            if errors:
                self.db.save_error_patterns_batch(errors)
                total_errors += len(errors)

        return total_errors

    def _detect_ui_elements(self, frames: List[FrameData]) -> int:
        """UI 요소 감지"""
        if not self.ui_detector:
            return 0

        total_ui = 0

        # UI 감지는 비용이 높으므로 키프레임만 처리
        keyframes = [f for f in frames if f.is_keyframe]
        print(f"[INFO] {len(keyframes)}개 키프레임에서 UI 요소 감지")

        for i, frame in enumerate(keyframes):
            if i % 5 == 0:
                print(f"[INFO] UI 감지 중: {i+1}/{len(keyframes)}")

            # 이미지 로드
            if frame.image_path:
                import cv2
                image = cv2.imread(frame.image_path)
            elif frame.image_data is not None:
                image = frame.image_data
            else:
                continue

            # UI 요소 감지
            ui_elements = self.ui_detector.detect_elements(
                image, frame.frame_id
            )

            # DB 저장
            if ui_elements:
                self.db.save_ui_elements_batch(ui_elements)
                total_ui += len(ui_elements)

        return total_ui

    def _generate_text_embeddings(self, frames: List[FrameData]) -> int:
        """텍스트 임베딩 생성"""
        if not self.text_embedder:
            return 0

        # 각 프레임의 OCR 결과 가져오기
        texts = []
        frame_ids = []

        for frame in frames:
            ocr_results = self.db.get_ocr_results_by_frame(frame.frame_id)

            if ocr_results:
                # OCR 텍스트 결합
                frame_text = " ".join([ocr.text for ocr in ocr_results])
                texts.append(frame_text)
                frame_ids.append(frame.frame_id)

        if not texts:
            print("[INFO] 텍스트 임베딩 생성할 텍스트 없음")
            return 0

        # 배치 임베딩 생성
        print(f"[INFO] {len(texts)}개 텍스트 임베딩 생성 중...")
        embeddings = self.text_embedder.embed_batch(texts, show_progress=True)

        # DB 저장
        count = self.db.add_text_embeddings_batch(texts, embeddings, frame_ids)

        return count


def create_cctv_processor(
    db_handler: DatabaseHandler,
    use_ocr: bool = True,
    use_gpu: bool = False
) -> CCTVProcessor:
    """
    CCTVProcessor 생성 헬퍼 함수

    Args:
        db_handler: DatabaseHandler 인스턴스
        use_ocr: OCR 사용 여부
        use_gpu: GPU 사용 여부

    Returns:
        CCTVProcessor 인스턴스
    """
    # 각 컴포넌트 초기화
    ocr_annotator = None
    if use_ocr:
        try:
            ocr_annotator = create_ocr_annotator(
                languages=["ko", "en"],
                use_gpu=use_gpu
            )
        except Exception as e:
            print(f"[WARNING] OCR 초기화 실패: {e}")

    action_extractor = create_action_extractor()
    error_detector = create_error_detector(use_color=True, use_vlm=False)
    ui_detector = create_ui_element_detector(method="traditional_cv")

    text_embedder = None
    try:
        text_embedder = create_text_embedder(use_gpu=use_gpu)
    except Exception as e:
        print(f"[WARNING] TextEmbedder 초기화 실패: {e}")

    return CCTVProcessor(
        db_handler=db_handler,
        ocr_annotator=ocr_annotator,
        action_extractor=action_extractor,
        error_detector=error_detector,
        ui_detector=ui_detector,
        text_embedder=text_embedder
    )


if __name__ == "__main__":
    # 사용 예시
    import sys
    from .db_handler import DatabaseHandler
    from .config import DatabaseConfig

    if len(sys.argv) < 2:
        print("Usage: python -m test.video_frame_parser.cctv_processor <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # DatabaseHandler 초기화
    config = DatabaseConfig()
    db = DatabaseHandler(config)
    db.initialize()
    db.init_text_faiss_index()

    # CCTVProcessor 생성
    processor = create_cctv_processor(db, use_ocr=True, use_gpu=False)

    # 비디오 처리
    results = processor.process_cctv_video(
        video_path=video_path,
        extract_ocr=True,
        extract_actions=True,
        detect_errors=True,
        detect_ui_elements=False,  # 비용이 높으므로 기본 비활성화
        generate_text_embeddings=True
    )

    # 결과 출력
    print("\n" + "="*60)
    print("처리 결과:")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")

    # 통계
    stats = db.get_stats()
    print("\n" + "="*60)
    print("데이터베이스 통계:")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")

    db.close()
