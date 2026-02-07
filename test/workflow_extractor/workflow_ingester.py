"""
Workflow Ingester - 어노테이션을 DB에 저장하는 파이프라인

수동/자동 추출된 워크플로우 어노테이션을 기존 DatabaseHandler,
FrameAnalyzer, TextEmbedder를 활용하여 MongoDB + FAISS에 저장합니다.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python 또는 numpy가 설치되지 않았습니다.")

try:
    from video_frame_parser.models import (
        FrameData,
        FrameType,
        AnalysisResult,
        AnalysisStatus,
        ActionSequence,
    )
    from video_frame_parser.config import (
        DatabaseConfig,
        AnalyzerConfig,
        ExtractorConfig,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARNING] video_frame_parser 모델을 불러올 수 없습니다.")

try:
    from video_frame_parser.db_handler import DatabaseHandler
    DB_HANDLER_AVAILABLE = True
except ImportError:
    DB_HANDLER_AVAILABLE = False
    print("[WARNING] DatabaseHandler를 불러올 수 없습니다.")

try:
    from video_frame_parser.analyzer import FrameAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("[WARNING] FrameAnalyzer를 불러올 수 없습니다.")

try:
    from video_frame_parser.text_embedder import TextEmbedder
    TEXT_EMBEDDER_AVAILABLE = True
except ImportError:
    TEXT_EMBEDDER_AVAILABLE = False
    print("[WARNING] TextEmbedder를 불러올 수 없습니다.")

try:
    from video_frame_parser.extractor import VideoFrameExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    print("[WARNING] VideoFrameExtractor를 불러올 수 없습니다.")

from .models import WorkflowAnnotation, WorkflowStep


class WorkflowIngester:
    """
    워크플로우 어노테이션을 DB에 저장하는 파이프라인.

    파이프라인:
    1. WorkflowAnnotation JSON 로드
    2. 각 단계의 스크린샷 프레임 → CLIP 임베딩 생성 (FrameAnalyzer)
    3. 단계 설명 → 텍스트 임베딩 생성 (TextEmbedder)
    4. ActionSequence 레코드 생성
    5. FrameData + AnalysisResult 생성
    6. MongoDB + FAISS에 저장 (DatabaseHandler)
    """

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        analyzer_config: Optional[AnalyzerConfig] = None,
        use_gpu: bool = False,
    ):
        """
        Args:
            db_config: 데이터베이스 설정
            analyzer_config: CLIP 분석기 설정
            use_gpu: GPU 사용 여부
        """
        self._db_config = db_config or DatabaseConfig()
        self._analyzer_config = analyzer_config or AnalyzerConfig()
        self._use_gpu = use_gpu

        self._db: Optional[DatabaseHandler] = None
        self._analyzer: Optional[FrameAnalyzer] = None
        self._text_embedder: Optional[TextEmbedder] = None
        self._extractor: Optional[VideoFrameExtractor] = None
        self._initialized = False

    def initialize(self) -> None:
        """컴포넌트 초기화"""
        if self._initialized:
            return

        # DatabaseHandler 초기화
        if DB_HANDLER_AVAILABLE:
            self._db = DatabaseHandler(self._db_config)
            self._db.initialize(embedding_dim=self._analyzer_config.embedding_dim)
            print("[INFO] DatabaseHandler 초기화 완료")
        else:
            print("[WARNING] DatabaseHandler 사용 불가 - MongoDB/FAISS 저장 건너뜁니다.")

        # FrameAnalyzer 초기화
        if ANALYZER_AVAILABLE:
            self._analyzer = FrameAnalyzer(self._analyzer_config)
            self._analyzer.initialize()
            print("[INFO] FrameAnalyzer 초기화 완료")
        else:
            print("[WARNING] FrameAnalyzer 사용 불가 - CLIP 임베딩 생성 건너뜁니다.")

        # TextEmbedder 초기화
        if TEXT_EMBEDDER_AVAILABLE:
            device = "cuda" if self._use_gpu else "cpu"
            self._text_embedder = TextEmbedder(device=device)
            print("[INFO] TextEmbedder 초기화 완료")

            # 텍스트 FAISS 인덱스 초기화
            if self._db is not None:
                self._db.init_text_faiss_index(embedding_dim=self._text_embedder.dimension)
        else:
            print("[WARNING] TextEmbedder 사용 불가 - 텍스트 임베딩 건너뜁니다.")

        # VideoFrameExtractor 초기화
        if EXTRACTOR_AVAILABLE:
            self._extractor = VideoFrameExtractor()
            print("[INFO] VideoFrameExtractor 초기화 완료")
        else:
            print("[WARNING] VideoFrameExtractor 사용 불가 - 프레임 추출은 cv2 직접 사용합니다.")

        self._initialized = True
        print("[INFO] WorkflowIngester 초기화 완료")

    def ingest_annotation(self, annotation: WorkflowAnnotation) -> Dict[str, Any]:
        """
        단일 워크플로우 어노테이션을 DB에 저장합니다.

        Args:
            annotation: 워크플로우 어노테이션

        Returns:
            인제스트 결과 요약
        """
        if not self._initialized:
            self.initialize()

        result = {
            "workflow_id": annotation.workflow_id,
            "frames_saved": 0,
            "embeddings_saved": 0,
            "text_embeddings_saved": 0,
            "action_sequences_saved": 0,
            "errors": [],
        }

        # 1. 영상에서 프레임 추출
        print(f"[INFO] 워크플로우 인제스트 시작: {annotation.workflow_id}")
        frames = self._extract_step_frames(annotation)
        if not frames:
            result["errors"].append("프레임 추출 실패")
            return result

        # 2. 프레임을 DB에 저장
        frame_count = self._save_frames(frames)
        result["frames_saved"] = frame_count

        # 3. CLIP 임베딩 생성 및 저장
        if self._analyzer is not None:
            embed_count = self._generate_and_save_embeddings(frames)
            result["embeddings_saved"] = embed_count

        # 4. 텍스트 임베딩 생성 및 저장
        if self._text_embedder is not None:
            text_count = self._generate_and_save_text_embeddings(
                annotation, frames
            )
            result["text_embeddings_saved"] = text_count

        # 5. ActionSequence 생성 및 저장
        action_count = self._create_and_save_action_sequences(annotation, frames)
        result["action_sequences_saved"] = action_count

        print(f"[INFO] 인제스트 완료: {result}")
        return result

    def ingest_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        JSON 파일에서 어노테이션을 로드하여 인제스트합니다.

        Args:
            json_path: 어노테이션 JSON 파일 경로

        Returns:
            인제스트 결과 요약
        """
        print(f"[INFO] JSON 로드: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotation = WorkflowAnnotation.from_dict(data)
        return self.ingest_annotation(annotation)

    def ingest_batch(self, json_dir: str) -> List[Dict[str, Any]]:
        """
        디렉토리 내 모든 어노테이션 JSON을 일괄 인제스트합니다.

        Args:
            json_dir: 어노테이션 JSON 디렉토리

        Returns:
            각 어노테이션의 인제스트 결과 리스트
        """
        json_dir = Path(json_dir)
        results = []

        json_files = sorted(json_dir.glob("workflow_*.json"))
        print(f"[INFO] {len(json_files)}개 어노테이션 파일 발견")

        for json_file in json_files:
            try:
                result = self.ingest_from_json(str(json_file))
                results.append(result)
            except Exception as e:
                print(f"[ERROR] 인제스트 실패: {json_file.name} - {e}")
                results.append({
                    "workflow_id": json_file.stem,
                    "errors": [str(e)],
                })

        success = sum(1 for r in results if not r.get("errors"))
        print(f"[INFO] 일괄 인제스트 완료: {success}/{len(results)} 성공")
        return results

    def _extract_step_frames(
        self, annotation: WorkflowAnnotation
    ) -> List[FrameData]:
        """어노테이션의 각 단계에서 프레임을 추출합니다."""
        if not CV2_AVAILABLE:
            print("[ERROR] opencv-python이 필요합니다.")
            return []

        video_path = annotation.video_path
        if not Path(video_path).exists():
            print(f"[ERROR] 영상 파일 없음: {video_path}")
            return []

        frames = []

        # VideoFrameExtractor 사용 가능 시 활용
        if self._extractor is not None:
            metadata = self._extractor.open(video_path)
            video_id = metadata.video_id

            timestamps = [step.timestamp for step in annotation.steps]
            frames = self._extractor.extract_frames_batch(timestamps)
            self._extractor.close()
        else:
            # cv2 직접 사용 (폴백)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] 영상 열기 실패: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            video_id = self._generate_video_id(video_path)

            for step in annotation.steps:
                frame_num = int(step.timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    print(f"[WARNING] 프레임 읽기 실패: 단계 {step.step_number}, "
                          f"frame {frame_num}")
                    continue

                frame_id = f"{video_id}_f{frame_num:08d}"
                frame_data = FrameData(
                    frame_id=frame_id,
                    video_id=video_id,
                    frame_number=frame_num,
                    timestamp=step.timestamp,
                    image_data=frame,
                    frame_type=FrameType.KEYFRAME,
                    is_keyframe=True,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                )
                frames.append(frame_data)

            cap.release()

        print(f"[INFO] {len(frames)}개 프레임 추출 완료 (영상: {Path(video_path).name})")
        return frames

    def _save_frames(self, frames: List[FrameData]) -> int:
        """프레임을 DB에 저장합니다."""
        if self._db is None:
            return 0

        count = self._db.save_frames_batch(frames)
        print(f"[INFO] {count}개 프레임 DB 저장 완료")
        return count

    def _generate_and_save_embeddings(self, frames: List[FrameData]) -> int:
        """CLIP 임베딩을 생성하고 저장합니다."""
        if self._analyzer is None or self._db is None:
            return 0

        results = self._analyzer.analyze_frames_batch(frames)
        saved = 0

        for result in results:
            if result.status == AnalysisStatus.COMPLETED:
                self._db.save_analysis_result(result)
                saved += 1

        print(f"[INFO] {saved}개 CLIP 임베딩 저장 완료")
        return saved

    def _generate_and_save_text_embeddings(
        self,
        annotation: WorkflowAnnotation,
        frames: List[FrameData],
    ) -> int:
        """텍스트 임베딩을 생성하고 저장합니다."""
        if self._text_embedder is None or self._db is None:
            return 0

        texts = []
        frame_ids = []

        for i, step in enumerate(annotation.steps):
            # 단계 설명 텍스트 구성
            text = f"{step.action_type}: {step.target_description}"
            if step.input_text:
                text += f" [{step.input_text}]"
            if step.notes:
                text += f" ({step.notes})"

            texts.append(text)

            # 대응하는 프레임 ID 매칭
            if i < len(frames):
                frame_ids.append(frames[i].frame_id)
            else:
                frame_ids.append(f"step_{step.step_number}")

        if not texts:
            return 0

        # 배치 임베딩 생성
        embeddings = self._text_embedder.embed_batch(texts)

        # DB 저장
        count = self._db.add_text_embeddings_batch(texts, embeddings, frame_ids)
        print(f"[INFO] {count}개 텍스트 임베딩 저장 완료")
        return count

    def _create_and_save_action_sequences(
        self,
        annotation: WorkflowAnnotation,
        frames: List[FrameData],
    ) -> int:
        """ActionSequence를 생성하고 저장합니다."""
        if not frames:
            return 0

        video_id = frames[0].video_id

        # 전체 워크플로우를 하나의 ActionSequence로 변환
        actions = []
        for step in annotation.steps:
            action_dict = {
                "type": step.action_type,
                "target": step.target_description,
                "timestamp": step.timestamp,
            }
            if step.coordinates:
                action_dict["x"] = step.coordinates[0]
                action_dict["y"] = step.coordinates[1]
            if step.input_text:
                action_dict["text"] = step.input_text
            if step.notes:
                action_dict["notes"] = step.notes
            actions.append(action_dict)

        # 시작/종료 프레임 ID
        start_frame_id = frames[0].frame_id
        end_frame_id = frames[-1].frame_id
        start_time = annotation.steps[0].timestamp
        end_time = annotation.steps[-1].timestamp

        action_id = hashlib.md5(
            f"{annotation.workflow_id}_{start_time}_{end_time}".encode()
        ).hexdigest()[:16]

        action_sequence = ActionSequence(
            action_id=action_id,
            video_id=video_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            start_time=start_time,
            end_time=end_time,
            actions=actions,
            description=annotation.description,
            success=annotation.success,
            extraction_method=annotation.extraction_method,
            confidence=1.0 if annotation.extraction_method == "manual" else 0.8,
        )

        # DB 저장
        if self._db is not None:
            self._db.save_action_sequence(action_sequence)
            print(f"[INFO] ActionSequence 저장 완료: {action_id}")
            return 1

        return 0

    def _generate_video_id(self, video_path: str) -> str:
        """파일 기반 비디오 ID 생성"""
        path = Path(video_path)
        stat = path.stat()
        id_str = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(id_str.encode()).hexdigest()[:16]

    def close(self) -> None:
        """리소스 정리"""
        if self._db is not None:
            self._db.close()
        if self._analyzer is not None:
            self._analyzer.cleanup()
        if self._extractor is not None:
            self._extractor.close()
        print("[INFO] WorkflowIngester 종료")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(
        description="워크플로우 어노테이션 DB 인제스트 도구"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ingest 서브커맨드 (단일 파일)
    ingest_parser = subparsers.add_parser("ingest", help="단일 어노테이션 인제스트")
    ingest_parser.add_argument("json_file", help="어노테이션 JSON 파일 경로")
    ingest_parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ingest_parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 안 함")

    # batch 서브커맨드 (디렉토리)
    batch_parser = subparsers.add_parser("batch", help="일괄 인제스트")
    batch_parser.add_argument("json_dir", help="어노테이션 JSON 디렉토리")
    batch_parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    batch_parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 안 함")

    args = parser.parse_args()

    if args.command in ("ingest", "batch"):
        db_config = DatabaseConfig(mongo_uri=args.mongo_uri)
        use_gpu = not args.no_gpu

        with WorkflowIngester(db_config=db_config, use_gpu=use_gpu) as ingester:
            if args.command == "ingest":
                result = ingester.ingest_from_json(args.json_file)
                print(f"\n결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                results = ingester.ingest_batch(args.json_dir)
                for r in results:
                    print(f"  {r.get('workflow_id')}: "
                          f"frames={r.get('frames_saved', 0)}, "
                          f"embeddings={r.get('embeddings_saved', 0)}, "
                          f"errors={r.get('errors', [])}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
