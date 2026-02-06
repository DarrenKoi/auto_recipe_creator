"""
RAG Context Manager

하이브리드 검색 기반 RAG 컨텍스트 관리자 (시각적 + 텍스트 + 시간적)
"""

from typing import List, Optional, Tuple
import time
import numpy as np
from PIL import Image
import io

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from video_frame_parser.models import (
    FrameData, ActionSequence, ErrorPattern, UIElement,
    OCRResult, RAGContext
)
from video_frame_parser.db_handler import DatabaseHandler
from video_frame_parser.text_embedder import TextEmbedder
from video_frame_parser.analyzer import FrameAnalyzer


class RAGContextManager:
    """
    RAG 컨텍스트 관리자

    하이브리드 검색 (시각적 + 텍스트)로 관련 컨텍스트를 검색하고
    시간적 확장과 메타데이터 enrichment 수행
    """

    def __init__(
        self,
        db_handler: DatabaseHandler,
        text_embedder: Optional[TextEmbedder] = None,
        frame_analyzer: Optional[FrameAnalyzer] = None,
        visual_weight: float = 0.6,
        text_weight: float = 0.4
    ):
        """
        Args:
            db_handler: DatabaseHandler 인스턴스
            text_embedder: TextEmbedder 인스턴스 (옵션)
            frame_analyzer: FrameAnalyzer 인스턴스 (옵션)
            visual_weight: 시각적 유사도 가중치
            text_weight: 텍스트 유사도 가중치
        """
        self.db = db_handler
        self.text_embedder = text_embedder
        self.frame_analyzer = frame_analyzer
        self.visual_weight = visual_weight
        self.text_weight = text_weight

        print("[INFO] RAGContextManager initialized")
        print(f"[INFO] Visual weight: {visual_weight}, Text weight: {text_weight}")

    def retrieve_context(
        self,
        current_screen: bytes,
        query_text: Optional[str] = None,
        top_k: int = 3,
        temporal_window: float = 5.0,
        include_actions: bool = True,
        include_errors: bool = True,
        include_ui_elements: bool = True,
        include_ocr: bool = True
    ) -> RAGContext:
        """
        현재 화면에 대한 RAG 컨텍스트 검색

        Args:
            current_screen: 현재 화면 이미지 (bytes)
            query_text: 쿼리 텍스트 (옵션)
            top_k: 검색할 프레임 수
            temporal_window: 시간적 확장 범위 (초)
            include_actions: 작업 시퀀스 포함 여부
            include_errors: 에러 패턴 포함 여부
            include_ui_elements: UI 요소 포함 여부
            include_ocr: OCR 결과 포함 여부

        Returns:
            RAGContext
        """
        start_time = time.time()

        # 1. 하이브리드 검색 (시각적 + 텍스트)
        similar_frames, scores = self._hybrid_search(
            current_screen, query_text, top_k
        )

        # 2. 시간적 확장 (전후 프레임)
        temporal_frames = self._expand_temporal(
            similar_frames, temporal_window
        )

        # 3. 메타데이터 enrichment
        action_sequences = []
        error_patterns = []
        ui_elements = []
        ocr_results = []

        if include_actions:
            action_sequences = self._enrich_with_actions(similar_frames)

        if include_errors:
            error_patterns = self._enrich_with_errors(similar_frames)

        if include_ui_elements:
            ui_elements = self._enrich_with_ui_elements(similar_frames)

        if include_ocr:
            ocr_results = self._enrich_with_ocr(similar_frames)

        # 4. RAGContext 생성
        elapsed_ms = (time.time() - start_time) * 1000

        context = RAGContext(
            similar_frames=similar_frames,
            action_sequences=action_sequences,
            error_patterns=error_patterns,
            ui_elements=ui_elements,
            temporal_frames=temporal_frames,
            retrieval_scores=scores,
            retrieval_time_ms=elapsed_ms,
            query_text=query_text,
            retrieval_method="hybrid" if query_text else "visual",
            top_k=top_k,
            ocr_results=ocr_results
        )

        print(f"[INFO] Retrieved RAG context in {elapsed_ms:.1f}ms")
        print(f"[INFO] Found {len(similar_frames)} similar frames, "
              f"{len(action_sequences)} actions, {len(error_patterns)} errors")

        return context

    def _hybrid_search(
        self,
        current_screen: bytes,
        query_text: Optional[str],
        top_k: int
    ) -> Tuple[List[FrameData], List[float]]:
        """
        하이브리드 검색 (시각적 + 텍스트)

        Args:
            current_screen: 현재 화면
            query_text: 쿼리 텍스트
            top_k: 검색할 프레임 수

        Returns:
            (similar_frames, scores) 튜플
        """
        # 시각적 검색
        visual_results = self._visual_search(current_screen, top_k * 2)

        if query_text and self.text_embedder:
            # 텍스트 검색
            text_results = self._text_search(query_text, top_k * 2)

            # 하이브리드 랭킹
            combined_results = self._hybrid_ranking(
                visual_results, text_results, top_k
            )
        else:
            # 시각적 검색만
            combined_results = visual_results[:top_k]

        # FrameData 객체 로드
        frames = []
        scores = []

        for frame_id, score in combined_results:
            frame = self.db.get_frame(frame_id)
            if frame:
                frames.append(frame)
                scores.append(score)

        return frames, scores

    def _visual_search(
        self,
        current_screen: bytes,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        시각적 유사도 검색

        Args:
            current_screen: 현재 화면
            top_k: 검색할 프레임 수

        Returns:
            (frame_id, similarity_score) 리스트
        """
        if not self.frame_analyzer:
            print("[WARNING] FrameAnalyzer not available for visual search")
            return []

        # 현재 화면 임베딩 생성
        image = Image.open(io.BytesIO(current_screen))
        current_embedding = self.frame_analyzer.extract_features(np.array(image))

        # FAISS 검색
        results = self.db.search_similar_embeddings(current_embedding, top_k)

        return results

    def _text_search(
        self,
        query_text: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        텍스트 유사도 검색

        Args:
            query_text: 쿼리 텍스트
            top_k: 검색할 프레임 수

        Returns:
            (frame_id, similarity_score) 리스트
        """
        if not self.text_embedder:
            print("[WARNING] TextEmbedder not available for text search")
            return []

        # 쿼리 임베딩 생성
        query_embedding = self.text_embedder.embed_text(query_text)

        # FAISS 검색
        results = self.db.search_by_text(query_embedding, top_k)

        return results

    def _hybrid_ranking(
        self,
        visual_results: List[Tuple[str, float]],
        text_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        하이브리드 랭킹 (시각적 + 텍스트 점수 결합)

        Args:
            visual_results: 시각적 검색 결과
            text_results: 텍스트 검색 결과
            top_k: 반환할 결과 수

        Returns:
            (frame_id, combined_score) 리스트 (정렬됨)
        """
        # 점수 정규화
        visual_scores = {fid: score for fid, score in visual_results}
        text_scores = {fid: score for fid, score in text_results}

        # 모든 프레임 ID 수집
        all_frame_ids = set(visual_scores.keys()) | set(text_scores.keys())

        # 결합 점수 계산
        combined = []
        for frame_id in all_frame_ids:
            visual_score = visual_scores.get(frame_id, 0.0)
            text_score = text_scores.get(frame_id, 0.0)

            # 가중 평균
            combined_score = (
                self.visual_weight * visual_score +
                self.text_weight * text_score
            )

            combined.append((frame_id, combined_score))

        # 점수로 정렬
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:top_k]

    def _expand_temporal(
        self,
        similar_frames: List[FrameData],
        temporal_window: float
    ) -> List[FrameData]:
        """
        시간적 확장 (전후 프레임 추가)

        Args:
            similar_frames: 유사 프레임 리스트
            temporal_window: 확장 범위 (초)

        Returns:
            확장된 프레임 리스트
        """
        temporal_frames = []

        for frame in similar_frames:
            # 전후 프레임 조회
            video_frames = self.db.get_frames_by_video(frame.video_id)

            for vframe in video_frames:
                time_diff = abs(vframe.timestamp - frame.timestamp)
                if time_diff <= temporal_window and vframe.frame_id != frame.frame_id:
                    temporal_frames.append(vframe)

        return temporal_frames

    def _enrich_with_actions(
        self,
        frames: List[FrameData]
    ) -> List[ActionSequence]:
        """
        작업 시퀀스로 enrichment

        Args:
            frames: 프레임 리스트

        Returns:
            ActionSequence 리스트
        """
        actions = []

        for frame in frames:
            frame_actions = self.db.get_action_sequences_by_frame(frame.frame_id)
            actions.extend(frame_actions)

        # 중복 제거
        seen = set()
        unique_actions = []
        for action in actions:
            if action.action_id not in seen:
                seen.add(action.action_id)
                unique_actions.append(action)

        return unique_actions

    def _enrich_with_errors(
        self,
        frames: List[FrameData]
    ) -> List[ErrorPattern]:
        """
        에러 패턴으로 enrichment

        Args:
            frames: 프레임 리스트

        Returns:
            ErrorPattern 리스트
        """
        errors = []

        for frame in frames:
            frame_errors = self.db.get_error_patterns_by_frame(frame.frame_id)
            errors.extend(frame_errors)

        return errors

    def _enrich_with_ui_elements(
        self,
        frames: List[FrameData]
    ) -> List[UIElement]:
        """
        UI 요소로 enrichment

        Args:
            frames: 프레임 리스트

        Returns:
            UIElement 리스트
        """
        ui_elements = []

        for frame in frames:
            frame_ui = self.db.get_ui_elements_by_frame(frame.frame_id)
            ui_elements.extend(frame_ui)

        return ui_elements

    def _enrich_with_ocr(
        self,
        frames: List[FrameData]
    ) -> List[OCRResult]:
        """
        OCR 결과로 enrichment

        Args:
            frames: 프레임 리스트

        Returns:
            OCRResult 리스트
        """
        ocr_results = []

        for frame in frames:
            frame_ocr = self.db.get_ocr_results_by_frame(frame.frame_id)
            ocr_results.extend(frame_ocr)

        return ocr_results


def create_rag_context_manager(
    db_handler: DatabaseHandler,
    use_text_search: bool = True,
    use_visual_search: bool = True
) -> RAGContextManager:
    """
    RAGContextManager 생성 헬퍼 함수

    Args:
        db_handler: DatabaseHandler 인스턴스
        use_text_search: 텍스트 검색 사용 여부
        use_visual_search: 시각적 검색 사용 여부

    Returns:
        RAGContextManager 인스턴스
    """
    text_embedder = None
    frame_analyzer = None

    if use_text_search:
        from video_frame_parser.text_embedder import create_text_embedder
        text_embedder = create_text_embedder(use_gpu=False)

    if use_visual_search:
        from video_frame_parser.analyzer import create_frame_analyzer
        frame_analyzer = create_frame_analyzer(device="cpu")

    return RAGContextManager(
        db_handler=db_handler,
        text_embedder=text_embedder,
        frame_analyzer=frame_analyzer
    )


if __name__ == "__main__":
    # 사용 예시
    from video_frame_parser.db_handler import DatabaseHandler
    from video_frame_parser.config import DatabaseConfig

    # DatabaseHandler 초기화
    config = DatabaseConfig()
    db = DatabaseHandler(config)
    db.initialize()

    # RAGContextManager 생성
    rag_manager = create_rag_context_manager(db)

    # 테스트 (실제 이미지 필요)
    # current_screen = open("test_screen.png", "rb").read()
    # context = rag_manager.retrieve_context(
    #     current_screen=current_screen,
    #     query_text="RCS 로그인 화면",
    #     top_k=3
    # )
    # print(f"Found {len(context.similar_frames)} similar frames")

    print("[INFO] RAGContextManager ready")
