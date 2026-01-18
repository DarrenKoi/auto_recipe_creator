"""
Tests for FrameAnalyzer

프레임 분석기 단위 테스트 (CLIP 임베딩 생성)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_frame_parser.analyzer import FrameAnalyzer
from video_frame_parser.config import AnalyzerConfig
from video_frame_parser.models import FrameData, AnalysisResult, AnalysisStatus, FrameType


@pytest.fixture
def mock_frame():
    """테스트용 목 프레임 데이터"""
    # 랜덤 이미지 데이터 생성 (BGR 형식)
    image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    return FrameData(
        frame_id="test_video_f00000001",
        video_id="test_video",
        frame_number=1,
        timestamp=0.033,
        image_data=image_data,
        frame_type=FrameType.REGULAR,
        width=640,
        height=480,
        channels=3,
    )


@pytest.fixture
def mock_frames():
    """테스트용 목 프레임 데이터 배열"""
    frames = []
    for i in range(5):
        image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(FrameData(
            frame_id=f"test_video_f{i:08d}",
            video_id="test_video",
            frame_number=i,
            timestamp=i * 0.033,
            image_data=image_data,
            frame_type=FrameType.REGULAR,
            width=640,
            height=480,
            channels=3,
        ))
    return frames


class TestFrameAnalyzerWithMock:
    """FrameAnalyzer 테스트 (CLIP 모킹)"""

    @patch('video_frame_parser.analyzer.CLIP_AVAILABLE', False)
    def test_initialize_without_clip(self):
        """CLIP 없이 초기화 시 에러"""
        analyzer = FrameAnalyzer()

        with pytest.raises(RuntimeError, match="CLIP is required"):
            analyzer.initialize()

    @patch('video_frame_parser.analyzer.TORCH_AVAILABLE', False)
    def test_initialize_without_torch(self):
        """PyTorch 없이 초기화 시 에러"""
        analyzer = FrameAnalyzer()

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            analyzer.initialize()


class TestFrameAnalyzerConfig:
    """AnalyzerConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = AnalyzerConfig()

        assert config.clip_model == "ViT-B/32"
        assert config.device in ("cuda", "cpu", "auto")
        assert config.batch_size > 0

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = AnalyzerConfig(
            clip_model="ViT-L/14",
            device="cpu",
            batch_size=16,
        )

        assert config.clip_model == "ViT-L/14"
        assert config.device == "cpu"
        assert config.batch_size == 16


class TestAnalysisResult:
    """AnalysisResult 테스트"""

    def test_create_result(self, mock_frame):
        """분석 결과 생성 테스트"""
        embedding = np.random.randn(512).astype(np.float32)

        result = AnalysisResult(
            result_id=f"{mock_frame.frame_id}_result",
            frame_id=mock_frame.frame_id,
            video_id=mock_frame.video_id,
            embedding=embedding,
            embedding_model="ViT-B/32",
            status=AnalysisStatus.COMPLETED,
            analyzed_at=datetime.now(),
            processing_time_ms=15.5,
        )

        assert result.result_id == f"{mock_frame.frame_id}_result"
        assert result.status == AnalysisStatus.COMPLETED
        assert result.embedding.shape == (512,)

    def test_to_dict(self, mock_frame):
        """딕셔너리 변환 테스트"""
        result = AnalysisResult(
            result_id=f"{mock_frame.frame_id}_result",
            frame_id=mock_frame.frame_id,
            video_id=mock_frame.video_id,
            status=AnalysisStatus.COMPLETED,
            analyzed_at=datetime.now(),
        )

        data = result.to_dict()

        assert "result_id" in data
        assert "frame_id" in data
        assert "status" in data
        assert data["status"] == "completed"

    def test_from_dict(self, mock_frame):
        """딕셔너리에서 복원 테스트"""
        original = AnalysisResult(
            result_id=f"{mock_frame.frame_id}_result",
            frame_id=mock_frame.frame_id,
            video_id=mock_frame.video_id,
            status=AnalysisStatus.COMPLETED,
            match_confidence=0.95,
            analyzed_at=datetime.now(),
        )

        data = original.to_dict()
        restored = AnalysisResult.from_dict(data)

        assert restored.result_id == original.result_id
        assert restored.status == original.status
        assert restored.match_confidence == original.match_confidence


class TestSimilarityFunctions:
    """유사도 계산 함수 테스트"""

    def test_compute_similarity(self):
        """코사인 유사도 계산 테스트"""
        analyzer = FrameAnalyzer()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])

        similarity = analyzer.compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-6

    def test_compute_similarity_orthogonal(self):
        """직교 벡터 유사도 테스트"""
        analyzer = FrameAnalyzer()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])

        similarity = analyzer.compute_similarity(emb1, emb2)
        assert abs(similarity) < 1e-6

    def test_find_similar_frames(self):
        """유사 프레임 검색 테스트"""
        analyzer = FrameAnalyzer()

        query = np.array([1.0, 0.0, 0.0])
        candidates = [
            np.array([1.0, 0.0, 0.0]),  # 가장 유사
            np.array([0.8, 0.6, 0.0]),  # 두 번째
            np.array([0.0, 1.0, 0.0]),  # 직교
        ]

        results = analyzer.find_similar_frames(query, candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # 첫 번째가 가장 유사
        assert results[0][1] > results[1][1]  # 유사도 내림차순

    def test_match_state(self):
        """상태 매칭 테스트"""
        analyzer = FrameAnalyzer()

        frame_embedding = np.array([1.0, 0.0, 0.0])
        state_embeddings = {
            "state_a": np.array([0.9, 0.1, 0.0]),
            "state_b": np.array([0.0, 1.0, 0.0]),
            "state_c": np.array([0.0, 0.0, 1.0]),
        }

        result = analyzer.match_state(frame_embedding, state_embeddings, threshold=0.8)

        assert result is not None
        assert result[0] == "state_a"
        assert result[1] > 0.8


class TestEmbeddingDimension:
    """임베딩 차원 테스트"""

    def test_embedding_dim_vit_b_32(self):
        """ViT-B/32 임베딩 차원"""
        config = AnalyzerConfig(clip_model="ViT-B/32")
        analyzer = FrameAnalyzer(config)

        assert analyzer.embedding_dim == 512

    def test_embedding_dim_vit_l_14(self):
        """ViT-L/14 임베딩 차원"""
        config = AnalyzerConfig(clip_model="ViT-L/14")
        analyzer = FrameAnalyzer(config)

        assert analyzer.embedding_dim == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
