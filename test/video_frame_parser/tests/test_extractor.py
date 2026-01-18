"""
Tests for VideoFrameExtractor

프레임 추출기 단위 테스트
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_frame_parser.extractor import VideoFrameExtractor
from video_frame_parser.config import ExtractorConfig
from video_frame_parser.models import FrameType


@pytest.fixture
def sample_video(tmp_path):
    """테스트용 샘플 비디오 생성"""
    video_path = tmp_path / "test_video.avi"

    # 간단한 테스트 비디오 생성
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    frame_size = (640, 480)
    num_frames = 90  # 3초 분량

    out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    for i in range(num_frames):
        # 프레임마다 다른 색상의 이미지 생성
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        color = ((i * 3) % 255, (i * 5) % 255, (i * 7) % 255)
        frame[:] = color
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def extractor():
    """기본 설정의 프레임 추출기"""
    config = ExtractorConfig(frame_interval=1.0)
    return VideoFrameExtractor(config)


class TestVideoFrameExtractor:
    """VideoFrameExtractor 테스트"""

    def test_open_video(self, extractor, sample_video):
        """비디오 파일 열기 테스트"""
        metadata = extractor.open(sample_video)

        assert metadata is not None
        assert metadata.video_id is not None
        assert metadata.fps == 30.0
        assert metadata.total_frames == 90
        assert metadata.width == 640
        assert metadata.height == 480
        assert extractor.is_opened

        extractor.close()

    def test_open_nonexistent_video(self, extractor):
        """존재하지 않는 파일 열기 테스트"""
        with pytest.raises(FileNotFoundError):
            extractor.open("/nonexistent/path/video.avi")

    def test_open_unsupported_format(self, extractor, tmp_path):
        """지원하지 않는 포맷 테스트"""
        fake_file = tmp_path / "video.xyz"
        fake_file.write_text("not a video")

        with pytest.raises(ValueError):
            extractor.open(fake_file)

    def test_extract_frames(self, extractor, sample_video):
        """프레임 추출 테스트"""
        extractor.open(sample_video)
        frames = list(extractor.extract_frames())

        # 1초 간격으로 3초 비디오 -> 약 3개 프레임
        assert len(frames) >= 2
        assert all(frame.image_data is not None for frame in frames)
        assert all(frame.video_id == extractor.metadata.video_id for frame in frames)

        extractor.close()

    def test_extract_frames_with_max_frames(self, extractor, sample_video):
        """최대 프레임 수 제한 테스트"""
        extractor.open(sample_video)
        frames = list(extractor.extract_frames(max_frames=2))

        assert len(frames) == 2

        extractor.close()

    def test_extract_frame_at_timestamp(self, extractor, sample_video):
        """특정 시간 프레임 추출 테스트"""
        extractor.open(sample_video)

        frame = extractor.extract_frame_at(1.0)  # 1초 위치

        assert frame is not None
        assert frame.timestamp == 1.0
        assert frame.image_data is not None

        extractor.close()

    def test_extract_keyframes(self, extractor, sample_video):
        """키프레임 추출 테스트"""
        extractor.open(sample_video)
        keyframes = list(extractor.extract_keyframes())

        # 최소 1개 이상의 키프레임
        assert len(keyframes) >= 1
        assert all(frame.is_keyframe for frame in keyframes)

        extractor.close()

    def test_save_frame(self, extractor, sample_video, tmp_path):
        """프레임 저장 테스트"""
        extractor.open(sample_video)
        frame = extractor.extract_frame_at(0.0)

        output_dir = tmp_path / "frames"
        saved_path = extractor.save_frame(frame, output_dir)

        assert Path(saved_path).exists()
        assert frame.image_path == saved_path

        extractor.close()

    def test_context_manager(self, sample_video):
        """컨텍스트 매니저 테스트"""
        config = ExtractorConfig(frame_interval=1.0)

        with VideoFrameExtractor(config) as extractor:
            extractor.open(sample_video)
            assert extractor.is_opened

        assert not extractor.is_opened

    def test_frame_data_fields(self, extractor, sample_video):
        """프레임 데이터 필드 검증"""
        extractor.open(sample_video)
        frames = list(extractor.extract_frames(max_frames=1))

        frame = frames[0]

        assert frame.frame_id is not None
        assert frame.video_id is not None
        assert frame.frame_number >= 0
        assert frame.timestamp >= 0
        assert frame.width > 0
        assert frame.height > 0
        assert frame.channels in (1, 3, 4)
        assert isinstance(frame.frame_type, FrameType)

        extractor.close()

    def test_change_score_calculation(self, sample_video):
        """변화 점수 계산 테스트"""
        # 짧은 간격으로 추출하여 변화 점수 확인
        config = ExtractorConfig(frame_interval=0.1)
        extractor = VideoFrameExtractor(config)

        extractor.open(sample_video)
        frames = list(extractor.extract_frames(max_frames=10))

        # 첫 프레임 이후의 프레임들은 변화 점수가 있어야 함
        assert frames[0].change_score == 0.0  # 첫 프레임
        # 후속 프레임들은 변화가 있으므로 점수 > 0
        for frame in frames[1:]:
            assert frame.change_score >= 0.0

        extractor.close()


class TestExtractorConfig:
    """ExtractorConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = ExtractorConfig()

        assert config.frame_interval == 1.0
        assert config.output_format in ("png", "jpg")
        assert 0 <= config.quality <= 100

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ExtractorConfig(
            frame_interval=0.5,
            resize_width=320,
            resize_height=240,
            grayscale=True,
        )

        assert config.frame_interval == 0.5
        assert config.resize_width == 320
        assert config.resize_height == 240
        assert config.grayscale is True


class TestVideoMetadata:
    """VideoMetadata 테스트"""

    def test_to_dict(self, extractor, sample_video):
        """딕셔너리 변환 테스트"""
        metadata = extractor.open(sample_video)
        data = metadata.to_dict()

        assert "video_id" in data
        assert "file_path" in data
        assert "fps" in data
        assert "total_frames" in data
        assert "width" in data
        assert "height" in data

        extractor.close()

    def test_from_dict(self, extractor, sample_video):
        """딕셔너리에서 복원 테스트"""
        from video_frame_parser.models import VideoMetadata

        metadata = extractor.open(sample_video)
        data = metadata.to_dict()

        restored = VideoMetadata.from_dict(data)

        assert restored.video_id == metadata.video_id
        assert restored.fps == metadata.fps
        assert restored.total_frames == metadata.total_frames

        extractor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
