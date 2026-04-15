"""workflow_1 전용 동영상 프레임 추출 유틸리티.

`test/video_frame_parser` 의 프레임 추출 핵심 로직만
workflow_1 내부에서 독립적으로 재사용할 수 있도록 옮긴 경량 버전이다.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """프레임 타입."""

    KEYFRAME = "keyframe"
    REGULAR = "regular"
    TRANSITION = "transition"
    STATIC = "static"


@dataclass
class ExtractorConfig:
    """프레임 추출기 설정."""

    frame_interval: float | None = 1.0
    keyframes_only: bool = False
    output_format: str = "png"
    quality: int = 95
    resize_width: int | None = None
    resize_height: int | None = None
    grayscale: bool = False


@dataclass
class VideoMetadata:
    """동영상 메타데이터."""

    video_id: str
    file_path: str
    file_name: str
    duration: float
    fps: float
    total_frames: int
    width: int
    height: int
    codec: str
    fourcc: str
    file_size: int
    created_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class FrameData:
    """추출된 프레임 데이터."""

    frame_id: str
    video_id: str
    frame_number: int
    timestamp: float
    image_path: str | None = None
    image_data: np.ndarray | None = None
    frame_type: FrameType = FrameType.REGULAR
    is_keyframe: bool = False
    change_score: float = 0.0
    width: int = 0
    height: int = 0
    channels: int = 3
    extracted_at: datetime = field(default_factory=datetime.now)


class VideoFrameExtractor:
    """동영상 파일에서 샘플 프레임을 추출한다."""

    SUPPORTED_FORMATS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(self, config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        self._cap: cv2.VideoCapture | None = None
        self._current_video_path: Path | None = None
        self._metadata: VideoMetadata | None = None

    def open(self, video_path: str | Path) -> VideoMetadata:
        """동영상을 열고 메타데이터를 반환한다."""
        resolved_path = Path(video_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Video file not found: {resolved_path}")
        if resolved_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {resolved_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        self.close()

        self._cap = cv2.VideoCapture(str(resolved_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {resolved_path}")

        self._current_video_path = resolved_path
        self._metadata = self._extract_metadata(resolved_path)
        logger.info(
            "Opened video: %s, duration=%.2fs, fps=%.2f, size=%sx%s",
            resolved_path.name,
            self._metadata.duration,
            self._metadata.fps,
            self._metadata.width,
            self._metadata.height,
        )
        return self._metadata

    def close(self) -> None:
        """열린 동영상을 닫는다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._current_video_path = None
            self._metadata = None

    def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """동영상 메타데이터를 읽는다."""
        if self._cap is None:
            raise RuntimeError("No active cv2.VideoCapture")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * idx)) & 0xFF) for idx in range(4))
        duration = total_frames / fps if fps > 0 else 0.0
        codec = fourcc if fourcc.strip() else "unknown"
        file_size = video_path.stat().st_size
        video_id = self._generate_video_id(video_path)

        return VideoMetadata(
            video_id=video_id,
            file_path=str(video_path.absolute()),
            file_name=video_path.name,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            codec=codec,
            fourcc=fourcc,
            file_size=file_size,
        )

    def _generate_video_id(self, video_path: Path) -> str:
        """파일 기반 고유 video id 를 만든다."""
        stat = video_path.stat()
        raw = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _generate_frame_id(self, video_id: str, frame_number: int) -> str:
        """frame id 를 만든다."""
        return f"{video_id}_f{frame_number:08d}"

    def extract_frames(
        self,
        start_time: float = 0.0,
        end_time: float | None = None,
        max_frames: int | None = None,
    ):
        """샘플 간격 기준으로 프레임을 순차 추출한다."""
        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video opened. Call open() first.")

        metadata = self._metadata
        if end_time is None:
            end_time = metadata.duration

        start_frame = int(start_time * metadata.fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_step = 1
        if self.config.frame_interval:
            frame_step = int(self.config.frame_interval * metadata.fps)
        frame_step = max(1, frame_step)

        extracted_count = 0
        prev_frame = None
        current_frame_num = start_frame

        while True:
            if max_frames and extracted_count >= max_frames:
                break

            current_time = current_frame_num / metadata.fps if metadata.fps > 0 else 0.0
            if current_time > end_time:
                break

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = self._cap.read()
            if not ret:
                break

            change_score = 0.0
            if prev_frame is not None:
                change_score = self._calculate_change_score(prev_frame, frame)

            frame_type = self._determine_frame_type(
                change_score=change_score,
                is_first=(current_frame_num == start_frame),
            )
            processed_frame = self._preprocess_frame(frame)

            yield FrameData(
                frame_id=self._generate_frame_id(metadata.video_id, current_frame_num),
                video_id=metadata.video_id,
                frame_number=current_frame_num,
                timestamp=current_time,
                image_data=processed_frame,
                frame_type=frame_type,
                is_keyframe=(current_frame_num % max(1, int(metadata.fps * 5)) < frame_step),
                change_score=change_score,
                width=processed_frame.shape[1],
                height=processed_frame.shape[0],
                channels=processed_frame.shape[2] if len(processed_frame.shape) > 2 else 1,
            )

            prev_frame = frame.copy()
            current_frame_num += frame_step
            extracted_count += 1

        logger.info("Extracted %s frames from %s", extracted_count, metadata.file_name)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """리사이즈/그레이스케일 등 최소 전처리를 수행한다."""
        result = frame.copy()

        if self.config.grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = np.expand_dims(result, axis=-1)

        if self.config.resize_width and self.config.resize_height:
            result = cv2.resize(
                result,
                (self.config.resize_width, self.config.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        elif self.config.resize_width:
            aspect_ratio = result.shape[0] / result.shape[1]
            new_height = int(self.config.resize_width * aspect_ratio)
            result = cv2.resize(
                result,
                (self.config.resize_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        elif self.config.resize_height:
            aspect_ratio = result.shape[1] / result.shape[0]
            new_width = int(self.config.resize_height * aspect_ratio)
            result = cv2.resize(
                result,
                (new_width, self.config.resize_height),
                interpolation=cv2.INTER_AREA,
            )

        return result

    def _calculate_change_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """두 프레임 간 평균 밝기 차이를 0~1 점수로 계산한다."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(np.mean(diff) / 255.0)

    def _determine_frame_type(self, *, change_score: float, is_first: bool) -> FrameType:
        """간단한 규칙으로 프레임 타입을 정한다."""
        if is_first:
            return FrameType.KEYFRAME
        if change_score > 0.3:
            return FrameType.TRANSITION
        if change_score < 0.01:
            return FrameType.STATIC
        return FrameType.REGULAR

    def save_frame(self, frame_data: FrameData, output_dir: str | Path, prefix: str = "") -> str:
        """프레임 이미지를 파일로 저장한다."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{prefix}{frame_data.frame_id}.{self.config.output_format}"
        output_path = out_dir / filename

        if frame_data.image_data is None:
            raise ValueError("Frame has no image data")

        format_name = self.config.output_format.lower()
        if format_name in {"jpg", "jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
        elif format_name == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (self.config.quality // 12)]
        else:
            params = []

        cv2.imwrite(str(output_path), frame_data.image_data, params)
        frame_data.image_path = str(output_path)
        return str(output_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def metadata(self) -> VideoMetadata | None:
        """현재 열린 동영상 메타데이터."""
        return self._metadata

    @property
    def is_opened(self) -> bool:
        """동영상 open 상태."""
        return self._cap is not None and self._cap.isOpened()


__all__ = [
    "ExtractorConfig",
    "FrameData",
    "FrameType",
    "VideoFrameExtractor",
    "VideoMetadata",
]
