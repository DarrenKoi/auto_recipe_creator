"""
Video Frame Extractor

동영상 파일에서 프레임을 추출하는 모듈.
AVI, MP4, MOV 등 다양한 포맷 지원.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, List, Tuple
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass

from .config import ExtractorConfig
from .models import FrameData, VideoMetadata, FrameType

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    동영상에서 프레임을 추출하는 클래스.

    Features:
    - 프레임 간격 기반 추출
    - 키프레임 추출
    - 변화 감지 기반 추출
    - 배치 추출 지원
    """

    SUPPORTED_FORMATS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_video_path: Optional[Path] = None
        self._metadata: Optional[VideoMetadata] = None

    def open(self, video_path: str | Path) -> VideoMetadata:
        """
        동영상 파일을 열고 메타데이터를 반환합니다.

        Args:
            video_path: 동영상 파일 경로

        Returns:
            VideoMetadata: 동영상 메타데이터

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 포맷일 때
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {video_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # 기존 캡처 객체 해제
        self.close()

        # 새 동영상 열기
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self._current_video_path = video_path
        self._metadata = self._extract_metadata(video_path)

        logger.info(
            f"Opened video: {video_path.name}, "
            f"Duration: {self._metadata.duration:.2f}s, "
            f"FPS: {self._metadata.fps:.2f}, "
            f"Resolution: {self._metadata.width}x{self._metadata.height}"
        )

        return self._metadata

    def close(self):
        """동영상 파일을 닫습니다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._current_video_path = None
            self._metadata = None

    def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """동영상 메타데이터를 추출합니다."""
        cap = self._cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        duration = total_frames / fps if fps > 0 else 0

        # 코덱 정보
        codec = fourcc if fourcc.strip() else "unknown"

        # 파일 크기
        file_size = video_path.stat().st_size

        # 비디오 ID 생성 (파일 해시 기반)
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
        """파일 기반 고유 ID 생성"""
        stat = video_path.stat()
        id_string = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]

    def _generate_frame_id(self, video_id: str, frame_number: int) -> str:
        """프레임 고유 ID 생성"""
        return f"{video_id}_f{frame_number:08d}"

    def extract_frames(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[FrameData]:
        """
        프레임을 순차적으로 추출합니다.

        Args:
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초), None이면 끝까지
            max_frames: 최대 추출 프레임 수

        Yields:
            FrameData: 추출된 프레임 데이터
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video opened. Call open() first.")

        metadata = self._metadata
        config = self.config

        # 종료 시간 설정
        if end_time is None:
            end_time = metadata.duration

        # 시작 위치로 이동
        start_frame = int(start_time * metadata.fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 프레임 간격 계산
        if config.frame_interval:
            frame_step = int(config.frame_interval * metadata.fps)
        else:
            frame_step = 1

        frame_step = max(1, frame_step)

        extracted_count = 0
        prev_frame = None
        current_frame_num = start_frame

        while True:
            # 종료 조건 체크
            if max_frames and extracted_count >= max_frames:
                break

            current_time = current_frame_num / metadata.fps
            if current_time > end_time:
                break

            # 프레임 읽기
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = self._cap.read()

            if not ret:
                break

            # 변화 감지
            change_score = 0.0
            if prev_frame is not None:
                change_score = self._calculate_change_score(prev_frame, frame)

            # 프레임 타입 결정
            frame_type = self._determine_frame_type(
                frame, prev_frame, change_score, current_frame_num == start_frame
            )

            # 이미지 전처리
            processed_frame = self._preprocess_frame(frame)

            # FrameData 생성
            frame_data = FrameData(
                frame_id=self._generate_frame_id(metadata.video_id, current_frame_num),
                video_id=metadata.video_id,
                frame_number=current_frame_num,
                timestamp=current_time,
                image_data=processed_frame,
                frame_type=frame_type,
                is_keyframe=(current_frame_num % (metadata.fps * 5) < frame_step),  # 약 5초마다 키프레임
                change_score=change_score,
                width=processed_frame.shape[1],
                height=processed_frame.shape[0],
                channels=processed_frame.shape[2] if len(processed_frame.shape) > 2 else 1,
            )

            yield frame_data

            prev_frame = frame.copy()
            current_frame_num += frame_step
            extracted_count += 1

        logger.info(f"Extracted {extracted_count} frames from {metadata.file_name}")

    def extract_keyframes(self) -> Iterator[FrameData]:
        """
        키프레임만 추출합니다 (씬 변화 감지 기반).

        Yields:
            FrameData: 키프레임 데이터
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video opened. Call open() first.")

        metadata = self._metadata
        threshold = 30.0  # 씬 변화 임계값

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_frame = None
        prev_hist = None

        for frame_num in range(metadata.total_frames):
            ret, frame = self._cap.read()
            if not ret:
                break

            # 히스토그램 기반 씬 변화 감지
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            is_keyframe = False
            if prev_hist is None:
                is_keyframe = True
            else:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                if diff > threshold:
                    is_keyframe = True

            if is_keyframe:
                change_score = 1.0 if prev_frame is not None else 0.0
                processed_frame = self._preprocess_frame(frame)
                timestamp = frame_num / metadata.fps

                frame_data = FrameData(
                    frame_id=self._generate_frame_id(metadata.video_id, frame_num),
                    video_id=metadata.video_id,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    image_data=processed_frame,
                    frame_type=FrameType.KEYFRAME,
                    is_keyframe=True,
                    change_score=change_score,
                    width=processed_frame.shape[1],
                    height=processed_frame.shape[0],
                    channels=processed_frame.shape[2] if len(processed_frame.shape) > 2 else 1,
                )

                yield frame_data

            prev_frame = frame
            prev_hist = hist

    def extract_frame_at(self, timestamp: float) -> Optional[FrameData]:
        """
        특정 시간의 프레임을 추출합니다.

        Args:
            timestamp: 추출할 시간 (초)

        Returns:
            FrameData or None
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video opened. Call open() first.")

        metadata = self._metadata
        frame_num = int(timestamp * metadata.fps)

        if frame_num < 0 or frame_num >= metadata.total_frames:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()

        if not ret:
            return None

        processed_frame = self._preprocess_frame(frame)

        return FrameData(
            frame_id=self._generate_frame_id(metadata.video_id, frame_num),
            video_id=metadata.video_id,
            frame_number=frame_num,
            timestamp=timestamp,
            image_data=processed_frame,
            frame_type=FrameType.REGULAR,
            is_keyframe=False,
            change_score=0.0,
            width=processed_frame.shape[1],
            height=processed_frame.shape[0],
            channels=processed_frame.shape[2] if len(processed_frame.shape) > 2 else 1,
        )

    def extract_frames_batch(
        self,
        timestamps: List[float],
    ) -> List[FrameData]:
        """
        여러 시간대의 프레임을 배치로 추출합니다.

        Args:
            timestamps: 추출할 시간 목록 (초)

        Returns:
            추출된 FrameData 목록
        """
        results = []
        for ts in sorted(timestamps):
            frame_data = self.extract_frame_at(ts)
            if frame_data:
                results.append(frame_data)
        return results

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (리사이즈, 그레이스케일 등)"""
        config = self.config
        result = frame.copy()

        # 그레이스케일 변환
        if config.grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = np.expand_dims(result, axis=-1)

        # 리사이즈
        if config.resize_width and config.resize_height:
            result = cv2.resize(
                result,
                (config.resize_width, config.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        elif config.resize_width:
            aspect_ratio = result.shape[0] / result.shape[1]
            new_height = int(config.resize_width * aspect_ratio)
            result = cv2.resize(
                result, (config.resize_width, new_height), interpolation=cv2.INTER_AREA
            )
        elif config.resize_height:
            aspect_ratio = result.shape[1] / result.shape[0]
            new_width = int(config.resize_height * aspect_ratio)
            result = cv2.resize(
                result, (new_width, config.resize_height), interpolation=cv2.INTER_AREA
            )

        return result

    def _calculate_change_score(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> float:
        """두 프레임 간의 변화량 계산 (0~1)"""
        # 그레이스케일 변환
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 구조적 유사도 계산 (SSIM 대신 간단한 MSE 사용)
        diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.mean(diff) / 255.0

        return float(score)

    def _determine_frame_type(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        change_score: float,
        is_first: bool,
    ) -> FrameType:
        """프레임 타입 결정"""
        if is_first:
            return FrameType.KEYFRAME

        if change_score > 0.3:
            return FrameType.TRANSITION
        elif change_score < 0.01:
            return FrameType.STATIC
        else:
            return FrameType.REGULAR

    def save_frame(
        self, frame_data: FrameData, output_dir: str | Path, prefix: str = ""
    ) -> str:
        """
        프레임을 파일로 저장합니다.

        Args:
            frame_data: 저장할 프레임 데이터
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{prefix}{frame_data.frame_id}.{self.config.output_format}"
        output_path = output_dir / filename

        if frame_data.image_data is None:
            raise ValueError("Frame has no image data")

        if self.config.output_format.lower() in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
        elif self.config.output_format.lower() == "png":
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
    def metadata(self) -> Optional[VideoMetadata]:
        """현재 열린 동영상의 메타데이터"""
        return self._metadata

    @property
    def is_opened(self) -> bool:
        """동영상이 열려있는지 확인"""
        return self._cap is not None and self._cap.isOpened()
