"""
Action Extractor for CCTV Footage

CCTV 영상에서 전문가의 마우스/키보드 작업 시퀀스 추출
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np
import cv2

from .models import ActionSequence, FrameData


class ActionExtractor:
    """
    CCTV 영상에서 작업 시퀀스 추출

    마우스 커서 추적, 클릭 감지, 키보드 입력 추론을 통해
    전문가의 작업 패턴을 학습
    """

    def __init__(
        self,
        cursor_template: Optional[np.ndarray] = None,
        click_threshold: float = 0.05,
        min_action_interval: float = 0.1
    ):
        """
        Args:
            cursor_template: 마우스 커서 템플릿 (None이면 자동 감지)
            click_threshold: 클릭 감지 임계값 (색상 변화량)
            min_action_interval: 최소 작업 간격 (초)
        """
        self.cursor_template = cursor_template
        self.click_threshold = click_threshold
        self.min_action_interval = min_action_interval

        print("[INFO] ActionExtractor initialized")

    def extract_from_frames(
        self,
        frames: List[FrameData],
        video_id: str,
        manual_annotations: Optional[List[Dict[str, Any]]] = None
    ) -> List[ActionSequence]:
        """
        프레임 시퀀스에서 작업 추출

        Args:
            frames: FrameData 리스트 (시간순 정렬)
            video_id: 비디오 ID
            manual_annotations: 수동 어노테이션 (옵션)

        Returns:
            ActionSequence 리스트
        """
        if manual_annotations:
            # 수동 어노테이션 사용
            return self._extract_from_manual_annotations(
                frames, video_id, manual_annotations
            )
        else:
            # 자동 추출 (현재는 기본 구현)
            return self._extract_automatic(frames, video_id)

    def _extract_from_manual_annotations(
        self,
        frames: List[FrameData],
        video_id: str,
        annotations: List[Dict[str, Any]]
    ) -> List[ActionSequence]:
        """
        수동 어노테이션에서 작업 시퀀스 생성

        Args:
            frames: FrameData 리스트
            video_id: 비디오 ID
            annotations: 수동 어노테이션
                [{
                    "start_time": 10.5,
                    "end_time": 15.2,
                    "actions": [
                        {"type": "click", "x": 100, "y": 200, "timestamp": 10.5},
                        {"type": "type", "text": "admin", "timestamp": 11.0}
                    ],
                    "description": "RCS 로그인 수행",
                    "success": True
                }]

        Returns:
            ActionSequence 리스트
        """
        sequences = []

        for annotation in annotations:
            # 시작/종료 프레임 찾기
            start_frame = self._find_frame_at_time(frames, annotation["start_time"])
            end_frame = self._find_frame_at_time(frames, annotation["end_time"])

            if not start_frame or not end_frame:
                print(f"[WARNING] Could not find frames for annotation at {annotation['start_time']}")
                continue

            # ActionSequence 생성
            sequence = ActionSequence(
                action_id=self._generate_action_id(video_id),
                video_id=video_id,
                start_frame_id=start_frame.frame_id,
                end_frame_id=end_frame.frame_id,
                start_time=annotation["start_time"],
                end_time=annotation["end_time"],
                actions=annotation["actions"],
                description=annotation.get("description", ""),
                success=annotation.get("success", True),
                extracted_at=datetime.now(),
                extraction_method="manual",
                confidence=1.0,
                error_type=annotation.get("error_type"),
                recovery_action=annotation.get("recovery_action")
            )

            sequences.append(sequence)

        print(f"[INFO] Extracted {len(sequences)} action sequences from manual annotations")
        return sequences

    def _extract_automatic(
        self,
        frames: List[FrameData],
        video_id: str
    ) -> List[ActionSequence]:
        """
        자동으로 작업 시퀀스 추출 (기본 구현)

        Args:
            frames: FrameData 리스트
            video_id: 비디오 ID

        Returns:
            ActionSequence 리스트
        """
        print("[INFO] Automatic action extraction (basic implementation)")

        if len(frames) < 2:
            return []

        # 광학 흐름 기반 커서 추적
        cursor_positions = self._track_cursor_optical_flow(frames)

        # 클릭 감지
        click_events = self._detect_clicks(frames, cursor_positions)

        # 작업 시퀀스로 그룹화
        sequences = self._group_into_sequences(
            frames, video_id, cursor_positions, click_events
        )

        print(f"[INFO] Extracted {len(sequences)} action sequences automatically")
        return sequences

    def _track_cursor_optical_flow(
        self,
        frames: List[FrameData]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        광학 흐름 기반 커서 위치 추적

        Args:
            frames: FrameData 리스트

        Returns:
            각 프레임의 커서 위치 [(x, y), ...] (None이면 감지 실패)
        """
        # 기본 구현: 프레임 차이 기반 움직임 감지
        positions = [None] * len(frames)

        # TODO: 실제 커서 추적 구현
        # - 템플릿 매칭
        # - 광학 흐름 (Lucas-Kanade or Farneback)
        # - 색상 기반 감지 (흰색 커서)

        return positions

    def _detect_clicks(
        self,
        frames: List[FrameData],
        cursor_positions: List[Optional[Tuple[int, int]]]
    ) -> List[Dict[str, Any]]:
        """
        클릭 이벤트 감지

        Args:
            frames: FrameData 리스트
            cursor_positions: 커서 위치 리스트

        Returns:
            클릭 이벤트 리스트
                [{"frame_idx": 10, "x": 100, "y": 200, "timestamp": 10.5}, ...]
        """
        clicks = []

        # TODO: 실제 클릭 감지 구현
        # - 버튼 영역의 색상 변화 감지
        # - 커서 위치의 픽셀 변화 감지
        # - 프레임 간 차이 분석

        return clicks

    def _group_into_sequences(
        self,
        frames: List[FrameData],
        video_id: str,
        cursor_positions: List[Optional[Tuple[int, int]]],
        click_events: List[Dict[str, Any]]
    ) -> List[ActionSequence]:
        """
        작업들을 시퀀스로 그룹화

        Args:
            frames: FrameData 리스트
            video_id: 비디오 ID
            cursor_positions: 커서 위치
            click_events: 클릭 이벤트

        Returns:
            ActionSequence 리스트
        """
        # TODO: 작업 시퀀스 그룹화 로직
        # - 시간 간격 기반 그룹화
        # - 화면 변화 기반 그룹화
        # - 유사 작업 패턴 인식

        return []

    def _find_frame_at_time(
        self,
        frames: List[FrameData],
        timestamp: float
    ) -> Optional[FrameData]:
        """
        특정 시간의 프레임 찾기

        Args:
            frames: FrameData 리스트
            timestamp: 타임스탬프 (초)

        Returns:
            해당 시간의 프레임 (없으면 None)
        """
        # 이진 탐색으로 가장 가까운 프레임 찾기
        if not frames:
            return None

        closest_frame = None
        min_diff = float('inf')

        for frame in frames:
            diff = abs(frame.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame

        return closest_frame

    def _generate_action_id(self, video_id: str) -> str:
        """작업 시퀀스 고유 ID 생성"""
        return f"{video_id}_action_{uuid.uuid4().hex[:8]}"

    def export_annotation_template(
        self,
        frames: List[FrameData],
        output_path: str
    ):
        """
        수동 어노테이션 템플릿 생성

        Args:
            frames: FrameData 리스트
            output_path: 출력 파일 경로 (JSON)
        """
        import json

        template = {
            "video_id": frames[0].video_id if frames else "unknown",
            "total_frames": len(frames),
            "duration": frames[-1].timestamp if frames else 0.0,
            "annotations": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "actions": [
                        {
                            "type": "click",
                            "x": 100,
                            "y": 200,
                            "timestamp": 1.0,
                            "description": "Click server input field"
                        },
                        {
                            "type": "type",
                            "text": "192.168.1.100",
                            "timestamp": 2.0,
                            "description": "Type server address"
                        }
                    ],
                    "description": "RCS 서버 주소 입력",
                    "success": True,
                    "error_type": None,
                    "recovery_action": None
                }
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Annotation template saved to {output_path}")

    def load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """
        어노테이션 파일 로드

        Args:
            annotation_path: 어노테이션 파일 경로 (JSON)

        Returns:
            어노테이션 리스트
        """
        import json

        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("annotations", [])


def create_action_extractor() -> ActionExtractor:
    """
    ActionExtractor 생성 헬퍼 함수

    Returns:
        ActionExtractor 인스턴스
    """
    return ActionExtractor()


if __name__ == "__main__":
    # 사용 예시: 어노테이션 템플릿 생성
    import sys

    extractor = create_action_extractor()

    # 템플릿 생성
    output_path = "annotation_template.json"
    extractor.export_annotation_template([], output_path)
    print(f"[INFO] Created annotation template: {output_path}")

    print("\n[INFO] To annotate CCTV footage:")
    print("1. Edit the JSON file with actual action sequences")
    print("2. Use extract_from_frames() with manual_annotations parameter")
