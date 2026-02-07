"""
Workflow Annotator - 수동 어노테이션 CLI 도구

엔지니어가 화면 녹화에 구조화된 작업 단계를 어노테이션하는 CLI 도구.
각 타임스탬프에서 프레임을 추출하여 시각적 확인이 가능하며,
결과를 JSON으로 저장하여 검토/편집할 수 있습니다.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python이 설치되지 않았습니다. pip install opencv-python")

from .models import (
    WorkflowStep,
    WorkflowAnnotation,
    ActionType,
    RecipeType,
)


class WorkflowAnnotator:
    """
    화면 녹화 수동 어노테이션 도구.

    엔지니어가 영상을 보면서 각 단계를 입력하면
    타임스탬프 기반 프레임 추출 + JSON 저장을 수행합니다.
    """

    VALID_ACTIONS = [e.value for e in ActionType]
    VALID_RECIPE_TYPES = [e.value for e in RecipeType]

    def __init__(self, output_dir: str = "./workflow_annotations"):
        """
        Args:
            output_dir: 어노테이션 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[str] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration: float = 0.0
        self._width: int = 0
        self._height: int = 0

    def open_video(self, video_path: str) -> Dict[str, Any]:
        """
        영상 파일을 열고 메타데이터를 반환합니다.

        Args:
            video_path: 영상 파일 경로

        Returns:
            영상 메타데이터 딕셔너리

        Raises:
            FileNotFoundError: 파일이 없을 때
            RuntimeError: 영상을 열 수 없을 때
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("[ERROR] opencv-python이 필요합니다.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[ERROR] 영상 파일을 찾을 수 없습니다: {video_path}")

        self.close_video()
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"[ERROR] 영상을 열 수 없습니다: {video_path}")

        self._video_path = video_path
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0

        metadata = {
            "video_path": video_path,
            "fps": self._fps,
            "total_frames": self._total_frames,
            "width": self._width,
            "height": self._height,
            "duration": self._duration,
        }
        print(f"[INFO] 영상 열기 완료: {Path(video_path).name}")
        print(f"[INFO] 해상도: {self._width}x{self._height}, "
              f"FPS: {self._fps:.2f}, 길이: {self._duration:.2f}초")
        return metadata

    def close_video(self):
        """영상 파일을 닫습니다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def extract_frame_at(self, timestamp: float) -> Optional[str]:
        """
        특정 타임스탬프의 프레임을 추출하여 파일로 저장합니다.

        Args:
            timestamp: 추출할 시간 (초)

        Returns:
            저장된 파일 경로 또는 None
        """
        if self._cap is None:
            print("[ERROR] 영상이 열려있지 않습니다.")
            return None

        frame_num = int(timestamp * self._fps)
        if frame_num < 0 or frame_num >= self._total_frames:
            print(f"[ERROR] 유효하지 않은 타임스탬프: {timestamp}초 (범위: 0~{self._duration:.2f}초)")
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()

        if not ret:
            print(f"[ERROR] 프레임 읽기 실패: frame {frame_num}")
            return None

        # 프레임 저장
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{frame_num:08d}_t{timestamp:.2f}s.png"
        output_path = frames_dir / filename
        cv2.imwrite(str(output_path), frame)

        return str(output_path)

    def _generate_workflow_id(self) -> str:
        """워크플로우 고유 ID 생성"""
        timestamp = datetime.now().isoformat()
        video_name = Path(self._video_path).name if self._video_path else "unknown"
        id_str = f"{video_name}_{timestamp}"
        return hashlib.md5(id_str.encode()).hexdigest()[:12]

    def annotate_interactive(self, video_path: str) -> Optional[WorkflowAnnotation]:
        """
        대화형 CLI로 워크플로우를 어노테이션합니다.

        Args:
            video_path: 영상 파일 경로

        Returns:
            WorkflowAnnotation 또는 None (취소 시)
        """
        self.open_video(video_path)

        print("\n=== 워크플로우 어노테이션 시작 ===")
        print(f"영상: {Path(video_path).name}")
        print(f"길이: {self._duration:.2f}초 ({self._total_frames} 프레임)")

        # 기본 정보 입력
        print(f"\n레시피 타입 옵션: {', '.join(self.VALID_RECIPE_TYPES)}")
        recipe_type = input("레시피 타입: ").strip()
        if recipe_type not in self.VALID_RECIPE_TYPES:
            print(f"[WARNING] 알 수 없는 타입 '{recipe_type}', OTHER로 설정합니다.")
            recipe_type = RecipeType.OTHER.value

        description = input("워크플로우 설명: ").strip()
        annotated_by = input("작성자: ").strip()
        tags_input = input("태그 (쉼표 구분): ").strip()
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

        # 단계별 어노테이션
        steps: List[WorkflowStep] = []
        step_number = 1

        print(f"\n--- 단계 입력 (완료하려면 빈 타임스탬프 입력) ---")
        print(f"작업 타입 옵션: {', '.join(self.VALID_ACTIONS)}")

        while True:
            print(f"\n[단계 {step_number}]")
            timestamp_str = input(f"  타임스탬프 (초, 범위 0~{self._duration:.2f}): ").strip()

            if not timestamp_str:
                break

            try:
                timestamp = float(timestamp_str)
            except ValueError:
                print("[ERROR] 숫자를 입력하세요.")
                continue

            # 해당 프레임 추출 및 표시
            frame_path = self.extract_frame_at(timestamp)
            if frame_path:
                print(f"  프레임 저장됨: {frame_path}")

            frame_num = int(timestamp * self._fps)

            action_type = input(f"  작업 타입 ({'/'.join(self.VALID_ACTIONS)}): ").strip()
            if action_type not in self.VALID_ACTIONS:
                print(f"[WARNING] 알 수 없는 작업 타입 '{action_type}', click으로 설정합니다.")
                action_type = ActionType.CLICK.value

            target_desc = input("  대상 설명 (예: Login 버튼): ").strip()

            # 좌표 입력 (선택)
            coords_str = input("  좌표 (x,y / 없으면 Enter): ").strip()
            coordinates = None
            if coords_str:
                try:
                    parts = coords_str.split(",")
                    coordinates = (int(parts[0].strip()), int(parts[1].strip()))
                except (ValueError, IndexError):
                    print("[WARNING] 좌표 형식 오류, 무시합니다.")

            # 입력 텍스트 (type 작업 시)
            input_text = None
            if action_type in (ActionType.TYPE.value, ActionType.HOTKEY.value):
                input_text = input("  입력 텍스트: ").strip() or None

            notes = input("  메모 (없으면 Enter): ").strip() or None

            step = WorkflowStep(
                step_number=step_number,
                action_type=action_type,
                target_description=target_desc,
                timestamp=timestamp,
                screenshot_frame=frame_num,
                coordinates=coordinates,
                input_text=input_text,
                notes=notes,
            )
            steps.append(step)
            print(f"  [INFO] 단계 {step_number} 추가 완료: {action_type} → {target_desc}")
            step_number += 1

        if not steps:
            print("[WARNING] 입력된 단계가 없습니다. 취소합니다.")
            self.close_video()
            return None

        # 성공 여부
        success_str = input("\n워크플로우 성공 여부 (y/n): ").strip().lower()
        success = success_str in ("y", "yes", "1", "true")

        # WorkflowAnnotation 생성
        workflow = WorkflowAnnotation(
            workflow_id=self._generate_workflow_id(),
            video_path=video_path,
            recipe_type=recipe_type,
            description=description,
            steps=steps,
            total_duration=self._duration,
            success=success,
            annotated_by=annotated_by,
            extraction_method="manual",
            video_resolution=(self._width, self._height),
            tags=tags,
        )

        # JSON 저장
        output_path = self.save_annotation(workflow)
        print(f"\n[INFO] 어노테이션 저장 완료: {output_path}")
        print(f"[INFO] 총 {len(steps)}개 단계, 워크플로우 ID: {workflow.workflow_id}")

        self.close_video()
        return workflow

    def annotate_from_dict(
        self,
        video_path: str,
        annotation_data: Dict[str, Any],
    ) -> WorkflowAnnotation:
        """
        딕셔너리 데이터로 어노테이션을 생성합니다 (프로그래밍 방식).

        Args:
            video_path: 영상 파일 경로
            annotation_data: 어노테이션 데이터
                - recipe_type: str
                - description: str
                - annotated_by: str
                - success: bool
                - tags: List[str] (선택)
                - steps: List[Dict] (각 단계 데이터)

        Returns:
            WorkflowAnnotation
        """
        self.open_video(video_path)

        steps = []
        for i, step_data in enumerate(annotation_data.get("steps", []), start=1):
            timestamp = step_data["timestamp"]
            frame_num = int(timestamp * self._fps)

            # 프레임 추출
            self.extract_frame_at(timestamp)

            step = WorkflowStep(
                step_number=i,
                action_type=step_data.get("action_type", ActionType.CLICK.value),
                target_description=step_data.get("target_description", ""),
                timestamp=timestamp,
                screenshot_frame=frame_num,
                coordinates=tuple(step_data["coordinates"]) if step_data.get("coordinates") else None,
                input_text=step_data.get("input_text"),
                notes=step_data.get("notes"),
                confidence=step_data.get("confidence", 1.0),
                duration=step_data.get("duration"),
            )
            steps.append(step)

        workflow = WorkflowAnnotation(
            workflow_id=self._generate_workflow_id(),
            video_path=video_path,
            recipe_type=annotation_data.get("recipe_type", RecipeType.OTHER.value),
            description=annotation_data.get("description", ""),
            steps=steps,
            total_duration=self._duration,
            success=annotation_data.get("success", True),
            annotated_by=annotation_data.get("annotated_by", "system"),
            extraction_method=annotation_data.get("extraction_method", "manual"),
            video_resolution=(self._width, self._height),
            tags=annotation_data.get("tags", []),
        )

        output_path = self.save_annotation(workflow)
        print(f"[INFO] 어노테이션 저장 완료: {output_path}")

        self.close_video()
        return workflow

    def save_annotation(self, annotation: WorkflowAnnotation) -> str:
        """
        어노테이션을 JSON 파일로 저장합니다.

        Args:
            annotation: 저장할 어노테이션

        Returns:
            저장된 파일 경로
        """
        filename = f"workflow_{annotation.workflow_id}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)

        return str(output_path)

    def load_annotation(self, filepath: str) -> WorkflowAnnotation:
        """
        JSON 파일에서 어노테이션을 로드합니다.

        Args:
            filepath: JSON 파일 경로

        Returns:
            WorkflowAnnotation
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WorkflowAnnotation.from_dict(data)

    def list_annotations(self) -> List[Dict[str, Any]]:
        """
        저장된 어노테이션 목록을 반환합니다.

        Returns:
            어노테이션 요약 리스트
        """
        annotations = []
        for path in sorted(self.output_dir.glob("workflow_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                annotations.append({
                    "file": str(path),
                    "workflow_id": data.get("workflow_id"),
                    "recipe_type": data.get("recipe_type"),
                    "description": data.get("description"),
                    "steps_count": len(data.get("steps", [])),
                    "success": data.get("success"),
                    "annotated_by": data.get("annotated_by"),
                    "created_at": data.get("created_at"),
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARNING] 파일 읽기 실패: {path} ({e})")
        return annotations

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_video()
        return False


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(
        description="워크플로우 어노테이션 도구"
    )
    subparsers = parser.add_subparsers(dest="command")

    # annotate 서브커맨드
    annotate_parser = subparsers.add_parser("annotate", help="영상 어노테이션")
    annotate_parser.add_argument("video", help="영상 파일 경로")
    annotate_parser.add_argument(
        "--output-dir", default="./workflow_annotations",
        help="결과 저장 디렉토리 (기본: ./workflow_annotations)"
    )

    # list 서브커맨드
    list_parser = subparsers.add_parser("list", help="저장된 어노테이션 목록")
    list_parser.add_argument(
        "--output-dir", default="./workflow_annotations",
        help="어노테이션 디렉토리 (기본: ./workflow_annotations)"
    )

    # view 서브커맨드
    view_parser = subparsers.add_parser("view", help="어노테이션 상세 보기")
    view_parser.add_argument("file", help="어노테이션 JSON 파일 경로")

    args = parser.parse_args()

    if args.command == "annotate":
        annotator = WorkflowAnnotator(output_dir=args.output_dir)
        annotator.annotate_interactive(args.video)

    elif args.command == "list":
        annotator = WorkflowAnnotator(output_dir=args.output_dir)
        annotations = annotator.list_annotations()
        if not annotations:
            print("[INFO] 저장된 어노테이션이 없습니다.")
        else:
            print(f"\n총 {len(annotations)}개 어노테이션:")
            for ann in annotations:
                status = "성공" if ann["success"] else "실패"
                print(f"  [{ann['workflow_id']}] {ann['recipe_type']} - "
                      f"{ann['description']} ({ann['steps_count']}단계, {status})")

    elif args.command == "view":
        annotator = WorkflowAnnotator()
        annotation = annotator.load_annotation(args.file)
        print(f"\n=== 워크플로우: {annotation.workflow_id} ===")
        print(f"타입: {annotation.recipe_type}")
        print(f"설명: {annotation.description}")
        print(f"작성자: {annotation.annotated_by}")
        print(f"성공: {'예' if annotation.success else '아니오'}")
        print(f"단계: {len(annotation.steps)}개")
        print(f"영상: {annotation.video_path}")
        print()
        for step in annotation.steps:
            coords = f" @ ({step.coordinates[0]}, {step.coordinates[1]})" if step.coordinates else ""
            text = f" [{step.input_text}]" if step.input_text else ""
            print(f"  {step.step_number}. [{step.timestamp:.2f}s] "
                  f"{step.action_type}: {step.target_description}{coords}{text}")
            if step.notes:
                print(f"     메모: {step.notes}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
