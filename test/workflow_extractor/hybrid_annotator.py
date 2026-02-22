"""
Hybrid Annotator - VLM 자동 추출 + 사람 리뷰 하이브리드 어노테이션

VLM이 자동 추출한 워크플로우를 사람이 리뷰/수정하는 대화형 CLI 도구.
신뢰도 등급에 따라 자동 승인/리뷰 필요/수동 입력 필요로 분류합니다.

파이프라인:
1. AutoExtractor로 자동 추출 (또는 기존 JSON 로드)
2. 각 단계를 신뢰도 등급별로 분류
3. 대화형 CLI로 리뷰/수정
4. 리뷰 완료된 어노테이션 저장
"""

import json
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from .models import (
    WorkflowAnnotation,
    WorkflowStep,
    ExtractionMethod,
    ReviewStatus,
    ConfidenceTier,
    compute_confidence_tier,
)
from .auto_extractor import AutoExtractor, AutoExtractorConfig


@dataclass
class HybridAnnotatorConfig:
    """하이브리드 어노테이터 설정"""

    # 신뢰도 등급 임계값
    high_confidence_threshold: float = 0.7  # 이상이면 자동 승인
    low_confidence_threshold: float = 0.3  # 미만이면 수동 입력 필요

    # 자동 승인된 단계도 표시할지 여부
    show_auto_accepted: bool = True

    # 출력 디렉토리
    output_dir: str = "./hybrid_annotations"

    # 리뷰어 이름
    reviewer_name: str = ""


class HybridAnnotator:
    """
    VLM 자동 추출 결과를 사람이 리뷰하는 하이브리드 어노테이터.

    신뢰도 등급별 처리:
    - HIGH (>= 0.7): 자동 승인, 선택적 오버라이드 가능
    - MEDIUM (0.3 ~ 0.7): 리뷰 필수 (승인/수정/삭제)
    - LOW (< 0.3): 수동 입력 필요 플래그
    """

    def __init__(self, config: Optional[HybridAnnotatorConfig] = None):
        """
        Args:
            config: 하이브리드 어노테이터 설정
        """
        self.config = config or HybridAnnotatorConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_review(
        self,
        video_path: str,
        reviewer: str = "",
        recipe_type: str = "OTHER",
        description: str = "",
        extractor_config: Optional[AutoExtractorConfig] = None,
    ) -> Optional[WorkflowAnnotation]:
        """
        자동 추출 후 리뷰를 수행합니다.

        Args:
            video_path: 영상 파일 경로
            reviewer: 리뷰어 이름
            recipe_type: 레시피 타입
            description: 워크플로우 설명
            extractor_config: AutoExtractor 설정 (없으면 기본값)

        Returns:
            리뷰 완료된 WorkflowAnnotation 또는 None
        """
        reviewer = reviewer or self.config.reviewer_name

        # AutoExtractor 설정: 낮은 신뢰도 결과도 보존
        if extractor_config is None:
            extractor_config = AutoExtractorConfig()
        extractor_config.discard_low_confidence = False

        print(f"[INFO] 하이브리드 어노테이션 시작: {Path(video_path).name}")

        # 1. 자동 추출
        extractor = AutoExtractor(extractor_config)
        annotation = extractor.extract_workflow(
            video_path=video_path,
            recipe_type=recipe_type,
            description=description,
        )

        if annotation is None:
            print("[ERROR] 자동 추출 실패")
            return None

        # 2. 리뷰
        reviewed = self.review_annotation(annotation, reviewer)
        return reviewed

    def review_annotation(
        self,
        annotation: WorkflowAnnotation,
        reviewer: str = "",
    ) -> WorkflowAnnotation:
        """
        어노테이션의 각 단계를 리뷰합니다.

        Args:
            annotation: 리뷰할 어노테이션
            reviewer: 리뷰어 이름

        Returns:
            리뷰 완료된 WorkflowAnnotation
        """
        reviewer = reviewer or self.config.reviewer_name or "anonymous"

        # 원본 ID 보존
        source_id = annotation.workflow_id

        print(f"\n{'='*60}")
        print(f"  하이브리드 리뷰 시작")
        print(f"  워크플로우: {annotation.workflow_id}")
        print(f"  영상: {Path(annotation.video_path).name}")
        print(f"  총 단계: {len(annotation.steps)}개")
        print(f"  리뷰어: {reviewer}")
        print(f"{'='*60}\n")

        reviewed_steps: List[WorkflowStep] = []
        total = len(annotation.steps)

        for idx, step in enumerate(annotation.steps):
            tier = compute_confidence_tier(step.confidence)
            step.confidence_tier = tier

            result = self._review_step_interactive(step, idx, total)

            if result is None:
                # 삭제됨
                continue
            elif isinstance(result, list):
                # 단계 추가됨 (기존 + 새로운 단계)
                reviewed_steps.extend(result)
            else:
                reviewed_steps.append(result)

        # 단계 번호 재정렬
        for i, step in enumerate(reviewed_steps, start=1):
            step.step_number = i

        # 어노테이션 업데이트
        annotation.steps = reviewed_steps
        annotation.extraction_method = ExtractionMethod.HYBRID.value
        annotation.review_status = ReviewStatus.HUMAN_APPROVED.value
        annotation.reviewed_by = reviewer
        annotation.reviewed_at = datetime.now().isoformat()
        annotation.source_annotation_id = source_id
        annotation.review_stats = self.get_review_stats(annotation)

        # 저장
        output_path = self._save_annotation(annotation)
        print(f"\n[INFO] 리뷰 완료: {output_path}")
        self._print_review_summary(annotation)

        return annotation

    def review_from_json(
        self,
        json_path: str,
        reviewer: str = "",
    ) -> Optional[WorkflowAnnotation]:
        """
        JSON 파일에서 어노테이션을 로드하여 리뷰합니다.

        Args:
            json_path: 어노테이션 JSON 파일 경로
            reviewer: 리뷰어 이름

        Returns:
            리뷰 완료된 WorkflowAnnotation 또는 None
        """
        print(f"[INFO] JSON 로드: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            annotation = WorkflowAnnotation.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(f"[ERROR] JSON 로드 실패: {e}")
            return None

        return self.review_annotation(annotation, reviewer)

    def _review_step_interactive(
        self,
        step: WorkflowStep,
        idx: int,
        total: int,
    ) -> Optional[Any]:
        """
        단일 단계에 대한 대화형 리뷰를 수행합니다.

        Args:
            step: 리뷰할 단계
            idx: 현재 인덱스 (0-based)
            total: 전체 단계 수

        Returns:
            - WorkflowStep: 승인/수정된 단계
            - List[WorkflowStep]: 기존 단계 + 추가된 단계
            - None: 삭제됨
        """
        tier = step.confidence_tier

        # 자동 승인 처리 (HIGH 등급)
        if tier == ConfidenceTier.HIGH.value:
            if self.config.show_auto_accepted:
                self._display_step(step, idx, total, tier)
                print("  → 자동 승인 (HIGH 신뢰도)")
                print()

            step.review_status = ReviewStatus.AUTO_ACCEPTED.value
            return step

        # 리뷰 필요 (MEDIUM/LOW 등급)
        self._display_step(step, idx, total, tier)

        if tier == ConfidenceTier.LOW.value:
            print("  ⚠ 낮은 신뢰도 — 수동 확인 필요")
            step.review_status = ReviewStatus.NEEDS_REVIEW.value

        while True:
            choice = input(
                "  [a]승인  [c]수정  [d]삭제  [n]단계 추가  [Enter=승인]: "
            ).strip().lower()

            if choice in ("", "a"):
                step.review_status = ReviewStatus.HUMAN_APPROVED.value
                step.reviewed_by = self.config.reviewer_name or "anonymous"
                step.reviewed_at = datetime.now().isoformat()
                return step

            elif choice == "c":
                self._snapshot_vlm_suggestion(step)
                corrected = self._prompt_correction(step)
                corrected.review_status = ReviewStatus.HUMAN_CORRECTED.value
                corrected.reviewed_by = self.config.reviewer_name or "anonymous"
                corrected.reviewed_at = datetime.now().isoformat()
                return corrected

            elif choice == "d":
                confirm = input("  삭제 확인 (y/N): ").strip().lower()
                if confirm == "y":
                    print(f"  → 단계 {idx + 1} 삭제됨")
                    return None
                continue

            elif choice == "n":
                new_step = self._prompt_new_step(step.timestamp)
                if new_step:
                    step.review_status = ReviewStatus.HUMAN_APPROVED.value
                    step.reviewed_by = self.config.reviewer_name or "anonymous"
                    step.reviewed_at = datetime.now().isoformat()
                    return [step, new_step]
                continue

            else:
                print("  잘못된 입력입니다. a/c/d/n 중 선택하세요.")

    def _prompt_correction(self, step: WorkflowStep) -> WorkflowStep:
        """단계 필드별 수정을 받습니다."""
        print("\n  --- 수정 모드 (Enter로 기존 값 유지) ---")

        action_type = input(
            f"  작업 타입 [{step.action_type}]: "
        ).strip()
        if action_type:
            step.action_type = action_type

        target = input(
            f"  대상 [{step.target_description}]: "
        ).strip()
        if target:
            step.target_description = target

        coords_str = input(
            f"  좌표 [{step.coordinates}]: "
        ).strip()
        if coords_str:
            try:
                parts = coords_str.replace("(", "").replace(")", "").split(",")
                step.coordinates = (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                print("  [WARNING] 좌표 형식 오류 — 기존 값 유지")

        input_text = input(
            f"  입력 텍스트 [{step.input_text or '없음'}]: "
        ).strip()
        if input_text:
            step.input_text = input_text if input_text != "-" else None

        notes = input(
            f"  설명 [{step.notes or '없음'}]: "
        ).strip()
        if notes:
            step.notes = notes if notes != "-" else None

        confidence_str = input(
            f"  신뢰도 [{step.confidence}]: "
        ).strip()
        if confidence_str:
            try:
                step.confidence = float(confidence_str)
                step.confidence_tier = compute_confidence_tier(step.confidence)
            except ValueError:
                print("  [WARNING] 신뢰도 형식 오류 — 기존 값 유지")

        print("  → 수정 완료\n")
        return step

    def _prompt_new_step(self, after_timestamp: float) -> Optional[WorkflowStep]:
        """새 단계를 수동 입력받습니다."""
        print("\n  --- 새 단계 추가 ---")

        action_type = input("  작업 타입 (click/type/select/...): ").strip()
        if not action_type:
            print("  → 추가 취소")
            return None

        target = input("  대상 설명: ").strip()
        if not target:
            print("  → 추가 취소")
            return None

        timestamp_str = input(
            f"  시간 (초) [{after_timestamp + 0.5:.2f}]: "
        ).strip()
        try:
            timestamp = float(timestamp_str) if timestamp_str else after_timestamp + 0.5
        except ValueError:
            timestamp = after_timestamp + 0.5

        coords_str = input("  좌표 (x,y) [없음]: ").strip()
        coordinates = None
        if coords_str:
            try:
                parts = coords_str.replace("(", "").replace(")", "").split(",")
                coordinates = (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                pass

        input_text = input("  입력 텍스트 [없음]: ").strip() or None
        notes = input("  설명 [없음]: ").strip() or None

        new_step = WorkflowStep(
            step_number=0,  # 이후 재정렬됨
            action_type=action_type,
            target_description=target,
            timestamp=timestamp,
            screenshot_frame=0,
            coordinates=coordinates,
            input_text=input_text,
            notes=notes,
            confidence=1.0,
            confidence_tier=ConfidenceTier.HIGH.value,
            review_status=ReviewStatus.HUMAN_APPROVED.value,
            reviewed_by=self.config.reviewer_name or "anonymous",
            reviewed_at=datetime.now().isoformat(),
        )
        print("  → 단계 추가 완료\n")
        return new_step

    def _display_step(
        self,
        step: WorkflowStep,
        idx: int,
        total: int,
        tier: str,
    ) -> None:
        """단계를 CLI에 표시합니다."""
        tier_labels = {
            ConfidenceTier.HIGH.value: "HIGH - 자동 승인",
            ConfidenceTier.MEDIUM.value: "MEDIUM - 리뷰 필요",
            ConfidenceTier.LOW.value: "LOW - 수동 확인 필요",
        }
        tier_label = tier_labels.get(tier, tier)

        print(f"=== 단계 {idx + 1}/{total} [{tier_label}] ===")
        print(f"  시간: {step.timestamp:.2f}초 (프레임 #{step.screenshot_frame})")
        print(f"  작업: {step.action_type} → \"{step.target_description}\" "
              f"(신뢰도: {step.confidence:.2f})")
        if step.coordinates:
            print(f"  좌표: ({step.coordinates[0]}, {step.coordinates[1]})")
        if step.input_text:
            print(f"  입력: \"{step.input_text}\"")
        if step.notes:
            print(f"  설명: {step.notes}")

    def _snapshot_vlm_suggestion(self, step: WorkflowStep) -> None:
        """수정 전 VLM 원본 제안을 보존합니다."""
        if step.original_vlm_suggestion is not None:
            return  # 이미 보존됨

        step.original_vlm_suggestion = {
            "action_type": step.action_type,
            "target_description": step.target_description,
            "coordinates": list(step.coordinates) if step.coordinates else None,
            "input_text": step.input_text,
            "notes": step.notes,
            "confidence": step.confidence,
            "confidence_tier": step.confidence_tier,
        }

    def print_diff(self, annotation: WorkflowAnnotation) -> None:
        """VLM 제안 vs 사람 수정 비교를 출력합니다."""
        print(f"\n{'='*60}")
        print(f"  VLM 제안 vs 사람 수정 비교")
        print(f"  워크플로우: {annotation.workflow_id}")
        print(f"{'='*60}\n")

        has_diff = False
        for step in annotation.steps:
            if step.original_vlm_suggestion is None:
                continue
            if step.review_status != ReviewStatus.HUMAN_CORRECTED.value:
                continue

            has_diff = True
            orig = step.original_vlm_suggestion
            print(f"--- 단계 {step.step_number} ---")

            fields = [
                ("작업 타입", "action_type", step.action_type),
                ("대상", "target_description", step.target_description),
                ("입력", "input_text", step.input_text),
                ("설명", "notes", step.notes),
                ("신뢰도", "confidence", step.confidence),
            ]

            for label, key, current in fields:
                original = orig.get(key)
                if original != current:
                    print(f"  {label}: \"{original}\" → \"{current}\"")

            # 좌표 비교
            orig_coords = orig.get("coordinates")
            curr_coords = list(step.coordinates) if step.coordinates else None
            if orig_coords != curr_coords:
                print(f"  좌표: {orig_coords} → {curr_coords}")

            print()

        if not has_diff:
            print("  수정된 단계가 없습니다.\n")

    def get_review_stats(self, annotation: WorkflowAnnotation) -> Dict[str, int]:
        """리뷰 상태별 통계를 반환합니다."""
        stats: Dict[str, int] = {}
        for step in annotation.steps:
            status = step.review_status
            stats[status] = stats.get(status, 0) + 1
        return stats

    def _print_review_summary(self, annotation: WorkflowAnnotation) -> None:
        """리뷰 요약을 출력합니다."""
        stats = annotation.review_stats or self.get_review_stats(annotation)

        print(f"\n{'='*60}")
        print(f"  리뷰 요약")
        print(f"{'='*60}")
        print(f"  총 단계: {len(annotation.steps)}개")

        status_labels = {
            ReviewStatus.AUTO_ACCEPTED.value: "자동 승인",
            ReviewStatus.HUMAN_APPROVED.value: "사람 승인",
            ReviewStatus.HUMAN_CORRECTED.value: "사람 수정",
            ReviewStatus.REJECTED.value: "거절",
            ReviewStatus.NEEDS_REVIEW.value: "추가 리뷰 필요",
            ReviewStatus.PENDING.value: "대기",
        }

        for status, count in sorted(stats.items()):
            label = status_labels.get(status, status)
            print(f"  {label}: {count}개")
        print()

    def _save_annotation(self, annotation: WorkflowAnnotation) -> str:
        """리뷰된 어노테이션을 JSON으로 저장합니다."""
        filename = f"hybrid_{annotation.workflow_id}.json"
        output_path = self._output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)

        return str(output_path)

    @staticmethod
    def list_annotations(
        directory: str = "./hybrid_annotations",
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        디렉토리에서 어노테이션 목록을 반환합니다.

        Args:
            directory: 어노테이션 디렉토리
            status_filter: 리뷰 상태 필터 (없으면 전체)

        Returns:
            어노테이션 요약 리스트
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"[WARNING] 디렉토리 없음: {directory}")
            return []

        summaries = []
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                review_status = data.get("review_status", "pending")
                if status_filter and review_status != status_filter:
                    continue

                summaries.append({
                    "file": str(json_file),
                    "workflow_id": data.get("workflow_id", ""),
                    "video_path": data.get("video_path", ""),
                    "steps": len(data.get("steps", [])),
                    "review_status": review_status,
                    "reviewed_by": data.get("reviewed_by"),
                    "extraction_method": data.get("extraction_method", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return summaries


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(
        description="하이브리드 어노테이션 도구 — VLM 자동 추출 + 사람 리뷰"
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract-and-review 서브커맨드
    extract_parser = subparsers.add_parser(
        "extract-and-review",
        help="영상에서 자동 추출 후 리뷰"
    )
    extract_parser.add_argument("video", help="영상 파일 경로")
    extract_parser.add_argument("--reviewer", default="", help="리뷰어 이름")
    extract_parser.add_argument("--recipe-type", default="OTHER", help="레시피 타입")
    extract_parser.add_argument("--description", default="", help="워크플로우 설명")
    extract_parser.add_argument("--output-dir", default="./hybrid_annotations")
    extract_parser.add_argument("--vlm-url", help="VLM API URL")
    extract_parser.add_argument("--vlm-key", help="VLM API 키")
    extract_parser.add_argument("--vlm-model", default="qwen-vl-max")
    extract_parser.add_argument(
        "--vlm-provider", default="qwen_vl",
        choices=["qwen_vl", "openai_gpt4v", "anthropic_claude", "qwen3_vl"],
    )

    # review 서브커맨드
    review_parser = subparsers.add_parser(
        "review",
        help="기존 자동 추출 JSON 리뷰"
    )
    review_parser.add_argument("json_file", help="어노테이션 JSON 파일 경로")
    review_parser.add_argument("--reviewer", default="", help="리뷰어 이름")
    review_parser.add_argument("--output-dir", default="./hybrid_annotations")

    # stats 서브커맨드
    stats_parser = subparsers.add_parser(
        "stats",
        help="리뷰 통계 출력"
    )
    stats_parser.add_argument("json_file", help="어노테이션 JSON 파일 경로")

    # diff 서브커맨드
    diff_parser = subparsers.add_parser(
        "diff",
        help="VLM 제안 vs 사람 수정 비교"
    )
    diff_parser.add_argument("json_file", help="어노테이션 JSON 파일 경로")

    # list 서브커맨드
    list_parser = subparsers.add_parser(
        "list",
        help="어노테이션 목록"
    )
    list_parser.add_argument("--dir", default="./hybrid_annotations", help="디렉토리")
    list_parser.add_argument("--status", default=None, help="리뷰 상태 필터")

    args = parser.parse_args()

    if args.command == "extract-and-review":
        extractor_config = AutoExtractorConfig(
            vlm_api_url=args.vlm_url,
            vlm_api_key=args.vlm_key,
            vlm_model=args.vlm_model,
            vlm_provider=args.vlm_provider,
        )
        annotator_config = HybridAnnotatorConfig(
            output_dir=args.output_dir,
            reviewer_name=args.reviewer,
        )
        annotator = HybridAnnotator(annotator_config)
        result = annotator.extract_and_review(
            video_path=args.video,
            reviewer=args.reviewer,
            recipe_type=args.recipe_type,
            description=args.description,
            extractor_config=extractor_config,
        )
        if result:
            print(f"\n완료: {len(result.steps)}개 단계 리뷰됨")

    elif args.command == "review":
        annotator_config = HybridAnnotatorConfig(
            output_dir=args.output_dir,
            reviewer_name=args.reviewer,
        )
        annotator = HybridAnnotator(annotator_config)
        result = annotator.review_from_json(args.json_file, reviewer=args.reviewer)
        if result:
            print(f"\n완료: {len(result.steps)}개 단계 리뷰됨")

    elif args.command == "stats":
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotation = WorkflowAnnotation.from_dict(data)
        annotator = HybridAnnotator()
        annotator._print_review_summary(annotation)

    elif args.command == "diff":
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotation = WorkflowAnnotation.from_dict(data)
        annotator = HybridAnnotator()
        annotator.print_diff(annotation)

    elif args.command == "list":
        summaries = HybridAnnotator.list_annotations(
            directory=args.dir,
            status_filter=args.status,
        )
        if not summaries:
            print("어노테이션이 없습니다.")
        else:
            print(f"\n{'='*60}")
            print(f"  어노테이션 목록 ({len(summaries)}개)")
            print(f"{'='*60}")
            for s in summaries:
                print(f"  {s['workflow_id']}  "
                      f"[{s['review_status']}]  "
                      f"{s['steps']}단계  "
                      f"{s.get('reviewed_by') or '-'}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
