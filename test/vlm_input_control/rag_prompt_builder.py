"""
RAG Prompt Builder

RAG 컨텍스트를 VLM 프롬프트로 변환
"""

from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from video_frame_parser.models import (
    RAGContext, FrameData, ActionSequence, ErrorPattern,
    UIElement, OCRResult
)


class RAGPromptBuilder:
    """
    RAG 컨텍스트를 VLM 프롬프트로 변환

    전문가 예시, 작업 시퀀스, 에러 패턴, UI 요소 정보를
    구조화된 프롬프트로 포맷팅
    """

    def __init__(
        self,
        max_similar_frames: int = 3,
        max_actions: int = 5,
        max_errors: int = 3,
        max_ui_elements: int = 10,
        include_temporal: bool = False
    ):
        """
        Args:
            max_similar_frames: 포함할 최대 유사 프레임 수
            max_actions: 포함할 최대 작업 시퀀스 수
            max_errors: 포함할 최대 에러 패턴 수
            max_ui_elements: 포함할 최대 UI 요소 수
            include_temporal: 시간적 컨텍스트 포함 여부
        """
        self.max_similar_frames = max_similar_frames
        self.max_actions = max_actions
        self.max_errors = max_errors
        self.max_ui_elements = max_ui_elements
        self.include_temporal = include_temporal

    def build_rag_augmented_prompt(
        self,
        base_prompt: str,
        rag_context: RAGContext
    ) -> str:
        """
        RAG 컨텍스트로 프롬프트 증강

        Args:
            base_prompt: 기본 프롬프트
            rag_context: RAG 컨텍스트

        Returns:
            증강된 프롬프트
        """
        sections = [base_prompt, ""]

        # 구분선
        sections.append("# 참고: CCTV 영상에서 추출한 전문가 작업 예시")
        sections.append("")

        # 유사 화면
        if rag_context.similar_frames:
            sections.append(self._format_similar_frames(
                rag_context.similar_frames[:self.max_similar_frames],
                rag_context.retrieval_scores[:self.max_similar_frames]
            ))
            sections.append("")

        # 전문가 작업 시퀀스
        if rag_context.action_sequences:
            sections.append(self._format_action_sequences(
                rag_context.action_sequences[:self.max_actions]
            ))
            sections.append("")

        # 에러 패턴
        if rag_context.error_patterns:
            sections.append(self._format_error_patterns(
                rag_context.error_patterns[:self.max_errors]
            ))
            sections.append("")

        # UI 요소
        if rag_context.ui_elements:
            sections.append(self._format_ui_elements(
                rag_context.ui_elements[:self.max_ui_elements]
            ))
            sections.append("")

        # OCR 결과 (간략)
        if rag_context.ocr_results:
            sections.append(self._format_ocr_summary(rag_context.ocr_results))
            sections.append("")

        # 시간적 컨텍스트
        if self.include_temporal and rag_context.temporal_frames:
            sections.append(self._format_temporal_context(
                rag_context.temporal_frames
            ))
            sections.append("")

        # 마무리 지시
        sections.append("위의 전문가 작업 예시를 참고하여, 현재 화면을 분석하고 최선의 작업을 제안해주세요.")

        return "\n".join(sections)

    def _format_similar_frames(
        self,
        frames: List[FrameData],
        scores: List[float]
    ) -> str:
        """유사 프레임 포맷팅"""
        lines = [f"## 유사한 화면 (Top-{len(frames)})"]
        lines.append("")

        for i, (frame, score) in enumerate(zip(frames, scores), 1):
            lines.append(f"{i}. **프레임**: {frame.frame_id}")
            lines.append(f"   - **유사도**: {score:.3f}")
            lines.append(f"   - **비디오**: {frame.video_id}")
            lines.append(f"   - **시간**: {frame.timestamp:.1f}초")
            if frame.frame_type:
                lines.append(f"   - **타입**: {frame.frame_type.value}")
            lines.append("")

        return "\n".join(lines)

    def _format_action_sequences(
        self,
        actions: List[ActionSequence]
    ) -> str:
        """작업 시퀀스 포맷팅"""
        lines = ["## 전문가 작업 시퀀스"]
        lines.append("")

        for i, action in enumerate(actions, 1):
            lines.append(f"{i}. **작업**: {action.description}")
            lines.append(f"   - **기간**: {action.start_time:.1f}초 ~ {action.end_time:.1f}초 "
                        f"({action.end_time - action.start_time:.1f}초)")
            lines.append(f"   - **성공**: {'✓ 성공' if action.success else '✗ 실패'}")

            # 작업 상세
            if action.actions:
                lines.append(f"   - **작업 상세**:")
                for j, act in enumerate(action.actions[:5], 1):  # 최대 5개
                    act_type = act.get("type", "unknown")
                    if act_type == "click":
                        lines.append(f"     {j}) 클릭: ({act.get('x')}, {act.get('y')})")
                    elif act_type == "type":
                        lines.append(f"     {j}) 입력: \"{act.get('text')}\"")
                    elif act_type == "wait":
                        lines.append(f"     {j}) 대기: {act.get('duration')}초")
                    else:
                        lines.append(f"     {j}) {act_type}")

                if len(action.actions) > 5:
                    lines.append(f"     ... 외 {len(action.actions) - 5}개 작업")

            # 에러/복구
            if not action.success and action.error_type:
                lines.append(f"   - **에러**: {action.error_type}")
                if action.recovery_action:
                    lines.append(f"   - **복구**: {action.recovery_action}")

            lines.append("")

        return "\n".join(lines)

    def _format_error_patterns(
        self,
        errors: List[ErrorPattern]
    ) -> str:
        """에러 패턴 포맷팅"""
        lines = ["## 주의: 발생 가능한 에러"]
        lines.append("")

        for i, error in enumerate(errors, 1):
            lines.append(f"{i}. **에러 타입**: {error.error_type} "
                        f"(심각도: {error.severity})")
            lines.append(f"   - **메시지**: {error.error_message}")
            lines.append(f"   - **위치**: {error.bbox}")
            lines.append(f"   - **복구 방법**: {error.recovery_action}")
            lines.append("")

        return "\n".join(lines)

    def _format_ui_elements(
        self,
        elements: List[UIElement]
    ) -> str:
        """UI 요소 포맷팅"""
        lines = ["## 감지된 UI 요소"]
        lines.append("")

        # 타입별 그룹화
        by_type = {}
        for element in elements:
            if element.element_type not in by_type:
                by_type[element.element_type] = []
            by_type[element.element_type].append(element)

        for elem_type, elems in by_type.items():
            lines.append(f"### {elem_type.capitalize()}s")
            for i, elem in enumerate(elems[:self.max_ui_elements], 1):
                label_text = f'"{elem.label}"' if elem.label else "(라벨 없음)"
                lines.append(f"{i}. {label_text} - 위치: {elem.bbox}")
            lines.append("")

        return "\n".join(lines)

    def _format_ocr_summary(
        self,
        ocr_results: List[OCRResult]
    ) -> str:
        """OCR 결과 요약 포맷팅"""
        lines = ["## 화면 내 텍스트"]
        lines.append("")

        # 신뢰도 높은 텍스트만 (>0.7)
        high_conf = [ocr for ocr in ocr_results if ocr.confidence > 0.7]

        if high_conf:
            # 언어별 그룹화
            korean = [ocr for ocr in high_conf if ocr.language in ["ko", "mixed"]]
            english = [ocr for ocr in high_conf if ocr.language == "en"]

            if korean:
                lines.append("**한국어 텍스트**:")
                for ocr in korean[:10]:  # 최대 10개
                    lines.append(f"- {ocr.text} (신뢰도: {ocr.confidence:.2f})")
                lines.append("")

            if english:
                lines.append("**영어 텍스트**:")
                for ocr in english[:10]:  # 최대 10개
                    lines.append(f"- {ocr.text} (confidence: {ocr.confidence:.2f})")
                lines.append("")
        else:
            lines.append("(신뢰도 높은 텍스트 없음)")
            lines.append("")

        return "\n".join(lines)

    def _format_temporal_context(
        self,
        temporal_frames: List[FrameData]
    ) -> str:
        """시간적 컨텍스트 포맷팅"""
        lines = ["## 시간적 컨텍스트 (전후 프레임)"]
        lines.append("")

        if not temporal_frames:
            lines.append("(전후 프레임 없음)")
        else:
            lines.append(f"총 {len(temporal_frames)}개 프레임")

        lines.append("")
        return "\n".join(lines)

    def build_compact_prompt(
        self,
        base_prompt: str,
        rag_context: RAGContext,
        max_length: int = 2000
    ) -> str:
        """
        압축된 프롬프트 생성 (토큰 제한이 있는 경우)

        Args:
            base_prompt: 기본 프롬프트
            rag_context: RAG 컨텍스트
            max_length: 최대 문자 길이

        Returns:
            압축된 프롬프트
        """
        # 우선순위: 작업 시퀀스 > 에러 패턴 > UI 요소 > 유사 프레임
        sections = [base_prompt, "", "# 참고: 전문가 작업 예시", ""]

        # 작업 시퀀스 (간략)
        if rag_context.action_sequences:
            sections.append("**작업**:")
            for action in rag_context.action_sequences[:2]:
                success = "✓" if action.success else "✗"
                sections.append(f"- {action.description} {success}")
            sections.append("")

        # 에러 패턴 (간략)
        if rag_context.error_patterns:
            sections.append("**주의 에러**:")
            for error in rag_context.error_patterns[:2]:
                sections.append(f"- {error.error_type}: {error.recovery_action}")
            sections.append("")

        prompt = "\n".join(sections)

        # 길이 초과 시 추가 압축
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "\n..."

        return prompt


def create_rag_prompt_builder(
    max_similar_frames: int = 3,
    compact: bool = False
) -> RAGPromptBuilder:
    """
    RAGPromptBuilder 생성 헬퍼 함수

    Args:
        max_similar_frames: 최대 유사 프레임 수
        compact: 압축 모드

    Returns:
        RAGPromptBuilder 인스턴스
    """
    if compact:
        return RAGPromptBuilder(
            max_similar_frames=1,
            max_actions=2,
            max_errors=2,
            max_ui_elements=5,
            include_temporal=False
        )
    else:
        return RAGPromptBuilder(
            max_similar_frames=max_similar_frames,
            max_actions=5,
            max_errors=3,
            max_ui_elements=10,
            include_temporal=False
        )


if __name__ == "__main__":
    # 사용 예시
    from video_frame_parser.models import (
        RAGContext, FrameData, ActionSequence, FrameType
    )
    from datetime import datetime

    # 테스트 데이터
    test_frames = [
        FrameData(
            frame_id="frame_001",
            video_id="video_001",
            frame_number=100,
            timestamp=10.0,
            frame_type=FrameType.KEYFRAME
        )
    ]

    test_actions = [
        ActionSequence(
            action_id="action_001",
            video_id="video_001",
            start_frame_id="frame_001",
            end_frame_id="frame_050",
            start_time=10.0,
            end_time=15.0,
            actions=[
                {"type": "click", "x": 100, "y": 200, "timestamp": 10.5},
                {"type": "type", "text": "admin", "timestamp": 11.0},
            ],
            description="RCS 로그인 수행",
            success=True
        )
    ]

    test_context = RAGContext(
        similar_frames=test_frames,
        action_sequences=test_actions,
        error_patterns=[],
        ui_elements=[],
        temporal_frames=[],
        retrieval_scores=[0.95],
        retrieval_time_ms=80.0,
        query_text="RCS 로그인",
        retrieval_method="hybrid",
        top_k=3
    )

    # 프롬프트 생성
    builder = create_rag_prompt_builder()
    prompt = builder.build_rag_augmented_prompt(
        base_prompt="화면을 분석하여 RCS 로그인에 필요한 작업을 제안해주세요.",
        rag_context=test_context
    )

    print(prompt)
    print(f"\n[INFO] Prompt length: {len(prompt)} characters")
