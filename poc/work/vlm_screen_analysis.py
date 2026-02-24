"""
VLM Screen Analysis Module

VLM(Vision Language Model)을 활용하여 화면 상태를 분석합니다.
Qwen3-VL API 또는 다른 VLM API와 연동하여 화면 이해 기능을 제공합니다.

요구사항: FR-02 (VLM 기반 화면 상태 인식)
테스트 케이스: TC-02 (상태 매칭), TC-03 (미등록 상태), TC-10 (AI 채팅 응답)
"""

import os
import base64
import json
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient


@dataclass
class ScreenAnalysisResult:
    """화면 분석 결과"""
    state_id: str
    state_name: str
    confidence: float
    description: str
    ui_elements: List[Dict[str, Any]]
    suggested_actions: List[str]
    raw_response: str
    processing_time_ms: float


@dataclass
class MeasurementJudgment:
    """측정 결과 판단"""
    success: bool
    confidence: float
    failure_reason: Optional[str]
    suggested_adjustment: Optional[Dict[str, Any]]
    raw_response: str


class VLMScreenAnalyzer:
    """VLM 기반 화면 분석 클래스"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            api_key: API 키 (환경변수에서도 읽음)
            api_base_url: OpenAI 호환 API 기본 URL
            model_name: 사용할 모델 이름 (예: Qwen3-VL-30B-Instruct)
        """
        self.api_key = api_key or os.environ.get("VLM_API_KEY")
        self.api_base_url = (
            api_base_url
            or os.environ.get("VLM_API_URL")
            or os.environ.get("VLM_API_BASE_URL")
        )
        self.model_name = model_name or os.environ.get("VLM_MODEL_NAME", "")
        self.vlm_client = LangChainOpenAICompatibleVLMClient(
            base_url=self.api_base_url or "",
            api_key=self.api_key or "",
            timeout_sec=60.0,
        )

        # 상태 정의 템플릿
        self.state_definitions: Dict[str, Dict] = {}

        print(f"[INFO] VLMScreenAnalyzer 초기화 - Model: {self.model_name}")

    def load_state_definitions(self, definitions: Dict[str, Dict]):
        """
        상태 정의를 로드합니다.

        Args:
            definitions: 상태 정의 딕셔너리
        """
        self.state_definitions = definitions
        print(f"[INFO] {len(definitions)}개의 상태 정의 로드됨")

    def _encode_image_to_base64(self, image_data: bytes) -> str:
        """이미지를 Base64로 인코딩합니다."""
        return base64.b64encode(image_data).decode('utf-8')

    def _build_analysis_prompt(self, task: str = "state_recognition") -> str:
        """
        분석 프롬프트를 생성합니다.

        Args:
            task: 작업 타입

        Returns:
            프롬프트 문자열
        """
        state_context = ""
        if self.state_definitions:
            state_list = "\n".join([
                f"- {sid}: {sdef.get('state_name', '')} - {sdef.get('description', '')}"
                for sid, sdef in self.state_definitions.items()
            ])
            state_context = f"\n\n알려진 상태 목록:\n{state_list}"

        base_prompts = {
            "state_recognition": f"""당신은 GUI 화면 분석 전문가입니다. 주어진 스크린샷을 분석하여 현재 화면의 상태를 파악해주세요.

다음 정보를 JSON 형식으로 반환해주세요:
{{
    "state_id": "화면 상태 식별자 (예: main_menu, recipe_editor, error_popup)",
    "state_name": "화면 상태 이름 (한글)",
    "confidence": 0.0-1.0 사이의 확신도,
    "description": "현재 화면에 대한 상세 설명",
    "ui_elements": [
        {{"name": "요소 이름", "type": "button/input/label/etc", "location": "위치 설명"}}
    ],
    "suggested_actions": ["가능한 액션 1", "가능한 액션 2"]
}}{state_context}

분석 결과를 JSON으로만 반환해주세요.""",

            "measurement_judgment": """당신은 반도체 측정 장비의 결과 분석 전문가입니다. 주어진 측정 결과 화면을 분석하여 측정 성공 여부를 판단해주세요.

다음 정보를 JSON 형식으로 반환해주세요:
{
    "success": true/false,
    "confidence": 0.0-1.0 사이의 확신도,
    "failure_reason": "실패 시 원인 (position_offset, focus_error, pattern_mismatch 등)",
    "suggested_adjustment": {
        "direction": "left/right/up/down",
        "amount": "small/medium/large"
    }
}

측정 성공 기준:
- 측정값이 명확하게 표시되어 있음
- 에러 메시지가 없음
- 측정 패턴이 올바르게 인식됨

분석 결과를 JSON으로만 반환해주세요.""",

            "general_query": """당신은 GUI 화면 분석 전문가입니다. 주어진 화면에 대해 사용자의 질문에 답변해주세요.

답변은 명확하고 간결하게 해주세요."""
        }

        return base_prompts.get(task, base_prompts["general_query"])

    def analyze_screen(
        self,
        image_data: bytes,
        task: str = "state_recognition",
    ) -> Optional[ScreenAnalysisResult]:
        """
        화면을 분석하여 상태를 인식합니다.

        Args:
            image_data: PNG 이미지 바이트 데이터
            task: 분석 작업 유형 (state_recognition, measurement_judgment)

        Returns:
            ScreenAnalysisResult 또는 None
        """
        start_time = time.time()

        prompt = self._build_analysis_prompt(task)

        # API 호출
        response = self._call_vlm_api(image_data, prompt)

        if not response:
            return None

        processing_time = (time.time() - start_time) * 1000

        # 응답 파싱
        try:
            # JSON 블록 추출
            json_str = self._extract_json_from_response(response)
            result_data = json.loads(json_str)

            return ScreenAnalysisResult(
                state_id=result_data.get("state_id", "unknown"),
                state_name=result_data.get("state_name", "알 수 없음"),
                confidence=result_data.get("confidence", 0.0),
                description=result_data.get("description", ""),
                ui_elements=result_data.get("ui_elements", []),
                suggested_actions=result_data.get("suggested_actions", []),
                raw_response=response,
                processing_time_ms=processing_time
            )
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 실패: {e}")
            return ScreenAnalysisResult(
                state_id="parse_error",
                state_name="파싱 오류",
                confidence=0.0,
                description=response,
                ui_elements=[],
                suggested_actions=[],
                raw_response=response,
                processing_time_ms=processing_time
            )

    def judge_measurement(self, image_data: bytes) -> Optional[MeasurementJudgment]:
        """
        측정 결과를 판단합니다.

        Args:
            image_data: 측정 결과 화면 이미지

        Returns:
            MeasurementJudgment 또는 None
        """
        prompt = self._build_analysis_prompt("measurement_judgment")

        response = self._call_vlm_api(image_data, prompt)

        if not response:
            return None

        try:
            json_str = self._extract_json_from_response(response)
            result_data = json.loads(json_str)

            return MeasurementJudgment(
                success=result_data.get("success", False),
                confidence=result_data.get("confidence", 0.0),
                failure_reason=result_data.get("failure_reason"),
                suggested_adjustment=result_data.get("suggested_adjustment"),
                raw_response=response
            )
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 실패: {e}")
            return None

    def ask_about_screen(self, image_data: bytes, question: str) -> Optional[str]:
        """
        화면에 대한 질문에 답변합니다 (TC-10: AI 채팅 응답).

        Args:
            image_data: 화면 이미지
            question: 사용자 질문

        Returns:
            답변 문자열 또는 None
        """
        prompt = f"""당신은 GUI 화면 분석 전문가입니다. 주어진 화면에 대해 다음 질문에 답변해주세요.

질문: {question}

답변은 명확하고 간결하게 해주세요."""

        response = self._call_vlm_api(image_data, prompt)
        return response

    def _call_vlm_api(self, image_data: bytes, prompt: str) -> Optional[str]:
        """
        VLM API 호출 (OpenAI 호환 형식).

        Args:
            image_data: 이미지 데이터
            prompt: 프롬프트

        Returns:
            API 응답 텍스트 또는 None
        """
        if not self.api_base_url:
            print("[INFO] VLM API URL이 설정되지 않았습니다. Mock 응답을 반환합니다.")
            return self._get_mock_response(prompt)

        if not self.model_name:
            print("[ERROR] VLM_MODEL_NAME이 설정되지 않았습니다.")
            return None

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_mime = (
            "image/webp"
            if image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP'
            else "image/png"
        )
        request = ChatImageRequest(
            model=self.model_name,
            system_message=(
                "당신은 GUI 화면 분석 전문가입니다. "
                "반드시 요청된 출력 형식(JSON 등)을 정확히 지켜 응답하세요."
            ),
            user_text=prompt,
            image_b64=image_base64,
            image_mime=image_mime,
            temperature=0.1,
        )

        try:
            return self.vlm_client.chat_with_image(request)
        except Exception as e:
            print(f"[ERROR] VLM API 호출 실패: {e}")
            return self._get_mock_response(prompt)

    def _get_mock_response(self, prompt: str) -> str:
        """테스트용 Mock 응답을 반환합니다."""
        print("[INFO] Mock 응답 생성 중...")

        if "state_recognition" in prompt or "상태" in prompt:
            return json.dumps({
                "state_id": "mock_main_menu",
                "state_name": "메인 메뉴 (Mock)",
                "confidence": 0.85,
                "description": "이것은 테스트용 Mock 응답입니다. 실제 VLM API가 연결되면 실제 분석 결과가 반환됩니다.",
                "ui_elements": [
                    {"name": "샘플 버튼", "type": "button", "location": "화면 중앙"},
                    {"name": "검색창", "type": "input", "location": "상단"}
                ],
                "suggested_actions": ["버튼 클릭", "텍스트 입력"]
            }, ensure_ascii=False)

        elif "measurement" in prompt or "측정" in prompt:
            return json.dumps({
                "success": True,
                "confidence": 0.92,
                "failure_reason": None,
                "suggested_adjustment": None
            }, ensure_ascii=False)

        else:
            return "이것은 테스트용 Mock 응답입니다. VLM API가 연결되면 실제 답변이 제공됩니다."

    def _extract_json_from_response(self, response: str) -> str:
        """응답에서 JSON 블록을 추출합니다."""
        # JSON 블록 찾기
        start_markers = ['{', '```json\n{', '```\n{']
        end_markers = ['}', '}\n```', '}\n```']

        for start, end in zip(start_markers, end_markers):
            if start in response:
                start_idx = response.find(start)
                if '```' in start:
                    start_idx = response.find('{', start_idx)

                # 마지막 } 찾기
                end_idx = response.rfind('}')
                if end_idx > start_idx:
                    return response[start_idx:end_idx + 1]

        return response
