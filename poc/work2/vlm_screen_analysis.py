"""`poc.work2` screen analysis with optional PaddleOCR assist pipeline."""

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient
from poc.work2.flask_vlm import apply_pipeline_env_defaults
from poc.work2.logger import log_vlm_call
from poc.work2.pipeline_ocr import build_ocr_extra_instructions, collect_ocr_hint_result
from poc.work2.prompts import (
    build_general_query_prompt,
    build_measurement_judgment_prompt,
    build_state_recognition_prompt,
)


@dataclass
class ScreenAnalysisResult:
    """화면 분석 결과."""

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
    """측정 결과 판단."""

    success: bool
    confidence: float
    failure_reason: Optional[str]
    suggested_adjustment: Optional[Dict[str, Any]]
    raw_response: str


class VLMScreenAnalyzer:
    """screen_analysis purpose VLM + PaddleOCR assist 기반 화면 분석 클래스."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        pipeline_config: Optional[dict[str, object]] = None,
    ):
        self.pipeline_config = pipeline_config or apply_pipeline_env_defaults()
        self.api_key = api_key or str(self.pipeline_config.get("screen_analysis_api_key", "") or "")
        self.api_base_url = (
            api_base_url
            or str(self.pipeline_config.get("screen_analysis_api_url", "") or "")
            or os.environ.get("VLM_API_URL")
            or os.environ.get("VLM_API_BASE_URL")
        )
        self.model_name = (
            model_name
            or str(self.pipeline_config.get("screen_analysis_model_name", "") or "")
            or os.environ.get("VLM_MODEL_NAME", "")
        )
        self.vlm_client = LangChainOpenAICompatibleVLMClient(
            base_url=self.api_base_url or "",
            api_key=self.api_key or "",
            timeout_sec=60.0,
        )
        self.state_definitions: Dict[str, Dict] = {}

        print(f"[INFO] VLMScreenAnalyzer 초기화 - Model: {self.model_name}")

    def load_state_definitions(self, definitions: Dict[str, Dict]):
        """상태 정의를 로드한다."""
        self.state_definitions = definitions
        print(f"[INFO] {len(definitions)}개의 상태 정의 로드됨")

    def _build_analysis_prompt(
        self,
        task: str = "state_recognition",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        *,
        question: str = "",
        extra_instructions: Iterable[str] | None = None,
    ) -> str:
        if task == "state_recognition":
            return build_state_recognition_prompt(
                image_width=image_width,
                image_height=image_height,
                state_definitions=self.state_definitions,
                extra_instructions=extra_instructions,
            )
        if task == "measurement_judgment":
            return build_measurement_judgment_prompt(extra_instructions=extra_instructions)
        return build_general_query_prompt(
            question=question,
            extra_instructions=extra_instructions,
        )

    @staticmethod
    def _image_mime(image_data: bytes) -> str:
        if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    def _build_pipeline_instructions(
        self,
        *,
        image_data: bytes,
        image_width: Optional[int],
        image_height: Optional[int],
        extra_instructions: Iterable[str] | None = None,
        ocr_context_label: str = "",
        ocr_focus_words: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        instructions = [item.strip() for item in (extra_instructions or ()) if item and item.strip()]
        if not image_width or not image_height:
            return tuple(instructions)

        ocr_result = collect_ocr_hint_result(
            image_b64=base64.b64encode(image_data).decode("utf-8"),
            image_width=image_width,
            image_height=image_height,
            image_mime=self._image_mime(image_data),
            pipeline_config=self.pipeline_config,
            context_label=ocr_context_label,
            focus_words=ocr_focus_words,
        )
        instructions.extend(build_ocr_extra_instructions(ocr_result))
        return tuple(instructions)

    def analyze_screen(
        self,
        image_data: bytes,
        task: str = "state_recognition",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        *,
        extra_instructions: Iterable[str] | None = None,
        ocr_context_label: str = "",
        ocr_focus_words: Iterable[str] | None = None,
    ) -> Optional[ScreenAnalysisResult]:
        """화면을 분석하여 상태를 인식한다."""
        start_time = time.time()
        pipeline_instructions = self._build_pipeline_instructions(
            image_data=image_data,
            image_width=image_width,
            image_height=image_height,
            extra_instructions=extra_instructions,
            ocr_context_label=ocr_context_label,
            ocr_focus_words=ocr_focus_words,
        )
        prompt = self._build_analysis_prompt(
            task=task,
            image_width=image_width,
            image_height=image_height,
            extra_instructions=pipeline_instructions,
        )
        response = self._call_vlm_api(image_data, prompt)

        if not response:
            return None

        processing_time = (time.time() - start_time) * 1000
        try:
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
                processing_time_ms=processing_time,
            )
        except json.JSONDecodeError as exc:
            print(f"[ERROR] JSON 파싱 실패: {exc}")
            return ScreenAnalysisResult(
                state_id="parse_error",
                state_name="파싱 오류",
                confidence=0.0,
                description=response,
                ui_elements=[],
                suggested_actions=[],
                raw_response=response,
                processing_time_ms=processing_time,
            )

    def judge_measurement(
        self,
        image_data: bytes,
        *,
        extra_instructions: Iterable[str] | None = None,
    ) -> Optional[MeasurementJudgment]:
        """측정 결과를 판단한다."""
        prompt = self._build_analysis_prompt(
            "measurement_judgment",
            extra_instructions=extra_instructions,
        )
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
                raw_response=response,
            )
        except json.JSONDecodeError as exc:
            print(f"[ERROR] JSON 파싱 실패: {exc}")
            return None

    def ask_about_screen(
        self,
        image_data: bytes,
        question: str,
        *,
        extra_instructions: Iterable[str] | None = None,
    ) -> Optional[str]:
        """화면에 대한 질문에 답변한다."""
        prompt = self._build_analysis_prompt(
            "general_query",
            question=question,
            extra_instructions=extra_instructions,
        )
        return self._call_vlm_api(image_data, prompt)

    def _call_vlm_api(self, image_data: bytes, prompt: str) -> Optional[str]:
        if not self.api_base_url:
            print("[INFO] VLM API URL이 설정되지 않았습니다. Mock 응답을 반환합니다.")
            return self._get_mock_response(prompt)

        if not self.model_name:
            print("[ERROR] VLM_MODEL_NAME이 설정되지 않았습니다.")
            return None

        request = ChatImageRequest(
            model=self.model_name,
            system_message=(
                "당신은 GUI 화면 분석 전문가입니다. "
                "반드시 요청된 출력 형식(JSON 등)을 정확히 지켜 응답하세요."
            ),
            user_text=prompt,
            image_b64=base64.b64encode(image_data).decode("utf-8"),
            image_mime=self._image_mime(image_data),
            temperature=0.1,
        )

        start_ms = time.time()
        try:
            result = self.vlm_client.chat_with_image(request)
            log_vlm_call(
                service="screen_analysis",
                model=self.model_name,
                status="ok",
                latency_ms=(time.time() - start_ms) * 1000,
                token_usage=self.vlm_client.last_token_usage,
                endpoint=self.vlm_client.endpoint,
            )
            return result
        except Exception as exc:
            log_vlm_call(
                service="screen_analysis",
                model=self.model_name,
                status="error",
                latency_ms=(time.time() - start_ms) * 1000,
                error=str(exc),
                endpoint=self.vlm_client.endpoint,
            )
            print(f"[ERROR] VLM API 호출 실패: {exc}")
            return self._get_mock_response(prompt)

    def _get_mock_response(self, prompt: str) -> str:
        if "측정" in prompt or "measurement" in prompt:
            return json.dumps(
                {
                    "success": True,
                    "confidence": 0.92,
                    "failure_reason": None,
                    "suggested_adjustment": None,
                },
                ensure_ascii=False,
            )

        if "상태" in prompt or "state" in prompt:
            return json.dumps(
                {
                    "state_id": "mock_main_menu",
                    "state_name": "메인 메뉴 (Mock)",
                    "confidence": 0.85,
                    "description": "이것은 테스트용 Mock 응답입니다.",
                    "ui_elements": [
                        {
                            "name": "샘플 버튼",
                            "type": "button",
                            "location": "화면 중앙",
                            "x": 640,
                            "y": 360,
                            "coord_anchor": "element_center",
                        }
                    ],
                    "suggested_actions": ["버튼 클릭"],
                },
                ensure_ascii=False,
            )

        return "이것은 테스트용 Mock 응답입니다."

    def _extract_json_from_response(self, response: str) -> str:
        start_markers = ["{", "```json\n{", "```\n{"]
        for start in start_markers:
            if start in response:
                start_idx = response.find(start)
                if "```" in start:
                    start_idx = response.find("{", start_idx)
                end_idx = response.rfind("}")
                if end_idx > start_idx:
                    return response[start_idx : end_idx + 1]
        return response
