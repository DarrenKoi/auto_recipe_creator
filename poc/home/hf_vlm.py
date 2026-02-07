"""
Hugging Face VLM Provider

무료 Hugging Face Inference API를 사용한 Vision Language Model 통합.
GPU 없이 CPU만으로 동작하며, 집에서 학습용으로 사용할 수 있습니다.

Rate Limits (무료 계정):
- 약 100-300 requests/hour
- 모델에 따라 다름

Supported Models:
- Qwen/Qwen2-VL-7B-Instruct (추천)
- llava-hf/llava-1.5-7b-hf
- microsoft/Florence-2-large

Usage:
    from poc.home.hf_vlm import HuggingFaceVLM

    vlm = HuggingFaceVLM(token="hf_xxxx")
    result = vlm.analyze_screen(image_bytes, "화면에 보이는 버튼을 찾아주세요")
"""

import os
import base64
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from io import BytesIO

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HFModel(Enum):
    """사용 가능한 HuggingFace 모델"""
    # Vision-Language Models (chat_completion with images)
    QWEN2_VL_7B = "Qwen/Qwen2-VL-7B-Instruct"
    QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"

    # Image-to-Text Models (captioning)
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    GIT_LARGE = "microsoft/git-large-coco"

    # Object Detection
    DETR = "facebook/detr-resnet-50"
    YOLOS = "hustvl/yolos-tiny"


@dataclass
class VLMResponse:
    """VLM 응답 결과"""
    success: bool
    content: str
    model: str
    latency_ms: float
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class DetectedObject:
    """탐지된 객체"""
    label: str
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]


class HuggingFaceVLM:
    """
    Hugging Face 무료 Inference API를 사용한 VLM 클래스

    집에서 GPU 없이 GUI 자동화를 학습하기 위한 용도.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: HFModel = HFModel.QWEN2_VL_7B,
        timeout: int = 60
    ):
        """
        Args:
            token: HuggingFace API 토큰 (없으면 HF_TOKEN 환경변수 사용)
            model: 사용할 모델
            timeout: API 타임아웃 (초)
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub이 설치되지 않았습니다.\n"
                "설치: uv sync --extra home"
            )

        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            print("[WARNING] HF_TOKEN이 설정되지 않았습니다.")
            print("[WARNING] 익명 요청은 rate limit이 더 낮습니다.")
            print("[INFO] 토큰 발급: https://huggingface.co/settings/tokens")

        self.model = model
        self.timeout = timeout

        # HF Inference Client 초기화
        self.client = InferenceClient(
            token=self.token,
            timeout=timeout
        )

        print(f"[INFO] HuggingFace VLM 초기화 완료")
        print(f"[INFO] 모델: {model.value}")
        print(f"[INFO] 타임아웃: {timeout}초")

    def analyze_screen(
        self,
        image: bytes,
        prompt: str,
        max_tokens: int = 1024
    ) -> VLMResponse:
        """
        화면 이미지를 분석하여 텍스트 응답 반환

        Args:
            image: 이미지 바이트 (PNG/JPEG/WebP)
            prompt: 분석 요청 프롬프트
            max_tokens: 최대 응답 토큰 수

        Returns:
            VLMResponse: 분석 결과
        """
        import time

        start_time = time.time()

        try:
            # 이미지를 base64로 인코딩
            image_b64 = base64.b64encode(image).decode("utf-8")

            # Chat completion with image (multimodal)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            response = self.client.chat_completion(
                model=self.model.value,
                messages=messages,
                max_tokens=max_tokens
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            return VLMResponse(
                success=True,
                content=content,
                model=self.model.value,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            # 일반적인 오류 메시지 개선
            if "401" in error_msg:
                error_msg = "인증 실패: HF_TOKEN을 확인해주세요"
            elif "429" in error_msg:
                error_msg = "Rate limit 초과: 잠시 후 다시 시도해주세요"
            elif "503" in error_msg:
                error_msg = "모델 로딩 중: 잠시 후 다시 시도해주세요 (cold start)"

            print(f"[ERROR] VLM 호출 실패: {error_msg}")

            return VLMResponse(
                success=False,
                content="",
                model=self.model.value,
                latency_ms=latency_ms,
                error=error_msg
            )

    def detect_objects(self, image: bytes) -> List[DetectedObject]:
        """
        이미지에서 객체 탐지 (Object Detection)

        UI 요소 탐지에 유용합니다.

        Args:
            image: 이미지 바이트

        Returns:
            List[DetectedObject]: 탐지된 객체 목록
        """
        try:
            # DETR 또는 YOLOS 사용
            results = self.client.object_detection(
                image=image,
                model=HFModel.DETR.value
            )

            detected = []
            for obj in results:
                bbox = obj.get("box", {})
                detected.append(DetectedObject(
                    label=obj.get("label", "unknown"),
                    score=obj.get("score", 0.0),
                    bbox=[
                        int(bbox.get("xmin", 0)),
                        int(bbox.get("ymin", 0)),
                        int(bbox.get("xmax", 0)),
                        int(bbox.get("ymax", 0))
                    ]
                ))

            return detected

        except Exception as e:
            print(f"[ERROR] 객체 탐지 실패: {e}")
            return []

    def caption_image(self, image: bytes) -> str:
        """
        이미지 캡션 생성 (Image-to-Text)

        Args:
            image: 이미지 바이트

        Returns:
            str: 이미지 설명
        """
        try:
            result = self.client.image_to_text(
                image=image,
                model=HFModel.GIT_LARGE.value
            )

            if isinstance(result, str):
                return result
            elif hasattr(result, 'generated_text'):
                return result.generated_text
            else:
                return str(result)

        except Exception as e:
            print(f"[ERROR] 이미지 캡션 실패: {e}")
            return ""

    def analyze_ui_elements(
        self,
        image: bytes,
        return_json: bool = True
    ) -> VLMResponse:
        """
        UI 요소 분석 (GUI 자동화용)

        화면에서 클릭 가능한 요소들을 찾아 JSON으로 반환합니다.

        Args:
            image: 화면 캡처 이미지
            return_json: JSON 형식으로 반환 요청

        Returns:
            VLMResponse: UI 요소 분석 결과
        """
        if return_json:
            prompt = """
Analyze this screenshot and identify all interactive UI elements.
Return the result as a JSON object with the following format:

{
  "screen_type": "application type or window name",
  "ui_elements": [
    {
      "name": "element description",
      "type": "button|input|checkbox|dropdown|link|menu|label",
      "location": "top-left|top-center|top-right|center|bottom-left|etc",
      "text": "visible text on the element (if any)"
    }
  ],
  "possible_actions": ["action 1", "action 2"]
}

Only respond with valid JSON, no additional text.
"""
        else:
            prompt = """
Describe this screenshot:
1. What application or window is shown?
2. List all visible buttons, inputs, and interactive elements
3. What actions can be performed on this screen?
"""

        return self.analyze_screen(image, prompt)


def get_recommended_models() -> Dict[str, str]:
    """
    작업별 추천 모델 목록

    Returns:
        Dict[str, str]: 작업 -> 모델 매핑
    """
    return {
        "screen_analysis": HFModel.QWEN2_VL_7B.value,
        "ui_detection": HFModel.QWEN2_VL_2B.value,
        "object_detection": HFModel.DETR.value,
        "image_caption": HFModel.GIT_LARGE.value,
    }
