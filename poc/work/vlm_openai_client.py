"""OpenAI-compatible VLM client utilities."""

import json as _json
from dataclasses import dataclass
from typing import Any

import requests

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False


@dataclass(frozen=True)
class ChatImageRequest:
    """VLM 채팅 이미지 요청 데이터."""
    model: str
    system_message: str
    user_text: str
    image_b64: str
    image_mime: str = "image/webp"
    temperature: float = 0.0


class OpenAICompatibleVLMClient:
    """Minimal client for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, base_url: str, api_key: str = "", timeout_sec: float = 120.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_sec = timeout_sec

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _coerce_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return str(content)

    def chat_with_image(self, request: ChatImageRequest) -> str:
        b64_len_kb = len(request.image_b64) * 3 / 4 / 1024
        print(f"[INFO] VLM 요청: model={request.model}, image={b64_len_kb:.1f}KB ({request.image_mime})")

        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{request.image_mime};base64,{request.image_b64}"},
                    },
                ],
            },
        )

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )

        # 응답 상세 로그
        print(f"[INFO] VLM 응답 status={response.status_code}")
        print(f"[INFO] VLM 응답 headers: content-type={response.headers.get('content-type', 'N/A')}")
        try:
            raw_body = response.json()
            # choices 내용은 길 수 있으므로 요약 출력
            log_body = dict(raw_body)
            if "choices" in log_body:
                log_body["choices"] = f"[{len(raw_body['choices'])} item(s)]"
            print(f"[INFO] VLM 응답 body (요약): {_json.dumps(log_body, ensure_ascii=False)}")
        except Exception:
            print(f"[INFO] VLM 응답 body (raw): {response.text[:500]}")

        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"VLM response has no choices: {_json.dumps(data, ensure_ascii=False)[:300]}")
        message = choices[0].get("message") or {}
        return self._coerce_content(message.get("content", ""))


class LangChainOpenAICompatibleVLMClient:
    """LangChain ChatOpenAI 기반 OpenAI 호환 VLM 클라이언트."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_sec: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    @property
    def openai_base_url(self) -> str:
        if not self.base_url:
            return ""
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    @property
    def endpoint(self) -> str:
        if not self.openai_base_url:
            return ""
        return f"{self.openai_base_url}/chat/completions"

    def _build_chat_model(self, model: str, temperature: float) -> Any:
        if not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError(
                "langchain_openai가 설치되어 있지 않습니다. "
                "pip install langchain-openai langchain-core"
            )

        modern_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": self.timeout_sec,
            "max_retries": self.max_retries,
        }
        if self.api_key:
            modern_kwargs["api_key"] = self.api_key
        if self.openai_base_url:
            modern_kwargs["base_url"] = self.openai_base_url

        legacy_kwargs: dict[str, Any] = {
            "model_name": model,
            "temperature": temperature,
            "request_timeout": self.timeout_sec,
            "max_retries": self.max_retries,
        }
        if self.api_key:
            legacy_kwargs["openai_api_key"] = self.api_key
        if self.openai_base_url:
            legacy_kwargs["openai_api_base"] = self.openai_base_url

        try:
            return ChatOpenAI(**modern_kwargs)
        except Exception as modern_error:
            try:
                return ChatOpenAI(**legacy_kwargs)
            except Exception:
                raise modern_error

    def chat_with_image(self, request: ChatImageRequest) -> str:
        b64_len_kb = len(request.image_b64) * 3 / 4 / 1024
        print(f"[INFO] VLM 요청: model={request.model}, image={b64_len_kb:.1f}KB ({request.image_mime})")
        if self.endpoint:
            print(f"[INFO] LangChain endpoint={self.endpoint}")

        chat_model = self._build_chat_model(
            model=request.model,
            temperature=request.temperature,
        )

        data_url = f"data:{request.image_mime};base64,{request.image_b64}"
        messages = []
        if request.system_message:
            messages.append(SystemMessage(content=request.system_message))
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": request.user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ),
        )
        response = chat_model.invoke(messages)

        content = OpenAICompatibleVLMClient._coerce_content(getattr(response, "content", ""))
        print(f"[INFO] VLM 응답(content) 길이={len(content)}")
        return content

