"""OpenAI-compatible VLM client utilities."""

from dataclasses import dataclass

import requests

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False


@dataclass(frozen=True)
class ChatImageRequest:
    model: str
    system_message: str
    user_text: str
    image_b64: str
    image_mime: str = "image/png"
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
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_message},
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
            ],
            "temperature": request.temperature,
        }

        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("VLM response has no choices")
        message = choices[0].get("message") or {}
        return self._coerce_content(message.get("content", ""))


class LangChainOpenAIVLMClient:
    """LangChain ChatOpenAI-backed VLM client for easy LangGraph migration."""

    def __init__(self, base_url: str, api_key: str = "", timeout_sec: float = 120.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_sec = timeout_sec

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

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
        if not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError(
                "langchain_openai is not installed. "
                "Install it to use LangChainOpenAIVLMClient."
            )

        llm = ChatOpenAI(
            model=request.model,
            temperature=request.temperature,
            api_key=self.api_key or None,
            base_url=self.base_url or None,
            timeout=self.timeout_sec,
        )

        response = llm.invoke(
            [
                SystemMessage(content=request.system_message),
                HumanMessage(
                    content=[
                        {"type": "text", "text": request.user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{request.image_mime};base64,{request.image_b64}"},
                        },
                    ]
                ),
            ]
        )
        return self._coerce_content(response.content)
