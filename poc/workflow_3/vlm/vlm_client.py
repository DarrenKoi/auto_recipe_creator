"""workflow_3 용 간단한 Flask proxy VLM 클라이언트."""

import base64
import json as _json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from poc.workflow_3.vlm.flask_vlm import (
    ALL_VLM_SERVICES,
    get_service_by_slug,
    resolve_service_api_key,
    resolve_service_proxy_url,
)
from poc.workflow_3.logger import log_vlm_call


DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class ChatImageRequest:
    """VLM 채팅 이미지 요청 데이터."""

    model: str
    system_message: str
    user_text: str
    image_b64: str
    image_mime: str = "image/webp"
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    frequency_penalty: float | None = None
    stream: bool = False


class OpenAICompatibleVLMClient:
    """OpenAI-compatible chat-completions endpoint 용 최소 클라이언트."""

    def __init__(self, base_url: str, api_key: str = "", timeout_sec: float = 120.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_sec = timeout_sec
        self.last_token_usage: dict[str, int] = {}

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
    def _coerce_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            for key in (
                "text",
                "content",
                "value",
                "output_text",
                "reasoning_content",
                "arguments",
                "function",
            ):
                value = content.get(key)
                if value is None:
                    continue
                text = OpenAICompatibleVLMClient._coerce_content(value)
                if text:
                    return text
            return _json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                text = OpenAICompatibleVLMClient._coerce_content(item)
                if text:
                    chunks.append(text)
            return "\n".join(chunks).strip()
        return str(content)

    @classmethod
    def _extract_text_from_message_like(cls, payload: object) -> str:
        if isinstance(payload, dict):
            for key in (
                "content",
                "text",
                "output_text",
                "reasoning_content",
                "tool_calls",
                "function_call",
                "function",
            ):
                value = payload.get(key)
                if value is None:
                    continue
                text = cls._coerce_content(value)
                if text:
                    return text
            return ""
        return cls._coerce_content(payload)

    @staticmethod
    def _looks_like_chat_wrapper(data: object) -> bool:
        if not isinstance(data, dict):
            return False
        return any(
            key in data
            for key in ("choices", "usage", "id", "object", "model", "created")
        )

    @classmethod
    def _extract_text_from_choice(cls, choice: object) -> str:
        if not isinstance(choice, dict):
            return ""

        for key in ("message", "delta"):
            value = choice.get(key)
            if value is None:
                continue
            text = cls._extract_text_from_message_like(value)
            if text:
                return text

        for key in ("text", "content", "output_text", "reasoning_content"):
            value = choice.get(key)
            if value is None:
                continue
            text = cls._coerce_content(value)
            if text:
                return text

        return ""

    @classmethod
    def _extract_text_from_json_body(cls, data: object) -> str:
        if not isinstance(data, dict):
            return ""

        choices = data.get("choices") or []
        if isinstance(choices, list):
            chunks: list[str] = []
            for choice in choices:
                text = cls._extract_text_from_choice(choice)
                if text:
                    chunks.append(text)
            joined = "".join(chunks).strip()
            if joined:
                return joined

        for key in ("output_text", "text", "content"):
            value = data.get(key)
            if value is None:
                continue
            text = cls._coerce_content(value)
            if text:
                return text

        message = data.get("message")
        if message is not None:
            return cls._extract_text_from_message_like(message).strip()

        if not cls._looks_like_chat_wrapper(data):
            return _json.dumps(data, ensure_ascii=False)

        return ""

    @classmethod
    def _extract_text_from_sse_body(cls, body_text: str) -> str:
        if not body_text:
            return ""

        chunks: list[str] = []
        for raw_line in body_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                payload = _json.loads(payload_text)
            except Exception:
                chunks.append(payload_text)
                continue

            text = cls._extract_text_from_json_body(payload)
            if text:
                chunks.append(text)

        return "".join(chunks).strip()

    def chat_with_image(self, request: ChatImageRequest) -> str:
        """이미지 포함 chat completions 요청을 보내고 응답 텍스트를 반환한다."""
        b64_len_kb = len(request.image_b64) * 3 / 4 / 1024
        print(f"[INFO] VLM 요청: model={request.model}, image={b64_len_kb:.1f}KB ({request.image_mime})")

        messages: list[dict] = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{request.image_mime};base64,{request.image_b64}"
                        },
                    },
                ],
            },
        )

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.stream:
            payload["stream"] = True

        try:
            response = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is None:
                print(f"[ERROR] VLM 응답 실패: model={request.model}, error={exc}")
            else:
                print(
                    f"[ERROR] VLM 응답 실패: model={request.model}, "
                    f"status={status_code}, error={exc}"
                )
            raise

        body_text = response.text

        try:
            data = response.json()
        except ValueError:
            data = None

        if isinstance(data, dict):
            self.last_token_usage = data.get("usage") or {}
            text = self._extract_text_from_json_body(data)
            if text:
                usage = self.last_token_usage or {}
                print(
                    f"[INFO] VLM 응답 성공: model={request.model}, "
                    f"prompt={usage.get('prompt_tokens', '?')}, "
                    f"completion={usage.get('completion_tokens', '?')}, "
                    f"total={usage.get('total_tokens', '?')}"
                )
                return text
            if self._looks_like_chat_wrapper(data):
                raise ValueError(
                    "VLM JSON response has no usable assistant text. "
                    f"body={_json.dumps(data, ensure_ascii=False)[:500]}"
                )

        sse_text = self._extract_text_from_sse_body(body_text)
        if sse_text:
            print(
                f"[INFO] VLM 응답 성공: model={request.model}, "
                "prompt=?, completion=?, total=?"
            )
            return sse_text

        stripped_body = body_text.strip()
        if stripped_body:
            print(
                f"[INFO] VLM 응답 성공: model={request.model}, "
                "prompt=?, completion=?, total=?"
            )
            return stripped_body

        raise ValueError("VLM response has no usable text content.")


def _detect_image_mime(image_bytes: bytes, fallback: str = "image/webp") -> str:
    """이미지 header 를 보고 MIME type 을 추정한다."""
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def list_supported_services() -> list[dict[str, str]]:
    """workflow_3 에 하드코딩된 서비스 목록을 반환한다."""
    return [
        {
            "service": service.route_slug,
            "display_name": service.display_name,
            "model_name": service.model_name,
            "api_url": service.api_url,
        }
        for service in ALL_VLM_SERVICES
        if service.enabled
    ]


@dataclass(frozen=True)
class Workflow1VLMResponse:
    """VLM 호출 결과."""

    service_slug: str
    model_name: str
    api_url: str
    text: str
    token_usage: dict[str, int]


class Workflow1VLMClient:
    """service slug 기반의 단순 VLM 클라이언트."""

    def __init__(
        self,
        service_slug: str,
        *,
        api_key: str = "",
        model_name: str | None = None,
        api_url: str | None = None,
        timeout_sec: float = 120.0,
        log_name: str = "vlm_calls",
    ):
        service_entry = get_service_by_slug(service_slug)
        if service_entry is None:
            supported = ", ".join(item["service"] for item in list_supported_services())
            raise ValueError(
                f"알 수 없는 service slug: {service_slug}. "
                f"현재 지원: {supported}"
            )

        resolved_api_url = (api_url or resolve_service_proxy_url(service_slug)).strip().rstrip("/")
        resolved_model_name = (model_name or service_entry.model_name).strip()
        if not resolved_api_url:
            raise ValueError(f"service={service_slug} API URL 이 비어 있습니다.")
        if not resolved_model_name:
            raise ValueError(f"service={service_slug} model name 이 비어 있습니다.")

        self.service_slug = service_slug
        self.display_name = service_entry.display_name
        self.api_url = resolved_api_url
        self.api_key = (api_key or resolve_service_api_key(service_slug)).strip()
        self.model_name = resolved_model_name
        self.timeout_sec = timeout_sec
        self.log_name = log_name.strip() or "vlm_calls"
        self.prefer_stream = service_entry.prefer_stream
        self._client = OpenAICompatibleVLMClient(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout_sec=self.timeout_sec,
        )

    @property
    def endpoint(self) -> str:
        """실제 chat completions endpoint."""
        return self._client.endpoint

    def chat_with_image_b64(
        self,
        *,
        image_b64: str,
        system_message: str,
        user_text: str,
        image_mime: str = "image/webp",
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        frequency_penalty: float | None = None,
        model_name: str | None = None,
        stream: bool | None = None,
    ) -> Workflow1VLMResponse:
        """이미 base64 인코딩된 이미지를 사용해 요청한다."""
        request = ChatImageRequest(
            model=(model_name or self.model_name).strip(),
            system_message=system_message,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            stream=self.prefer_stream if stream is None else bool(stream),
        )
        started_at = time.time()
        try:
            text = self._client.chat_with_image(request)
        except Exception as exc:
            log_vlm_call(
                service=self.service_slug,
                model=request.model,
                status="error",
                latency_ms=(time.time() - started_at) * 1000,
                token_usage=dict(self._client.last_token_usage or {}),
                error=str(exc),
                endpoint=self.endpoint,
                log_name=self.log_name,
            )
            raise
        log_vlm_call(
            service=self.service_slug,
            model=request.model,
            status="ok",
            latency_ms=(time.time() - started_at) * 1000,
            token_usage=dict(self._client.last_token_usage or {}),
            endpoint=self.endpoint,
            response_text=text,
            log_name=self.log_name,
        )
        return Workflow1VLMResponse(
            service_slug=self.service_slug,
            model_name=request.model,
            api_url=self.api_url,
            text=text,
            token_usage=dict(self._client.last_token_usage or {}),
        )

    def chat_with_image_bytes(
        self,
        *,
        image_bytes: bytes,
        system_message: str,
        user_text: str,
        image_mime: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        frequency_penalty: float | None = None,
        model_name: str | None = None,
        stream: bool | None = None,
    ) -> Workflow1VLMResponse:
        """raw image bytes 를 base64 인코딩해서 요청한다."""
        mime = (image_mime or _detect_image_mime(image_bytes)).strip() or "image/webp"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return self.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime=mime,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model_name=model_name,
            stream=stream,
        )

    def chat_with_image_path(
        self,
        *,
        image_path: str | Path,
        system_message: str,
        user_text: str,
        image_mime: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        frequency_penalty: float | None = None,
        model_name: str | None = None,
        stream: bool | None = None,
    ) -> Workflow1VLMResponse:
        """로컬 이미지 파일을 읽어 요청한다."""
        path = Path(image_path)
        image_bytes = path.read_bytes()
        return self.chat_with_image_bytes(
            image_bytes=image_bytes,
            system_message=system_message,
            user_text=user_text,
            image_mime=image_mime,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model_name=model_name,
            stream=stream,
        )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "ChatImageRequest",
    "OpenAICompatibleVLMClient",
    "Workflow1VLMClient",
    "Workflow1VLMResponse",
    "list_supported_services",
]
