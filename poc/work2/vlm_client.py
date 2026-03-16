"""간단한 Flask proxy VLM 클라이언트.

목적:
- coworkers 가 `poc/work2/flask_vlm.py`의 service slug 만 골라 바로 호출한다.
- `screen_analysis`, `main_tabs` 같은 purpose-based shared config 없이 쓴다.
- 이미지 bytes/path/base64 입력을 받아 OpenAI-compatible `/v1/chat/completions` 를 호출한다.

사용 예:
    from pathlib import Path

    from poc.work2.vlm_client import Work2VLMClient

    client = Work2VLMClient(service_slug="ui-venus")
    result = client.chat_with_image_path(
        image_path=Path("debug_images/sample.webp"),
        system_message="Respond only with JSON.",
        user_text="Describe clickable UI elements.",
    )
    print(result.text)
"""

from __future__ import annotations

import base64
import json as _json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from poc.work2.flask_vlm import get_service_by_slug, resolve_service_proxy_url
from poc.work2.logger import log_vlm_call


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
        }

        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )

        print(f"[INFO] VLM 응답 status={response.status_code}")
        print(
            f"[INFO] VLM 응답 headers: "
            f"content-type={response.headers.get('content-type', 'N/A')}"
        )
        try:
            raw_body = response.json()
            log_body = dict(raw_body)
            if "choices" in log_body:
                log_body["choices"] = f"[{len(raw_body['choices'])} item(s)]"
            print(
                f"[INFO] VLM 응답 body (요약): "
                f"{_json.dumps(log_body, ensure_ascii=False)}"
            )
        except Exception:
            print(f"[INFO] VLM 응답 body (raw): {response.text[:500]}")

        response.raise_for_status()

        data = response.json()
        self.last_token_usage = data.get("usage") or {}
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(
                f"VLM response has no choices: "
                f"{_json.dumps(data, ensure_ascii=False)[:300]}"
            )
        message = choices[0].get("message") or {}
        return self._coerce_content(message.get("content", ""))


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
    """work2 에 하드코딩된 서비스 목록을 반환한다."""
    from poc.work2.flask_vlm import ALL_VLM_SERVICES

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
class Work2VLMResponse:
    """VLM 호출 결과."""

    service_slug: str
    model_name: str
    api_url: str
    text: str
    token_usage: dict[str, int]


class Work2VLMClient:
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
        self.api_key = api_key.strip()
        self.model_name = resolved_model_name
        self.timeout_sec = timeout_sec
        self.log_name = log_name.strip() or "vlm_calls"
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
        model_name: str | None = None,
    ) -> Work2VLMResponse:
        """이미 base64 인코딩된 이미지를 사용해 요청한다."""
        request = ChatImageRequest(
            model=(model_name or self.model_name).strip(),
            system_message=system_message,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
            temperature=temperature,
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
        return Work2VLMResponse(
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
        model_name: str | None = None,
    ) -> Work2VLMResponse:
        """raw image bytes 를 base64 인코딩해서 요청한다."""
        mime = (image_mime or _detect_image_mime(image_bytes)).strip() or "image/webp"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return self.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime=mime,
            temperature=temperature,
            model_name=model_name,
        )

    def chat_with_image_path(
        self,
        *,
        image_path: str | Path,
        system_message: str,
        user_text: str,
        image_mime: str | None = None,
        temperature: float = 0.0,
        model_name: str | None = None,
    ) -> Work2VLMResponse:
        """로컬 이미지 파일을 읽어 요청한다."""
        path = Path(image_path)
        image_bytes = path.read_bytes()
        return self.chat_with_image_bytes(
            image_bytes=image_bytes,
            system_message=system_message,
            user_text=user_text,
            image_mime=image_mime,
            temperature=temperature,
            model_name=model_name,
        )


def send_image_request(
    *,
    service_slug: str,
    image_bytes: bytes,
    system_message: str,
    user_text: str,
    api_key: str = "",
    image_mime: str | None = None,
    temperature: float = 0.0,
    model_name: str | None = None,
    timeout_sec: float = 120.0,
) -> Work2VLMResponse:
    """한 번만 쓸 때 사용할 편의 함수."""
    client = Work2VLMClient(
        service_slug=service_slug,
        api_key=api_key,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )
    return client.chat_with_image_bytes(
        image_bytes=image_bytes,
        system_message=system_message,
        user_text=user_text,
        image_mime=image_mime,
        temperature=temperature,
        model_name=model_name,
    )


__all__ = [
    "ChatImageRequest",
    "OpenAICompatibleVLMClient",
    "Work2VLMClient",
    "Work2VLMResponse",
    "list_supported_services",
    "send_image_request",
]
