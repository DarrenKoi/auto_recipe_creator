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
from dataclasses import dataclass
from pathlib import Path

from poc.work.vlm_openai_client import ChatImageRequest, OpenAICompatibleVLMClient
from poc.work2.flask_vlm import get_service_by_slug, resolve_service_proxy_url


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
        text = self._client.chat_with_image(request)
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
    "Work2VLMClient",
    "Work2VLMResponse",
    "list_supported_services",
    "send_image_request",
]
