"""poc.work2.vlm_client 응답 파싱 테스트."""

import json

import pytest

from poc.work2.vlm_client import (
    ChatImageRequest,
    OpenAICompatibleVLMClient,
    Work2VLMClient,
)


class DummyResponse:
    """requests.Response 대체용 최소 객체."""

    def __init__(self, *, status_code: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_chat_with_image_reads_standard_message_content(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                }
            ).encode("utf-8"),
        )

    monkeypatch.setattr("poc.work2.vlm_client.requests.post", fake_post)

    client = OpenAICompatibleVLMClient("http://example.com/v1")
    text = client.chat_with_image(
        ChatImageRequest(
            model="ui-venus-1.5-8b",
            system_message="json only",
            user_text="ping",
            image_b64="dGVzdA==",
        )
    )

    assert text == "ok"


def test_chat_with_image_reads_ui_tars_sse_content(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(
            status_code=200,
            body=(
                b'data: {"choices":[{"delta":{"content":"{\\"coord_system\\": \\"relative_1000\\","}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" \\"login_button\\": {\\"x\\": 512, \\"y\\": 824}}"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr("poc.work2.vlm_client.requests.post", fake_post)

    client = OpenAICompatibleVLMClient("http://example.com/v1")
    text = client.chat_with_image(
        ChatImageRequest(
            model="ui-tars-1.5-7b",
            system_message="json only",
            user_text="ping",
            image_b64="dGVzdA==",
        )
    )

    assert json.loads(text) == {
        "coord_system": "relative_1000",
        "login_button": {"x": 512, "y": 824},
    }


def test_chat_with_image_raises_when_chat_wrapper_content_is_null(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 1,
                    },
                }
            ).encode("utf-8"),
        )

    monkeypatch.setattr("poc.work2.vlm_client.requests.post", fake_post)

    client = OpenAICompatibleVLMClient("http://example.com/v1")
    with pytest.raises(ValueError, match="no usable assistant text"):
        client.chat_with_image(
            ChatImageRequest(
                model="ui-tars-1.5-7b",
                system_message="json only",
                user_text="ping",
                image_b64="dGVzdA==",
            )
        )


def test_work2_client_enables_stream_by_default_for_ui_tars(monkeypatch):
    captured_payload: dict[str, object] = {}

    def fake_post(*args, **kwargs):
        captured_payload.update(kwargs["json"])
        return DummyResponse(
            status_code=200,
            body=(
                b'data: {"choices":[{"delta":{"content":"{\\"coord_system\\": \\"relative_1000\\"}"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr("poc.work2.vlm_client.requests.post", fake_post)

    client = Work2VLMClient(service_slug="ui-tars")
    response = client.chat_with_image_b64(
        image_b64="dGVzdA==",
        image_mime="image/webp",
        system_message="json only",
        user_text="ping",
    )

    assert captured_payload["stream"] is True
    assert json.loads(response.text) == {"coord_system": "relative_1000"}
