"""
LLM 대화 클라이언트

OpenAI 호환 API에 대화 이력을 포함하여 요청.
"""

import requests

from server.config import LLMConfig


def send_chat(messages: list[dict], config: LLMConfig) -> str:
    """대화 이력을 포함하여 LLM에 요청 후 응답 텍스트 반환.

    Args:
        messages: OpenAI 형식 메시지 리스트 [{"role": "...", "content": "..."}]
        config: LLM API 설정

    Returns:
        LLM 응답 텍스트
    """
    url = f"{config.api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    print(f"[INFO] LLM 요청: model={config.model_name}, messages={len(messages)}개")
    response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_sec)
    if not response.ok:
        print(f"[ERROR] LLM status={response.status_code}, body={response.text[:500]}")
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[INFO] LLM 응답 길이={len(content)}")
    return content
