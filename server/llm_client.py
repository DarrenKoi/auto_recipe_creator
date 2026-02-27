"""
LLM 대화 클라이언트

LangGraph 그래프 또는 requests 폴백을 통한 LLM 호출.
"""

import requests

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from server.config import LLMConfig


def send_chat(messages: list[dict], config: LLMConfig, graph=None, channel_id: str = "") -> str:
    """대화 이력을 포함하여 LLM에 요청 후 응답 텍스트 반환.

    Args:
        messages: OpenAI 형식 메시지 리스트 [{"role": "...", "content": "..."}]
        config: LLM API 설정
        graph: 컴파일된 LangGraph 그래프 (없으면 requests 폴백)
        channel_id: 채널 ID (LangGraph thread_id로 사용)

    Returns:
        LLM 응답 텍스트
    """
    if graph is not None and LANGCHAIN_AVAILABLE:
        return _send_via_graph(messages, graph, channel_id)
    return _send_via_requests(messages, config)


def _send_via_graph(messages: list[dict], graph, channel_id: str) -> str:
    """LangGraph 그래프를 통한 LLM 호출."""
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    print(f"[INFO] LangGraph 호출: thread_id={channel_id}, messages={len(lc_messages)}개")

    result = graph.invoke(
        {"messages": lc_messages, "response": ""},
        config={"configurable": {"thread_id": channel_id}},
    )
    return result["response"]


def _send_via_requests(messages: list[dict], config: LLMConfig) -> str:
    """기존 requests 기반 LLM 호출 (폴백)."""
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

    print(f"[INFO] LLM 요청 (requests 폴백): model={config.model_name}, messages={len(messages)}개")
    response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_sec)
    if not response.ok:
        print(f"[ERROR] LLM status={response.status_code}, body={response.text[:500]}")
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[INFO] LLM 응답 길이={len(content)}")
    return content
