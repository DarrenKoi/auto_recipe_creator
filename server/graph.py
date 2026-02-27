"""
LangGraph 대화 그래프

단일 노드(LLM 호출) 그래프 — 채널별 대화 이력 기반 응답 생성.
"""

from typing import TypedDict

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from server.config import LLMConfig


class ChatState(TypedDict):
    """그래프 상태 정의."""
    messages: list  # LangChain BaseMessage 리스트
    response: str   # LLM 응답 텍스트


def build_chat_model(config: LLMConfig):
    """LLMConfig로부터 ChatOpenAI 인스턴스 생성.

    OpenAI 호환 로컬 API에 연결하기 위해 base_url을 설정한다.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph / langchain-openai가 설치되어 있지 않습니다. "
            "pip install langgraph langchain-openai langchain-core"
        )

    base_url = config.api_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_sec,
        "base_url": base_url,
        "streaming": False,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    else:
        kwargs["api_key"] = "not-needed"

    print(f"[INFO] ChatOpenAI 생성: model={config.model_name}, base_url={base_url}")
    return ChatOpenAI(**kwargs)


def create_chat_graph(config: LLMConfig):
    """LangGraph 대화 그래프 생성 및 컴파일.

    Args:
        config: LLM API 설정

    Returns:
        컴파일된 LangGraph 그래프
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph가 설치되어 있지 않습니다")

    llm = build_chat_model(config)

    def chat_node(state: ChatState) -> dict:
        """LLM 호출 노드."""
        messages = state["messages"]
        print(f"[INFO] LangGraph chat_node: messages={len(messages)}개")

        ai_message = llm.invoke(messages)
        content = ai_message.content or ""
        print(f"[INFO] LangGraph 응답 길이={len(content)}")

        return {"response": content}

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_edge(START, "chat")
    graph_builder.add_edge("chat", END)

    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)

    print("[INFO] LangGraph 대화 그래프 컴파일 완료")
    return graph
