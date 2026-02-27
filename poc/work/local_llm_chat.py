"""
로컬 LLM 텍스트 요청 템플릿

requests를 사용하여 OpenAI 호환 로컬 LLM에 텍스트를 보내고 응답을 받는 모듈.

Usage:
    uv run python poc/work/local_llm_chat.py
"""

import requests


# ── 설정 (하드코딩) ──────────────────────────────────────────
API_URL = "http://localhost:8000/v1"    # OpenAI 호환 API 베이스 URL
API_KEY = "your-api-key-here"           # API 키
MODEL_NAME = "your-model-name"          # 모델 이름
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TIMEOUT_SEC = 120.0
# ─────────────────────────────────────────────────────────────


def send_text(user_text: str, system_prompt: str = "") -> str:
    """텍스트를 LLM에 보내고 응답 텍스트를 반환."""
    url = f"{API_URL.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    print(f"[INFO] LLM 요청: model={MODEL_NAME}, url={url}")
    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
    if not response.ok:
        print(f"[ERROR] status={response.status_code}, body={response.text[:500]}")
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[INFO] LLM 응답 길이={len(content)}")
    return content


if __name__ == "__main__":
    result = send_text("안녕하세요, 테스트 메시지입니다.")
    print(f"응답: {result}")
