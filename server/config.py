"""
서버 설정 모듈

MongoDB, LLM, Flask 서버 설정을 관리.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class MongoConfig:
    """MongoDB 접속 설정."""
    uri: str = "mongodb://localhost:27017"   # MongoDB 연결 URI
    database: str = "chat_server"            # 데이터베이스 이름
    collection: str = "messages"             # 메시지 컬렉션


@dataclass
class LLMConfig:
    """LLM API 설정."""
    api_url: str = "http://localhost:8000/v1"  # OpenAI 호환 API 베이스 URL
    api_key: str = ""                          # API 키
    model_name: str = ""                       # 모델 이름
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_sec: float = 120.0
    system_prompt: str = ""                    # 기본 시스템 프롬프트


@dataclass
class ServerConfig:
    """Flask 서버 설정."""
    host: str = "0.0.0.0"        # 바인드 주소
    port: int = 5000             # 포트
    debug: bool = False          # Flask 디버그 모드


@dataclass
class AppConfig:
    """통합 설정."""
    mongo: MongoConfig
    llm: LLMConfig
    server: ServerConfig

    @classmethod
    def load(cls) -> "AppConfig":
        """환경변수에서 설정 로드."""
        if DOTENV_AVAILABLE:
            load_dotenv(override=True)

        return cls(
            mongo=MongoConfig(
                uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017").strip(),
                database=os.environ.get("MONGO_DATABASE", "chat_server").strip(),
                collection=os.environ.get("MONGO_COLLECTION", "messages").strip(),
            ),
            llm=LLMConfig(
                api_url=os.environ.get("LLM_API_URL", "http://localhost:8000/v1").strip(),
                api_key=os.environ.get("LLM_API_KEY", "").strip(),
                model_name=os.environ.get("LLM_MODEL_NAME", "").strip(),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
                timeout_sec=float(os.environ.get("LLM_TIMEOUT_SEC", "120.0")),
                system_prompt=os.environ.get("LLM_SYSTEM_PROMPT", "").strip(),
            ),
            server=ServerConfig(
                host=os.environ.get("SERVER_HOST", "0.0.0.0").strip(),
                port=int(os.environ.get("SERVER_PORT", "5000")),
                debug=os.environ.get("SERVER_DEBUG", "").strip().lower() in ("true", "1", "yes"),
            ),
        )
