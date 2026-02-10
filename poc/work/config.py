"""
PoC 통합 설정 모듈

.env 파일에서 환경변수를 로드하여 각 모듈 설정을 제공.

Usage:
    from poc.work.config import PocConfig
    config = PocConfig.load()
    config.print_summary()
"""

import os
from dataclasses import dataclass
from typing import Optional

# python-dotenv 임포트 가드
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def _bool(key: str, default: bool = False) -> bool:
    """환경변수를 bool로 변환"""
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


def _int(key: str, default: int = 0) -> int:
    """환경변수를 int로 변환"""
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"[WARNING] {key}='{val}' → int 변환 실패, 기본값 {default} 사용")
        return default


def _float(key: str, default: float = 0.0) -> float:
    """환경변수를 float로 변환"""
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"[WARNING] {key}='{val}' → float 변환 실패, 기본값 {default} 사용")
        return default


def _str(key: str, default: str = "") -> str:
    """환경변수를 문자열로 반환"""
    return os.environ.get(key, default).strip()


@dataclass
class VLMConfig:
    """VLM API 설정"""
    api_url: str = ""              # VLM API 엔드포인트
    api_key: str = ""              # API 인증 키
    model_name: str = ""           # 모델 이름
    provider: str = "qwen3_vl"     # VLM 제공자 (qwen3_vl, kimi_2, openai_gpt4v 등)


@dataclass
class OpenSearchConfig:
    """OpenSearch 설정"""
    url: str = "https://localhost:9200"     # OpenSearch URL
    index: str = "recipe_automation"        # 인덱스 이름
    username: str = ""                      # 인증 사용자명
    password: str = ""                      # 인증 비밀번호


@dataclass
class RCSConfig:
    """RCS 접속 정보"""
    server: str = ""               # RCS 서버 주소
    username: str = ""             # 사용자명
    password: str = ""             # 비밀번호


@dataclass
class OperationConfig:
    """실행 모드 및 동작 설정"""
    safe_mode: bool = True         # True: 실제 입력 없음 (분석만)
    use_webp: bool = True          # WebP 변환 사용
    max_image_size: int = 1280     # 리사이즈 최대 픽셀
    use_rag: bool = False          # RAG 사용 여부
    demo_type: str = "screen_analysis"  # 데모 유형
    action_delay: float = 0.5      # 액션 후 대기 시간 (초)
    # RCS 에이전트 설정
    rcs_tool_name: str = "CD-SEM Recipe Editor"  # 선택할 도구 이름
    max_steps_login: int = 15      # 로그인 최대 스텝
    max_steps_tool: int = 10       # 도구 선택 최대 스텝
    # RAG 평가 설정
    eval_cases_file: str = ""      # 평가 케이스 JSON 파일 경로
    eval_output_file: str = "evaluation_results.json"  # 결과 출력 파일
    # RAG 데모 설정
    capture_region: str = ""       # 캡처 영역 (x,y,width,height)
    query_text: str = ""           # 쿼리 텍스트
    compare_mode: bool = False     # With/Without RAG 비교 모드


@dataclass
class PocConfig:
    """PoC 통합 설정"""
    vlm: VLMConfig
    opensearch: OpenSearchConfig
    rcs: RCSConfig
    operation: OperationConfig

    @classmethod
    def load(cls, dotenv_path: Optional[str] = None) -> "PocConfig":
        """
        환경변수에서 설정을 로드

        Args:
            dotenv_path: .env 파일 경로 (None이면 자동 탐색)

        Returns:
            PocConfig 인스턴스
        """
        # .env 파일 로드
        if DOTENV_AVAILABLE:
            if dotenv_path:
                load_dotenv(dotenv_path)
            else:
                # poc/.env → 프로젝트 루트/.env 순서로 탐색
                poc_dir = os.path.dirname(os.path.abspath(__file__))
                env_file = os.path.join(poc_dir, ".env")
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                else:
                    load_dotenv()  # 자동 탐색
            print("[INFO] .env 파일 로드 완료")
        else:
            print("[WARNING] python-dotenv 미설치 — 시스템 환경변수만 사용")

        vlm = VLMConfig(
            api_url=_str("VLM_API_URL"),
            api_key=_str("VLM_API_KEY"),
            model_name=_str("VLM_MODEL_NAME", "Qwen3-VL-30B-Instruct"),
            provider=_str("VLM_PROVIDER", "qwen3_vl"),
        )

        opensearch = OpenSearchConfig(
            url=_str("OPENSEARCH_URL", "https://localhost:9200"),
            index=_str("OPENSEARCH_INDEX", "recipe_automation"),
            username=_str("OPENSEARCH_USERNAME"),
            password=_str("OPENSEARCH_PASSWORD"),
        )

        rcs = RCSConfig(
            server=_str("RCS_SERVER"),
            username=_str("RCS_USERNAME"),
            password=_str("RCS_PASSWORD"),
        )

        operation = OperationConfig(
            safe_mode=_bool("SAFE_MODE", True),
            use_webp=_bool("USE_WEBP", True),
            max_image_size=_int("MAX_IMAGE_SIZE", 1280),
            use_rag=_bool("USE_RAG", False),
            demo_type=_str("DEMO_TYPE", "screen_analysis"),
            action_delay=_float("ACTION_DELAY", 0.5),
            rcs_tool_name=_str("RCS_TOOL_NAME", "CD-SEM Recipe Editor"),
            max_steps_login=_int("MAX_STEPS_LOGIN", 15),
            max_steps_tool=_int("MAX_STEPS_TOOL", 10),
            eval_cases_file=_str("EVAL_CASES_FILE"),
            eval_output_file=_str("EVAL_OUTPUT_FILE", "evaluation_results.json"),
            capture_region=_str("CAPTURE_REGION"),
            query_text=_str("QUERY_TEXT"),
            compare_mode=_bool("COMPARE_MODE", False),
        )

        return cls(vlm=vlm, opensearch=opensearch, rcs=rcs, operation=operation)

    def get_vlm_provider(self):
        """VLMProvider enum 반환"""
        from .vlm_screen_analysis import VLMProvider

        provider_map = {
            "qwen3_vl": VLMProvider.QWEN3_VL,
            "qwen_vl": VLMProvider.QWEN_VL,
            "kimi_2": VLMProvider.KIMI_2,
            "openai_gpt4v": VLMProvider.OPENAI_GPT4V,
            "anthropic_claude": VLMProvider.ANTHROPIC_CLAUDE,
            "local": VLMProvider.LOCAL,
        }

        key = self.vlm.provider.lower()
        if key not in provider_map:
            print(f"[WARNING] 알 수 없는 VLM_PROVIDER='{self.vlm.provider}', "
                  f"기본값 QWEN3_VL 사용")
            return VLMProvider.QWEN3_VL
        return provider_map[key]

    def print_summary(self):
        """설정 요약 출력 (비밀번호 마스킹)"""
        print()
        print("=" * 60)
        print("  PoC 설정 요약")
        print("=" * 60)
        print(f"  VLM API URL:    {self.vlm.api_url or '(미설정)'}")
        print(f"  VLM Provider:   {self.vlm.provider}")
        print(f"  VLM Model:      {self.vlm.model_name}")
        print(f"  VLM API Key:    {'****' if self.vlm.api_key else '(미설정)'}")
        print(f"  OpenSearch URL: {self.opensearch.url}")
        print(f"  OpenSearch Idx: {self.opensearch.index}")
        print(f"  RCS Server:     {self.rcs.server or '(미설정)'}")
        print(f"  RCS User:       {self.rcs.username or '(미설정)'}")
        print(f"  RCS Password:   {'****' if self.rcs.password else '(미설정)'}")
        print(f"  Safe Mode:      {self.operation.safe_mode}")
        print(f"  Image Format:   {'WebP' if self.operation.use_webp else 'PNG'}")
        print(f"  Max Image Size: {self.operation.max_image_size}px")
        print(f"  Use RAG:        {self.operation.use_rag}")
        print(f"  Demo Type:      {self.operation.demo_type}")
        print("=" * 60)
        print()
