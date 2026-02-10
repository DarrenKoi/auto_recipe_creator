"""
Home 환경 설정 모듈

.env 파일에서 환경변수를 로드하여 Home 데모 설정을 제공.

Usage:
    from poc.home.config import HomeConfig
    config = HomeConfig.load()
"""

import os
from dataclasses import dataclass

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


def _str(key: str, default: str = "") -> str:
    """환경변수를 문자열로 반환"""
    return os.environ.get(key, default).strip()


@dataclass
class HomeConfig:
    """Home 데모 설정"""
    hf_token: str = ""             # HuggingFace API 토큰
    model: str = "qwen2_vl_7b"    # 사용할 모델
    demo_mode: str = "all"        # 데모 모드
    safe_mode: bool = True        # 실제 입력 여부

    @classmethod
    def load(cls) -> "HomeConfig":
        """환경변수에서 설정을 로드"""
        # .env 파일 로드
        if DOTENV_AVAILABLE:
            home_dir = os.path.dirname(os.path.abspath(__file__))
            env_file = os.path.join(home_dir, ".env")
            if os.path.exists(env_file):
                load_dotenv(env_file)
            else:
                load_dotenv()
            print("[INFO] .env 파일 로드 완료")
        else:
            print("[WARNING] python-dotenv 미설치 — 시스템 환경변수만 사용")

        return cls(
            hf_token=_str("HF_TOKEN"),
            model=_str("HOME_MODEL", "qwen2_vl_7b"),
            demo_mode=_str("HOME_DEMO_MODE", "all"),
            safe_mode=_bool("HOME_SAFE_MODE", True),
        )

    def get_hf_model(self):
        """HFModel enum 반환"""
        from poc.home.hf_vlm import HFModel

        model_map = {
            "qwen2_vl_7b": HFModel.QWEN2_VL_7B,
            "qwen2_vl_2b": HFModel.QWEN2_VL_2B,
            "llava": HFModel.LLAVA_1_5_7B,
        }

        key = self.model.lower()
        if key not in model_map:
            print(f"[WARNING] 알 수 없는 HOME_MODEL='{self.model}', "
                  f"기본값 qwen2_vl_7b 사용")
            return HFModel.QWEN2_VL_7B
        return model_map[key]

    def print_summary(self):
        """설정 요약 출력"""
        print()
        print("=" * 60)
        print("  Home 데모 설정 요약")
        print("=" * 60)
        print(f"  HF Token:    {'****' if self.hf_token else '(미설정)'}")
        print(f"  Model:       {self.model}")
        print(f"  Demo Mode:   {self.demo_mode}")
        print(f"  Safe Mode:   {self.safe_mode}")
        print("=" * 60)
        print()
